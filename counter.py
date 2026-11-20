from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .crossing import (CrossingFSM, CrossingEvent as FsmCrossingEvent,
                       LineGeometry, anchor_point as _anchor_impl,
                       normal_from_line)
from .identity import TrackletLinker
from .quality import TrackQualityGate


def _seq_len(v) -> int:
    """``len(v)`` 的安全版本: 非序列返回 -1 (用于识别 tlwh 这类 4 元序列)。"""
    try:
        return len(v)
    except TypeError:
        return -1


# --------------------------------------------------------------------------- #
@dataclass
class CrossingEvent:
    track_id: int
    direction: str          # "in" | "out"
    raw_direction: str      # "fwd" | "bwd"
    frame_id: int
    point: Tuple[float, float]


@dataclass
class ZoneEvent:
    track_id: int
    direction: str          # "enter" | "exit"
    frame_id: int


def _anchor_point(tlwh: Sequence[float], mode: str) -> Tuple[float, float]:
    """锚点 (兼容旧名): 默认底边中点 (脚点)。实现见 ``crossing.anchor_point``。"""
    return _anchor_impl(tlwh, mode)


# --------------------------------------------------------------------------- #
class LineCounter:
    """虚拟计数线 (legacy 算法)。方向由显式 ``in_normal`` 定义, 归一化坐标。

    与旧实现的唯一区别是**方向语义显式化**: 有符号距离 ``d = (p - p1)·in_normal``,
    ``d > 0`` 为 IN 侧; 由 OUT 侧翻到 IN 侧记 ``in``, 反之记 ``out``。
    这样交换 ``p1``/``p2`` 不会反转 IN/OUT。
    """

    def __init__(self, cfg_line: dict, anchor: str = "bottom_center", dedupe_window: int = 60,
                 gate: Optional[TrackQualityGate] = None):
        self.cfg_line = dict(cfg_line or {})
        self.p1_n = np.array(self.cfg_line["p1"], dtype=float)
        self.p2_n = np.array(self.cfg_line["p2"], dtype=float)
        self.in_direction = self.cfg_line.get("in_direction", "fwd")
        self.in_normal_n = self.cfg_line.get("in_normal")
        self.line_id = str(self.cfg_line.get("line_id", "") or "")
        self.line_version = int(self.cfg_line.get("line_version", 1) or 1)
        self.hysteresis = float(self.cfg_line.get("hysteresis", 8.0))
        self.min_track_age = int(self.cfg_line.get("min_track_age", 4))
        self.min_displacement = float(self.cfg_line.get("min_displacement", 6.0))
        self.cooldown_frames = int(self.cfg_line.get("cooldown_frames", 25))
        # 可选: 只在线段有效范围内 / ROI 内计数 (默认关闭以保持 legacy 行为)
        self.enforce_gate = bool(self.cfg_line.get("enforce_segment", False))
        self.dedupe_window = int(dedupe_window)
        self.anchor = anchor
        self.gate = gate

        self.geom: Optional[LineGeometry] = None
        self._ready = False

        # 每身份状态
        self._state: Dict[int, dict] = {}
        self._events: List[CrossingEvent] = []

    # ------------------------------------------------------------------ #
    def _ensure_geometry(self, width: int, height: int):
        if self._ready:
            return
        self.geom = LineGeometry.from_cfg(self.cfg_line, width, height)
        self._ready = True

    def _signed(self, p: np.ndarray) -> float:
        """带符号垂直距离: >0 在 IN 侧, <0 在 OUT 侧 (像素)。"""
        return self.geom.signed_distance(p)

    @property
    def in_normal(self) -> Optional[np.ndarray]:
        return None if self.geom is None else self.geom.in_normal

    def _raw_to_inout(self, raw: str) -> str:
        """兼容旧接口的映射 (方向语义已内化到 ``in_normal``)。

        ``raw`` 现由有符号距离直接给出: ``"bwd"`` = OUT→IN = 进井, ``"fwd"`` = 出井。
        """
        return "in" if raw == "bwd" else "out"

    # ------------------------------------------------------------------ #
    def update(self, frame_id: int, observations: Sequence[Tuple[int, Sequence[float]]],
               width: int, height: int) -> List[CrossingEvent]:
        """observations: [(track_id, tlwh), ...] 返回本帧新增的穿越事件。"""
        self._ensure_geometry(width, height)
        new_events: List[CrossingEvent] = []

        for ob in observations:
            tid, tlwh = ob[0], ob[1]
            p = np.array(_anchor_point(tlwh, self.anchor), dtype=float)
            st = self._state.get(tid)
            if st is None:
                st = {
                    "age": 1,
                    "last_eval_p": p.copy(),
                    "last_side": self._signed(p),
                    "side_state": 0,     # -1 / 0 / +1 已确认侧
                    "last_count_frame": -10 ** 9,
                }
                self._state[tid] = st
                continue

            st["age"] += 1
            side = self._signed(p)
            moved = float(np.linalg.norm(p - st["last_eval_p"]))

            # 位移不足 → 认为几乎静止, 不更新评估基准 (抗抖动)
            if moved < self.min_displacement and st["side_state"] != 0:
                continue

            # 更新"已确认侧"状态机 (带死区)
            prev_state = st["side_state"]
            if side > self.hysteresis:
                st["side_state"] = 1
            elif side < -self.hysteresis:
                st["side_state"] = -1
            # else 保持 (处于死区内不翻转)

            crossed = None
            if prev_state != 0 and st["side_state"] != 0 and st["side_state"] != prev_state:
                # -1(OUT) -> +1(IN) = 进井; +1 -> -1 = 出井
                crossed = "bwd" if st["side_state"] > 0 else "fwd"

            st["last_eval_p"] = p.copy()
            st["last_side"] = side

            if crossed is None:
                continue
            if st["age"] < self.min_track_age:
                continue
            if frame_id - st["last_count_frame"] < self.cooldown_frames:
                continue
            # 误检门: 轨迹质量不达标 -> 不计数 (判断并消除)
            if self.gate is not None and not self.gate.passes(tid):
                st["last_count_frame"] = frame_id
                continue
            # 可选: 有限线段 / ROI 空间约束
            if self.enforce_gate:
                a = st.get("last_eval_p")
                cross = self.geom.crossing_point(a, p) if a is not None else None
                if cross is None or not self.geom.in_valid_space(cross):
                    st["last_count_frame"] = frame_id
                    continue

            raw = crossed
            direction = self._raw_to_inout(raw)

            # 同身份同向去重 (抑制 re-ID 碎片化重复计数)
            if self._is_duplicate(tid, raw, frame_id):
                st["last_count_frame"] = frame_id
                continue

            ev = CrossingEvent(track_id=int(tid), direction=direction, raw_direction=raw,
                               frame_id=int(frame_id), point=(float(p[0]), float(p[1])))
            self._events.append(ev)
            new_events.append(ev)
            st["last_count_frame"] = frame_id

        return new_events

    def _is_duplicate(self, tid: int, raw: str, frame_id: int) -> bool:
        for ev in reversed(self._events):
            if frame_id - ev.frame_id > self.dedupe_window:
                break
            if ev.track_id == tid and ev.raw_direction == raw:
                return True
        return False

    # ------------------------------------------------------------------ #
    @property
    def events(self) -> List[CrossingEvent]:
        return self._events

    def counts(self) -> Dict[str, int]:
        c = {"in": 0, "out": 0}
        for ev in self._events:
            c[ev.direction] += 1
        return c

    def geometry(self):
        if self.geom is None:
            return None, None
        return self.geom.p1, self.geom.p2

    def line_describe(self) -> Dict[str, Any]:
        return {
            "algorithm": "legacy",
            "line_id": self.line_id,
            "line_version": self.line_version,
            "geometry": (None if self.geom is None else self.geom.describe()),
            "hysteresis_px": self.hysteresis,
            "cooldown_frames": self.cooldown_frames,
            "dedupe_window": self.dedupe_window,
            "enforce_segment": self.enforce_gate,
        }

    def per_id_counts(self) -> Dict[int, Dict[str, int]]:
        d: Dict[int, Dict[str, int]] = {}
        for ev in self._events:
            d.setdefault(ev.track_id, {"in": 0, "out": 0})[ev.direction] += 1
        return d


# --------------------------------------------------------------------------- #
class FSMLineCounter:
    """虚拟计数线 (fsm_v1 算法): 双稳定区 FSM + 有限线段/ROI + 逻辑身份。

    与 ``LineCounter`` 保持**完全相同的对外接口** (``update`` / ``events`` /
    ``counts`` / ``geometry`` / ``per_id_counts`` / ``in_direction``), 因此
    ``Counter`` 与 ``pipeline`` 无需感知实现差异, 只需 ``counter.algorithm`` 切换。

    身份生命周期: 本类自己从观测流里检测 track 的"出现/消失", 维护
    ``TrackletLinker``; FSM 以 ``person_uid`` 为键, 从而对 ID 碎片化免疫。
    """

    def __init__(self, cfg_line: dict, anchor: str = "bottom_center",
                 gate: Optional[TrackQualityGate] = None, fps: float = 25.0,
                 identity_cfg: Optional[dict] = None, algorithm: str = "fsm_v1",
                 gate_by_identity: bool = True):
        self.cfg_line = dict(cfg_line or {})
        self.anchor = anchor
        self.gate = gate
        self.fps = float(fps or 25.0)
        self.algorithm = algorithm
        self.gate_by_identity = bool(gate_by_identity)
        self.in_direction = self.cfg_line.get("in_direction", "fwd")
        self.line_id = str(self.cfg_line.get("line_id", "") or "")
        self.line_version = int(self.cfg_line.get("line_version", 1) or 1)

        c = dict(identity_cfg or {})
        self.identity_enabled = bool(c.get("enabled", True))
        self.linker = TrackletLinker(
            enabled=self.identity_enabled,
            max_gap_sec=float(c.get("max_gap_sec", 1.5)),
            fps=self.fps,
            max_dist_px=float(c.get("max_dist_px", 60.0)),
            max_dist_rel=float(c.get("max_dist_rel", 1.0)),
            dir_cos_min=float(c.get("dir_cos_min", 0.0)),
            margin_rel=float(c.get("margin_rel", 1.35)),
            margin_px=float(c.get("margin_px", 25.0)),
            use_appearance=bool(c.get("use_appearance", True)),
            app_min_cos=float(c.get("app_min_cos", 0.75)),
            # 任务C: UID 唯一性修复 (默认开启)
            strict_unique_uid=bool(c.get("strict_unique_uid", True)),
            prefer_same_track=bool(c.get("prefer_same_track", True)),
        )

        self.fsm: Optional[CrossingFSM] = None
        self.geom: Optional[LineGeometry] = None
        self._ready = False
        self._active: Dict[int, str] = {}          # track_id -> person_uid
        self._uid_active: Dict[str, int] = {}      # person_uid -> 活跃 track 数
        # 由 ``Counter`` 在构造完成后按运行模式设置: 当质量门按逻辑身份求值时,
        # 观测统计也由本类按 ``person_uid`` 累计 (Counter 不再按 track_id 累计)。
        self.owns_gate_observation = False

    # ------------------------------------------------------------------ #
    def _ensure_geometry(self, width: int, height: int):
        if self._ready:
            return
        self.geom = LineGeometry.from_cfg(self.cfg_line, width, height)
        self.fsm = CrossingFSM(
            self.geom,
            k_start=int(self.cfg_line.get("k_start", 3)),
            k_end=int(self.cfg_line.get("k_end", 3)),
            band_rel=float(self.cfg_line.get("band_rel", 0.2)),
            band_min=float(self.cfg_line.get("band_min", 4.0)),
            band_max=float(self.cfg_line.get("band_max", 120.0)),
            band_px=float(self.cfg_line.get("band_px", 0.0) or 0.0),
            max_pending_sec=float(self.cfg_line.get("max_pending_sec", 3.0)),
            jump_rel=float(self.cfg_line.get("jump_rel", 0.8)),
            jump_px=float(self.cfg_line.get("jump_px", 30.0)),
            cooldown_frames=int(self.cfg_line.get("event_cooldown_frames", 25)),
            min_score=(None if self.cfg_line.get("min_score") is None
                       else float(self.cfg_line["min_score"])),
            gate=self.gate, anchor=self.anchor, fps=self.fps,
            line_id=self.line_id, line_version=self.line_version,
            algorithm=self.algorithm,
            gate_by_identity=self.gate_by_identity,
            gate_blocks_state=bool(self.cfg_line.get("gate_blocks_state", False)),
            osc_guard=bool(self.cfg_line.get("osc_guard", True)),
            osc_window_sec=float(self.cfg_line.get("osc_window_sec", 3.0)),
            osc_boost=float(self.cfg_line.get("osc_boost", 1.0)),
            osc_min_flips=int(self.cfg_line.get("osc_min_flips", 2)),
            osc_episode_min=int(self.cfg_line.get("osc_episode_min", 5)),
            # 阶段四 · 任务B: A/B 通行约束与事件时间语义 (默认关闭 = V3 行为)
            min_side_displacement_px=float(
                self.cfg_line.get("min_side_displacement_px", 0.0) or 0.0),
            max_normal_step_rel=float(
                self.cfg_line.get("max_normal_step_rel", 0.0) or 0.0),
            event_time=str(self.cfg_line.get("event_time", "confirm") or "confirm"),
            # 阶段四 · 任务3: 锚点-身体一致性门 (默认关闭 = V3 行为)
            anchor_body_corroboration=float(
                self.cfg_line.get("anchor_body_corroboration", 0.0) or 0.0),
        )
        self._ready = True

    # ------------------------------------------------------------------ #
    def update(self, frame_id: int, observations: Sequence[Sequence],
               width: int, height: int) -> List[FsmCrossingEvent]:
        self._ensure_geometry(width, height)
        ts = float(frame_id) / self.fps

        obs_norm: List[Tuple[int, Sequence[float], Optional[float]]] = []
        keys: Dict[int, str] = {}
        seen = set()
        for ob in observations:
            tid = int(ob[0])
            tlwh = ob[1]
            score = float(ob[2]) if len(ob) >= 3 and ob[2] is not None else None
            seen.add(tid)
            uid, first = self.linker.update(
                tid, _anchor_point(tlwh, self.anchor), frame_id, ts,
                bbox_h=float(tlwh[3]) if len(tlwh) >= 4 else 0.0,
            )
            self._active[tid] = uid
            self._uid_active[uid] = self._uid_active.get(uid, 0) + (1 if first else 0)
            keys[tid] = uid
            obs_norm.append((tid, tlwh, score))
            # 质量门按逻辑身份累计时, 这里用 uid 而不是 track_id 喂统计 ——
            # 否则一个被切成多条 2–3 帧短轨迹的真人会被整体判为"观测不足"。
            if self.owns_gate_observation and self.gate is not None:
                self.gate.observe(uid, tlwh, score, frame_id, width, height)

        # 轨迹消失 → 移入 finished 池, 并关闭对应的 FSM 身份 (不补记事件)
        gone = [t for t in list(self._active) if t not in seen]
        for tid in gone:
            uid = self._active.pop(tid, None)
            # 用该轨迹**最后被观测到**的帧/时间为准 (而不是当前空帧), 断轨间隔才准确
            self.linker.end(tid)
            if uid is not None:
                n = self._uid_active.get(uid, 1) - 1
                if n <= 0:
                    self._uid_active.pop(uid, None)
                    self.fsm.close_identity(uid)
                else:
                    self._uid_active[uid] = n

        return self.fsm.update(frame_id, obs_norm, width, height,
                               timestamp=ts, identity_keys=keys)

    # ------------------------------------------------------------------ #
    @property
    def events(self) -> List[FsmCrossingEvent]:
        return [] if self.fsm is None else self.fsm.events

    @property
    def in_normal(self) -> Optional[np.ndarray]:
        return None if self.geom is None else self.geom.in_normal

    def counts(self) -> Dict[str, int]:
        if self.fsm is None:
            return {"in": 0, "out": 0}
        return self.fsm.counts()

    def geometry(self):
        if self.geom is None:
            return None, None
        return self.geom.p1, self.geom.p2

    def per_id_counts(self) -> Dict[int, Dict[str, int]]:
        return {} if self.fsm is None else self.fsm.per_id_counts()

    def line_describe(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "line_id": self.line_id,
            "line_version": self.line_version,
            "geometry": (None if self.geom is None else self.geom.describe()),
            "identity": self.linker.summary(),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "line": self.line_describe(),
            "fsm": (None if self.fsm is None else self.fsm.diagnostics()),
        }


# --------------------------------------------------------------------------- #
class ZoneCounter:
    """多边形区域: 当前占用人数 + 可选进入/离开事件。"""

    def __init__(self, cfg_zone: dict, anchor: str = "bottom_center",
                 gate: Optional[TrackQualityGate] = None):
        self.poly_n = np.array(cfg_zone["polygon"], dtype=float)
        self.name = cfg_zone.get("name", "zone")
        self.count_enter = bool(cfg_zone.get("count_enter", True))
        self.count_exit = bool(cfg_zone.get("count_exit", True))
        self.anchor = anchor
        self.gate = gate
        self._poly = None
        self._inside: Dict[int, bool] = {}
        self._occupancy_series: List[int] = []
        self._events: List[ZoneEvent] = []

    def _ensure_geometry(self, width: int, height: int):
        if self._poly is None:
            self._poly = np.stack(
                [self.poly_n[:, 0] * width, self.poly_n[:, 1] * height], axis=1
            ).astype(np.float32)

    @staticmethod
    def _point_in_poly(pt, poly) -> bool:
        """纯 numpy 射线法 (不再依赖 cv2, 便于无 GPU/无 cv2 环境单测)。"""
        from .crossing import point_in_polygon

        return point_in_polygon(pt, poly)

    def update(self, frame_id: int, observations, width: int, height: int) -> List[ZoneEvent]:
        self._ensure_geometry(width, height)
        new_events: List[ZoneEvent] = []
        seen = set()
        for ob in observations:
            tid, tlwh = ob[0], ob[1]
            p = _anchor_point(tlwh, self.anchor)
            inside = self._point_in_poly(p, self._poly)
            prev = self._inside.get(tid)
            if prev is None:
                self._inside[tid] = inside
                seen.add(tid)
                continue
            if inside != prev:
                self._inside[tid] = inside
                if inside and self.count_enter and (self.gate is None or self.gate.passes(tid)):
                    ev = ZoneEvent(int(tid), "enter", int(frame_id))
                    self._events.append(ev)
                    new_events.append(ev)
                elif (not inside) and self.count_exit and (self.gate is None or self.gate.passes(tid)):
                    ev = ZoneEvent(int(tid), "exit", int(frame_id))
                    self._events.append(ev)
                    new_events.append(ev)
            seen.add(tid)
        occupancy = sum(1 for v in self._inside.values() if v)
        self._occupancy_series.append(occupancy)
        return new_events

    @property
    def occupancy(self) -> int:
        return sum(1 for v in self._inside.values() if v)

    @property
    def events(self) -> List[ZoneEvent]:
        return self._events

    def counts(self) -> Dict[str, int]:
        c = {"enter": 0, "exit": 0}
        for ev in self._events:
            c[ev.direction] += 1
        return c

    def geometry(self):
        return self._poly


# --------------------------------------------------------------------------- #
class Counter:
    """计数门面: 组合 线计数 + 区域计数。

    ``counter.algorithm`` 选择计数算法 (``legacy`` 默认 | ``fsm_v1``)。
    两种实现对外接口一致, 因此上层 (pipeline / viz / 汇总脚本) 无需改动。
    """

    def __init__(self, cfg: "Config", fps: Optional[float] = None):
        c = cfg.get("counter")
        anchor = c.get("anchor", "bottom_center")
        dedupe = int(c.get("dedupe_window", 60))
        self.mode = c.get("mode", "line")
        self.algorithm = str(c.get("algorithm", "legacy") or "legacy").lower()
        if self.algorithm in ("fsm", "v1", "fsm1"):
            self.algorithm = "fsm_v1"
        self.anchor = anchor
        self.quality = TrackQualityGate(c.get("quality", {}))
        self.line = None
        self.zone: Optional[ZoneCounter] = None

        fps = float(fps or cfg.get("io.fps_override", 0) or 0) or 25.0

        if "line" in self.mode:
            if self.algorithm == "fsm_v1":
                self.line = FSMLineCounter(
                    c["line"], anchor=anchor, gate=self.quality, fps=fps,
                    identity_cfg=c.get("identity", {}), algorithm=self.algorithm,
                    gate_by_identity=bool(
                        c.get("quality", {}).get("by_identity", True)))
            else:
                self.line = LineCounter(c["line"], anchor=anchor, dedupe_window=dedupe,
                                        gate=self.quality)
        if "zone" in self.mode:
            z = c.get("zone", {})
            if z.get("enabled", False) or self.mode == "zone":
                self.zone = ZoneCounter(z, anchor=anchor, gate=self.quality)

        # 质量统计的归属: fsm_v1 + 按身份门控 + 无区域计数时, 由 FSMLineCounter
        # 按 person_uid 累计; 其余情况由 Counter 按 track_id 累计 (legacy 行为不变)。
        self._line_owns_gate = (
            self.zone is None
            and isinstance(self.line, FSMLineCounter)
            and bool(getattr(self.line, "gate_by_identity", False))
            and self.quality.enabled
        )
        if self._line_owns_gate:
            self.line.owns_gate_observation = True

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(observations) -> List[Tuple[int, Sequence[float], Optional[float]]]:
        """兼容三种观测格式:

        * ``(tid, tlwh)``
        * ``(tid, tlwh, score)``
        * ``(tid, (tlwh, score))``  ← ``streaming.FrameStreamFeeder`` 的嵌套载荷

        第三种是 GPU 流式路径的实际形态 (``pipeline.feed_counter`` 把 ``(tlwh, score)``
        作为 payload 入队, feeder 原样转发)。**必须在这里展开**, 否则 ``tlwh`` 会以
        嵌套 2 元组的形式流到 ``quality.observe`` / ``crossing``, 触发
        ``not enough values to unpack (expected 4, got 2)``。
        """
        out = []
        for ob in observations:
            if len(ob) >= 3:
                out.append((ob[0], ob[1], ob[2]))
                continue
            payload = ob[1]
            if (_seq_len(payload) == 2 and _seq_len(payload[0]) == 4):
                out.append((ob[0], payload[0], payload[1]))
            else:
                out.append((ob[0], payload, None))
        return out

    def update(self, frame_id: int, observations, width: int, height: int):
        obs = self._normalize(observations)
        # 先累计质量统计 (所有输出轨迹, 不只越线的)
        # fsm_v1 按逻辑身份门控时, 统计由 FSMLineCounter 以 person_uid 为键累计,
        # 这里就不再按 track_id 累计 (否则同一份统计被两套键各记一遍)。
        if not self._line_owns_gate:
            for tid, tlwh, score in obs:
                self.quality.observe(tid, tlwh, score, frame_id, width, height)

        line_events, zone_events = [], []
        if self.line is not None:
            line_events = self.line.update(frame_id, obs, width, height)
        if self.zone is not None:
            zone_events = self.zone.update(frame_id, obs, width, height)
        return line_events, zone_events

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        out: dict = {"mode": self.mode, "anchor": self.anchor,
                     "algorithm": self.algorithm}
        if self.line is not None:
            counts = self.line.counts()
            out["line"] = {
                "in": counts["in"],
                "out": counts["out"],
                "net": counts["in"] - counts["out"],
                "in_direction": self.line.in_direction,
                "events": len(self.line.events),
            }
        if self.zone is not None:
            zc = self.zone.counts()
            out["zone"] = {
                "name": self.zone.name,
                "occupancy_final": self.zone.occupancy,
                "enter": zc["enter"],
                "exit": zc["exit"],
                "occupancy_max": max(self.zone._occupancy_series) if self.zone._occupancy_series else 0,
            }
        out["quality"] = self.quality.summary()
        return out

    def diagnostics(self) -> Dict[str, Any]:
        """计数层的可追溯诊断 (legacy 返回配置摘要, fsm_v1 额外返回 FSM/身份统计)。"""
        if self.line is None:
            return {"algorithm": self.algorithm}
        fn = getattr(self.line, "diagnostics", None)
        if callable(fn):
            return fn()
        return self.line.line_describe()

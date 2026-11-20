from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _norm(v) -> Optional[np.ndarray]:
    a = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return None
    return a / n


def _cos(a, b) -> float:
    ua, ub = _norm(a), _norm(b)
    if ua is None or ub is None:
        return 0.0
    return float(np.clip(np.dot(ua, ub), -1.0, 1.0))


# --------------------------------------------------------------------------- #
@dataclass
class _Active:
    track_id: int
    person_uid: str
    anchor: np.ndarray
    first_frame: int
    last_frame: int
    last_ts: float
    bbox_h: float
    velocity: np.ndarray
    n_obs: int = 1
    embedding: Optional[np.ndarray] = None


@dataclass
class _Finished:
    person_uid: str
    track_id: int
    last_anchor: np.ndarray
    last_frame: int
    last_ts: float
    bbox_h: float
    velocity: np.ndarray
    n_obs: int = 0
    embedding: Optional[np.ndarray] = None
    source_track_ids: Tuple[int, ...] = ()


@dataclass
class LinkDecision:
    """一次关联决策的完整记录 (便于"误合并"事后追因)。"""

    frame_id: int
    timestamp: float
    new_track_id: int
    person_uid: str
    linked: bool
    reason: str
    from_track_id: Optional[int] = None
    gap_sec: float = 0.0
    distance_px: float = 0.0
    allowed_px: float = 0.0
    appearance_cos: Optional[float] = None
    n_candidates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "timestamp_sec": round(float(self.timestamp), 3),
            "new_track_id": int(self.new_track_id),
            "person_uid": self.person_uid,
            "linked": bool(self.linked),
            "reason": self.reason,
            "from_track_id": self.from_track_id,
            "gap_sec": round(float(self.gap_sec), 3),
            "distance_px": round(float(self.distance_px), 1),
            "allowed_px": round(float(self.allowed_px), 1),
            "appearance_cos": (None if self.appearance_cos is None
                               else round(float(self.appearance_cos), 4)),
            "n_candidates": int(self.n_candidates),
        }


# --------------------------------------------------------------------------- #
class TrackletLinker:
    """把碎裂的 ``track_id`` 归并成逻辑 ``person_uid``。"""

    def __init__(self, *, enabled: bool = True,
                 max_gap_sec: float = 1.5, fps: float = 25.0,
                 max_dist_px: float = 60.0, max_dist_rel: float = 1.0,
                 max_speed_px_per_frame: float = 40.0,
                 dir_cos_min: float = 0.0, dir_min_move_px: float = 20.0,
                 margin_rel: float = 1.35, margin_px: float = 25.0,
                 use_appearance: bool = True, app_min_cos: float = 0.75,
                 prefix: str = "person",
                 strict_unique_uid: bool = True,
                 prefer_same_track: bool = True,
                 same_track_max_dist_rel: float = 3.0):
        self.enabled = bool(enabled)
        self.max_gap_sec = float(max_gap_sec)
        self.fps = float(fps or 25.0)
        self.max_dist_px = float(max_dist_px)
        self.max_dist_rel = float(max_dist_rel)
        self.max_speed_px_per_frame = float(max_speed_px_per_frame)
        self.dir_cos_min = float(dir_cos_min)
        self.dir_min_move_px = float(dir_min_move_px)
        self.margin_rel = float(margin_rel)
        self.margin_px = float(margin_px)
        self.use_appearance = bool(use_appearance)
        self.app_min_cos = float(app_min_cos)
        self.prefix = str(prefix)
        # ---- 任务C 修复: UID 时间重叠复用 ---------------------------------- #
        # 1) ``strict_unique_uid``: 一个 ``person_uid`` 只要还有**活跃**轨迹在用,
        #    就绝不允许被另一条 tracklet 复用 (模块文档承诺过的约束, 之前未落实)。
        # 2) ``prefer_same_track``: ``track_id`` 是 ByteTrack 给出的**同一物理目标**
        #    的最强证据。若某 ``track_id`` 短暂丢帧后重现, 且它自己上一次的
        #    tracklet 仍在 finished 池内, 就直接续回原 UID —— 否则两条轨迹
        #    同时闪断时会**互相抢走对方的 UID** (实测 `…15490940` 中 tid 34 的
        #    UID 在 f=448 被 person_010 抢走, 而 person_010 原本属于 tid 43)。
        self.strict_unique_uid = bool(strict_unique_uid)
        self.prefer_same_track = bool(prefer_same_track)
        # 同一 track_id 重现时放宽的空间门 (按框高倍数)。理由: track_id 本身
        # 就是 ByteTrack 给出的"同一目标"判定, 它的可信度高于一次欧氏距离;
        # 而"跨轨迹抢占 UID"恰恰发生在锚点瞬移之后 —— 那里距离门必然挡不住,
        # 于是最近邻会选到**别人**的 tracklet。放宽只对"自己的 UID"生效。
        self.same_track_max_dist_rel = float(same_track_max_dist_rel)
        self.n_uid_swap_prevented = 0

        self._active: Dict[int, _Active] = {}
        self._finished: List[_Finished] = []
        self._uid_of: Dict[int, str] = {}          # track_id -> person_uid (全历史)
        self._uid_tracks: Dict[str, List[int]] = {}
        self._decisions: List[LinkDecision] = []
        self._next_id = 0
        self.n_linked = 0
        self.n_created = 0

    # ------------------------------------------------------------------ #
    def _new_uid(self) -> str:
        self._next_id += 1
        return "%s_%03d" % (self.prefix, self._next_id)

    def _remember(self, uid: str, tid: int) -> None:
        self._uid_of[tid] = uid
        self._uid_tracks.setdefault(uid, [])
        if tid not in self._uid_tracks[uid]:
            self._uid_tracks[uid].append(tid)

    # ------------------------------------------------------------------ #
    def update(self, track_id: int, anchor: Sequence[float], frame_id: int,
               timestamp: float, bbox_h: float = 0.0,
               embedding=None) -> Tuple[str, bool]:
        """喂入一帧观测, 返回 ``(person_uid, 是否为本轨迹的首次决策)``。

        ``是否首次决策`` 为 True 时调用方可据此决定是否迁移 FSM 状态。
        """
        tid = int(track_id)
        p = np.asarray(anchor, dtype=float).reshape(2)

        act = self._active.get(tid)
        if act is not None:
            # 速度单位统一为 **像素/帧**, 与 jump_gate / 外推位置保持同一量纲
            dframes = max(int(frame_id) - act.last_frame, 1)
            v = (p - act.anchor) / dframes
            act.velocity = 0.7 * act.velocity + 0.3 * v if act.n_obs > 1 else v
            act.anchor = p
            act.last_frame, act.last_ts = int(frame_id), float(timestamp)
            if bbox_h:
                act.bbox_h = bbox_h if act.bbox_h <= 0 else 0.7 * act.bbox_h + 0.3 * float(bbox_h)
            act.n_obs += 1
            if embedding is not None and self.use_appearance:
                e = np.asarray(embedding, dtype=float).reshape(-1)
                act.embedding = e if act.embedding is None else 0.7 * act.embedding + 0.3 * e
            return act.person_uid, False

        # ---- 新轨迹: 尝试与最近结束的 tracklet 关联 ---- #
        self._prune(float(timestamp))
        uid, decision = self._try_link(tid, p, int(frame_id), float(timestamp),
                                       float(bbox_h), embedding)
        self._remember(uid, tid)
        self._active[tid] = _Active(
            track_id=tid, person_uid=uid, anchor=p, first_frame=int(frame_id),
            last_frame=int(frame_id), last_ts=float(timestamp),
            bbox_h=float(bbox_h), velocity=np.zeros(2),
            embedding=(None if embedding is None else np.asarray(embedding, dtype=float)),
        )
        self._decisions.append(decision)
        if decision.linked:
            self.n_linked += 1
        else:
            self.n_created += 1
        return uid, True

    # ------------------------------------------------------------------ #
    def _try_link(self, tid: int, p: np.ndarray, frame_id: int, ts: float,
                  bbox_h: float, embedding) -> Tuple[str, LinkDecision]:
        if not self.enabled or not self._finished:
            uid = self._new_uid()
            return uid, LinkDecision(frame_id, ts, tid, uid, False, "linker_disabled"
                                     if not self.enabled else "no_candidate")

        allowed = self.max_dist_px + self.max_dist_rel * max(bbox_h, 1.0)

        # ---- 修复 2: 同一 track_id 重现 → 优先续回它自己的 UID ---- #
        prev_uid = self._uid_of.get(tid)
        if self.prefer_same_track and prev_uid is not None:
            own = [f for f in self._finished if f.person_uid == prev_uid]
            if own:
                f = own[0]
                if ts - f.last_ts <= self.max_gap_sec:
                    dist = float(np.linalg.norm(
                        p - (f.last_anchor + f.velocity * min(
                            float(max(frame_id - f.last_frame, 1)),
                            self.max_speed_px_per_frame))))
                    allowed_self = max(
                        allowed, self.same_track_max_dist_rel * max(bbox_h, 1.0))
                    if dist <= allowed_self:
                        self._finished.remove(f)
                        self._remember(f.person_uid, f.track_id)
                        return f.person_uid, LinkDecision(
                            frame_id, ts, tid, f.person_uid, True,
                            "same_track_relink", from_track_id=f.track_id,
                            gap_sec=ts - f.last_ts, distance_px=dist,
                            allowed_px=allowed_self, n_candidates=1)

        # ---- 修复 1: 正在被活跃轨迹占用的 UID 不得复用 ---- #
        active_uids = ({a.person_uid for a in self._active.values()}
                       if self.strict_unique_uid else None)

        scored: List[Tuple[float, _Finished, float, float, Optional[float]]] = []
        for f in self._finished:
            if active_uids is not None and f.person_uid in active_uids:
                self.n_uid_swap_prevented += 1
                continue
            gap = ts - f.last_ts
            if gap < 0 or gap > self.max_gap_sec:
                continue
            dframes = max(frame_id - f.last_frame, 1)
            pred = f.last_anchor + f.velocity * min(float(dframes),
                                                    self.max_speed_px_per_frame)
            dist = float(np.linalg.norm(p - pred))

            # 运动方向一致性 (仅在位移足够大且历史速度可信时启用)
            move = p - f.last_anchor
            move_n = float(np.linalg.norm(move))
            speed = float(np.linalg.norm(f.velocity))
            if speed > 0.5 and move_n > max(self.dir_min_move_px, 0.3 * max(bbox_h, 1.0)):
                if _cos(move, f.velocity) < self.dir_cos_min:
                    continue

            # 外观门控 (可选): 没有 embedding 时不做外观判断, 只靠时空
            acos = None
            if (self.use_appearance and embedding is not None
                    and f.embedding is not None):
                acos = _cos(embedding, f.embedding)
                if acos < self.app_min_cos:
                    continue

            if dist > allowed:
                continue
            scored.append((dist, f, gap, allowed, acos))

        if not scored:
            uid = self._new_uid()
            return uid, LinkDecision(frame_id, ts, tid, uid, False, "no_candidate",
                                     n_candidates=0)

        scored.sort(key=lambda s: s[0])
        best = scored[0]
        # 唯一性: 存在第二名且优势不足 → 拿不准, 不合并
        if len(scored) >= 2:
            second = scored[1]
            if not (best[0] * self.margin_rel + self.margin_px <= second[0]):
                uid = self._new_uid()
                return uid, LinkDecision(
                    frame_id, ts, tid, uid, False, "ambiguous_candidates",
                    from_track_id=best[1].track_id, gap_sec=best[2],
                    distance_px=best[0], allowed_px=best[3],
                    appearance_cos=best[4], n_candidates=len(scored))

        f = best[1]
        self._finished.remove(f)
        self._remember(f.person_uid, f.track_id)
        return f.person_uid, LinkDecision(
            frame_id, ts, tid, f.person_uid, True, "linked_unique_best",
            from_track_id=f.track_id, gap_sec=best[2], distance_px=best[0],
            allowed_px=best[3], appearance_cos=best[4], n_candidates=len(scored))

    # ------------------------------------------------------------------ #
    def end(self, track_id: int, frame_id: Optional[int] = None,
            timestamp: Optional[float] = None) -> None:
        """轨迹消失时调用, 把活跃轨迹移入 finished 池。"""
        tid = int(track_id)
        act = self._active.pop(tid, None)
        if act is None:
            return
        self._finished.append(_Finished(
            person_uid=act.person_uid, track_id=tid, last_anchor=act.anchor,
            last_frame=(act.last_frame if frame_id is None else int(frame_id)),
            last_ts=(act.last_ts if timestamp is None else float(timestamp)),
            bbox_h=act.bbox_h, velocity=act.velocity, n_obs=act.n_obs,
            embedding=act.embedding,
            source_track_ids=tuple(self._uid_tracks.get(act.person_uid, [tid])),
        ))

    def _prune(self, now_ts: float) -> None:
        keep = []
        for f in self._finished:
            if now_ts - f.last_ts <= self.max_gap_sec:
                keep.append(f)
        self._finished = keep

    # ------------------------------------------------------------------ #
    def uid_of(self, track_id: int) -> Optional[str]:
        return self._uid_of.get(int(track_id))

    def tracks_of(self, uid: str) -> Tuple[int, ...]:
        return tuple(self._uid_tracks.get(str(uid), ()))

    def active_uids(self) -> List[str]:
        return [a.person_uid for a in self._active.values()]

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_track_ids": len(self._uid_of),
            "n_person_uids": len(self._uid_tracks),
            "n_linked": self.n_linked,
            "n_created": self.n_created,
            "n_active": len(self._active),
            "n_finished": len(self._finished),
            "max_gap_sec": self.max_gap_sec,
            "max_dist_px": self.max_dist_px,
            "max_dist_rel": self.max_dist_rel,
            "dir_cos_min": self.dir_cos_min,
            "appearance_used": bool(self.use_appearance),
            "app_min_cos": self.app_min_cos,
            "strict_unique_uid": self.strict_unique_uid,
            "prefer_same_track": self.prefer_same_track,
            "n_uid_swap_prevented": self.n_uid_swap_prevented,
            "decisions": [d.to_dict() for d in self._decisions],
        }

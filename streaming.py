from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


class FrameStreamFeeder:
    """按帧升序、有界延迟地把观测提交给计数层。

    参数
    ----
    update_fn : ``(frame_id, batch) -> None``
        计数层的喂入函数; ``batch`` 为 ``[(identity, tlwh, score), ...]`` 或 ``None``
        (空帧)。**每一帧都会被调用一次**, 包括没有任何观测的空帧。
    max_buffered : int
        允许水位落后的最大帧数; 超限即强制定案最老的未定身份。
    enabled : bool
        False 时退化为"收尾统一回放"(旧行为), 用于等价性对照。
    """

    def __init__(self, update_fn: Callable[[int, Optional[list]], None], *,
                 max_buffered: int = 600, enabled: bool = True):
        self._update = update_fn
        self.max_buffered = max(1, int(max_buffered))
        self.enabled = bool(enabled)

        self._obs: Dict[int, List[Tuple[Any, Any, Any]]] = defaultdict(list)
        self._pending_min: Dict[Any, int] = {}          # internal -> 最早未定观测帧
        self._pending_export: Dict[Any, int] = {}       # internal -> 当前 export id
        self.next_feed_frame: int = 1
        self.max_frame_seen: int = 0

        # 强制定案时的回调: (internal, export_id) -> None
        self.on_force_finalize: Optional[Callable[[Any, int], None]] = None

        self.stats: Dict[str, int] = {
            "fed_frames": 0, "max_lag_frames": 0, "forced_final": 0,
        }

    # ------------------------------------------------------------------ #
    # 生产者侧
    # ------------------------------------------------------------------ #
    def add(self, frame_id: int, identity: Any, payload: Any = None) -> None:
        """登记一条观测 (允许晚于 ``frame_id`` 才被调用)。"""
        f = int(frame_id)
        self._obs[f].append((identity, payload))
        if f > self.max_frame_seen:
            self.max_frame_seen = f

    def note_pending(self, internal: Any, frame_id: int, export_id: int = -1) -> None:
        """标记内部身份 ``internal`` 尚未定案 → 水位不得越过 ``frame_id``。"""
        f = int(frame_id)
        prev = self._pending_min.get(internal)
        self._pending_min[internal] = f if prev is None else min(prev, f)
        self._pending_export[internal] = int(export_id)

    def resolve(self, internal: Any) -> None:
        """内部身份已定案 → 解除水位阻塞。"""
        self._pending_min.pop(internal, None)
        self._pending_export.pop(internal, None)

    # ------------------------------------------------------------------ #
    # 消费者侧
    # ------------------------------------------------------------------ #
    def barrier(self) -> int:
        """可安全提交的帧号上界 (不含)。"""
        if not self.enabled:
            return 10 ** 18
        return min(self._pending_min.values()) if self._pending_min else 10 ** 18

    def _feed_upto(self, upto: int) -> None:
        while self.next_feed_frame < upto:
            batch = self._obs.pop(self.next_feed_frame, None)
            if batch:
                batch.sort(key=lambda o: str(o[0]))   # 同帧内按 id 排序, 保证可复现
            self._update(self.next_feed_frame, batch if batch else None)
            self.stats["fed_frames"] += 1
            self.next_feed_frame += 1

    def force_finalize_oldest(self) -> bool:
        """冻结最早未定的身份, 使其观测可被提交。返回是否发生了动作。"""
        if not self._pending_min:
            return False
        internal = min(self._pending_min, key=lambda k: self._pending_min[k])
        export = int(self._pending_export.get(internal, internal))
        self._pending_min.pop(internal, None)
        self._pending_export.pop(internal, None)
        self.stats["forced_final"] += 1
        if self.on_force_finalize is not None:
            self.on_force_finalize(internal, export)
        return True

    def tick(self, current_frame: int) -> None:
        """每处理完一帧调用一次: 提交可安全提交的帧, 并执行缓存上限保护。"""
        if not self.enabled:
            return
        upto = min(self.barrier(), int(current_frame) + 1)
        self._feed_upto(upto)
        lag = (int(current_frame) + 1) - self.next_feed_frame
        if lag > self.stats["max_lag_frames"]:
            self.stats["max_lag_frames"] = lag
        # 缓存超限 → 强制定案, 直到水位跟上或没有未定身份
        guard = 0
        while ((int(current_frame) + 1) - self.next_feed_frame) > self.max_buffered \
                and self._pending_min and guard < 10000:
            guard += 1
            if not self.force_finalize_oldest():
                break
            upto = min(self.barrier(), int(current_frame) + 1)
            self._feed_upto(upto)

    def flush(self) -> None:
        """收尾: 把剩余帧 (直到 max_frame_seen) 全部提交。"""
        if not self.enabled:
            # 兼容旧行为: 从第 1 帧一路补到最后一帧
            self._feed_upto(self.max_frame_seen + 1)
            return
        self.clear_pending()
        self._feed_upto(max(self.max_frame_seen + 1, self.next_feed_frame))

    def clear_pending(self) -> None:
        """解除全部水位阻塞 (收尾时调用: 残留未定身份已按当前 id 补发完毕)。"""
        self._pending_min.clear()
        self._pending_export.clear()

    # ------------------------------------------------------------------ #
    @property
    def n_buffered_frames(self) -> int:
        return len(self._obs)

    def lag_sec(self, fps: float) -> float:
        return round(self.stats["max_lag_frames"] / max(float(fps), 1e-6), 3)

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fed_frames": self.stats["fed_frames"],
            "max_lag_frames": self.stats["max_lag_frames"],
            "forced_final": self.stats["forced_final"],
            "max_buffered_frames": self.max_buffered,
            "residual_buffered_frames": self.n_buffered_frames,
            "next_feed_frame": self.next_feed_frame,
        }

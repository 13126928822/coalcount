"""性能与资源消耗统计: FPS / 时延分位 / 显存 / 内存 / CPU / 分阶段耗时."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np


def _rss_mb() -> Optional[float]:
    """进程常驻内存 (MB)。Linux 用 resource, 其它平台尽力而为。"""
    try:
        import resource  # noqa: WPS433 (仅 Unix)

        # ru_maxrss: Linux 单位 KB, macOS 单位 Byte
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        return v / 1024.0 if sys.platform != "darwin" else v / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


class PerfMonitor:
    """按帧统计总耗时与各阶段耗时, 并记录峰值资源占用。"""

    STAGES = ("extract", "detect", "mask", "track", "count", "viz", "write")

    def __init__(self, enabled: bool = True, sync_cuda: bool = True, warmup_frames: int = 3):
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.warmup_frames = warmup_frames
        self._t0 = None
        self._frames = 0
        self._frame_times: List[float] = []
        self._stage_times: Dict[str, List[float]] = {s: [] for s in self.STAGES}
        self._stage_cur: Dict[str, float] = {}
        self._peak_gpu_mb: Optional[float] = None
        self._start_rss = _rss_mb()
        self._peak_rss = self._start_rss
        self._cpu_t0 = None
        self._cpu_t1 = None

    # ------------------------------------------------------------------ #
    def start(self):
        if not self.enabled:
            return
        self._t0 = time.perf_counter()
        self._cpu_t0 = os.times()
        self._reset_gpu_peak()

    def _reset_gpu_peak(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        except Exception:
            pass

    def _sync(self):
        if not self.sync_cuda:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t
            self._stage_cur[name] = self._stage_cur.get(name, 0.0) + dt
            if name in self._stage_times:
                self._stage_times[name].append(dt)

    def end_frame(self, frame_index: int):
        if not self.enabled:
            return
        self._sync()
        self._frames += 1
        if self._frames > self.warmup_frames:
            now = time.perf_counter()
            self._frame_times.append(now - self._last)
        self._last = time.perf_counter()
        # 记录峰值资源
        try:
            import torch

            if torch.cuda.is_available():
                cur = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self._peak_gpu_mb = max(self._peak_gpu_mb or 0.0, cur)
        except Exception:
            pass
        r = _rss_mb()
        if r is not None:
            self._peak_rss = max(self._peak_rss or 0.0, r)

    def begin_frames(self):
        """在所有初始化完成后调用, 标记计时起点。"""
        if not self.enabled:
            return
        self._sync()
        self._last = time.perf_counter()

    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        total = (time.perf_counter() - self._t0) if self._t0 else 0.0
        ft = np.array(self._frame_times, dtype=float) if self._frame_times else np.array([0.0])
        cpu = None
        if self._cpu_t0 is not None:
            t1 = os.times()
            wall = max(total, 1e-9)
            cpu = (t1.user + t1.system - self._cpu_t0.user - self._cpu_t0.system) / wall * 100.0
        stages = {
            k: (float(np.mean(v)) * 1000.0 if v else 0.0) for k, v in self._stage_times.items()
        }
        measured = len(self._frame_times)
        fps = (measured / ft.sum()) if ft.sum() > 0 else 0.0
        return {
            "enabled": True,
            "frames": self._frames,
            "frames_measured": measured,
            "warmup_frames": self.warmup_frames,
            "total_s": round(total, 3),
            "fps": round(fps, 2),
            "latency_ms": {
                "mean": round(float(ft.mean()) * 1000.0, 2),
                "p50": round(float(np.percentile(ft, 50)) * 1000.0, 2),
                "p95": round(float(np.percentile(ft, 95)) * 1000.0, 2),
                "max": round(float(ft.max()) * 1000.0, 2),
            },
            "stage_ms_mean": {k: round(v, 2) for k, v in stages.items()},
            "peak_gpu_mem_mb": round(self._peak_gpu_mb, 1) if self._peak_gpu_mb else None,
            "rss_mb": round(self._peak_rss, 1) if self._peak_rss else None,
            "cpu_percent": round(cpu, 1) if cpu is not None else None,
        }

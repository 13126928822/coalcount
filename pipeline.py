from __future__ import annotations
# NOTE: this is a partial-disclosure build. Calling run_video() without the omitted core modules raises a clear RuntimeError.

import json
import os
import os.path as osp
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
# PARTIAL DISCLOSURE: core imports are replaced by explicit API stubs.
from ._core_api import (Counter, _anchor_point, load_store, lookup_line,
                        apply_to_config, describe_line, scene_name_from_path,
                        CONFIG_KEYS, FrameStreamFeeder)
from .models import Detector, build_mask_manager, build_tracker
from .perf import PerfMonitor
from . import viz as vizmod

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("coalcount")

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")


@dataclass
class RunResult:
    name: str
    input_path: str
    ok: bool = True
    error: str = ""
    frames: int = 0
    width: int = 0
    height: int = 0
    fps_src: float = 0.0
    unique_ids: int = 0
    counts: Dict[str, Any] = field(default_factory=dict)
    perf: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input": self.input_path,
            "ok": self.ok,
            "error": self.error,
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "fps_src": round(self.fps_src, 3),
            "unique_ids": self.unique_ids,
            "counts": self.counts,
            "perf": self.perf,
            "outputs": self.outputs,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# --------------------------------------------------------------------------- #
def _list_frames(d: str) -> List[str]:
    fs = [osp.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMAGE_EXT)]
    fs.sort()
    return fs


# 常见的"容器目录名" (frames/img1/...) 的判定在 lines.scene_name_from_path 中统一维护


def derive_scene_name(path: str) -> str:
    """从输入路径推导场景名。若末级目录是 frames/img1 这类通用名, 取上一级。"""
    return scene_name_from_path(path)


class FrameSource:
    """统一的帧来源: 视频(内存流式解码, 不落盘) 或 帧目录。

    ``iter_frames()`` 每次调用返回一个**新的**迭代器, 产出 ``(序号, 图像或路径)``:
      * 视频模式 → 重新打开解码器 (seek 一次后顺序读, 避免反复定位导致坏帧),
        直接产出解码后的 ndarray (内存流式, **不写盘**);
      * 帧目录模式 → 直接产出路径, 由 ``Detector.infer`` 内部 imread。

    这样"视频输入"不必先抽帧到磁盘, 也天然支持只处理前 N 秒/N 帧。
    """

    def __init__(self, n, fps, width, height, *, paths=None, video_path=None,
                 start_frame=1, stride=1, max_frames=0, frames_dir=None,
                 extracted=False, save_dir=None, pace_fps=0.0):
        self.n = int(n)
        self.fps = float(fps)
        # 实时节流: >0 时按该帧率投喂 (模拟摄像头实时输入), 用于生产压测。
        self.pace_fps = float(pace_fps or 0.0)
        self.width = int(width)
        self.height = int(height)
        self.paths = paths
        self.video_path = video_path
        self.start_frame = int(start_frame)
        self.stride = max(1, int(stride))
        self.max_frames = int(max_frames)
        self.frames_dir = frames_dir
        self.extracted = bool(extracted)
        self.save_dir = save_dir

    @property
    def lazy(self) -> bool:
        """True = 视频内存流式 (逐帧解码, 不落盘)。"""
        return self.paths is None

    def iter_frames(self):
        if self.paths is not None:
            _t0 = time.perf_counter()
            for i, p in enumerate(self.paths, 1):
                if self.pace_fps > 0:
                    _dt = (_t0 + i / self.pace_fps) - time.perf_counter()
                    if _dt > 0:
                        time.sleep(_dt)
                yield i, p
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        if self.start_frame > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame - 1)
        idx = 0
        taken = 0
        _t0 = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % self.stride != 0:
                    continue
                taken += 1
                if self.pace_fps > 0:
                    _dt = (_t0 + taken / self.pace_fps) - time.perf_counter()
                    if _dt > 0:
                        time.sleep(_dt)
                yield taken, frame
                if self.max_frames and taken >= self.max_frames:
                    break
        finally:
            cap.release()


def open_source(
    input_path: str,
    work_dir: str,
    start_frame: int = 1,
    max_frames: int = 0,
    max_seconds: float = 0.0,
    stride: int = 1,
    stride_allowed: bool = True,
    materialize: bool = False,
    decode: str = "memory",
    pace_fps: float = 0.0,
) -> FrameSource:
    """打开输入为 FrameSource。

    * 帧目录输入 → 直接按路径读 (可选 materialize 复制成连续编号目录, 供 EdgeTAM 使用);
    * 视频输入 → 默认 ``decode="memory"`` 内存流式解码 (不落盘);
      ``decode="extract"`` 或 ``materialize=True`` 时退回"先抽帧到磁盘"的老路径。
    """
    if osp.isdir(input_path):
        all_files = _list_frames(input_path)
        if not all_files:
            raise FileNotFoundError(f"目录内没有图像: {input_path}")
        # 帧目录输入同样支持 start_frame / stride / max_frames
        sel = all_files[max(0, start_frame - 1):]
        if stride > 1:
            sel = sel[::stride]
        if max_frames:
            sel = sel[:max_frames]
        if not sel:
            raise RuntimeError(f"抽帧后为空: {input_path}")

        if materialize:
            frames_dir = osp.join(work_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for f in os.listdir(frames_dir):
                if f.lower().endswith(IMAGE_EXT):
                    os.remove(osp.join(frames_dir, f))
            new_files = []
            for i, src in enumerate(sel, 1):
                dst = osp.join(frames_dir, f"{i:06d}.jpg")
                shutil.copyfile(src, dst)
                new_files.append(dst)
            sel = new_files
        else:
            frames_dir = input_path

        img0 = cv2.imread(sel[0])
        if img0 is None:
            raise RuntimeError(f"无法读取帧: {sel[0]}")
        h, w = img0.shape[:2]
        return FrameSource(
            len(sel), 25.0 / float(stride), w, h, paths=sel,
            frames_dir=frames_dir, extracted=bool(materialize),
            save_dir=frames_dir, pace_fps=pace_fps,
        )

    if not osp.isfile(input_path):
        raise FileNotFoundError(f"输入不存在: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if W <= 0 or H <= 0:
        ok0, f0 = cap.read()
        if ok0:
            H, W = f0.shape[:2]
    cap.release()

    if not stride_allowed:
        stride = 1
    # 时长上限 (秒) → 帧数上限
    if max_seconds and max_seconds > 0:
        cap_n = int(max_seconds * fps / max(stride, 1))
        max_frames = cap_n if not max_frames else min(max_frames, cap_n)

    if decode == "extract" or materialize:
        return _extract_video_frames(input_path, work_dir, start_frame, max_frames, stride, fps)

    # ---- 内存流式解码 (不落盘) ---- #
    avail = max(0, total - (start_frame - 1)) if total > 0 else 0
    if avail > 0:
        n = int(np.ceil(avail / float(stride)))
        if max_frames:
            n = min(n, max_frames)
    else:
        n = int(max_frames) if max_frames else 0
    logger.info(
        f"[source] 视频内存流式解码 (不抽帧): {osp.basename(input_path)} "
        f"{W}x{H} fps={fps:.2f} 计划帧数≈{n if n else '未知'}"
    )
    return FrameSource(
        n, fps / float(stride), W, H, video_path=input_path,
        start_frame=start_frame, stride=stride, max_frames=max_frames,
        extracted=False, save_dir=work_dir, pace_fps=pace_fps,
    )


def _extract_video_frames(input_path, work_dir, start_frame, max_frames, stride, fps):
    """把视频抽帧到 work_dir/frames (EdgeTAM 需要帧目录时使用)。"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    frames_dir = osp.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.lower().endswith(IMAGE_EXT):
            os.remove(osp.join(frames_dir, f))

    # 长 H.264/HEVC 监控流: 反复 set(POS_FRAMES) 会读到坏帧 → seek 一次后顺序读
    if start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    files: List[str] = []
    idx = 0
    taken = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride != 0:
            continue
        out_name = osp.join(frames_dir, f"{taken + 1:06d}.jpg")
        cv2.imwrite(out_name, frame)
        files.append(out_name)
        taken += 1
        if max_frames and taken >= max_frames:
            break
    cap.release()

    if not files:
        raise RuntimeError(f"未抽取到任何帧: {input_path}")
    img0 = cv2.imread(files[0])
    h, w = img0.shape[:2]
    return FrameSource(len(files), fps / float(stride), w, h, paths=files,
                       frames_dir=frames_dir, extracted=True, save_dir=frames_dir)


# --------------------------------------------------------------------------- #
def _load_detector(cfg: Config, device):
    return Detector.build(cfg, device)


def calibrate_auto_line(cfg: Config, fr: "FrameSource", detector, device, fps: float,
                        anchor: str = "bottom_center"):
    """用轨迹主运动方向自动标定计数线 (适配任意相机角度)。

    原理:
      1. 跑一遍 检测+跟踪 (不做计数/可视化), 收集每条轨迹的起止锚点;
      2. 取位移显著的轨迹, 用单位位移向量之和估计主运动方向 theta
         (单位向量求和比平均位移更抗"个别快速目标"的干扰);
      3. 计数线过画面中心、与主运动方向垂直, 并把"沿主运动方向穿越"定义为入井;
      4. 场景几乎静止(无显著运动)时返回 None, 上层回退到配置里的固定线。

    返回 ``(p1_norm, p2_norm, in_direction, in_normal)`` 或 None。

    ``in_normal`` 是**朝入井侧的单位法向量**, 等于估计出的主运动方向 —— 因此
    自动线也走显式方向语义, 与人工线、UI 箭头、计数逻辑完全同一套定义。
    """
    tracker = build_tracker(cfg, save_folder=fr.save_dir, frame_rate=fps)
    min_age = int(cfg.get("counter.line.min_track_age", 4))
    min_disp = float(cfg.get("counter.line.auto_min_displacement", 25.0))

    traj = defaultdict(list)
    for i, img in fr.iter_frames():
        outputs, img_info = detector.infer(img)
        if outputs[0] is None:
            continue
        try:
            targets, *_ = tracker.update(
                outputs[0], [img_info["height"], img_info["width"]], detector.test_size,
                prediction_mask=None, tracklet_mask_dict=None,
                frame_img=img_info["raw_img"], vis_type="no_vis", dets_from_file=False,
            )
        except Exception:
            continue
        for t in targets:
            traj[int(t.track_id)].append(_anchor_point(t.last_det_tlwh, anchor))

    unit = []
    for tid, seq in traj.items():
        if len(seq) < min_age:
            continue
        v = np.asarray(seq[-1], dtype=float) - np.asarray(seq[0], dtype=float)
        n = float(np.linalg.norm(v))
        if n >= min_disp:
            unit.append(v / n)

    if len(unit) < 3:
        logger.warning(f"[auto-line] 显著运动轨迹不足 ({len(unit)}), 回退固定线")
        return None

    m = np.sum(np.asarray(unit), axis=0)
    if float(np.linalg.norm(m)) < 0.5:   # 方向分散(如双向人流) → 不强行定线
        logger.warning("[auto-line] 运动方向过于分散, 回退固定线")
        return None
    theta = float(np.arctan2(m[1], m[0]))

    W, H = fr.width, fr.height
    flow = np.array([np.cos(theta), np.sin(theta)])
    d = np.array([-flow[1], flow[0]])          # 垂直于主运动方向
    L = 1.8 * float(np.hypot(W, H))
    c = np.array([W / 2.0, H / 2.0])
    # p1/p2 的顺序决定箭头 (normal = (-dy, dx)) 朝向。取 p1 = c + 0.5L*d,
    # p2 = c - 0.5L*d 使 normal 与主运动方向 flow 同向 —— 即"顺着箭头 = 顺着
    # 主运动 = 进井", 与人工画线的语义 (箭头标 IN) 一致。
    p1 = c + 0.5 * L * d
    p2 = c - 0.5 * L * d
    p1n = [float(p1[0] / W), float(p1[1] / H)]
    p2n = [float(p2[0] / W), float(p2[1] / H)]
    logger.info(
        f"[auto-line] 主运动方向 theta={np.degrees(theta):.1f}deg, "
        f"线 p1={np.round(p1n,3).tolist()} p2={np.round(p2n,3).tolist()} (n_traj={len(unit)})"
    )
    return p1n, p2n, "fwd", [float(flow[0]), float(flow[1])]


def run_video(
    cfg: Config,
    input_path: str,
    out_dir: Optional[str] = None,
    name: Optional[str] = None,
) -> RunResult:
    """跑一个视频/帧序列场景, 返回 RunResult (不抛异常, 失败信息写入 result)。"""
    t_start = time.perf_counter()
    name = name or derive_scene_name(input_path)
    out_dir = out_dir or osp.join(cfg.path(cfg.get("io.out_dir")), name)
    os.makedirs(out_dir, exist_ok=True)

    res = RunResult(name=name, input_path=input_path)

    import torch

    device = cfg.get("run.device", "cuda:0")
    if "cuda" in str(device) and not torch.cuda.is_available():
        logger.warning("[run] CUDA 不可用, 回退 CPU")
        device = "cpu"
    device = torch.device(device)

    mask_enabled = bool(cfg.get("mask.enabled", False))
    stride = int(cfg.get("io.stride", 1))
    if mask_enabled and stride != 1:
        logger.warning("[run] 掩码传播开启时不支持 stride>1, 已强制 stride=1")
        stride = 1

    try:
        fr = open_source(
            input_path,
            work_dir=out_dir,
            start_frame=int(cfg.get("io.start_frame", 1)),
            max_frames=int(cfg.get("io.max_frames", 0)),
            max_seconds=float(cfg.get("io.max_seconds", 0.0) or 0.0),
            stride=stride,
            stride_allowed=not mask_enabled,
            materialize=mask_enabled,
            decode=str(cfg.get("io.decode", "memory")),
            pace_fps=float(cfg.get("io.pace_fps", 0.0) or 0.0),
        )
    except Exception as e:
        res.ok, res.error = False, f"open_source: {e}"
        res.elapsed_s = time.perf_counter() - t_start
        return res

    W, H, fps = fr.width, fr.height, fr.fps
    if cfg.get("io.fps_override"):
        fps = float(cfg.get("io.fps_override"))
    res.frames, res.width, res.height, res.fps_src = fr.n, W, H, fps
    logger.info(f"[run] {name}: {fr.n or '?'} frames @ {W}x{H}, fps={fps:.2f}, "
                f"mode={'memory' if fr.lazy else 'frames'}")

    # ---- 模型 ---- #
    try:
        detector = _load_detector(cfg, device)
    except Exception as e:
        res.ok, res.error = False, f"detector: {e}"
        res.elapsed_s = time.perf_counter() - t_start
        return res

    mask_manager = None
    if mask_enabled:
        try:
            mask_manager = build_mask_manager(cfg, fr.frames_dir)
        except Exception as e:
            logger.warning(f"[run] 掩码管理器初始化失败, 降级为不使用掩码: {e}")
            mask_manager = None

    # ---- 计数线解析: 人工绘制 > 自动标定 > 配置固定 ---- #
    # 注意: 批量运行时 cfg 是**共享对象**。下面会临时改写 counter.line.* / anchor,
    # 必须在 Counter 构造完成后立刻还原, 否则本条视频的线会泄漏给后续视频
    # (人工线会把后续场景的 auto 误关、并沿用上一条线)。
    _line_keys = CONFIG_KEYS
    orig_line = {k: cfg.get(k) for k in _line_keys}

    line_source = "config"
    line_file = cfg.get("counter.line_file")
    if line_file:
        try:
            store = load_store(cfg.path(line_file))
            rec = lookup_line(store, input_path, name)
            if rec is not None:
                apply_to_config(cfg, rec)
                line_source = f"manual:{name}"
                logger.info(f"[line] {name}: 使用人工绘制计数线 {describe_line(rec)}")
            else:
                logger.info(f"[line] {name}: 未找到人工计数线, 回退配置/自动")
        except Exception as e:
            logger.warning(f"[line] 读取计数线文件失败, 忽略: {e}")

    # ---- 自动计数线标定 (可选) ---- #
    if line_source == "config" and cfg.get("counter.line.auto", False):
        if mask_manager is not None:
            logger.warning("[auto-line] 与掩码传播同时开启会显著变慢; 本次标定不使用掩码")
        try:
            auto = calibrate_auto_line(
                cfg, fr, detector, device, fps, anchor=cfg.get("counter.anchor", "bottom_center")
            )
            if auto is not None:
                p1n, p2n, indir, nrm = auto
                cfg.set("counter.line.p1", p1n)
                cfg.set("counter.line.p2", p2n)
                cfg.set("counter.line.in_direction", indir)
                cfg.set("counter.line.in_normal", list(nrm))
                cfg.set("counter.line.line_id", "auto")
                line_source = "auto"
        except Exception as e:
            logger.warning(f"[auto-line] 标定失败, 使用固定线: {e}")

    tracker = build_tracker(cfg, save_folder=out_dir, frame_rate=fps)
    counter = Counter(cfg, fps=fps)
    anchor_mode = cfg.get("counter.anchor", "bottom_center")

    # 记录本次实际生效的计数线, 随后立即还原共享 cfg
    line_used: Dict[str, Any] = {
        "source": line_source,
        "p1": list(cfg.get("counter.line.p1")),
        "p2": list(cfg.get("counter.line.p2")),
        "in_direction": (counter.line.in_direction if counter.line is not None
                         else cfg.get("counter.line.in_direction")),
        "in_normal": cfg.get("counter.line.in_normal"),
        "ref_size": cfg.get("counter.line.ref_size"),
        "line_id": cfg.get("counter.line.line_id"),
        "line_version": cfg.get("counter.line.line_version"),
        "roi": cfg.get("counter.line.roi"),
        "seg_tolerance": cfg.get("counter.line.seg_tolerance"),
        "anchor": anchor_mode,
    }
    if line_source.startswith("manual") and line_file:
        line_used["file"] = cfg.path(line_file)
    for _k, _v in orig_line.items():
        cfg.set(_k, _v)
    perf = PerfMonitor(
        enabled=bool(cfg.get("perf.enabled", True)),
        sync_cuda=bool(cfg.get("perf.sync_cuda", True)),
        warmup_frames=int(cfg.get("perf.warmup_frames", 3)),
    )

    save_video = bool(cfg.get("io.save_video", True))
    save_mot = bool(cfg.get("io.save_mot", True))
    vis_on = bool(cfg.get("io.vis", True))
    vis_scale = float(cfg.get("io.vis_scale", 1.0))
    writer = None
    if save_video:
        vpath = osp.join(out_dir, "vis.mp4")
        writer = cv2.VideoWriter(
            vpath, cv2.VideoWriter_fourcc(*str(cfg.get("io.video_fourcc", "mp4v"))),
            max(fps, 1.0), (int(W * vis_scale), int(H * vis_scale)),
        )
        res.outputs["video"] = vpath

    # ---- 逐帧主循环 ---- #
    prediction = None
    tracklet_mask_dict = None
    online_tlwhs: List[Any] = []
    online_ids_mask_manager: List[int] = []
    new_tracks: List[Any] = []
    removed_tracks_ids: List[int] = []

    mot_records: List[str] = []
    pending_records = defaultdict(list)     # internal_id -> [(frame, tlwh, score)]
    pending_obs = defaultdict(list)          # internal_id -> [(frame, tlwh, score)]
    seen_identities = set()
    id_remap: Dict[int, int] = {}            # internal -> final export id
    frame_times: List[float] = []
    n_det_total = 0                          # 累计检出框数 (含未成轨迹的)
    n_det_hi_total = 0                       # 累计"高置信"检出 (score >= track_thresh)
    n_frames_with_det = 0
    track_thresh = float(cfg.get("tracker.track_thresh", 0.5))

    draw_traj = bool(cfg.get("io.draw_trajectory", True)) and (vis_on or save_video)
    trail_len = max(2, int(cfg.get("io.trail_len", 60)))
    trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))

    # ★ 计数器消费契约: update() 期望"一帧一次, 且携带该帧全部观测"; 并且必须
    #   覆盖每一帧(含无观测的空帧), 否则 FSM 无法判定轨迹"已消失", 身份无法收尾。
    #   旧实现逐条喂入 → 同一帧内其余轨迹被判为 gone, 每帧反复 end/重建身份,
    #   导致 person_uid 数量虚高、FSM 双稳定区证据被切断。legacy 算法对同一 track
    #   的内部状态是帧序递推的, 交付顺序改变不影响其结果。
    #
    # ★ V3 流式改造: 不再"视频结束后统一回放", 而是**边跑边按帧升序喂入**。
    #   难点在于 tracker 的 export_id 可能迟到 (身份未定期间观测要先缓存),
    #   因此某些帧的观测会在更晚的时刻才被"补齐"。做法是维护一个**提交水位**
    #   (barrier = 所有未定身份中最早的观测帧号): [next_feed_frame, barrier)
    #   内的帧已不可能再变化, 立即按帧升序喂给计数层 (含空帧)。这样计数结果与
    #   HUD 都是准实时的, 延迟上界 = 身份判定延迟, 而不是整段视频。
    #
    #   缓存上限 ``counter.max_buffered_frames``: 若水位落后当前帧太多 (身份迟迟
    #   未定, 例如目标长期被遮挡), 就强制冻结最老的未定身份并推进水位, 保证
    #   内存有界、延迟有上界。收尾时把残留帧一次喂完 —— 最终结果与"结束后统一
    #   回放"逐帧一致 (见 tools/_test_streaming.py)。
    #
    #   全部水位/缓存逻辑封装在 coalcount.streaming.FrameStreamFeeder, 便于单测。
    def _to_obs(batch):
        """把 feeder 的批原样交给计数层。

        批元素形如 ``(identity, (tlwh, score))``; **展开由
        ``Counter._normalize`` 统一负责** (它在兼容层里同时接受
        ``(tid, tlwh)`` / ``(tid, tlwh, score)`` / ``(tid, (tlwh, score))``),
        避免在这里再写一份容易与单测脱节的适配逻辑。
        """
        return batch if batch else ()

    feeder = FrameStreamFeeder(
        lambda f, batch: counter.update(f, _to_obs(batch), W, H),
        max_buffered=int(cfg.get("counter.max_buffered_frames", 600)),
        enabled=bool(cfg.get("counter.streaming", True)),
    )

    def feed_counter(identity: int, frame_id: int, tlwh, score=None):
        # 只入队, 不立即计算 —— 由 feeder 按帧升序提交。
        feeder.add(int(frame_id), int(identity), (tlwh, score))

    def _on_force_finalize(internal, export):
        """缓存超限时冻结该身份: 定案 id 并补发它此前缓存的观测/记录。"""
        id_remap[internal] = export
        seen_identities.add(export)
        for (f, tlwh_, sc_) in pending_obs.pop(internal, []):
            feed_counter(export, f, tlwh_, sc_)
        for (f, tlwh_, sc_) in pending_records.pop(internal, []):
            mot_records.append(_mot_line(f, export, tlwh_, sc_))

    feeder.on_force_finalize = _on_force_finalize

    # ---- 生产压测: 逐帧记录流式缓存水位 (输入队列长度) ---- #
    _qlog_path = str(cfg.get("io.queue_log", "") or "")
    _qlog = None
    if _qlog_path:
        try:
            os.makedirs(osp.dirname(osp.abspath(_qlog_path)), exist_ok=True)
            _qlog = open(_qlog_path, "w", encoding="utf-8")
            _qlog.write("frame,wall_s,lag_frames,n_buffered\n")
        except Exception as _e:  # pragma: no cover
            logger.warning(f"[p4] 无法写队列日志 {_qlog_path}: {_e}")
            _qlog = None

    perf.start()
    perf.begin_frames()

    n_processed = 0
    for i, img in fr.iter_frames():
        n_processed = i
        t_frame = time.perf_counter()
        with perf.stage("detect"):
            outputs, img_info = detector.infer(img)
        raw_img = img_info["raw_img"]
        if outputs[0] is not None:
            n_det_total += int(outputs[0].shape[0])
            n_det_hi_total += int((outputs[0][:, 4] >= track_thresh).sum())
            n_frames_with_det += 1

        online_targets = []
        if outputs[0] is not None:
            with perf.stage("mask"):
                if mask_manager is not None and i > 1:
                    prediction, tracklet_mask_dict = mask_manager.get_updated_masks(
                        i, online_tlwhs, online_ids_mask_manager, new_tracks, removed_tracks_ids
                    )
            with perf.stage("track"):
                (online_targets, removed_tracks_ids, new_tracks, detections_per_assoc_step,
                 all_before) = tracker.update(
                    outputs[0], [img_info["height"], img_info["width"]], detector.test_size,
                    prediction_mask=prediction, tracklet_mask_dict=tracklet_mask_dict,
                    frame_img=raw_img, vis_type=cfg.get("io.vis_type", "basic"),
                    dets_from_file=False,
                )

        online_tlwhs, online_ids_mask_manager = [], []
        online_tlwhs_vis, online_ids_vis, online_scores_vis = [], [], []

        with perf.stage("count"):
            for t in online_targets:
                tlwh = t.last_det_tlwh
                internal = int(t.track_id)
                export = int(getattr(t, "export_id", internal))
                final = bool(getattr(t, "export_id_final", False))
                sc = float(t.score)

                online_tlwhs.append(tlwh)
                online_ids_mask_manager.append(internal)

                if final:
                    id_remap[internal] = export
                    # 冲刷该内部 id 之前缓存的观测/记录 (身份已定)
                    if internal in pending_obs:
                        for (f, tlwh_, sc_) in pending_obs.pop(internal):
                            feed_counter(export, f, tlwh_, sc_)
                    if internal in pending_records:
                        for (f, tlwh_, sc_) in pending_records.pop(internal):
                            mot_records.append(_mot_line(f, export, tlwh_, sc_))
                    # 身份已定 → 解除水位阻塞
                    feeder.resolve(internal)
                    seen_identities.add(export)
                    feed_counter(export, i, tlwh, sc)
                    mot_records.append(_mot_line(i, export, tlwh, sc))
                    online_tlwhs_vis.append(tlwh)
                    online_ids_vis.append(export)
                    online_scores_vis.append(sc)
                else:
                    # 身份未定 → 缓存, 暂不计数/输出 (避免碎片化重复计数)
                    pending_obs[internal].append((i, tlwh, sc))
                    pending_records[internal].append((i, tlwh, sc))
                    # 记录该身份最早未定观测的帧号 → 水位不得越过它
                    feeder.note_pending(internal, i, export)
                    online_tlwhs_vis.append(tlwh)
                    online_ids_vis.append(-1)
                    online_scores_vis.append(sc)

        # ---- 流式提交: 把已定案的帧按序喂给计数层 (含空帧) ---- #
        feeder.tick(i)
        if _qlog is not None:
            _qlog.write(f"{i},{time.perf_counter() - t_start:.4f},"
                        f"{(int(i) + 1) - feeder.next_feed_frame},"
                        f"{feeder.n_buffered_frames}\n")

        # ---- 可视化 ---- #
        if vis_on or save_video:
            with perf.stage("viz"):
                frame = raw_img
                if cfg.get("io.vis_type", "basic") != "no_vis" and online_tlwhs_vis:
                    from yolox.utils.visualize import plot_tracking_basic

                    frame, _, _ = plot_tracking_basic(
                        raw_img, online_tlwhs_vis, online_ids_vis,
                        frame_id=i, prediction_mask=prediction,
                    )
                else:
                    frame = np.ascontiguousarray(raw_img.copy())

                # 锚点轨迹 + 误检标记
                n_rejected = 0
                if draw_traj:
                    for tlwh, tid in zip(online_tlwhs_vis, online_ids_vis):
                        if tid < 0:
                            continue
                        trails[tid].append(_anchor_point(tlwh, anchor_mode))
                    vizmod.draw_trails(frame, trails)
                    for tlwh, tid in zip(online_tlwhs_vis, online_ids_vis):
                        if tid >= 0 and not counter.quality.passes(tid):
                            n_rejected += 1
                            vizmod.draw_rejected(frame, tlwh)

                if counter.line is not None:
                    p1, p2 = counter.line.geometry()
                    vizmod.draw_line(frame, p1, p2, counter.line.in_direction,
                                     in_normal=getattr(counter.line, "in_normal", None))
                if counter.zone is not None:
                    vizmod.draw_zone(frame, counter.zone.geometry(), counter.zone.name,
                                     counter.zone.occupancy)
                inst_fps = None
                if len(frame_times) >= 5:
                    inst_fps = 1.0 / (np.mean(frame_times[-10:]) + 1e-9)
                vizmod.draw_hud(frame, i, len(online_tlwhs_vis), counter.summary(), inst_fps,
                                n_rejected=n_rejected)
                if vis_scale != 1.0:
                    frame = cv2.resize(frame, (int(W * vis_scale), int(H * vis_scale)))
        else:
            frame = raw_img

        if writer is not None:
            with perf.stage("write"):
                writer.write(frame)

        frame_times.append(time.perf_counter() - t_frame)
        perf.end_frame(i)

    # ---- 收尾: 冲刷未定身份的缓存 ---- #
    for internal, obs_list in list(pending_obs.items()):
        final_id = id_remap.get(internal, internal)
        for (f, tlwh_, sc_) in obs_list:
            feed_counter(final_id, f, tlwh_, sc_)
        seen_identities.add(final_id)
    for internal, recs in list(pending_records.items()):
        final_id = id_remap.get(internal, internal)
        for (f, tlwh_, sc) in recs:
            mot_records.append(_mot_line(f, final_id, tlwh_, sc))
    # 残留的未定身份不再阻塞水位 (其观测已按当前 id 补发)
    feeder.clear_pending()

    # ---- 计数器: 把剩余帧按帧号升序喂完 (含无观测的空帧) ---- #
    # 与 tools/replay_count.py 的离线回放保持同一契约, 保证 GPU 端与回放端可逐事件对齐。
    # 流式模式下前面已提交 [1, next_feed_frame); 这里补完剩下的 (含残留未定身份)。
    feeder.flush()

    if _qlog is not None:
        try:
            _qlog.flush()
            _qlog.close()
        except Exception:  # pragma: no cover
            pass

    if writer is not None:
        writer.release()

    # ---- 输出 ---- #
    mot_records.sort(key=lambda s: int(s.split(",", 1)[0]))
    if save_mot:
        p = osp.join(out_dir, "mot.txt")
        with open(p, "w") as f:
            f.writelines(mot_records)
        res.outputs["mot"] = p

    counts = counter.summary()
    counts["unique_ids"] = len(seen_identities)
    counts["frames"] = n_processed
    # V3: 流式喂入统计 —— 用于验证"计数延迟有上界、缓存有上限"
    counts["streaming"] = feeder.summary()
    counts["streaming"]["max_lag_sec"] = feeder.lag_sec(fps)
    # 记录本次实际使用的计数线 (人工绘制 / 自动标定 / 配置固定), 便于追溯复现
    if counter.line is not None:
        counts["line_used"] = dict(line_used)
        # 向后兼容: 旧的汇总脚本 (_report_full/_aggregate_runs) 读 auto_line
        if line_source == "auto":
            counts["auto_line"] = {"p1": line_used["p1"], "p2": line_used["p2"],
                                   "in_direction": line_used["in_direction"]}
    # 源声明帧数 vs 实际可解码帧数: 明显偏少说明视频被截断/损坏 (矿上样本常见半包文件)
    if fr.n and n_processed < 0.9 * fr.n:
        counts["frame_shortfall"] = {"planned": fr.n, "processed": n_processed,
                                     "ratio": round(n_processed / float(fr.n), 3)}
        logger.warning(
            f"[warn] {name}: 源声明 {fr.n} 帧, 实际只解码出 {n_processed} 帧 "
            f"({n_processed/float(fr.n)*100:.0f}%) — 视频可能被截断或损坏"
        )
    counts["detections_total"] = n_det_total
    counts["det_per_frame"] = round(n_det_total / max(n_processed, 1), 3)
    # 真正驱动跟踪的是 score>=track_thresh 的框; 原始 det/frame 在 conf=0.01 下
    # 主要被低分噪声主导, 做尺度对比时会误导。
    counts["det_hi_total"] = n_det_hi_total
    counts["det_per_frame_hi"] = round(n_det_hi_total / max(n_processed, 1), 3)
    counts["track_thresh"] = track_thresh
    counts["frames_with_det_ratio"] = round(n_frames_with_det / max(n_processed, 1), 3)
    res.counts = counts
    res.frames = n_processed
    res.unique_ids = len(seen_identities)
    res.perf = perf.report()

    if cfg.get("io.save_counts", True):
        p = osp.join(out_dir, "counts.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": name,
                    "input": input_path,
                    "counts": counts,
                    "line_events": [
                        {"id": e.track_id, "dir": e.direction, "frame": e.frame_id,
                         "x": round(e.point[0], 1), "y": round(e.point[1], 1)}
                        for e in (counter.line.events if counter.line else [])
                    ],
                    "zone_events": [
                        {"id": e.track_id, "dir": e.direction, "frame": e.frame_id}
                        for e in (counter.zone.events if counter.zone else [])
                    ],
                },
                f, ensure_ascii=False, indent=2,
            )
        res.outputs["counts"] = p

        # ---- 新增: 逐事件 JSONL (含 person_uid / 线版本 / 拒绝原因), 便于事件级复盘 ----
        # 旧的 counts.json 字段全部保留, 这里只是**新增**文件, 不破坏任何既有读取方。
        ep = osp.join(out_dir, "events.jsonl")
        with open(ep, "w", encoding="utf-8") as f:
            for k, e in enumerate(counter.line.events if counter.line else [], 1):
                row = {"event_id": "%s#%d" % (name, k), "video_id": name,
                       "camera_id": name, "camera": name}
                to_dict = getattr(e, "to_dict", None)
                if callable(to_dict):
                    row.update(to_dict())
                else:
                    row.update({"direction": e.direction, "track_id": int(e.track_id),
                                "frame_id": int(e.frame_id),
                                "crossing_xy": [round(e.point[0], 1), round(e.point[1], 1)]})
                row.setdefault("line_id", line_used.get("line_id"))
                row.setdefault("line_version", line_used.get("line_version"))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        res.outputs["events"] = ep

        # ---- 新增: 被拒候选 (解释"为什么这里没计数") + 计数层配置快照 ----
        diag = counter.diagnostics()
        dp = osp.join(out_dir, "counter_diagnostics.json")
        with open(dp, "w", encoding="utf-8") as f:
            json.dump({"name": name, "line_used": line_used, "diagnostics": diag},
                      f, ensure_ascii=False, indent=2)
        res.outputs["diagnostics"] = dp

        p = osp.join(out_dir, "perf.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(res.perf, f, ensure_ascii=False, indent=2)
        res.outputs["perf"] = p

    if not cfg.get("io.save_frames", False) and fr.extracted and fr.frames_dir:
        shutil.rmtree(fr.frames_dir, ignore_errors=True)

    res.elapsed_s = time.perf_counter() - t_start
    logger.info(
        f"[run] {name} 完成: frames={res.frames} ids={res.unique_ids} "
        f"counts={counts.get('line')} fps={res.perf.get('fps')} "
        f"gpu_peak={res.perf.get('peak_gpu_mem_mb')}MB"
    )
    return res


def _mot_line(frame_id, tid, tlwh, score) -> str:
    x, y, w, h = tlwh
    return f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.2f},-1,-1,-1\n"

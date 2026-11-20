from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import Config

try:  # loguru 在仓库依赖里
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("coalcount")


# --------------------------------------------------------------------------- #
# 检测器尺度注册表
# --------------------------------------------------------------------------- #
# scale -> 说明 + (exp_file, person_ckpt, raw_coco_ckpt, preproc)
#   person_ckpt : num_classes=1 的权重 (直接可用)
#   raw_coco_ckpt : 官方 80 类 COCO 权重 (可自动转换成 person_ckpt)
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nano": {
        "desc": "YOLOX-nano  depth0.33/width0.25/depthwise  ~0.9M params  (最小)",
        "exp_file": "exps/example/mot/yolox_nano_coco_person.py",
        "person_ckpt": "pretrained/yolox_nano_person.pth",
        "raw_coco_ckpt": "pretrained/yolox_nano.pth",
        "coco_name": "yolox_nano",
        "preproc": "yolox",
        "params_m": 0.912,
    },
    "tiny": {
        "desc": "YOLOX-tiny  depth0.33/width0.375  ~5.0M params",
        "exp_file": "exps/example/mot/yolox_tiny_mix_det.py",
        "person_ckpt": "pretrained/yolox_tiny_person.pth",
        "raw_coco_ckpt": "pretrained/yolox_tiny.pth",
        "coco_name": "yolox_tiny",
        "preproc": "yolox",
        "params_m": 5.05,
    },
    "s": {
        "desc": "YOLOX-s  depth0.33/width0.50  ~9.0M params",
        "exp_file": "exps/example/mot/yolox_s_mix_det.py",
        "person_ckpt": "pretrained/yolox_s_person.pth",
        "raw_coco_ckpt": "pretrained/yolox_s.pth",
        "coco_name": "yolox_s",
        "preproc": "yolox",
        "params_m": 8.97,
    },
    # ---- 煤矿数据微调档位 (在 249 上基于 COCO 权重微调, 单类"工人") ----
    # 训练: <REMOTE_HOME>/disk/xlz/YOLOX  exps/example/coal/yolox_s_coal1.py
    # 数据: DsLMF coal_miner + UCMD coal_miner + person_helmet 工人 (统一为1类)
    # 预处理与官方 COCO 权重一致 -> preproc="yolox"
    "s_coal": {
        "desc": "YOLOX-s 煤矿工人微调 (单类, 3数据集合并)  ~9.0M params",
        "exp_file": "exps/example/coal/yolox_s_coal1.py",
        "person_ckpt": "pretrained/yolox_s_coal_person.pth",
        "raw_coco_ckpt": "pretrained/yolox_s.pth",
        "coco_name": "yolox_s",
        "preproc": "yolox",
        "params_m": 8.97,
    },
    "m": {
        "desc": "YOLOX-m  depth0.67/width0.75  ~25.3M params",
        "exp_file": "exps/example/mot/yolox_m_mix_det.py",
        "person_ckpt": "pretrained/yolox_m_person.pth",
        "raw_coco_ckpt": "pretrained/yolox_m.pth",
        "coco_name": "yolox_m",
        "preproc": "yolox",
        "params_m": 25.3,
    },
    "l": {
        "desc": "YOLOX-l  depth1.00/width1.00  ~54.2M params",
        "exp_file": "exps/example/mot/yolox_l_mix_det.py",
        "person_ckpt": "pretrained/yolox_l_person.pth",
        "raw_coco_ckpt": "pretrained/yolox_l.pth",
        "coco_name": "yolox_l",
        "preproc": "yolox",
        "params_m": 54.2,
    },
    "x": {
        "desc": "YOLOX-x  depth1.33/width1.25  ~99.1M params",
        "exp_file": "exps/example/mot/yolox_x_mix_det.py",
        "person_ckpt": "pretrained/yolox_x_person.pth",
        "raw_coco_ckpt": "pretrained/yolox_x.pth",
        "coco_name": "yolox_x",
        "preproc": "yolox",
        "params_m": 99.1,
    },
    # 仓库自带权重 (配 fork 的默认预处理)
    "x_sports": {
        "desc": "YOLOX-x + SportsMOT 混合权重 (仓库自带, 默认预处理)",
        "exp_file": "exps/example/mot/yolox_x_mix_det.py",
        "person_ckpt": "pretrained/yolox_x_sports_mix.pth.tar",
        "raw_coco_ckpt": None,
        "coco_name": None,
        "preproc": "default",
        "params_m": 99.1,
    },
    "x_mot17": {
        "desc": "ByteTrack x_mot17 权重 (仓库自带, 默认预处理)",
        "exp_file": "exps/example/mot/yolox_x_mix_det.py",
        "person_ckpt": "pretrained/bytetrack_x_mot17.pth.tar",
        "raw_coco_ckpt": None,
        "coco_name": None,
        "preproc": "default",
        "params_m": 99.1,
    },
}

PERSON_CLASS_ID = 0  # COCO 第 0 类 = person


def list_scales() -> Dict[str, str]:
    return {k: v["desc"] for k, v in MODEL_REGISTRY.items()}


# --------------------------------------------------------------------------- #
# 官方 COCO 80 类 → person 单类 转换 / 下载
# --------------------------------------------------------------------------- #
def convert_coco_to_person(src_ckpt: str, dst_ckpt: str) -> int:
    """把 COCO 80 类 YOLOX 权重的 cls_preds 切片到第 0 类(person)。

    YOLOXHead 的 reg_preds / obj_preds 跨类共享, 只有 cls_preds 每类一个通道,
    因此只需切 cls_preds 即可得到纯 person 单类检测器。
    返回切片的张量个数。
    """
    sd = torch.load(src_ckpt, map_location="cpu")
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    new_state = {}
    n_sliced = 0
    for k, v in state.items():
        if k.startswith("head.cls_preds.") and (k.endswith(".weight") or k.endswith(".bias")):
            if v.shape[0] > 1:
                new_state[k] = v[PERSON_CLASS_ID : PERSON_CLASS_ID + 1].contiguous()
                n_sliced += 1
                continue
        new_state[k] = v
    torch.save({"model": new_state}, dst_ckpt)
    logger.info(f"[convert] COCO→person: {osp.basename(src_ckpt)} -> {osp.basename(dst_ckpt)} ({n_sliced} tensors sliced)")
    return n_sliced


def _maybe_download(url: str, dst: str) -> bool:
    import urllib.request

    os.makedirs(osp.dirname(dst), exist_ok=True)
    try:
        logger.info(f"[download] {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return osp.getsize(dst) > 0
    except Exception as e:  # pragma: no cover
        logger.warning(f"[download] 失败: {e}")
        if osp.exists(dst):
            os.remove(dst)
        return False


def resolve_detector_files(cfg: Config) -> Tuple[str, str, str, str]:
    """解析出 (exp_file, ckpt, preproc, scale)。必要时自动转换/下载。"""
    scale = cfg.get("detector.scale", "nano")
    exp_file = cfg.get("detector.exp_file")
    ckpt = cfg.get("detector.ckpt")
    preproc = cfg.get("detector.preproc")

    if exp_file is None or ckpt is None:
        if scale not in MODEL_REGISTRY:
            raise KeyError(
                f"未知检测器尺度 '{scale}'; 可用: {list(MODEL_REGISTRY)}"
            )
        entry = MODEL_REGISTRY[scale]
        exp_file = exp_file or entry["exp_file"]
        ckpt = ckpt or entry["person_ckpt"]
        preproc = preproc or entry["preproc"]

        ckpt_abs = cfg.path(ckpt)
        # 单类 person 权重不存在 → 尝试用官方 COCO 权重自动转换
        if not osp.exists(ckpt_abs) and cfg.get("detector.auto_convert", True):
            raw = cfg.path(entry.get("raw_coco_ckpt"))
            if (raw is None or not osp.exists(raw)) and cfg.get("detector.auto_download", False):
                name = entry.get("coco_name")
                if name:
                    tmpl = cfg.get("detector.coco_url_template")
                    raw = cfg.path(osp.join("pretrained", f"{name}.pth"))
                    _maybe_download(tmpl.format(name=name), raw)
            if raw and osp.exists(raw):
                convert_coco_to_person(raw, ckpt_abs)
            else:
                raise FileNotFoundError(
                    f"检测器权重缺失: {ckpt_abs}\n"
                    f"  可放官方 COCO 权重到 {entry.get('raw_coco_ckpt')} 后自动转换, "
                    f"或打开 detector.auto_download=true 自动下载。"
                )

    if preproc is None:
        preproc = "yolox"
    return cfg.path(exp_file), cfg.path(ckpt), preproc, scale


# --------------------------------------------------------------------------- #
# 检测器封装
# --------------------------------------------------------------------------- #
class Detector:
    """YOLOX 检测器: 统一封装 预处理 + 前向 + 后处理。"""

    def __init__(self, model, exp, device, fp16: bool, det_preproc: str):
        self.model = model
        self.exp = exp
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.det_preproc = det_preproc
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @classmethod
    def build(cls, cfg: Config, device) -> "Detector":
        from yolox.data.data_augment import preproc, preproc_yolox_official  # noqa: F401
        from yolox.exp import get_exp
        from yolox.utils import fuse_model, get_model_info

        exp_file, ckpt, preproc_name, scale = resolve_detector_files(cfg)
        exp = get_exp(exp_file, None)

        if cfg.get("detector.conf_thresh") is not None:
            exp.test_conf = float(cfg.get("detector.conf_thresh"))
        if cfg.get("detector.nms_thresh") is not None:
            exp.nmsthre = float(cfg.get("detector.nms_thresh"))
        ts = cfg.get("detector.test_size")
        if ts:
            exp.test_size = (int(ts[0]), int(ts[1]))

        model = exp.get_model().to(device)
        logger.info(f"[detector] scale={scale} | {get_model_info(model, exp.test_size)}")
        model.eval()

        ckpt_obj = torch.load(ckpt, map_location="cpu")
        state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                f"[detector] 加载 {osp.basename(ckpt)}: missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            logger.info(f"[detector] 加载 {osp.basename(ckpt)}: 0 missing / 0 unexpected")

        if cfg.get("detector.fuse", True):
            model = fuse_model(model)
        fp16 = bool(cfg.get("detector.fp16", True)) and "cuda" in str(device)
        if fp16:
            model = model.half()

        return cls(model, exp, device, fp16, preproc_name)

    def infer(self, img):
        """img 可以是图像路径 (str) 或已解码的 ndarray (BGR)。

        视频"内存流式"模式下直接传 ndarray, 避免抽帧落盘。
        """
        from yolox.data.data_augment import preproc, preproc_yolox_official
        from yolox.utils import postprocess

        fname = "frame"
        if isinstance(img, str):
            fname = osp.basename(img)
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(f"无法读取帧: {fname}")
        h, w = img.shape[:2]
        img_info = {"file_name": fname, "height": h, "width": w, "raw_img": img}

        if self.det_preproc == "yolox":
            inp, ratio = preproc_yolox_official(img, self.test_size)
        else:
            inp, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio

        inp = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)
        if self.fp16:
            inp = inp.half()
        with torch.no_grad():
            outputs = self.model(inp)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


# --------------------------------------------------------------------------- #
# 可选子模块: 掩码传播 / re-ID
# --------------------------------------------------------------------------- #
def build_mask_manager(cfg: Config, frames_dir: str):
    """构建 SAM/EdgeTAM 掩码管理器; 未启用返回 None (上层据此降级)。"""
    if not cfg.get("mask.enabled", False):
        return None
    from types import SimpleNamespace

    from mask_propagation.mask_manager__edgetam import MaskManager

    mask_args = SimpleNamespace(
        with_mask_removal=bool(cfg.get("mask.with_mask_removal", False)),
        with_mask_update=bool(cfg.get("mask.with_mask_update", False)),
        mask_update_every_k_frames=int(cfg.get("mask.mask_update_every_k_frames", 1)),
    )
    logger.info("[mask] EdgeTAM/SAM 掩码传播已启用")
    return MaskManager(input_frame_dir=frames_dir, args=mask_args)


def make_tracker_args(cfg: Config) -> Any:
    """把配置翻译成 tracker 需要的 args 命名空间。"""
    from types import SimpleNamespace

    t = cfg.as_dict()
    args = SimpleNamespace(
        track_thresh=float(t["tracker"]["track_thresh"]),
        track_buffer=int(t["tracker"]["track_buffer"]),
        cmc_method=t["tracker"]["cmc_method"],
        cmc_interval=int(t["tracker"]["cmc_interval"]),
        cmc_downscale=int(t["tracker"]["cmc_downscale"]),
    )
    r = t["reid"]
    if r["enabled"]:
        args.reid_model_path = cfg.path(r["model_path"])
        args.reid_model_name = r["model_name"]
        args.reid_device = r["device"]
        args.reid_min_det_score = float(r["min_det_score"])
        args.reid_min_box_area = float(r["min_box_area"])
        args.reid_min_frame_gap = int(r["min_frame_gap"])
        args.reid_sim_thresh = float(r["sim_thresh"])
        args.reid_reconnect_enabled = bool(r["reconnect_enabled"])
        args.reid_new_max_len = int(r["new_max_len"])
        args.reid_min_updates = int(r["min_updates"])
    else:
        # reid_model_path=None → ReIDManager 自动禁用, 轨迹退化为 track_id
        args.reid_model_path = None
        args.reid_model_name = r["model_name"]
        args.reid_device = "cpu"
    return args


def build_tracker(cfg: Config, save_folder: str, frame_rate: int = 30):
    from yolox.tracker.mcbyteplusplus_tracker__with_reid import McBytePlusPlusTracker

    args = make_tracker_args(cfg)
    return McBytePlusPlusTracker(
        args, frame_rate=int(frame_rate), save_folder=save_folder, enable_logging=False
    )

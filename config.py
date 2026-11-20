from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# 缺省配置 (与 Config.get 的 dotted 访问对应)
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Any] = {
    "detector": {
        # 尺度别名, 见 models.MODEL_REGISTRY: nano|tiny|s|m|l|x|x_sports|x_mot17
        "scale": "nano",
        # 显式覆盖 (为 None 时使用注册表)
        "exp_file": None,
        "ckpt": None,
        "preproc": None,          # default | yolox
        "fp16": True,
        "fuse": True,
        "conf_thresh": None,      # 覆盖 exp.test_conf (后处理置信度)
        "nms_thresh": None,
        "test_size": None,        # [h, w] 覆盖
        # 官方 COCO 权重 → person 单类 自动转换/下载
        "auto_convert": True,
        "auto_download": False,
        "coco_url_template": "https://gh-proxy.com/https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{name}.pth",
    },
    "tracker": {
        "track_thresh": 0.5,
        "track_buffer": 30,
        # 煤矿井下摄像头基本固定 → 默认关闭相机运动补偿。
        # 实测 1920x1080 下 ORB CMC 会把 FPS 从 ~27 拉到 ~6, 收益为负。
        # 云台/抖动/手持相机请设为 orb|ecc|sparseOptFlow。
        "cmc_method": "none",      # none | orb | ecc | sparseOptFlow
        "cmc_interval": 1,
        "cmc_downscale": 4,
    },
    # ---- 非必须子功能开关 ---- #
    "mask": {                      # SAM / EdgeTAM 掩码传播 (最重的子模块)
        "enabled": False,
        "checkpoint": "mask_propagation/EdgeTAM/checkpoints/edgetam.pt",
        "model_cfg": "configs/edgetam.yaml",
        "with_mask_removal": False,
        "with_mask_update": False,
        "mask_update_every_k_frames": 1,
    },
    "reid": {                      # 在线 re-ID (人员相似/遮挡重连)
        "enabled": True,
        "model_name": "osnet_x0_25",   # osnet_x0_25 | osnet_x1_0
        "model_path": "deep-person-reid/pretrained/osnet_x0_25_msmt17.pth",
        "device": "cuda",
        "sim_thresh": 0.7,
        "reconnect_enabled": True,
        "new_max_len": 10,
        "min_updates": 2,
        "min_det_score": 0.6,
        "min_box_area": 1024,
        "min_frame_gap": 5,
    },
    # ---- 计数 ---- #
    "counter": {
        "mode": "line",            # line | zone | line+zone
        # 计数算法: legacy(历史瞬时判据, 默认) | fsm_v1(双稳定区 FSM, 见 crossing.py)
        # 默认保持 legacy, 待验证通过再切换生产行为。
        "algorithm": "legacy",
        "anchor": "bottom_center",  # bottom_center(脚点, 推荐) | center
        # 人工绘制的计数线存储文件 (tools/draw_count_line.py 产出)。
        # 存在且命中当前场景时优先于下面的 line.auto / p1 / p2; 置空则禁用。
        "line_file": "configs/coalcount/lines.json",
        # 计数线 (归一化坐标 0~1)。
        # ★ 方向语义: 优先用显式单位法向量 in_normal (朝入井侧); 未给时按
        #   normal = (-dy, dx) (in_direction="bwd" 取负) 推导 —— 与 UI 画的 IN 箭头一致。
        "line": {
            "auto": False,           # True: 用轨迹主运动方向自动标定 (忽略 p1/p2)
            "auto_min_displacement": 25.0,  # 自动标定时视为"显著运动"的最小位移(像素)
            "p1": [0.50, 0.15],
            "p2": [0.50, 0.85],
            "in_direction": "fwd",   # 兼容字段: 决定箭头朝向 (fwd/bwd)
            "in_normal": None,       # 显式单位法向量 [nx, ny], 朝入井侧; 优先于 in_direction
            "line_id": "",           # 线标识 (可追溯)
            "line_version": 1,       # 线语义版本号
            "ref_size": [0, 0],      # in_normal 的参考分辨率 [W, H] (审计用)
            "roi": None,             # 可选: 有效闸口多边形 (归一化), 交点必须落在其中
            "seg_tolerance": 0.0,    # 有限线段两端容差 (线长比例, 建议 0.05)
            "hysteresis": 8.0,       # [legacy] 死区(像素): 抑制线附近抖动
            "min_track_age": 4,      # [legacy] 轨迹至少被观测多少帧才计入
            "min_displacement": 6.0,  # [legacy] 相邻评估间的最小位移(像素)
            "cooldown_frames": 25,   # [legacy] 同一 ID 两次同向计数的最小间隔
            # ---- fsm_v1 专用 ----
            "k_start": 3,            # 起始侧需连续确认帧数
            "k_end": 3,              # 终止侧需连续确认帧数
            "band_px": 0.0,          # 单侧带宽 H (像素); >0 时固定, 否则按框高自适应
            "band_rel": 0.2,         # H = clip(band_rel * bbox_h, band_min, band_max)
            "band_min": 4.0,
            "band_max": 120.0,
            "max_pending_sec": 3.0,  # 候选最长有效时间(秒), 超时不计数
            "event_cooldown_frames": 25,  # [fsm_v1] 同一身份两次事件的最小间隔(帧)。
            #   独立于 legacy 的 cooldown_frames: 后者是针对"瞬时判据抖动"调的
            #   大值(实测配置里出现过 90), 若被 FSM 继承会把正常的快速往返
            #   (出井后 3.6s 内又进井) 当成重复事件压掉。
            "jump_rel": 0.8,         # 锚点跳变门: 允许位移 = jump_px + jump_rel*bbox_h*Δt
            "jump_px": 30.0,
            "gate_blocks_state": False,  # [fsm_v1] 质量门否决是否阻断状态推进 (False=V3 行为)
            #   True 复现 V2: 门否决即跳过观测 → 轨迹开头被否会导致 stable_side
            #   建不起来, 后续真实穿越被误判成"初始化"而静默丢失。
            "osc_guard": True,       # [fsm_v1] 位移自适应带宽 (徘徊抑制)
            "osc_window_sec": 3.0,   #   振荡检测的滑动时间窗(秒)
            "osc_boost": 2.0,        #   H_eff = H * (1 + osc_boost * osc), osc=1-|净位移|/|行程|
            "osc_min_flips": 2,      #   窗口内稳定侧至少翻转多少次才算"徘徊"。
            #   正常通行 (进一次/出一次) 翻转 ≤1 → 带宽不受影响, 保护擦边合法事件;
            #   真正的来回徘徊实测翻转 3–6 次。设 0 可退化为"无条件放大"(不推荐)。
            "osc_episode_min": 2,    #   片段最小驻留(观测数): 短于此的越侧视为脚点抖动, 不计翻转。
            #   ★ 该组合 (min_flips=2, episode_min=2, boost=2) 是 9 段有效视频上
            #     的**零回归**最优: TP 不变 (73), FP 35→34, F1 0.6404→0.6432,
            #     err9 40→39, 且 `一号人行斜井井口2` 的误差保持 0。
            #     更大的 boost (≥4) 或更松的门槛 (min_flips=1) 会开始吞掉真实事件
            #     (一号 err 0→2..4), 违反任务书"不得以抑制合法反向通行换低 FP"。
            "min_score": None,       # 参与 FSM 的最低检测置信度 (None=不启用)
            "enforce_segment": False,  # [legacy] 是否启用有限线段/ROI 约束
        },
        # 逻辑身份关联 (仅 fsm_v1 使用): 把碎裂的 track_id 归并成 person_uid
        "identity": {
            "enabled": True,
            "max_gap_sec": 1.5,      # 允许的最大断轨间隔(秒)
            "max_dist_px": 60.0,     # 允许的最大预测位置偏差(像素)
            "max_dist_rel": 1.0,     # 再加上 max_dist_rel * bbox_h
            "dir_cos_min": 0.0,      # 运动方向一致性下限 (余弦)
            "margin_rel": 1.35,      # 唯一最优判定: best*rel+px <= second 才合并
            "margin_px": 25.0,
            "use_appearance": True,  # 有 embedding 时才做外观门控
            "app_min_cos": 0.75,
        },
        # 区域 (归一化多边形), 用于"当前在井口区域人数"等
        "zone": {
            "enabled": False,
            "polygon": [[0.3, 0.4], [0.7, 0.4], [0.7, 0.9], [0.3, 0.9]],
            "name": "gate_zone",
            "count_enter": True,
            "count_exit": True,
        },
        "dedupe_window": 60,       # 同一身份同向去重窗口(帧), 抑制碎片化重复计数
        # V3: 流式按帧批处理 (取代"视频结束后统一回放")。
        #   计数层在运行过程中就被喂入已定案的帧, 因此 HUD 实时、结果延迟有上界;
        #   关闭 (False) 时退回旧行为 (收尾统一回放), 用于等价性对照测试。
        "streaming": True,
        "max_buffered_frames": 600,  # 未定身份导致的缓存上限(帧); 超限即强制定案最老身份
        # 误检判断与消除 (轨迹质量门): 越线时按轨迹统计判定是否可信
        "quality": {
            "enabled": True,
            "min_hits": 4,             # 最少匹配检测次数
            "min_hit_ratio": 0.0,      # 匹配次数/存在帧跨度 (0=关闭)
            "min_score": 0.0,          # 轨迹平均检测置信度下限 (0=关闭)
            # V3: 几何判据改为"滑动窗口 + 最小样本数", 并重新定标
            #   min_aspect 1.0 -> 0.2  (井口近景/多人合框时 h/w<1 是常态, 不是误检)
            #   min_height_ratio 0.03 -> 0.0 (远距离小框同样是真目标)
            #   实测: 旧阈值在 …103513550 单场景就误杀 12 条真实出井轨迹
            "min_aspect": 0.2,         # 框高宽比 h/w 下限
            "max_aspect": 8.0,         # 框高宽比 h/w 上限
            "min_height_ratio": 0.0,   # 框高/画面高 下限
            "max_height_ratio": 0.98,  # 框高/画面高 上限
            "window": 30,              # 几何判据滑动窗口长度 (观测数)
            "min_samples": 10,         # 观测数不足时不做几何比例判据
            # [fsm_v1] 质量门按 person_uid (True) 还是 track_id (False) 统计。
            #   实测 **False 更优**: 按身份聚合会把"刚入画时框偏宽扁"的阶段也
            #   纳入统计窗口, 反而把真实轨迹整体否决 (66/34/54 vs 68/35/52)。
            #   按 track_id 统计则每条轨迹各自定标, 短碎片不会被长轨迹拖累。
            "by_identity": False,
        },
    },
    "io": {
        "out_dir": "coalcount_outputs",
        "save_video": True,
        "save_mot": True,
        "save_counts": True,
        "save_frames": False,      # 是否保留抽取的帧
        "decode": "memory",        # memory(视频内存流式解码, 不落盘) | extract(先抽帧到临时目录)
        "vis": True,
        "vis_scale": 1.0,
        "draw_trajectory": True,   # 是否绘制锚点轨迹 (直观显示穿越过程)
        "trail_len": 60,           # 轨迹保留帧数
        "start_frame": 1,
        "max_frames": 0,           # 0 = 全部
        "max_seconds": 0.0,        # >0 时按源帧率换算成 max_frames (视频时长上限, 秒)
        "stride": 1,
        "fps_override": None,
        "video_fourcc": "mp4v",
    },
    "perf": {
        "enabled": True,
        "sync_cuda": True,
        "warmup_frames": 3,
    },
    "run": {
        "device": "cuda:0",
        "tag": "",                 # 输出子目录后缀
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """嵌套字典 + dotted 访问 的配置对象。"""

    def __init__(self, data: Dict[str, Any]):
        self._d = _deep_merge(DEFAULTS, data or {})

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(data)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Config":
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    f"读取 {p} 需要 PyYAML; 请 `pip install pyyaml` 或改用 .json 配置"
                ) from e
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        cfg = cls(data)
        cfg._source = str(p)
        return cfg

    # ------------------------------------------------------------------ #
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self._d
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)

    # ------------------------------------------------------------------ #
    def path(self, p: str | os.PathLike | None) -> Optional[str]:
        """把配置里的相对路径解析为绝对路径 (相对仓库根)。"""
        if p in (None, ""):
            return None
        p = Path(p)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return str(p)

    def save(self, path: str | os.PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self._d, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def describe(self) -> List[str]:
        """人类可读的配置摘要 (用于日志)。"""
        lines = []
        det = self._d["detector"]
        lines.append(
            f"detector: scale={det['scale']} preproc={det['preproc']} "
            f"fp16={det['fp16']} conf={det['conf_thresh']}"
        )
        lines.append(
            f"switches: mask={self._d['mask']['enabled']} "
            f"reid={self._d['reid']['enabled']} cmc={self._d['tracker']['cmc_method']}"
        )
        lines.append(
            f"counter: mode={self._d['counter']['mode']} "
            f"anchor={self._d['counter']['anchor']} "
            f"in_dir={self._d['counter']['line']['in_direction']} "
            f"auto={self._d['counter']['line']['auto']} "
            f"line_file={self._d['counter'].get('line_file')} "
            f"quality={self._d['counter']['quality']['enabled']}"
        )
        lines.append(
            f"io: out_dir={self._d['io']['out_dir']} decode={self._d['io']['decode']} "
            f"max_frames={self._d['io']['max_frames']} max_seconds={self._d['io']['max_seconds']} "
            f"stride={self._d['io']['stride']} trail={self._d['io']['draw_trajectory']}"
        )
        return lines

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

VERSION = 2

# 末级目录是这些通用名时不表达场景语义, 取上一级
_GENERIC_DIR_NAMES = {"frames", "img1", "img", "images", "jpgs", "rgb", "images_0"}

# 只剥离这些"真正的"媒体/图片扩展名。
# 关键: **不能**对任意名字直接 osp.splitext —— 煤矿视频常用 IP 命名
# (``192.168.1.35_01_20251025154558464.mp4``), splitext 会在第一个点处切开,
# 把场景名主干算成 ``192.168.1``; 于是同一网段的所有视频会互相串场
# (第 2 条视频查到第 1 条的记录/计数线)。见 _test_lines.test_no_cross_scene_lookup。
_MEDIA_EXTS = {
    ".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".rmvb", ".3gp", ".mts",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}


def stem_of(name: str) -> str:
    """取文件名主干: **仅当**末段是已知媒体扩展名时才去掉, 否则原样返回。

    这样 ``a.mp4`` -> ``a``, 而 ``192.168.1.35_01_20251025154558464`` 保持原样
    (不会被 splitext 误切成 ``192.168.1``)。
    """
    base = osp.basename(str(name).replace("\\", "/"))
    root, ext = osp.splitext(base)
    return root if ext.lower() in _MEDIA_EXTS else base


def record_matches_input(rec: Dict[str, Any], input_path: str) -> bool:
    """判断一条已存记录是否**确实属于当前输入视频** (防串场, 纵深防御)。

    比对记录里的 ``source`` 与 ``input_path`` 的文件名主干: 不一致即认为记录
    属于别的视频。用于交互标注时忽略历史遗留的串场记录, 让它被重新标注覆盖。
    任一侧无法解析时放行 (不做过度拦截)。
    """
    src = str((rec or {}).get("source") or "")
    if not src or not input_path:
        return True
    a, b = stem_of(src), stem_of(input_path)
    if not a or not b:
        return True
    return a == b


# --------------------------------------------------------------------------- #
def scene_name_from_path(path: str) -> str:
    """从输入路径推导场景名 (与 pipeline.derive_scene_name 保持一致)。"""
    p = osp.normpath(str(path))
    base = osp.basename(p)
    if base.lower() in _GENERIC_DIR_NAMES:
        parent = osp.basename(osp.dirname(p))
        if parent:
            return parent
    return osp.splitext(base)[0] or "scene"


def _norm_key(path: str) -> str:
    """把路径归一化成稳定的比较键 (尽量取绝对路径, 失败则退回原串)。"""
    s = str(path)
    try:
        return osp.normpath(osp.abspath(s))
    except Exception:  # pragma: no cover
        return osp.normpath(s)


# --------------------------------------------------------------------------- #
# 方向语义: 显式法向量 (v2) 与旧 schema 的确定性迁移
# --------------------------------------------------------------------------- #
def derived_in_normal(rec: Dict[str, Any]) -> Optional[List[float]]:
    """按历史箭头约定推导朝"入井侧"的单位法向量 (无 p1/p2/分辨率 时返回 None)。

    **必须在像素坐标空间推导**:

        a = (p1.x * W, p1.y * H);  b = (p2.x * W, p2.y * H)
        u = unit(b - a);  normal = (-u.y, u.x);  in_direction == "bwd" → normal = -normal

    与 ``ui.draw_line_overlay`` / ``viz.draw_line`` 画 "IN" 箭头的公式**完全一致**
    (那两个函数拿到的是像素坐标端点, 所以法向量也是在像素空间算的)。

    ⚠️ 2026-10-11 修正 —— 之前直接在**归一化坐标**里算 ``(-dy, dx)``, 对
    ``2560x1440`` 这类非正方形画幅会引入 ~5° 的系统性角度偏差 (归一化空间把
    x 拉伸了 ``W/H`` 倍, 斜率被扭曲)。后果是"生效的计数线"与"标注员看到的
    箭头"不是同一条线: 在 ``35_01_…5317122`` 上, 一条在线上徘徊的长轨迹
    (track 80) 因为偏转后的线恰好扫过它的抖动区间, 凭空多出 6 次穿越事件
    (GPU 端旧实现 5 事件 → 本实现 11 事件)。修正后两者逐事件一致。

    分辨率未知时返回 ``None``, 交由 ``LineGeometry.from_cfg`` 在运行时用真实
    ``W/H`` 推导 —— 宁可退回"端点顺序决定方向"的旧行为, 也不要注入一个错的
    方向 (错方向是静默的, 而端点顺序是可审计的)。
    """
    try:
        p1 = (float(rec["p1"][0]), float(rec["p1"][1]))
        p2 = (float(rec["p2"][0]), float(rec["p2"][1]))
    except Exception:
        return None
    w = float(rec.get("width") or 0.0)
    h = float(rec.get("height") or 0.0)
    if w <= 0.0 or h <= 0.0:
        return None
    dx, dy = (p2[0] - p1[0]) * w, (p2[1] - p1[1]) * h
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-12:
        return None
    ux, uy = dx / n, dy / n
    nx, ny = -uy, ux
    if str(rec.get("in_direction", "fwd")).lower() == "bwd":
        nx, ny = -nx, -ny
    ln = (nx * nx + ny * ny) ** 0.5
    if ln < 1e-12:
        return None
    return [nx / ln, ny / ln]


def effective_in_normal(rec: Dict[str, Any]) -> Optional[List[float]]:
    """记录实际生效的单位法向量: 显式 ``in_normal`` 优先, 否则按箭头约定推导。"""
    n = (rec or {}).get("in_normal")
    if n:
        try:
            nx, ny = float(n[0]), float(n[1])
            ln = (nx * nx + ny * ny) ** 0.5
            if ln > 1e-12:
                return [nx / ln, ny / ln]
        except Exception:
            pass
    return derived_in_normal(rec or {})


def migrate_record(rec: Dict[str, Any], scene: str = "") -> Dict[str, Any]:
    """把一条记录迁移到 v2 视图 (**不修改入参**, 返回新 dict)。

    * 补齐 ``in_normal`` (由旧箭头约定在**像素空间**确定性推导) —— 行为与旧版
      ``counter._signed`` **完全一致** (旧版就是拿像素坐标算的); 若记录缺
      ``width``/``height`` 则无法还原像素空间方向, 此时**不**注入 ``in_normal``,
      交由 ``LineGeometry.from_cfg`` 在运行时按真实分辨率推导。
    * 补齐 ``line_id`` / ``line_version``;
    * 记录 ``_migrated_from`` = 原 ``version``, 便于回滚与审计。
    若记录本来就有 ``in_normal``, 只补齐缺失字段, 不动方向。
    """
    out = dict(rec or {})
    out.setdefault("line_id", scene or "")
    src_ver = int(out.get("line_version", out.get("version", 1)) or 1)
    if not out.get("in_normal"):
        n = derived_in_normal(out)
        if n is not None:
            out["in_normal"] = n
            out["_migrated_from"] = src_ver
            out.setdefault("ref_size", [int(out.get("width") or 0),
                                        int(out.get("height") or 0)])
    out.setdefault("line_version", 2 if out.get("in_normal") else src_ver)
    out.setdefault("seg_tolerance", 0.0)
    return out


def migrate_store(store: Dict[str, Any], *, in_place: bool = False) -> Dict[str, Any]:
    """迁移整个存储; 默认返回新对象 (原 store 不变)。"""
    src = store if in_place else dict(store or {})
    scenes = dict(src.get("scenes") or {})
    for name, rec in list(scenes.items()):
        scenes[name] = migrate_record(rec or {}, name)
    src["scenes"] = scenes
    src["version"] = VERSION
    return src


# --------------------------------------------------------------------------- #
def empty_store() -> Dict[str, Any]:
    return {"version": VERSION, "anchor_default": "bottom_center",
            "scenes": {}, "paths": {}}


def load_store(path: str | os.PathLike) -> Dict[str, Any]:
    """读取计数线存储; 文件不存在/损坏时返回空存储 (不抛异常)。"""
    store = empty_store()
    if not path:
        return store
    p = str(path)
    if not osp.isfile(p):
        return store
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return store
    if not isinstance(data, dict):
        return store
    store["version"] = int(data.get("version", VERSION))
    store["anchor_default"] = data.get("anchor_default", "bottom_center")
    sc = data.get("scenes")
    store["scenes"] = sc if isinstance(sc, dict) else {}
    pa = data.get("paths")
    store["paths"] = pa if isinstance(pa, dict) else {}
    # 保证每条记录都有 paths 反查 (兼容手工编辑过的文件)
    for name, rec in list(store["scenes"].items()):
        src = (rec or {}).get("source")
        if src:
            store["paths"].setdefault(str(src), name)
    return store


def save_store(path: str | os.PathLike, store: Dict[str, Any]) -> None:
    p = str(path)
    d = osp.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    store.setdefault("version", VERSION)
    store.setdefault("anchor_default", "bottom_center")
    store.setdefault("scenes", {})
    store.setdefault("paths", {})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
def set_scene(store: Dict[str, Any], scene: str, p1, p2, *,
              in_direction: str = "fwd", anchor: str = "bottom_center",
              source: str = "", width: int = 0, height: int = 0,
              note: str = "", in_normal=None, line_id: str = "",
              roi=None, seg_tolerance: float = 0.0,
              line_version: int = VERSION) -> Dict[str, Any]:
    """写入/覆盖一条场景计数线, 并维护 paths 反查表。

    ``in_normal`` 未显式给出时, 按箭头约定自动推导并**写入记录** —— 这样新画的线
    从第一天起就带显式方向语义, 不依赖端点顺序。
    """
    rec = {
        "p1": [float(p1[0]), float(p1[1])],
        "p2": [float(p2[0]), float(p2[1])],
        "in_direction": in_direction,
        "anchor": anchor,
        "source": str(source or ""),
        "width": int(width or 0),
        "height": int(height or 0),
        "note": note,
        "line_id": str(line_id or scene),
        "line_version": int(line_version or VERSION),
        "seg_tolerance": float(seg_tolerance or 0.0),
    }
    if in_normal is not None:
        rec["in_normal"] = [float(in_normal[0]), float(in_normal[1])]
        rec["ref_size"] = [int(width or 0), int(height or 0)]
    else:
        n = derived_in_normal(rec)
        if n is not None:
            rec["in_normal"] = n
            rec["ref_size"] = [int(width or 0), int(height or 0)]
    if roi:
        rec["roi"] = [[float(q[0]), float(q[1])] for q in roi]
    store.setdefault("scenes", {})[scene] = rec
    if source:
        store.setdefault("paths", {})[str(source)] = scene
    store["version"] = max(int(store.get("version", VERSION) or VERSION), VERSION)
    return rec


def resolve_scene(store: Dict[str, Any], input_path: str,
                  scene: Optional[str] = None) -> Optional[str]:
    """按 路径/场景名/文件名 依次查找, 返回命中的**场景名** (未命中 None)。

    查找优先级:
      1) paths 反查表 (原始路径、归一化路径、文件名)
      2) paths 表归一化后比对
      3) 场景名 / 文件名(去扩展名) 直接匹配
    因此换机器、换挂载点导致绝对路径变化时, 只要文件名一致仍能命中。
    """
    if not store:
        return None
    scenes = store.get("scenes") or {}
    if not scenes:
        return None
    paths = store.get("paths") or {}

    keys = []
    if input_path:
        keys.append(str(input_path))
        keys.append(_norm_key(input_path))
        keys.append(osp.basename(str(input_path)))
    if scene:
        keys.append(str(scene))
    if input_path:
        keys.append(scene_name_from_path(input_path))

    # 1) 通过 paths 反查表
    for k in keys:
        name = paths.get(k)
        if name and name in scenes:
            return name

    # 2) 归一化后直接比对 paths 表
    if input_path:
        nk = _norm_key(input_path)
        for raw, name in paths.items():
            if _norm_key(raw) == nk and name in scenes:
                return name

    # 3) 直接按场景名 / 文件名匹配。
    #    文件名比较必须走 stem_of (只剥离真实媒体扩展名), 否则含点的场景名
    #    (如 192.168.1.x_...) 会被 splitext 截成公共前缀而互相串场。
    for k in keys:
        if k in scenes:
            return k
    for k in keys:
        ks = stem_of(k)
        if not ks:
            continue
        for name in scenes:
            if stem_of(name) == ks:
                return name
    return None


def lookup(store: Dict[str, Any], input_path: str,
           scene: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """按 路径/场景名/文件名 依次查找已保存的计数线, 命中返回**迁移后**的记录副本。

    返回值一定带 ``in_normal`` (显式或按旧箭头约定推导) 与 ``line_version``,
    因此上层可以只依赖 ``in_normal`` 判定方向, 不再关心端点顺序。
    """
    name = resolve_scene(store, input_path, scene)
    if name is None:
        return None
    rec = dict((store.get("scenes") or {}).get(name) or {})
    return migrate_record(rec, name)


def apply_to_config(cfg, rec: Dict[str, Any]) -> None:
    """把一条计数线记录写入 Config (counter.line.* / counter.anchor)。"""
    cfg.set("counter.line.p1", list(rec["p1"]))
    cfg.set("counter.line.p2", list(rec["p2"]))
    cfg.set("counter.line.in_direction", rec.get("in_direction", "fwd"))
    # ★ 显式方向语义: 有 in_normal 就以它为准 (交换端点不再影响 IN/OUT)
    n = effective_in_normal(rec)
    cfg.set("counter.line.in_normal", list(n) if n else None)
    cfg.set("counter.line.line_id", rec.get("line_id") or "")
    cfg.set("counter.line.line_version", int(rec.get("line_version", 1) or 1))
    cfg.set("counter.line.ref_size", list(rec.get("ref_size") or
                                          [int(rec.get("width") or 0),
                                           int(rec.get("height") or 0)]))
    cfg.set("counter.line.roi", [list(map(float, q)) for q in rec["roi"]] if rec.get("roi") else None)
    cfg.set("counter.line.seg_tolerance", float(rec.get("seg_tolerance", 0.0) or 0.0))
    # 手工线优先于自动标定
    cfg.set("counter.line.auto", False)
    if rec.get("anchor"):
        cfg.set("counter.anchor", rec["anchor"])


# apply_to_config 可能写到的全部配置键。
# 批量运行时 Config 是共享对象, pipeline.run_video 会在 Counter 构造后
# 用这个列表把配置**原样还原**, 避免本条视频的线泄漏给后续视频。
# 修改 apply_to_config 时务必同步维护此处。
CONFIG_KEYS = (
    "counter.line.p1",
    "counter.line.p2",
    "counter.line.in_direction",
    "counter.line.in_normal",
    "counter.line.line_id",
    "counter.line.line_version",
    "counter.line.ref_size",
    "counter.line.roi",
    "counter.line.seg_tolerance",
    "counter.line.auto",
    "counter.anchor",
)


def describe(rec: Dict[str, Any]) -> str:
    p1 = rec.get("p1", [0, 0])
    p2 = rec.get("p2", [0, 0])
    n = effective_in_normal(rec)
    ns = "None" if n is None else f"({n[0]:+.3f},{n[1]:+.3f})"
    # 保留 ``in=<fwd|bwd>`` 便于旧日志/脚本比对, 并补充权威的 in_normal 与版本
    return (f"p1=({p1[0]:.3f},{p1[1]:.3f}) p2=({p2[0]:.3f},{p2[1]:.3f}) "
            f"in={rec.get('in_direction', 'fwd')} in_normal={ns} "
            f"v{rec.get('line_version', 1)} anchor={rec.get('anchor', 'bottom_center')}")

from __future__ import annotations

import csv
import importlib.util
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

VERSION = 1


def _load_lines_module():
    """按路径加载同目录的 lines.py, 复用其场景匹配逻辑 (两者都无 torch 依赖)。"""
    p = osp.join(osp.dirname(osp.abspath(__file__)), "lines.py")
    spec = importlib.util.spec_from_file_location("cc_lines_for_labels", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_L = _load_lines_module()
scene_name_from_path = _L.scene_name_from_path


# --------------------------------------------------------------------------- #
def empty_store() -> Dict[str, Any]:
    return {"version": VERSION, "scenes": {}, "paths": {}}


def load_store(path: str | os.PathLike) -> Dict[str, Any]:
    """读取标签存储; 文件不存在/损坏时返回空存储 (不抛异常)。"""
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
    sc = data.get("scenes")
    store["scenes"] = sc if isinstance(sc, dict) else {}
    pa = data.get("paths")
    store["paths"] = pa if isinstance(pa, dict) else {}
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
    store.setdefault("scenes", {})
    store.setdefault("paths", {})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
def make_record(source: str, *, width: int = 0, height: int = 0, fps: float = 0.0,
                total_frames: int = 0, line: Optional[dict] = None,
                events: Optional[List[dict]] = None, resume_frame: int = 1,
                note: str = "") -> Dict[str, Any]:
    ev = sorted((dict(e) for e in (events or [])), key=lambda e: int(e.get("frame", 0)))
    return {
        "source": str(source or ""),
        "width": int(width or 0),
        "height": int(height or 0),
        "fps": float(fps or 0.0),
        "total_frames": int(total_frames or 0),
        "line": dict(line) if line else {},
        "events": ev,
        "totals": totals_of(ev),
        "resume_frame": int(resume_frame or 1),
        "note": note,
    }


def totals_of(events: List[dict]) -> Dict[str, int]:
    c = {"in": 0, "out": 0}
    for e in events or []:
        d = e.get("dir")
        if d in c:
            c[d] += 1
    c["net"] = c["in"] - c["out"]
    c["n_events"] = c["in"] + c["out"]
    return c


def set_scene(store: Dict[str, Any], scene: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """写入/覆盖一条场景标签, 并维护 paths 反查表。"""
    rec = dict(record)
    rec["events"] = sorted((dict(e) for e in rec.get("events", [])),
                           key=lambda e: int(e.get("frame", 0)))
    rec["totals"] = totals_of(rec["events"])
    store.setdefault("scenes", {})[scene] = rec
    if rec.get("source"):
        store.setdefault("paths", {})[str(rec["source"])] = scene
    return rec


def lookup(store: Dict[str, Any], input_path: str,
           scene: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """按 路径/场景名/文件名 查找标签记录 (优先级同 lines.lookup)。"""
    name = _L.resolve_scene(store, input_path, scene)
    if name is None:
        return None
    return dict((store.get("scenes") or {}).get(name) or {})


def resolve_scene(store: Dict[str, Any], input_path: str,
                  scene: Optional[str] = None) -> Optional[str]:
    return _L.resolve_scene(store, input_path, scene)


def remove_scene(store: Dict[str, Any], scene: str) -> bool:
    rec = (store.get("scenes") or {}).pop(scene, None)
    if rec is None:
        return False
    src = rec.get("source")
    if src and (store.get("paths") or {}).get(str(src)) == scene:
        store["paths"].pop(str(src), None)
    return True


# --------------------------------------------------------------------------- #
def to_csv(store: Dict[str, Any], path: str) -> str:
    """导出汇总 CSV (便于 Excel / 后续脚本直接比对)。"""
    p = str(path)
    d = osp.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    rows = summary_rows(store)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scene", "gt_in", "gt_out", "gt_net", "n_events",
                    "fps", "total_frames", "resume_frame", "source", "note"])
        for r in rows:
            w.writerow([r["scene"], r["in"], r["out"], r["net"], r["n_events"],
                        r["fps"], r["total_frames"], r["resume_frame"],
                        r["source"], r["note"]])
    return p


def summary_rows(store: Dict[str, Any]) -> List[dict]:
    out = []
    for scene, rec in (store.get("scenes") or {}).items():
        t = rec.get("totals") or totals_of(rec.get("events") or [])
        out.append({
            "scene": scene,
            "in": int(t.get("in", 0)),
            "out": int(t.get("out", 0)),
            "net": int(t.get("net", 0)),
            "n_events": int(t.get("n_events", 0)),
            "fps": rec.get("fps", 0.0),
            "total_frames": rec.get("total_frames", 0),
            "resume_frame": rec.get("resume_frame", 1),
            "source": rec.get("source", ""),
            "note": rec.get("note", ""),
        })
    out.sort(key=lambda r: r["scene"])
    return out


def is_done(rec: Optional[Dict[str, Any]]) -> bool:
    """该场景是否已标完 (标完 = 有记录且 resume_frame 已到末尾或显式标记)。"""
    if not rec:
        return False
    return bool(rec.get("note") == "done" or rec.get("finished"))

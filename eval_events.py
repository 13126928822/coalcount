from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# 事件容器
# --------------------------------------------------------------------------- #
class Ev:
    __slots__ = ("t", "dir", "track_id", "person_uid", "frame", "raw")

    def __init__(self, t: float, direction: str, track_id=None, person_uid=None,
                 frame=None, raw=None):
        self.t = float(t)
        self.dir = str(direction)
        self.track_id = track_id
        self.person_uid = person_uid
        self.frame = frame
        self.raw = raw or {}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels(path: str) -> Dict[str, dict]:
    data = _load_json(path)
    return dict(data.get("scenes") or {})


def gt_events(rec: dict) -> List[Ev]:
    fps = float(rec.get("fps") or 25.0)
    out = []
    for e in rec.get("events") or []:
        t = e.get("t")
        if t is None:
            t = float(e.get("frame", 0)) / fps
        out.append(Ev(float(t), e.get("dir", ""), frame=e.get("frame"), raw=e))
    out.sort(key=lambda x: x.t)
    return out


def pred_events(scene_dir: str, fps_default: float = 25.0,
                direction_map: str = "as-is") -> Tuple[List[Ev], str]:
    """读取一个场景的算法事件; 优先 events.jsonl, 退回 counts.json。"""
    ep = osp.join(scene_dir, "events.jsonl")
    if osp.isfile(ep):
        out = []
        with open(ep, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = r.get("timestamp_sec")
                if t is None:
                    t = float(r.get("frame_id", 0)) / fps_default
                out.append(Ev(float(t), r.get("direction", ""),
                              track_id=r.get("track_id"), person_uid=r.get("person_uid"),
                              frame=r.get("frame_id"), raw=r))
        out.sort(key=lambda x: x.t)
        return _remap(out, direction_map), "events.jsonl"

    cp = osp.join(scene_dir, "counts.json")
    if osp.isfile(cp):
        data = _load_json(cp)
        out = []
        for e in data.get("line_events") or []:
            t = float(e.get("frame", 0)) / fps_default
            out.append(Ev(t, e.get("dir", ""), track_id=e.get("id"),
                          frame=e.get("frame"), raw=e))
        out.sort(key=lambda x: x.t)
        return _remap(out, direction_map), "counts.json"

    return [], "missing"


def _remap(evs: List[Ev], direction_map: str) -> List[Ev]:
    if direction_map != "swap":
        return evs
    for e in evs:
        e.dir = {"in": "out", "out": "in"}.get(e.dir, e.dir)
    return evs


# --------------------------------------------------------------------------- #
# 保序一对一匹配 (最多匹配 + 最小总时间差)
# --------------------------------------------------------------------------- #
def match_ordered(pred: Sequence[Ev], gt: Sequence[Ev], tol: float
                  ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """返回 (matched 下标对, 未匹配 pred 下标, 未匹配 gt 下标)。"""
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    NEG = (-1, 0)
    # dp[i][j]: 用前 i 个 pred 与前 j 个 gt 的最优 (匹配数, -总代价)
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]     # 0=skip pred,1=skip gt,2=match
    for i in range(n + 1):
        dp[i][0] = (0, 0)
    for j in range(m + 1):
        dp[0][j] = (0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i - 1][j]
            bk = 0
            if dp[i][j - 1] > best:
                best, bk = dp[i][j - 1], 1
            dt = abs(pred[i - 1].t - gt[j - 1].t)
            if dt <= tol:
                prev = dp[i - 1][j - 1]
                cand = (prev[0] + 1, prev[1] - int(round(dt * 1000)))
                if cand > best:
                    best, bk = cand, 2
            dp[i][j] = best
            back[i][j] = bk

    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        bk = back[i][j]
        if bk == 2:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif bk == 1:
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    mi = {p for p, _ in pairs}
    mj = {q for _, q in pairs}
    return pairs, [k for k in range(n) if k not in mi], [k for k in range(m) if k not in mj]


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# --------------------------------------------------------------------------- #
def eval_scene(pred: List[Ev], gt: List[Ev], tol: float, direction_flip_ok: bool = False):
    """同一场景内按方向分别做一对一匹配。"""
    res: Dict[str, Any] = {}
    tot = {"tp": 0, "fp": 0, "fn": 0}
    details: List[dict] = []
    for d in ("in", "out"):
        P = [e for e in pred if e.dir == d]
        G = [e for e in gt if e.dir == d]
        pairs, up, ug = match_ordered(P, G, tol)
        s = _prf(len(pairs), len(up), len(ug))
        res[d] = s
        tot["tp"] += s["tp"]
        tot["fp"] += s["fp"]
        tot["fn"] += s["fn"]
        for i, j in pairs:
            details.append({"direction": d, "status": "TP",
                            "pred_t": round(P[i].t, 3), "gt_t": round(G[j].t, 3),
                            "dt": round(P[i].t - G[j].t, 3),
                            "pred_track_id": P[i].track_id,
                            "pred_person_uid": P[i].person_uid})
        for i in up:
            details.append({"direction": d, "status": "FP",
                            "pred_t": round(P[i].t, 3), "gt_t": None, "dt": None,
                            "pred_track_id": P[i].track_id,
                            "pred_person_uid": P[i].person_uid})
        for j in ug:
            details.append({"direction": d, "status": "FN",
                            "pred_t": None, "gt_t": round(G[j].t, 3), "dt": None,
                            "pred_track_id": None, "pred_person_uid": None})

    res["overall"] = _prf(tot["tp"], tot["fp"], tot["fn"])

    # 方向混淆诊断: 忽略方向做一次匹配, 统计"时间能配上但方向相反"的数量。
    # 这些**不计入 TP**, 只用于说明有多少 FP/FN 其实来自方向判反。
    pairs_nd, _, _ = match_ordered(sorted(pred, key=lambda x: x.t),
                                   sorted(gt, key=lambda x: x.t), tol)
    conf = 0
    P_all = sorted(pred, key=lambda x: x.t)
    G_all = sorted(gt, key=lambda x: x.t)
    for i, j in pairs_nd:
        if P_all[i].dir != G_all[j].dir:
            conf += 1
    res["direction_confused_pairs"] = conf
    res["details"] = details
    res["n_pred"] = len(pred)
    res["n_gt"] = len(gt)
    return res


# --------------------------------------------------------------------------- #
def _find_scene_dir(pred_dir: str, scene: str) -> Optional[str]:
    cands = [scene]
    # 兼容 "…15490940" vs "…15490940" 之类的轻微差异: 用主干匹配
    try:
        names = os.listdir(pred_dir)
    except Exception:
        return None
    for n in names:
        if n == scene:
            return osp.join(pred_dir, n)
    key = scene.replace(".", "_")
    for n in names:
        if n.replace(".", "_") == key:
            return osp.join(pred_dir, n)
    for n in names:
        if scene in n or n in scene:
            return osp.join(pred_dir, n)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("eval_events", description="事件级一对一匹配评估")
    ap.add_argument("--labels", required=True, help="labels.json (人工逐事件真值)")
    ap.add_argument("--pred-dir", required=True, help="算法结果目录 (每场景一个子目录)")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--tolerance", type=float, default=1.0, help="事件时间容差 (秒)")
    ap.add_argument("--pred-direction-map", default="as-is", choices=["as-is", "swap"],
                    help="swap: 事后交换算法 in/out (仅用于旧结果的可比统计)")
    ap.add_argument("--exclude-scenes", default="",
                    help="逗号分隔, 从正式验收中排除的场景 (如旧线标注不一致的视频)")
    ap.add_argument("--tag", default="", help="写入 metrics.json 的版本标签")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    labels = load_labels(a.labels)
    excluded = {s.strip() for s in a.exclude_scenes.split(",") if s.strip()}

    rows, all_details = [], []
    agg = {"tp": 0, "fp": 0, "fn": 0}
    agg_dir = {"in": {"tp": 0, "fp": 0, "fn": 0}, "out": {"tp": 0, "fp": 0, "fn": 0}}
    agg_excl = {"tp": 0, "fp": 0, "fn": 0}
    n_err_total = 0.0
    n_err_total_excl = 0.0
    conf_total = 0

    for scene, rec in sorted(labels.items()):
        sd = _find_scene_dir(a.pred_dir, scene)
        fps = float(rec.get("fps") or 25.0)
        if sd is None:
            print(f"[warn] 找不到 {scene} 的算法结果目录, 跳过")
            continue
        pe, src = pred_events(sd, fps_default=fps, direction_map=a.pred_direction_map)
        ge = gt_events(rec)
        r = eval_scene(pe, ge, a.tolerance)

        gt_in = sum(1 for e in ge if e.dir == "in")
        gt_out = sum(1 for e in ge if e.dir == "out")
        pr_in = sum(1 for e in pe if e.dir == "in")
        pr_out = sum(1 for e in pe if e.dir == "out")
        err = abs(pr_in - gt_in) + abs(pr_out - gt_out)

        row = {
            "scene": scene, "pred_source": src, "frames_gt": rec.get("total_frames"),
            "fps": fps,
            "gt_in": gt_in, "gt_out": gt_out, "pred_in": pr_in, "pred_out": pr_out,
            "d_in": pr_in - gt_in, "d_out": pr_out - gt_out,
            "abs_err": err,
            "tp": r["overall"]["tp"], "fp": r["overall"]["fp"], "fn": r["overall"]["fn"],
            "precision": r["overall"]["precision"], "recall": r["overall"]["recall"],
            "f1": r["overall"]["f1"],
            "tp_in": r["in"]["tp"], "fp_in": r["in"]["fp"], "fn_in": r["in"]["fn"],
            "tp_out": r["out"]["tp"], "fp_out": r["out"]["fp"], "fn_out": r["out"]["fn"],
            "direction_confused_pairs": r["direction_confused_pairs"],
            "excluded": scene in excluded,
            "annotation_status": "stale_line" if scene in excluded else "ok",
        }
        rows.append(row)
        n_err_total += err
        conf_total += r["direction_confused_pairs"]
        for k in agg:
            agg[k] += row[k]
        for d in ("in", "out"):
            for k in ("tp", "fp", "fn"):
                agg_dir[d][k] += row[f"{k}_{d}"]
        if scene not in excluded:
            for k in agg:
                agg_excl[k] += row[k]
            n_err_total_excl += err

        for d in r["details"]:
            d["scene"] = scene
            all_details.append(d)

    overall = _prf(agg["tp"], agg["fp"], agg["fn"])
    overall_excl = _prf(agg_excl["tp"], agg_excl["fp"], agg_excl["fn"])
    per_dir = {d: _prf(agg_dir[d]["tp"], agg_dir[d]["fp"], agg_dir[d]["fn"])
               for d in ("in", "out")}

    n_scenes = len(rows)
    metrics = {
        "tag": a.tag,
        "labels": a.labels,
        "pred_dir": a.pred_dir,
        "tolerance_sec": a.tolerance,
        "pred_direction_map": a.pred_direction_map,
        "excluded_scenes": sorted(excluded),
        "n_scenes": n_scenes,
        "n_scenes_effective": n_scenes - len([r for r in rows if r["excluded"]]),
        "event_overall": overall,
        "event_overall_effective": overall_excl,
        "event_per_direction": per_dir,
        "direction_confused_pairs_total": conf_total,
        "sum_abs_count_error": n_err_total,
        "sum_abs_count_error_effective": n_err_total_excl,
        "mean_abs_count_error_per_scene": round(n_err_total / n_scenes, 2) if n_scenes else 0.0,
        "per_scene": rows,
    }
    with open(osp.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(osp.join(a.out, "scenario_compare.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scene", "gt_in", "gt_out", "pred_in", "pred_out", "d_in", "d_out",
                    "abs_err", "tp", "fp", "fn", "precision", "recall", "f1",
                    "dir_confused", "excluded", "annotation_status"])
        for r in rows:
            w.writerow([r["scene"], r["gt_in"], r["gt_out"], r["pred_in"], r["pred_out"],
                        r["d_in"], r["d_out"], r["abs_err"], r["tp"], r["fp"], r["fn"],
                        r["precision"], r["recall"], r["f1"],
                        r["direction_confused_pairs"], r["excluded"],
                        r["annotation_status"]])

    with open(osp.join(a.out, "events_match.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scene", "direction", "status", "pred_t", "gt_t", "dt",
                    "pred_track_id", "pred_person_uid"])
        for d in all_details:
            w.writerow([d["scene"], d["direction"], d["status"], d["pred_t"], d["gt_t"],
                        d["dt"], d["pred_track_id"], d["pred_person_uid"]])

    lines = [
        f"# 事件级评估 {a.tag}".rstrip(),
        "",
        f"- 真值: `{a.labels}`",
        f"- 算法: `{a.pred_dir}` (方向映射: {a.pred_direction_map})",
        f"- 时间容差: {a.tolerance} 秒 | 场景数: {n_scenes} "
        f"(正式统计 {metrics['n_scenes_effective']} 段, 排除 {sorted(excluded) or '无'})",
        "",
        "## 事件级指标 (总计)",
        "",
        "| 方向 | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d in ("in", "out"):
        s = per_dir[d]
        lines.append(f"| {d} | {s['tp']} | {s['fp']} | {s['fn']} | "
                     f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |")
    lines.append(f"| **合计** | {overall['tp']} | {overall['fp']} | {overall['fn']} | "
                 f"{overall['precision']:.3f} | {overall['recall']:.3f} | "
                 f"{overall['f1']:.3f} |")
    lines += [
        "",
        f"- 方向判反 (时间可配但方向相反) 对数: **{conf_total}** (不计入 TP)",
        f"- 场景绝对计数误差 Σ(|Δin|+|Δout|): **{n_err_total:.0f}** "
        f"(排除旧线场景后 {n_err_total_excl:.0f})",
        f"- 每场景 MAE: {metrics['mean_abs_count_error_per_scene']}",
        "",
        "## 逐场景",
        "",
        "| 场景 | GT in/out | Pred in/out | Σ|Δ| | TP/FP/FN | P | R | F1 | 方向判反 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        mark = " ⚠️旧线" if r["excluded"] else ""
        lines.append(
            f"| {r['scene']}{mark} | {r['gt_in']}/{r['gt_out']} | "
            f"{r['pred_in']}/{r['pred_out']} | {r['abs_err']} | "
            f"{r['tp']}/{r['fp']}/{r['fn']} | {r['precision']:.2f} | "
            f"{r['recall']:.2f} | {r['f1']:.2f} | {r['direction_confused_pairs']} |")
    with open(osp.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[输出] {a.out}/metrics.json, scenario_compare.csv, events_match.csv, summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import os.path as osp
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load(mod_name: str, filename: str):
    p = osp.join(REPO_ROOT, "coalcount", filename)
    spec = importlib.util.spec_from_file_location(mod_name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


LB = _load("cc_labels", "labels.py")


# --------------------------------------------------------------------------- #
def load_gt(path: str) -> Dict[str, Dict[str, int]]:
    """读真值 -> {scene: {in,out,net}}"""
    if path.endswith(".csv"):
        out = {}
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                out[row["scene"]] = {"in": int(row["gt_in"]), "out": int(row["gt_out"]),
                                     "net": int(row["gt_net"])}
        return out
    store = LB.load_store(path)
    return {r["scene"]: {"in": r["in"], "out": r["out"], "net": r["net"]}
            for r in LB.summary_rows(store)}


def _counts_from_obj(obj: Any, scene_hint: str = "") -> Optional[Dict[str, int]]:
    """从 counts.json 的 dict 里取 line 计数。"""
    if not isinstance(obj, dict):
        return None
    c = obj.get("counts")
    if not isinstance(c, dict):
        return None
    line = c.get("line")
    if not isinstance(line, dict):
        return None
    return {"in": int(line.get("in", 0)), "out": int(line.get("out", 0)),
            "net": int(line.get("net", 0)), "scene": obj.get("name") or scene_hint}


def _read_json(p: str):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_run(path: str) -> Dict[str, Dict[str, int]]:
    """读一个算法的结果 -> {scene: {in,out,net}}"""
    path = path if osp.isabs(path) else osp.join(REPO_ROOT, path)
    out: Dict[str, Dict[str, int]] = {}

    if osp.isfile(path):
        data = _read_json(path)
        if not data:
            return out
        # 批量 summary.json
        results = data.get("results")
        if isinstance(results, list):
            for r in results:
                rec = _counts_from_obj(r, r.get("name", ""))
                if rec and rec.get("scene"):
                    out[rec["scene"]] = rec
            return out
        # 单个 counts.json
        rec = _counts_from_obj(data, osp.basename(osp.dirname(path)))
        if rec and rec.get("scene"):
            out[rec["scene"]] = rec
        return out

    if osp.isdir(path):
        # 目录本身可能就是某个场景
        rec = _counts_from_obj(_read_json(osp.join(path, "counts.json")) or {},
                              osp.basename(osp.normpath(path)))
        if rec and rec.get("scene"):
            out[rec["scene"]] = rec
            return out
        for name in sorted(os.listdir(path)):
            sub = osp.join(path, name)
            if not osp.isdir(sub):
                continue
            rec = _counts_from_obj(_read_json(osp.join(sub, "counts.json")) or {}, name)
            if rec and rec.get("scene"):
                out[rec["scene"]] = rec
    return out


def _match_scene(gt_scene: str, run: Dict[str, Dict[str, int]]) -> Optional[Dict[str, int]]:
    if gt_scene in run:
        return run[gt_scene]
    stem = osp.splitext(gt_scene)[0]
    for k, v in run.items():
        if k == stem or osp.splitext(osp.basename(k))[0] == stem:
            return v
    return None


# --------------------------------------------------------------------------- #
def build_table(gt: Dict[str, Dict[str, int]], runs: Dict[str, Dict[str, Dict[str, int]]]
                ) -> Tuple[List[dict], Dict[str, dict]]:
    """返回 (逐场景行, 每个 run 的汇总指标)。"""
    scenes = sorted(gt.keys())
    rows: List[dict] = []
    for sc in scenes:
        row = {"scene": sc, "gt": gt[sc], "runs": {}}
        for rname, run in runs.items():
            got = _match_scene(sc, run)
            if got is None:
                row["runs"][rname] = None
            else:
                row["runs"][rname] = {
                    "in": got["in"], "out": got["out"], "net": got["net"],
                    "d_in": got["in"] - gt[sc]["in"],
                    "d_out": got["out"] - gt[sc]["out"],
                }
        rows.append(row)

    agg: Dict[str, dict] = {}
    for rname in runs:
        n = 0
        ae_in = ae_out = 0
        tot_in = tot_out = 0
        gt_in = gt_out = 0
        exact_in = exact_out = 0
        for r in rows:
            got = r["runs"].get(rname)
            gt_in += r["gt"]["in"]
            gt_out += r["gt"]["out"]
            if got is None:
                continue
            n += 1
            tot_in += got["in"]
            tot_out += got["out"]
            ae_in += abs(got["d_in"])
            ae_out += abs(got["d_out"])
            exact_in += int(got["d_in"] == 0)
            exact_out += int(got["d_out"] == 0)
        agg[rname] = {
            "n_scenes": n, "n_gt": len(rows),
            "gt_in": gt_in, "gt_out": gt_out,
            "algo_in": tot_in, "algo_out": tot_out,
            "bias_in": tot_in - gt_in, "bias_out": tot_out - gt_out,
            "mae_in": (ae_in / n) if n else None,
            "mae_out": (ae_out / n) if n else None,
            "exact_in": exact_in, "exact_out": exact_out,
        }
    return rows, agg


def print_table(rows: List[dict], agg: Dict[str, dict], names: List[str]):
    def cell(v):
        return f"{v:>4d}" if isinstance(v, int) else "   -"

    w_scene = max([len("scene")] + [len(r["scene"]) for r in rows]) + 1
    hdr = f"{'scene':{w_scene}s} {'GT_IN':>6s} {'GT_OUT':>6s}"
    for nm in names:
        hdr += f" | {nm+' IN':>10s} {'OUT':>5s} {'dIN':>5s} {'dOUT':>5s}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['scene']:{w_scene}s} {r['gt']['in']:6d} {r['gt']['out']:6d}"
        for nm in names:
            got = r["runs"].get(nm)
            if got is None:
                line += f" | {'-':>10s} {'-':>5s} {'-':>5s} {'-':>5s}"
            else:
                di = f"{got['d_in']:+d}"
                do = f"{got['d_out']:+d}"
                line += (f" | {got['in']:10d} {got['out']:5d} {di:>5s} {do:>5s}")
        print(line)
    print("-" * len(hdr))

    tot = f"{'合计':{w_scene}s} {agg[names[0]]['gt_in']:6d} {agg[names[0]]['gt_out']:6d}"
    for nm in names:
        a = agg[nm]
        tot += (f" | {a['algo_in']:10d} {a['algo_out']:5d} "
                f"{a['bias_in']:+5d} {a['bias_out']:+5d}")
    print(tot)

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else "-"

    for label, key in (("MAE(每场景)", "mae"), ("完全一致(场景数)", "exact")):
        line = f"{label:{w_scene}s} {'':6s} {'':6s}"
        for nm in names:
            a = agg[nm]
            if key == "mae":
                line += (f" | {'':10s} {'':5s} {fmt(a['mae_in']):>5s} "
                         f"{fmt(a['mae_out']):>5s}")
            else:
                line += (f" | {'':10s} {'':5s} "
                         f"{str(a['exact_in'])+'/'+str(a['n_scenes']):>5s} "
                         f"{str(a['exact_out'])+'/'+str(a['n_scenes']):>5s}")
        print(line)
    print("=" * len(hdr))
    print("dIN/dOUT = 算法 - 真值 (正数=多计, 负数=漏计)\n")


def to_csv(rows: List[dict], agg: Dict[str, dict], names: List[str], path: str) -> str:
    path = path if osp.isabs(path) else osp.join(REPO_ROOT, path)
    d = osp.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["scene", "gt_in", "gt_out", "gt_net"]
        for nm in names:
            head += [f"{nm}_in", f"{nm}_out", f"{nm}_d_in", f"{nm}_d_out"]
        w.writerow(head)
        for r in rows:
            line = [r["scene"], r["gt"]["in"], r["gt"]["out"], r["gt"]["net"]]
            for nm in names:
                got = r["runs"].get(nm)
                line += (["", "", "", ""] if got is None
                         else [got["in"], got["out"], got["d_in"], got["d_out"]])
            w.writerow(line)
        w.writerow([])
        for nm in names:
            a = agg[nm]
            w.writerow([f"[{nm}] 合计", a["gt_in"], a["gt_out"], "",
                        a["algo_in"], a["algo_out"], a["bias_in"], a["bias_out"]])
            w.writerow([f"[{nm}] MAE/场景", "", "", "",
                        round(a["mae_in"], 3) if a["mae_in"] is not None else "",
                        round(a["mae_out"], 3) if a["mae_out"] is not None else "", "", ""])
    return path


def to_markdown(rows: List[dict], agg: Dict[str, dict], names: List[str],
                labels_path: str, path: str) -> str:
    path = path if osp.isabs(path) else osp.join(REPO_ROOT, path)
    d = osp.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    L = []
    L.append("# 人工真值 vs 算法计数 对照\n")
    L.append(f"- 真值: `{labels_path}`")
    L.append(f"- 对比算法: {', '.join(names)}")
    L.append(f"- 场景数: {len(rows)}\n")

    head = "| 场景 | 真值 IN | 真值 OUT |"
    sep = "|---|---|---|"
    for nm in names:
        head += f" {nm} IN | {nm} OUT | ΔIN | ΔOUT |"
        sep += "---|---|---|---|"
    L.append(head)
    L.append(sep)

    def _row(scene: str, gt_in: Any, gt_out: Any, cells: List[str]) -> str:
        # NOTE: 统一用 " | ".join 拼装。旧实现把 base 串(尾部已带 "|")与
        # " | xxx" 直接相加, 会多出一个空单元格, 渲染出多余的竖线列。
        parts = [str(scene), str(gt_in), str(gt_out)] + [str(c) for c in cells]
        return "| " + " | ".join(parts) + " |"

    for r in rows:
        cells: List[str] = []
        for nm in names:
            got = r["runs"].get(nm)
            if got is None:
                cells += ["-", "-", "-", "-"]
            else:
                cells += [got["in"], got["out"],
                          f"{got['d_in']:+d}", f"{got['d_out']:+d}"]
        L.append(_row(r["scene"], r["gt"]["in"], r["gt"]["out"], cells))

    a0 = agg[names[0]]
    cells = []
    for nm in names:
        a = agg[nm]
        cells += [f"**{a['algo_in']}**", f"**{a['algo_out']}**",
                  f"**{a['bias_in']:+d}**", f"**{a['bias_out']:+d}**"]
    L.append(_row("**合计**", f"**{a0['gt_in']}**", f"**{a0['gt_out']}**", cells))
    L.append("")

    L.append("## 误差指标\n")
    L.append("| 算法 | 参与场景 | MAE(IN) | MAE(OUT) | 偏差(IN) | 偏差(OUT) | IN 完全一致 | OUT 完全一致 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nm in names:
        a = agg[nm]
        mae_i = f"{a['mae_in']:.2f}" if a["mae_in"] is not None else "-"
        mae_o = f"{a['mae_out']:.2f}" if a["mae_out"] is not None else "-"
        L.append(f"| {nm} | {a['n_scenes']}/{a['n_gt']} | {mae_i} | {mae_o} | "
                 f"{a['bias_in']:+d} | {a['bias_out']:+d} | "
                 f"{a['exact_in']}/{a['n_scenes']} | {a['exact_out']}/{a['n_scenes']} |")
    L.append("")
    L.append("> Δ = 算法 − 真值。正数表示多计, 负数表示漏计。")
    L.append("> MAE 为逐场景绝对误差的平均, 反映单条视频的稳定性; 偏差反映系统性高/低估。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        "compare_counts",
        description="对比人工真值与不同算法的计数结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--labels", default="configs/coalcount/labels.json",
                    help="真值文件 (labels.json 或 labels.csv)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH",
                    help="算法结果, 可重复。PATH 可为 summary.json / 输出根目录 / counts.json")
    ap.add_argument("--out-csv", default=None, help="导出对照 CSV")
    ap.add_argument("--out-md", default=None, help="导出 Markdown 报告")
    a = ap.parse_args(argv)

    labels_path = a.labels if osp.isabs(a.labels) else osp.join(REPO_ROOT, a.labels)
    gt = load_gt(labels_path)
    if not gt:
        print(f"[错误] 真值里没有任何场景: {labels_path}\n"
              f"        先跑 tools/count_label.py 生成标签。")
        return 2
    print(f"真值: {labels_path}  ({len(gt)} 个场景)")

    runs: Dict[str, Dict[str, Dict[str, int]]] = {}
    for spec in a.run:
        if "=" not in spec:
            print(f"[警告] --run 需要 NAME=PATH 形式, 跳过: {spec}")
            continue
        name, path = spec.split("=", 1)
        r = load_run(path)
        if not r:
            print(f"[警告] {name}: 没读到任何计数结果 ({path})")
        runs[name] = r
        print(f"算法 {name}: {len(r)} 个场景  ({path})")
    if not runs:
        print("\n[提示] 没有指定 --run, 只显示真值:\n")
        names = []
    else:
        names = list(runs.keys())

    rows, agg = build_table(gt, runs) if names else (
        [{"scene": s, "gt": v, "runs": {}} for s, v in sorted(gt.items())], {})

    if names:
        print_table(rows, agg, names)
    else:
        print(f"{'scene':44s} {'IN':>5s} {'OUT':>5s} {'NET':>5s}")
        for r in rows:
            print(f"{r['scene']:44s} {r['gt']['in']:5d} {r['gt']['out']:5d} {r['gt']['net']:5d}")

    if a.out_csv and names:
        print("CSV:", to_csv(rows, agg, names, a.out_csv))
    if a.out_md and names:
        print("Markdown:", to_markdown(rows, agg, names, a.labels, a.out_md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""命令行入口: 单场景 / 批量场景 运行, 配置覆盖, 结果汇总."""
from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import sys
from typing import List

from .config import Config, REPO_ROOT
from .models import MODEL_REGISTRY, list_scales
from .pipeline import run_video, derive_scene_name

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("coalcount")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "coalcount",
        description="煤矿人员入井/升井 AI 计数 (基于 McByte++ 的无训练长时多目标跟踪)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=None, help="JSON/YAML 配置文件")
    p.add_argument("--input", "-i", action="append", default=[], help="视频文件或帧目录 (可重复)")
    p.add_argument("--videos-file", default=None, help="每行一个输入路径的清单文件 (批量)")
    p.add_argument("--out", "-o", default=None, help="输出根目录 (覆盖配置 io.out_dir)")
    p.add_argument("--tag", default=None, help="输出子目录后缀, 便于区分不同实验")

    # 覆盖项
    p.add_argument("--detector", default=None, help="检测器尺度: nano|tiny|s|m|l|x|x_sports|x_mot17")
    p.add_argument("--device", default=None, help="如 cuda:0 / cpu")
    p.add_argument("--mask", dest="mask", action="store_true", default=None, help="启用 SAM/EdgeTAM 掩码传播")
    p.add_argument("--no-mask", dest="mask", action="store_false", help="关闭掩码传播")
    p.add_argument("--reid", dest="reid", action="store_true", default=None, help="启用 re-ID")
    p.add_argument("--no-reid", dest="reid", action="store_false", help="关闭 re-ID")
    p.add_argument("--no-cmc", action="store_true", help="关闭相机运动补偿")
    p.add_argument("--cmc-method", default=None,
                   choices=["none", "orb", "ecc", "sparseOptFlow"],
                   help="相机运动补偿方法 (覆盖配置)")
    p.add_argument("--max-frames", type=int, default=None, help="每个场景最多处理帧数 (0=全部)")
    p.add_argument("--start-frame", type=int, default=None, help="起始帧 (1-based)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="每个场景最多处理的原视频时长(秒), 按源帧率换算成帧数")
    p.add_argument("--stride", type=int, default=None, help="抽帧步长 (掩码开启时强制 1)")
    p.add_argument("--decode", default=None, choices=["memory", "extract"],
                   help="视频解码方式: memory=内存流式不落盘(默认) | extract=先抽帧到临时目录")
    p.add_argument("--no-video", action="store_true", help="不输出可视化视频")
    p.add_argument("--no-vis", action="store_true", help="不做可视化 (更快)")
    p.add_argument("--no-trail", action="store_true", help="不绘制锚点轨迹")
    p.add_argument("--vis-scale", type=float, default=None,
                   help="可视化输出缩放 (0.5 = 半分辨率, 减小视频体积)")
    p.add_argument("--no-quality", action="store_true", help="关闭误检质量门 (不做误检判断)")
    p.add_argument("--keep-frames", action="store_true",
                   help="保留从视频抽取的中间帧目录 (默认处理后删除)")
    p.add_argument("--line", nargs=4, type=float, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"), help="计数线归一化坐标")
    p.add_argument("--lines-file", default=None,
                   help="人工绘制计数线存储文件 (默认取配置 counter.line_file); "
                        "按场景自动套用, 优先于 --line 与自动标定")
    p.add_argument("--no-lines", action="store_true",
                   help="忽略人工计数线文件, 强制使用配置/自动标定")
    p.add_argument("--in-direction", default=None, choices=["fwd", "bwd"], help="哪个方向算'入井'")
    p.add_argument("--anchor", default=None, choices=["bottom_center", "center"], help="锚点模式")
    p.add_argument("--set", dest="set", action="append", default=[], metavar="KEY=VALUE",
                   help="覆盖任意配置项 (可重复); 值优先按 JSON 解析, 失败则按字符串。"
                        "例: --set counter.algorithm=fsm_v1 --set counter.line.band_rel=0.16")

    p.add_argument("--list-scales", action="store_true", help="列出可用检测器尺度后退出")
    p.add_argument("--dump-config", action="store_true", help="打印最终配置后退出")
    return p


def _apply_overrides(cfg: Config, a: argparse.Namespace) -> Config:
    if a.detector:
        if a.detector not in MODEL_REGISTRY:
            raise SystemExit(f"未知尺度 {a.detector}; 可选: {list(MODEL_REGISTRY)}")
        cfg.set("detector.scale", a.detector)
        # 切换尺度时清空显式覆盖, 交给注册表解析
        cfg.set("detector.exp_file", None)
        cfg.set("detector.ckpt", None)
        cfg.set("detector.preproc", None)
    if a.device:
        cfg.set("run.device", a.device)
    if a.mask is not None:
        cfg.set("mask.enabled", bool(a.mask))
    if a.reid is not None:
        cfg.set("reid.enabled", bool(a.reid))
    if a.no_cmc:
        cfg.set("tracker.cmc_method", "none")
    if a.cmc_method:
        cfg.set("tracker.cmc_method", a.cmc_method)
    if a.max_frames is not None:
        cfg.set("io.max_frames", int(a.max_frames))
    if a.start_frame is not None:
        cfg.set("io.start_frame", int(a.start_frame))
    if a.max_seconds is not None:
        cfg.set("io.max_seconds", float(a.max_seconds))
    if a.stride is not None:
        cfg.set("io.stride", int(a.stride))
    if a.decode:
        cfg.set("io.decode", a.decode)
    if a.no_video:
        cfg.set("io.save_video", False)
    if a.no_vis:
        cfg.set("io.vis", False)
        cfg.set("io.save_video", False)
    if a.no_trail:
        cfg.set("io.draw_trajectory", False)
    if a.vis_scale is not None:
        cfg.set("io.vis_scale", float(a.vis_scale))
    if a.no_quality:
        cfg.set("counter.quality.enabled", False)
    if a.keep_frames:
        cfg.set("io.save_frames", True)
    if a.out:
        cfg.set("io.out_dir", a.out)
    if a.tag:
        cfg.set("run.tag", a.tag)
    if a.line:
        cfg.set("counter.line.p1", [a.line[0], a.line[1]])
        cfg.set("counter.line.p2", [a.line[2], a.line[3]])
        # 显式 --line 表示"就要这条线", 除非同时指定了 --lines-file
        if not a.lines_file:
            cfg.set("counter.line_file", None)
            cfg.set("counter.line.auto", False)
    if a.no_lines:
        cfg.set("counter.line_file", None)
    if a.lines_file:
        cfg.set("counter.line_file", a.lines_file)
    if a.in_direction:
        cfg.set("counter.line.in_direction", a.in_direction)
    if a.anchor:
        cfg.set("counter.anchor", a.anchor)
    # 通用覆盖放最后, 保证优先级最高 (便于消融实验逐项开关)
    for kv in (a.set or []):
        if "=" not in kv:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式, 收到: {kv}")
        key, val = kv.split("=", 1)
        key, val = key.strip(), val.strip()
        if not key:
            raise SystemExit(f"--set 的 KEY 不能为空: {kv}")
        try:
            parsed = json.loads(val)
        except Exception:
            parsed = val
        cfg.set(key, parsed)
    return cfg


def _collect_inputs(a: argparse.Namespace) -> List[str]:
    inputs = list(a.input)
    if a.videos_file:
        with open(a.videos_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    inputs.append(s)
    return inputs


def _print_table(results: List[dict]):
    hdr = (f"{'scene':38s} {'frames':>7s} {'ids':>5s} {'IN':>5s} {'OUT':>5s} {'NET':>5s} "
           f"{'line':>9s} {'DET/f':>7s} {'FPS':>7s} {'GPU_MB':>7s}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        line = r.get("counts", {}).get("line", {}) or {}
        perf = r.get("perf", {}) or {}
        lu = r.get("counts", {}).get("line_used", {}) or {}
        src = str(lu.get("source", ""))
        if src.startswith("manual"):
            src = "manual"
        name = (r["name"][:37] + "…") if len(r["name"]) > 38 else r["name"]
        print(
            f"{name:38s} {r.get('frames', 0):7d} {r.get('unique_ids', 0):5d} "
            f"{line.get('in', 0):5d} {line.get('out', 0):5d} {line.get('net', 0):5d} "
            f"{src:>9s} "
            f"{r.get('counts', {}).get('det_per_frame_hi', 0):7.2f} "
            f"{perf.get('fps', 0):7.2f} {str(perf.get('peak_gpu_mem_mb')):>7s}"
        )
    print("=" * len(hdr))
    tot_in = sum((r.get("counts", {}).get("line", {}) or {}).get("in", 0) for r in results)
    tot_out = sum((r.get("counts", {}).get("line", {}) or {}).get("out", 0) for r in results)
    ok = sum(1 for r in results if r.get("ok", True))
    print(f"合计: 场景 {ok}/{len(results)} 成功 | 进 {tot_in} | 出 {tot_out}\n")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    if a.list_scales:
        print("可用检测器尺度:")
        for k, v in list_scales().items():
            print(f"  {k:10s} {v}")
        return 0

    base = Config.from_file(a.config) if a.config else Config.from_dict({})
    cfg = _apply_overrides(base, a)

    if a.dump_config:
        print(json.dumps(cfg.as_dict(), ensure_ascii=False, indent=2))
        return 0

    inputs = _collect_inputs(a)
    if not inputs:
        print("[错误] 未提供输入。用 --input <视频> 或 --videos-file <清单>")
        return 2

    for line in cfg.describe():
        logger.info("cfg | " + line)

    out_root = cfg.path(cfg.get("io.out_dir"))
    tag = cfg.get("run.tag") or ""
    if tag:
        out_root = osp.join(out_root, tag)
    os.makedirs(out_root, exist_ok=True)

    results: List[dict] = []
    # 单输入且显式指定 --out 时, 直接把 --out 当作该场景的输出目录 (不再套一层场景名)
    single_explicit_out = bool(a.out) and len(inputs) == 1
    for idx, inp in enumerate(inputs, 1):
        name = derive_scene_name(inp)
        if name in {r["name"] for r in results}:
            name = f"{name}_{idx}"
        logger.info(f"[{idx}/{len(inputs)}] >>> {name}")
        run_dir = out_root if single_explicit_out else osp.join(out_root, name)
        try:
            r = run_video(cfg, inp, out_dir=run_dir, name=name)
        except Exception as e:  # 兜底: 单场景失败不影响批量
            from .pipeline import RunResult

            r = RunResult(name=name, input_path=inp, ok=False, error=str(e))
        results.append(r.to_dict())
        if not r.ok:
            logger.error(f"[{idx}/{len(inputs)}] 失败: {name} -> {r.error}")

    # 汇总
    summary = {
        "config": cfg.as_dict(),
        "n_scenes": len(results),
        "n_ok": sum(1 for r in results if r["ok"]),
        "results": results,
    }
    sp = osp.join(out_root, "summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cp = osp.join(out_root, "summary.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scene", "ok", "frames", "unique_ids", "in", "out", "net",
                    "line_source", "det_per_frame_hi", "det_per_frame", "n_rejected",
                    "fps", "latency_mean_ms", "latency_p95_ms", "gpu_peak_mb", "rss_mb", "error"])
        for r in results:
            line = r.get("counts", {}).get("line", {}) or {}
            perf = r.get("perf", {}) or {}
            lat = perf.get("latency_ms", {}) or {}
            q = r.get("counts", {}).get("quality", {}) or {}
            lu = r.get("counts", {}).get("line_used", {}) or {}
            w.writerow([
                r["name"], r["ok"], r.get("frames", 0), r.get("unique_ids", 0),
                line.get("in", 0), line.get("out", 0), line.get("net", 0),
                lu.get("source", ""),
                r.get("counts", {}).get("det_per_frame_hi", ""),
                r.get("counts", {}).get("det_per_frame", ""),
                q.get("n_rejected", ""),
                perf.get("fps", 0), lat.get("mean", ""), lat.get("p95", ""),
                perf.get("peak_gpu_mem_mb", ""), perf.get("rss_mb", ""), r.get("error", ""),
            ])

    _print_table(results)
    logger.info(f"汇总: {sp}")
    logger.info(f"汇总: {cp}")
    print(json.dumps({"summary_json": sp, "summary_csv": cp}, ensure_ascii=False))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

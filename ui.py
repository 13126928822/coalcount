from __future__ import annotations

import os
import os.path as osp
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 颜色 (BGR)
GREEN = (80, 220, 80)
RED = (0, 0, 255)
YELLOW = (0, 220, 220)
BLUE = (255, 160, 60)
ORANGE = (60, 180, 255)
MAGENTA = (200, 80, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (170, 170, 170)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Unicode 安全的图片读写
# --------------------------------------------------------------------------- #
def imread(path: str):
    """读图 (中文路径安全)。失败返回 None。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return cv2.imread(path)
    if data is None or data.size == 0:
        return None
    try:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path: str, img) -> bool:
    """写图 (中文路径安全)。"""
    ext = osp.splitext(path)[1] or ".jpg"
    try:
        ok, buf = cv2.imencode(ext, img)
    except Exception:
        return bool(cv2.imwrite(path, img))
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return bool(cv2.imwrite(path, img))


# --------------------------------------------------------------------------- #
def basename_any(path: str) -> str:
    """跨平台取文件名 (清单里可能是 Linux 路径, 在 Windows 上也要取对)。"""
    return osp.basename(str(path).replace("\\", "/"))


def put_text(img, text: str, org, scale: float = 0.7, color=WHITE,
             thick: int = 2, bg: bool = True):
    """带半透明黑底的文字 (任何背景下都看得清)。"""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg:
        x0, y0 = max(0, x - 4), max(0, y - th - 6)
        x1, y1 = x + tw + 6, y + bl + 4
        sub = img[y0:y1, x0:x1]
        if sub.size:
            cv2.rectangle(img, (x0, y0), (x1, y1), BLACK, -1)
            cv2.addWeighted(sub, 0.55, img[y0:y1, x0:x1], 0.45, 0, img[y0:y1, x0:x1])
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return img


def gui_available() -> bool:
    """预检当前 cv2 是否支持图形窗口 (headless 版 / 无显示器会失败)。"""
    try:
        cv2.namedWindow("__cc_probe__", cv2.WINDOW_NORMAL)
    except cv2.error:
        return False
    try:
        cv2.destroyWindow("__cc_probe__")
    except cv2.error:
        pass
    return True


def destroy_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


# --------------------------------------------------------------------------- #
def line_geometry(p1_n, p2_n, width: int, height: int):
    """归一化端点 -> 像素端点。"""
    p1 = (float(p1_n[0]) * width, float(p1_n[1]) * height)
    p2 = (float(p2_n[0]) * width, float(p2_n[1]) * height)
    return p1, p2


def draw_line_overlay(img, p1, p2, in_direction: str = "fwd",
                      hysteresis_px: float = 0.0, arrow_len: int = 70,
                      scale: float = 1.0, show_points: bool = True,
                      tag: str = "IN", in_normal=None):
    """画计数线 + 端点 + "进井"方向箭头 (+ 可选死区带)。

    方向语义 (2026-10-11 起唯一):
      * 首选显式单位法向量 ``in_normal`` (朝入井侧);
      * 未给时按历史约定 ``normal = (-dy, dx)`` (``in_direction="bwd"`` 取负) 推导。
    两种情况都与 ``coalcount/crossing.py`` 的 ``in_normal`` 及计数逻辑**同一套公式**,
    保证"标注员看到的 IN 箭头"就是"实际生效的进井方向"。

    p1/p2 为**像素坐标**; scale 用于把线宽/箭头长度按显示缩放做补偿。
    """
    if p1 is None or p2 is None:
        return img
    a = (int(round(p1[0])), int(round(p1[1])))
    b = (int(round(p2[0])), int(round(p2[1])))

    v = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return img
    d = v / n
    normal = np.array([-d[1], d[0]])
    if in_direction == "bwd":
        normal = -normal
    if in_normal is not None:
        nn = np.array([float(in_normal[0]), float(in_normal[1])], dtype=float)
        ln = float(np.linalg.norm(nn))
        if ln > 1e-9:
            normal = nn / ln

    thick = max(2, int(round(3 * scale)))
    # 死区带 (±hysteresis)
    if hysteresis_px and hysteresis_px > 0:
        half = max(2.0, float(hysteresis_px))
        q1 = np.array(p1, float) + normal * half
        q2 = np.array(p2, float) + normal * half
        r1 = np.array(p1, float) - normal * half
        r2 = np.array(p2, float) - normal * half
        poly = np.array([[q1, q2, r2, r1]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, poly, YELLOW)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    cv2.line(img, a, b, YELLOW, thick, cv2.LINE_AA)

    mid = (np.array(a, float) + np.array(b, float)) / 2.0
    tip = mid + normal * (arrow_len * max(1.0, scale))
    cv2.arrowedLine(img, tuple(mid.astype(int)), tuple(tip.astype(int)),
                    GREEN, thick, tipLength=0.25)
    if tag:
        put_text(img, tag, (int(tip[0]) + 8, int(tip[1])), 0.7 * max(0.7, scale), GREEN)

    if show_points:
        r = max(5, int(round(6 * scale)))
        cv2.circle(img, a, r, GREEN, -1, cv2.LINE_AA)
        cv2.circle(img, a, r, BLACK, 1, cv2.LINE_AA)
        cv2.circle(img, b, r, BLUE, -1, cv2.LINE_AA)
        cv2.circle(img, b, r, BLACK, 1, cv2.LINE_AA)
        put_text(img, "P1", (a[0] + 10, a[1] - 8), 0.6, GREEN)
        put_text(img, "P2", (b[0] + 10, b[1] - 8), 0.6, BLUE)
    return img


def fit_scale(width: int, height: int, max_side: int) -> float:
    """返回显示缩放系数 (>=1 表示需要缩小)。"""
    if not max_side or max_side <= 0:
        return 1.0
    s = max(width, height) / float(max_side)
    return 1.0 if s <= 1.0 else s


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def help_overlay(img, lines: Sequence[str], width: int = 640):
    """居中半透明帮助面板。"""
    h, w = img.shape[:2]
    bw = min(width, w - 40)
    lh = 30
    bh = 30 + lh * len(lines)
    x0, y0 = max(0, (w - bw) // 2), max(0, (h - bh) // 2)
    y1 = min(h, y0 + bh)
    sub = img[y0:y1, x0:x0 + bw]
    if sub.size:
        cv2.rectangle(sub, (0, 0), (bw, y1 - y0), BLACK, -1)
        cv2.addWeighted(sub, 0.8, img[y0:y1, x0:x0 + bw], 0.2, 0, img[y0:y1, x0:x0 + bw])
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0 + 16, y0 + 30 + lh * i), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    return img

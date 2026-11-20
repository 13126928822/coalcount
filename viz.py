"""可视化: 在跟踪结果帧上叠加计数线/区域与计数 HUD."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_GREEN = (0, 200, 0)
_RED = (0, 0, 255)
_BLUE = (255, 160, 0)
_YELLOW = (0, 220, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

# 稳定的身份调色板 (BGR), 按 id 取模, 保证同一目标颜色不跳变
_PALETTE = [
    (0, 220, 220),   # 黄
    (80, 220, 80),   # 绿
    (255, 160, 60),  # 蓝
    (200, 80, 255),  # 紫
    (60, 180, 255),  # 橙
    (255, 120, 200), # 粉
    (140, 255, 180), # 浅绿
    (255, 220, 120), # 浅蓝
]


def id_color(tid: int):
    """给轨迹一个稳定颜色。"""
    return _PALETTE[int(tid) % len(_PALETTE)]



def _put(img, text, org, scale=0.7, color=_WHITE, thick=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    if bg:
        cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 6, y + bl + 4), _BLACK, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_line(img, p1, p2, in_direction: str = "fwd", in_normal=None):
    """画计数线 + 方向箭头 (箭头指向'进井'方向)。

    方向优先取显式 ``in_normal`` (朝入井侧的单位法向量); 未给时按
    ``normal = (-dy, dx)`` (``in_direction="bwd"`` 取负) 推导。
    与 ``ui.draw_line_overlay`` 和计数逻辑共用同一套语义。
    """
    if p1 is None or p2 is None:
        return img
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    cv2.line(img, p1, p2, _YELLOW, 3, cv2.LINE_AA)
    cv2.circle(img, p1, 6, _YELLOW, -1)
    cv2.circle(img, p2, 6, _YELLOW, -1)

    v = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    n = np.linalg.norm(v) + 1e-9
    d = v / n
    # 法向: 指向 in 侧
    normal = np.array([-d[1], d[0]])
    if in_direction == "bwd":
        normal = -normal
    if in_normal is not None:
        nn = np.asarray(in_normal, dtype=float).reshape(-1)
        ln = float(np.linalg.norm(nn))
        if ln > 1e-9:
            normal = nn / ln
    mid = (np.array(p1) + np.array(p2)) / 2.0
    tip = mid + normal * 60
    cv2.arrowedLine(img, tuple(mid.astype(int)), tuple(tip.astype(int)), _GREEN, 3, tipLength=0.25)
    _put(img, "IN", (int(tip[0]) + 6, int(tip[1])), 0.7, _GREEN)
    return img


def draw_trails(img, trails, thickness: int = 3, alpha: float = 0.75, head_radius: int = 5):
    """绘制每个身份的锚点轨迹 (折线 + 头部圆点)。

    trails: {identity: [(x, y), ...]}, 按时间顺序 (旧->新), 像素坐标。
    轨迹线以半透明叠加, 不遮挡检测框; 头部圆点标出当前锚点 (脚点)。
    一眼能看出"目标从哪来、往哪去、有没有穿过计数线"。
    """
    if not trails:
        return img
    overlay = img.copy()
    heads = []
    for tid, pts in trails.items():
        if pts is None or len(pts) < 2:
            if pts:
                heads.append((tid, pts[-1]))
            continue
        col = id_color(tid)
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [arr], False, col, thickness, cv2.LINE_AA)
        heads.append((tid, pts[-1]))
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    # 头部锚点画实心, 保证在任何背景下都可见
    for tid, p in heads:
        cv2.circle(img, (int(p[0]), int(p[1])), head_radius, id_color(tid), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), head_radius, _BLACK, 1, cv2.LINE_AA)
    return img


def draw_zone(img, poly, name: str = "zone", occupancy: Optional[int] = None):
    if poly is None:
        return img
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, _BLUE, 2, cv2.LINE_AA)
    c = pts.reshape(-1, 2).mean(axis=0).astype(int)
    label = name if occupancy is None else f"{name}: {occupancy}"
    _put(img, label, (int(c[0]) - 30, int(c[1])), 0.6, _BLUE)
    return img


def draw_rejected(img, tlwh, label: str = "FP?"):
    """把被判为误检的轨迹框标红并打标签 (判断+消除的可视化证据)。"""
    x, y, w, h = [int(v) for v in tlwh]
    cv2.rectangle(img, (x, y), (x + w, y + h), _RED, 2, cv2.LINE_AA)
    _put(img, label, (x, max(y - 6, 14)), 0.6, _RED)
    return img


def draw_hud(img, frame_id: int, num_tracks: int, summary: dict, fps_inst: Optional[float] = None,
             n_rejected: int = 0):
    h, w = img.shape[:2]
    scale = max(0.6, min(w, h) / 1000.0)
    y = int(30 * scale) + 10
    persons = f"frame {frame_id}  persons {num_tracks}"
    if n_rejected:
        persons += f"  (rej {n_rejected})"
    _put(img, persons, (12, y), scale, _WHITE)
    y += int(34 * scale)

    line = summary.get("line")
    if line:
        _put(img, f"IN {line['in']}   OUT {line['out']}   NET {line['net']}",
             (12, y), scale, _GREEN)
        y += int(34 * scale)
    zone = summary.get("zone")
    if zone:
        _put(img, f"ZONE {zone['name']} occ {zone.get('occupancy_final', 0)}",
             (12, y), scale, _BLUE)
        y += int(34 * scale)
    if fps_inst:
        _put(img, f"{fps_inst:.1f} FPS", (12, y), scale, _YELLOW)
    return img

# -*- coding: utf-8 -*-
"""
七夕花束插画（两行祝福 + 竖直爱心）
运行后会在当前目录生成 qixi_bouquet_text_v3.png
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, PathPatch, Polygon, Circle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib import patheffects as pe
from matplotlib import font_manager as fm

random.seed(42)
np.random.seed(42)

# ---------- 基础绘图工具 ----------
def cubic_bezier_path(p0, p1, p2, p3):
    verts = [p0, p1, p2, p3]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)

def draw_stem(ax, start, end, curve=0.1, width=2.5, color="#2e8b57"):
    x0, y0 = start; x3, y3 = end
    dx, dy = x3 - x0, y3 - y0
    p1 = (x0 + dx*0.33 - dy*curve, y0 + dy*0.33 + dx*curve)
    p2 = (x0 + dx*0.66 + dy*curve, y0 + dy*0.66 - dx*curve)
    path = cubic_bezier_path((x0, y0), p1, p2, (x3, y3))
    ax.add_patch(PathPatch(path, fill=False, lw=width, capstyle="round",
                           joinstyle="round", color=color, alpha=0.95))

def draw_leaf(ax, attach_pt, size=0.05, angle_deg=0, color="#3c9d6d"):
    x, y = attach_pt
    w = size; h = size*1.6
    p0 = (0, 0)
    left  = cubic_bezier_path(p0, (-w*0.6, h*0.2), (-w*0.9, h*0.8), (0, h))
    right = cubic_bezier_path(p0, ( w*0.6, h*0.2), ( w*0.9, h*0.8), (0, h))
    trans = Affine2D().rotate_deg_around(0, 0, angle_deg).translate(x, y)
    ax.add_patch(PathPatch(trans.transform_path(left),  facecolor=color, edgecolor="none", alpha=0.98))
    ax.add_patch(PathPatch(trans.transform_path(right), facecolor=color, edgecolor="none", alpha=0.98))

def draw_flower(ax, center, petal_len=0.12, petal_w=0.06, petals=8,
                petal_color="#ff8fb1", core_color="#ffd166", edge_alpha=0.15):
    cx, cy = center
    angles = np.linspace(0, 360, petals, endpoint=False)
    r_offset = petal_len*0.3
    for ang in angles:
        rad = np.deg2rad(ang)
        ox = cx + r_offset*np.cos(rad)
        oy = cy + r_offset*np.sin(rad)
        e = Ellipse((ox, oy), width=petal_w, height=petal_len, angle=ang,
                    facecolor=petal_color, edgecolor="k", lw=0.6, alpha=0.98)
        ax.add_patch(e)
    ax.add_patch(Circle((cx, cy), radius=petal_len*0.55, facecolor=petal_color,
                        edgecolor="none", alpha=edge_alpha))
    ax.add_patch(Circle((cx, cy), radius=petal_w*0.35, facecolor=core_color,
                        edgecolor="k", lw=0.5))

def draw_wrap_and_ribbon(ax):
    poly_pts = np.array([
        [0.50, 0.08],
        [0.35, 0.18],
        [0.26, 0.52],
        [0.74, 0.52],
        [0.65, 0.18],
    ])
    wrap = Polygon(poly_pts, closed=True, facecolor="#f2e9e4", edgecolor="#e3d5ca", lw=2, alpha=0.98)
    ax.add_patch(wrap)
    left_fold = Polygon(np.array([[0.28, 0.52], [0.18, 0.60], [0.32, 0.62], [0.36, 0.52]]),
                        closed=True, facecolor="#efe2db", edgecolor="#e3d5ca", lw=1.5, alpha=0.98)
    right_fold = Polygon(np.array([[0.64, 0.52], [0.68, 0.52], [0.82, 0.60], [0.72, 0.62]]),
                         closed=True, facecolor="#efe2db", edgecolor="#e3d5ca", lw=1.5, alpha=0.98)
    ax.add_patch(left_fold); ax.add_patch(right_fold)
    ax.add_patch(Circle((0.5, 0.22), radius=0.018, facecolor="#e63946", edgecolor="none", alpha=0.95))
    left_bow = cubic_bezier_path((0.5, 0.22), (0.45, 0.25), (0.40, 0.22), (0.42, 0.19))
    right_bow = cubic_bezier_path((0.5, 0.22), (0.55, 0.25), (0.60, 0.22), (0.58, 0.19))
    ax.add_patch(PathPatch(left_bow, facecolor="#ff6b6b", edgecolor="#c1121f", lw=1.2, alpha=0.95))
    ax.add_patch(PathPatch(right_bow, facecolor="#ff6b6b", edgecolor="#c1121f", lw=1.2, alpha=0.95))
    tail1 = cubic_bezier_path((0.5, 0.22), (0.49, 0.15), (0.47, 0.12), (0.46, 0.09))
    tail2 = cubic_bezier_path((0.5, 0.22), (0.51, 0.15), (0.53, 0.12), (0.54, 0.09))
    ax.add_patch(PathPatch(tail1, edgecolor="#c1121f", lw=2.0, facecolor="none"))
    ax.add_patch(PathPatch(tail2, edgecolor="#c1121f", lw=2.0, facecolor="none"))

def background_gradient(ax):
    h, w = 900, 650
    top = np.array([1.00, 0.94, 0.96])
    bottom = np.array([1.00, 1.00, 1.00])
    t = np.linspace(0, 1, h)[:, None]
    grad = top*(1-t) + bottom*t
    grad = np.tile(grad, (1, w, 1))
    ax.imshow(grad, extent=[0,1,0,1], origin="lower", interpolation="bilinear", zorder=-10)

def pick_chinese_font():
    candidates = [
        "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC",
        "SimHei", "STHeiti", "Arial Unicode MS", "DejaVu Sans"
    ]
    for name in candidates:
        try:
            path = fm.findfont(name, fallback_to_default=False)
            if os.path.exists(path):
                return name
        except Exception:
            continue
    return "sans-serif"

# ---------- 爱心 & 文字 ----------
def draw_heart_upright(ax, center=(0.5, 0.56), size=0.025, color="#ff4d6d", edge="#b22234", lw=1.2):
    """竖直向上的心形"""
    cx, cy = center
    s = size
    t = np.linspace(0, 2*np.pi, 200)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    x = x / max(abs(x)) * s * 2.0
    y = y / max(abs(y)) * s * 2.0
    verts = np.column_stack([x+cx, y+cy])
    path = Path(verts, closed=True)
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor=edge, lw=lw, alpha=0.95))

def add_greeting_multiline(ax, fontsize=25):
    font_name = pick_chinese_font()
    t1 = ax.text(0.5, 0.60, "2025.8.29 七夕节快乐", ha="center", va="center",
                 fontsize=fontsize, fontname=font_name, color="#222",
                 alpha=0.98, zorder=20)
    t2 = ax.text(0.5, 0.54, "航航     周周", ha="center", va="center",
                 fontsize=fontsize, fontname=font_name, color="#222",
                 alpha=0.98, zorder=20)
    for t in (t1, t2):
        t.set_path_effects([pe.withStroke(linewidth=4, foreground="white", alpha=0.9)])
    draw_heart_upright(ax, center=(0.5, 0.54), size=0.030, color="#ff4d6d", edge="#b22234", lw=1.0)

# ---------- 主函数 ----------
def make_bouquet(out_png="qixi_bouquet_text_v3.png"):
    fig = plt.figure(figsize=(7.2, 10.4), dpi=150)
    ax = plt.gca()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")

    background_gradient(ax)

    centers = [
        (0.42, 0.70), (0.58, 0.72), (0.50, 0.80),
        (0.33, 0.60), (0.67, 0.62), (0.40, 0.58), (0.60, 0.58)
    ]
    petal_palette = ["#ff8fb1", "#ffa1cf", "#ffc8dd", "#f4978e",
                     "#f8a5c2", "#cdb4db", "#ffb5a7", "#ffd6a5", "#e5989b"]
    core_palette  = ["#ffd166", "#ffe066", "#ffdd99", "#fde68a"]

    for i, (cx, cy) in enumerate(centers):
        start = (0.50 + (i-3)*0.01, 0.08 + abs(i-3)*0.002)
        draw_stem(ax, start, (cx, cy-0.08), curve=0.12 + (i%3)*0.03, width=2.5, color="#2e8b57")
        mid = ((start[0]+cx)/2, (start[1]+cy)/2)
        draw_leaf(ax, (mid[0]-0.03, mid[1]+0.02), size=0.05, angle_deg=random.choice([10, 25, -15, -30]))
        draw_leaf(ax, (mid[0]+0.03, mid[1]), size=0.045, angle_deg=random.choice([160, 200, -170]))

    for cx, cy in centers:
        petals = random.choice([7, 8, 9, 10])
        petal_len = random.uniform(0.10, 0.14)
        petal_w   = random.uniform(0.05, 0.07)
        pc = random.choice(petal_palette)
        cc = random.choice(core_palette)
        draw_flower(ax, (cx, cy), petal_len=petal_len, petal_w=petal_w, petals=petals,
                    petal_color=pc, core_color=cc, edge_alpha=0.18)

    draw_wrap_and_ribbon(ax)
    draw_leaf(ax, (0.44, 0.18), size=0.08, angle_deg=-40, color="#4da768")
    draw_leaf(ax, (0.56, 0.18), size=0.085, angle_deg=210, color="#4da768")

    add_greeting_multiline(ax, fontsize=25)

    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

if __name__ == "__main__":
    make_bouquet("qixi_bouquet_text_v3.png")

# -*- coding: utf-8 -*-
"""
均值基线 + 极端偏离高亮（C, D, L）
- 横轴: t = t0..T（可下采样 step）
- 纵轴: f_C, f_D, f_L
- 三条虚线: 从 tstart(含) 到 T 的均值 (C/D/L 各一条)
- 高亮实线: |f - mean| 处于前 q_abs 分位（比如 q_abs=0.90=前10%）的那些时间点上画实线，其他点不画
- 额外: 用 ★ 标注每条曲线的“绝对偏离最大”的那个峰值
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 颜色
COLOR_C = '#1f77b4'
COLOR_D = '#ff7f0e'
COLOR_L = '#2ca02c'

parser = argparse.ArgumentParser()
parser.add_argument("--loner_csv", type=str,
    default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv")
parser.add_argument("--outdir", type=str,
    default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191220扫描b")
parser.add_argument("--tstart", type=int, default=0, help="均值基线的起始索引 t_start（0 表示从 t0 开始）")
parser.add_argument("--step", type=int, default=10, help="下采样步长（>=1）")
parser.add_argument("--q_abs", type=float, default=0.90, help="绝对偏离分位阈值（0.90=前10%）")
parser.add_argument("--dpi", type=int, default=450)
args = parser.parse_args()

outdir = Path(args.outdir).expanduser().resolve() / "event_study"
outdir.mkdir(parents=True, exist_ok=True)

# 读数据
df = pd.read_csv(args.loner_csv)
df.columns = [c.strip() for c in df.columns]
need = {"frac_C", "frac_D", "frac_L"}
if not need <= set(df.columns):
    raise KeyError(f"缺少必要列：{need}")

# 下采样
if args.step > 1:
    df = df.iloc[::args.step, :].reset_index(drop=True)

C = df["frac_C"].to_numpy(float)
D = df["frac_D"].to_numpy(float)
L = df["frac_L"].to_numpy(float)
T = len(C)
t = np.arange(T)

# 基线区间 [tstart, T)
t0 = max(0, int(args.tstart))
if t0 >= T - 1:
    raise ValueError(f"tstart={t0} 太靠后，序列长度为 {T}")

# 计算均值（基线）
mu_C = np.nanmean(C[t0:])
mu_D = np.nanmean(D[t0:])
mu_L = np.nanmean(L[t0:])

# 计算“相对均值偏离”
dev_C = C - mu_C
dev_D = D - mu_D
dev_L = L - mu_L

# 阈值（极端偏离前 q_abs 分位，基于绝对偏离）
thr_C = np.nanquantile(np.abs(dev_C[t0:]), args.q_abs)
thr_D = np.nanquantile(np.abs(dev_D[t0:]), args.q_abs)
thr_L = np.nanquantile(np.abs(dev_L[t0:]), args.q_abs)

# 构造高亮曲线：非极端点填 NaN（这样只会连极端区间的线段）
C_hl = C.copy(); C_hl[~(np.abs(dev_C) >= thr_C)] = np.nan
D_hl = D.copy(); D_hl[~(np.abs(dev_D) >= thr_D)] = np.nan
L_hl = L.copy(); L_hl[~(np.abs(dev_L) >= thr_L)] = np.nan

# 找每条曲线的“绝对偏离最大”峰值点（基于 t0..T-1 区间）
peak_C = t0 + int(np.nanargmax(np.abs(dev_C[t0:])))
peak_D = t0 + int(np.nanargmax(np.abs(dev_D[t0:])))
peak_L = t0 + int(np.nanargmax(np.abs(dev_L[t0:])))

# 画图
fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)

# 均值虚线（水平）
ax.hlines(mu_C, t0, T-1, colors=COLOR_C, linestyles='--', lw=1.6, label=r"$\overline{f_C}$")
ax.hlines(mu_D, t0, T-1, colors=COLOR_D, linestyles='--', lw=1.6, label=r"$\overline{f_D}$")
ax.hlines(mu_L, t0, T-1, colors=COLOR_L, linestyles='--', lw=1.6, label=r"$\overline{f_L}$")

# 高亮实线（只在极端偏离的点/区间上画）
# 为避免 NaN 引发连线错误，Matplotlib 会自动只连连续的非 NaN 段
ax.plot(t, C_hl, color=COLOR_C, lw=2.0, label=r"$f_C$ (top |dev|)")
ax.plot(t, D_hl, color=COLOR_D, lw=2.0, label=r"$f_D$ (top |dev|)")
ax.plot(t, L_hl, color=COLOR_L, lw=2.0, label=r"$f_L$ (top |dev|)")

# 峰值标记（★）
ax.scatter([peak_C], [C[peak_C]], s=55, color=COLOR_C, marker='*', zorder=5)
ax.scatter([peak_D], [D[peak_D]], s=55, color=COLOR_D, marker='*', zorder=5)
ax.scatter([peak_L], [L[peak_L]], s=55, color=COLOR_L, marker='*', zorder=5)

# 视觉元素
if args.step <= 1:
    ax.set_xlabel("t (steps)")
else:
    ax.set_xlabel(f"t (×{args.step} original steps)")
ax.set_ylabel("fraction (f_C, f_D, f_L)")
ax.grid(True, ls='--', alpha=0.25)
ax.set_xlim(0, T-1)

# 图例放右上角；若遮挡可改 loc 或 bbox_to_anchor
ax.legend(fontsize=8, frameon=True, loc="upper right", ncol=2)

# 标注信息
ax.axvline(t0, color='k', lw=1, alpha=0.5)
ax.text(t0, ax.get_ylim()[1], f" baseline from t={t0}", va='top', ha='left', fontsize=8)

ax.set_title(f"Mean baselines (dashed) & top-|deviation| segments (solid)\n"
             f"q_abs={args.q_abs:.2f}, step={args.step}, tstart={t0}")

# 保存
png = outdir / f"mean_highlight_step{args.step}_q{int(args.q_abs*100)}_tstart{t0}.png"
svg = outdir / f"mean_highlight_step{args.step}_q{int(args.q_abs*100)}_tstart{t0}.svg"
fig.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)  # SVG 不需要 dpi

plt.show()

# 控制台摘要
print(f"长度 T={T}, 下采样 step={args.step}, 基线区间=[{t0}, {T-1}]")
print(f"mu_C={mu_C:.4f}, mu_D={mu_D:.4f}, mu_L={mu_L:.4f}")
print(f"阈值: thr_C={thr_C:.4g}, thr_D={thr_D:.4g}, thr_L={thr_L:.4g}")
print(f"峰值: tC*={peak_C} |dev|={abs(dev_C[peak_C]):.4g}; "
      f"tD*={peak_D} |dev|={abs(dev_D[peak_D]):.4g}; "
      f"tL*={peak_L} |dev|={abs(dev_L[peak_L]):.4g}")
print(f"图已保存到: {outdir}")

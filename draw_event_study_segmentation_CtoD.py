# -*- coding: utf-8 -*-
"""
Event-study: C shock → future ΔD（线性版，不用函数封装）
按 ΔC 的分位段定义事件，绘制 E[Δf_D(t+h) | ΔC ∈ bin]。
主图：多分段平滑曲线；右上角 inset：各分段的平均 ΔC。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import make_interp_spline

# ========= 统一字体 =========
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========= 参数 =========
parser = argparse.ArgumentParser()
parser.add_argument("--loner_csv", type=str,
    default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv")
parser.add_argument("--outdir", type=str,
    default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191220扫描b")
parser.add_argument("--h", type=int, default=30)
parser.add_argument("--step2", type=int, default=10)
args = parser.parse_args()

outroot = Path(args.outdir).expanduser().resolve() / "event_study"
outroot.mkdir(parents=True, exist_ok=True)

# ========= 读数据（需要 C/L/D 三列）=========
df = pd.read_csv(args.loner_csv)
df.columns = [c.strip() for c in df.columns]
if not {"frac_C", "frac_L", "frac_D"} <= set(df.columns):
    raise KeyError("缺少必要列：frac_C, frac_L, frac_D")

# ========= 只做 step=step2（例如 10）=========
for step in [args.step2]:
    tag = f"step{step}"
    dfl = df.iloc[::step, :].reset_index(drop=True) if step > 1 else df.reset_index(drop=True)

    C = dfl["frac_C"].to_numpy(float)
    D = dfl["frac_D"].to_numpy(float)

    # 一阶差分
    dC = np.diff(C, prepend=C[0])   # 事件变量
    dD = np.diff(D, prepend=D[0])   # 被解释变量

    # ===== 分段（按 ΔC 的分位）=====
    QUANTILE_BINS = [
        (0.00, 0.05),
        (0.05, 0.15),
        (0.15, 0.25),
        (0.75, 0.85),
        (0.85, 0.95),
        (0.95, 1.00),
    ]

    H = args.h
    x = np.arange(H + 1)

    bin_labels = []
    curves = []           # (label, xs, ys, color)
    mean_dC_bins = []     # 各分段平均 ΔC（用于 inset）
    n_events_bins = []    # 各分段事件数
    n_points_bins = []    # 各分段曲线可用点数

    cmap = plt.cm.viridis
    colors = [cmap(t) for t in np.linspace(0.15, 0.90, len(QUANTILE_BINS))]

    for (qlo, qhi), col in zip(QUANTILE_BINS, colors):
        # 1) 事件索引（基于 ΔC 分位）
        vlo, vhi = np.quantile(dC, [qlo, qhi])
        idx = np.where((dC >= vlo) & (dC < vhi))[0] if qhi < 1.0 else np.where((dC >= vlo) & (dC <= vhi))[0]

        # 2) 分段标签与平均 ΔC
        bin_text = f"[{int(qlo*100)}%, {int(qhi*100)}%]"
        bin_labels.append(bin_text)
        mean_dC = float(dC[idx].mean()) if len(idx) else np.nan
        mean_dC_bins.append(mean_dC)
        n_events_bins.append(int(len(idx)))

        # 3) 条件期望：E[ ΔD_{t+h} | ΔC ∈ 分段 ]
        m_bin = []
        for h in range(H + 1):
            valid = idx[(idx + h) < len(dD)]
            m_bin.append(dD[valid + h].mean() if len(valid) else np.nan)
        m_bin = np.array(m_bin, dtype=float)

        # 4) 平滑（样条；点少回退移动平均或原样）
        legend_lbl = rf"{bin_text}  ($\overline{{\Delta f_C}}={mean_dC:.3g}$)"
        mask = np.isfinite(m_bin)
        x_valid, y_valid = x[mask], m_bin[mask]
        n_points_bins.append(int(len(x_valid)))

        if len(x_valid) >= 4:
            xs = np.linspace(x_valid.min(), x_valid.max(), 200)
            try:
                ys = make_interp_spline(x_valid, y_valid, k=3)(xs)
            except Exception:
                k = 3
                ys = np.convolve(y_valid, np.ones(k)/k, mode='same') if len(y_valid) >= k else y_valid
                xs = x_valid
        else:
            xs, ys = x_valid, y_valid

        curves.append((legend_lbl, xs, ys, col))

    # ===== 画图：主图 + 右上角 inset（平均 ΔC）=====
    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)

    for (lbl, xs, ys, col) in curves:
        ax.plot(xs, ys, lw=1.8, color=col, label=lbl)
        # 可选：稀疏散点显示采样
        # if len(xs) > 0:
        #     ax.scatter(xs[::max(1, len(xs)//10)], ys[::max(1, len(xs)//10)], s=8, color=col, alpha=0.45)

    if step == 1:
        ax.set_xlabel("Horizon h (time steps)")
    else:
        ax.set_xlabel(f"Horizon h (×{step} original steps)")
    ax.set_ylabel(r"$E[\Delta f_D(t+h)]$")
    ax.axhline(0, color='gray', ls='--', alpha=0.5, lw=1)
    ax.grid(True, ls='--', alpha=0.25)
    ax.set_title(f"Event-study by ΔC quantile bins ({tag})")
    ax.legend(fontsize=8, frameon=True, loc="lower right")

    # inset：各分段平均 ΔC
    ax_in = inset_axes(ax, width="34%", height="38%", loc="upper right", borderpad=0.8)
    ax_in.set_zorder(5); ax.set_zorder(1)
    ax_in.set_facecolor("white"); ax_in.patch.set_alpha(1.0)

    y_pos = np.arange(len(bin_labels))
    for i, (mu, col) in enumerate(zip(mean_dC_bins, colors)):
        ax_in.barh(y_pos[i], mu if np.isfinite(mu) else 0.0, color=col, alpha=0.95, height=0.8, zorder=6)

    ax_in.set_yticks(y_pos)
    ax_in.set_yticklabels(bin_labels, fontsize=8)
    ax_in.set_xlabel(r"$\overline{\Delta f_C}$", fontsize=8)
    ax_in.tick_params(axis='x', labelsize=8)
    ax_in.grid(False)
    ax_in.axvline(0, color='gray', lw=1, alpha=0.6, zorder=4)

    # 保存
    png_path = outroot / f"event_study_CtoD_quantilebins_{tag}.png"
    svg_path = outroot / f"event_study_CtoD_quantilebins_{tag}.svg"
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg_path, dpi=450, bbox_inches="tight", pad_inches=0.02)

    # 摘要：每个分段的事件数与可用点数
    summary = {lab: {"n_events": ne, "n_points": npnts, "mean_dC": float(mu) if np.isfinite(mu) else np.nan}
               for lab, ne, npnts, mu in zip(bin_labels, n_events_bins, n_points_bins, mean_dC_bins)}
    print(f"[{tag}] 分段摘要：")
    for k, v in summary.items():
        print(f"  {k}: n_events={v['n_events']}, n_points={v['n_points']}, meanΔC={v['mean_dC']:.3g}")

plt.show()
print(f"Done. Figures saved to: {outroot}")

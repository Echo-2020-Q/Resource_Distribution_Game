# -*- coding: utf-8 -*-
"""
Event-study for Loner → future ΔC（线性版，不用函数封装）

生成四张图：
  - event_study_step1.png
  - event_study_diff_step1.png
  - event_study_step10.png
  - event_study_diff_step10.png
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

# ========= 配色 =========
COLOR_POS = '#264385'  # ΔL top
COLOR_NEG = '#e23e35' # ΔL bottom
COLOR_DIFF = '#2ca02c' # diff

# ========= 参数 =========
parser = argparse.ArgumentParser()
parser.add_argument("--loner_csv", type=str,
    default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv")
parser.add_argument("--outdir", type=str, default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191220扫描b")
parser.add_argument("--h", type=int, default=30)
parser.add_argument("--q_low", type=float, default=0.1)
parser.add_argument("--q_high", type=float, default=0.9)
parser.add_argument("--step2", type=int, default=10)
args = parser.parse_args()

outroot = Path(args.outdir).expanduser().resolve() / "event_study"
outroot.mkdir(parents=True, exist_ok=True)

# ========= 读数据 =========
df = pd.read_csv(args.loner_csv)
df.columns = [c.strip() for c in df.columns]
if not {"frac_C", "frac_L"} <= set(df.columns):
    raise KeyError("缺少必要列：frac_C, frac_L")

# ========= 处理两种 step =========
for step in [ args.step2]:
    tag = f"step{step}"
    if step <= 1:
        dfl = df.reset_index(drop=True)
    else:
        dfl = df.iloc[::step, :].reset_index(drop=True)

    C = dfl["frac_C"].to_numpy(float)
    L = dfl["frac_L"].to_numpy(float)

    dC = np.diff(C, prepend=C[0])
    dL = np.diff(L, prepend=L[0])
    # ===== 多分段事件研究：例如 [0,5%], (5%,15%], (15%,25%] =====
    # 你可以按需修改分段（分位区间），注意取值在 [0,1] 之间
    QUANTILE_BINS = [
        (0.00, 0.05),
        (0.05, 0.15),
        (0.15, 0.25),
        (0.75, 0.85),
        (0.85, 0.95),
        (0.95, 1.00),
        # 需要更多段，继续在这里加：(0.25, 0.50), (0.50, 0.75), ...
    ]

    # 计算每个分段的 ΔC 曲线 E[ΔC | ΔL ∈ 分段]，以及该分段的平均 ΔL
    H = args.h
    x = np.arange(H + 1)
    bin_labels = []  # 分段文本标签
    curves = []  # list of (label, y_values)
    mean_dL_bins = []  # 每个分段的平均 ΔL


    # 为分段曲线准备一组颜色（按分段顺序渐变）
    cmap = plt.cm.viridis
    colors = [cmap(t) for t in np.linspace(0.15, 0.90, len(QUANTILE_BINS))]

    for (qlo, qhi), col in zip(QUANTILE_BINS, colors):
        # 分位点阈值
        vlo, vhi = np.quantile(dL, [qlo, qhi])
        # 落在该分段内的索引（右端开区间，最后一段可改为闭区间）
        if qhi < 1.0:
            idx = np.where((dL >= vlo) & (dL < vhi))[0]
        else:
            idx = np.where((dL >= vlo) & (dL <= vhi))[0]

        # 计算该分段的 ΔL 平均大小（便于在图中展示）
        mean_dL = float(dL[idx].mean()) if len(idx) else np.nan
        mean_dL_bins.append(mean_dL)
        bin_text = f"[{int(qlo * 100)}%, {int(qhi * 100)}%]"
        bin_labels.append(bin_text)

        # 计算 E[ΔC_{t+h} | ΔL ∈ 分段]
        m_bin = []
        for h in range(H + 1):
            valid = idx[(idx + h) < len(dC)]
            m_bin.append(dC[valid + h].mean() if len(valid) else np.nan)
        m_bin = np.array(m_bin, dtype=float)

        legend_lbl = rf"{bin_text}  ($\overline{{\Delta f_L}}={mean_dL:.3g}$)"

        mask = np.isfinite(m_bin)
        x_valid = x[mask]
        y_valid = m_bin[mask]

        if len(x_valid) >= 4:  # 三次样条至少4个点更稳
            x_smooth = np.linspace(x_valid.min(), x_valid.max(), 200)
            try:
                spl = make_interp_spline(x_valid, y_valid, k=3)
                y_smooth = spl(x_smooth)
            except Exception:
                # 如果样条失败，退化为移动平均
                kernel = 3
                if len(y_valid) >= kernel:
                    y_ma = np.convolve(y_valid, np.ones(kernel) / kernel, mode='same')
                    x_smooth, y_smooth = x_valid, y_ma
                else:
                    x_smooth, y_smooth = x_valid, y_valid
        else:
            # 点太少，直接用原曲线
            x_smooth, y_smooth = x_valid, y_valid

        # 存“平滑后的 x,y”
        curves.append((legend_lbl, x_smooth, y_smooth, col))

    # ====== 单张图展示：主轴是多条分段曲线 + 右上角嵌入柱状图显示均值ΔL ======
    # fig = plt.figure(figsize=(6.8, 4.6))
    # ax = fig.add_axes([0.10, 0.12, 0.65, 0.78])  # 主轴
    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    for (lbl, xs, ys, col) in curves:
        ax.plot(xs, ys, lw=1.8, color=col, label=lbl)
        # （可选）在原始点位上加淡色小标记，帮助审稿人看到真实采样
        ax.scatter(xs[::20], ys[::20], s=5, color=col, alpha=0.5)
    # 坐标轴与标题
    if step == 1:
        ax.set_xlabel("Horizon h (time steps)")
    else:
        ax.set_xlabel(f"h(time steps)")
    ax.set_ylabel(r"$\Delta \bar{f_C}(t+h)$")
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.grid(True, ls='--', alpha=0.25)
    ax.set_title("")
    ax.legend(fontsize=8, frameon=True, loc="lower right")

    # ====== 右上角嵌入柱状图（显示每段的平均ΔL）——嵌入主图右上角，层级更高 ======
    ax_in = inset_axes(
        ax,
        width="34%",  # 小图宽度占主轴的百分比（可调：30%~40%）
        height="38%",  # 小图高度占主轴的百分比（可调）
        loc="upper right",  # 放在主图右上角
        borderpad=0.8  # 与边界的留白（单位：字体大小）
    )
    ax_in.set_zorder(5)  # 小图层级更高
    ax.set_zorder(1)  # 主图层级较低
    ax_in.set_facecolor("white")  # 背景白色
    ax_in.patch.set_alpha(0.95)  # 略透明，盖在线上也清晰

    y_pos = np.arange(len(bin_labels))
    for i, (mu, col) in enumerate(zip(mean_dL_bins, colors)):
        ax_in.barh(
            y_pos[i],
            mu if np.isfinite(mu) else 0.0,
            color=col,
            alpha=0.95,
            height=0.8,
            zorder=6  # 柱子再抬高一层
        )

    ax_in.set_yticks(y_pos)
    ax_in.set_yticklabels(bin_labels, fontsize=8)
    ax_in.set_xlabel(r"$\overline{\Delta f_L}$", fontsize=8)
    ax_in.tick_params(axis='x', labelsize=8)
    ax_in.grid(False)
    ax_in.axvline(0, color='gray', lw=1, alpha=0.6, zorder=4)

    # 保存为单张图（SVG + PNG）
    fig.tight_layout()
    fig.savefig(outroot / f"event_study_quantilebins_{tag}.png", dpi=450)
    fig.savefig(outroot / f"event_study_quantilebins_{tag}.svg", dpi=450)
    plt.show()

    # 控制台摘要
    print(f"[{tag}] 分段样本数：", {lab: len(xs) for (lab, xs, ys, col) in curves})







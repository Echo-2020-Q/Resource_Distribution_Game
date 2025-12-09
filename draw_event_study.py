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
for step in [1, args.step2]:
    tag = f"step{step}"
    if step <= 1:
        dfl = df.reset_index(drop=True)
    else:
        dfl = df.iloc[::step, :].reset_index(drop=True)

    C = dfl["frac_C"].to_numpy(float)
    L = dfl["frac_L"].to_numpy(float)

    dC = np.diff(C, prepend=C[0])
    dL = np.diff(L, prepend=L[0])

    # 事件划分
    ql, qh = np.quantile(dL, [args.q_low, args.q_high])
    pos_idx = np.where(dL >= qh)[0]
    neg_idx = np.where(dL <= ql)[0]

    H = args.h
    m_pos, m_neg = [], []
    for h in range(H+1):
        vpos = pos_idx[(pos_idx + h) < len(dC)]
        vneg = neg_idx[(neg_idx + h) < len(dC)]
        m_pos.append(dC[vpos + h].mean() if len(vpos) else np.nan)
        m_neg.append(dC[vneg + h].mean() if len(vneg) else np.nan)
    m_pos, m_neg = np.array(m_pos), np.array(m_neg)
    m_diff = m_pos - m_neg

    # ===== 图1: 两类事件曲线 =====
    plt.figure()
    x = np.arange(H+1)
    plt.plot(x, m_pos, label=r"$\Delta L_{Top}$", color=COLOR_POS, lw=1.5)
    plt.plot(x, m_neg, label=r"$\Delta L_{Bottom}$", color=COLOR_NEG, lw=1.5)
    plt.axhline(0, color='gray', ls='--', alpha=0.5)
    plt.xlabel("h(time steps)")
    plt.ylabel("$\Delta f_C(t+h)$")
    plt.title(f"")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outroot / f"event_study_{tag}.png", dpi=450)
    plt.savefig(outroot / f"event_study_{tag}.svg", dpi=450)


    # ===== 图2: 差值曲线 =====
    plt.figure()
    plt.plot(x, m_diff, color=COLOR_DIFF, lw=2)
    plt.axhline(0, color='gray', ls='--', alpha=0.5)
    if np.any(np.isfinite(m_diff)):
        h_star = int(np.nanargmax(m_diff))
        plt.scatter([h_star], [m_diff[h_star]], s=32, color=COLOR_DIFF)
        plt.annotate(f"max @ h={h_star}\n{m_diff[h_star]:.3g}",
                     xy=(h_star, m_diff[h_star]), xytext=(10, 10),
                     textcoords="offset points", ha="left", fontsize=8)
    plt.xlabel("horizon h (raw steps)")
    plt.ylabel("difference in mean ΔC")
    plt.title(f"Event-study diff (Loner, {tag}): top - bottom")
    plt.tight_layout()
    plt.savefig(outroot / f"event_study_diff_{tag}.png", dpi=150)


    print(f"[{tag}] pos={len(pos_idx)}, neg={len(neg_idx)}, best_h={np.nanargmax(m_diff)}, "
          f"max_diff={np.nanmax(m_diff):.4f}")

plt.show()
print(f"完成！图已保存到 {outroot}")


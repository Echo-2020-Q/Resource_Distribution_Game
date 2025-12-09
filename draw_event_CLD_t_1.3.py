# -*- coding: utf-8 -*-
"""
plot_event_study_cdl_nofunc.py
事件研究（锚点可选：D_up / D_down / C_up / C_down / L_up / L_down） + 95% CI + 降采样/平滑（无函数版）

依赖：numpy, pandas, matplotlib
示例：
  python plot_event_study_cdl_nofunc.py --csv "capture_window_Loner.csv" --outdir "./out_event_study" \
      --ds 20 --q 0.75 --pre 10 --post 12 --gap 8 --win 11 --panel --baseline window --anchor_mode D_up
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as mticker
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
matplotlib.rcParams['xtick.labelsize'] = 24  # x轴刻度标签字体大小
matplotlib.rcParams['ytick.labelsize'] = 24  # y轴刻度标签字体大小
matplotlib.rcParams['legend.fontsize'] = 26  # 图例字体大小
matplotlib.rcParams['figure.titlesize'] = 24  # 图形标题字体大小
matplotlib.rcParams['font.weight'] = 'bold'  # 全局字体粗细
matplotlib.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签字体粗细

# ---------- 工具：对 fill_between 做健壮性封装 ----------
def _safe_arrays(x, lo, hi):
    x  = np.asarray(x,  dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if not (x.size == lo.size == hi.size):
        raise ValueError(f"shape mismatch: x={x.shape}, lo={lo.shape}, hi={hi.shape}")
    mask = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    return x, lo, hi, mask

def safe_fill_between(ax, x, lo, hi, **kw):
    x, lo, hi, mask = _safe_arrays(x, lo, hi)
    ax.fill_between(x, lo, hi, where=mask, interpolate=True, **kw)

# ========== 参数 ==========
ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=str, default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_"
                                           r"NoLoner.csv", help="输入CSV，如 capture_window_NoLoner.csv")
ap.add_argument("--outdir", type=str, default=r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\out_event_study_Loner", help="输出目录")
ap.add_argument("--ds", type=int, default=20, help="降采样因子（原始步→显示步）")
ap.add_argument("--q", type=float, default=0.8, help="差分分位数阈值（越小样本越多），如 0.75")
ap.add_argument("--pre", type=int, default=20, help="锚点左窗口长度（显示步）")
ap.add_argument("--post", type=int, default=20, help="锚点右窗口长度（显示步）")
ap.add_argument("--gap", type=int, default=8, help="锚点最小间隔（显示步）")
ap.add_argument("--win", type=int, default=7, help="中值平滑窗口（奇数）")
ap.add_argument("--panel", action="store_true", help="输出三行面板（默认关闭：输出 C/D/L 三张单图）")
ap.add_argument("--seed", type=int, default=1234, help="bootstrap 随机种子")
ap.add_argument("--baseline", type=str, default="window",
                choices=["t0", "window", "none"],
                help="水平虚线基线：t0=τ=0处数值；window=窗口平均；none=不画")
ap.add_argument("--anchor_mode", type=str, default="D_up",
                choices=["D_up", "D_down", "C_up", "C_down", "L_up", "L_down"],
                help="锚点模式：变量_方向（上升 up / 下降 down）")

args = ap.parse_args()

CSV = Path(args.csv)
OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
DS, Q, PRE, POST, GAP, WIN, PANEL, SEED = args.ds, args.q, args.pre, args.post, args.gap, args.win, args.panel, args.seed

# ========== 读取并鲁棒识别列 ==========
df0 = pd.read_csv(CSV)
cols = {c.lower(): c for c in df0.columns}
def _pick(*names):
    for n in names:
        if n in cols: return cols[n]
    return None

t_col = _pick("t","time","step","iter","episode") or df0.columns[0]
c_col = _pick("frac_c","fc","c","c_frac","f_c")
d_col = _pick("frac_d","fd","d","d_frac","f_d")
l_col = _pick("frac_l","fl","l","l_frac","f_l")
if any(x is None for x in [c_col, d_col, l_col]):
    raise ValueError("无法从表头自动识别 C/D/L 列名，请检查列名或手动修改脚本。")

df = df0[[t_col, c_col, d_col, l_col]].copy()
df.columns = ["t","C","D","L"]
for k in ["t","C","D","L"]:
    df[k] = pd.to_numeric(df[k], errors="coerce")
df = df.dropna().reset_index(drop=True)

# 归一化（若存在微漂移）
s = df[["C","D","L"]].sum(axis=1)
if (np.abs(s-1.0) > 1e-6).any():
    df[["C","D","L"]] = df[["C","D","L"]].div(s, axis=0)

# ========== 降采样 + 中值平滑 ==========
df_ds = df.iloc[::DS, :].copy()
df_ds["C_s"] = df_ds["C"].rolling(WIN, center=True).median()
df_ds["D_s"] = df_ds["D"].rolling(WIN, center=True).median()
df_ds["L_s"] = df_ds["L"].rolling(WIN, center=True).median()
df_ds = df_ds.dropna().reset_index(drop=True)

# 一阶差分（上升/下降强度）
df_ds["dC"] = df_ds["C_s"].diff()
df_ds["dD"] = df_ds["D_s"].diff()
df_ds["dL"] = df_ds["L_s"].diff()

# ========== 锚点选择（支持 up/down 与 C/D/L） ==========
# 解析 anchor_mode
anchor_var, anchor_dir = args.anchor_mode.split("_")  # 例如 "D_up" -> ("D", "up")
diff_col = f"d{anchor_var}"                           # dC / dD / dL

if anchor_dir == "up":
    thr = df_ds[diff_col].quantile(Q)                 # 上分位阈值：较大上升
    pick_mask = df_ds[diff_col] >= thr
    anchor_desc = f"{anchor_var} rising"
elif anchor_dir == "down":
    thr = df_ds[diff_col].quantile(1 - Q)             # 下分位阈值：较大下降
    pick_mask = df_ds[diff_col] <= thr
    anchor_desc = f"{anchor_var} falling"
else:
    raise ValueError("anchor_mode must end with _up or _down")

cand = df_ds.index[pick_mask].tolist()

# 最小间隔去重
anchors, last = [], -10**9
for i in cand:
    if i - last >= GAP:
        anchors.append(i); last = i

# ========== 对齐并堆叠 ==========
lags = np.arange(-PRE, POST+1)
def _collect(var):
    mats = []
    for a in anchors:
        lo, hi = a-PRE, a+POST
        if lo < 0 or hi >= len(df_ds):
            continue
        mats.append(df_ds.loc[lo:hi, var].to_numpy())
    return (np.vstack(mats) if mats else None)

MC = _collect("C_s")
MD = _collect("D_s")
ML = _collect("L_s")

if MC is None:
    raise RuntimeError("没有足够的锚点可用于事件研究，请调小 q、缩短窗口或减小 gap。")

N_USED = MC.shape[0]

# ========== bootstrap 均值与 95% CI ==========
if SEED is not None:
    rng = np.random.default_rng(SEED)
else:
    rng = np.random.default_rng()

def _mean_ci(M, alpha=0.05):
    mean = M.mean(axis=0)
    n = M.shape[0]
    B = min(2000, max(200, 20*n))
    boots = np.empty((B, M.shape[1]), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = M[idx].mean(axis=0)
    lo = np.percentile(boots, 100*alpha/2, axis=0)
    hi = np.percentile(boots, 100*(1-alpha/2), axis=0)
    return mean, lo, hi

C_mean, C_lo, C_hi = _mean_ci(MC)
D_mean, D_lo, D_hi = _mean_ci(MD)
L_mean, L_lo, L_hi = _mean_ci(ML)

# --- 计算基线 ---
if args.baseline == "t0":
    base_C, base_D, base_L = C_mean[PRE], D_mean[PRE], L_mean[PRE]
elif args.baseline == "window":
    base_C = float(np.nanmean(C_mean))
    base_D = float(np.nanmean(D_mean))
    base_L = float(np.nanmean(L_mean))
elif args.baseline == "none":
    base_C = base_D = base_L = None
else:
    raise ValueError("baseline must be one of: t0, window, none")

# ========== 作图 ==========
style_map = {
    "C": dict(color="tab:blue",  linestyle="-", linewidth=2),
    "D": dict(color="tab:red",   linestyle="-", linewidth=2),
    "L": dict(color="#dbb428",   linestyle="-", linewidth=2),
}
fill_alpha_panel = 0.18
fill_alpha_single = 0.20

# 标题/文件名标签
mode_tag = args.anchor_mode
tag = f"{mode_tag}_q{int(Q*100):02d}_DS{DS}_pre{PRE}_post{POST}"

# ========== 面板图 ==========
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
panel_data = [
    (axes[0], "C", C_mean, C_lo, C_hi, base_C),
    (axes[1], "D", D_mean, D_lo, D_hi, base_D),
    (axes[2], "L", L_mean, L_lo, L_hi, base_L),
]
for ax, name, mean, lo, hi, base in panel_data:
    x, lo1, hi1, mask = _safe_arrays(lags, lo, hi)
    style = style_map.get(name, {})
    safe_fill_between(ax, x, lo1, hi1, color=style.get("color", None),
                      alpha=fill_alpha_panel, linewidth=0)
    ax.plot(x[mask], np.asarray(mean).reshape(-1)[mask],
            label=f"{name}(without Loner)", **style)
    ax.axvline(0, ls="--", lw=1)
    if base is not None:
        ax.axhline(base, ls=":", lw=1, color=style.get("color", "gray"), alpha=0.55)
    # ax.set_ylabel("The frequency of strategies")
    if name == "D":
        ax.legend(loc="lower left")
    else:
        ax.legend(loc="upper right")
    ax.grid(False)

axes[1].set_ylabel("The frequency of strategies", labelpad=20)
axes[1].yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))
axes[-1].set_xlabel("Time steps")
#axes[0].set_title(f"Event study aligned on '{anchor_desc}' (3-panel with 95% CI)")
axes[0].set_title("")
plt.tight_layout()
out_png = OUTDIR / f"event_study_panel_{tag}.png"
out_svg = OUTDIR / f"event_study_panel_{tag}.svg"
out_pdf = OUTDIR / f"event_study_panel_{tag}.pdf"
plt.savefig(out_png, dpi=450)
plt.savefig(out_svg)
plt.savefig(out_pdf)
print(f"[panel] n={N_USED} -> {out_png}/{out_svg}/{out_pdf}")

# ========== 三张单图 ==========
figs = []
baseline_map = {"C": base_C, "D": base_D, "L": base_L}
for name, mean, lo, hi in [("D", D_mean, D_lo, D_hi),
                           ("C", C_mean, C_lo, C_hi),
                           ("L", L_mean, L_lo, L_hi)]:
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    x, lo1, hi1, mask = _safe_arrays(lags, lo, hi)
    style = style_map.get(name, {})
    safe_fill_between(ax, x, lo1, hi1,
                      color=style.get("color", None),
                      alpha=fill_alpha_single, linewidth=0)
    ax.plot(x[mask], np.asarray(mean).reshape(-1)[mask],
            label=f"{name}(without Loner)", **style)
    ax.axvline(0, ls="--", lw=1)
    base = baseline_map[name]
    if base is not None:
        ax.axhline(base, ls=":", lw=1, color=style.get("color", "gray"), alpha=0.55)
    ax.grid(False); ax.legend(loc="best")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("The frequency of strategies")
    ax.set_title("")
    plt.tight_layout()
    out_png = OUTDIR / f"event_study_single_{tag}_{name}.png"
    out_svg = OUTDIR / f"event_study_single_{tag}_{name}.svg"
    fig.savefig(out_png, dpi=450)
    fig.savefig(out_svg)
    figs.append(fig)
    print(f"[single-{name}] n={N_USED} -> {out_png}")

plt.show()

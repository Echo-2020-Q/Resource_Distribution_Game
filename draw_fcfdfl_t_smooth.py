# -*- coding: utf-8 -*-
"""
plot_fc_fd_fl_logt_fixed_linear_smooth.py
顺序执行版：读取两份 CSV，log 轴绘制 fc/fd/fl。
为减小抖动：按对数时间做分箱，并对每箱取“中位数(默认)/均值”平滑。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ========= 路径 =========
CSV_LONER   = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv"
CSV_NOLONER = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_NoLoner.csv"
OUTDIR      = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191220扫描b"

# ========= 平滑参数（可调）=========
SMOOTH_BINS = 300   # 分箱数：越小越平滑（比如 150/200/300/400）
USE_MEDIAN  = False  # True=箱内用中位数，False=用均值
SHOW_RAW    = True # 叠加原始曲线（半透明），便于对比

# ========= 字体 =========
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ====== 读取 CSV：Loner ======
if not os.path.exists(CSV_LONER):
    raise FileNotFoundError(f"CSV 不存在：{CSV_LONER}")
dfL = pd.read_csv(CSV_LONER)
colsL = list(dfL.columns)

# 自动识别列名（Loner）
cands_t  = ["t","time","step","iter","epoch","round"]
cands_fc = ["fc","frac_C","frac_c","C","c"]
cands_fd = ["fd","frac_D","frac_d","D","d"]
cands_fl = ["fl","frac_L","frac_l","L","l"]

t_colL = next((c for c in cands_t  if c in colsL), None) or {c.lower(): c for c in colsL}.get("t","time")
fc_colL = next((c for c in cands_fc if c in colsL), None) or {c.lower(): c for c in colsL}.get("fc","frac_c")
fd_colL = next((c for c in cands_fd if c in colsL), None) or {c.lower(): c for c in colsL}.get("fd","frac_d")
fl_colL = next((c for c in cands_fl if c in colsL), None) or {c.lower(): c for c in colsL}.get("fl","frac_l")
if None in [t_colL, fc_colL, fd_colL, fl_colL]:
    raise KeyError(f"[Loner] 列名不完整，实际列：{colsL}")

tL  = pd.to_numeric(dfL[t_colL],  errors="coerce")
fcL = pd.to_numeric(dfL[fc_colL], errors="coerce")
fdL = pd.to_numeric(dfL[fd_colL], errors="coerce")
flL = pd.to_numeric(dfL[fl_colL], errors="coerce")
maskL = ~(tL.isna() | fcL.isna() | fdL.isna() | flL.isna())
tL, fcL, fdL, flL = tL[maskL].values, fcL[maskL].values, fdL[maskL].values, flL[maskL].values

# ====== 读取 CSV：NoLoner ======
if not os.path.exists(CSV_NOLONER):
    raise FileNotFoundError(f"CSV 不存在：{CSV_NOLONER}")
dfN = pd.read_csv(CSV_NOLONER)
colsN = list(dfN.columns)

t_colN = next((c for c in cands_t  if c in colsN), None) or {c.lower(): c for c in colsN}.get("t","time")
fc_colN = next((c for c in cands_fc if c in colsN), None) or {c.lower(): c for c in colsN}.get("fc","frac_c")
fd_colN = next((c for c in cands_fd if c in colsN), None) or {c.lower(): c for c in colsN}.get("fd","frac_d")
fl_colN = next((c for c in cands_fl if c in colsN), None) or {c.lower(): c for c in colsN}.get("fl","frac_l")
if None in [t_colN, fc_colN, fd_colN, fl_colN]:
    raise KeyError(f"[NoLoner] 列名不完整，实际列：{colsN}")

tN  = pd.to_numeric(dfN[t_colN],  errors="coerce")
fcN = pd.to_numeric(dfN[fc_colN], errors="coerce")
fdN = pd.to_numeric(dfN[fd_colN], errors="coerce")
flN = pd.to_numeric(dfN[fl_colN], errors="coerce")
maskN = ~(tN.isna() | fcN.isna() | fdN.isna() | flN.isna())
tN, fcN, fdN, flN = tN[maskN].values, fcN[maskN].values, fdN[maskN].values, flN[maskN].values

# ====== 准备 log-x 与 log 分箱 ======
def prep_log_and_bins(t_arr):
    tmin = float(np.nanmin(t_arr)) if len(t_arr) else 0.0
    shift = 1.0 - tmin if tmin <= 0 else 0.0
    x = t_arr + shift
    # log 分箱（几何等距）
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()+1e-12), SMOOTH_BINS+1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return x, shift, edges, centers

xL, shiftL, edgesL, centersL = prep_log_and_bins(tL)
xN, shiftN, edgesN, centersN = prep_log_and_bins(tN)

# ====== 箱内聚合（中位数/均值）======
def bin_aggregate(x, y, edges, use_median=True):
    out = np.full(len(edges)-1, np.nan, dtype=float)
    for i in range(len(edges)-1):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.any(m):
            out[i] = (np.median(y[m]) if use_median else np.mean(y[m]))
    # 可再做一次很轻的 3 点滚动均值，让线条更平顺
    y2 = pd.Series(out).rolling(3, center=True, min_periods=1).mean().values
    return y2

fcL_sm = bin_aggregate(xL, fcL, edgesL, USE_MEDIAN)
fdL_sm = bin_aggregate(xL, fdL, edgesL, USE_MEDIAN)
flL_sm = bin_aggregate(xL, flL, edgesL, USE_MEDIAN)

fcN_sm = bin_aggregate(xN, fcN, edgesN, USE_MEDIAN)
fdN_sm = bin_aggregate(xN, fdN, edgesN, USE_MEDIAN)
flN_sm = bin_aggregate(xN, flN, edgesN, USE_MEDIAN)

# 有些箱可能为空，去掉 NaN 点
mL = ~np.isnan(fcL_sm) & ~np.isnan(fdL_sm) & ~np.isnan(flL_sm)
mN = ~np.isnan(fcN_sm) & ~np.isnan(fdN_sm) & ~np.isnan(flN_sm)

# ====== Loner：三曲线 ======
plt.figure(figsize=(9,5.4))
if SHOW_RAW:
    plt.plot(xL, fcL, color="#264385",alpha=0.25, lw=1)
    plt.plot(xL, fdL, color="#e23e35",alpha=0.25, lw=1)
    plt.plot(xL, flL, color="#dbb428",alpha=0.25, lw=1)
plt.plot(centersL[mL], fcL_sm[mL], color="#264385",label="C", lw=1.5)
plt.plot(centersL[mL], fdL_sm[mL], color="#e23e35",label="D", lw=1.5)
plt.plot(centersL[mL], flL_sm[mL], color="#dbb428",label="L", lw=1.5)
plt.xscale("log")
xlabL = "t" if shiftL == 0 else f"t (shift {shiftL:.3g}, log scale)"
plt.xlabel(xlabL); plt.ylabel("$f_C$, $f_D$, $f_L$")
plt.title("")
#plt.grid(False, which="both", ls="--", alpha=0.35)
plt.grid(False)
plt.legend(); plt.tight_layout()
out1 = os.path.join(OUTDIR, "Loner_fc_fd_fl_logt_smoothed.png")
out1_svg = os.path.join(OUTDIR, "Loner_fc_fd_fl_logt_smoothed.svg")
plt.savefig(out1, dpi=300)
plt.savefig(out1_svg,dpi=600)
# ====== NoLoner：三曲线 ======
plt.figure(figsize=(9,5.4))
if SHOW_RAW:
    plt.plot(xN, fcN,  color="#264385",alpha=0.25, lw=1)
    plt.plot(xN, fdN, color="#e23e35",alpha=0.25, lw=1)
    plt.plot(xN, flN, color="#eae936",alpha=0.25, lw=1)
plt.plot(centersN[mN], fcN_sm[mN],  color="#264385",label="C", lw=1.5)
plt.plot(centersN[mN], fdN_sm[mN], color="#e23e35",label="D", lw=1.5)
plt.plot(centersN[mN], flN_sm[mN], color="#eae936",label="L", lw=1.5)
plt.xscale("log")
xlabN = "t" if shiftN == 0 else f"t (shift {shiftN:.3g}, log scale)"
plt.xlabel(xlabN); plt.ylabel("$f_C$, $f_D$, $f_L$")
plt.title("")
#plt.grid(False, which="both", ls="--", alpha=0.35)
plt.grid(False)
plt.legend(); plt.tight_layout()
out2 = os.path.join(OUTDIR, "NoLoner_fc_fd_fl_logt_smoothed.png")
out2_svg = os.path.join(OUTDIR, "NoLoner_fc_fd_fl_logt_smoothed.svg")
plt.savefig(out2, dpi=300)
plt.savefig(out2_svg, dpi=600)

# ====== fc 对比（平滑后 + raw） ======
plt.figure(figsize=(9, 5.4))

# 若要显示 raw，请确保前面把 SHOW_RAW = True
if SHOW_RAW:
    # 轻度下采样，避免太密：最多 ~5000 个点
    skipL = max(1, len(xL) // 10000)
    skipN = max(1, len(xN) // 10000)
    plt.plot(xL[::skipL], fcL[::skipL], color="#264385",lw=1, alpha=0.25)
    plt.plot(xN[::skipN], fcN[::skipN], color="#e23e35",lw=1, alpha=0.25)

# 平滑后的曲线
plt.plot(centersL[mL], fcL_sm[mL], color="#264385",label="C(with Loner)", lw=1.5, zorder=3)
plt.plot(centersN[mN], fcN_sm[mN], color="#e23e35",label="C(without Loner)", lw=1.5, zorder=3)

plt.xscale("log")
plt.xlabel("t")
plt.ylabel("$f_C$")
plt.title("")  # 半角括号
#plt.grid(True, which="both", ls="--", alpha=0.35)
plt.grid(False)
plt.legend()
plt.tight_layout()

out3 = os.path.join(OUTDIR, "compare_fc_logt_smoothed.png")
out3_svg = os.path.join(OUTDIR, "compare_fc_logt_smoothed.svg")
plt.savefig(out3, dpi=300)
plt.savefig(out3_svg, dpi=600)
plt.show()


print("[OK] 保存：")
print(" -", out1)
print(" -", out2)
print(" -", out3)

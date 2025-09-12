# -*- coding: utf-8 -*-
# 作用：绘制 L→ΔC、D→ΔL、C→ΔD 的“相关-滞后”曲线（原始步可选 + 10×下采样）
# 说明：横轴 lag 的单位：
#   - 原始步图：1 格 = 1 个原始时间步
#   - 下采样图：1 格 ≈ DS_STEP 个原始时间步
# 依赖：numpy, pandas, matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 基本配置（按需修改） ==========
LONER_CSV   = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv"     # 你的 Loner CSV 路径
NOLONER_CSV = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_NoLoner.csv"    # 你的 NoLoner CSV 路径
OUTDIR      = Path("./cc_plots")              # 输出目录

PLOT_RAW  = True   # 是否绘制“原始步”的相关-滞后曲线
PLOT_DS   = True   # 是否绘制“下采样（10×）”的曲线
DS_STEP   = 10     # 下采样因子（每一点≈DS_STEP个原始步）
MAX_LAG   = 200    # 最大滞后（在对应域内）
SHOW_FIGS = True   # 是否 plt.show() 弹出图窗

OUTDIR.mkdir(parents=True, exist_ok=True)

# ========== 工具函数 ==========
def pick(df, names):
    """从候选列名中选择首个存在的列名"""
    for n in names:
        if n in df.columns:
            return n
    return None

def zscore(x):
    """z 标准化；常量序列返回全 0，避免相关为 NaN"""
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / (sd + 1e-12)

def xcorr_lead(X, dY, max_lag=200):
    """返回 lags, cors；cors[k] = corr( X_t, dY_{t+k} )"""
    Xz, dYz = zscore(X), zscore(dY)
    lags = np.arange(0, max_lag + 1)
    cors = []
    for k in lags:
        if k == 0:
            x, y = Xz, dYz
        else:
            x, y = Xz[:-k], dYz[k:]
        if len(x) < 3 or len(y) < 3:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(r):
                r = np.nan
        cors.append(r)
    return lags, np.array(cors, float)

def plot_corr_curve(lags, cors, title, xlabel, outfile=None, show=True):
    """绘制一条相关-滞后曲线，并在标题中标注最佳滞后与相关"""
    plt.figure()
    plt.plot(lags, cors)
    # 标出峰值
    mask = np.isfinite(cors)
    if np.any(mask):
        kbest = int(lags[np.nanargmax(np.where(mask, cors, -1e9))])
        rbest = float(cors[kbest])
        plt.scatter([kbest], [rbest], s=50)
        title = f"{title}\n(best lag={kbest}, r≈{rbest:.3f})"
        print(f"{title}")
    else:
        print(f"{title}（无有效相关点）")
    plt.xlabel(xlabel)
    plt.ylabel("corr( X(t), ΔY(t+lag) )")
    plt.title(title)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=180)
    if show:
        plt.show()
    else:
        plt.close()

# ========== 读取数据 ==========
dL = pd.read_csv(LONER_CSV)
dN = pd.read_csv(NOLONER_CSV)
dL.columns = [c.strip() for c in dL.columns]
dN.columns = [c.strip() for c in dN.columns]

# 列名（带兜底）
cC_L = pick(dL, ["frac_C", "C_frac", "fC", "frac_c"])
cD_L = pick(dL, ["frac_D", "D_frac", "fD", "frac_d"])
cL_L = pick(dL, ["frac_L", "L_frac", "fL", "frac_l"])

cC_N = pick(dN, ["frac_C", "C_frac", "fC", "frac_c"])
cD_N = pick(dN, ["frac_D", "D_frac", "fD", "frac_d"])
# NoLoner 不会有 L 与 dL

# ========== 构造 Δ 序列 ==========
# 原始步
dL_raw = dL.copy()
dL_raw["dC"] = dL_raw[cC_L].diff().fillna(0.0)
dL_raw["dD"] = dL_raw[cD_L].diff().fillna(0.0)
dL_raw["dL"] = dL_raw[cL_L].diff().fillna(0.0)

dN_raw = dN.copy()
dN_raw["dC"] = dN_raw[cC_N].diff().fillna(0.0)
dN_raw["dD"] = dN_raw[cD_N].diff().fillna(0.0)

# 下采样域
dL_ds = dL.iloc[::DS_STEP, :].reset_index(drop=True).copy()
dL_ds["dC"] = dL_ds[cC_L].diff().fillna(0.0)
dL_ds["dD"] = dL_ds[cD_L].diff().fillna(0.0)
dL_ds["dL"] = dL_ds[cL_L].diff().fillna(0.0)

dN_ds = dN.iloc[::DS_STEP, :].reset_index(drop=True).copy()
dN_ds["dC"] = dN_ds[cC_N].diff().fillna(0.0)
dN_ds["dD"] = dN_ds[cD_N].diff().fillna(0.0)

# ========== 绘图（Loner）==========
# ——— L → ΔC
if PLOT_RAW:
    lags, cors = xcorr_lead(dL_raw[cL_L].values, dL_raw["dC"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[Loner RAW] L→ΔC",
        xlabel="lag（原始时间步）",
        outfile=OUTDIR / "L_to_dC_Loner_RAW.png",
        show=SHOW_FIGS
    )
if PLOT_DS:
    lags, cors = xcorr_lead(dL_ds[cL_L].values, dL_ds["dC"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[Loner DS] L→ΔC",
        xlabel=f"lag（每格≈{DS_STEP} 个原始时间步）",
        outfile=OUTDIR / "L_to_dC_Loner_DS.png",
        show=SHOW_FIGS
    )

# ——— D → ΔL
if PLOT_RAW:
    lags, cors = xcorr_lead(dL_raw[cD_L].values, dL_raw["dL"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[Loner RAW] D→ΔL",
        xlabel="lag（原始时间步）",
        outfile=OUTDIR / "D_to_dL_Loner_RAW.png",
        show=SHOW_FIGS
    )
if PLOT_DS:
    lags, cors = xcorr_lead(dL_ds[cD_L].values, dL_ds["dL"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[Loner DS] D→ΔL",
        xlabel=f"lag（每格≈{DS_STEP} 个原始时间步）",
        outfile=OUTDIR / "D_to_dL_Loner_DS.png",
        show=SHOW_FIGS
    )

# ——— C → ΔD
if PLOT_RAW:
    lags, cors = xcorr_lead(dL_raw[cC_L].values, dL_raw["dD"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[Loner RAW] C→ΔD",
        xlabel="lag（原始时间步）",
        outfile=OUTDIR / "C_to_dD_Loner_RAW.png",
        show=SHOW_FIGS
    )
if PLOT_DS:
    lags, cors = xcorr_lead(dL_ds[cC_L].values, dL_ds["dD"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[Loner DS] C→ΔD",
        xlabel=f"lag（每格≈{DS_STEP} 个原始时间步）",
        outfile=OUTDIR / "C_to_dD_Loner_DS.png",
        show=SHOW_FIGS
    )

# ========== 绘图（NoLoner）==========
# NoLoner 可做：C → ΔD
if PLOT_RAW:
    lags, cors = xcorr_lead(dN_raw[cC_N].values, dN_raw["dD"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[NoLoner RAW] C→ΔD",
        xlabel="lag（原始时间步）",
        outfile=OUTDIR / "C_to_dD_NoLoner_RAW.png",
        show=SHOW_FIGS
    )
if PLOT_DS:
    lags, cors = xcorr_lead(dN_ds[cC_N].values, dN_ds["dD"].values, max_lag=MAX_LAG)
    plot_corr_curve(
        lags, cors,
        title="[NoLoner DS] C→ΔD",
        xlabel=f"lag（每格≈{DS_STEP} 个原始时间步）",
        outfile=OUTDIR / "C_to_dD_NoLoner_DS.png",
        show=SHOW_FIGS
    )

print(f"完成。图片也已保存到：{OUTDIR.resolve()}")

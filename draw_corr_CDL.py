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

# ========= 统一字体 =========
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 基本配置（按需修改） ==========
LONER_CSV   = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv"
NOLONER_CSV = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_NoLoner.csv"
OUTDIR      = Path("./cc_plots")              # 输出目录

PLOT_RAW  = True   # 是否绘制“原始步”的相关-滞后曲线
PLOT_DS   = True   # 是否绘制“下采样（10×）”的曲线
DS_STEP   = 10     # 下采样因子（每一点≈DS_STEP个原始步）
MAX_LAG   = 200    # 最大滞后（在对应域内）
SHOW_FIGS = True   # 是否 plt.show() 弹出图窗

OUTDIR.mkdir(parents=True, exist_ok=True)

# ========== 读取数据 ==========
dL = pd.read_csv(LONER_CSV)
dN = pd.read_csv(NOLONER_CSV)
dL.columns = [c.strip() for c in dL.columns]
dN.columns = [c.strip() for c in dN.columns]

# ========= 列名（逐项确定，不用函数）=========
candidates_C = ["frac_C", "C_frac", "fC", "frac_c"]
candidates_D = ["frac_D", "D_frac", "fD", "frac_d"]
candidates_L = ["frac_L", "L_frac", "fL", "frac_l"]

cC_L = None
for n in candidates_C:
    if n in dL.columns:
        cC_L = n; break
cD_L = None
for n in candidates_D:
    if n in dL.columns:
        cD_L = n; break
cL_L = None
for n in candidates_L:
    if n in dL.columns:
        cL_L = n; break

cC_N = None
for n in candidates_C:
    if n in dN.columns:
        cC_N = n; break
cD_N = None
for n in candidates_D:
    if n in dN.columns:
        cD_N = n; break

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

# ========== 共同量 ==========
lags = n

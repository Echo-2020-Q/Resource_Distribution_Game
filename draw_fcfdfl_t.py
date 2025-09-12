# -*- coding: utf-8 -*-
"""
plot_fc_fd_fl_logt_fixed_linear.py
不使用自定义函数/类，按顺序直接执行：
- 读取 Loner / NoLoner 两个 CSV
- 自动识别 t / fc / fd / fl 列名（大小写不敏感）
- 若 t 含 0 或负数，整体平移到 >=1 再用对数横轴
- 生成三张图：
    1) Loner_fc_fd_fl_logt.png
    2) NoLoner_fc_fd_fl_logt.png
    3) compare_fc_logt.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 先把 Times 作为第一选择，缺字时按顺序回退到中文字体
mpl.rcParams['font.family'] = ['Times New Roman',
                               'Microsoft YaHei',   # 微软雅黑
                               'SimHei',            # 黑体
                               'Noto Sans CJK SC',  # 谷歌 Noto（若已装）
                               'Arial Unicode MS']  # 覆盖面大，机器上有的话
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False  # 修复负号显示

# 可选：强制所有中文文本用中文字体（保留英文字体为 Times）
# from matplotlib.font_manager import FontProperties
# zh = FontProperties(family='Microsoft YaHei')
# plt.title("含中文的标题", fontproperties=zh)

# ========= 修改这里：CSV 路径 =========
CSV_LONER   = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv"
CSV_NOLONER = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_NoLoner.csv"
OUTDIR      = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191220扫描b"

# ========= 字体（可选） =========
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

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

# 先精确匹配，否则大小写不敏感匹配
t_colL = None
for c in cands_t:
    if c in colsL:
        t_colL = c; break
if t_colL is None:
    lowerL = {c.lower(): c for c in colsL}
    for c in cands_t:
        if c.lower() in lowerL:
            t_colL = lowerL[c.lower()]; break

fc_colL = None
for c in cands_fc:
    if c in colsL:
        fc_colL = c; break
if fc_colL is None:
    lowerL = {c.lower(): c for c in colsL}
    for c in cands_fc:
        if c.lower() in lowerL:
            fc_colL = lowerL[c.lower()]; break

fd_colL = None
for c in cands_fd:
    if c in colsL:
        fd_colL = c; break
if fd_colL is None:
    lowerL = {c.lower(): c for c in colsL}
    for c in cands_fd:
        if c.lower() in lowerL:
            fd_colL = lowerL[c.lower()]; break

fl_colL = None
for c in cands_fl:
    if c in colsL:
        fl_colL = c; break
if fl_colL is None:
    lowerL = {c.lower(): c for c in colsL}
    for c in cands_fl:
        if c.lower() in lowerL:
            fl_colL = lowerL[c.lower()]; break

missingL = [name for name, col in [("t", t_colL), ("fc", fc_colL), ("fd", fd_colL), ("fl", fl_colL)] if col is None]
if missingL:
    raise KeyError(f"[Loner] 无法找到这些列：{missingL}；可用列为：{colsL}")

# 转数值 + 过滤 NaN
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

# 自动识别列名（NoLoner）
t_colN = None
for c in cands_t:
    if c in colsN:
        t_colN = c; break
if t_colN is None:
    lowerN = {c.lower(): c for c in colsN}
    for c in cands_t:
        if c.lower() in lowerN:
            t_colN = lowerN[c.lower()]; break

fc_colN = None
for c in cands_fc:
    if c in colsN:
        fc_colN = c; break
if fc_colN is None:
    lowerN = {c.lower(): c for c in colsN}
    for c in cands_fc:
        if c.lower() in lowerN:
            fc_colN = lowerN[c.lower()]; break

fd_colN = None
for c in cands_fd:
    if c in colsN:
        fd_colN = c; break
if fd_colN is None:
    lowerN = {c.lower(): c for c in colsN}
    for c in cands_fd:
        if c.lower() in lowerN:
            fd_colN = lowerN[c.lower()]; break

fl_colN = None
for c in cands_fl:
    if c in colsN:
        fl_colN = c; break
if fl_colN is None:
    lowerN = {c.lower(): c for c in colsN}
    for c in cands_fl:
        if c.lower() in lowerN:
            fl_colN = lowerN[c.lower()]; break

missingN = [name for name, col in [("t", t_colN), ("fc", fc_colN), ("fd", fd_colN), ("fl", fl_colN)] if col is None]
if missingN:
    raise KeyError(f"[NoLoner] 无法找到这些列：{missingN}；可用列为：{colsN}")

# 转数值 + 过滤 NaN
tN  = pd.to_numeric(dfN[t_colN],  errors="coerce")
fcN = pd.to_numeric(dfN[fc_colN], errors="coerce")
fdN = pd.to_numeric(dfN[fd_colN], errors="coerce")
flN = pd.to_numeric(dfN[fl_colN], errors="coerce")
maskN = ~(tN.isna() | fcN.isna() | fdN.isna() | flN.isna())
tN, fcN, fdN, flN = tN[maskN].values, fcN[maskN].values, fdN[maskN].values, flN[maskN].values

# ====== 确保输出目录 ======
os.makedirs(OUTDIR, exist_ok=True)

# ====== Loner：绘制三曲线（log x） ======
tminL = float(np.nanmin(tL)) if len(tL) else 0.0
shiftL = 1.0 - tminL if tminL <= 0 else 0.0
xL = tL + shiftL
xlabL = "t (log scale)" if shiftL == 0 else f"t (shift {shiftL:.3g}, log scale)"
plt.figure(figsize=(8,5))
plt.plot(xL, fcL, label="fc")
plt.plot(xL, fdL, label="fd")
plt.plot(xL, flL, label="fl")
plt.xscale("log")
plt.xlabel(xlabL); plt.ylabel("fraction")
plt.title("Loner: fc/fd/fl 演化")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(); plt.tight_layout()
out1 = os.path.join(OUTDIR, "Loner_fc_fd_fl_logt.png")
plt.savefig(out1, dpi=220)
plt.show()

# ====== NoLoner：绘制三曲线（log x） ======
tminN = float(np.nanmin(tN)) if len(tN) else 0.0
shiftN = 1.0 - tminN if tminN <= 0 else 0.0
xN = tN + shiftN
xlabN = "t (log scale)" if shiftN == 0 else f"t (shift {shiftN:.3g}, log scale)"
plt.figure(figsize=(8,5))
plt.plot(xN, fcN, label="fc")
plt.plot(xN, fdN, label="fd")
# NoLoner 时 fl 可能为 0（如果列存在即画出，便于比较），这里仍绘制
plt.plot(xN, flN, label="fl")
plt.xscale("log")
plt.xlabel(xlabN); plt.ylabel("fraction")
plt.title("NoLoner: fc/fd/fl 演化")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(); plt.tight_layout()
out2 = os.path.join(OUTDIR, "NoLoner_fc_fd_fl_logt.png")
plt.savefig(out2, dpi=220)
plt.show()
# ====== 比较图：仅 fc（log x） ======
plt.figure(figsize=(8,5))
lab1 = f"Loner fc (shift {shiftL:.3g})" if shiftL != 0 else "Loner fc"
lab2 = f"NoLoner fc (shift {shiftN:.3g})" if shiftN != 0 else "NoLoner fc"
plt.plot(xL, fcL, label=lab1)
plt.plot(xN, fcN, label=lab2)
plt.xscale("log")
plt.xlabel("t (log scale)")
plt.ylabel("fraction")
plt.title("fc 对比 (Loner vs NoLoner)")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(); plt.tight_layout()
out3 = os.path.join(OUTDIR, "compare_fc_logt.png")
plt.savefig(out3, dpi=220)
plt.show()
print("[OK] 保存：")
print(" -", out1)
print(" -", out2)
print(" -", out3)

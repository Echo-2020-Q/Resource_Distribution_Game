# ==== 0) 准备环境 ====
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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

CBAR_LABEL_SIZE = 26  # 颜色条标签字体大小

# ==== 1) 读取并筛选数据 ====
CSV = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\扫描sigma\sweep_allow_forced_l_in_cd-eta-r-sigma_T50000_rep50_20250916-110326.csv"
df = pd.read_csv(CSV, comment="#")
df = df[df["allow_forced_l_in_cd"] == True].copy()

# ==== 2) 输出目录（和 CSV 同目录下建 heatmaps 文件夹）====
OUTDIR = os.path.join(os.path.dirname(CSV), "heatmaps")
os.makedirs(OUTDIR, exist_ok=True)

# ==== 3) frac_C_mean 的热力图（sigma × r）====
pvt = df.pivot_table(index="sigma", columns="r", values="frac_C_mean", aggfunc="mean")
pvt = pvt.sort_index().reindex(sorted(pvt.columns), axis=1)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
Z = pvt.to_numpy()
# 用 viridis 连续色图，并设置插值方式为 bilinear
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="YlGnBu", interpolation="bilinear")
# 横坐标：固定在 1.0, 1.1, ..., 2.0
xticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
ax.set_xticks([np.argmin(np.abs(pvt.columns - v)) for v in xticks])
ax.set_xticklabels([f"{v:.1f}" for v in xticks])
# 纵坐标：sigma 全部或抽样
s_vals = pvt.index.values
ytick_idx = np.linspace(0, len(s_vals) - 1, num=min(12, len(s_vals)), dtype=int)
ax.set_yticks(ytick_idx)
ax.set_yticklabels([f"{s_vals[i]:.2f}" for i in ytick_idx])

# 添加等值线
levels = [0.01, 0.35, 0.8]

# 分别指定颜色：0.01(即0)为黑色，0.35和0.8为白色
CS = ax.contour(Z, levels=levels, colors=['k', 'white', 'white'], linewidths=2)
fmt = {0.01: '0', 0.35: '0.35', 0.8: '0.8'}
labels = ax.clabel(CS, inline=True, fontsize=22, fmt=fmt)
for l in labels:
    l.set_rotation(0)

ax.set_xlabel("b")
ax.set_ylabel("$\sigma$")
ax.set_title("")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$f_c(∞)$", fontsize=CBAR_LABEL_SIZE)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_frac_C_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_frac_C_mean.png"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_frac_C_mean.pdf"), dpi=450)


# ==== 4) avg_res_mean 的热力图 ====
pvt = df.pivot_table(index="sigma", columns="r", values="avg_res_mean", aggfunc="mean")
pvt = pvt.sort_index().reindex(sorted(pvt.columns), axis=1)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
Z = pvt.to_numpy()
# 用 viridis 连续色图，并设置插值方式为 bilinear
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="YlOrBr", interpolation="bilinear")
# 横坐标：固定在 1.0, 1.1, ..., 2.0
xticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
ax.set_xticks([np.argmin(np.abs(pvt.columns - v)) for v in xticks])
ax.set_xticklabels([f"{v:.1f}" for v in xticks])
# 纵坐标：sigma 全部或抽样
s_vals = pvt.index.values
ytick_idx = np.linspace(0, len(s_vals) - 1, num=min(12, len(s_vals)), dtype=int)
ax.set_yticks(ytick_idx)
ax.set_yticklabels([f"{s_vals[i]:.2f}" for i in ytick_idx])
ax.set_xlabel("b")
ax.set_ylabel("$\sigma$")
ax.set_title("")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\bar{R}_{res}$", fontsize=CBAR_LABEL_SIZE)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_avg_res_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_avg_res_mean.png"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_avg_res_mean.pdf"), dpi=450)



# ==== 5) net_output_hat_mean 的热力图 ====
pvt = df.pivot_table(index="sigma", columns="r", values="net_output_hat_mean", aggfunc="mean")
pvt = pvt.sort_index().reindex(sorted(pvt.columns), axis=1)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
Z = pvt.to_numpy()
# 用 viridis 连续色图，并设置插值方式为 bilinear
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="YlGn", interpolation="bilinear")

# 添加 r_net 的等值线
levels_rnet = [60, 75, 95]
CS_rnet = ax.contour(Z, levels=levels_rnet, colors='white', linewidths=2)
fmt_rnet = {60: '60', 75: '75', 95: '95'}
labels_rnet = ax.clabel(CS_rnet, inline=True, fontsize=22, fmt=fmt_rnet)
for l in labels_rnet:
    l.set_rotation(0)

# 横坐标：固定在 1.0, 1.1, ..., 2.0
xticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
ax.set_xticks([np.argmin(np.abs(pvt.columns - v)) for v in xticks])
ax.set_xticklabels([f"{v:.1f}" for v in xticks])
# 纵坐标：sigma 全部或抽样
s_vals = pvt.index.values
ytick_idx = np.linspace(0, len(s_vals) - 1, num=min(12, len(s_vals)), dtype=int)
ax.set_yticks(ytick_idx)
ax.set_yticklabels([f"{s_vals[i]:.2f}" for i in ytick_idx])

ax.set_xlabel("b")
ax.set_ylabel("$\sigma$")
ax.set_title("")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\bar{r}_{net}$", fontsize=CBAR_LABEL_SIZE)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_net_output_hat_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_net_output_hat_mean.png"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_net_output_hat_mean.pdf"), dpi=450)

# ==== 6) Gini 指数的热力图（sigma × r）====
def _pivot_gini_mean(df_in):
    if "gini_mean" in df_in.columns:
        pvt_ = df_in.pivot_table(index="sigma", columns="r", values="gini_mean", aggfunc="mean")
    elif "gini" in df_in.columns:
        tmp = (df_in.groupby(["sigma", "r"], as_index=False)["gini"].mean()
                     .rename(columns={"gini": "gini_mean"}))
        pvt_ = tmp.pivot(index="sigma", columns="r", values="gini_mean")
    else:
        raise KeyError("CSV 中未找到 gini_mean 或 gini 列，无法绘制 Gini 热力图。")
    return pvt_.sort_index().reindex(sorted(pvt_.columns), axis=1)

pvt = _pivot_gini_mean(df)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

Z = pvt.to_numpy(dtype=float)

# 统一配色与显示设置
# RdYlGn_r：低值偏红，高值偏绿；对“不平等越高越糟”的解读更直观
cmap = plt.get_cmap('PuBu').copy()
cmap.set_bad("#f0f0f0")  # 缺失值用浅灰

im = ax.imshow(
    Z, origin="lower", aspect="auto",
    cmap=cmap, interpolation="bilinear"
)

# # 在图上叠加若干等值线，让结构更清晰（可选）
# try:
#     import numpy as _np
#     # 自动在数据范围内取 6 条等值线
#     vmin, vmax = _np.nanmin(Z), _np.nanmax(Z)
#     levels = _np.linspace(vmin, vmax, num=6)
#     cs = ax.contour(Z, levels=levels, colors="k", linewidths=0.6, alpha=0.6)
#     ax.clabel(cs, fmt="%.2f", fontsize=9, inline=True)
# except Exception:
#     pass

# 统一的刻度：横轴 r（b），纵轴 sigma
r_vals = pvt.columns.values
s_vals = pvt.index.values

# 横坐标：固定在 1.0, 1.1, ..., 2.0
xticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
ax.set_xticks([np.argmin(np.abs(pvt.columns - v)) for v in xticks])
ax.set_xticklabels([f"{v:.1f}" for v in xticks])

ytick_idx = np.linspace(0, len(s_vals) - 1, num=min(12, len(s_vals)), dtype=int)

# ax.set_xticks(xtick_idx)
# ax.set_xticklabels([f"{r_vals[i]:.2f}" for i in xtick_idx])
ax.set_yticks(ytick_idx)
ax.set_yticklabels([f"{s_vals[i]:.2f}" for i in ytick_idx])

ax.set_xlabel("b")
ax.set_ylabel(r"$\sigma$")
ax.set_title("")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Gini", fontsize=CBAR_LABEL_SIZE)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_gini_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_gini_mean.png"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_gini_mean.pdf"), dpi=450)

print("Saved to:", OUTDIR)
# 不立即 plt.show()，让你统一在文件末尾 show
plt.show()



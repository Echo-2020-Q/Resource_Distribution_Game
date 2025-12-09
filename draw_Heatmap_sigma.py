# ==== 0) 准备环境 ====
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# 横坐标：固定在 1.0, 1.2, ..., 2.0
xticks = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ax.set_xticks([np.argmin(np.abs(pvt.columns - v)) for v in xticks])
ax.set_xticklabels([f"{v:.1f}" for v in xticks])
# 纵坐标：sigma 全部或抽样
s_vals = pvt.index.values
ytick_idx = np.linspace(0, len(s_vals) - 1, num=min(12, len(s_vals)), dtype=int)
ax.set_yticks(ytick_idx)
ax.set_yticklabels([f"{s_vals[i]:.2f}" for i in ytick_idx])

r_vals = pvt.columns.values
s_vals = pvt.index.values
xtick_idx = np.linspace(0, len(r_vals) - 1, num=min(10, len(r_vals)), dtype=int)
ax.set_xticks(xtick_idx)
ax.set_xticklabels([f"{r_vals[i]:.2f}" for i in xtick_idx], rotation=0, ha="right")



ax.set_xlabel("b")
ax.set_ylabel("$\sigma$")
ax.set_title("")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("$f_c(∞)$")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_frac_C_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_frac_C_mean.png"), dpi=450)


# ==== 4) avg_res_mean 的热力图 ====
pvt = df.pivot_table(index="sigma", columns="r", values="avg_res_mean", aggfunc="mean")
pvt = pvt.sort_index().reindex(sorted(pvt.columns), axis=1)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
Z = pvt.to_numpy()
# 用 viridis 连续色图，并设置插值方式为 bilinear
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="YlOrBr", interpolation="bilinear")
# 横坐标：固定在 1.0, 1.2, ..., 2.0
xticks = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
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
cbar.set_label(r"$\bar{R}_{res}$")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_avg_res_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_avg_res_mean.png"), dpi=450)


# ==== 5) net_output_hat_mean 的热力图 ====
pvt = df.pivot_table(index="sigma", columns="r", values="net_output_hat_mean", aggfunc="mean")
pvt = pvt.sort_index().reindex(sorted(pvt.columns), axis=1)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
Z = pvt.to_numpy()
# 用 viridis 连续色图，并设置插值方式为 bilinear
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="YlGn", interpolation="bilinear")
# 横坐标：固定在 1.0, 1.2, ..., 2.0
xticks = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
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
cbar.set_label(r"$\bar{r}_{net}$")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_net_output_hat_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_net_output_hat_mean.png"), dpi=450)
plt.show()
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

# 让刻度数量在 8~10 以内，避免过密
xtick_idx = np.linspace(0, len(r_vals) - 1, num=min(10, len(r_vals)), dtype=int)
ytick_idx = np.linspace(0, len(s_vals) - 1, num=min(12, len(s_vals)), dtype=int)

ax.set_xticks(xtick_idx)
ax.set_xticklabels([f"{r_vals[i]:.2f}" for i in xtick_idx])
ax.set_yticks(ytick_idx)
ax.set_yticklabels([f"{s_vals[i]:.2f}" for i in ytick_idx])

ax.set_xlabel("b")
ax.set_ylabel(r"$\sigma$")
ax.set_title("")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Gini")

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "heatmap_gini_mean.svg"), dpi=450)
fig.savefig(os.path.join(OUTDIR, "heatmap_gini_mean.png"), dpi=450)
# 不立即 plt.show()，让你统一在文件末尾 show
plt.show()

print("Saved to:", OUTDIR)

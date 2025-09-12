import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= 修改这里：你的 CSV 路径 =========
CSV_PATH = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191220扫描b\sweep_allow_forced_l_in_cd-eta-r_T50000_rep50_20250912-191220.csv"

# ========= 字体 =========
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========= 读取 & 预处理 =========
df = pd.read_csv(CSV_PATH, comment="#")

# 统一 r 为数值，并量化为一位小数的 b
df["r"] = pd.to_numeric(df["r"], errors="coerce")
df["b"] = df["r"].round(1)

# 统一 allow_forced_l_in_cd 为布尔
df["allow_forced_l_in_cd"] = (
    df["allow_forced_l_in_cd"].astype(str).str.strip().str.lower()
      .map({"true": True, "false": False, "1": True, "0": False})
      .fillna(False).astype(bool)
)

# 只保留 b=1.0..2.0（步长0.1）
valid_b = np.round(np.arange(1.0, 2.0 + 1e-9, 0.1), 1)
df = df[df["b"].isin(valid_b)].copy()

# 选择净产出列（兼容不同命名）
if "net_output_hat_mean" in df.columns:
    net_col = "net_output_hat_mean"
else:
    net_col = "net_sum_res_mean"  # 若没有上面的列，退回到这个

# —— 关键：按 (b, allow_forced_l_in_cd) 聚合成唯一点 ——
agg_cols = ["gini_mean", "avg_res_mean", net_col]
agg_df = (
    df.groupby(["b", "allow_forced_l_in_cd"], as_index=False)[agg_cols]
      .mean(numeric_only=True)
      .sort_values(["b", "allow_forced_l_in_cd"])
      .reset_index(drop=True)
)

# 先在画图之前算一个全局的微移尺度
gmin, gmax = agg_df["gini_mean"].min(), agg_df["gini_mean"].max()
xr = (gmax - gmin) if np.isfinite(gmax - gmin) and (gmax > gmin) else 1.0
dx = 0.006 * xr   # 横向微移 ~0.6% 的Gini跨度，可按视觉调 0.004~0.01

# 自检：b≠2.0 时，每组应各有10个点
print("with Loner points (b≠2.0):", agg_df.query("allow_forced_l_in_cd==True and b<2.0").shape[0])
print("without Loner points (b≠2.0):", agg_df.query("allow_forced_l_in_cd==False and b<2.0").shape[0])

# ========= 画图配置 =========
label_bs = [1.0, 1.1, 1.2, 1.3,1.4,1.5, 1.6,1.7,1.8, 1.9]      # 只给这几个b加文本标注；要全标就用 list(valid_b[:-1])

# ------------------ 图1：Gini vs net ------------------
plt.figure(figsize=(8,5))
y_col = net_col

for allow_loner in [True]:
    sub = agg_df[(agg_df["allow_forced_l_in_cd"] == allow_loner) & (agg_df["b"] < 2.0)]
    plt.scatter(
        sub["gini_mean"],
        sub[y_col],
        facecolors="#82c9ff",  # 填充色
        edgecolors="none",  # 不画边框
        marker="o",
        alpha=0.8,
        s=40,
        label=("with Loner" if allow_loner else "without Loner")
    )
    # 标注部分 b
    # for _, row in sub.iterrows():
    #     b = float(row["b"])
    #     if b in label_bs:
    #         plt.text(row["gini_mean"], row[y_col], f"b={b:.1f}",
    #                  fontsize=8, ha="left", va="bottom")

for allow_loner in [False]:
    sub = agg_df[(agg_df["allow_forced_l_in_cd"] == allow_loner) & (agg_df["b"] < 2.0)]
    plt.scatter(
        sub["gini_mean"],
        sub[y_col],
        #c="#ff7f0e",
        facecolors="none",  # 面色透明
        edgecolors="#f2724d",
        alpha=0.8,
        s=40,
        label=("with Loner" if allow_loner else "without Loner")
    )


    # 标注部分 b
    # for _, row in sub.iterrows():
    #     b = float(row["b"])
    #     if b in label_bs:
    #         plt.text(row["gini_mean"], row[y_col], f"b={b:.1f}",
    #                  fontsize=8, ha="left", va="bottom")

# 底部画 I_collapse（不使用真实 b=2.0 的 Gini 值）
ax = plt.gca()
ax.autoscale(enable=True, axis="both", tight=False)
xmin, xmax = ax.get_xlim(); xr = xmax - xmin
ymin, ymax = ax.get_ylim(); yr = ymax - ymin
plt.scatter(xmax + 0.05*xr, ymin + 0.03*yr, facecolors="none", edgecolors="#8a95a9",
            linewidths=1.5, zorder=5, label=r"$I_{collapse}$ (with Loner)")
plt.scatter(xmax + 0.05*xr, ymin + 0.03*yr, color="#8a95a9", marker="x",
            zorder=5, label=r"$I_{collapse}$ (without Loner)")

plt.xlabel("Gini"); plt.ylabel(r"$\bar{r}_{net}$")
plt.legend(); plt.grid(False); plt.tight_layout()
out1 = os.path.join(os.path.dirname(CSV_PATH) or ".", "gini_vs_netoutput.png")
out1_svg = os.path.join(os.path.dirname(CSV_PATH) or ".", "gini_vs_netoutput.svg")
plt.savefig(out1, dpi=600)
plt.savefig(out1_svg,dpi=600)


# ------------------ 图2：Gini vs avg_res_mean ------------------
plt.figure(figsize=(8,5))
y_col = "avg_res_mean"

for allow_loner in [True]:
    sub = agg_df[(agg_df["allow_forced_l_in_cd"] == allow_loner) & (agg_df["b"] < 2.0)]
    plt.scatter(
        sub["gini_mean"],
        sub[y_col],
        facecolors="#82c9ff",  # 填充色
        edgecolors="none",  # 不画边框
        marker="o",
        alpha=0.8,
        s=40,
        label=("with Loner" if allow_loner else "without Loner")
    )
    # 标注部分 b
    # for _, row in sub.iterrows():
    #     b = float(row["b"])
    #     if b in label_bs:
    #         plt.text(row["gini_mean"], row[y_col], f"b={b:.1f}",
    #                  fontsize=8, ha="left", va="bottom")

for allow_loner in [False]:
    sub = agg_df[(agg_df["allow_forced_l_in_cd"] == allow_loner) & (agg_df["b"] < 2.0)]
    plt.scatter(
        sub["gini_mean"],
        sub[y_col],
        #c="#ff7f0e",
        facecolors="none",  # 面色透明
        edgecolors="#f2724d",
        alpha=0.8,
        s=40,
        label=("with Loner" if allow_loner else "without Loner")
    )
    # for _, row in sub.iterrows():
    #     b = float(row["b"])
    #     if b in label_bs:
    #         plt.text(row["gini_mean"], row[y_col], f"b={b:.1f}",
    #                  fontsize=8, ha="left", va="bottom")

ax = plt.gca()
ax.autoscale(enable=True, axis="both", tight=False)
xmin, xmax = ax.get_xlim(); xr = xmax - xmin
ymin, ymax = ax.get_ylim(); yr = ymax - ymin
plt.scatter(xmax + 0.05*xr, ymin + 0.03*yr, facecolors="none", edgecolors="#8a95a9",
            linewidths=1.5, zorder=5, label=r"$I_{collapse}$ (with Loner)")
plt.scatter(xmax + 0.05*xr, ymin + 0.03*yr, color="#8a95a9", marker="x",
            zorder=5, label=r"$I_{collapse}$ (without Loner)")

plt.xlabel("Gini"); plt.ylabel(r"$\bar{R}_{res}$")
plt.legend(); plt.grid(False); plt.tight_layout()
out2 = os.path.join(os.path.dirname(CSV_PATH) or ".", "gini_vs_avgres.png")
out2_svg = os.path.join(os.path.dirname(CSV_PATH) or ".", "gini_vs_avgres.svg")
plt.savefig(out2, dpi=600)
plt.savefig(out2_svg,dpi=600)

plt.show()

# ------------------ 图3：net_col vs avg_res_mean ------------------
plt.figure(figsize=(8,5))

# with Loner
sub = agg_df[(agg_df["allow_forced_l_in_cd"] == True) & (agg_df["b"] < 2.0)]
plt.scatter(
    sub[net_col],
    sub["avg_res_mean"],
    facecolors="#82c9ff", edgecolors="none",
    marker="o", alpha=0.8, s=40,
    label="with Loner"
)

# without Loner
sub = agg_df[(agg_df["allow_forced_l_in_cd"] == False) & (agg_df["b"] < 2.0)]
plt.scatter(
    sub[net_col],
    sub["avg_res_mean"],
    facecolors="none", edgecolors="#f2724d",
    marker="o", alpha=0.8, s=40,
    label="without Loner"
)

# 底部画 I_collapse（代表 b=2.0，不使用真实点）
ax = plt.gca()
ax.autoscale(enable=True, axis="both", tight=False)
xmin, xmax = ax.get_xlim(); xr = xmax - xmin
ymin, ymax = ax.get_ylim(); yr = ymax - ymin
x_ic_with    = xmin + 0.05 * xr
x_ic_without = xmin + 0.05 * xr
y_ic         = ymin + 0.03 * yr

plt.scatter(x_ic_with, y_ic, facecolors="none", edgecolors="#8a95a9",
            linewidths=1.5, zorder=5, label=r"$I_{collapse}$ (with Loner)")
plt.scatter(x_ic_without, y_ic, color="#8a95a9", marker="x",
            zorder=5, label=r"$I_{collapse}$ (without Loner)")

plt.xlabel(r"$\bar{r}_{net}$")  # x 轴：净产出
plt.ylabel(r"$\bar{R}_{res}$")  # y 轴：平均资源
plt.legend()
plt.grid(False)
plt.tight_layout()
out3 = os.path.join(os.path.dirname(CSV_PATH) or ".", "net_vs_avgres.png")
out3_svg = os.path.join(os.path.dirname(CSV_PATH) or ".", "net_vs_avgres.svg")
plt.savefig(out3, dpi=600)
plt.savefig(out3_svg, dpi=600)

plt.show()
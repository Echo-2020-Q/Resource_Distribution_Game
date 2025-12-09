# draw_with_without_loner_fixed.py
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ========= 修改这里：你的 CSV 路径 =========
# 建议用原始字符串 r"..." 或用正斜杠 / 以避免转义问题
CSV_PATH = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191220扫描b\sweep_allow_forced_l_in_cd-eta-r_T50000_rep50_20250912-191220.csv"

# ========= 可选：避免中文标题警告（系统装了中文字体才会生效）=========
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
#matplotlib.rcParams['font.size'] = 14  # 全局字体大小
matplotlib.rcParams['axes.labelsize'] = 26  # 坐标轴标签字体大小
matplotlib.rcParams['xtick.labelsize'] = 24  # x轴刻度标签字体大小
matplotlib.rcParams['ytick.labelsize'] = 24  # y轴刻度标签字体大小
matplotlib.rcParams['legend.fontsize'] = 18  # 图例字体大小
matplotlib.rcParams['figure.titlesize'] = 24  # 图形标题字体大小
matplotlib.rcParams['font.weight'] = 'bold'  # 全局字体粗细
matplotlib.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签字体粗细
#===========================================================================================
plt.close("all")  # 关闭之前的图
def must_have(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"CSV 中缺少必要列：{miss}\n实际列名为：{list(df.columns)}")
#=============在绘制前做样条插值
import numpy as np
try:
    from scipy.interpolate import make_interp_spline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def _smooth_plot(x, y, style, color, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # 按 x 排序，确保单调
    order = np.argsort(x)
    x, y = x[order], y[order]

    if _HAS_SCIPY and len(x) >= 4:
        # 三次样条插值，点数>=4 时生效
        k = min(3, len(x) - 1)
        xs = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=k)
        ys = spline(xs)
        plt.plot(xs, ys, linestyle=style, color=color, label=label, linewidth=3)
    else:
        # 没有 scipy 就用原始点直接画
        plt.plot(x, y, linestyle=style, color=color, label=label, linewidth=3)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[Error] 找不到文件：{CSV_PATH}")
        sys.exit(1)

    # 读取 CSV：跳过以 # 开头的注释行；丢弃全空列
    df = pd.read_csv(CSV_PATH, comment="#", engine="python").dropna(axis=1, how="all")

    # 必要列名（与你的 sweep 输出一致）
    b_col = "r"  # 作为横轴的收益 b
    loner_flag_col = "allow_forced_l_in_cd"
    frac_C_col, frac_D_col, frac_L_col = "frac_C_mean", "frac_D_mean", "frac_L_mean"
    avg_res_col = "avg_res_mean"
    net_output_col = "net_output_hat_mean"  # 如要看 hat 版本可改 "net_output_hat_mean"

    # 必要列名
    must_have(df, [b_col, loner_flag_col, frac_C_col, frac_D_col, frac_L_col, avg_res_col, net_output_col, "gini_mean"])

    # 标志布尔化
    df["_allow_loner_bool"] = df[loner_flag_col].astype(bool)

    # 按 (b, 是否允许Loner) 聚合取均值（若同一 b 有多副本/其他参数）
    # 聚合列
    agg_cols = [frac_C_col, frac_D_col, frac_L_col, avg_res_col, net_output_col, "gini_mean"]
    agg_df = (
        df.groupby([b_col, "_allow_loner_bool"], as_index=False)[agg_cols]
          .mean(numeric_only=True)
          .sort_values([b_col, "_allow_loner_bool"])
          .reset_index(drop=True)
    )

    # ===== 图 1：C/D/L 比例 vs b（有/无 Loner）=====
    plt.figure(figsize=(10, 8))

    # (策略, 是否允许Loner) -> (颜色, 线型)
    style_map = {
        ("C", True): ("#264385", "-"),  # C 有Loner：深蓝 实线- 虚线--#264385
        ("C", False): ("#85c7e5", "--"),  # C 无Loner：浅蓝#85c7e5
        ("D", True): ("#e23e35", "-"),  # D 有Loner：红 实线#e23e35
        ("D", False): ("#f39fa2", "--"),  # D 无Loner：橙 虚线#f39fa2
        ("L", True): ("#dbb428", "-"),  # L 有Loner：绿 实线#2ca02c
        ("L", False): ("#eae936", "--"),  # L 无Loner：浅绿 虚线#98df8a#eae936
    }

    for allow_loner in [True, False]:
        sub = agg_df[agg_df["_allow_loner_bool"] == allow_loner]
        label_suffix = "with Loner" if allow_loner else "without Loner"
        x = sub[b_col].to_numpy()

        for strat, colname in [("C", frac_C_col), ("D", frac_D_col), ("L", frac_L_col)]:
            color, ls = style_map[(strat, allow_loner)]
            _smooth_plot(
                x, sub[colname].to_numpy(),
                style=ls, color=color,
                label=f"{strat}({label_suffix})"
            )

    plt.xlabel("b")
    plt.ylabel("The frequency of strategies")
    plt.title("")
    plt.legend()
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1a_fractions_vs_b.png")
    out1_svg = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1a_fractions_vs_b.svg")
    out1_pdf = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1a_fractions_vs_b.pdf")
    plt.grid(False)
    plt.savefig(out1, dpi=600)
    plt.savefig(out1_svg, dpi=600)
    plt.savefig(out1_pdf)

    #plt.show()
    # plt.close()

    # ===== 图 2：avg_res_mean vs b（有/无 Loner）=====
    plt.figure(figsize=(10, 8))

    # (是否允许Loner) -> (颜色, 线型, 阈值)
    style_map = {
        True: ("#264385", "-", 20),  # 允许Loner：深蓝 实线；崩溃阈值=-500
        False: ("#85c7e5", "--", -999),  # 不允许Loner：浅蓝 虚线；崩溃阈值=-1000
    }

    for allow_loner in [True, False]:
        color, ls, collapse_thresh = style_map[allow_loner]
        sub = agg_df[agg_df["_allow_loner_bool"] == allow_loner]
        label = "with Loner" if allow_loner else "without Loner"

        # 1) 画正常点：大于等于 0 的
        mask_valid = sub[avg_res_col] >= 20
        x_valid = sub.loc[mask_valid, b_col].to_numpy()
        y_valid = sub.loc[mask_valid, avg_res_col].to_numpy()
        plt.plot(x_valid, y_valid, ls, color=color, label=label, linewidth=3)

        # 2) 标注 <= collapse_thresh 的崩溃点
        mask_collapse = sub[avg_res_col] <= collapse_thresh

    #-----手动标注collapse
    ax = plt.gca()
    # ymin, ymax = ax.get_ylim()
    # y_annot = ymin + 0.05 * (ymax - ymin)  # 取底部稍微往上，避免跑到轴外

    # 你要显示的内容：
    # txt = "b=2, I_collapse (with Loner)"
    # ax.annotate(
    #     txt,
    #     xy=(2.0, y_annot), xycoords="data",  # 指向 b=2.0, y=y_annot
    #     xytext=(0, 8), textcoords="offset points",  # 文字再往上偏移 10
    #     ha="center", va="bottom",
    #     fontsize=9, color="blue", rotation=45,
    #     arrowprops=dict(arrowstyle="-", lw=0.6, color="blue")
    # )
    # ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    y_point = ymin + 0.05 * (ymax - ymin)  # 在底部稍微抬高一点，代替 y=0

    # 画一个蓝色圆点
    plt.scatter(2.0, y_point, s=200, facecolors="none", edgecolors="#8a95a9", linewidths=2.5, zorder=8, label="$I_{collapse}$(with Loner)")

    # 画一个青色方块点（另一种情况）
    plt.scatter(2.0, y_point , s=200, color="#8a95a9", marker="x", linewidths=2.5, zorder=8,
                label="$I_{collapse}$(without Loner)")

    # 如果还想在同一个 b=2.0 的位置再加一个 “without Loner” 标注
    # txt2 = "b=2, I_collapse (without Loner)"
    # ax.annotate(
    #     txt2,
    #     xy=(2.0, y_annot), xycoords="data",
    #     xytext=(0, 8), textcoords="offset points",  # 再偏移大一点，避免重叠
    #     ha="center", va="bottom",
    #     fontsize=9, color="cyan", rotation=0,
    #     arrowprops=dict(arrowstyle="-", lw=0.6, color="cyan")
    # )

    plt.xlabel("b")
    plt.ylabel(r"$\bar{R}_{res}$")
    plt.title("")
    plt.legend()
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1c_avg_res_mean_vs_b.png")
    out2_svg = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1c_avg_res_mean_vs_b.svg")
    out2_pdf = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1c_avg_res_mean_vs_b.pdf")
    plt.savefig(out2, dpi=600)
    plt.savefig(out2_svg, dpi=600)
    plt.savefig(out2_pdf)
    plt.grid(False)
    # plt.close()

    # ===== 图 3：Net_output vs b（有/无 Loner；各自阈值）=====
    plt.figure(figsize=(10, 8))

    style_map = {
        True: {"color": "#e23e35", "ls": "-", "collapse_thresh": 1.0},  # with Loner
        False: {"color": "#f39fa2", "ls": "--", "collapse_thresh": 0.0},  # without Loner
    }
    HIDE_RANGE = (-999, 0)

    ax = plt.gca()

    for allow_loner in [True, False]:
        cfg = style_map[allow_loner]
        color = cfg["color"]
        ls = cfg["ls"]
        thr = cfg["collapse_thresh"]

        sub = agg_df[agg_df["_allow_loner_bool"] == allow_loner]
        label = "with Loner" if allow_loner else "without Loner"

        mask_draw = sub[net_output_col] > 0.5
        if HIDE_RANGE is not None:
            low, high = HIDE_RANGE
            mask_draw &= ~((sub[net_output_col] > low) & (sub[net_output_col] < high))

        x_draw = sub.loc[mask_draw, b_col].to_numpy()
        y_draw = sub.loc[mask_draw, net_output_col].to_numpy()

        # 每条线单独一个 label
        plt.plot(x_draw, y_draw, ls, color=color, label=label, linewidth=3)

    # 坐标和标题
    plt.xlabel("b")
    plt.ylabel(r"$\bar{r}_{net}$")
    plt.title("")
    plt.grid(False)
    # -----手动标注collapse
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    y_point = ymin + 0.05 * (ymax - ymin)  # 在底部稍微抬高一点，代替 y=0

    # 画一个蓝色圆点
    plt.scatter(2.0, y_point, s=200, facecolors="none", edgecolors="#8a95a9", linewidths=2.5, zorder=8,
                label="$I_{collapse}$(with Loner)")

    # 画一个青色方块点（另一种情况）
    plt.scatter(2.0, y_point, s=200, color="#8a95a9", marker="x", linewidths=2.5, zorder=8,
                label="$I_{collapse}$(without Loner)")


    plt.legend()  # 加图例
    plt.tight_layout()
    out3 = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1b_net_output_vs_b.png")
    out3_svg = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1b_net_output_vs_b.svg")
    out3_pdf = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1b_net_output_vs_b.pdf")
    plt.savefig(out3, dpi=600)
    plt.savefig(out3_svg, dpi=600)
    plt.savefig(out3_pdf)
    plt.grid(False)
    # plt.close()

    # ===== 图 4：Gini vs b（有/无 Loner；b=2.0 不画点，改为 I_collapse 标注）=====
    plt.figure(figsize=(10, 8))

    color_map = {
        True: "#dbb428",  # with Loner
        False: "#eae936",  # without Loner
    }
    ls_map = {
        True: "-",  # with Loner
        False: "--",  # without Loner
    }

    B_COLLAPSE = 2.0  # 需要特殊处理的 b 值
    ATOL = 1e-9  # 浮点比较容差

    for allow_loner in [True, False]:
        # 只画除 b=2.0 之外的点
        sub_all = agg_df[agg_df["_allow_loner_bool"] == allow_loner]
        mask = ~np.isclose(sub_all[b_col].to_numpy(dtype=float), B_COLLAPSE, atol=ATOL)
        sub = sub_all.loc[mask]

        x = sub[b_col].to_numpy()
        y = sub["gini_mean"].to_numpy()

        # 平滑曲线（无散点、无逐点文本）
        _smooth_plot(
            x, y,
            style=ls_map[allow_loner],
            color=color_map[allow_loner],
            label="with Loner" if allow_loner else "without Loner"
        )

    # 在所有线画完之后再统一标注 I_collapse（位于底部附近）
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    y_point = ymin + 0.05 * (ymax - ymin)  # 底部稍抬起，避免压在轴上

    # 空心圆：with Loner
    plt.scatter(
        B_COLLAPSE, y_point, s=200,
        facecolors="none",edgecolors="#8a95a9", linewidths=2.5, zorder=8,
        label=r"$I_{collapse}$ (with Loner)"
    )
    # 叉号：without Loner
    plt.scatter(
        B_COLLAPSE, y_point, s=200,
        color="#8a95a9", marker="x", linewidths=2.5, zorder=8,
        label=r"$I_{collapse}$ (without Loner)"
    )

    plt.xlabel("b")
    plt.ylabel("Gini")
    plt.title("")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    out4 = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1d_gini_vs_b.png")
    out4_svg = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1d_gini_vs_b.svg")
    out4_pdf = os.path.join(os.path.dirname(CSV_PATH) or ".", "Fig1d_gini_vs_b.pdf")
    plt.savefig(out4, dpi=600)
    plt.savefig(out4_svg, dpi=600)
    plt.savefig(out4_pdf)
    plt.show()

    print("✅ 4张图已生成：")
    print("  -", out1)
    print("  -", out2)
    print("  -", out3)
    print("  -", out4)

if __name__ == "__main__":
    main()

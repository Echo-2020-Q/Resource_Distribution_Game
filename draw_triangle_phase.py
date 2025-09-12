# draw_triangle_phase.py
# 读取包含 (t, frac_C, frac_D, frac_L) 的 CSV，
# 生成：timeseries_fractions.png、phase_cloud_fc_fd.png、ternary_cloud.png
# 并在丢弃 burn-in 之后输出均值/标准差/变异系数(CV)，另存 stationary_stats.csv。
# 示例：
# python draw_triangle_phase.py --csv "D:\...\trend.csv" --outdir out_figs
#python draw_triangle_phase.py --csv "D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\20250912-191224\allow_forced_l_in_cd=True_eta=0.0_r=1.52\rep0\trend.csv" --outdir out_figs  --stream-len 300 --stream-stride 2 --smooth-k 3
#3) 怎么调出示例那种“密集扇形”质感？

#--stream-stride 调小（比如 10 或 5）→ 曲线更密。

#--stream-len 调大（比如 200~300）→ 单条曲线更长、更弯。

#--smooth-k 调 3、5、7 这样的小奇数 → 曲线更圆滑，避免“抖动”。

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    raise KeyError(f"未找到列：候选 {candidates}；实际列：{list(df.columns)}")

def ternary_project(fC, fD, fL):
    # 等边三角形投影：C->(0,0), D->(1,0), L->(1/2, sqrt(3)/2)
    x = fD + 0.5 * fL
    y = (np.sqrt(3)/2.0) * fL
    return x, y

def main():
    ap = argparse.ArgumentParser(description="Plot f_C, f_D, f_L time series & clouds")
    ap.add_argument("--stream-len", type=int, default=150, help="每条小曲线的时间长度（步数），默认 150")
    ap.add_argument("--stream-stride", type=int, default=25, help="相邻曲线起点的步长（越小越密），默认 25")
    ap.add_argument("--smooth-k", type=int, default=5, help="滚动均值的平滑窗口（奇数更佳），默认 5；=1 则不平滑")
    ap.add_argument("--csv", required=True, help="输入 CSV 路径")
    ap.add_argument("--outdir", default=".", help="输出目录（默认当前）")
    ap.add_argument("--burn-frac", type=float, default=0.0,
                    help="burn-in 百分比（0~1，默认 0.10）")
    ap.add_argument("--burn-abs", type=float, default=None,
                    help="burn-in 绝对时间（单位与 t 同，优先于 burn-frac）")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(args.csv)

    # 标准化列名（仅用于匹配，不改原表）
    lower_map = {c: c.strip().lower() for c in df_raw.columns}
    df = df_raw.rename(columns=lower_map).copy()

    t_col = pick_column(df, ["t", "time", "round"])
    c_col = pick_column(df, ["frac_c", "f_c", "fc"])
    d_col = pick_column(df, ["frac_d", "f_d", "fd"])
    l_col = pick_column(df, ["frac_l", "f_l", "fl"])

    df = df[[t_col, c_col, d_col, l_col]].rename(
        columns={t_col:"t", c_col:"frac_C", d_col:"frac_D", l_col:"frac_L"}
    ).dropna()

    # 统一数值化、丢 NaN、按时间排序（更稳健）
    for c in ["t", "frac_C", "frac_D", "frac_L"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t","frac_C","frac_D","frac_L"]).sort_values("t")

    # 和为1的校验与必要时归一
    s = df["frac_C"] + df["frac_D"] + df["frac_L"]
    if np.nanmax(np.abs(s.to_numpy() - 1.0)) > 1e-3:
        df[["frac_C","frac_D","frac_L"]] = df[["frac_C","frac_D","frac_L"]].div(s, axis=0)

    # burn-in
    t_min, t_max = float(df["t"].min()), float(df["t"].max())
    if args.burn_abs is not None:
        burn_in = args.burn_abs
    else:
        burn_in = t_min + args.burn_frac * (t_max - t_min)

    obs = df[df["t"] > burn_in].copy()
    if obs.empty:
        raise ValueError("burn-in 过大，平稳段为空。请调小 --burn-frac 或使用 --burn-abs。")

    # ===== 统计表：均值、标准差、CV =====
    stats = {}
    for key in ["frac_C","frac_D","frac_L"]:
        mean = float(obs[key].mean())
        std  = float(obs[key].std(ddof=1))
        cv   = float(std / mean) if mean != 0 else np.nan
        stats[key] = {"mean": mean, "std": std, "CV": cv}
    stats_df = pd.DataFrame(stats).T
    stats_path = outdir / "stationary_stats.csv"
    stats_df.to_csv(stats_path, float_format="%.6g")
    print("平稳段统计：\n", stats_df)

    # ===== (A) 时间序列（单图，无子图） =====
    t  = df["t"].to_numpy()
    fC = df["frac_C"].to_numpy()
    fD = df["frac_D"].to_numpy()
    fL = df["frac_L"].to_numpy()

    plt.figure()
    ax = plt.gca()
    ax.plot(t, fC, label="f_C")
    ax.plot(t, fD, label="f_D")
    ax.plot(t, fL, label="f_L")
    ax.axhline(stats["frac_C"]["mean"], linestyle="--", linewidth=1, label="mean f_C (obs)")
    ax.axhline(stats["frac_D"]["mean"], linestyle="--", linewidth=1, label="mean f_D (obs)")
    ax.axhline(stats["frac_L"]["mean"], linestyle="--", linewidth=1, label="mean f_L (obs)")
    ax.set_xlabel("t (round)")
    ax.set_ylabel("fraction")
    ax.set_title("Time series of f_C, f_D, f_L")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "timeseries_fractions.png", dpi=180, bbox_inches="tight")

    # ===== (B) (f_C, f_D) 相位云团 =====
    C_obs = obs["frac_C"].to_numpy()
    D_obs = obs["frac_D"].to_numpy()
    plt.figure()
    ax = plt.gca()
    ax.scatter(C_obs, D_obs, s=8, alpha=0.6, label="obs window")
    ax.plot(C_obs[:1], D_obs[:1], marker="o", markersize=6, label="start")
    ax.plot(C_obs[-1:], D_obs[-1:], marker="x", markersize=6, label="end")
    ax.set_xlabel("f_C")
    ax.set_ylabel("f_D")
    ax.set_title("(f_C, f_D) phase cloud (stationary window)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "phase_cloud_fc_fd.png", dpi=180, bbox_inches="tight")

    # ===== (C) 三元投影：流线风格（多段短曲线） =====
    def rolling_mean(a, k):
        """简单滚动均值平滑，返回与原长度相同（边缘用最邻近填充）"""
        if k <= 1:
            return a
        s = pd.Series(a)
        b = s.rolling(k, center=True, min_periods=1).mean().to_numpy()
        return b

    # 1) 取平稳段数据并投影到等边三角形
    C_obs = obs["frac_C"].to_numpy()
    D_obs = obs["frac_D"].to_numpy()
    L_obs = obs["frac_L"].to_numpy()
    x_all = D_obs + 0.5 * L_obs
    y_all = (np.sqrt(3) / 2.0) * L_obs

    # 2) 画三角形边与顶点标签（黑白期刊风格）
    plt.figure()
    ax = plt.gca()
    tri_x = np.array([0.0, 1.0, 0.5, 0.0])
    tri_y = np.array([0.0, 0.0, np.sqrt(3) / 2.0, 0.0])
    ax.plot(tri_x, tri_y, color="0.2", linewidth=1.5)

    # 顶点文字更接近示例风格
    ax.text(0.5, np.sqrt(3) / 2.0 + 0.02, "L", ha="center", va="bottom", fontsize=16)
    ax.text(-0.02, -0.02, "D", ha="right", va="top", fontsize=16)
    ax.text(1.02, -0.02, "C", ha="left", va="top", fontsize=16)

    # 3) 多段短曲线：从时间序列中按“起点步长”抽取很多段，每段长度为 stream_len
    seg_len = int(args.stream_len)
    stride = int(args.stream_stride)
    k_smooth = int(args.smooth_k)

    n = len(x_all)
    starts = range(0, max(n - seg_len, 1), stride)

    for s in starts:
        xs = x_all[s:s + seg_len]
        ys = y_all[s:s + seg_len]
        if len(xs) < 2:
            continue
        # 轻微平滑，让曲线更圆润
        xs = rolling_mean(xs, k_smooth)
        ys = rolling_mean(ys, k_smooth)

        ax.plot(xs, ys,
                color="k", linewidth=0.6, alpha=0.9,
                solid_capstyle="round")

    # 4) 版式与范围
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3) / 2.0 + 0.05)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_title("")  # 期刊黑白图风格一般不写标题
    plt.tight_layout()
    plt.savefig(outdir / "ternary_cloud.png", dpi=300, bbox_inches="tight", metadata=None)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
plot_event_study_cdl.py
生成 C/D/L 的事件研究图（以 D 上升为锚点），含 95% CI、平滑与下采样。
可选：三角单纯形稀疏箭头轨迹、单变量美化版图。

依赖：numpy, pandas, matplotlib
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============== 基础配置 ===============
CSV_PATH = r"D:\PyCharm_Community_Edition_2024_01_04\Py_Projects\Resource_Distribution_Game\Resource_Distribution_Game\sweep_out\b=1.6_有无Loner机制\capture_window_Loner.csv"   # 修改成你的路径
OUTDIR    = Path("./out_event_study")
OUTDIR.mkdir(parents=True, exist_ok=True)

# 你可以在这里添加多组配置（会依次产图）
CONFIGS = [
    # DS=30, q=0.80（我们之前的 n≈128 那版）
    dict(tag="q80_DS30", DS=30, q=0.80, pre=12, post=20, win=11, gap=8),
    # DS=30, q=0.75（n≈143）
    dict(tag="q75_DS30", DS=30, q=0.75, pre=12, post=20, win=11, gap=8),
    # DS=30, q=0.70, 窗口缩短到 -10…+12（n≈196）
    dict(tag="q70_DS30_short", DS=30, q=0.70, pre=10, post=12, win=11, gap=8),
]

# 是否额外生成：三角轨迹图 & 单变量美化图
MAKE_TERNARY = True
MAKE_SINGLE  = True


# =============== 工具函数 ===============
def _robust_load_cdl(csv_path: str) -> pd.DataFrame:
    """读取 CSV 并尽量鲁棒地识别列名，返回列为 [t, C, D, L] 的 DataFrame。"""
    df0 = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df0.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    t_col = pick("t","time","step","iter","episode") or df0.columns[0]
    c_col = pick("frac_c","fc","c","c_frac","f_c")
    d_col = pick("frac_d","fd","d","d_frac","f_d")
    l_col = pick("frac_l","fl","l","l_frac","f_l")
    if not all([t_col, c_col, d_col, l_col]):
        raise KeyError(f"无法识别列名，请检查：{df0.columns.tolist()}")

    df = df0[[t_col, c_col, d_col, l_col]].copy()
    df.columns = ["t","C","D","L"]
    for k in ["t","C","D","L"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # 如有微小偏差，按和为 1 归一化
    s = df[["C","D","L"]].sum(axis=1)
    if (np.abs(s-1.0) > 1e-6).any():
        df[["C","D","L"]] = df[["C","D","L"]].div(s, axis=0)
    return df


def _smooth_downsample(df: pd.DataFrame, DS=30, win=11) -> pd.DataFrame:
    """下采样 + 中值平滑（win 取奇数较好）。"""
    d = df.iloc[::DS, :].copy()
    for k in ["C","D","L"]:
        d[f"{k}_s"] = d[k].rolling(win, center=True).median()
    d = d.dropna().reset_index(drop=True)
    for k in ["C","D","L"]:
        d[f"d{k}"] = d[f"{k}_s"].diff()
    return d


def _select_anchors_D_rise(dfs: pd.DataFrame, q=0.80, gap=8):
    """以“D 上升”为锚点：dD >= 分位数(q)，并用 gap 去重以避免锚点簇。"""
    thr = dfs["dD"].quantile(q)
    cand = dfs.index[dfs["dD"] >= thr].tolist()
    anchors, last = [], -10**9
    for i in cand:
        if i - last >= gap:
            anchors.append(i)
            last = i
    return anchors, thr


def _collect_windows(dfs: pd.DataFrame, anchors, pre=12, post=20, var="C_s"):
    """按锚点抽取窗口，堆叠为 [n_anchors, pre+post+1] 的矩阵。"""
    mats = []
    for a in anchors:
        lo, hi = a - pre, a + post
        if lo < 0 or hi >= len(dfs):
            continue
        mats.append(dfs.loc[lo:hi, var].to_numpy())
    return (np.vstack(mats) if mats else None)


def _mean_ci(M: np.ndarray, alpha=0.05, B=None):
    """按行（锚点）自举，返回均值和点位 95% CI。"""
    if M is None or len(M) == 0:
        return None, None, None
    mean = M.mean(axis=0)
    n = M.shape[0]
    if B is None:
        B = min(2000, max(200, 20*n))
    boots = np.empty((B, M.shape[1]), float)
    rng = np.random.default_rng(42)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = M[idx].mean(axis=0)
    lo = np.percentile(boots, 100*alpha/2, axis=0)
    hi = np.percentile(boots, 100*(1-alpha/2), axis=0)
    return mean, lo, hi


# =============== 画图函数 ===============
def plot_event_study_panel(df: pd.DataFrame, tag="q80_DS30",
                           DS=30, q=0.80, pre=12, post=20, win=11, gap=8,
                           outdir=OUTDIR):
    """三联图：C/D/L 在锚点（D 上升）对齐的平均响应 + 95% CI。"""
    dfs = _smooth_downsample(df, DS=DS, win=win)
    anchors, thr = _select_anchors_D_rise(dfs, q=q, gap=gap)
    lags = np.arange(-pre, post+1)

    mats = {k: _collect_windows(dfs, anchors, pre, post, f"{k}_s") for k in ["C","D","L"]}
    n_used = 0 if mats["C"] is None else mats["C"].shape[0]
    stats = {k: _mean_ci(mats[k]) for k in ["C","D","L"]}

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, k in zip(axes, ["C","D","L"]):
        mean, lo, hi = stats[k]
        base = mean[pre]
        ax.fill_between(lags, lo, hi, alpha=0.18, linewidth=0)
        ax.plot(lags, mean, lw=2, label=f"{k} (mean)")
        ax.axvline(0, ls="--", lw=1)
        ax.axhline(base, ls=":", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel(f"Lag from D rising (×{DS} original steps); anchors n={n_used} (q={q:.2f})")
    axes[0].set_title("Event study aligned on 'D rising' (3-panel with 95% CI)")

    out = outdir / f"event_study_panel_{tag}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    return out, n_used


def plot_ternary_sparse_arrows(df: pd.DataFrame, DS=30, win=11, step=5, outdir=OUTDIR, tag="ternary"):
    """三角单纯形稀疏箭头轨迹（去噪 + 抽样）。"""
    dfs = _smooth_downsample(df, DS=DS, win=win)
    C, D, L = dfs["C_s"].to_numpy(), dfs["D_s"].to_numpy(), dfs["L_s"].to_numpy()
    S = C + D + L
    C, D, L = C/S, D/S, L/S
    x = 0.5*(2*D + L)
    y = (np.sqrt(3)/2)*L

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    tri = np.array([[0,0],[1,0],[0.5, np.sqrt(3)/2],[0,0]])
    ax.plot(tri[:,0], tri[:,1], lw=1, color="0.4")

    ax.plot(x, y, lw=0.6, alpha=0.2, color="0.2")
    for i in range(0, len(x)-step, step):
        ax.arrow(x[i], y[i], x[i+1]-x[i], y[i+1]-y[i],
                 head_width=0.015, head_length=0.02, length_includes_head=True,
                 fc="tab:gray", ec="tab:gray", alpha=0.6)
    ax.scatter(x[0], y[0], c="tab:blue", s=35, label="Start")
    ax.scatter(x[-1], y[-1], c="tab:red", s=35, label="End")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="upper right")
    ax.set_title("Cycle trajectory in C–D–L simplex (smoothed & decimated)")

    out = outdir / f"ternary_sparse_arrows_{tag}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_single_beautified(df: pd.DataFrame, DS=30, q=0.80, pre=12, post=20, win=11, gap=8,
                           outdir=OUTDIR, tag="beauty"):
    """生成三张“单变量美化版”事件图（每张一条曲线+95% CI+注释）。"""
    dfs = _smooth_downsample(df, DS=DS, win=win)
    anchors, thr = _select_anchors_D_rise(dfs, q=q, gap=gap)
    lags = np.arange(-pre, post+1)
    mats = {k: _collect_windows(dfs, anchors, pre, post, f"{k}_s") for k in ["C","D","L"]}
    n_used = 0 if mats["C"] is None else mats["C"].shape[0]
    stats = {k: _mean_ci(mats[k]) for k in ["C","D","L"]}

    def _one(k: str):
        mean, lo, hi = stats[k]
        base = mean[pre]
        fig = plt.figure(figsize=(8, 4.8))
        plt.fill_between(lags, lo, hi, alpha=0.2, linewidth=0)
        plt.plot(lags, mean, linewidth=2, label=f"{k} (mean)")
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.axhline(base, linestyle=":", linewidth=1)
        plt.xlabel(f"Lag from D rising (×{DS} original steps),  n={n_used} (q={q:.2f})")
        plt.ylabel("Fraction")
        plt.title(f"Event study (anchor: D rising) — {k} with 95% CI")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")

        # 简单注释：D 在 0 处上冲；C 在 0 后下降；L 在稍后回升
        i0 = pre
        i_end = min(len(lags)-1, pre+8)
        seg = mean[i0:i_end+1] - base
        if k == "C":
            j = int(np.argmin(seg)) + i0
            plt.annotate("↓ after D↑", xy=(lags[j], mean[j]),
                         xytext=(lags[j]+4, mean[j]-0.01),
                         arrowprops=dict(arrowstyle="->", lw=1))
        elif k == "D":
            j = int(np.argmax(seg)) + i0
            plt.annotate("↑ at D↑", xy=(lags[j], mean[j]),
                         xytext=(lags[j]+3, mean[j]+0.01),
                         arrowprops=dict(arrowstyle="->", lw=1))
        elif k == "L":
            i_end2 = min(len(lags)-1, pre+12)
            seg2 = mean[i0:i_end2+1] - base
            j = int(np.argmax(seg2)) + i0
            plt.annotate("滞后↑", xy=(lags[j], mean[j]),
                         xytext=(lags[j]+3, mean[j]+0.01),
                         arrowprops=dict(arrowstyle="->", lw=1))

        out = outdir / f"event_study_{k}_{tag}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close(fig)
        return out

    outs = [_one(k) for k in ["D","C","L"]]
    return outs


# =============== 主流程（按你的 CONFIGS 批量产图） ===============
def main():
    df = _robust_load_cdl(CSV_PATH)

    for cfg in CONFIGS:
        tag  = cfg.get("tag", "run")
        DS   = cfg.get("DS", 30)
        q    = cfg.get("q", 0.80)
        pre  = cfg.get("pre", 12)
        post = cfg.get("post", 20)
        win  = cfg.get("win", 11)
        gap  = cfg.get("gap", 8)

        out, n_used = plot_event_study_panel(df, tag, DS, q, pre, post, win, gap, OUTDIR)
        print(f"[OK] {out}   anchors n={n_used}")

        if MAKE_SINGLE:
            outs = plot_single_beautified(df, DS, q, pre, post, win, gap, OUTDIR, tag)
            for p in outs:
                print(f"[OK] {p}")

    if MAKE_TERNARY:
        outt = plot_ternary_sparse_arrows(df, DS=30, win=11, step=5, outdir=OUTDIR, tag="DS30")
        print(f"[OK] {outt}")


if __name__ == "__main__":
    main()

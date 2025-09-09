# expOne_Three_fast_final_stats.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import multiprocessing as mp
import argparse
import time
import os, csv
import matplotlib
matplotlib.use("Agg")  # 服务器/无显示环境安全存图
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import os, csv, time, itertools
import numpy as np
import multiprocessing as mp
import threading
from tqdm.auto import tqdm   # 用 auto 适配 PyCharm/Jupyter/终端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===== 线程控制（避免/利用多线程）=====
def set_blas_threads(n: int):
    """控制 NumPy/MKL/OpenBLAS/numexpr 的线程数。"""
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

def _worker_init_reduce_blas_threads():
    """子进程 initializer：把每个子进程的 BLAS 线程数钉为 1，避免过度并行竞争。"""
    set_blas_threads(1)

# ===== 策略编码 =====
C, D, L = 0, 1, 2  # Cooperator / Defector / Loner


# ===== 参数 =====
@dataclass
class Params:
    Lsize: int = 60
    T: int = 10000
    fc: float = 0.5
    fd: float = 0.5
    fL: Optional[float] = None  # None=仅CD；否则CDL
    pi0: float = 5.0
    r: float = 1.6
    sigma: float = 0.25
    alpha: float = 1.0
    gamma: float = 10.0
    beta: float = 0.1
    Min_Res: float = -10000.0
    kappa: float = 1
    learn_signal: str = "cumulative"  # 'round' / 'cumulative' / 'per_strategy'
    seed: Optional[int] = 0
    allow_forced_l_in_cd: bool = False
    # === 新增：更新规则 ===
    update_rule: str = "fermi"  # 可选: 'fermi' / 'best_rational' / 'best_imitation'
    # === 新增：混合学习 & 归一化设置 ===
    eta: float = 0.5              # 混合权重：η ∈ [0,1]；最终 S = (1-η)*minmax(round) + η*minmax(cum)
    scale_low: float = 0.0        # min–max 目标下界
    scale_high: float = 1.0       # min–max 目标上界
    scale_eps: float = 1e-9       # 极小跨度保护，防止除零
    burn_in_frac: float = 0.5  # ★ 新增：用于计算各类“*_mean”时跳过前 burn_in_frac*T 的回合
    # 输出/采样控制（可选；让 BASE 里也能统一配置）
    out_dir: Optional[str] = None  # None=不写；"AUTO"=runs/<ts>；也可自定义字符串
    log_every: int = -1  # -1≈自动200点；0=不写；>=1=指定步长
    snap_every: int = 0  # 0=用比例点(0.1/0.5/1.0)；>0=每 N 回合存一张快照图


# ===== 工具 =====


def build_payoff_matrix(include_loner: bool, r: float, sigma: float) -> np.ndarray:
    if include_loner:
        return np.array([
            [1.0    , 0.0   , 0.0   ],
            [r      , 0.0   , 0.0   ],
            [sigma  , sigma , sigma ],
        ], dtype=float)
    else:
        return np.array([
            [1.0, 0.0],
            [r, 0.0],
        ], dtype=float)


def gini_allow_negative(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0)
    mean_abs = np.mean(np.abs(x))
    if mean_abs < 1e-12:
        return 0.0
    n = x.size
    diff_sum = np.sum(np.abs(x[:, None] - x[None, :]))
    return diff_sum / (2 * n * n * mean_abs)

def _progress_consumer_thread(q: "mp.Queue", total_ticks: int, desc: str = "Sweep (config×replica)"):
    """
    从队列 q 中持续读取“1”并累加到 tqdm 进度。
    预期收到 total_ticks 次 put(1) 后自动结束。
    """
    done = 0
    pbar = tqdm(total=total_ticks, desc=desc, dynamic_ncols=True, mininterval=0.2, smoothing=0)
    while done < total_ticks:
        try:
            q.get()  # 阻塞直到收到一个心跳
            done += 1
            pbar.update(1)
        except Exception:
            # 极端情况下（比如主进程关闭）直接跳出
            break
    pbar.close()

# ====== 向量化邻居工具（周期边界）======
def rolled(arr: np.ndarray):
    # 返回上、下、左、右四个“邻居阵”
    up = np.roll(arr, -1, axis=0)
    down = np.roll(arr, 1, axis=0)
    left = np.roll(arr, -1, axis=1)
    right = np.roll(arr, 1, axis=1)
    return up, down, left, right

# ★★★ 新增：四邻边类型统计 + D 周长（用于 CSV / 归因） ★★★
def edge_stats_four_neigh(S: np.ndarray):
    """
    四邻+环面。返回：
      counts: CC,CD,DD,CL,DL,LL,D_perim,E
      ratios: *_ratio（对 E 归一化）
    """
    right = np.roll(S, -1, axis=1)
    down  = np.roll(S, -1, axis=0)
    a1, b1 = S, right
    a2, b2 = S, down

    def cnt_pair(a,b,x,y):
        return ((a==x)&(b==y)).sum() + ((a==y)&(b==x)).sum()

    E = S.size * 2

    CC = ((a1==C)&(b1==C)).sum() + ((a2==C)&(b2==C)).sum()
    DD = ((a1==D)&(b1==D)).sum() + ((a2==D)&(b2==D)).sum()
    LL = ((a1==L)&(b1==L)).sum() + ((a2==L)&(b2==L)).sum()
    CD = cnt_pair(a1,b1,C,D) + cnt_pair(a2,b2,C,D)
    CL = cnt_pair(a1,b1,C,L) + cnt_pair(a2,b2,C,L)
    DL = cnt_pair(a1,b1,D,L) + cnt_pair(a2,b2,D,L)

    D_perim = CD + DL
    counts = dict(CC=CC, CD=CD, DD=DD, CL=CL, DL=DL, LL=LL, D_perim=D_perim, E=E)
    ratios = {k+"_ratio": counts[k]/E for k in ["CC","CD","DD","CL","DL","LL","D_perim"]}
    return {**counts, **ratios}
# ====== 主仿真（向量化 + 无绘图版）======
class SpatialGameFast:
    def __init__(
            self,
            p: Params,
            out_dir: str = None,  # 新：输出目录（若为 None 则不写文件）
            log_every: int = 0,  # 新：>0 时记录时间序列
            snapshots=(0.1, 0.5, 1.0),  # 新：相对时间点，用于出图
            save_plots: bool = False  # 新：是否保存分布图
    ):
        self.p = p
        if p.seed is not None:
            np.random.seed(p.seed)
        self.L = p.Lsize
        self.include_loner = bool(p.allow_forced_l_in_cd)
        if not self.include_loner:
            self.p.fL = None
        self.M = build_payoff_matrix(self.include_loner, p.r, p.sigma)
        self.strat = self._init_strategies()
        self.res = np.full((self.L, self.L), fill_value=p.pi0, dtype=float)
        self.prev_strat = self.strat.copy()
        self.just_forced_L = np.zeros((self.L, self.L), dtype=bool)
        self.cum_sigma = 0.0   #维持Loner所要消耗的资源
        self.cum_output = 0.0  # 累计产出值（仅 C&D）
        self.last_per_sigma = 0.0
        self.last_per_output = 0.0

        # ===== 新增：输出控制 =====
        self.out_dir = out_dir
        self.log_every = max(0, int(log_every))
        self.save_plots = bool(save_plots)
        # 将相对快照点换算为绝对回合编号，去重并排序
        snaps = set()
        for s in (snapshots or []):
            if isinstance(s, float):
                t = int(round(self.p.T * s))
            else:
                t = int(s)
            if 1 <= t <= self.p.T:
                snaps.add(t)
        # 结尾一定加上 T（最后一帧）
        snaps.add(self.p.T)
        self.snap_steps = sorted(snaps)

        # 时间序列缓存
        self._ts = []  # 每项：(t, frac_C, frac_D, frac_L, avg_res, gini, cum_sigma_hat, cum_output_hat, per_sigma, per_output, per_net, cum_net_hat)
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        # ★新增：snapshot.csv writer（按 log_every 写，避免文件过大）
        self._snapshot_writer = None
        self._snapshot_file = None
        if self.out_dir and self.log_every > 0:
            snap_path = os.path.join(self.out_dir, "snapshot.csv")
            self._snapshot_file = open(snap_path, "w", newline="", encoding="utf-8")
            fields = [
                "t","frac_C","frac_D","frac_L","avg_res","gini",
                "per_sigma","per_output","per_net","cum_net_hat",
                "edges_CC","edges_CD","edges_DD","edges_CL","edges_DL","edges_LL","edges_total",
                "edges_CC_ratio","edges_CD_ratio","edges_DD_ratio","edges_CL_ratio","edges_DL_ratio","edges_LL_ratio",
                "D_perimeter","D_perimeter_ratio"
            ]
            self._snapshot_writer = csv.DictWriter(self._snapshot_file, fieldnames=fields)
            self._snapshot_writer.writeheader()

        # ★新增：后半程均值累加器（burn-in=0.5T）
        self.E_per_round = 2 * (self.L ** 2)
        self.edge_acc = {k: 0 for k in ["CC","CD","DD","CL","DL","LL","D_perim"]}
        self.edge_samples = 0
        # 原：self.burn_in = int(0.5 * self.p.T)
        self.burn_in = int(max(0.0, min(1.0, getattr(self.p, "burn_in_frac", 0.5))) * self.p.T)  # ★

    # === 新增：把关键参数拼成短标签，用于标题/文件名 ===
    def _short_name(self, base_name: str) -> str:
        """
        Windows 路径过长时，生成短文件名；并把完整标签写到一个 .tag.txt 旁注里。
        """
        import hashlib
        p = self.p
        digest = hashlib.md5(base_name.encode("utf-8")).hexdigest()[:8]
        short = f"trend_CDL_L{p.Lsize}_T{p.T}_r{p.r}_pi{p.pi0}_k{p.kappa}_A{int(bool(p.allow_forced_l_in_cd))}_{digest}.png"
        return short

    def _safe_savefig(self, fig, base_name: str):
        # 确保目录存在（即使上游建过，这里再稳一手）
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
        save_path = os.path.join(self.out_dir, base_name)
        try:
            fig.savefig(save_path, bbox_inches="tight")
            return save_path
        except OSError as e:
            # Windows 超长路径兜底：换短文件名 + 写旁注
            short = self._short_name(base_name)
            short_path = os.path.join(self.out_dir, short)
            fig.savefig(short_path, bbox_inches="tight")
            note = short_path[:-4] + ".tag.txt"
            with open(note, "w", encoding="utf-8") as f:
                f.write(base_name + "\n")
            return short_path

    def _param_tag(self) -> str:
        p = self.p
        tag = (
            f"L={p.Lsize}_T={p.T}_pi0={p.pi0}_r={p.r}_sigma={p.sigma}_"
            f"alpha={p.alpha}_gamma={p.gamma}_beta={p.beta}_kappa={p.kappa}_"
            f"learn={p.learn_signal}_update={p.update_rule}_seed={p.seed}_"
            f"allowL={int(bool(p.allow_forced_l_in_cd))}"
        )
        return tag

    # === 新增：绘制 CDL 变化曲线（以及可选的 avg_res / gini） ===
    def _plot_trend_image(self):
        if not (self.out_dir and self._ts):
            return
        import matplotlib.pyplot as plt

        arr = np.asarray(self._ts, dtype=float)
        t = arr[:, 0]
        frac_C, frac_D, frac_L = arr[:, 1], arr[:, 2], arr[:, 3]
        avg_res, gini = arr[:, 4], arr[:, 5]

        fig = plt.figure(figsize=(8.8, 5.0), dpi=140)
        ax = plt.gca()

        ax.plot(t, frac_C, label="C (Cooperator)", linewidth=2)
        ax.plot(t, frac_D, label="D (Defector)", linewidth=2)
        ax.plot(t, frac_L, label="L (Loner)", linewidth=2)

        ax.set_xlabel("t (round)")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, frameon=False)

        # 右侧叠加 avg_res（更直观对照资源变化）
        ax2 = ax.twinx()
        ax2.plot(t, avg_res, linestyle="--", linewidth=1.5)
        ax2.set_ylabel("avg_res (right)")
        # 也可按需再叠加 gini：把下一行注释取消即可
        # ax2.plot(t, gini, linestyle=":", linewidth=1.5)

        # 标题含参数摘要
        tag = self._param_tag()
        plt.title(f"CDL Trend & avg_res  |  {tag}")
        plt.gcf().text(0.01, 0.99, tag, ha="left", va="top", fontsize=8, alpha=0.8)#保存名字

        # 输出路径：含参数摘要，便于 sweep 后检索
        fname = f"trend_CDL_{tag}.png".replace("/", "_")
        plt.tight_layout()
        self._safe_savefig(fig, fname)
        plt.close(fig)


    # === 小封装：在写完 CSV 后顺手出图 ===
    def _dump_trend_png(self):
        try:
            self._plot_trend_image()
        except Exception as e:
            # 出图失败不影响主流程（例如无 GUI 环境）
            print(f"[warn] plot trend failed: {e}")

    def _fractions(self):
        frac_C = float(np.mean(self.strat == C))
        frac_D = float(np.mean(self.strat == D))
        frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
        return frac_C, frac_D, frac_L

    # ★修改：新增 es（边统计）参数与 snapshot.csv 写入、均值累加
    def _record_trend(self, t, per_sigma_t=0.0, per_output_t=0.0, per_net_t=0.0, es: dict = None):
        frac_C, frac_D, frac_L = self._fractions()
        avg_res = float(np.mean(self.res))
        gini = gini_allow_negative(self.res)

        N = self.L * self.L
        if t > 0:
            cum_sigma_hat_t = (self.cum_sigma / (N * t)) if self.include_loner else 0.0
            cum_output_hat_t = self.cum_output / (N * t)
            cum_net_hat_t = (self.cum_output - self.cum_sigma) / (N * t)
        else:
            cum_sigma_hat_t = 0.0
            cum_output_hat_t = 0.0
            cum_net_hat_t = 0.0

        # 原时间序列缓存
        if self.log_every > 0 and (t == 0 or t % self.log_every == 0 or t == self.p.T):
            self._ts.append((
                t, frac_C, frac_D, frac_L,
                avg_res, gini,
                cum_sigma_hat_t, cum_output_hat_t,  # 归一化累计
                per_sigma_t, per_output_t, per_net_t,  # 每回合
                cum_net_hat_t  # 累计净产出（每人每回合）
            ))

        # snapshot.csv（含边统计）
        if self._snapshot_writer and (t == 0 or t % self.log_every == 0 or t == self.p.T):
            if es is None:
                es = edge_stats_four_neigh(self.strat)
            row = {
                "t": t,
                "frac_C": frac_C, "frac_D": frac_D, "frac_L": frac_L,
                "avg_res": avg_res, "gini": gini,
                "per_sigma": per_sigma_t, "per_output": per_output_t, "per_net": per_net_t,
                "cum_net_hat": cum_net_hat_t,
                "edges_CC": es["CC"], "edges_CD": es["CD"], "edges_DD": es["DD"],
                "edges_CL": es["CL"], "edges_DL": es["DL"], "edges_LL": es["LL"],
                "edges_total": es["E"],
                "edges_CC_ratio": es["CC_ratio"], "edges_CD_ratio": es["CD_ratio"],
                "edges_DD_ratio": es["DD_ratio"], "edges_CL_ratio": es["CL_ratio"],
                "edges_DL_ratio": es["DL_ratio"], "edges_LL_ratio": es["LL_ratio"],
                "D_perimeter": es["D_perim"], "D_perimeter_ratio": es["D_perim_ratio"],
            }
            self._snapshot_writer.writerow(row)

        # 后半程均值累加
        if es is None:
            es = edge_stats_four_neigh(self.strat)
        if t >= self.burn_in:
            for k in ["CC","CD","DD","CL","DL","LL","D_perim"]:
                self.edge_acc[k] += es[k]
            self.edge_samples += 1

    def _plot_strategy_map(self, t, fname):
        # 颜色：C=蓝，D=红，L=灰（如果没启 L，L 不会出现）
        cmap = ListedColormap(["#1f77b4", "#d62728", "#FFFF00"])
        plt.figure(figsize=(5.4, 5.4), dpi=120)
        plt.imshow(self.strat, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        plt.title(f"Strategy map @ t={t}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

    def _plot_resource_map(self, t, fname):
        arr = np.asarray(self.res, dtype=float)

        # 处理 NaN/Inf
        if not np.isfinite(arr).all():
            if np.isfinite(arr).any():
                arr = np.nan_to_num(arr, nan=0.0,
                                    posinf=np.max(arr[np.isfinite(arr)]),
                                    neginf=np.min(arr[np.isfinite(arr)]))
            else:
                arr = np.zeros_like(arr)

        # 稳健范围：5/95 分位
        vmin = float(np.nanpercentile(arr, 5))
        vmax = float(np.nanpercentile(arr, 95))

        # 兜底：极端情况下 5/95 相等，用整体最小/最大
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = float(np.min(arr))
            vmax = float(np.max(arr))

        # —— 关键改动 1：近乎常数时，给一个更宽的可视化窗口（例如 center±max(1, 5%)）——
        if np.isclose(vmin, vmax, rtol=0.0, atol=1e-9):
            center = 0.5 * (vmin + vmax)
            span = max(1.0, abs(center) * 0.05)  # 至少 ±1，也可按数据量级放大 5%
            vmin, vmax = center - span, center + span

        crosses_zero = (vmin < 0.0) and (vmax > 0.0)

        plt.figure(figsize=(5.8, 5.4), dpi=120)
        if crosses_zero:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            im = plt.imshow(arr, norm=norm, aspect="equal")
        else:
            im = plt.imshow(arr, vmin=vmin, vmax=vmax, aspect="equal")

        # —— 关键改动 2：禁用 colorbar 的 offset，显示绝对数值 ——
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        sf = ScalarFormatter(useOffset=False)
        sf.set_scientific(False)
        cb.formatter = sf
        cb.ax.yaxis.get_offset_text().set_visible(False)
        cb.update_ticks()

        # —— 关键改动 3：把真实 min/mean/max 打在标题里，便于和 CSV 对照 ——
        real_min = float(np.min(arr))
        real_mu = float(np.mean(arr))
        real_max = float(np.max(arr))
        plt.title(f"Resource map @ t={t}  (min={real_min:.2f}, mean={real_mu:.2f}, max={real_max:.2f})")

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fname, bbox_inches="tight")
        plt.close('all')


    def _maybe_snapshot_plots(self, t):
        if not (self.save_plots and self.out_dir):
            return
        if t in self.snap_steps:
            sp = os.path.join(self.out_dir, f"strat_t{t:06d}.png")
            rp = os.path.join(self.out_dir, f"res_t{t:06d}.png")
            self._plot_strategy_map(t, sp)
            self._plot_resource_map(t, rp)

    def _dump_trend_csv(self):
        if not (self.out_dir and self._ts):
            return
        csv_path = os.path.join(self.out_dir, "trend.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "t", "frac_C", "frac_D", "frac_L",
                "avg_res", "gini",
                "cum_sigma_hat", "cum_output_hat",
                "per_sigma", "per_output", "per_net",
                "cum_net_hat"
            ])
            w.writerows(self._ts)

    def _init_strategies(self) -> np.ndarray:
        L2 = self.L * self.L
        if self.include_loner and (self.p.fL is not None):
            assert abs(self.p.fc + self.p.fd + self.p.fL - 1.0) < 1e-8, "fc+fd+fL must be 1."
            cN = int(self.p.fc * L2)
            dN = int(self.p.fd * L2)
            lN = L2 - cN - dN
            pool = np.array([C] * cN + [D] * dN + [L] * lN, dtype=np.uint8)  # ← 改这里
        else:
            assert abs(self.p.fc + self.p.fd - 1.0) < 1e-8, "fc+fd must be 1."
            cN = int(self.p.fc * L2)
            dN = L2 - cN
            pool = np.array([C] * cN + [D] * dN, dtype=np.uint8)  # ← 改这里
        np.random.shuffle(pool)
        return pool.reshape(self.L, self.L)

    # ---- (1) 本回合收益：完全向量化 ----
    def _round_payoff_vec(self, strat: np.ndarray) -> np.ndarray:
        up, down, left, right = rolled(strat)
        # 利用收益矩阵做整阵索引
        payoff = (self.M[strat, up] +
                  self.M[strat, down] +
                  self.M[strat, left] +
                  self.M[strat, right])
        return payoff

    # ---- (2) 资源消耗：阈值规则（向量化）----
    def _consume_rule(self, res_after_gain: np.ndarray) -> np.ndarray:
        low_mask = (res_after_gain <= self.p.gamma)
        res_after_consume = np.where(low_mask,
                                     res_after_gain - self.p.alpha,
                                     res_after_gain - self.p.alpha- (res_after_gain - self.p.gamma) * self.p.beta
                                    )
        # 软下限：若会低于 Min_Res，则回退到旧值（在 caller 里做）
        return res_after_consume

    def _minmax_scale(self, X: np.ndarray, low: float = None, high: float = None) -> np.ndarray:
        if low is None:  low = self.p.scale_low
        if high is None: high = self.p.scale_high
        X = np.asarray(X, dtype=float)
        xmin = float(np.min(X));
        xmax = float(np.max(X))
        span = xmax - xmin
        if not np.isfinite(xmin) or not np.isfinite(xmax) or span < self.p.scale_eps:
            return np.zeros_like(X, dtype=float)  # 几乎常数：统一用 0，费米/Logit 比较时相当于无优势
        Z = (X - xmin) / span
        return Z * (high - low) + low

    def _build_signal(self, round_payoff: np.ndarray) -> np.ndarray:
        # 统一入口：根据 learn_signal 产出 S
        if self.p.learn_signal == "round":
            return round_payoff
        if self.p.learn_signal == "cumulative":
            return self.res
        if self.p.learn_signal == "per_strategy":
            S = np.empty_like(self.res)
            S[self.strat == C] = self.res[self.strat == C]
            S[self.strat == D] = round_payoff[self.strat == D]
            maskL = (self.strat == L)
            if np.any(maskL):
                S[maskL] = 0.5 * (self.res[maskL] + round_payoff[maskL])
            return S
        if self.p.learn_signal == "hybrid":
            # 关键：先各自min–max到同一量纲，再凸组合
            Sr = self._minmax_scale(round_payoff)
            Sc = self._minmax_scale(self.res)
            eta = float(np.clip(self.p.eta, 0.0, 1.0))
            return (1.0 - eta) * Sr + eta * Sc
        else:

            raise ValueError("learn_signal must be one of "
                         "'round','cumulative','per_strategy','hybrid'")

    # ---- (3) 费米更新：一次性随机方向并向量化 ----
    def _fermi_update_vec(self, S: np.ndarray):
        kappa = max(self.p.kappa, 1e-8)
        # 预先取得邻居的策略与 S
        s = self.strat
        S_up, S_down, S_left, S_right = rolled(S)
        s_up, s_down, s_left, s_right = rolled(s)

        # 每个格子随机选择一个邻居方向（0=up,1=down,2=left,3=right）
        dir_choice = np.random.randint(0, 4, size=s.shape)

        # 根据 dir_choice 选择邻居 S 与 邻居策略
        neighbor_S = np.where(
            dir_choice == 0, S_up,
            np.where(dir_choice == 1, S_down,
                     np.where(dir_choice == 2, S_left, S_right))
        )
        neighbor_s = np.where(
            dir_choice == 0, s_up,
            np.where(dir_choice == 1, s_down,
                     np.where(dir_choice == 2, s_left, s_right))
        )

        # 刚被强制 L 的细胞：本回合冻结为 L，不参与模仿
        frozen = self.just_forced_L
        # 费米采纳概率：p = 1/(1+exp((Si - Sj)/kappa))
        delta = np.clip(S - neighbor_S, -50.0, 50.0)
        adopt_p = 1.0 / (1.0 + np.exp(delta / kappa))
        randu = np.random.rand(*s.shape)
        adopt = (randu < adopt_p) & (~frozen)

        # 生成 next_strat
        next_s = np.where(adopt, neighbor_s, s)
        # 对冻结位置强制设为 L
        next_s = np.where(frozen, L, next_s)
        self.strat = next_s

    def _best_imitation_logit_update_vec(self, S: np.ndarray):
        """
        概率版：先找 payoff 最高的邻居 j*，再以
        p = 1/(1+exp((Si - Sj*)/kappa)) 的概率模仿其策略；否则保持原策略。
        冻结的格点维持为 L。
        """
        kappa = max(self.p.kappa, 1e-12)

        s = self.strat
        S_up, S_down, S_left, S_right = rolled(S)
        s_up, s_down, s_left, s_right = rolled(s)

        neigh_payoffs = np.stack([S_up, S_down, S_left, S_right], axis=0)  # (4,L,L)
        neigh_strats = np.stack([s_up, s_down, s_left, s_right], axis=0)  # (4,L,L)

        # tie-break 的极小噪声，避免整片并列
        eps = 1e-9
        noisy = neigh_payoffs + eps * np.random.rand(*neigh_payoffs.shape)
        argmax_idx = np.argmax(noisy, axis=0)  # (L,L)

        rows = np.arange(self.L)[:, None]
        cols = np.arange(self.L)[None, :]
        S_star = neigh_payoffs[argmax_idx, rows, cols]  # Sj*
        strat_star = neigh_strats[argmax_idx, rows, cols]  # 邻居策略

        Si = S  # 当前自身 payoff
        delta = np.clip(Si - S_star, -50.0, 50.0)
        adopt_p = 1.0 / (1.0 + np.exp(delta / kappa))

        randu = np.random.rand(*s.shape)
        adopt = (randu < adopt_p) & (~self.just_forced_L)  # 冻结不变

        next_s = np.where(adopt, strat_star, s)
        # 冻结位置强制为 L
        next_s = np.where(self.just_forced_L, L, next_s)
        self.strat = next_s

    def _best_rational_logit_update_vec(self):
        """
        概率版：理性 BR 的 Logit 采纳。
        - 先对每个候选动作 a 计算对四邻居的总收益 U[a]。
        - 记当前动作的预测收益 U_i、最优动作及其收益 U*。
        - 以 p = 1/(1+exp((U_i - U*)/kappa)) 的概率切换到最优动作；否则保持不变。
        冻结的格点维持为 L。
        """
        kappa = max(self.p.kappa, 1e-12)

        up, down, left, right = rolled(self.strat)
        A = 3 if self.include_loner else 2

        # U[a] for all a
        payoffs = []
        for a in range(A):
            pa = (self.M[a, up] + self.M[a, down] + self.M[a, left] + self.M[a, right])
            payoffs.append(pa)
        U = np.stack(payoffs, axis=0)  # (A,L,L)

        # 最优动作 a*
        eps = 1e-9
        argmax_a = np.argmax(U + eps * np.random.rand(*U.shape), axis=0)  # (L,L)

        rows = np.arange(self.L)[:, None]
        cols = np.arange(self.L)[None, :]

        # 当前动作的预测收益 U_i（按当前策略索引）
        Ui = U[self.strat, rows, cols]
        # 最优动作收益 U*
        Ustar = U[argmax_a, rows, cols]

        delta = np.clip(Ui - Ustar, -50.0, 50.0)
        adopt_p = 1.0 / (1.0 + np.exp(delta / kappa))
        randu = np.random.rand(self.L, self.L)
        adopt = (randu < adopt_p) & (~self.just_forced_L)

        next_s = np.where(adopt, argmax_a, self.strat)
        next_s = np.where(self.just_forced_L, L, next_s)
        self.strat = next_s

    def _best_response_update_vec(self):
        """
        同步 Best Response：
        - 对每个单元，枚举可用策略 a∈{C,D,(L)}，
          计算若选 a 与上下左右邻居对弈的总收益 sum_j M[a, s_j]；
        - 选取收益最大的策略；并对并列最优用极小随机扰动打破平局（保持可复现）；
        - 刚被强制转 L 的位置本回合冻结为 L。
        """
        # 邻居策略
        up, down, left, right = rolled(self.strat)

        # 可用动作数：仅当允许 L 时，A=3；否则 A=2
        A = 3 if self.include_loner else 2

        # 计算每个候选动作的总收益
        # payoffs 形状: (A, L, L)
        payoffs = []
        for a in range(A):
            pa = (self.M[a, up] + self.M[a, down] + self.M[a, left] + self.M[a, right])
            payoffs.append(pa)
        P = np.stack(payoffs, axis=0)

        # 为避免完全相等时出现大块同步翻转，引入极小噪声做随机tie-break（种子可控）
        eps = 1e-9
        P_noisy = P + eps * np.random.rand(*P.shape)

        # 选最优动作
        best = np.argmax(P_noisy, axis=0).astype(np.int64)   # (L,L)，取值 in {0,1,(2)}

        # 冻结：刚强制转 L 的位置本回合固定为 L
        if self.include_loner:
            best = np.where(self.just_forced_L, L, best)
        self.strat = best

    def _best_rational_update_vec(self):
        """
        理性 Best Response：
        - 枚举所有候选动作 a，计算对四邻居的总收益 sum_j M[a, s_j]。
        - 选取最大收益的动作。
        """
        up, down, left, right = rolled(self.strat)
        A = 3 if self.include_loner else 2
        payoffs = []
        for a in range(A):
            pa = (self.M[a, up] + self.M[a, down] +
                  self.M[a, left] + self.M[a, right])
            payoffs.append(pa)
        P = np.stack(payoffs, axis=0)

        eps = 1e-9
        P_noisy = P + eps * np.random.rand(*P.shape)
        best = np.argmax(P_noisy, axis=0).astype(np.int64)

        if self.include_loner:
            best = np.where(self.just_forced_L, L, best)
        self.strat = best

    def _best_imitation_update_vec(self, S: np.ndarray):
        """
        模仿型 Best Response：
        - 查看四个邻居的 payoff S。
        - 模仿 payoff 最大的邻居的策略。
        """
        s = self.strat
        S_up, S_down, S_left, S_right = rolled(S)
        s_up, s_down, s_left, s_right = rolled(s)

        neigh_payoffs = np.stack([S_up, S_down, S_left, S_right], axis=0)
        neigh_strats  = np.stack([s_up, s_down, s_left, s_right], axis=0)

        eps = 1e-9
        noisy_payoffs = neigh_payoffs + eps * np.random.rand(*neigh_payoffs.shape)
        max_idx = np.argmax(noisy_payoffs, axis=0)

        # 按 max_idx 取对应策略
        rows = np.arange(self.L)[:, None]
        cols = np.arange(self.L)[None, :]
        best_neighbor_strat = neigh_strats[max_idx, rows, cols]

        next_s = np.where(self.just_forced_L, L, best_neighbor_strat)
        self.strat = next_s


    def _force_loners_if_broke(self):
        """
        若上一回合不是 L 且当前资源<0：
        - 当 allow_forced_l_in_cd=True：转为 L，并在本回合冻结为 L；
        - 当 allow_forced_l_in_cd=False：不改变策略；
        只有在Loner允许的情况下，当前破产者资源均才会被钉为 0
        """
        self.just_forced_L[:] = False
        broke_now = (self.res < -0.1)

        if not np.any(broke_now):
            return

        # 仅针对“上一回合不是 L”的个体
        need_force = broke_now & (self.prev_strat != L)

        if self.p.allow_forced_l_in_cd and self.include_loner:
            # 允许强制转 L
            if np.any(need_force):
                self.strat[need_force] = L
                self.just_forced_L[need_force] = True
                self.res[need_force] = 0.0     # ← 只清这些位置，保持 self.res 为 2D 数组

    def _plot_extra_time_series(self):
        if not (self.out_dir and self._ts):
            return
        arr = np.asarray(self._ts, dtype=float)
        t = arr[:, 0]

        per_sigma = arr[:, 8]
        per_output = arr[:, 9]
        per_net = arr[:, 10]
        cum_net_hat = arr[:, 11]

        # 1) Loner 每回合消耗
        fig = plt.figure(figsize=(8.4, 4.4), dpi=140);
        ax = plt.gca()
        ax.plot(t, per_sigma, linewidth=1.6, label="per-round σ-consumption (L)")
        ax.set_xlabel("t");
        ax.set_ylabel("σ per round");
        ax.grid(True, alpha=0.3);
        ax.legend()
        self._safe_savefig(fig, "timeseries_per_sigma.png");
        plt.close(fig)

        # 2) 每回合净收益（产出-补贴）
        fig = plt.figure(figsize=(8.4, 4.4), dpi=140);
        ax = plt.gca()
        ax.plot(t, per_net, linewidth=1.6, label="per-round net = output - σ")
        ax.set_xlabel("t");
        ax.set_ylabel("net per round");
        ax.grid(True, alpha=0.3);
        ax.legend()
        self._safe_savefig(fig, "timeseries_per_net.png");
        plt.close(fig)

        # 3) 累计净产出（每人每回合，便于不同 L/T 可比）
        fig = plt.figure(figsize=(8.4, 4.4), dpi=140);
        ax = plt.gca()
        ax.plot(t[1:], cum_net_hat[1:], linewidth=1.8, label="cumulative net (per-capita per-round)")
        ax.set_xlabel("t");
        ax.set_ylabel("cum net hat");
        ax.grid(True, alpha=0.3);
        ax.legend()
        self._safe_savefig(fig, "timeseries_cum_net_hat.png");
        plt.close(fig)

    # ★新增：把边均值整理成汇总字典
    def _edge_means_for_summary(self) -> Dict[str, float]:
        n = max(1, self.edge_samples)
        E = self.E_per_round
        mean_ratio = {f"edges_{k}_mean_ratio": self.edge_acc[k] / (E * n)
                      for k in ["CC","CD","DD","CL","DL","LL"]}
        return {
            **mean_ratio,
            "D_perimeter_mean": self.edge_acc["D_perim"] / n,
            "D_perimeter_mean_ratio": self.edge_acc["D_perim"] / (E * n),
        }

    def run(self) -> Dict[str, float]:
        # —— 新增：t=0 的初始点 ——
        es0 = edge_stats_four_neigh(self.strat)  # ★新增
        self._record_trend(0, 0.0, 0.0, 0.0, es=es0)
        self._maybe_snapshot_plots(0)

        for t in range(1, self.p.T + 1):
            res_old = self.res.copy()
            self.prev_strat = self.strat.copy()

            # (1) 收益
            round_payoff = self._round_payoff_vec(self.strat)

            # === 当回合“产出/补贴/净收益” ===
            mask_CD = (self.strat == C) | (self.strat == D)
            per_output_t = float(np.sum(round_payoff[mask_CD]))
            if self.include_loner:
                nL = int(np.sum(self.strat == L))
                per_sigma_t = nL * 4.0 * self.p.sigma
            else:
                per_sigma_t = 0.0
            per_net_t = per_output_t - per_sigma_t

            # === 累计器 ===
            self.cum_output += per_output_t
            self.cum_sigma += per_sigma_t

            res_after_gain = res_old + round_payoff

            # (2) 消耗
            res_after_consume = self._consume_rule(res_after_gain)
            self.res = np.where(res_after_consume < self.p.Min_Res, res_old, res_after_consume)

            # (3) 破产转 L
            self._force_loners_if_broke()

            # (4) 学习信号（统一入口，支持 hybrid）
            S = self._build_signal(round_payoff)

            # (5) 策略更新
            if self.p.update_rule == "fermi":
                self._fermi_update_vec(S)
            elif self.p.update_rule == "best_rational":
                self._best_rational_update_vec()
            elif self.p.update_rule == "best_imitation":
                self._best_imitation_update_vec(S)
            elif self.p.update_rule == "best_imitation_logit":
                self._best_imitation_logit_update_vec(S)
            elif self.p.update_rule == "best_rational_logit":
                self._best_rational_logit_update_vec()
            else:
                raise ValueError("update_rule must be one of: "
                                 "'fermi','best_rational','best_imitation',"
                                 "'best_imitation_logit','best_rational_logit'")

            # —— 记录与快照（含边统计） ——
            es_t = edge_stats_four_neigh(self.strat)  # ★新增
            self._record_trend(t, per_sigma_t, per_output_t, per_net_t, es=es_t)
            self._maybe_snapshot_plots(t)

        # 结束统计
        frac_C = float(np.mean(self.strat == C))
        frac_D = float(np.mean(self.strat == D))
        frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
        avg_res = float(np.mean(self.res))
        sum_res = float(np.sum(self.res))
        gini = gini_allow_negative(self.res)
        cum_sigma = float(self.cum_sigma)
        cum_output = float(self.cum_output)
        net_output = cum_output - cum_sigma  # 新定义：净产出

        # === 新增：归一化（每格每回合） ===
        N = self.L * self.L
        T = self.p.T
        cum_sigma_hat = (cum_sigma / (N * T)) if self.include_loner else 0.0
        net_output_hat = net_output / (N * T)
        cum_output_hat = self.cum_output / (N * T)
        # ==================================

        # —— 关闭 snapshot.csv 文件句柄（若有）——
        if self._snapshot_file is not None:
            try:
                self._snapshot_file.close()
            except Exception:
                pass

        # —— 新增：保存时间序列 ——
        self._dump_trend_csv()
        # —— 新增：绘制 CDL 曲线图 ——
        self._dump_trend_png()

        self._plot_extra_time_series()

        # ★新增：将边类型/周长的“后半程均值”汇总进返回值
        edge_summary = self._edge_means_for_summary()

        return {
            "frac_C": frac_C, "frac_D": frac_D, "frac_L": frac_L,
            "avg_res": avg_res, "sum_res": sum_res, "gini": gini,
            "cum_sigma": cum_sigma,
            "cum_output": cum_output,
            "net_output": net_output,  # 新指标：累计产出 - 补贴
            "cum_sigma_hat": float(cum_sigma / (N * T)),
            "cum_output_hat": float(cum_output / (N * T)),
            "net_output_hat": float(net_output / (N * T)),
            **edge_summary,  # ★新增：edges_*_mean_ratio, D_perimeter_mean(_ratio)
        }


# ====== 单次运行的便捷函数（给并行用）======
def run_once(p: Params) -> Dict[str, float]:
    sim = SpatialGameFast(p)
    return sim.run()


# ====== CLI & 并行入口 ======
def main():
    parser = argparse.ArgumentParser(description="Fast spatial game (final-only stats).")
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument("--fc", type=float, default=0.5)
    parser.add_argument("--fd", type=float, default=0.5)
    parser.add_argument("--fL", type=float, default=-1.0, help="-1 表示仅CD；>=0表示CDL的L比例")
    parser.add_argument("--pi0", type=float, default=5.0)
    parser.add_argument("--r", type=float, default=1.6)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.5, help="混合学习权重 η，0~1：η*round + (1-η)*cumulative（均为min–max后）")
    parser.add_argument("--minres", type=float, default=-10000.0)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--learn", type=str, default="cumulative",
                        choices=["round", "cumulative", "per_strategy", "hybrid"])
    parser.add_argument(
        "--update", type=str, default="fermi",
        choices=["fermi", "best_rational", "best_imitation",
                 "best_imitation_logit", "best_rational_logit"],
        help="更新规则"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_forced_l", action="store_true", help="仅CD起始时也允许破产转L")
    parser.add_argument("--runs", type=int, default=1, help="并行独立重复次数（不同随机种子）")
    parser.add_argument("--burn", type=float, default=0.5,
                        help="burn-in fraction for summary means (0~1). 0=不用烧入，1=只看最后一个回合")
    parser.add_argument("--out", type=str, default="AUTO",
                        help="输出目录。AUTO=自动 runs/<时间戳>；空串\"\"或none=不写CSV/图；其它=自定义目录")
    parser.add_argument("--log-every", type=int, default=-1,
                        help="每多少回合写一行 snapshot/trend。-1=自动约200点；0=不写；>=1=指定步长")
    parser.add_argument("--snap-every", type=int, default=0,
                        help="每多少回合保存一次快照图（0=用比例点 0.1/0.5/1.0）")

    args = parser.parse_args()

    p = Params(
        Lsize=args.L, T=args.T, fc=args.fc, fd=args.fd,
        fL=None if args.fL < 0 else args.fL,
        pi0=args.pi0, r=args.r, sigma=args.sigma,
        alpha=args.alpha, gamma=args.gamma, beta=args.beta,
        Min_Res=args.minres, kappa=args.kappa,
        learn_signal=args.learn,
        eta=args.eta,
        seed=args.seed,
        allow_forced_l_in_cd=args.allow_forced_l,
        update_rule=args.update,   # ← 新增
        burn_in_frac=args.burn,
    )
    # === 输出目录 ===
    if args.out is None:
        out_dir = None
    else:
        out_str = str(args.out).strip()
        if out_str == "" or out_str.lower() == "none":
            out_dir = None
        elif out_str == "AUTO":
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join("runs", ts)
        else:
            out_dir = out_str

    # === 采样频率 ===
    if args.log_every < 0:
        log_every = (max(1, p.T // 200) if out_dir else 0)  # AUTO≈200个点
    else:
        log_every = int(args.log_every)

    # === 快照帧 ===
    if args.snap_every and args.snap_every > 0:
        snapshots = list(range(args.snap_every, p.T + 1, args.snap_every))
    else:
        snapshots = (0.1, 0.5, 1.0)

    # 根据“单次 or 并行”设置 BLAS 线程
    if args.runs == 1:
        # 单次长仿真：吃满 CPU
        set_blas_threads(os.cpu_count() or 1)
    else:
        # 并行：每个进程限制为 1 线程（真正的扩展在进程数上）
        set_blas_threads(1)

    if args.runs == 1:
         # —— 计算“生效”的输出设置（只用 CLI 解析得到的本地变量，避免依赖 BASE） ——
        out_dir_eff = out_dir
        log_every_eff = (log_every if log_every is not None else -1)

        if int(log_every_eff) < 0:
            log_every_eff = (max(1, p.T // 200) if out_dir_eff else 0)
        snapshots_eff = snapshots

        # 解析 out_dir（支持 "AUTO"/None/自定义路径）
        if out_dir_eff and str(out_dir_eff).lower() != "none":
            out_dir_real = (os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
                            if str(out_dir_eff).upper() == "AUTO" else str(out_dir_eff))
            os.makedirs(out_dir_real, exist_ok=True)
        else:
            out_dir_real = None

        sim = SpatialGameFast(
            p,
            out_dir=out_dir_real,
            log_every=int(log_every_eff),
            snapshots=snapshots_eff,
            save_plots=bool(out_dir_real)
        )

        t0 = time.time()
        out = sim.run()
        dt = time.time() - t0
        print(f"[Single run done in {dt:.2f}s]")
        print(
            f"Final @ T={p.T}: "
            f"C={out['frac_C']:.3f}  D={out['frac_D']:.3f}  L={out['frac_L']:.3f}  "
            f"avg_res={out['avg_res']:.3f}  sum_res={out['sum_res']:.3f}  gini={out['gini']:.3f}  "
            f"cum_sigma={out['cum_sigma']:.3f}  cum_output={out['cum_output']:.3f}  net_output={out['net_output']:.3f}"
        )
        if out_dir_real:
            print(f"Saved to: {out_dir_real}")






    else:

        # ===== 并行：只让 rep0 写 CSV/图，其余副本只算统计 =====

        N_PROCS = min(args.runs, os.cpu_count() or 1)

        results = []

        t0 = time.time()

        # —— 计算“生效”的输出/采样设置（与单次分支一致；不依赖 BASE） ——
        out_dir_eff = out_dir
        log_every_eff = (log_every if log_every is not None else -1)

        if int(log_every_eff) < 0:
            log_every_eff = (max(1, p.T // 200) if out_dir_eff else 0)
        snapshots_eff = snapshots

        # —— rep0：主进程内运行并写输出 ——

        if out_dir_eff and str(out_dir_eff).lower() != "none":

            out_dir_root = (os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))

                            if str(out_dir_eff).upper() == "AUTO" else str(out_dir_eff))

            rep0_dir = os.path.join(out_dir_root, "rep0")

            os.makedirs(rep0_dir, exist_ok=True)

        else:

            out_dir_root = None

            rep0_dir = None

        p0 = Params(**{**p.__dict__, "seed": (None if p.seed is None else p.seed)})

        sim0 = SpatialGameFast(

            p0,

            out_dir=rep0_dir,

            log_every=int(log_every_eff if rep0_dir else 0),

            snapshots=snapshots_eff,

            save_plots=bool(rep0_dir)

        )

        out0 = sim0.run()

        results.append(out0)

        # —— 其余副本：进程池计算，不写文件 ——

        jobs = []

        for k in range(1, args.runs):
            pk = Params(**{**p.__dict__, "seed": (None if p.seed is None else p.seed + k)})

            jobs.append(pk)

        if jobs:

            with mp.get_context("spawn").Pool(

                    processes=min(len(jobs), N_PROCS),

                    initializer=_worker_init_reduce_blas_threads,

                    maxtasksperchild=1

            ) as pool:

                for out in pool.imap_unordered(run_once, jobs, chunksize=1):
                    results.append(out)

        dt = time.time() - t0

        # ===== 聚合并打印 =====

        arrC = np.array([r["frac_C"] for r in results], dtype=float)

        arrD = np.array([r["frac_D"] for r in results], dtype=float)

        arrL = np.array([r["frac_L"] for r in results], dtype=float)

        arrAvg = np.array([r["avg_res"] for r in results], dtype=float)

        arrSum = np.array([r["sum_res"] for r in results], dtype=float)

        arrG = np.array([r["gini"] for r in results], dtype=float)

        arrCum = np.array([r["cum_sigma"] for r in results], dtype=float)

        arrNet = np.array([r["net_output"] for r in results], dtype=float)

        print(f"[{args.runs} runs done in {dt:.2f}s] mean±std:")

        print(f"C={arrC.mean():.3f}±{arrC.std():.3f}  D={arrD.mean():.3f}±{arrD.std():.3f}  "

              f"L={arrL.mean():.3f}±{arrL.std():.3f}")

        print(f"avg_res={arrAvg.mean():.3f}±{arrAvg.std():.3f}  "

              f"sum_res={arrSum.mean():.3f}±{arrSum.std():.3f}  "

              f"gini={arrG.mean():.3f}±{arrG.std():.3f}  "

              f"cum_sigma={arrCum.mean():.3f}±{arrCum.std():.3f}  "

              f"net_output={arrNet.mean():.3f}±{arrNet.std():.3f}")

        if rep0_dir:
            print(f"[rep0 outputs] saved to: {rep0_dir}")


# if __name__ == "__main__":
#     main()
#
# （下略：保留你原先的示例 main 与 sweep 注释）
# sweep_parallel.py
# -*- coding: utf-8 -*-
import os, csv, time, itertools
import numpy as np
import multiprocessing as mp


# ---- 1) 定义你想扫的参数网格（自行改动） ----
GRID = {
    #"r":                    [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "r":   [round(x, 2) for x in np.linspace(1.0, 2.0, 51)],
    "allow_forced_l_in_cd": [True, False],
    #"pi0":                  [ 5,  10],
    #"kappa":                [ 1 ],
    #"update_rule":          ['best_imitation_logit'] ,#"fermi","best_rational","best_imitation","best_imitation_logit","best_rational_logit"
    #"learn_signal":         [ 'cumulative' ,'round'],   #"round", "cumulative", "per_strategy", "hybrid"
    #"eta":                  [0, 0.2, 0.4, 0.6, 0.8, 1]          # η累计收益注重系数
    "eta": [round(x, 2) for x in np.linspace(0.0, 1.0, 51)],
    #"sigma":                [0.25,0.26,0.27,0.28,0.29,0.30]
    # 也可加入 "sigma": [0.2, 0.3], "allow_forced_l_in_cd": [False, True], ...
}
# 固定不变的公共参数
# 固定不变的公共参数（可在此统一调参）
BASE = Params(
    # --- 结构/时长 ---
    Lsize=50,            # 格子边长 L，环境大小 L x L（周期边界）
    T=50000,             # 演化总回合数

    # --- 初始策略配比（仅CD时用 fc, fd；CDL 用 fc, fd, fL）---
    fc=0.5,
    fd=0.5,
    fL=None,             # None=仅CD；给个数比如0.1时=CDL

    # --- 博弈参数 ---
    r=1.6,               # 背叛诱惑指数
    pi0=10.0,            # 初始资源 π0
    sigma=0.25,          # Loner 单边互动收益（每个邻边都拿 σ）
    alpha=1.0,           # 每回合固定消耗
    gamma=70.0,          # 资源阈值
    beta=0.1,            # 超阈后的比例消耗
    Min_Res=-10000.0,    # 资源软下限（跌破则回退）

    # --- 学习/更新规则 ---
    kappa=1,                             # 费米/Logit 温度
    learn_signal="hybrid",               # 'round' / 'cumulative' / 'per_strategy' / 'hybrid'
    eta=0.5,                             # hybrid 混合权重： (1-η)*minmax(round) + η*minmax(cum)
    update_rule="best_imitation_logit",  # 'fermi' / 'best_rational' / 'best_imitation' / 'best_imitation_logit' / 'best_rational_logit'
    allow_forced_l_in_cd=False,          # 仅CD起始时是否允许破产→L
    seed=0,                              # 随机种（sweep 副本会偏移）

    # --- 归一化刻度（一般不用改）---
    scale_low=0.0,
    scale_high=1.0,
    scale_eps=1e-9,

    # --- 统计均值的烧入比例（只影响 run() 返回的 *mean*；不影响 snapshot.csv）---
    burn_in_frac=0.5,    # 0=全程平均；1=只看最后一回合

    # --- 输出/采样控制（新加，可被 sweep 读取）---
    out_dir="AUTO",      # "AUTO"=runs/<时间戳>；None/""=不写；也可给自定义目录
    log_every= 25,        # -1≈自动 ~200 个采样点；1=每回合；100=每100回合
    snap_every=0         # 0=按比例点(0.1/0.5/1.0)；>0=每 N 回合存一张快照图
)


# 每组参数跑多少个随机种子副本求均值
REPLICAS = 50


# 并行进程数：用 “min(副本数, 逻辑CPU数)”，默认不再限制为 8
N_PROCS = min(REPLICAS, os.cpu_count() or 1)


# === 新增：本次运行的时间戳（文件夹名用） ===
RUN_TS = time.strftime("%Y%m%d-%H%M%S")   # 例如 "20250904-1130"
# ---- 2) 生成参数组合 ----
def iter_param_combos(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

# ---- 3) 单次 run（带参数合并）----
def run_one_config(args):
    """
    args: (config_dict, base_params, replicas, seed_base, q)
    子进程在每完成 1 个 replica 后向队列 q put(1)，
    供主进程的进度线程更新 tqdm。
    """
    cfg, base, replicas, seed0, q = args
    p_dict = {**base.__dict__, **cfg}
    results = []

    # 构建图像输出目录：按参数拼路径，避免混淆
    grid_name = "_".join([f"{k}={cfg[k]}" for k in sorted(cfg.keys())])
    run_root = os.path.join("sweep_out", RUN_TS)
    os.makedirs(run_root, exist_ok=True)
    save_root = os.path.join(run_root, grid_name)
    os.makedirs(save_root, exist_ok=True)

    for k in range(replicas):
        p_k = Params(**{**p_dict, "seed": (None if seed0 is None else seed0 + k)})

        # 仅首个副本出图，避免文件过多
        if k == 0:
              # rep0 的输出目录（之前缺少定义，导致 NameError）
            out_dir = os.path.join(save_root, "rep0")
            os.makedirs(out_dir, exist_ok=True)
            log_every_eff = (p_k.log_every if getattr(p_k, "log_every", -1) >= 0 else max(1, p_k.T // 200))
            snapshots_eff = (list(range(p_k.snap_every, p_k.T + 1, p_k.snap_every))
                             if getattr(p_k, "snap_every", 0) > 0 else (0.001, 0.01, 0.1, 0.5, 1.0))

            sim = SpatialGameFast(
                p_k,
                out_dir=out_dir,
                log_every=log_every_eff,
                snapshots=snapshots_eff,
                save_plots=True
            )
        else:
            sim = SpatialGameFast(p_k)

        out = sim.run()
        results.append(out)

        # —— 关键：每完成 1 个 replica，就向主进程上报一次 ——
        try:
            q.put(1)
        except Exception:
            pass  # 即使队列异常也不影响仿真结果

    # 聚合统计
    def agg(key):
        arr = np.array([r[key] for r in results], dtype=float)
        return float(arr.mean()), float(arr.std())

    meanC, stdC = agg("frac_C")
    meanD, stdD = agg("frac_D")
    meanL, stdL = agg("frac_L")
    meanAvg, stdAvg = agg("avg_res")
    meanSum, stdSum = agg("sum_res")
    meanG, stdG = agg("gini")
    meanCum, stdCum = agg("cum_sigma")
    meanNet, stdNet = agg("net_output")
    meanOut, stdOut = agg("cum_output")
    meanHatS, stdHatS = agg("cum_sigma_hat")
    meanHatN, stdHatN = agg("net_output_hat")
    meanHatO, stdHatO = agg("cum_output_hat")

    # 返回“配置 + 统计”
    outrow = {**cfg,
              "frac_C_mean": meanC, "frac_C_std": stdC,
              "frac_D_mean": meanD, "frac_D_std": stdD,
              "frac_L_mean": meanL, "frac_L_std": stdL,
              "avg_res_mean": meanAvg, "avg_res_std": stdAvg,
              "sum_res_mean": meanSum, "sum_res_std": stdSum,
              "gini_mean": meanG, "gini_std": stdG,
              "cum_sigma_mean": meanCum, "cum_sigma_std": stdCum,
              # —— 关键：把“净产出”也写入行 ——
              "net_output_mean": meanNet, "net_output_std": stdNet,

              # —— 归一化三项 ——
              "cum_output_mean": meanOut, "cum_output_std": stdOut,
              "cum_sigma_hat_mean": meanHatS, "cum_sigma_hat_std": stdHatS,
              "net_output_hat_mean": meanHatN, "net_output_hat_std": stdHatN,
              "cum_output_hat_mean": meanHatO, "cum_output_hat_std": stdHatO,
              }

    # 不在子进程里 print，避免 tqdm 乱序；必要的话可把日志写入 outrow 让主进程打印
    return outrow

# ---- 4) 主入口（并行执行并保存CSV）----
def main():
    combos = list(iter_param_combos(GRID))
    n_configs = len(combos)
    print(f"Total configs: {n_configs} × replicas {REPLICAS}")

    # ——— 心跳队列：子进程 put(1)，主进度条 +1
    manager = mp.Manager()
    prog_q = manager.Queue()

    # 进度总步数 = 配置数 × 每配置的副本数
    total_ticks = n_configs * REPLICAS

    # ——— 进度线程：在主进程里运行，专门消费队列、刷新 tqdm
    th = threading.Thread(target=_progress_consumer_thread,
                          args=(prog_q, total_ticks, "Sweep (config×replica)"),
                          daemon=True)
    th.start()

    # ——— 组装任务：把队列一并传给子进程
    tasks = [(cfg, BASE, REPLICAS, BASE.seed, prog_q) for cfg in combos]

    t0 = time.time()
    rows = []
    # Windows/多进程：spawn 更稳；chunksize=1 便于打散工作
    with mp.get_context("spawn").Pool(processes=N_PROCS) as pool:
        it = pool.imap_unordered(run_one_config, tasks, chunksize=1)
        for row in it:
            rows.append(row)
            # 如需在主进程打印每个 config 的概要，可在这里 print（不会打乱 tqdm）
            # print(f"[Config { {k: row[k] for k in GRID.keys()} }] "
            #       f"C={row['frac_C_mean']:.3f}, D={row['frac_D_mean']:.3f}, "
            #       f"L={row['frac_L_mean']:.3f}, avg_res={row['avg_res_mean']:.3f}, "
            #       f"gini={row['gini_mean']:.3f}, net_sum_res={row['net_sum_res_mean']:.3f}")
    dt = time.time() - t0

    # 等待进度线程自然结束（收到 total_ticks 个心跳后退出）
    th.join(timeout=1.0)

    # ===== 保存 CSV =====
    param_keys = sorted(GRID.keys())
    metric_keys = [
        "frac_C_mean", "frac_D_mean", "frac_L_mean",
        "avg_res_mean", "gini_mean",
        "sum_res_mean",
        "cum_sigma_mean",
        "cum_output_mean",
        "net_output_mean",  # ✅ 新增
        "cum_sigma_hat_mean",
        "cum_output_hat_mean",
        "net_output_hat_mean",  # ✅ 新增
    ]

    fieldnames = param_keys + metric_keys

    os.makedirs("sweep_out", exist_ok=True)

    rows_sorted = sorted(rows, key=lambda d: d.get("r", 0.0))
    grid_name = "-".join(param_keys)
    csv_name = f"sweep_{grid_name}_T{BASE.T}_rep{REPLICAS}_{RUN_TS}.csv"

    run_root = os.path.join("sweep_out", RUN_TS)
    os.makedirs(run_root, exist_ok=True)
    csv_path = os.path.join(run_root, csv_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("# GRID = " + repr(GRID) + "\n")
        f.write("# BASE = " + repr(BASE) + "\n")
        f.write(f"# REPLICAS = {REPLICAS}, T = {BASE.T}\n")

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Done in {dt:.1f}s. Saved: {csv_path}")

    # ===== 画汇总图：按 (allow_forced_l_in_cd, r) 聚合（自动对 eta 求平均）=====
    # 先按组聚合
    from collections import defaultdict
    bucket_gini = defaultdict(list)
    bucket_net_hat = defaultdict(list)

    for r in rows_sorted:
        key = (bool(r.get("allow_forced_l_in_cd", False)), float(r.get("r", 0.0)))
        if "gini_mean" in r:
            bucket_gini[key].append(float(r["gini_mean"]))
        if "net_output_hat_mean" in r:
            bucket_net_hat[key].append(float(r["net_output_hat_mean"]))

    def agg_dict(bucket):
        out = {}
        for (allow, rr), vals in bucket.items():
            if len(vals):
                out.setdefault(allow, {})[rr] = float(np.mean(vals))
        # 每个 allow 下按 r 排序成曲线
        for allow in out:
            out[allow] = dict(sorted(out[allow].items(), key=lambda kv: kv[0]))
        return out

    gini_curves = agg_dict(bucket_gini)
    net_hat_curves = agg_dict(bucket_net_hat)

    def plot_curves(curves, ylabel, fname):
        fig = plt.figure(figsize=(7.4,4.6), dpi=140); ax = plt.gca()
        for allow, d in curves.items():
            if not d: continue
            xs = np.array(list(d.keys()))
            ys = np.array(list(d.values()))
            label = "Allow L (forced→L)" if allow else "No L"
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        ax.set_xlabel("r"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(run_root, fname), bbox_inches="tight")
        plt.close(fig)

    # 画两张：Gini vs r；net_output_hat vs r
    plot_curves(gini_curves, "Gini (final)", "summary_gini_vs_r.png")
    plot_curves(net_hat_curves, "Net output (per-capita per-round)", "summary_net_output_hat_vs_r.png")

    print("Saved summary figures:",
          os.path.join(run_root, "summary_gini_vs_r.png"), " & ",
          os.path.join(run_root, "summary_net_output_hat_vs_r.png"))




if __name__ == "__main__":
    main()

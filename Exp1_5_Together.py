# expOne_Three_fast_final_stats.py
# -*- coding: utf-8 -*-
import os
import numpy as np
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


# ===== 工具 =====
def build_payoff_matrix(include_loner: bool, r: float, sigma: float) -> np.ndarray:
    if include_loner:
        return np.array([
            [1.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            [sigma, sigma, sigma],
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


# ====== 向量化邻居工具（周期边界）======
def rolled(arr: np.ndarray):
    # 返回上、下、左、右四个“邻居阵”
    up = np.roll(arr, -1, axis=0)
    down = np.roll(arr, 1, axis=0)
    left = np.roll(arr, -1, axis=1)
    right = np.roll(arr, 1, axis=1)
    return up, down, left, right


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
        self.cum_sigma = 0.0

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
        self._ts = []  # 每项：(t, frac_C, frac_D, frac_L, avg_res, gini)
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)


    def _fractions(self):
        frac_C = float(np.mean(self.strat == C))
        frac_D = float(np.mean(self.strat == D))
        frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
        return frac_C, frac_D, frac_L

    def _record_trend(self, t):
        if self.log_every > 0 and (t == 0 or t % self.log_every == 0 or t == self.p.T):
            frac_C, frac_D, frac_L = self._fractions()
            avg_res = float(np.mean(self.res))
            gini = gini_allow_negative(self.res)
            self._ts.append((t, frac_C, frac_D, frac_L, avg_res, gini))

    def _plot_strategy_map(self, t, fname):
        # 颜色：C=蓝，D=红，L=灰（如果没启 L，L 不会出现）
        cmap = ListedColormap(["#1f77b4", "#d62728", "FFFF00"])
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
        plt.close()

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
            w.writerow(["t", "frac_C", "frac_D", "frac_L", "avg_res", "gini"])
            w.writerows(self._ts)

    def _init_strategies(self) -> np.ndarray:
        L2 = self.L * self.L
        if self.include_loner and (self.p.fL is not None):
            assert abs(self.p.fc + self.p.fd + self.p.fL - 1.0) < 1e-8, "fc+fd+fL must be 1."
            cN = int(self.p.fc * L2)
            dN = int(self.p.fd * L2)
            lN = L2 - cN - dN
            pool = np.array([C] * cN + [D] * dN + [L] * lN, dtype=int)
        else:
            assert abs(self.p.fc + self.p.fd - 1.0) < 1e-8, "fc+fd must be 1."
            cN = int(self.p.fc * L2)
            dN = L2 - cN
            pool = np.array([C] * cN + [D] * dN, dtype=int)
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
                                     (1.0 - self.p.beta) * res_after_gain)
        # 软下限：若会低于 Min_Res，则回退到旧值（在 caller 里做）
        return res_after_consume

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

    def _force_loners_if_broke(self):
        """
        若上一回合不是 L 且当前资源<0：
        - 当 allow_forced_l_in_cd=True：转为 L，并在本回合冻结为 L；
        - 当 allow_forced_l_in_cd=False：不改变策略；
        只有在Loner允许的情况下，当前破产者资源均才会被钉为 0
        """
        self.just_forced_L[:] = False
        broke_now = (self.res < 0.0)

        if not np.any(broke_now):
            return

        # 仅针对“上一回合不是 L”的个体
        need_force = broke_now & (self.prev_strat != L)

        if self.p.allow_forced_l_in_cd and self.include_loner:
            # 允许强制转 L
            if np.any(need_force):
                self.strat[need_force] = L
                self.just_forced_L[need_force] = True

    def run(self) -> Dict[str, float]:
        # 只做极简统计：起始与最终
        # 初始
        # （可按需保留）init_frac_C = float(np.mean(self.strat == C))
        # 迭代
        # —— 新增：t=0 的初始点 ——
        self._record_trend(0)
        self._maybe_snapshot_plots(0)

        for t in range(1, self.p.T + 1):
            res_old = self.res.copy()
            self.prev_strat = self.strat.copy()

            # (1) 收益
            round_payoff = self._round_payoff_vec(self.strat)
            if self.include_loner:
                nL = int(np.sum(self.strat == L))
                self.cum_sigma += nL * 4.0 * self.p.sigma
            res_after_gain = res_old + round_payoff

            # (2) 消耗
            res_after_consume = self._consume_rule(res_after_gain)
            self.res = np.where(res_after_consume < self.p.Min_Res, res_old, res_after_consume)

            # (3) 破产转 L
            self._force_loners_if_broke()

            # (4) 学习信号
            if self.p.learn_signal == "round":
                S = round_payoff
            elif self.p.learn_signal == "cumulative":
                S = self.res
            elif self.p.learn_signal == "per_strategy":
                S = np.empty_like(self.res)
                S[self.strat == C] = self.res[self.strat == C]
                S[self.strat == D] = round_payoff[self.strat == D]
                maskL = (self.strat == L)
                if np.any(maskL):
                    S[maskL] = 0.5 * (self.res[maskL] + round_payoff[maskL])
            else:
                raise ValueError("learn_signal must be 'round'/'cumulative'/'per_strategy'")

            # (5) 费米更新
            self._fermi_update_vec(S)

            # —— 新增：记录与出图 ——
            self._record_trend(t)
            self._maybe_snapshot_plots(t)

        # 结束统计（保持你原来的）
        frac_C = float(np.mean(self.strat == C))
        frac_D = float(np.mean(self.strat == D))
        frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
        avg_res = float(np.mean(self.res))
        sum_res = float(np.sum(self.res))
        gini = gini_allow_negative(self.res)
        cum_sigma = float(self.cum_sigma)
        net_sum_res = sum_res - cum_sigma

        # —— 新增：保存时间序列 ——
        self._dump_trend_csv()

        return {
            "frac_C": frac_C, "frac_D": frac_D, "frac_L": frac_L,
            "avg_res": avg_res, "sum_res": sum_res, "gini": gini,
            "cum_sigma": cum_sigma, "net_sum_res": net_sum_res,
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
    parser.add_argument("--minres", type=float, default=-10000.0)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--learn", type=str, default="cumulative",
                        choices=["round", "cumulative", "per_strategy"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_forced_l", action="store_true", help="仅CD起始时也允许破产转L")
    parser.add_argument("--runs", type=int, default=1, help="并行独立重复次数（不同随机种子）")
    args = parser.parse_args()

    p = Params(
        Lsize=args.L, T=args.T, fc=args.fc, fd=args.fd,
        fL=None if args.fL < 0 else args.fL,
        pi0=args.pi0, r=args.r, sigma=args.sigma,
        alpha=args.alpha, gamma=args.gamma, beta=args.beta,
        Min_Res=args.minres, kappa=args.kappa,
        learn_signal=args.learn, seed=args.seed,
        allow_forced_l_in_cd=args.allow_forced_l
    )

    if args.runs == 1:
        t0 = time.time()
        out = run_once(p)
        dt = time.time() - t0
        print(f"[Single run done in {dt:.2f}s]")
        print(
            f"Final @ T={p.T}: "
            f"C={out['frac_C']:.3f}  D={out['frac_D']:.3f}  L={out['frac_L']:.3f}  "
            f"avg_res={out['avg_res']:.3f}  sum_res={out['sum_res']:.3f}  gini={out['gini']:.3f}  "
            f"cum_sigma={out['cum_sigma']:.3f}  net_sum_res={out['net_sum_res']:.3f}"
        )

    else:
        # 多进程跑多次，自动使用不同种子
        jobs = []
        for k in range(args.runs):
            pk = Params(**{**p.__dict__, "seed": (None if p.seed is None else p.seed + k)})
            jobs.append(pk)
        t0 = time.time()
        with mp.get_context("spawn").Pool(processes=min(args.runs, os.cpu_count() or 1)) as pool:
            results = pool.map(run_once, jobs)
        dt = time.time() - t0

        # 汇总
        arrC = np.array([r["frac_C"] for r in results])
        arrD = np.array([r["frac_D"] for r in results])
        arrL = np.array([r["frac_L"] for r in results])
        arrAvg = np.array([r["avg_res"] for r in results])
        arrSum = np.array([r["sum_res"] for r in results])
        arrG = np.array([r["gini"] for r in results])
        arrCum = np.array([r["cum_sigma"] for r in results])
        arrNet = np.array([r["net_sum_res"] for r in results])

        print(f"[{args.runs} runs done in {dt:.2f}s] mean±std:")
        print(f"C={arrC.mean():.3f}±{arrC.std():.3f}  D={arrD.mean():.3f}±{arrD.std():.3f}  "
              f"L={arrL.mean():.3f}±{arrL.std():.3f}")
        print(f"avg_res={arrAvg.mean():.3f}±{arrAvg.std():.3f}  "
              f"sum_res={arrSum.mean():.3f}±{arrSum.std():.3f}  "
              f"gini={arrG.mean():.3f}±{arrG.std():.3f}  "
              f"cum_sigma={arrCum.mean():.3f}±{arrCum.std():.3f}  "
              f"net_sum_res={arrNet.mean():.3f}±{arrNet.std():.3f}")


# if __name__ == "__main__":
#     main()
#
# if __name__ == "__main__":
#     p_cd = Params(
#         Lsize=60, T=10000,
#         fc=0.5, fd=0.5, fL=None,  # fL=None => 仅 CD 起始
#         pi0=5,  # 初始资源
#         r=1.8, sigma=0.25,  # r: 背叛诱惑系数, sigma: Loner低保
#         kappa=1,  # 费米学习因子
#         learn_signal="cumulative",  # 'per_strategy' / 'round' / 'cumulative'
#         seed=0,
#         allow_forced_l_in_cd=True  # 资源耗尽者是否变为 L
#     )
#
#     sim_cd = SpatialGameFast(p_cd)
#     result = sim_cd.run()
#
#     print("[CD-start] final fractions:",
#           f"C={result['frac_C']:.3f}",
#           f"D={result['frac_D']:.3f}",
#           f"L={result['frac_L']:.3f}",
#           f"avg_res={result['avg_res']:.3f}",
#           f"sum_res={result['sum_res']:.3f}",
#           f"gini={result['gini']:.3f}")


# sweep_parallel.py
# -*- coding: utf-8 -*-
import os, csv, time, itertools
import numpy as np
import multiprocessing as mp
import tqdm

# ---- 1) 定义你想扫的参数网格（自行改动） ----
GRID = {
    "r":                    [1.2, 1.4, 1.6, 1.8,2.0],
    "allow_forced_l_in_cd": [True , False],
    "pi0":                  [5, 6, 7, 8, 9, 10]

    #"sigma":                [0.25,0.26,0.27,0.28,0.29,0.30]
    # 也可加入 "sigma": [0.2, 0.3], "allow_forced_l_in_cd": [False, True], ...
}
# 固定不变的公共参数
BASE = Params(
    Lsize=50,   # 格子边长 L，环境大小 L x L（周期边界）
    T=50000,     # 演化总回合数
    fc=0.5,       # 初始 C 比例（仅 CD 模式时用 fc, fd；CDL 模式时用 fc, fd, fL）
    fd=0.5,
    fL=None,
    r=1.6,        #背叛诱惑指数
    pi0=5.0,    # 初始资源 π0（所有玩家相同）
    sigma=0.25, # Loner 的单边互动收益 σ（每个邻边都拿 σ）
    alpha=1.0,   # 每回合的固定损耗参数
    gamma=10.0,  #阈值
    beta = 0.1,  # 超出阈值后每回合消费当前资源的比例
    Min_Res=-10000.0,   #最低的资源值
    kappa=1,     # 费米学习规则 κ
    learn_signal="cumulative",
    seed=0,  # 会在副本里偏移
    allow_forced_l_in_cd=False,
)

# 每组参数跑多少个随机种子副本求均值
REPLICAS = 50

# 并行进程数（不给就用 CPU 核心数）
N_PROCS = min(8, os.cpu_count() or 1)

# === 新增：本次运行的时间戳（文件夹名用） ===
RUN_TS = time.strftime("%Y%m%d-%H%M%S")   # 例如 "20250904-1130"
# ---- 2) 生成参数组合 ----
def iter_param_combos(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

# ---- 3) 单次 run（带参数合并）----
def run_one_config(args):
    """args: (config_dict, base_params, replicas, seed_base)"""
    cfg, base, replicas, seed0 = args
    p_dict = {**base.__dict__, **cfg}
    results = []

    # 构建一个图像输出目录：按参数拼路径，避免混淆
    grid_name = "_".join([f"{k}={cfg[k]}" for k in sorted(cfg.keys())])
    # 本次运行的根目录：sweep_out/<时间戳>/
    run_root = os.path.join("sweep_out", RUN_TS)
    os.makedirs(run_root, exist_ok=True)

    # 每个参数组合的图像目录：sweep_out/<时间戳>/figs/<参数键值对>/
    save_root = os.path.join(run_root, "figs", grid_name)
    os.makedirs(save_root, exist_ok=True)

    os.makedirs(save_root, exist_ok=True)

    for k in range(replicas):
        p_k = Params(**{**p_dict, "seed": (None if seed0 is None else seed0 + k)})

        # 只让第一条副本出图，避免太多文件
        if k == 0:
            out_dir = os.path.join(save_root, "rep0")
            sim = SpatialGameFast(
                p_k,
                out_dir=out_dir,
                log_every=max(1, p_k.T // 200),  # 约 200 个点
                snapshots=(0.001, 0.01, 0.1, 0.5, 1.0),       # 10%、50%、100% 三张
                save_plots=True
            )
        else:
            sim = SpatialGameFast(p_k)  # 其它副本不出图

        out = sim.run()
        results.append(out)

    # 聚合
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
    meanNet, stdNet = agg("net_sum_res")

    # 返回“配置 + 统计”
    outrow = {**cfg,
              "frac_C_mean": meanC, "frac_C_std": stdC,
              "frac_D_mean": meanD, "frac_D_std": stdD,
              "frac_L_mean": meanL, "frac_L_std": stdL,
              "avg_res_mean": meanAvg, "avg_res_std": stdAvg,
              "sum_res_mean": meanSum, "sum_res_std": stdSum,
              "gini_mean": meanG, "gini_std": stdG,
              "cum_sigma_mean": meanCum, "cum_sigma_std": stdCum,
              "net_sum_res_mean": meanNet, "net_sum_res_std": stdNet}
    print(f"[Config {cfg}] C={meanC:.3f}, D={meanD:.3f}, L={meanL:.3f}, "
          f"avg_res={meanAvg:.3f}, gini={meanG:.3f}, "
          f"cum_sigma={meanCum:.3f}, net_sum_res={meanNet:.3f}")

    return outrow

# ---- 4) 主入口（并行执行并保存CSV）----
def main():
    import tqdm  # tqdm.auto 也行

    combos = list(iter_param_combos(GRID))
    total = len(combos)
    print(f"Total configs: {total} × replicas {REPLICAS}")

    tasks = [(cfg, BASE, REPLICAS, BASE.seed) for cfg in combos]

    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(processes=N_PROCS) as pool:
        # 用 imap_unordered 边出结果边更新进度
        for row in tqdm.tqdm(pool.imap_unordered(run_one_config, tasks),
                             total=total, desc="Sweep", smoothing=0.1):
            rows.append(row)
    dt = time.time() - t0

    # ===== 保存 CSV =====
    param_keys = sorted(GRID.keys())
    metric_keys = [
        "frac_C_mean", "frac_D_mean", "frac_L_mean",
        "avg_res_mean", "gini_mean", "cum_sigma_mean", "net_sum_res_mean"
    ]

    fieldnames = param_keys + metric_keys

    os.makedirs("sweep_out", exist_ok=True)

    # 按 r 值排序
    rows_sorted = sorted(rows, key=lambda d: d.get("r", 0.0))

    # 构造逻辑化文件名
    # 构造逻辑化文件名（文件名前仍带时间戳，目录也带时间戳，方便检索）
    grid_name = "-".join(param_keys)
    csv_name = f"sweep_{grid_name}_T{BASE.T}_rep{REPLICAS}_{RUN_TS}.csv"

    # 本次运行的根目录（和上面保持一致）
    run_root = os.path.join("sweep_out", RUN_TS)
    os.makedirs(run_root, exist_ok=True)

    csv_path = os.path.join(run_root, csv_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # ==== 写注释行，记录实验配置 ====
        f.write("# GRID = " + repr(GRID) + "\n")
        f.write("# BASE = " + repr(BASE) + "\n")
        f.write(f"# REPLICAS = {REPLICAS}, T = {BASE.T}\n")

        # ==== 写表头和数据 ====
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Done in {dt:.1f}s. Saved: {csv_path}")



if __name__ == "__main__":
    main()

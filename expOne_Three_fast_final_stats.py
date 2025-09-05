# expOne_Three_fast_final_stats.py
# -*- coding: utf-8 -*-
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import multiprocessing as mp
import argparse
import time

# ===== 策略编码 =====
C, D, L = 0, 1, 2  # Cooperator / Defector / Loner

# ===== 参数 =====
@dataclass
class Params:
    Lsize: int = 60
    T: int = 10000
    fc: float = 0.5
    fd: float = 0.5
    fL: Optional[float] = None      # None=仅CD；否则CDL
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
            [1.0,   0.0,   0.0],
            [r,     0.0,   0.0],
            [sigma, sigma, sigma],
        ], dtype=float)
    else:
        return np.array([
            [1.0, 0.0],
            [r,   0.0],
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
    up    = np.roll(arr, -1, axis=0)
    down  = np.roll(arr,  1, axis=0)
    left  = np.roll(arr, -1, axis=1)
    right = np.roll(arr,  1, axis=1)
    return up, down, left, right

# ====== 主仿真（向量化 + 无绘图版）======
class SpatialGameFast:
    def __init__(self, p: Params):
        self.p = p
        if p.seed is not None:
            np.random.seed(p.seed)
        self.L = p.Lsize
        # —— 只有当 allow_forced_l_in_cd=True 时，系统才启用 Loner 生态 ——
        self.include_loner = bool(p.allow_forced_l_in_cd)
        # 若不开启 Loner，则忽略任何传入的 fL，保证初始只有 C/D
        if not self.include_loner:
            self.p.fL = None
            # 根据是否启用 Loner 选择 2x2 或 3x3 收益矩阵
            self.M = build_payoff_matrix(self.include_loner, p.r, p.sigma)
        self.M = build_payoff_matrix(self.include_loner, p.r, p.sigma)
        self.strat = self._init_strategies()
        self.res = np.full((self.L, self.L), fill_value=p.pi0, dtype=float)
        self.prev_strat = self.strat.copy()
        self.just_forced_L = np.zeros((self.L, self.L), dtype=bool)
        # === 新增：累计“社会低保” σ ===
        self.cum_sigma = 0.0

    def _init_strategies(self) -> np.ndarray:
        L2 = self.L * self.L
        if self.include_loner and (self.p.fL is not None):
            assert abs(self.p.fc + self.p.fd + self.p.fL - 1.0) < 1e-8, "fc+fd+fL must be 1."
            cN = int(self.p.fc * L2)
            dN = int(self.p.fd * L2)
            lN = L2 - cN - dN
            pool = np.array([C]*cN + [D]*dN + [L]*lN, dtype=int)
        else:
            assert abs(self.p.fc + self.p.fd - 1.0) < 1e-8, "fc+fd must be 1."
            cN = int(self.p.fc * L2)
            dN = L2 - cN
            pool = np.array([C]*cN + [D]*dN, dtype=int)
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
        for t in range(1, self.p.T + 1):
            res_old = self.res.copy()
            self.prev_strat = self.strat.copy()

            # (1) 收益
            round_payoff = self._round_payoff_vec(self.strat)
            # === 新增：本回合 Loner 总 σ（四邻域 => 每个 L 拿 4σ）===
            if self.include_loner:
                nL = int(np.sum(self.strat == L))
                self.cum_sigma += nL * 4.0 * self.p.sigma
            # ================================================
            res_after_gain = res_old + round_payoff

            # (2) 消耗（带软下限）
            res_after_consume = self._consume_rule(res_after_gain)
            self.res = np.where(res_after_consume < self.p.Min_Res, res_old, res_after_consume)

            # (3) 破产转 L
            # (3) 破产转 L（函数内部会按 allow_forced_l_in_cd 判定是否转 L）
            self._force_loners_if_broke()

            # (4) 学习信号 S
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

            # (5) 费米更新（向量化）
            self._fermi_update_vec(S)

        # 结束统计
        frac_C = float(np.mean(self.strat == C))
        frac_D = float(np.mean(self.strat == D))
        frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
        avg_res = float(np.mean(self.res))
        sum_res = float(np.sum(self.res))
        gini = gini_allow_negative(self.res)
        # === 新增：累计 σ 与“净得资源总数” ===
        cum_sigma = float(self.cum_sigma)
        net_sum_res = sum_res - cum_sigma
        return {
            "frac_C": frac_C,
            "frac_D": frac_D,
            "frac_L": frac_L,
            "avg_res": avg_res,
            "sum_res": sum_res,
            "gini": gini,
            "cum_sigma": cum_sigma,  # 累计“社会低保”
            "net_sum_res": net_sum_res,  # 扣除低保后的净资源
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
        arrG  = np.array([r["gini"]    for r in results])
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


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    p_cd = Params(
        Lsize = 60, T = 10000,
        fc = 0.5, fd = 0.5, fL = None,     # fL=None => 仅 CD 起始
        pi0 = 5,                           # 初始资源
        r = 1.8, sigma = 0.25,             # r: 背叛诱惑系数, sigma: Loner低保
        kappa = 1,                         # 费米学习因子
        learn_signal = "cumulative",       # 'per_strategy' / 'round' / 'cumulative'
        seed = 0,
        allow_forced_l_in_cd = True       # 资源耗尽者是否变为 L
    )

    sim_cd = SpatialGameFast(p_cd)
    result = sim_cd.run()

    print("[CD-start] final fractions:",
          f"C={result['frac_C']:.3f}",
          f"D={result['frac_D']:.3f}",
          f"L={result['frac_L']:.3f}",
          f"avg_res={result['avg_res']:.3f}",
          f"sum_res={result['sum_res']:.3f}",
          f"gini={result['gini']:.3f}")

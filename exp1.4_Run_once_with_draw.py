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
    fL: Optional[float] = None      # None=仅CD；否则CDL（本实现仅在 allow_forced_l_in_cd=True 时启用 L 生态）
    pi0: float = 5.0
    r: float = 1.8
    sigma: float = 0.25
    alpha: float = 1.0
    gamma: float = 10.0
    beta: float = 0.1
    Min_Res: float = -10000.0
    kappa: float = 1.0
    learn_signal: str = "cumulative"  # 'round' / 'cumulative' / 'per_strategy'
    seed: Optional[int] = 0
    allow_forced_l_in_cd: bool = True

    # 统计与可视化控制
    measure_every: int = 10                 # 每多少回合记录一次统计
    snapshot_every: int = 0                 # <=0 不出图；>0 表示每隔多少回合出一张“面板图”
    snapshot_dir: str = "snapshots_fast"    # 快照目录
    curves_filename: str = "fractions_curve.png"  # 占比曲线图文件名（若 panels_in_one=False 时单独保存）
    panels_in_one: bool = True              # True: 单张图里上(策略/资源)+下(曲线)；False: 面板与曲线分图保存

    # 手动控制
    save_snapshots: bool = True             # 是否保存图片
    show_snapshots: bool = False            # 是否弹窗显示（需要 GUI 环境）
    print_every: int = 0                    # 每隔多少回合打印一次统计（0=不打印）

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

# ====== 主仿真（向量化 + 快照/曲线）======
class SpatialGameFast:
    def __init__(self, p: Params):
        self.p = p
        if p.seed is not None:
            np.random.seed(p.seed)
        self.L = p.Lsize

        # —— 只有当 allow_forced_l_in_cd=True 时，系统才启用 Loner 生态（3x3 矩阵） ——
        self.include_loner = bool(p.allow_forced_l_in_cd)
        if not self.include_loner:
            self.p.fL = None
        self.M = build_payoff_matrix(self.include_loner, p.r, p.sigma)

        self.strat = self._init_strategies()
        self.res = np.full((self.L, self.L), fill_value=p.pi0, dtype=float)
        self.prev_strat = self.strat.copy()
        self.just_forced_L = np.zeros((self.L, self.L), dtype=bool)

        # 累计“社会低保” σ（仅统计用途）
        self.cum_sigma = 0.0

        # 历史统计（用于占比曲线与面板标题）
        self.hist = {
            "t": [],
            "frac_C": [],
            "frac_D": [],
            "frac_L": [],
            "avg_res": [],
            "gini": [],
        }

        # 图像目录准备
        if self.p.snapshot_every and self.p.snapshot_every > 0 and self.p.save_snapshots:
            os.makedirs(self.p.snapshot_dir, exist_ok=True)

        # 策略色表（CD 或 CDL）
        from matplotlib.colors import ListedColormap  # 局部 import，避免无 pyplot 时的问题
        self.cmap_cd  = ListedColormap(["blue", "red"])
        self.cmap_cdl = ListedColormap(["blue", "red", "yellow"])

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
        return res_after_consume

    # ---- (3) 费米更新：一次性随机方向并向量化 ----
    def _fermi_update_vec(self, S: np.ndarray):
        kappa = max(self.p.kappa, 1e-8)
        s = self.strat
        S_up, S_down, S_left, S_right = rolled(S)
        s_up, s_down, s_left, s_right = rolled(s)

        dir_choice = np.random.randint(0, 4, size=s.shape)

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

        # 冻结：刚被强制 L 或“本回合新变为 L”（更稳健）
        frozen = self.just_forced_L | ((self.prev_strat != L) & (self.strat == L))

        delta = np.clip(S - neighbor_S, -50.0, 50.0)
        adopt_p = 1.0 / (1.0 + np.exp(delta / kappa))
        randu = np.random.rand(*s.shape)
        adopt = (randu < adopt_p) & (~frozen)

        next_s = np.where(adopt, neighbor_s, s)
        next_s = np.where(frozen, L, next_s)
        self.strat = next_s

    def _force_loners_if_broke(self):
        """
        若上一回合不是 L 且当前资源<0：
        - 当 allow_forced_l_in_cd=True：转为 L，并在本回合冻结为 L；
        - 资源同时钉为 0；
        """
        self.just_forced_L[:] = False
        broke_now = (self.res < 0.0)
        if not np.any(broke_now):
            return
        need_force = broke_now & (self.prev_strat != L)
        if self.p.allow_forced_l_in_cd and self.include_loner and np.any(need_force):
            self.strat[need_force] = L
            self.just_forced_L[need_force] = True
        # 破产者资源统一钉为 0（不管是否允许转 L）
        self.res[broke_now] = 0.0

    # -------- 记录一次统计 --------
    def _record_stats(self, t: int):
        frac_C = float(np.mean(self.strat == C))
        frac_D = float(np.mean(self.strat == D))
        frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
        avg_res = float(np.mean(self.res))
        gini = gini_allow_negative(self.res)
        self.hist["t"].append(t)
        self.hist["frac_C"].append(frac_C)
        self.hist["frac_D"].append(frac_D)
        self.hist["frac_L"].append(frac_L)
        self.hist["avg_res"].append(avg_res)
        self.hist["gini"].append(gini)

    # -------- 面板/曲线绘图 --------
    def _snapshot(self, t: int):
        if not (self.p.snapshot_every and self.p.snapshot_every > 0):
            return

        # 延迟导入 pyplot 与 TwoSlopeNorm，确保后端已在 main()/demo_run() 里设定
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        # 文件名
        panel_png = os.path.join(self.p.snapshot_dir, f"snapshot_t{t:05d}.png")
        curves_png = os.path.join(self.p.snapshot_dir, self.p.curves_filename)

        # 策略图设置
        if self.include_loner:
            cmap_strat = self.cmap_cdl
            vmax_strat = 2
        else:
            cmap_strat = self.cmap_cd
            vmax_strat = 1

        # 资源图规范化
        res_min, res_max = float(self.res.min()), float(self.res.max())
        if abs(res_max - res_min) < 1e-12:
            res_min -= 1e-6
            res_max += 1e-6
        use_center = (res_min < 0.0 < res_max)
        norm = TwoSlopeNorm(vmin=res_min, vcenter=0.0, vmax=res_max) if use_center else None

        # === 绘制 ===
        if self.p.panels_in_one:
            fig = plt.figure(figsize=(12, 8), dpi=120)
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.25, wspace=0.18)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(self.strat, vmin=0, vmax=vmax_strat, cmap=cmap_strat, interpolation="nearest")
            ax0.set_title(f"Strategies @ t={t}")
            ax0.set_xticks([]); ax0.set_yticks([])

            ax1 = fig.add_subplot(gs[0, 1])
            if norm is None:
                im1 = ax1.imshow(self.res, vmin=res_min, vmax=res_max, interpolation="nearest")
                ax1.set_title(f"Resources  min={res_min:.2f}, max={res_max:.2f}")
            else:
                im1 = ax1.imshow(self.res, norm=norm, cmap="RdBu_r", interpolation="nearest")
                ax1.set_title(f"Resources (center=0)  min={res_min:.2f}, max={res_max:.2f}")
            ax1.set_xticks([]); ax1.set_yticks([])
            cb = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cb.set_label("resource")
            cb.ax.yaxis.get_offset_text().set_visible(False)
            cb.ax.ticklabel_format(style="plain")

            ax2 = fig.add_subplot(gs[1, :])
            t_arr = np.array(self.hist["t"], dtype=int)
            ax2.plot(t_arr, self.hist["frac_C"], label="C", linewidth=1.5)
            ax2.plot(t_arr, self.hist["frac_D"], label="D", linewidth=1.5)
            if self.include_loner:
                ax2.plot(t_arr, self.hist["frac_L"], label="L", linewidth=1.5)
            ax2.set_ylim(0.0, 1.0)
            ax2.set_xlabel("t")
            ax2.set_ylabel("fraction")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
            ax2.set_title("Fraction of C/D/L over time")

            # 保存/显示/关闭（统一）
            if self.p.save_snapshots:
                fig.savefig(panel_png, bbox_inches="tight")
            if self.p.show_snapshots:
                plt.show(block=False); plt.pause(0.001)
            else:
                plt.close(fig)

        else:
            # 面板图（策略+资源）
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120, constrained_layout=True)
            axes[0].imshow(self.strat, vmin=0, vmax=vmax_strat, cmap=cmap_strat, interpolation="nearest")
            axes[0].set_title(f"Strategies @ t={t}")
            axes[0].set_xticks([]); axes[0].set_yticks([])

            if norm is None:
                im1 = axes[1].imshow(self.res, vmin=res_min, vmax=res_max, interpolation="nearest")
                axes[1].set_title(f"Resources  min={res_min:.2f}, max={res_max:.2f}")
            else:
                im1 = axes[1].imshow(self.res, norm=norm, cmap="RdBu_r", interpolation="nearest")
                axes[1].set_title(f"Resources (center=0)  min={res_min:.2f}, max={res_max:.2f}")
            axes[1].set_xticks([]); axes[1].set_yticks([])
            cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            cb.set_label("resource")
            cb.ax.yaxis.get_offset_text().set_visible(False)
            cb.ax.ticklabel_format(style="plain")

            if self.p.save_snapshots:
                fig.savefig(panel_png, bbox_inches="tight")
            if self.p.show_snapshots:
                plt.show(block=False); plt.pause(0.001)
            else:
                plt.close(fig)

            # 占比曲线（单独一张）
            fig2, ax2 = plt.subplots(figsize=(8, 3.5), dpi=120)
            t_arr = np.array(self.hist["t"], dtype=int)
            ax2.plot(t_arr, self.hist["frac_C"], label="C", linewidth=1.5)
            ax2.plot(t_arr, self.hist["frac_D"], label="D", linewidth=1.5)
            if self.include_loner:
                ax2.plot(t_arr, self.hist["frac_L"], label="L", linewidth=1.5)
            ax2.set_ylim(0.0, 1.0)
            ax2.set_xlabel("t")
            ax2.set_ylabel("fraction")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
            ax2.set_title("Fraction of C/D/L over time")
            if self.p.save_snapshots:
                fig2.savefig(curves_png, bbox_inches="tight")
            if self.p.show_snapshots:
                plt.show(block=False); plt.pause(0.001)
            else:
                plt.close(fig2)

    def run(self) -> Dict[str, float]:
        # 初始统计一次（t=0）
        self._record_stats(t=0)
        if self.p.snapshot_every and self.p.snapshot_every > 0:
            self._snapshot(t=0)

        for t in range(1, self.p.T + 1):
            res_old = self.res.copy()
            self.prev_strat = self.strat.copy()

            # (1) 收益
            round_payoff = self._round_payoff_vec(self.strat)
            if self.include_loner:
                nL = int(np.sum(self.strat == L))
                self.cum_sigma += nL * 4.0 * self.p.sigma
            res_after_gain = res_old + round_payoff

            # (2) 消耗（带软下限）
            res_after_consume = self._consume_rule(res_after_gain)
            self.res = np.where(res_after_consume < self.p.Min_Res, res_old, res_after_consume)

            # (3) 破产转 L + 资源清零
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

            # (6.1) 记录
            if (t % self.p.measure_every) == 0:
                self._record_stats(t)

            # (6.2) 定时打印（可选）
            if self.p.print_every and (t % self.p.print_every == 0):
                frac_C = float(np.mean(self.strat == C))
                frac_D = float(np.mean(self.strat == D))
                frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
                avg_res = float(np.mean(self.res))
                gini = gini_allow_negative(self.res)
                print(f"[t={t}] C={frac_C:.3f}  D={frac_D:.3f}  L={frac_L:.3f}  "
                      f"avg_res={avg_res:.3f}  gini={gini:.3f}")

            # (6.3) 定时面板图（保存/弹窗由开关决定）
            if self.p.snapshot_every and self.p.snapshot_every > 0 and (t % self.p.snapshot_every == 0):
                self._snapshot(t)

        # 结束统计
        frac_C = float(np.mean(self.strat == C))
        frac_D = float(np.mean(self.strat == D))
        frac_L = float(np.mean(self.strat == L)) if self.include_loner else 0.0
        avg_res = float(np.mean(self.res))
        sum_res = float(np.sum(self.res))
        gini = gini_allow_negative(self.res)
        cum_sigma = float(self.cum_sigma)
        net_sum_res = sum_res - cum_sigma
        return {
            "frac_C": frac_C,
            "frac_D": frac_D,
            "frac_L": frac_L,
            "avg_res": avg_res,
            "sum_res": sum_res,
            "gini": gini,
            "cum_sigma": cum_sigma,
            "net_sum_res": net_sum_res,
        }

# ====== 单次运行的便捷函数（给并行用）======
def run_once(p: Params) -> Dict[str, float]:
    sim = SpatialGameFast(p)
    return sim.run()

# ====== CLI & 并行入口（先设后端，再 import pyplot）======
def main():
    import matplotlib
    parser = argparse.ArgumentParser(description="Fast spatial game (stats + snapshots).")
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument("--fc", type=float, default=0.5)
    parser.add_argument("--fd", type=float, default=0.5)
    parser.add_argument("--fL", type=float, default=-1.0, help="-1=仅CD；>=0 表示CDL初始L比例（本实现只有启用 allow_forced_l 时才用到 L 生态）")
    parser.add_argument("--pi0", type=float, default=5.0)
    parser.add_argument("--r", type=float, default=1.8)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--minres", type=float, default=-10000.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--learn", type=str, default="cumulative",
                        choices=["round", "cumulative", "per_strategy"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_forced_l", action="store_true", help="仅CD起始时也允许破产转L（启用L生态）")
    parser.add_argument("--runs", type=int, default=1, help="并行独立重复次数（不同随机种子）")
    # 可视化/统计
    parser.add_argument("--measure_every", type=int, default=10)
    parser.add_argument("--snapshot_every", type=int, default=0)
    parser.add_argument("--snapshot_dir", type=str, default="snapshots_fast")
    parser.add_argument("--panels_in_one", action="store_true")
    parser.add_argument("--save_snapshots", action="store_true")
    parser.add_argument("--show_snapshots", action="store_true")
    parser.add_argument("--print_every", type=int, default=0)
    args = parser.parse_args()

    # 根据弹窗需求选择后端（必须在 import pyplot 之前）
    if args.show_snapshots:
        matplotlib.use("TkAgg")   # 或 "QtAgg"，看你的环境
    else:
        matplotlib.use("Agg")

    # 到这里才 import pyplot（_snapshot 内部也会延迟导入）
    import matplotlib.pyplot as plt  # noqa: F401

    p = Params(
        Lsize=args.L, T=args.T, fc=args.fc, fd=args.fd,
        fL=None if args.fL < 0 else args.fL,
        pi0=args.pi0, r=args.r, sigma=args.sigma,
        alpha=args.alpha, gamma=args.gamma, beta=args.beta,
        Min_Res=args.minres, kappa=args.kappa,
        learn_signal=args.learn, seed=args.seed,
        allow_forced_l_in_cd=args.allow_forced_l,
        measure_every=args.measure_every,
        snapshot_every=args.snapshot_every,
        snapshot_dir=args.snapshot_dir,
        panels_in_one=args.panels_in_one,
        save_snapshots=args.save_snapshots,
        show_snapshots=args.show_snapshots,
        print_every=args.print_every,
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
        if p.snapshot_every and p.snapshot_every > 0 and p.save_snapshots:
            print(f"Snapshots saved in: {p.snapshot_dir}")
    else:
        jobs = []
        for k in range(args.runs):
            pk = Params(**{**p.__dict__, "seed": (None if p.seed is None else p.seed + k)})
            jobs.append(pk)
        t0 = time.time()
        with mp.get_context("spawn").Pool(processes=min(args.runs, os.cpu_count() or 1)) as pool:
            results = pool.map(run_once, jobs)
        dt = time.time() - t0

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
        if p.snapshot_every and p.snapshot_every > 0 and p.save_snapshots:
            print(f"Snapshots saved in: {p.snapshot_dir}")

if __name__ == "__main__":
    main()

# ====== 小函数：直接在 Python 里调用，方便调参 ======
def demo_run(
    Lsize=50,
    T=500,
    fc=0.5,
    fd=0.5,
    fL=None,
    pi0=5.0,
    r=1.8,
    sigma=0.25,
    alpha=1.0,
    gamma=10.0,
    beta=0.1,
    Min_Res=-10000.0,
    kappa=1.0,
    learn_signal="cumulative", seed=0,
    allow_forced_l=True,
    measure_every=10, snapshot_every=50,
    snapshot_dir="snapshots_demo",
    panels_in_one=True,
    save_snapshots=True, show_snapshots=False,
    print_every=50,
):
    """
    运行一个快速仿真，返回最终结果，并可保存快照/曲线。
    """
    import matplotlib
    if show_snapshots:
        matplotlib.use("TkAgg")  # 或 "QtAgg"
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    p = Params(
        Lsize=Lsize, T=T, fc=fc, fd=fd, fL=fL,
        pi0=pi0, r=r, sigma=sigma,
        alpha=alpha, gamma=gamma, beta=beta,
        Min_Res=Min_Res, kappa=kappa,
        learn_signal=learn_signal, seed=seed,
        allow_forced_l_in_cd=allow_forced_l,
        measure_every=measure_every,
        snapshot_every=snapshot_every,
        snapshot_dir=snapshot_dir,
        panels_in_one=panels_in_one,
        save_snapshots=save_snapshots,
        show_snapshots=show_snapshots,
        print_every=print_every,
    )

    sim = SpatialGameFast(p)
    result = sim.run()

    print("[demo_run done]")
    print(
        f"C={result['frac_C']:.3f}  D={result['frac_D']:.3f}  L={result['frac_L']:.3f}  "
        f"avg_res={result['avg_res']:.3f}  sum_res={result['sum_res']:.3f}  "
        f"gini={result['gini']:.3f}  cum_sigma={result['cum_sigma']:.3f}  "
        f"net_sum_res={result['net_sum_res']:.3f}"
    )

    if snapshot_every > 0 and save_snapshots:
        print(f"Snapshots saved in: {snapshot_dir}")

    return result

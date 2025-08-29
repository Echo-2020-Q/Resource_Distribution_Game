import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
from multiprocessing import Process, Queue
import atexit

# ============ 基本设定 ============
C, D, L = 0, 1, 2  # 策略编码：Cooperator, Defector, Loner



@dataclass
class Params:
    Lsize: int = 50                 # 格子边长 L，环境大小 L x L（周期边界）
    T: int = 2000                    # 演化总回合数
    fc: float = 0.5                 # 初始 C 比例（仅 CD 模式时用 fc, fd；CDL 模式时用 fc, fd, fL）
    fd: float = 0.5
    fL: Optional[float] = None      # 若 None => 仅 CD；否则 => CDL，且 fc+fd+fL=1
    pi0: float = 5.0                # 初始资源 π0（所有玩家相同）
    r: float = 1.6                  # 背叛指数（D 对 C 的诱惑收益）
    sigma: float = 0.25             # Loner 的单边互动收益 σ（每个邻边都拿 σ）
    alpha:float = 1.0               # 每回合的固定损耗参数
    beta:float=0.1                  #超出阈值后每回合的消费
    Min_Res:float = -36             #最低的资源值
    gamma:float= 10                  #阈值
    kappa: float = 0.1              # 费米学习规则 κ
    learn_signal: str = "cumulative"     # "round"=学本回合收益；"cumulative"=学累计资源
    seed: Optional[int] = 42        # 随机种子（可设为 None）
    measure_every: int = 1          # 每多少回合记录一次统计
    allow_forced_l_in_cd: bool = False  # 若仅 CD 初始，也允许资源耗尽者强制变 L
    snapshot_every: int = 50        # 每隔多少回合保存一次快照（<=0 代表不保存）
    snapshot_dir: str = "snapshots"  # 快照保存目录
    show_snapshots: bool = False  # True: plt.show()；False: 仅保存文件

# ============ 工具函数 ============
def _viewer_process(q: Queue):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    while True:
        fname = q.get()
        if fname is None:
            break

        arr = plt.imread(fname)
        h, w = arr.shape[:2]   # 图像高、宽（像素）

        # ==== 自动调节窗口大小 ====
        dpi = 100              # 分辨率，可改大/小
        scale = 1.0            # 比例因子，调节窗口大小
        figsize = (w / dpi * scale, h / dpi * scale)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(arr)
        ax.axis("off")
        ax.set_title(fname)

        plt.show()  # 阻塞，直到关闭窗口




def build_payoff_matrix(include_loner: bool, r: float, sigma: float) -> np.ndarray:
    """
    返回行玩家的收益矩阵 M，M[row_strategy, col_strategy]。
    两策略：[[1, 0],
            [r, 0]]
    三策略：[[1, 0, 0],
            [r, 0, 0],
            [sigma, sigma, sigma]]
    """
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

def torus_idx(i: int, L: int) -> int:
    """周期边界索引修正。"""
    return i % L

def von_neumann_neighbors(i: int, j: int, L: int) -> List[Tuple[int, int]]:
    """四邻域（上、下、左、右），周期边界。"""
    return [
        (torus_idx(i-1, L), j),
        (torus_idx(i+1, L), j),
        (i, torus_idx(j-1, L)),
        (i, torus_idx(j+1, L)),
    ]

def gini_coefficient(x: np.ndarray) -> float:
    """计算 Gini 系数（x >= 0）。"""
    x = x.flatten().astype(float)
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# ============ 主仿真类 ============
class SpatialGame:
    def __init__(self, p: Params):
        self.p = p
        if p.seed is not None:
            np.random.seed(p.seed)
        self.L = p.Lsize

        self.include_loner = (p.fL is not None)
        self.M = build_payoff_matrix(self.include_loner or p.allow_forced_l_in_cd, p.r, p.sigma)
        # 说明：即使初始仅 CD，若允许强制转 L，也需要 3x3 矩阵支持 L 的出现

        # 初始化策略格
        self.strat = self._init_strategies()
        # 上一回合策略（用于“上一回合不是L且破产→L”的判定）
        self.prev_strat = self.strat.copy()
        # 本回合“刚被强制转L”的掩码
        self.just_forced_L = np.zeros((self.L, self.L), dtype=bool)

        # 初始化资源
        self.res = np.full((self.L, self.L), fill_value=p.pi0, dtype=float)

        # 统计记录
        self.history: Dict[str, List[float]] = {
            "t": [],
            "frac_C": [],
            "frac_D": [],
            "frac_L": [],
            "avg_res": [],
            "gini": [],
        }
        # 策略可视化颜色：C=蓝, D=红, L=黄
        # 注意：仅当存在 L（含强制转 L）时才会用到黄色
        self.cmap_strat_cd  = ListedColormap(["blue", "red"])
        self.cmap_strat_cdl = ListedColormap(["blue", "red", "yellow"])

        # 准备输出目录
        if self.p.snapshot_every and self.p.snapshot_every > 0:
            os.makedirs(self.p.snapshot_dir, exist_ok=True)
            # —— 用于“后台看图”的队列与进程（只在 show_snapshots=True 时启用）——
        self._view_q: Optional[Queue] = None
        self._viewer: Optional[Process] = None
        if self.p.show_snapshots:
            self._view_q = Queue(maxsize=8)
            self._viewer = Process(target=_viewer_process, args=(self._view_q,), daemon=True)
            self._viewer.start()

            # 退出时优雅关闭
            def _cleanup():
                try:
                    if self._view_q is not None:
                        self._view_q.put(None)
                except Exception:
                    pass

            atexit.register(_cleanup)


    def _init_strategies(self) -> np.ndarray:
        L2 = self.L * self.L
        if self.include_loner:
            assert abs(self.p.fc + self.p.fd + self.p.fL - 1.0) < 1e-8, "fc+fd+fL must be 1."
            counts = [int(self.p.fc * L2), int(self.p.fd * L2)]
            counts.append(L2 - sum(counts))  # 剩余给 L，避免四舍五入误差
            pool = np.array([C]*counts[0] + [D]*counts[1] + [L]*counts[2], dtype=int)
        else:
            assert abs(self.p.fc + self.p.fd - 1.0) < 1e-8, "fc+fd must be 1."
            pool = np.array([C]*int(self.p.fc * L2) + [D]*(L2 - int(self.p.fc * L2)), dtype=int)

        np.random.shuffle(pool)
        return pool.reshape(self.L, self.L)

    def _compute_round_payoffs(self) -> np.ndarray:
        """
        计算每个个体本回合总收益（与四邻域逐对交互，行玩家视角）。
        注意：这里每条边会被两个端点各自计算一次（方向不同），恰好分别对应行玩家与列玩家的收益。
        """
        payoff = np.zeros((self.L, self.L), dtype=float)
        for i in range(self.L):
            for j in range(self.L):
                s_ij = self.strat[i, j]
                for ni, nj in von_neumann_neighbors(i, j, self.L):
                    s_nb = self.strat[ni, nj]
                    payoff[i, j] += self.M[s_ij, s_nb]
        return payoff  # 与 4 个邻居的“行收益”总和

    def _fermi_update(self, round_payoff: np.ndarray, frozen_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        费米学习（同步更新）。frozen_mask=True 的格子本回合不更新策略（保持原样）。
        """
        kappa = max(self.p.kappa, 1e-8)
        next_strat = self.strat.copy()

        # 选择学习信号
        if self.p.learn_signal == "round":
            S = round_payoff
        elif self.p.learn_signal == "cumulative":
            S = self.res
        else:
            raise ValueError("learn_signal must be 'round' or 'cumulative'")

        # 若未提供冻结掩码，视为全 False
        if frozen_mask is None:
            frozen_mask = np.zeros_like(self.strat, dtype=bool)

        for i in range(self.L):
            for j in range(self.L):
                # 本回合刚被强制转L的个体：跳过更新
                if frozen_mask[i, j]:
                    next_strat[i, j] = L
                    continue

                # （可选）若你还想让已经是 L 的个体也不学习，可在此处再加：
                # if self.strat[i, j] == L:
                #     next_strat[i, j] = L
                #     continue

                # 随机挑邻居
                ni, nj = von_neumann_neighbors(i, j, self.L)[np.random.randint(4)]
                Si, Sj = S[i, j], S[ni, nj]

                delta = np.clip(Si - Sj, -50.0, 50.0)
                p_adopt = 1.0 / (1.0 + np.exp(delta / kappa))

                if np.random.rand() < p_adopt:
                    next_strat[i, j] = self.strat[ni, nj]

        return next_strat

    def _force_loners_if_broke(self):
        """
           资源消耗后，若某个体在“上一回合不是 Loner”且当前资源<=0，
           则在允许出现 L 的前提下（include_loner 或 allow_forced_l_in_cd），将其强制转为 L。
           同时将其资源钉为 0。
           """
        broke_mask = (self.res <= 0.0)

        # 先清空上一轮的标记
        self.just_forced_L[:] = False

        if np.any(broke_mask):
            # 仅针对上一回合不是 L 的个体
            if hasattr(self, "prev_strat"):
                need_force_mask = broke_mask & (self.prev_strat != L)
            else:
                # 兜底：若没有 prev_strat（正常不会发生），则只看 broke
                need_force_mask = broke_mask

            if self.include_loner or self.p.allow_forced_l_in_cd:
                # 允许出现 L 才进行强制转化
                self.strat[need_force_mask] = L
                # 记录“刚被强制转L”的个体
                self.just_forced_L[need_force_mask] = True
            # 无论是否允许出现 L，都把破产者资源钉成 0（避免负值）
            self.res[broke_mask] = 0.0

    def _stats(self, t: int):
        L2 = self.L * self.L
        frac_C = np.mean(self.strat == C)
        frac_D = np.mean(self.strat == D)
        frac_L = np.mean(self.strat == L) if (self.include_loner or self.p.allow_forced_l_in_cd) else 0.0
        avg_res = float(np.mean(self.res))
        gini = gini_coefficient(self.res.clip(min=0.0))
        self.history["t"].append(t)
        self.history["frac_C"].append(frac_C)
        self.history["frac_D"].append(frac_D)
        self.history["frac_L"].append(frac_L)
        self.history["avg_res"].append(avg_res)
        self.history["gini"].append(gini)

    def run(self) -> Dict[str, List[float]]:


        # 初始统计
        self._stats(t=0)
        if self.p.snapshot_every and self.p.snapshot_every > 0:
            self._snapshot(t=0)

        for t in range(1, self.p.T + 1):
            # 先存旧资源，用于“跌破下限则不变”的回退
            res_old = self.res.copy()

            # 存下上一回合的策略，用于“非 L 且破产 → 强制转 L”
            self.prev_strat = self.strat.copy()

            # (1) 本回合逐对收益
            # (1) 本回合逐对收益
            round_payoff = self._compute_round_payoffs()
            res_after_gain = res_old + round_payoff

            # (2) 统一消耗（阈值规则）
            cost = self.p.gamma
            low_mask = (res_after_gain <= self.p.gamma)  # 低资源：减固定 cost
            res_after_consume = np.where(low_mask,
                                         res_after_gain - cost,
                                         (1 - self.p.beta) * res_after_gain)  # 高资源：按比例扣
            # (2.5) 应用“有下限但不硬夹”的规则：
            # 若这一步更新会让资源 < Min_Res，则保持为旧值 res_old；否则采用新值
            self.res = np.where(res_after_consume < self.p.Min_Res, res_old, res_after_consume)

            # (3) 强制转 L
            if self.p.allow_forced_l_in_cd:
                self._force_loners_if_broke()

            # (4) 费米学习
            next_strat = self._fermi_update(round_payoff, frozen_mask=self.just_forced_L)
            self.strat = next_strat


            # (5) 记录
            if (t % self.p.measure_every) == 0:
                self._stats(t=t)

            # (6) 快照
            if self.p.snapshot_every and (t % self.p.snapshot_every == 0):
                # 先即时打印（不会被看图阻塞）
                frac_C = float(np.mean(self.strat == C))
                frac_D = float(np.mean(self.strat == D))
                frac_L = float(np.mean(self.strat == L)) if (self.include_loner or self.p.allow_forced_l_in_cd) else 0.0
                avg_res = float(np.mean(self.res))
                gini = gini_coefficient(self.res.clip(min=0.0))
                print(
                    f"[snapshot @ t={t}] C={frac_C:.3f} D={frac_D:.3f} L={frac_L:.3f} avg_res={avg_res:.3f} gini={gini:.3f}")

                # 再把图交给子进程显示
                self._snapshot(t=t)

        return self.history

    def _snapshot(self, t: int):


        """保存策略分布（左）与资源热力图（右）的联合快照。"""
        # 选择策略色表
        if self.include_loner or self.p.allow_forced_l_in_cd:
            cmap_strat = self.cmap_strat_cdl
            vmax_strat = 2  # 0,1,2
        else:
            cmap_strat = self.cmap_strat_cd
            vmax_strat = 1  # 0,1

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        # 左：策略分布
        im0 = axes[0].imshow(self.strat, vmin=0, vmax=vmax_strat, interpolation='nearest', cmap=cmap_strat)
        axes[0].set_title(f"Strategies @ t={t}")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # 右：资源热力图（稳健处理）
        res_min = float(self.res.min())
        res_max = float(self.res.max())

        # 同值特判：给一个极小范围避免除零/排序错误
        if (res_min - res_max)<1e-6:
            res_min -= 1
            res_max += 1

        # 决定用哪个规范化
        if res_min < 0.0 and res_max > 0.0:
            # 跨零：以 0 为中心的双坡度
            norm = TwoSlopeNorm(vmin=res_min, vcenter=0.0, vmax=res_max)
            im1 = axes[1].imshow(self.res, norm=norm, cmap='RdBu_r', interpolation='nearest')
            axes[1].set_title(f"Resources (center=0)  min={res_min:.2f}, max={res_max:.2f}")
        else:
            # 同号：不要用 TwoSlopeNorm，直接普通区间
            im1 = axes[1].imshow(self.res, vmin=res_min, vmax=res_max, interpolation='nearest')
            axes[1].set_title(f"Resources  min={res_min:.2f}, max={res_max:.2f}")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("resource")

        # 关闭科学计数法的缩放标注
        cbar.ax.yaxis.get_offset_text().set_visible(False)
        cbar.ax.ticklabel_format(style="plain")  # 强制用普通数字

        # 文件名与保存
        fname = os.path.join(self.p.snapshot_dir, f"snapshot_t{t:05d}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)  # 立刻关闭，避免主进程占着窗口

        # —— 关键：把文件名交给子进程弹窗；主循环可以继续 print/计算 ——
        if self.p.show_snapshots and self._view_q is not None:
            try:
                self._view_q.put_nowait(fname)
            except Exception:
                # 队列满也无妨，忽略这个快照的弹窗
                pass


# ============ 示例用法 ============
if __name__ == "__main__":
    # 例 1：仅 C、D 起始；允许破产者强制转 L；学“本回合收益”
    p_cd = Params(
        Lsize=60, T=10000,
        fc=0.5, fd=0.5, fL=None,     # fL=None => 仅 CD 起始
        pi0=10,                      #初始资源
        r=1.8, sigma=0.25,           #r是背叛诱惑系数 sigma是Loner的低保收入
        kappa=0.1,                   #费米学习因子
        learn_signal="cumulative",
        seed=0,
        measure_every=10, #每多少回合记录一次统计
        allow_forced_l_in_cd=False,   # 资源耗尽者将变为 L
        snapshot_every=20,  # 每 20 回合一张
        snapshot_dir="snapshots_cd",  # 输出目录
        show_snapshots=True  # 不弹窗，只保存文件
    )
    sim_cd = SpatialGame(p_cd)
    hist_cd = sim_cd.run()
    print("[CD-start] final fractions:",
          f"C={hist_cd['frac_C'][-1]:.3f}",
          f"D={hist_cd['frac_D'][-1]:.3f}",
          f"L={hist_cd['frac_L'][-1]:.3f}",
          f"avg_res={hist_cd['avg_res'][-1]:.3f}",
          f"gini={hist_cd['gini'][-1]:.3f}")

    # 例 2：C、D、L 起始；学“累计资源”
    # p_cdl = Params(
    #     Lsize=60, T=5000,
    #     fc=0.5, fd=0.3, fL=0.2,      # fc+fd+fL=1
    #     pi0=5,
    #     r=1.6, sigma=0.25,
    #     kappa=0.1,
    #     learn_signal="cumulative",
    #     seed=1,
    #     measure_every=10,
    #     snapshot_every = 20,  # 每 20 回合一张
    #     snapshot_dir = "snapshots_cd",  # 输出目录
    #     show_snapshots = True  # 不弹窗，只保存文件
    # )
    # sim_cdl = SpatialGame(p_cdl)
    # hist_cdl = sim_cdl.run()
    # print("[CDL-start] final fractions:",
    #       f"C={hist_cdl['frac_C'][-1]:.3f}",
    #       f"D={hist_cdl['frac_D'][-1]:.3f}",
    #       f"L={hist_cdl['frac_L'][-1]:.3f}",
    #       f"avg_res={hist_cdl['avg_res'][-1]:.3f}",
    #       f"gini={hist_cdl['gini'][-1]:.3f}")

# sweep_parallel.py
# -*- coding: utf-8 -*-
import os, csv, time, itertools
import numpy as np
import multiprocessing as mp
import tqdm
from expOne_Three_fast_final_stats import Params, SpatialGameFast  # ← 用你刚才那份加速版

# ---- 1) 定义你想扫的参数网格（自行改动） ----
GRID = {
    "r":                    [1.2, 1.4, 1.6, 1.8],
    "allow_forced_l_in_cd": [True , False],

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
REPLICAS = 8

# 并行进程数（不给就用 CPU 核心数）
N_PROCS = min(8, os.cpu_count() or 1)

# ---- 2) 生成参数组合 ----
def iter_param_combos(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

# ---- 3) 单次 run（带参数合并）----
def run_one_config(args):
    """args: (config_dict, base_params, replicas, seed_base)"""
    cfg, base, replicas, seed0 = args
    # 合并参数：覆盖 BASE 中对应字段
    p_dict = {**base.__dict__, **cfg}
    # 注意：allow_forced_l_in_cd 若在 cfg 里，需确保 include_loner 逻辑符合你的需求

    # 多副本
    results = []
    for k in range(replicas):
        p_k = Params(**{**p_dict, "seed": (None if seed0 is None else seed0 + k)})
        out = SpatialGameFast(p_k).run()
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
    metric_keys = ["frac_C_mean", "frac_D_mean", "frac_L_mean", "avg_res_mean", "gini_mean","cum_sigma", "net_sum_res"]
    fieldnames = param_keys + metric_keys

    os.makedirs("sweep_out", exist_ok=True)

    # 按 r 值排序
    rows_sorted = sorted(rows, key=lambda d: d.get("r", 0.0))

    # 构造逻辑化文件名
    grid_name = "-".join(param_keys)              # e.g. "allow_forced_l_in_cd-r"
    timestamp = time.strftime("%Y%m%d-%H%M%S")    # e.g. "20250904-1130"
    csv_name = f"sweep_{grid_name}_T{BASE.T}_rep{REPLICAS}_{timestamp}.csv"
    csv_path = os.path.join("sweep_out", csv_name)

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

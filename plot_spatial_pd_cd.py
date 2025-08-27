#!/usr/bin/env python3
"""
Plotting script for 2-strategy spatial Prisoner's Dilemma (C/D)
- Runs Q-learning (with neighbor reward sharing) and Fermi baseline
- Saves time series CSV and two PNG figures
Usage:
    python plot_spatial_pd_cd.py \
        --L 80 --T 5000 --record-every 10 --b 1.6 --seed 7 \
        --alpha 0.1 --gamma 0.9 --epsilon 0.02 --alpha-r 0.7 --K 0.1 \
        --module spatial_pd_cd.py \
        --out-prefix results_spatial_pd_cd
"""
import argparse
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_module(module_path: str):
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("spatial_pd_cd", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spatial_pd_cd"] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=80)
    parser.add_argument("--T", type=int, default=5000)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--b", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=7)
    # Q-learning params
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--alpha-r", type=float, default=0.7, dest="alpha_r")
    # Fermi param
    parser.add_argument("--K", type=float, default=0.1)
    # IO
    parser.add_argument("--module", type=str, default="spatial_pd_cd.py")
    parser.add_argument("--out-prefix", type=str, default="results_spatial_pd_cd")
    args = parser.parse_args()

    mod = load_module(args.module)

    # Q-learning experiment
    p_q = mod.PDParams(L=args.L, b=args.b, rule="qlearning",
                       alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                       alpha_r=args.alpha_r, seed=args.seed)
    env_q = mod.SpatialPD(p_q)
    out_q = env_q.run(T=args.T, record_every=args.record_every)

    # Fermi experiment
    p_f = mod.PDParams(L=args.L, b=args.b, rule="fermi",
                       K=args.K, seed=args.seed)
    env_f = mod.SpatialPD(p_f)
    out_f = env_f.run(T=args.T, record_every=args.record_every)

    t = np.arange(args.record_every, args.T + 1, args.record_every)

    # Save CSV
    csv_path = f"{args.out_prefix}_timeseries.csv"
    df = pd.DataFrame({
        "t": t,
        "fC_qlearning": out_q["f_C"],
        "cb_qlearning": out_q["chessboard"],
        "fC_fermi": out_f["f_C"],
        "cb_fermi": out_f["chessboard"]
    })
    df.to_csv(csv_path, index=False)

    # Plot: cooperation fraction
    plt.figure()
    plt.plot(t, out_q["f_C"], label="Q-learning")
    plt.plot(t, out_f["f_C"], label="Fermi")
    plt.xlabel("Time (steps)")
    plt.ylabel("Cooperation fraction")
    plt.title("Cooperation fraction over time")
    plt.legend()
    png1 = f"{args.out_prefix}_fC_over_time.png"
    plt.savefig(png1, dpi=160, bbox_inches="tight")
    plt.close()

    # Plot: chessboard ratio
    plt.figure()
    plt.plot(t, out_q["chessboard"], label="Q-learning")
    plt.plot(t, out_f["chessboard"], label="Fermi")
    plt.xlabel("Time (steps)")
    plt.ylabel("Chessboard ratio")
    plt.title("Chessboard ratio over time")
    plt.legend()
    png2 = f"{args.out_prefix}_chessboard_over_time.png"
    plt.savefig(png2, dpi=160, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(" -", csv_path)
    print(" -", png1)
    print(" -", png2)

if __name__ == "__main__":
    main()

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

C, D = 1, 0  # Strategy encoding

@dataclass
class PDParams:
    L: int = 100
    b: float = 1.6
    rule: str = "qlearning"      # 'qlearning' or 'fermi'
    alpha: float = 0.1           # Q-learning: learning rate
    gamma: float = 0.9           # Q-learning: discount
    epsilon: float = 0.02        # Q-learning: exploration
    alpha_r: float = 0.6         # Q-learning: neighbor reward sharing
    K: float = 0.1               # Fermi: temperature
    seed: Optional[int] = None

class SpatialPD:
    def __init__(self, params: PDParams):
        self.p = params
        if self.p.seed is not None:
            np.random.seed(self.p.seed)
        self.L = self.p.L
        self.strat = np.random.randint(0, 2, size=(self.L, self.L), dtype=np.int8)
        if self.p.rule == "qlearning":
            self.Q = np.zeros((self.L, self.L, 5, 2), dtype=np.float32)
            self.prev_state = self._neighbor_nC(self.strat)

    # ---------- neighborhood helpers ----------
    def _roll(self, arr, dx, dy):
        return np.roll(np.roll(arr, dx, axis=0), dy, axis=1)

    def _neighbor_nC(self, strat: np.ndarray) -> np.ndarray:
        up    = self._roll(strat, -1,  0)
        down  = self._roll(strat,  1,  0)
        left  = self._roll(strat,  0, -1)
        right = self._roll(strat,  0,  1)
        return up + down + left + right

    def _payoffs(self, strat: np.ndarray) -> np.ndarray:
        nC = self._neighbor_nC(strat)
        payoff = np.where(strat == C, nC * 1.0, nC * self.p.b)
        return payoff

    def _avg_neighbor_payoff(self, payoff: np.ndarray) -> np.ndarray:
        up    = self._roll(payoff, -1,  0)
        down  = self._roll(payoff,  1,  0)
        left  = self._roll(payoff,  0, -1)
        right = self._roll(payoff,  0,  1)
        return (up + down + left + right) / 4.0

    # ---------- metrics ----------
    def cooperation_fraction(self) -> float:
        return float(self.strat.mean())

    def chessboard_ratio(self) -> float:
        s = self.strat
        up    = self._roll(s, -1,  0)
        down  = self._roll(s,  1,  0)
        left  = self._roll(s,  0, -1)
        right = self._roll(s,  0,  1)
        opp = 1 - s
        mask = (up == opp) & (down == opp) & (left == opp) & (right == opp)
        return float(mask.mean())

    # ---------- step ----------
    def step(self):
        if self.p.rule == "qlearning":
            return self._step_qlearning()
        elif self.p.rule == "fermi":
            return self._step_fermi()
        else:
            raise ValueError("Unknown rule: " + self.p.rule)

    def _step_qlearning(self) -> Dict[str, float]:
        s_t = self.prev_state
        a_t = self.strat
        payoff = self._payoffs(a_t)
        avg_nb = self._avg_neighbor_payoff(payoff)
        r_t = (1.0 - self.p.alpha_r) * payoff + self.p.alpha_r * avg_nb
        s_next = self._neighbor_nC(a_t)

        maxQ_next = np.max(self.Q[np.arange(self.L)[:,None], np.arange(self.L)[None,:], s_next, :], axis=-1)
        Qa = self.Q[np.arange(self.L)[:,None], np.arange(self.L)[None,:], s_t, a_t]
        Qa_new = (1 - self.p.alpha) * Qa + self.p.alpha * (r_t + self.p.gamma * maxQ_next)
        self.Q[np.arange(self.L)[:,None], np.arange(self.L)[None,:], s_t, a_t] = Qa_new

        Qnext_all = self.Q[np.arange(self.L)[:,None], np.arange(self.L)[None,:], s_next, :]
        greedy_action = (Qnext_all[...,1] > Qnext_all[...,0]).astype(np.int8)
        explore = (np.random.rand(self.L, self.L) < self.p.epsilon)
        random_action = np.random.randint(0, 2, size=(self.L, self.L), dtype=np.int8)
        a_next = np.where(explore, random_action, greedy_action).astype(np.int8)
        self.strat = a_next
        self.prev_state = self._neighbor_nC(self.strat)
        return {"f_C": float(self.strat.mean()), "chessboard": self.chessboard_ratio(), "avg_payoff": float(payoff.mean())}

    def _step_fermi(self) -> Dict[str, float]:
        payoff = self._payoffs(self.strat)

        up_s    = self._roll(self.strat, -1,  0)
        down_s  = self._roll(self.strat,  1,  0)
        left_s  = self._roll(self.strat,  0, -1)
        right_s = self._roll(self.strat,  0,  1)
        nb_s_stack = np.stack([up_s, down_s, left_s, right_s], axis=0)

        up_p    = self._roll(payoff, -1,  0)
        down_p  = self._roll(payoff,  1,  0)
        left_p  = self._roll(payoff,  0, -1)
        right_p = self._roll(payoff,  0,  1)
        nb_p_stack = np.stack([up_p, down_p, left_p, right_p], axis=0)

        dirs = np.random.randint(0, 4, size=(self.L, self.L))
        sel_idx = dirs[None, :, :]
        nb_strat = np.take_along_axis(nb_s_stack, sel_idx, axis=0)[0]
        nb_pay   = np.take_along_axis(nb_p_stack, sel_idx, axis=0)[0]

        diff = nb_pay - payoff
        prob = 1.0 / (1.0 + np.exp(-diff / max(self.p.K, 1e-8)))
        flip = (np.random.rand(self.L, self.L) < prob)

        new_strat = self.strat.copy()
        new_strat[flip] = nb_strat[flip]
        self.strat = new_strat
        return {"f_C": float(self.strat.mean()), "chessboard": self.chessboard_ratio(), "avg_payoff": float(payoff.mean())}

    # ---------- run ----------
    def run(self, T: int, record_every: int = 1) -> Dict[str, np.ndarray]:
        rec_fC, rec_cb, rec_pay = [], [], []
        for t in range(1, T+1):
            stats = self.step()
            if t % record_every == 0:
                rec_fC.append(stats["f_C"])
                rec_cb.append(stats["chessboard"])
                rec_pay.append(stats["avg_payoff"])
        return {"f_C": np.array(rec_fC, dtype=float),
                "chessboard": np.array(rec_cb, dtype=float),
                "avg_payoff": np.array(rec_pay, dtype=float)}

if __name__ == "__main__":
    p = PDParams(L=50, b=1.6, rule="qlearning", alpha=0.1, gamma=0.9, epsilon=0.02, alpha_r=0.7, seed=42)
    env = SpatialPD(p)
    out = env.run(T=500, record_every=1)
    print("Q-learning demo done. Final f_C =", out["f_C"][-1])

    p2 = PDParams(L=50, b=1.6, rule="fermi", K=0.1, seed=42)
    env2 = SpatialPD(p2)
    out2 = env2.run(T=500, record_every=1)
    print("Fermi demo done. Final f_C =", out2["f_C"][-1])

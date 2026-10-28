"""Verbosity policies: fixed baselines, pooled, personalized, and risk-constrained.

The adaptive policies keep a belief over each user's preferred level (a grid
posterior) and update it from weak feedback only: "show more" requests,
"too long" complaints and occasional ratings. They never see the true
preference. Their *model* of the user is deliberately a little wrong
(different penalties and signal rates than the simulator) so results are not
an artifact of perfect knowledge. They do know the completeness floor
TYPE_MIN (the shortest level that still contains the answer); in a real
system that would come from an answer-quality check.
"""
import numpy as np
from sim import TOKENS, DEFAULT, TYPE_MIN, sigmoid

GRID = np.arange(1.0, 5.01, 0.5)          # candidate preferred levels
LEVELS = np.arange(1, 6)

# The policy's assumed user model (differs from sim.SimParams on purpose).
A_UNDER, A_OVER, A_INCOMPLETE = 0.9, 0.7, 1.5
A_MORE_P, A_MORE_K = 0.5, 2.5
A_LONG_P, A_LONG_K = 0.3, 2.5
A_RATING_SD = 0.6


def assumed_sat(level, pref, qtype):
    s = (4.8 - A_UNDER * np.maximum(pref - level, 0.0)
         - A_OVER * np.maximum(level - pref, 0.0)
         - A_INCOMPLETE * np.maximum(TYPE_MIN[qtype] - level, 0.0))
    return np.clip(s, 1.0, 5.0)


# S[q][l-1, j] = assumed satisfaction at level l if the user's preference is GRID[j]
_S = [np.array([[assumed_sat(l, g, q) for g in GRID] for l in LEVELS]) for q in range(6)]


def _prior(mean, sd):
    p = np.exp(-0.5 * ((GRID - mean) / sd) ** 2)
    return p / p.sum()


def _normalize(p):
    p = p + 1e-12
    return p / p.sum()


class FixedPolicy:
    def __init__(self, level):
        self.level = level
        self.name = f"fixed-{level}"

    def choose(self, uid, q, user=None):
        return self.level

    def update(self, uid, q, obs):
        pass


class OraclePolicy:
    """Knows the true preference. An upper bound, not a deployable policy."""

    def __init__(self, params, lam):
        self.P, self.lam = params, lam
        self.name = "oracle"

    def choose(self, uid, q, user=None):
        from sim import mean_sat
        u = [mean_sat(l, user.pref[q], q, self.P) - self.lam * TOKENS[l - 1] / 100 for l in LEVELS]
        return int(LEVELS[int(np.argmax(u))])

    def update(self, uid, q, obs):
        pass


class AdaptivePolicy:
    """mode: 'pooled' (one belief per query type, shared by everyone),
    'personalized' (per user x type, shrunk toward the pooled belief),
    'safe' (personalized + a dissatisfaction-risk constraint relative to DEFAULT)."""

    def __init__(self, mode="personalized", lam=0.15, eps=0.1, delta=0.4, risk=0.25,
                 forget=0.97, prior_sd=1.2, shrink=0.7, name=None):
        self.mode, self.lam, self.eps, self.delta, self.risk = mode, lam, eps, delta, risk
        self.forget, self.prior_sd, self.shrink = forget, prior_sd, shrink
        self.name = name or mode
        self.pooled = {q: _prior(3.0, 1.5) for q in range(6)}
        self.beliefs = {}
        self.counts = {}

    # ---- beliefs ----------------------------------------------------------
    def _mean(self, b):
        return float(b @ GRID)

    def _offset(self, uid):
        res = [self._mean(self.beliefs[(uid, q)]) - self._mean(self.pooled[q])
               for q in range(6) if (uid, q) in self.beliefs and self.counts[(uid, q)] >= 3]
        return self.shrink * float(np.mean(res)) if res else 0.0

    def _belief(self, uid, q):
        if self.mode == "pooled":
            return self.pooled[q]
        key = (uid, q)
        if key not in self.beliefs:
            m = self._mean(self.pooled[q]) + self._offset(uid)
            self.beliefs[key] = _prior(m, self.prior_sd)
            self.counts[key] = 0
        return self.beliefs[key]

    # ---- decision ---------------------------------------------------------
    def choose(self, uid, q, user=None):
        b = self._belief(uid, q)
        S = _S[q]
        lo = int(TYPE_MIN[q])
        util = S @ b - self.lam * TOKENS / 100.0
        allowed = LEVELS >= lo
        if self.mode == "safe":
            d = max(DEFAULT, lo)
            diff = S - S[d - 1]
            drop_prob = (diff < -self.delta).astype(float) @ b
            exp_diff = diff @ b
            ok = (exp_diff >= -self.eps) & (drop_prob <= self.risk)
            ok[d - 1] = True
            allowed = allowed & ok
        util = np.where(allowed, util, -np.inf)
        return int(LEVELS[int(np.argmax(util))])

    # ---- learning ---------------------------------------------------------
    def _lik(self, obs, q):
        L = obs["shown"]
        lik = np.ones_like(GRID)
        if L < 5:
            p = A_MORE_P * sigmoid(A_MORE_K * (GRID - L - 0.4))
            lik *= p if obs["needs_more"] else (1 - p)
        if L > 1:
            p = A_LONG_P * sigmoid(A_LONG_K * (L - GRID - 0.8))
            lik *= p if obs["too_long"] else (1 - p)
        if obs["rating"] is not None:
            mu = assumed_sat(obs["final"], GRID, q)
            lik *= np.exp(-0.5 * ((obs["rating"] - mu) / A_RATING_SD) ** 2)
        return lik

    def update(self, uid, q, obs):
        lik = self._lik(obs, q)
        f = self.forget if self.mode != "pooled" else 0.995
        self.pooled[q] = _normalize((self.pooled[q] ** f) * lik)
        if self.mode != "pooled":
            key = (uid, q)
            b = self._belief(uid, q)
            self.beliefs[key] = _normalize((b ** self.forget) * lik)
            self.counts[key] += 1
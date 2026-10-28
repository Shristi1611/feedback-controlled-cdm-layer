"""Simulated users with hidden, context-dependent answer-length preferences.

Everything here is *assumed* behaviour, not measured human data. The point is to
test whether a policy can trade tokens for satisfaction safely when users differ.
"""
from dataclasses import dataclass
import numpy as np

# Verbosity levels 1..5 and the tokens an answer costs at each level.
TOKENS = np.array([40, 100, 200, 380, 600], dtype=float)
DEFAULT = 3  # what a provider would serve to everyone today

QTYPES = ["quick_fact", "casual", "how_to", "explain", "code", "analysis"]
TYPE_BASE = np.array([1.5, 2.0, 3.0, 3.5, 3.0, 4.0])   # population-typical preferred level
TYPE_FREQ = np.array([0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
TYPE_MIN = np.array([1, 1, 2, 2, 2, 3])                 # lowest level that still contains the answer

ARCHETYPES = ["terse", "typical", "thorough", "contextual", "arbitrary"]
ARCH_P = np.array([0.20, 0.25, 0.20, 0.15, 0.20])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SimParams:
    under_pen: float = 1.0        # satisfaction lost per level the answer is too short
    over_pen: float = 0.55        # ... per level it is too long
    incomplete_pen: float = 1.5   # ... per level below the completeness floor
    noise: float = 0.35           # rating noise
    rating_prob: float = 0.08     # chance the user leaves an explicit rating
    expand_enabled: bool = True   # "show more" button (reversible output compression)
    more_prob_expand: float = 0.70   # chance of asking for more when too short, with button
    more_prob_reask: float = 0.40    # ... without button (user must re-ask)
    expand_friction: float = 0.3
    reask_friction: float = 0.8
    toolong_prob: float = 0.35    # chance of an explicit "too long" signal
    drift_turn: int = -1          # if >=0, 30% of users shift preference by +-1 at this turn


def mean_sat(level, pref, qtype, P: SimParams):
    s = (4.8
         - P.under_pen * np.maximum(pref - level, 0.0)
         - P.over_pen * np.maximum(level - pref, 0.0)
         - P.incomplete_pen * np.maximum(TYPE_MIN[qtype] - level, 0.0))
    return np.clip(s, 1.0, 5.0)


class User:
    def __init__(self, rng, arch):
        self.arch = arch
        base = TYPE_BASE.copy()
        if arch == "terse":
            pref = base - 1.2 + rng.normal(0, 0.3, 6)
        elif arch == "thorough":
            pref = base + 1.2 + rng.normal(0, 0.3, 6)
        elif arch == "typical":
            pref = base + rng.normal(0, 0.3, 6)
        elif arch == "contextual":   # short for simple asks, long for hard ones
            simple = np.array([1, 1, 1, 0, 0, 0], bool)
            pref = np.where(simple, base - 1.0, base + 1.0) + rng.normal(0, 0.3, 6)
        else:                        # arbitrary: unrelated to any stereotype
            pref = rng.uniform(1, 5, 6)
        self.pref = np.clip(pref, 1.0, 5.0)

    def drift(self, rng):
        self.pref = np.clip(self.pref + rng.choice([-1.0, 1.0]), 1.0, 5.0)


def make_users(rng, n):
    archs = rng.choice(len(ARCHETYPES), size=n, p=ARCH_P)
    return [User(rng, ARCHETYPES[a]) for a in archs]


def respond(user, qtype, level, P: SimParams, rng):
    """One turn. Returns what the provider pays and what the policy can observe."""
    pref = user.pref[qtype]
    tokens = TOKENS[level - 1]
    final = level
    friction = 0.0
    needs_more = False
    if level < 5 and pref > level:
        p = (P.more_prob_expand if P.expand_enabled else P.more_prob_reask)
        if rng.random() < p * sigmoid(3.0 * (pref - level - 0.4)):
            needs_more = True
            final = level + 1
            if P.expand_enabled:
                tokens += TOKENS[final - 1] - TOKENS[level - 1]   # incremental text only
                friction = P.expand_friction
            else:
                tokens += TOKENS[final - 1]                       # a whole new answer
                friction = P.reask_friction
    too_long = False
    if level > 1 and level > pref:
        too_long = rng.random() < P.toolong_prob * sigmoid(3.0 * (level - pref - 0.8))
    exp_sat = mean_sat(final, pref, qtype, P) - friction
    sat = float(np.clip(exp_sat + rng.normal(0, P.noise), 1.0, 5.0))
    rating = sat if rng.random() < P.rating_prob else None
    return dict(tokens=float(tokens), exp_sat=float(exp_sat), sat=sat,
                needs_more=needs_more, too_long=too_long, rating=rating,
                shown=level, final=final)
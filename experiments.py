"""Run policies against simulated users and report tokens vs satisfaction."""
import sys
import numpy as np
from sim import (SimParams, make_users, respond, TYPE_FREQ, TOKENS, DEFAULT, ARCHETYPES)
from policies import FixedPolicy, OraclePolicy, AdaptivePolicy

N_USERS, N_TURNS = 150, 60


def run(policy_factory, params: SimParams, seed: int, n_users=N_USERS, n_turns=N_TURNS):
    """Same users and same query sequence for every policy given the same seed."""
    rng_world = np.random.default_rng(seed)
    users = make_users(rng_world, n_users)
    queries = rng_world.choice(6, size=(n_users, n_turns), p=TYPE_FREQ)
    drift_mask = rng_world.random(n_users) < 0.3
    rng_resp = np.random.default_rng(seed + 10_000)
    pol = policy_factory()
    rows = np.zeros((n_users, n_turns, 5))  # tokens, exp_sat, dissatisfied, level, needs_more
    for t in range(n_turns):
        if t == params.drift_turn:
            for u, user in enumerate(users):
                if drift_mask[u]:
                    user.drift(rng_world)
        for u, user in enumerate(users):
            q = int(queries[u, t])
            level = pol.choose(u, q, user)
            obs = respond(user, q, level, params, rng_resp)
            pol.update(u, q, obs)
            rows[u, t] = (obs["tokens"], obs["exp_sat"], float(obs["sat"] <= 2.5), level, float(obs["needs_more"]))
    return rows, [u.arch for u in users]


def summarize(rows):
    return dict(tokens=rows[..., 0].mean(), sat=rows[..., 1].mean(), dissat=rows[..., 2].mean() * 100,
                early_dissat=rows[:, :10, 2].mean() * 100, late_dissat=rows[:, 30:, 2].mean() * 100,
                early_sat=rows[:, :10, 1].mean(), late_sat=rows[:, 30:, 1].mean())


def evaluate(factories, params, seeds):
    out = {}
    for name, fac in factories.items():
        per = [summarize(run(fac, params, s)[0]) for s in seeds]
        out[name] = {k: (np.mean([p[k] for p in per]), np.std([p[k] for p in per])) for k in per[0]}
    return out


def table(res, base="default", title=""):
    print(f"\n{title}")
    b = res[base]
    hdr = f"{'policy':<14}{'tok/turn':>9}{'save%':>7}{'sat':>7}{'dsat%':>7}{'early dsat%':>12}{'late dsat%':>11}"
    print(hdr)
    print("-" * len(hdr))
    for n, r in res.items():
        save = 100 * (1 - r["tokens"][0] / b["tokens"][0])
        print(f"{n:<14}{r['tokens'][0]:>9.0f}{save:>7.1f}{r['sat'][0]:>7.2f}{r['dissat'][0]:>7.1f}"
              f"{r['early_dissat'][0]:>12.1f}{r['late_dissat'][0]:>11.1f}")


def factories(lam=0.15, params=None, **safe_kw):
    params = params or SimParams()
    return {
        "default": lambda: FixedPolicy(DEFAULT),
        "always-short": lambda: FixedPolicy(1),
        "always-long": lambda: FixedPolicy(5),
        "pooled": lambda: AdaptivePolicy("pooled", lam=lam),
        "personalized": lambda: AdaptivePolicy("personalized", lam=lam),
        "safe": lambda: AdaptivePolicy("safe", lam=lam, **safe_kw),
        "oracle": lambda: OraclePolicy(params, lam),
    }


if __name__ == "__main__":
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
    for expand in (True, False):
        P = SimParams(expand_enabled=expand)
        res = evaluate(factories(params=P), P, seeds)
        table(res, title=f"MAIN  (show-more button {'ON' if expand else 'OFF'}, {len(seeds)} seeds)")
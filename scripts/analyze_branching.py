"""Analyse the branching-validity experiment (scripts/branching_validity.py).

    python scripts/analyze_branching.py results/branching/raw.jsonl --out results/branching/summary.json

A "state" is one saved game (round R, index i); a "branch" is one re-seeded future from it.
Placements: 1 is best. HP+gold at +h rounds counts 0 for a hero already out.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

Z95, Z80 = 1.959964, 0.841621
INDEPENDENT_SD = 2.25  # SD of one hero placement over independent games (README; 200 games of hero=mimic: 2.24)


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    states = defaultdict(lambda: defaultdict(dict))  # (round, index) -> policy -> k -> row
    for row in rows:
        states[(row["round"], row["index"])][row["policy"]][row["k"]] = row
    return rows, states


def proxy(row, h, kind="hp+gold"):
    t = row["trajectory"][h - 1]
    if t is None:
        return 0.0
    return {"hp+gold": t["hp"] + t["gold"], "hp": t["hp"], "gold": t["gold"]}[kind]


def anova(groups):
    """One-way random-effects decomposition over equal-ish groups (list of 1-D arrays)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    k = np.mean([len(g) for g in groups])
    allv = np.concatenate(groups)
    within = float(np.mean([g.var(ddof=1) for g in groups]))
    means = np.array([g.mean() for g in groups])
    msb = float(k * means.var(ddof=1))
    between = max(0.0, (msb - within) / k)
    return {
        "states": len(groups), "branches_per_state": float(k), "n": int(allv.size),
        "mean": float(allv.mean()),
        "total_sd": float(allv.std(ddof=1)),
        "within_sd": math.sqrt(within),  # pooled SD of branches from one state
        "between_sd": math.sqrt(between),  # SD of the states' true means (sampling noise of K branches removed)
        "icc": between / (between + within) if between + within > 0 else None,  # share explained by the state
        "raw_share": float(means.var(ddof=0) / allv.var(ddof=0)) if allv.var() > 0 else None,  # uncorrected
    }


def bootstrap_icc(groups, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, float) for g in groups]
    vals = []
    for _ in range(reps):
        pick = [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        vals.append(anova(pick)["icc"])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort(kind="mergesort")
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v))
        for value in np.unique(v):  # average ties
            ranks[v == value] = ranks[v == value].mean()
        return ranks

    return corr(rank(x), rank(y))


def correlations(states, policies, h, kind="hp+gold", alive_only=False):
    xs, ys, xw, yw = [], [], [], []
    for pols in states.values():
        for pol in policies:
            rows = list(pols.get(pol, {}).values())
            if alive_only:
                rows = [r for r in rows if r["trajectory"][h - 1] is not None]
            if not rows:
                continue
            x = np.array([proxy(r, h, kind) for r in rows])
            y = np.array([r["place"] for r in rows], float)
            xs += list(x)
            ys += list(y)
            if len(rows) >= 2:
                xw += list(x - x.mean())
                yw += list(y - y.mean())
    return {"n": len(xs), "pearson": corr(xs, ys), "spearman": spearman(xs, ys),
            "within_state_pearson": corr(xw, yw)}


def policy_difference(states, a, b, metric):
    """Per state: mean over branches of metric(a) - metric(b), branches paired by re-seed k."""
    d, pair_var, paired_a, paired_b, changed = [], [], [], [], []
    for key, pols in sorted(states.items()):
        if a not in pols or b not in pols:
            continue
        ks = sorted(set(pols[a]) & set(pols[b]))
        if len(ks) < 2:
            continue
        va = np.array([metric(pols[a][k]) for k in ks], float)
        vb = np.array([metric(pols[b][k]) for k in ks], float)
        d.append(float(va.mean() - vb.mean()))
        pair_var.append(float((va - vb).var(ddof=1)))
        paired_a += list(va - va.mean())
        paired_b += list(vb - vb.mean())
        changed.append(any(pols[b][k]["plan_differs_from_mimic"] or pols[a][k]["plan_differs_from_mimic"] for k in ks))
    d = np.array(d)
    S = len(d)
    K = np.mean([len(set(p[a]) & set(p[b])) for p in states.values() if a in p and b in p])
    v_pair = float(np.mean(pair_var))  # variance of one paired branch difference within a state
    tau2 = max(0.0, float(d.var(ddof=1)) - v_pair / K)  # spread of the true effect across states
    return {
        "states": S, "branches_per_state": float(K),
        "mean": float(d.mean()), "ci95": float(Z95 * d.std(ddof=1) / math.sqrt(S)),
        "sd_of_state_differences": float(d.std(ddof=1)),
        "paired_branch_sd": math.sqrt(v_pair),
        "effect_sd_across_states": math.sqrt(tau2),
        "crn_correlation": corr(paired_a, paired_b),  # within-state correlation of a and b branches with the same k
        "states_where_plan_ever_differed": int(sum(changed)),
        "mean_where_plan_differed": float(d[np.array(changed)].mean()) if any(changed) else None,
        "_tau2": tau2, "_v_pair": v_pair,
    }


def sample_size(diff, half_width=0.3, ks=(1, 2, 5, 10, 20, 50, 100)):
    out = []
    for K in ks:
        var_state = diff["_tau2"] + diff["_v_pair"] / K
        s_ci = math.ceil(Z95 ** 2 * var_state / half_width ** 2)
        s_pow = math.ceil((Z95 + Z80) ** 2 * var_state / half_width ** 2)
        out.append({"branches_per_policy": K, "states_for_ci_halfwidth": s_ci, "playouts_ci": 2 * K * s_ci,
                    "states_for_80pct_power": s_pow, "playouts_power": 2 * K * s_pow})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows, states = load(args.raw)
    rounds = sorted({r for r, _ in states})
    policies = sorted({row["policy"] for row in rows})
    summary = {"rows": len(rows), "states": len(states), "rounds": rounds, "policies": policies}

    # timing and size of snapshots
    summary["snapshot"] = {
        "snapshot_seconds_mean": float(np.mean([r["snapshot_seconds"] for r in rows])),
        "snapshot_seconds_max": float(np.max([r["snapshot_seconds"] for r in rows])),
        "restore_seconds_mean": float(np.mean([r["restore_seconds"] for r in rows])),
        "restore_seconds_max": float(np.max([r["restore_seconds"] for r in rows])),
        "snapshot_mb_mean": float(np.mean([r["snapshot_bytes"] for r in rows]) / 1e6),
        "branch_seconds_mean_by_round": {R: float(np.mean([r["branch_seconds"] for r in rows if r["round"] == R]))
                                         for R in rounds},
    }

    # 1-2. spread of placement within a state, share explained by the state
    var = {}
    for pol in policies + ["all"]:
        for R in rounds + ["all"]:
            groups = [[row["place"] for row in pols[p].values()]
                      for (r, _), pols in states.items() if R in ("all", r)
                      for p in (policies if pol == "all" else [pol]) if p in pols]
            if len(groups) < 3:
                continue
            res = anova(groups)
            if pol == "mimic" or pol == "all":
                res["icc_ci95_bootstrap"] = bootstrap_icc(groups)
            var[f"{pol}/R{R}"] = res
    summary["placement_variance"] = var

    # 3. does hero HP+gold a few rounds after the branch predict the final placement?
    cor = {}
    for h in (1, 3, 5, 8):
        for kind in ("hp+gold", "hp", "gold"):
            cor[f"+{h} {kind}"] = correlations(states, policies, h, kind)
        cor[f"+{h} hp+gold alive only"] = correlations(states, policies, h, "hp+gold", alive_only=True)
    for R in rounds:
        sub = {k: v for k, v in states.items() if k[0] == R}
        cor[f"+5 hp+gold R{R}"] = correlations(sub, policies, 5)
    summary["proxy_correlation"] = cor

    # 4-5. mimic minus fast8, pooled over states, and what it would take to resolve 0.3
    if {"mimic", "fast8"} <= set(policies):
        diff = {}
        for R in rounds + ["all"]:
            sub = {k: v for k, v in states.items() if R in ("all", k[0])}
            diff[f"place R{R}"] = policy_difference(sub, "mimic", "fast8", lambda r: r["place"])
        diff["hp+gold +5 Rall"] = policy_difference(states, "mimic", "fast8", lambda r: proxy(r, 5))
        diff["hp+gold +3 Rall"] = policy_difference(states, "mimic", "fast8", lambda r: proxy(r, 3))
        summary["mimic_minus_fast8"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                        for k, v in diff.items()}
        summary["sample_size_for_0.3"] = {
            "with_common_random_numbers": sample_size(diff["place Rall"]),
            "independent_branches": sample_size({**diff["place Rall"],
                                                 "_v_pair": 2 * var["all/Rall"]["within_sd"] ** 2}),
            "independent_games_per_policy": math.ceil((Z95 * INDEPENDENT_SD * math.sqrt(2) / 0.3) ** 2),
        }
        fired = [bool(r["plan_differs_from_mimic"]) for r in rows if r["policy"] == "fast8"]
        summary["fast8_plan_ever_differs_share"] = float(np.mean(fired))
        summary["fast8_plan_differs_share_by_round"] = {
            R: float(np.mean([bool(r["plan_differs_from_mimic"]) for r in rows if r["policy"] == "fast8" and r["round"] == R]))
            for R in rounds}

    text = json.dumps(summary, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

"""Re-analysis of decision bank v1 (results/bank/realistic_set4.jsonl): how big are the true
differences between candidates, and what do the anchors score without reading the labels they are
scored on? These are the numbers quoted in README "正式题库 v1" and docs/BANK_V2.md.

    python scripts/bank1_effects.py results/bank/realistic_set4.jsonl

- tau: random-effects (method of moments) SD of the true per-state gap between two candidates:
  mean over candidate pairs of (mean paired diff)^2 - (paired variance / n), all branches of the item.
- Anchors scored on the label set relative to the discovery-best candidate, CI clustered by source
  game: `worst` as v1 defines it (reads the same label branches: winner's curse), the cross-fit
  worst and oracle (pick on one half of the label branches, score on the other, swap, average),
  and fixed candidates.
"""
import collections
import json
import statistics as st
import sys

import numpy as np


def load(path):
    items = {}
    for line in open(path):
        d = json.loads(line)
        items[d["id"]] = d  # a later line with the same id replaces the earlier one
    return [d for d in items.values() if not d.get("dropped")]


def places(item):
    out = collections.defaultdict(dict)
    for b in item["branches"]:
        out[b["cand"]][b["k"]] = b["place"]
    return out


def pair_stats(item):
    P = places(item)
    names = sorted(P)
    rows = []
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            ds = [P[a][k] - P[c][k] for k in P[a] if k in P[c]]
            if len(ds) >= 3:
                rows.append((st.mean(ds), st.variance(ds), len(ds)))
    return rows


def tau(rows):
    t2 = st.mean([m * m - v / n for m, v, n in rows])
    return max(t2, 0.0) ** 0.5, t2


def clustered(values, games):
    groups = collections.defaultdict(list)
    for v, g in zip(values, games):
        groups[g].append(v)
    tot = [sum(x) for x in groups.values()]
    n = [len(x) for x in groups.values()]
    m = sum(tot) / sum(n)
    r = np.array([t - m * k for t, k in zip(tot, n)])
    G = len(groups)
    return m, 1.96 * np.sqrt(G / (G - 1) * np.sum(r ** 2)) / sum(n)


def main(path):
    items = load(path)
    print(f"{path}: {len(items)} usable items")
    by_point = collections.defaultdict(list)
    for d in items:
        by_point[d["point"]] += pair_stats(d)
    print("\ntrue gap between two candidates (tau, places)")
    for p, rows in sorted(by_point.items()):
        print(f"  {p}: tau {tau(rows)[0]:.2f}  ({len(rows)} pairs)")
    late = [d for d in items if d["point"] != "3-2"]
    for name, key in [("hp", lambda d: d["public_state"]["hp"]), ("gold", lambda d: d["public_state"]["gold"]),
                      ("losses_to_death", lambda d: d["public_state"]["losses_to_death"])]:
        vals = sorted(key(d) for d in items)
        q1, q2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
        groups = collections.defaultdict(list)
        for d in late:
            x = key(d)
            groups["low" if x <= q1 else "mid" if x <= q2 else "high"] += pair_stats(d)
        print(f"  4-1/5-1 by {name} (cuts {q1}, {q2}): " +
              ", ".join(f"{g} {tau(r)[0]:.2f}" for g, r in sorted(groups.items())))
    groups = collections.defaultdict(list)
    for d in late:
        lm = d["label"]["label_mean"]
        mp = st.mean([v for v in lm.values() if v is not None])
        groups["top (<=3)" if mp <= 3 else "mid" if mp <= 5.5 else "bottom (>5.5)"] += pair_stats(d)
    print("  4-1/5-1 by state strength (mean label placement): " +
          ", ".join(f"{g} {tau(r)[0]:.2f}" for g, r in sorted(groups.items())))

    res = collections.defaultdict(list)
    games = []
    for d in items:
        P = places(d)
        names = list(P)
        kd, kl = d["kd"], d["kl"]
        lab = list(range(kd, kd + kl))
        best = d["label"]["best"]

        def mean(c, ks):
            return float(np.mean([P[c][k] for k in ks]))

        def regret(c, ks):
            return mean(c, ks) - mean(best, ks)

        halves = (lab[::2], lab[1::2])

        def crossfit(pick):
            return float(np.mean([regret(pick(names, key=lambda c: mean(c, S)), T)
                                  for S, T in (halves, halves[::-1])]))

        res["worst (v1: reads the label set)"].append(max(regret(c, lab) for c in names))
        res["worst (cross-fit)"].append(crossfit(max))
        res["oracle (cross-fit)"].append(crossfit(min))
        for c in ("save", "roll"):
            res[f"always:{c}"].append(regret(c, lab))
        res["always: the point's second candidate"].append(
            regret([c for c in names if c not in ("save", "roll")][0], lab))
        games.append(d["game"])
    print("\nanchors, regret vs the discovery-best on the label set (95% CI by source game)")
    for k, v in res.items():
        m, ci = clustered(v, games)
        print(f"  {k:38s} {m:+.2f} ±{ci:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/bank/realistic_set4.jsonl")

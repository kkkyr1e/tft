"""Placement statistics. The game, not the seat, is the independent unit: seats in one lobby
share a zero-sum outcome, so every interval here is computed over per-game values."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_Z95 = NormalDist().inv_cdf(0.975)


def _mean_ci(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"n": n, "mean": mean, "ci95": None, "low": None, "high": None}
    half = _Z95 * float(values.std(ddof=1)) / math.sqrt(n)
    return {"n": n, "mean": mean, "ci95": half, "low": mean - half, "high": mean + half}


def summarize(results: list[dict], policy: str) -> dict:
    """Average placement of every seat that ran `policy`, with a 95% interval over games."""
    per_game, top4, wins, places = [], [], [], []
    for result in results:
        if not result["finished"]:
            continue
        mine = [result["placements"][seat] for seat, name in result["lobby"].items() if name == policy]
        if not mine:
            continue
        per_game.append(np.mean(mine))
        top4.append(np.mean([place <= 4 for place in mine]))
        wins.append(float(1 in mine))
        places.extend(mine)
    if not per_game:
        raise ValueError(f"no finished game contains policy {policy!r}")
    summary = _mean_ci(np.asarray(per_game, dtype=float))
    summary.update(
        policy=policy,
        top4_rate=float(np.mean(top4)),
        games_won_rate=float(np.mean(wins)),
        histogram=np.bincount(places, minlength=9)[1:].tolist(),
    )
    return summary


def paired_diff(results_a: list[dict], results_b: list[dict], policy_a: str, policy_b: str) -> dict:
    """Difference in average placement (A minus B; negative means A is better) over games that
    share a seed. Use it for A/B runs where only the seats under test changed."""

    def by_seed(results, policy):
        table = {}
        for result in results:
            if not result["finished"]:
                continue
            mine = [result["placements"][s] for s, name in result["lobby"].items() if name == policy]
            if mine:
                table[result["seed"]] = float(np.mean(mine))
        return table

    a, b = by_seed(results_a, policy_a), by_seed(results_b, policy_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two runs share no seeds")
    diff = np.asarray([a[seed] - b[seed] for seed in shared])
    summary = _mean_ci(diff)
    va, vb = [a[seed] for seed in shared], [b[seed] for seed in shared]
    # correlation near 0 means the shared seed bought nothing; near 1 means pairing is doing the work
    summary["correlation"] = float(np.corrcoef(va, vb)[0, 1]) if len(shared) > 2 and np.std(va) > 0 and np.std(vb) > 0 else None
    summary["unpaired_ci95"] = (
        _Z95 * math.sqrt(np.var([a[s] for s in shared], ddof=1) / len(shared) + np.var([b[s] for s in shared], ddof=1) / len(shared))
        if len(shared) > 1
        else None
    )
    return summary


def games_needed(sd_per_game: float, half_width: float) -> int:
    """Games required for a 95% interval of the given half width."""
    return math.ceil((_Z95 * sd_per_game / half_width) ** 2)

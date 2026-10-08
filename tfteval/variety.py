"""Strategic variety: which comps an agent finishes with, and whether early signals steer the choice.

Reported next to strength, never weighted into it (README, "Benchmark v1"). Everything here reads
recorded rows (tfteval/record.py) of the hero seat, and only games the hero finished in the top 4:
variety among losing games says little about strategy.

* comp_distribution: the final comps (the target comp in the hero's last recorded row; for a seat
  without one, the comp whose units cover most of the final board) and their Shannon entropy in bits,
  with the maximum for the number of comps in the game and the effective number of comps (2^H).
* responsiveness: mutual information between one early signal at the signal round (default 3-1) and
  the final comp, for three signals: the component held most (item bench and units; ties by name), the
  chosen trait held (none if no chosen unit), the comp most contested on the other boards
  (public.contested_by_comp at the start of the round; none if nothing is contested). Plug-in MI in
  bits, the Miller-Madow bias-corrected value, MI / H(final comp), and a permutation null (the final
  comps shuffled; fixed seed) with its mean and the share of shuffles at or above the observed MI. With
  a few hundred games and a dozen comps the plug-in MI is biased upward: read it against the null.
* noninferiority: variety counts as a plus only if the hero's mean placement is not worse than the
  comparison agent's by more than the margin (the upper end of the 95% interval of hero - comparison,
  paired by seed, at most the margin).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from tfteval import stages
from tfteval.record import unit_chosen, unit_items
from tfteval.rubric import COMPONENTS

NONE = "none"


def entropy(counts) -> float:
    """Shannon entropy in bits of a count vector (or Counter)."""
    values = np.asarray(list(counts.values()) if isinstance(counts, dict) else list(counts), float)
    values = values[values > 0]
    if values.sum() == 0:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log2(p)).sum())


def board_comp(units: list, comps: dict) -> str | None:
    """The comp whose units cover most of the board (ties: the first listed); None for an empty board."""
    names = [u[0] for u in units]
    if not names:
        return None
    best = max(comps.items(), key=lambda kv: (sum(n in set(kv[1]) for n in names), -list(comps).index(kv[0])))
    return best[0] if any(n in set(best[1]) for n in names) else None


def final_comp(rows: list, comps: dict | None = None) -> str | None:
    """The comp the seat ended with: the target comp of its last recorded row, else its final board's."""
    if not rows:
        return None
    for row in reversed(rows):
        if row.get("comp"):
            return row["comp"]
        if row.get("board") is not None:
            break
    if comps is None:
        from tfteval.rubric import default_comps

        comps = default_comps()
    return board_comp(rows[-1].get("board") or [], comps)


def early_signals(rows: list, at="3-1") -> dict | None:
    """The three early signals at planning phase `at` (None when the seat has no row there)."""
    idx = stages.parse_label(at)
    row = next((r for r in rows if r["round"] == idx), None)
    if row is None:
        return None
    units = list(row.get("board") or []) + list(row.get("bench") or [])
    held = Counter(i for i in row.get("items") or [] if i in COMPONENTS)
    for u in units:
        held.update(i for i in unit_items(u) if i in COMPONENTS)
    item = min(held, key=lambda i: (-held[i], i)) if held else NONE
    chosen = sorted(c for c in (unit_chosen(u) for u in units) if c)
    contested = row.get("contested") or {}
    top = min(contested, key=lambda t: (-contested[t], t)) if contested and max(contested.values()) > 0 else NONE
    return {"item": item, "chosen": chosen[0] if chosen else NONE, "contested": top}


def comp_distribution(comps: list, universe: int | None = None) -> dict:
    """Counts and entropy of final comps (None entries are left out)."""
    counts = Counter(c for c in comps if c)
    n = sum(counts.values())
    h = entropy(counts)
    k = universe or len(counts)
    return {"n": n, "counts": dict(counts.most_common()), "distinct": len(counts), "entropy_bits": h,
            "max_bits": math.log2(k) if k > 1 else 0.0, "normalized": h / math.log2(k) if k > 1 else None,
            "effective_comps": 2 ** h if n else None}


def mutual_information(xs: list, ys: list, permutations: int = 1000, seed: int = 0) -> dict:
    """Plug-in MI (bits) between two categorical sequences, Miller-Madow corrected MI, MI / H(Y), and a
    permutation null (ys shuffled `permutations` times with a fixed seed)."""
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys differ in length")
    if n == 0:
        return {"n": 0, "mi_bits": None}

    def mi(a, b):
        joint = Counter(zip(a, b))
        return entropy(Counter(a)) + entropy(Counter(b)) - entropy(joint), joint

    value, joint = mi(xs, ys)
    kx, ky, kxy = len(set(xs)), len(set(ys)), len(joint)
    corrected = value + (kx + ky - kxy - 1) / (2 * n * math.log(2))  # Miller-Madow on each of the 3 entropies
    hy = entropy(Counter(ys))
    rng = np.random.default_rng(seed)
    ys_arr = np.asarray(ys, dtype=object)
    null = [mi(xs, list(rng.permutation(ys_arr)))[0] for _ in range(permutations)] if permutations else []
    return {"n": n, "mi_bits": value, "mi_miller_madow": corrected, "normalized": value / hy if hy > 0 else None,
            "null_mean": float(np.mean(null)) if null else None,
            "p_value": (1 + sum(v >= value - 1e-12 for v in null)) / (1 + len(null)) if null else None,
            "x_categories": kx, "y_categories": ky}


def responsiveness(games: list, permutations: int = 1000) -> dict:
    """MI between each early signal and the final comp over `games`: dicts with `signals` and `comp`."""
    usable = [g for g in games if g.get("signals") and g.get("comp")]
    ys = [g["comp"] for g in usable]
    out = {"n": len(usable)}
    for name in ("item", "chosen", "contested"):
        out[name] = mutual_information([g["signals"][name] for g in usable], ys, permutations)
    return out


def noninferiority(diff: dict | None, margin: float) -> dict:
    """`diff`: hero - comparison mean placement with a 95% interval (tfteval.stats.paired_diff). Variety
    is a plus only when the upper end is at most `margin`."""
    if not diff or diff.get("mean") is None:
        return {"margin": margin, "noninferior": None, "note": "no comparison agent: variety is descriptive only"}
    high = diff["mean"] + diff["ci95"] if diff.get("ci95") is not None else None
    ok = high is not None and high <= margin
    note = (f"hero - comparison placement {diff['mean']:+.2f}, 95% upper end {high:+.2f} "
            f"{'<=' if ok else '>'} margin {margin}: " if high is not None else "no interval: ")
    note += "variety may count as a plus" if ok else "variety is not reported as a plus"
    return {"margin": margin, "diff": diff["mean"], "upper": high, "n": diff.get("n"), "noninferior": ok,
            "note": note}

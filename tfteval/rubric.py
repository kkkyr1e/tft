"""Rubric v1: rule-based checks on recorded trajectories (tfteval/record.py rows).

docs/PLAN.md section 4: the rubric answers "why did it lose", by log rules only. Two kinds of checks:

Hard gates (correctness; reported as counts, never validated against placement):

* `fallbacks`        actions the runner replaced with a random one because the policy raised;
* `illegal`          actions the observation's action mask did not allow;
* `level_short`      plan-executor seats: rounds that ended below the level the executor was told to reach
                     (knobs level_to, capped at the profile's max level) while the gold left at the end of the
                     planning phase would have paid for the missing xp. Usually the round's 15 actions ran out
                     (the rule bot's buys, sells and moves come first); `level_short_idle` counts the ones where
                     the seat still passed at least once, which only an executor bug explains;
* `roll_below_floor` plan-executor seats: rounds with a roll that left less gold than the round's roll
                     floor (knobs roll_floor: the plan's, lowered by `spend` and `survival`).

Items v1 (flags; their association with placement is what scripts/validate_rubric.py estimates):

* `died_with_gold`          eliminated holding more than `died_gold` (20) gold at the end of the planning
                            phase before the fight that knocked the seat out;
* `banked_without_leveling` more than `bank_gold` (50) gold at the end of the planning phase in `bank_rounds`
                            (2) or more consecutive rounds while below the rules profile's max level (set4 9,
                            set18 10; Simulator/game/rules.py via tfteval.stages.rules_profile);
* `items_on_non_carries`    from `items_from` (4-1) on: a round ending with a completed item on a fielded unit
                            that is not a carry. For a plan-executor seat with a target comp the carries are
                            the plan's `carry` and the comp's units; for any other seat (or before a comp) the
                            `fallback_carries` (2) fielded units with the highest cost x stars (then cost; the
                            order the executor places items in). Counted in rounds; the flag is "at least one
                            round".

Thresholds are parameters (THRESHOLDS; benchmarks/v1.json repeats them). `score_seat` scores one seat's rows,
`score_result` every recorded seat of a game result, `rates` aggregates scores by policy, and
`within_state_association` is the estimator scripts/validate_rubric.py reports.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np

from tfteval import stages
from tfteval.record import unit_items

THRESHOLDS = {"died_gold": 20, "bank_gold": 50, "bank_rounds": 2, "items_from": "4-1", "fallback_carries": 2}
ITEMS = ("died_with_gold", "banked_without_leveling", "items_on_non_carries")
GATES = ("fallbacks", "illegal", "level_short", "level_short_idle", "roll_below_floor")

# Simulator/battle/item_stats.py basic_items; tests/test_rubric.py checks it against the simulator.
COMPONENTS = frozenset({"bf_sword", "chain_vest", "giants_belt", "needlessly_large_rod", "negatron_cloak",
                        "recurve_bow", "sparring_gloves", "spatula", "tear_of_the_goddess"})
NOT_ITEMS = frozenset({"champion_duplicator", "magnetic_remover", "reforger", "kayn_rhast", "kayn_shadowassassin"})

_Z95 = NormalDist().inv_cdf(0.975)


def is_completed(item) -> bool:
    """A completed (combined) item: anything but a component or a consumable / Kayn form."""
    return bool(item) and item not in COMPONENTS and item not in NOT_ITEMS


def max_level(rules=None) -> int:
    """The rules profile's max level (set4 9, set18 10), read like tfteval.planner.level_costs."""
    return int(stages.rules_profile(rules).max_level)


def default_comps() -> dict:
    """trait -> units of the simulator's team comps (the executor's COMPS)."""
    from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS, TEAM_COMPS

    return dict(zip(TEAM_COMP_TRAITS, TEAM_COMPS))


def _end(row: dict) -> dict:
    """End-of-planning gold / level / xp (the start of the round when the end was not recorded)."""
    end = row.get("end") or {}
    return {"gold": end.get("gold", row.get("gold", 0)), "level": end.get("level", row.get("level", 1)),
            "xp": end.get("xp", row.get("xp", 0))}


# --------------------------------------------------------------------------- items

def died_with_gold(rows: list, eliminated: bool, gold: int = THRESHOLDS["died_gold"]) -> dict:
    """Eliminated with more than `gold` gold at the end of the last planning phase."""
    held = _end(rows[-1])["gold"] if rows else None
    flag = bool(eliminated and held is not None and held > gold)
    return {"flag": flag, "gold": held if eliminated else None}


def banked_without_leveling(rows: list, rules=None, gold: int = THRESHOLDS["bank_gold"],
                            rounds: int = THRESHOLDS["bank_rounds"]) -> dict:
    """Runs of consecutive rounds ending with more than `gold` gold below the profile's max level; flagged
    when a run lasts `rounds` or more. `rounds` in the output counts the rounds of the flagged runs."""
    top = max_level(rules)
    runs, run, last = [], 0, None
    for row in rows:
        end = _end(row)
        hit = end["gold"] > gold and end["level"] < top
        if hit and last is not None and row["round"] == last + 1 and run:
            run += 1
        elif hit:
            if run:
                runs.append(run)
            run = 1
        else:
            if run:
                runs.append(run)
            run = 0
        last = row["round"]
    if run:
        runs.append(run)
    long = [r for r in runs if r >= rounds]
    return {"flag": bool(long), "rounds": sum(long), "longest": max(runs, default=0), "max_level": top}


def carries(row: dict, comps: dict | None, fallback: int = THRESHOLDS["fallback_carries"]):
    """(set of carry unit names or None, list of fielded units counted as carries by position).

    Plan-executor seat with a target comp: the plan's carry and the comp's units (by name). Otherwise the
    `fallback` fielded units with the highest cost x stars, then cost (by position on the board)."""
    comp = row.get("comp")
    if "plan" in row and comp:
        if comps is None:
            comps = default_comps()
        if comp in comps:
            names = set(comps[comp])
            carry = (row.get("plan") or {}).get("carry")
            if carry:
                names.add(carry)
            return names, None
    board = row.get("board") or []
    order = sorted(range(len(board)), key=lambda i: (-int(board[i][1]) * int(board[i][2]), -int(board[i][2]),
                                                     str(board[i][0]), i))
    return None, set(order[:fallback])


def misplaced_items(row: dict, comps: dict | None, fallback: int = THRESHOLDS["fallback_carries"]) -> int:
    """Completed items on fielded units that are not carries (see carries) at the end of the round's planning."""
    names, top = carries(row, comps, fallback)
    count = 0
    for i, unit in enumerate(row.get("board") or []):
        done = sum(is_completed(item) for item in unit_items(unit))
        if not done:
            continue
        carry = (unit[0] in names) if names is not None else (i in top)
        if not carry:
            count += done
    return count


def items_on_non_carries(rows: list, comps: dict | None = None, start=THRESHOLDS["items_from"],
                         fallback: int = THRESHOLDS["fallback_carries"]) -> dict:
    """Rounds from `start` on that end with a completed item on a non-carry (misplaced_items)."""
    first = stages.parse_label(start)
    checked = flagged = items = 0
    for row in rows:
        if row["round"] < first or "board" not in row:
            continue
        checked += 1
        n = misplaced_items(row, comps, fallback)
        items += n
        flagged += n > 0
    return {"flag": flagged > 0, "rounds": flagged, "checked": checked, "items": items}


# --------------------------------------------------------------------------- hard gates

def xp_gold(level: int, xp: int, target: int, rules=None) -> int:
    """Gold for the xp buys missing from (level, xp) to `target` under the rules profile."""
    profile = stages.rules_profile(rules)
    costs = profile.level_costs
    missing = max(0, sum(costs[lv] for lv in range(level, min(target, len(costs)))) - xp)
    return math.ceil(missing / profile.xp_per_purchase) * profile.xp_purchase_cost


def plan_gates(rows: list, rules=None) -> dict:
    """Plan-vs-execution mismatches of a plan-executor seat (rows with `knobs`)."""
    top = max_level(rules)
    out = {"plan_rounds": 0, "level_short": 0, "level_short_idle": 0, "roll_below_floor": 0}
    for row in rows:
        knobs = row.get("knobs")
        if not knobs:
            continue
        out["plan_rounds"] += 1
        end = _end(row)
        target = min(int(knobs.get("level_to", 0)), top)
        if "end" in row and end["level"] < target and end["gold"] >= xp_gold(end["level"], end["xp"], target, rules):
            out["level_short"] += 1
            out["level_short_idle"] += (row.get("actions") or {}).get("pass", 0) > 0
        if row.get("roll_low") is not None and row["roll_low"] < int(knobs.get("roll_floor", 0)):
            out["roll_below_floor"] += 1
    return out


# --------------------------------------------------------------------------- scoring

def score_seat(rows: list, *, rules=None, place: int | None = None, eliminated_round: int | None = None,
               comps: dict | None = None, thresholds: dict | None = None, from_round=None, **meta) -> dict:
    """Rubric v1 for one seat in one game. `rows` are the seat's recorded rows; `from_round` (a label or
    index) scores only the rows from that planning phase on (a branch's own future). `eliminated_round`
    None means the seat was not knocked out (it won, or the game was cut short)."""
    t = {**THRESHOLDS, **(thresholds or {})}
    if from_round is not None:
        first = stages.parse_label(from_round)
        rows = [r for r in rows if r["round"] >= first]
    eliminated = eliminated_round is not None
    died = died_with_gold(rows, eliminated, t["died_gold"])
    bank = banked_without_leveling(rows, rules, t["bank_gold"], t["bank_rounds"])
    items = items_on_non_carries(rows, comps, t["items_from"], t["fallback_carries"])
    gates = {"fallbacks": sum(r.get("fallbacks", 0) for r in rows), "illegal": sum(r.get("illegal", 0) for r in rows),
             **plan_gates(rows, rules)}
    return {**meta, "place": place, "eliminated_round": eliminated_round, "rounds": len(rows),
            "flags": {"died_with_gold": died["flag"], "banked_without_leveling": bank["flag"],
                      "items_on_non_carries": items["flag"]},
            "detail": {"died_gold": died["gold"], "banked_rounds": bank["rounds"], "banked_longest": bank["longest"],
                       "max_level": bank["max_level"], "items_rounds": items["rounds"],
                       "items_checked": items["checked"], "items_misplaced": items["items"]},
            "gates": gates}


def score_result(result: dict, seats=None, comps: dict | None = None, thresholds: dict | None = None,
                 from_round=None) -> list:
    """score_seat for every recorded seat (or `seats`) of a GameResult (as JSON)."""
    out = []
    records = result.get("records") or {}
    for seat in sorted(seats if seats is not None else records):
        rows = records.get(seat)
        if not rows:
            continue
        out.append(score_seat(rows, rules=result.get("rules"), place=result["placements"].get(seat),
                              eliminated_round=(result.get("eliminated") or {}).get(seat), comps=comps,
                              thresholds=thresholds, from_round=from_round, seat=seat, seed=result.get("seed"),
                              policy=result["lobby"].get(seat)))
    return out


def wilson(k: int, n: int) -> dict:
    """Rate k/n with a 95% Wilson interval."""
    if n == 0:
        return {"k": k, "n": n, "rate": None, "low": None, "high": None}
    p = k / n
    z2 = _Z95 ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return {"k": k, "n": n, "rate": p, "low": low, "high": high}


def rates(scores: list, key: str = "policy") -> dict:
    """Per policy (or another score field): each item's rate with a Wilson interval, and the hard gates'
    totals and per-seat means. Seats of one game share a lobby, so with several seats of a policy per game
    the intervals are a little too narrow; the benchmark records one hero seat per game."""
    groups = defaultdict(list)
    for s in scores:
        groups[s.get(key)].append(s)
    out = {}
    for name, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(group)
        items = {item: wilson(sum(bool(s["flags"][item]) for s in group), n) for item in ITEMS}
        details = {k: float(np.mean([s["detail"][k] or 0 for s in group])) for k in
                   ("banked_rounds", "items_rounds", "items_misplaced")}
        gates = {g: {"total": int(sum(s["gates"].get(g, 0) for s in group)),
                     "per_seat": float(np.mean([s["gates"].get(g, 0) for s in group]))} for g in GATES}
        gates["seats_with_any"] = sum(any(s["gates"].get(g, 0) for g in GATES) for s in group)
        out[name] = {"n": n, "items": items, "detail_means": details, "gates": gates}
    return out


# --------------------------------------------------------------------------- validation (within state)

NOTE = ("Association within saved state, not a causal effect: in branches re-seeded from the same state, "
        "how much worse (positive) or better the hero placed when the item fired than when it did not. "
        "Whatever made the item fire (bad luck, a weak board) can also cost placement.")


def within_state_association(branches: list) -> dict:
    """Within-state association of a 0/1 flag with placement.

    `branches`: dicts with `state` (the saved state's key), `cluster` (the source game: states from one game
    are not independent), `place` and `flag`. Placement and flag are demeaned within each state (state fixed
    effects); the slope of demeaned placement on demeaned flag (`fe_diff`) is a weighted mean of the
    within-state differences "mean placement flagged - mean placement not flagged", weighted by n p (1 - p)
    of each state; only states with both flagged and unflagged branches contribute. Its 95% interval is
    clustered by source game (sandwich variance with a G / (G - 1) correction, t with G - 1 df).
    `mean_state_diff` is the plain mean of those per-state differences, averaged within each source game
    first, with a t interval over source games."""
    from tfteval.stats import cluster_mean_ci, t_quantile

    states = defaultdict(list)
    for b in branches:
        if b.get("place") is None:
            continue
        states[b["state"]].append(b)
    rows, per_state = [], []
    for key, group in states.items():
        place = np.array([b["place"] for b in group], float)
        flag = np.array([float(bool(b["flag"])) for b in group])
        if 0 < flag.sum() < len(flag):
            per_state.append((group[0]["cluster"], float(place[flag == 1].mean() - place[flag == 0].mean())))
        for b, p, f in zip(group, place - place.mean(), flag - flag.mean()):
            rows.append((b["cluster"], p, f))
    flagged = sum(bool(b["flag"]) for g in states.values() for b in g)
    out = {"states": len(states), "states_mixed": len(per_state), "branches": sum(len(g) for g in states.values()),
           "flagged": flagged, "note": NOTE}
    sxx = sum(f * f for _, _, f in rows)
    if not per_state or sxx == 0:
        out.update(fe_diff=None, ci95=None, low=None, high=None, clusters=0, mean_state_diff={"n": 0})
        return out
    beta = sum(f * p for _, p, f in rows) / sxx
    score = defaultdict(float)
    for cluster, p, f in rows:
        score[cluster] += f * (p - beta * f)
    used = {c for c, _ in per_state}
    g = len(used)
    if g >= 2:
        var = (g / (g - 1)) * sum(v * v for v in score.values()) / sxx ** 2
        half = t_quantile(0.975, g - 1) * math.sqrt(var)
    else:
        half = None
    by_cluster = defaultdict(list)
    for cluster, d in per_state:
        by_cluster[cluster].append(d)
    out.update(fe_diff=beta, ci95=half, low=None if half is None else beta - half,
               high=None if half is None else beta + half, clusters=g,
               mean_state_diff=cluster_mean_ci([float(np.mean(v)) for v in by_cluster.values()]))
    return out

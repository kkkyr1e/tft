"""Contrastive pairs for the state-reading score (docs/BANK_V2.md).

A pair is one saved state of a source game (lineup, seed, planning phase `point`) on two sides, each a
counterfactual edit of one feature of the hero (e.g. HP set to 35 on one side and 85 on the other), and
two candidates of the bank's menu (bank2.CANDIDATES, e.g. roll_all and save) played on both sides with
the same branch seeds k = 0..K-1. Branch k re-seeds the game from (game seed, round, k) whatever the
edit (branching.branch), so the four cells (side x candidate) of one k share their randomness and the
interaction

    I_k = (A - B on side 0) - (A - B on side 1)          (places; negative: A gains on side 0)

is paired by k. Racing is off: every cell gets the same K branches.

The edit is applied after the state is rebuilt from the recipe and checked against the unedited
fingerprint. An HP edit also shifts the stance planner's HP history (StancePlanner.hp_seen) by the
same amount, so the HP lost over the last rounds (its "bleeding" test) is the base state's on both
sides; without that, a large downward edit reads as a sudden loss of HP. Upward gold edits are nearly
inert (interest is capped at 50 and the round's action cap limits rolls), so gold edits should go down.
"""

from __future__ import annotations

import math
import time

import numpy as np

from tfteval import bank, bank2

SCHEMA = 1
AXES = ("hp", "gold")
HP_RANGE = (1, 100)   # the simulator one-hot encodes health 1..100 (Player observation)


def hero_planner(game, seat: str):
    """The hero's planner object (through ActionLog and CommitPlanner wrappers), or None."""
    policy = game.seat_policies[seat]
    policy = getattr(policy, "policy", policy)  # ActionLog
    planner = getattr(policy, "planner", None)
    return getattr(planner, "base", planner)  # CommitPlanner


def apply_edit(game, seat: str, edit: dict) -> dict:
    """Set the hero's features in `edit` (axis -> value) in place; returns {axis: [before, after]}."""
    unknown = set(edit) - set(AXES)
    if unknown:
        raise ValueError(f"cannot edit {sorted(unknown)}; axes are {AXES}")
    player = game.player(seat)
    changed = {}
    if "hp" in edit:
        hp, before = int(edit["hp"]), int(player.health)
        if not HP_RANGE[0] <= hp <= HP_RANGE[1]:
            raise ValueError(f"hp {hp} outside {HP_RANGE}")
        player.health = hp
        history = getattr(hero_planner(game, seat), "hp_seen", None)
        if isinstance(history, dict):
            for rnd in history:
                history[rnd] += hp - before
        changed["hp"] = [before, hp]
    if "gold" in edit:
        gold, before = int(edit["gold"]), int(player.gold)
        if gold < 0:
            raise ValueError(f"gold {gold} < 0")
        player.gold = gold
        changed["gold"] = [before, gold]
    return changed


def candidate(name: str, rounds: int = bank2.COMMIT_ROUNDS) -> dict:
    return {"name": name, "desc": bank2.CANDIDATES[name]["desc"], "rounds": int(rounds),
            "spec": {k: v for k, v in bank2.CANDIDATES[name].items() if k != "desc"}}


def pair_id(lineup: str, seed: int, point: str, axis: str, values) -> str:
    return f"{lineup}#{seed}@{point}:{axis}={values[0]}|{values[1]}"


def base_recipe(lineup: str, seed: int, point: str, settings: dict | None = None,
                harness: str | None = None) -> tuple[dict, object]:
    """The recipe of the source game at `point` and its game played there (hero alive, a plan seat)."""
    from tfteval import simfixes
    from tfteval.planner import PlanPolicy

    recipe = bank.make_recipe(lineup, seed, point, settings)
    recipe.update(harness_commit=harness)
    game = bank.play_recipe(recipe)
    if (game.sim, game.sim_options, game.rules) != (recipe["sim"], recipe["sim_options"], recipe["rules"]):
        raise RuntimeError(f"game played {game.sim} {game.sim_options} {game.rules}, the recipe says "
                           f"{recipe['sim']} {recipe['sim_options']} {recipe['rules']}")
    if not bank.hero_alive(game, recipe):
        raise RuntimeError(f"hero {recipe['hero_seat']} is out before {point} in {lineup}#{seed}")
    if not isinstance(game.seat_policies[recipe["hero_seat"]], PlanPolicy):
        raise ValueError(f"the hero of {lineup!r} is not a plan seat; the continuation needs its executor state")
    recipe["sim_commit"], recipe["sim_fixes"] = simfixes.sim_commit(), list(game.sim_fixes)
    recipe["carousel_pickers"] = list(getattr(game, "carousel_pickers", []))
    return recipe, game


def play_side(spec: dict, side: int) -> dict:
    """One side of a pair: rebuild the state, apply the side's edit, play every candidate on k = 0..K-1.
    `spec`: lineup, seed, point, axis, values (two), cands (names), k, continuation, settings, harness."""
    from tfteval.branching import snapshot

    started = time.time()
    recipe, game = base_recipe(spec["lineup"], spec["seed"], spec["point"], spec.get("settings"), spec.get("harness"))
    seat = recipe["hero_seat"]
    fp = bank.fingerprint(game, seat)
    base = {"fingerprint_hash": bank.fingerprint_hash(fp), "features": bank2.trigger_features(bank.hero_view(game, seat)[0])}
    edit = {spec["axis"]: spec["values"][side]}
    changed = apply_edit(game, seat, edit)
    state, _, comp_now = bank.hero_view(game, seat)
    edited = bank.fingerprint_hash(bank.fingerprint(game, seat))
    cands = [candidate(c) for c in spec["cands"]]
    for c in cands:
        c["now"] = bank.resolve(c["spec"], state)
    snap = snapshot(game)
    del game
    branches = []
    for k in range(int(spec["k"])):
        for c in cands:
            b = bank.play_branch(snap, seat, c, k, spec["continuation"])
            branches.append({"side": side, **b})
    return {"side": side, "edit": edit, "changed": changed, "recipe": recipe, "base": base,
            "fingerprint_hash": edited, "public_state": state, "comp_now": comp_now,
            "features": bank2.trigger_features(state), "candidates": cands, "branches": branches,
            "seconds": round(time.time() - started, 2)}


def assemble(spec: dict, sides: list[dict]) -> dict:
    """The pair record from its two side results (both must start from the same unedited state)."""
    s0, s1 = sorted(sides, key=lambda s: s["side"])
    if s0["base"]["fingerprint_hash"] != s1["base"]["fingerprint_hash"]:
        raise RuntimeError(f"the two sides of {pair_id(spec['lineup'], spec['seed'], spec['point'], spec['axis'], spec['values'])} "
                           f"rebuilt different states: {s0['base']['fingerprint_hash']} != {s1['base']['fingerprint_hash']}")
    return {"schema": SCHEMA, "id": pair_id(spec["lineup"], spec["seed"], spec["point"], spec["axis"], spec["values"]),
            "game": f"{spec['lineup']}#{spec['seed']}", "lineup": spec["lineup"], "seed": int(spec["seed"]),
            "point": spec["point"], "axis": spec["axis"], "values": list(spec["values"]), "cands": list(spec["cands"]),
            "k": int(spec["k"]), "continuation": spec["continuation"], "recipe": s0["recipe"], "base": s0["base"],
            "sides": [{k: v for k, v in s.items() if k not in ("recipe", "base", "branches")}
                      for s in (s0, s1)],
            "branches": s0["branches"] + s1["branches"]}


# --------------------------------------------------------------------------- analysis

def cells(pair: dict) -> dict:
    """{(side, cand): {k: place}}."""
    out: dict = {}
    for b in pair["branches"]:
        if b.get("place") is not None:
            out.setdefault((b["side"], b["cand"]), {})[b["k"]] = b["place"]
    return out


def _summary(values) -> dict:
    v = np.asarray(values, float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else math.nan
    return {"mean": float(v.mean()) if n else math.nan, "sd": sd, "se": sd / math.sqrt(n) if n > 1 else math.nan, "n": n}


def contrasts(pair: dict, a: str | None = None, b: str | None = None) -> dict:
    """Side contrasts A - B (places, paired by k) and their interaction, on the ks all four cells have."""
    a, b = a or pair["cands"][0], b or pair["cands"][1]
    c = cells(pair)
    ks = sorted(set.intersection(*(set(c.get((s, x), {})) for s in (0, 1) for x in (a, b))))
    d = {s: [c[(s, a)][k] - c[(s, b)][k] for k in ks] for s in (0, 1)}
    return {"a": a, "b": b, "ks": len(ks),
            "side": {s: _summary(d[s]) for s in (0, 1)},
            "mean_place": {f"{s}:{x}": float(np.mean([c[(s, x)][k] for k in ks])) for s in (0, 1) for x in (a, b)},
            "interaction": _summary([d[0][i] - d[1][i] for i in range(len(ks))]),
            "side_corr": float(np.corrcoef(d[0], d[1])[0, 1]) if len(ks) > 2 and np.std(d[0]) and np.std(d[1]) else None}


def pooled(estimates: list[dict]) -> dict:
    """Pool the pairs' interactions: the plain mean over pairs with its between-pair SE (random effects
    in the simplest form), and the inverse-variance (fixed-effect) mean with DerSimonian-Laird tau^2."""
    m = np.array([e["mean"] for e in estimates], float)
    se = np.array([e["se"] for e in estimates], float)
    n = len(m)
    out = {"pairs": n, "mean": float(m.mean()) if n else math.nan,
           "se_between": float(m.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan}
    if n and np.all(se > 0):
        w = 1 / se ** 2
        fe = float((w * m).sum() / w.sum())
        q = float((w * (m - fe) ** 2).sum())
        tau2 = max(0.0, (q - (n - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if n > 1 else 0.0
        wr = 1 / (se ** 2 + tau2)
        out.update(fixed=fe, fixed_se=float(1 / math.sqrt(w.sum())), q=q, tau2=tau2,
                   random=float((wr * m).sum() / wr.sum()), random_se=float(1 / math.sqrt(wr.sum())))
    return out

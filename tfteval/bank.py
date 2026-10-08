"""Decision bank: saved decision points, candidate commitments, labels from re-seeded branches, scoring.

An item is one hero seat at one decision point of one source game, three candidate decisions and their
labels. The item does not hold the 4.7 MB snapshot; it holds the recipe that rebuilds the state (the
lineup of seat policies, the game seed, the hero seat, the round, the economy rules, the simulator
profile (tfteval.runner.SIM_PROFILES, "realistic" by default, with any overrides) and the exact
TFTConfig options it set, whether plan seats pick their own carousel unit, the simulator and harness
commits). `rebuild` plays the recipe with `branching.play_to`, which
is deterministic under PYTHONHASHSEED=0, and checks a fingerprint of the state (hero HP, gold, level,
xp, board, bench, items, shop, everyone's HP, the step count) against the one stored at build time.

Candidates are commitments, not single actions: plan-field overrides held for a few planning phases
(`POINTS[label]["rounds"]`) on top of the hero's continuation planner, which then plays on alone
(`CommitPlanner`). The overrides replace the economy fields of the plan (ECONOMY_FIELDS: level_to,
roll_floor, level_by, spend, xp_buys, survival) and keep the rest (comp, carry, field_comp, hold,
fodder). The continuation is `stance` by default; every other seat keeps its own policy. A candidate is
a small declarative spec (CANDIDATES) resolved against the round-start state of each committed round:

    level     the level to buy xp toward: "hold" (none), "curve" (stance.LEVEL_CURVE, the lobby's
              curve), "+1" (one above the level at the decision point), "8+" (8, or one above the
              decision point's level when that is already 8 or more; any "N+" works the same way)
              or a number
    keep      xp is bought only with the gold above this (xp_buys, partial levels allowed)
    roll_floor            reroll while gold - 2 >= this (999: no rerolls)
    roll_floor_at_target  the roll floor once the target level is reached
    fodder    true: the plan field `fodder` is set (field the weakest units: lose on purpose)
    then      "base": once the target level is reached at the start of a committed round, that round
              is the base planner's own (no overrides); the commitment is "get there, then play on"

(`fodder` and `then` are only used by the bank v2 candidates, tfteval/bank2.py.)

Menus per decision point are in POINTS (one table); the five points are 3-2, 3-5, 4-1, 4-5 and 5-1.

Labels. Every candidate is played from the same snapshot with re-seeds k = 0 .. Kd+Kl-1; the same k
across candidates is the same random future (common random numbers) until the hero's actions differ.
The DISCOVERY set (k < Kd) picks the best candidate (lowest mean placement; ties: menu order). The
LABEL set (Kd <= k < Kd+Kl) estimates each candidate's expected placement and its regret, the paired
mean of place(candidate, k) - place(best, k). Picking and estimating on separate branches avoids the
winner's curse; the regret is left unclipped, so a candidate the discovery set wrongly ranked below the
best can show a small negative regret, which keeps the mean regret of an agent unbiased. The item is
`clear` when every other (non-duplicate) candidate is worse than the best by more than the 95% interval
of the paired difference, else `ambiguous` (both are kept and reported apart). The interval uses the
paired differences pooled over the item's candidates, with a floor of SD_FLOOR places on the standard
deviation (with two or three label branches a sample SD of 0 happens by chance). Per-branch placements
and per-round action hashes are stored, so labels can be recomputed (`label_item`).

Duplicates. Two candidates whose hero actions are identical round by round in every branch played the
same game (same re-seed, same actions, so the same placement). They are grouped (`duplicates`); an item
where all candidates are one group has no decision in it and is dropped.

Scoring. An agent sees the item's public state (tfteval.planner.describe, which only reads
tfteval.public for the other players) and the candidate names and descriptions, and picks one name.
Rule planners from the repo are mapped to the nearest candidate (`nearest_candidate`); any callable can
be scored (`CallableAgent`), e.g. a model later. The score is the mean regret of the choices with a 95%
interval clustered by source game, and the share of items where the choice is within the label interval
of the best, for clear and ambiguous items and per decision point, on the dev and held-out source games
(`split`: by a hash of the seed, about 70/30).
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import random as _random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from tfteval import stages
from tfteval.planner import SPEND_CAP, compile_knobs, xp_to_level
from tfteval.stats import t_quantile

SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- candidates

ECONOMY_FIELDS = ("level_to", "roll_floor", "level_by", "spend", "xp_buys", "survival")

CANDIDATES = {
    "save": {"level": "curve", "keep": 50, "roll_floor": 999,
             "desc": "Save: no rerolls; buy xp only toward the lobby curve level (6 at 3-2, 7 at 3-5, 8 from 4-2) "
                     "and only with the gold above 50, so the interest stays at the maximum."},
    "level": {"level": "+1", "keep": 30, "roll_floor": 999,
              "desc": "Level: buy xp toward one level above the current one with the gold above 30 (this round "
                      "if that is enough, else the next), no rerolls."},
    "roll": {"level": "hold", "roll_floor": 15,
             "desc": "Roll: reroll down to 15 gold every round for upgrades (stabilize), no xp."},
    "fast8": {"level": "8+", "keep": 10, "roll_floor": 999, "roll_floor_at_target": 20,
              "desc": "Fast 8: buy xp toward level 8 (toward 9 when already 8) with the gold above 10, no rerolls "
                      "until there, then reroll down to 20."},
    "cap_out": {"level": 9, "keep": 10, "roll_floor": 999,
                "desc": "Cap out: buy xp toward level 9 with the gold above 10, no rerolls."},
}

# decision point -> commitment length (planning phases) and the menu, in tie-break order
POINTS = {
    "3-2": {"rounds": 2, "menu": ("save", "level", "roll")},
    "3-5": {"rounds": 2, "menu": ("save", "level", "roll")},
    "4-1": {"rounds": 3, "menu": ("save", "fast8", "roll")},
    "4-5": {"rounds": 2, "menu": ("save", "fast8", "roll")},
    "5-1": {"rounds": 2, "menu": ("save", "cap_out", "roll")},
}

SD_FLOOR = 1.0  # places; paired branch differences within a state measured 1.4-3.0 (README, branching)
DEV_SHARE = 70  # percent of source games (by seed) in the dev split


def menu(point: str) -> list[dict]:
    """The candidates of a decision point: name, description, spec and commitment length."""
    spec = POINTS[point]
    return [{"name": name, "desc": CANDIDATES[name]["desc"], "rounds": spec["rounds"],
             "spec": {k: v for k, v in CANDIDATES[name].items() if k != "desc"}} for name in spec["menu"]]


def max_level(state: dict) -> int:
    """The highest level under the state's rules profile (describe() names it; None: TFT_RULES)."""
    return int(stages.rules_profile(state.get("rules")).max_level)


def target_level(spec: dict, state: dict, anchor_level: int) -> int:
    level = int(state["level"])
    want = spec.get("level", "hold")
    if want == "hold":
        target = level
    elif want == "curve":
        from tfteval.stance import level_curve

        target = max(level, level_curve(int(state["round"])))
    elif want == "+1":
        target = anchor_level + 1
    elif isinstance(want, str) and want.endswith("+") and want[:-1].isdigit():  # "8+": 8, or one above
        floor = int(want[:-1])
        target = floor if anchor_level < floor else anchor_level + 1
    else:
        target = int(want)
    return min(target, max_level(state))


def resolve(spec: dict, state: dict, anchor_level: int | None = None) -> dict | None:
    """The economy plan fields of a candidate spec for this round's state (describe() output);
    `anchor_level` is the hero's level at the decision point (default: the current level). None (only
    for a spec with `then: "base"`) when the target level is already reached: the round is the base
    planner's own."""
    level, xp, gold = int(state["level"]), int(state["xp"]), int(state["gold"])
    target = target_level(spec, state, level if anchor_level is None else anchor_level)
    fields = {"level_to": level, "roll_floor": int(spec.get("roll_floor", 999))}
    need = xp_to_level(level, xp, target, state.get("rules"))
    if need <= 0 and spec.get("then") == "base":
        return None
    if need > 0:
        buys = min(math.ceil(need / 4), max(0, (gold - int(spec.get("keep", 0))) // 4))
        if buys:
            fields["xp_buys"] = buys
    elif "roll_floor_at_target" in spec:
        fields["roll_floor"] = int(spec["roll_floor_at_target"])
    if spec.get("fodder"):
        fields["fodder"] = True
    return fields


def override(plan: dict, fields: dict) -> dict:
    """The plan with its economy fields replaced by `fields` (the other fields kept)."""
    out = {k: v for k, v in plan.items() if k not in ECONOMY_FIELDS}
    out.update(fields)
    return out


class CommitPlanner:
    """A base planner whose economy is overridden by a candidate for `rounds` planning phases, from the
    first round it is asked about; then it is the base planner alone. The base is asked every round (it
    keeps its own state up to date, e.g. stance's hp history and its fast 8 bookkeeping) and its
    non-economy fields are kept. Picklable, so a game holding it can be snapshotted."""

    def __init__(self, base, candidate: str | dict, rounds: int, name: str | None = None):
        if isinstance(candidate, str):
            candidate = {"name": candidate, **{k: v for k, v in CANDIDATES[candidate].items() if k != "desc"}}
        self.base, self.candidate, self.rounds = base, dict(candidate), int(rounds)
        self.name = name or f"{getattr(base, 'name', 'base')}+{candidate.get('name', 'commit')}"
        self.start: int | None = None
        self.anchor: dict | None = None  # hero state at the decision point
        self.applied: list[int] = []  # rounds the overrides were applied
        self.changed: list[int] = []  # ... and changed the compiled knobs

    @property
    def context(self):
        return self.base.context  # AttributeError when the base has none: PlanPolicy checks hasattr

    def committed(self, idx: int) -> bool:
        return self.start is not None and self.start <= idx < self.start + self.rounds

    def plan(self, state: dict, comps: dict, comp_now: str | None) -> dict:
        plan = self.base.plan(state, comps, comp_now)
        idx = int(state["round"])
        if self.start is None:
            self.start = idx
            self.anchor = {k: state[k] for k in ("round", "level", "xp", "gold", "hp")}
        if not self.committed(idx):
            return plan
        fields = resolve(self.candidate, state, self.anchor["level"])
        if fields is None:  # `then: "base"` and the target is reached: this round is the base's own
            return plan
        new = override(plan, fields)
        new["commit"] = self.candidate.get("name")
        self.applied.append(idx)
        if compile_knobs(new, state) != compile_knobs(plan, state):
            self.changed.append(idx)
        return new


def plan_kind(kind: str):
    """A fresh planner object of a plan-seat kind (a VARIANTS economy or a STANCE_KINDS stance)."""
    from tfteval.planner import VARIANTS, make_plan_policy
    from tfteval.stance import STANCE_KINDS, make_stance_policy

    if kind in STANCE_KINDS:
        return make_stance_policy(kind).planner
    if kind in VARIANTS or kind == "llm":
        return make_plan_policy(kind).planner
    raise ValueError(f"{kind!r} is not a plan-seat kind")


def _commit_apply(candidate: dict, rounds: int, continuation: str, old):
    from tfteval.branching import switch_planner

    planner = getattr(old, "planner", None)
    base = planner if getattr(planner, "name", None) == continuation else plan_kind(continuation)
    return switch_planner(CommitPlanner(base, candidate, rounds))(old)


def commit_switch(candidate: str | dict, rounds: int, continuation: str = "stance"):
    """A `branch(switch=...)` entry: the hero keeps its executor state (switch_planner) and plans with
    CommitPlanner(continuation, candidate, rounds). The continuation is the hero's own planner object
    when it already is that kind (it keeps its history), else a fresh one."""
    if isinstance(candidate, str):
        candidate = {"name": candidate, **{k: v for k, v in CANDIDATES[candidate].items() if k != "desc"}}
    return functools.partial(_commit_apply, dict(candidate), int(rounds), continuation)


# --------------------------------------------------------------------------- recipes and state

def parse_seats(text: str) -> list[str]:
    names = []
    for part in text.split(","):
        name, _, count = part.partition(":")
        names.extend([name.strip()] * int(count or 1))
    return names


def parse_lineup(text: str) -> tuple[str, list[str]]:
    """`HERO@OPPONENTS`, e.g. `stance@rule:7` or `mimic@rule:4,stance:3`: the hero's policy before the
    decision point (a plan-seat kind) and the 7 other seats."""
    hero, _, opponents = text.partition("@")
    others = parse_seats(opponents or "rule:7")
    if len(others) != 7:
        raise ValueError(f"lineup {text!r} needs 7 opponents, got {len(others)}")
    return hero.strip(), others


def hero_seat(seed: int) -> str:
    return f"player_{(8 - seed % 8) % 8}"


def lobby_specs(seed: int, hero: str, opponents: list[str]) -> dict:
    """seat -> make_policy spec, the hero rotated by the seed as in branch_compare.py / run_lobby.py."""
    names = [f"hero={hero}"] + list(opponents)
    shift = seed % 8
    order = names[shift:] + names[:shift]
    return {f"player_{i}": name for i, name in enumerate(order)}


def harness_commit() -> str | None:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "tfteval", "scripts"],
                               capture_output=True, text=True, timeout=10).stdout.strip()  # untracked code too
        return head.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


SETTING_KEYS = ("rules", "sim", "sim_options", "pickers")


def sim_settings(sim: str | None = None, rules: str | None = None, pickers: bool | None = None) -> dict:
    """The simulator settings games are played with, as tfteval.runner.Game resolves them: the economy
    rules (`rules`, else TFT_RULES, default set4), the simulator profile with any overrides (`sim`, e.g.
    "realistic,rng_streams=shared", else TFT_SIM, default realistic) and the exact TFTConfig options it
    sets, and whether plan seats pick their own carousel unit (else TFT_PICKERS, default on)."""
    from tfteval.runner import sim_options

    spec, options = sim_options(sim)
    if pickers is None:
        pickers = os.environ.get("TFT_PICKERS", "1") != "0"
    return {"rules": rules or os.environ.get("TFT_RULES", "set4"), "sim": spec, "sim_options": options,
            "pickers": bool(pickers)}


def recipe_settings(recipe: dict) -> dict:
    """A recipe's simulator settings (SETTING_KEYS). Recipes made before the simulator profiles (no
    `sim` key) ran every fork option off and no carousel pickers: sim "default", pickers off."""
    if "sim" not in recipe:
        return {"rules": recipe.get("rules") or "set4", "sim": "default", "sim_options": {}, "pickers": False}
    return {"rules": recipe.get("rules") or "set4", "sim": recipe["sim"], "sim_options": dict(recipe["sim_options"]),
            "pickers": bool(recipe["pickers"])}


def make_recipe(lineup: str, seed: int, point: str, settings: dict | None = None) -> dict:
    hero, opponents = parse_lineup(lineup)
    settings = settings or sim_settings()
    return {"lineup": lineup, "seed": int(seed), "point": point, "round": stages.parse_label(point),
            "hero_seat": hero_seat(seed), "lobby": lobby_specs(seed, hero, opponents),
            **{key: settings[key] for key in SETTING_KEYS}, "hashseed": os.environ.get("PYTHONHASHSEED")}


def game_kwargs(recipe: dict) -> dict:
    """Keyword arguments of runner.Game for a recipe. The profile's options are resolved again and must
    equal the recorded ones (a profile redefined since the item was built would rebuild another game)."""
    from tfteval.runner import sim_options

    settings = recipe_settings(recipe)
    _, options = sim_options(settings["sim"])
    if options != settings["sim_options"]:
        raise RuntimeError(f"simulator profile {settings['sim']!r} now sets {options}, the recipe recorded "
                           f"{settings['sim_options']}")
    return {"rules": settings["rules"], "sim": settings["sim"], "pickers": settings["pickers"]}


def new_game(recipe: dict):
    from tfteval import make_policy
    from tfteval.runner import Game

    if recipe.get("hashseed") not in (None, os.environ.get("PYTHONHASHSEED")):
        raise RuntimeError(f"recipe built with PYTHONHASHSEED={recipe['hashseed']!r}, this process has "
                           f"{os.environ.get('PYTHONHASHSEED')!r}")
    seats = {seat: make_policy(spec) for seat, spec in recipe["lobby"].items()}
    return Game(seats, recipe["seed"], **game_kwargs(recipe))


def play_recipe(recipe: dict):
    """The game of a recipe played to the planning phase of its round (branching.play_to)."""
    from tfteval import make_policy
    from tfteval.branching import play_to

    if recipe.get("hashseed") not in (None, os.environ.get("PYTHONHASHSEED")):
        raise RuntimeError(f"recipe built with PYTHONHASHSEED={recipe['hashseed']!r}, this process has "
                           f"{os.environ.get('PYTHONHASHSEED')!r}")
    seats = {seat: make_policy(spec) for seat, spec in recipe["lobby"].items()}
    return play_to(seats, recipe["seed"], recipe["round"], **game_kwargs(recipe))


def hero_alive(game, recipe: dict) -> bool:
    seat = recipe["hero_seat"]
    return not game.done and game.round == recipe["round"] and game.player(seat) is not None \
        and seat not in game.placements


def fingerprint(game, seat: str) -> dict:
    """What a rebuilt state must match: the hero's full state and everyone's HP at this step."""
    from tfteval import public

    p = game.player(seat)
    board = sorted([x, y, public.unit_tag(u)] for x, col in enumerate(p.board) for y, u in enumerate(col)
                   if public.is_unit(u))
    players = public.alive_players(game.env)
    return {"round": int(game.round), "steps": int(game.steps),
            "hero": {"hp": float(p.health), "gold": int(p.gold), "level": int(p.level), "xp": int(p.exp),
                     "streak": public.streak(p), "board": board,
                     "bench": [public.unit_tag(u) if public.is_unit(u) else None for u in p.bench],
                     "items": [i for i in p.item_bench if i], "shop": [str(s) for s in p.shop if s]},
            "lobby_hp": {s: float(q.health) for s, q in players.items()}}


def fingerprint_hash(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


def hero_view(game, seat: str) -> tuple[dict, dict, str | None]:
    """The hero's round-start state as its planner gets it (PlanPolicy.act), the comps and the comp."""
    from tfteval.planner import describe

    policy = game.seat_policies[seat]
    policy = getattr(policy, "policy", policy)  # ActionLog
    player = game.player(seat)
    info = game.infos.get(seat, {}) if isinstance(getattr(game, "infos", None), dict) else {}
    executor = getattr(policy, "executor", None)
    comps = getattr(policy, "comps", None)
    if comps is None:
        from tfteval.executor import COMPS

        comps = dict(COMPS)
    comp_now = None
    if executor is not None and executor.comp_number >= 0:
        comp_now = policy.traits[executor.comp_number]
    shop = info.get("shop", player.shop)
    state = describe(player, shop, info.get("game_round", game.round), game.env, seat=seat, comps=comps, comp=comp_now,
                     candidates=info.get("opponent_candidates"))
    return state, comps, comp_now


def rebuild(recipe: dict, expect: str | None = None):
    """The recipe's game at its decision point; with `expect` (a fingerprint hash) the state is checked."""
    game = play_recipe(recipe)
    if not hero_alive(game, recipe):
        raise RuntimeError(f"hero {recipe['hero_seat']} is not alive at round {recipe['round']} on rebuild")
    got = fingerprint_hash(fingerprint(game, recipe["hero_seat"]))
    if expect is not None and got != expect:
        raise RuntimeError(f"rebuilt state differs from the bank's: fingerprint {got} != {expect}")
    return game


# --------------------------------------------------------------------------- branches

class ActionLog:
    """Wraps the hero's policy and hashes its actions round by round (as scripts/branch_compare.py)."""

    def __init__(self, policy):
        self.policy, self.name, self.rounds = policy, policy.name, {}

    def reset(self, seed):
        self.policy.reset(seed)

    def act(self, observation, info, agent, env):
        action = self.policy.act(observation, info, agent, env)
        rnd = int(info.get("game_round", 1))
        self.rounds.setdefault(rnd, hashlib.sha256()).update(repr(list(map(int, action))).encode())
        return action

    def digests(self) -> dict:
        return {str(r): h.hexdigest()[:12] for r, h in sorted(self.rounds.items())}


def play_branch(snap, seat: str, candidate: dict, k: int, continuation: str) -> dict:
    """One re-seeded branch with the hero committed to `candidate`, played until the hero is out."""
    from tfteval.branching import branch

    started = time.time()
    g = branch(snap, reseed=k, switch={seat: commit_switch(candidate["spec"] | {"name": candidate["name"]},
                                                           candidate["rounds"], continuation)})
    log = g.seat_policies[seat] = ActionLog(g.seat_policies[seat])
    while not g.done and seat not in g.placements:  # the hero's place is final once it is out
        g.run(until_round=g.round + 1)
    planner = log.policy.planner
    return {"cand": candidate["name"], "k": k, "place": g.placements.get(seat), "actions": log.digests(),
            "applied": list(planner.applied), "changed": list(planner.changed), "steps": g.steps,
            "fallbacks": g.fallbacks.get(seat, 0), "seconds": round(time.time() - started, 2)}


def item_id(lineup: str, seed: int, point: str) -> str:
    return f"{lineup}#{seed}@{point}"


def split(seed: int) -> str:
    """dev or heldout for a source game, by a hash of its seed (whole games, ~70/30)."""
    h = int.from_bytes(hashlib.sha256(f"bank-split:{int(seed)}".encode()).digest()[:8], "big")
    return "dev" if h % 100 < DEV_SHARE else "heldout"


def build_item(lineup: str, seed: int, point: str, kd: int, kl: int, continuation: str = "stance",
               settings: dict | None = None, harness: str | None = None, early_drop: bool = False) -> dict:
    """Play a source game to the decision point, branch every candidate Kd + Kl times and label it."""
    from tfteval import simfixes
    from tfteval.branching import snapshot

    if point not in POINTS:
        raise ValueError(f"no decision point {point!r}; choose from {list(POINTS)}")
    started = time.time()
    recipe = make_recipe(lineup, seed, point, settings)
    recipe.update(sim_commit=None, harness_commit=harness)
    cands = menu(point)
    item = {"schema": SCHEMA, "id": item_id(lineup, seed, point), "game": f"{lineup}#{seed}", "lineup": lineup,
            "seed": int(seed), "point": point, "split": split(seed), "recipe": recipe, "continuation": continuation,
            "kd": kd, "kl": kl, "candidates": cands, "dropped": None}
    game = play_recipe(recipe)
    recipe["sim_commit"], recipe["sim_fixes"] = simfixes.sim_commit(), list(game.sim_fixes)
    recipe["carousel_pickers"] = list(getattr(game, "carousel_pickers", []))
    if (game.sim, game.sim_options, game.rules) != (recipe["sim"], recipe["sim_options"], recipe["rules"]):
        raise RuntimeError(f"game played {game.sim} {game.sim_options} {game.rules}, the recipe says "
                           f"{recipe['sim']} {recipe['sim_options']} {recipe['rules']}")
    rebuild_seconds = time.time() - started
    if not hero_alive(game, recipe):
        item["dropped"] = "hero out before the decision point"
        item["timing"] = {"rebuild_seconds": round(rebuild_seconds, 2)}
        return item
    seat = recipe["hero_seat"]
    from tfteval.planner import PlanPolicy

    if not isinstance(game.seat_policies[seat], PlanPolicy):
        raise ValueError(f"the hero of {lineup!r} is not a plan seat; the continuation needs its executor state")
    fp = fingerprint(game, seat)
    state, _, comp_now = hero_view(game, seat)
    item.update(fingerprint=fp, fingerprint_hash=fingerprint_hash(fp), public_state=state, comp_now=comp_now)
    for c in cands:
        c["now"] = resolve(c["spec"], state)
    snap = snapshot(game)
    del game
    branches = []
    for k in range(kd):
        for c in cands:
            branches.append({**play_branch(snap, seat, c, k, continuation), "set": "discovery"})
    names = [c["name"] for c in cands]
    if early_drop and len(duplicate_groups(branches, names)) == 1:
        item["dropped"] = "all candidates duplicate on the discovery set"
    else:
        for k in range(kd, kd + kl):
            for c in cands:
                branches.append({**play_branch(snap, seat, c, k, continuation), "set": "label"})
    item["branches"] = branches
    if not item["dropped"]:
        lab = label_item(item)
        item["label"] = lab
        item["dropped"] = lab.get("dropped")
    secs = [b["seconds"] for b in branches]
    item["timing"] = {"rebuild_seconds": round(rebuild_seconds, 2), "branches": len(branches),
                      "branch_seconds_mean": round(float(np.mean(secs)), 2) if secs else None,
                      "item_seconds": round(time.time() - started, 2)}
    return item


def extend_item(item: dict, kl: int, harness: str | None = None) -> dict:
    """The item with label branches added up to `kl` (re-seeds kd+old kl .. kd+kl-1) and relabelled.
    The state is rebuilt from the recipe and checked against the fingerprint; the simulator must be the
    one the item was built on, since the new branches have to come from the same game dynamics."""
    from tfteval import simfixes
    from tfteval.branching import snapshot

    if item.get("dropped"):
        raise ValueError(f"{item['id']} is dropped ({item['dropped']}); not extended")
    if simfixes.sim_commit() != item["recipe"].get("sim_commit"):
        raise RuntimeError(f"{item['id']} was built on simulator {item['recipe'].get('sim_commit')}, "
                           f"this one is {simfixes.sim_commit()}")
    started = time.time()
    kd, old = int(item["kd"]), int(item["kl"])
    game = rebuild(item["recipe"], item["fingerprint_hash"])
    seat = item["recipe"]["hero_seat"]
    snap = snapshot(game)
    del game
    branches = list(item["branches"])
    for k in range(kd + old, kd + kl):
        for c in item["candidates"]:
            branches.append({**play_branch(snap, seat, c, k, item["continuation"]), "set": "label"})
    new = {**item, "kl": max(old, kl), "branches": branches}
    new["extended"] = list(item.get("extended", [])) + [{"from_kl": old, "to_kl": kl, "harness_commit": harness,
                                                          "seconds": round(time.time() - started, 2)}]
    lab = label_item(new)
    new["label"], new["dropped"] = lab, lab.get("dropped")
    return new


# --------------------------------------------------------------------------- labels

def t975(df: int) -> float:
    """Two-sided 95% Student-t quantile; exact for df < 5 (where stats.t_quantile is off)."""
    exact = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
    if df < 1:
        return math.inf
    return exact.get(df) or t_quantile(0.975, df)


def duplicate_groups(branches: list[dict], names: list[str]) -> list[list[str]]:
    """Candidates whose hero actions are identical round by round in every branch they share."""
    acts = defaultdict(dict)
    for b in branches:
        acts[b["cand"]][b["k"]] = b["actions"]
    groups: list[list[str]] = []
    for name in names:
        for group in groups:
            ref = group[0]
            ks = set(acts[name]) | set(acts[ref])
            if ks and all(acts[name].get(k) == acts[ref].get(k) for k in ks):
                group.append(name)
                break
        else:
            groups.append([name])
    return groups


def label_item(item: dict, sd_floor: float = SD_FLOOR) -> dict:
    """Best candidate on the discovery set, regrets and intervals on the label set (see the module doc)."""
    names = [c["name"] for c in item["candidates"]]
    kd, kl = int(item["kd"]), int(item["kl"])
    disc, lab = list(range(kd)), list(range(kd, kd + kl))
    places = defaultdict(dict)
    for b in item["branches"]:
        places[b["cand"]][b["k"]] = b["place"]
    out = {"duplicates": duplicate_groups(item["branches"], names)}
    missing = [(c, k) for c in names for k in disc + lab if places[c].get(k) is None]
    if missing:
        return {**out, "dropped": f"unfinished or missing branches: {missing[:4]}"}
    if len(out["duplicates"]) == 1:
        return {**out, "dropped": "all candidates duplicate"}
    d_mean = {c: float(np.mean([places[c][k] for k in disc])) for c in names}
    best = min(names, key=lambda c: (d_mean[c], names.index(c)))
    l_mean = {c: float(np.mean([places[c][k] for k in lab])) for c in names}
    diffs = {c: np.array([places[c][k] - places[best][k] for k in lab], float) for c in names}
    regret = {c: float(diffs[c].mean()) for c in names}
    group_of = {c: g for g in out["duplicates"] for c in g}
    others = [g[0] for g in out["duplicates"] if best not in g]  # one per group
    df = sum(kl - 1 for _ in others)
    pooled = float(np.mean([diffs[c].var(ddof=1) for c in others])) if kl > 1 and others else 0.0
    sd = max(math.sqrt(pooled), sd_floor)
    half = t975(df) * sd / math.sqrt(kl) if df > 0 else math.inf
    within = {c: bool(regret[c] - half <= 0) for c in names}
    clear = bool(others) and all(regret[c] - half > 0 for c in others)
    return {**out, "dropped": None, "best": best, "best_group": group_of[best],
            "discovery_mean": d_mean, "label_mean": l_mean, "regret": regret,
            "ci95": None if math.isinf(half) else half, "sd": sd, "df": df, "within_ci": within,
            "kind": "clear" if clear else "ambiguous"}


# --------------------------------------------------------------------------- agents

def plan_signature(plan: dict, state: dict) -> tuple[int, int]:
    """(gold the plan puts into xp, gold it puts into rerolls) this round, as intended: xp toward the
    plan's target level (level_to, a level_by target, or xp_buys) and rerolls down to its floor (or a
    spend target), rerolls capped at SPEND_CAP (what one round's 15 actions can spend)."""
    knobs = compile_knobs(plan, state)
    level, xp, gold = int(state["level"]), int(state["xp"]), int(state["gold"])
    target = max(level, int(knobs["level_to"]))
    if plan.get("level_by"):
        target = max(target, int(plan["level_by"]["level"]))
    need = xp_to_level(level, xp, min(target, max_level(state)), state.get("rules"))
    buys = max(math.ceil(need / 4), int(knobs["xp_buys"]))
    xp_gold = min(4 * buys, 4 * (gold // 4))
    floor = int(knobs["roll_floor"])
    if plan.get("spend") and not knobs["survival"]:
        floor = min(floor, int(plan["spend"]["to"]))
    roll_gold = min(max(0, gold - xp_gold - floor), SPEND_CAP)
    return xp_gold, roll_gold


def nearest_candidate(plan: dict, state: dict, candidates: list[dict]) -> str:
    """The candidate whose first-round plan is closest to `plan`: L1 distance of plan_signature (gold
    into xp, gold into rerolls). Ties go to the earlier candidate in the menu. Only the first round of a
    commitment can be compared: a planner says what it does now, not what it will do next round."""
    mine = plan_signature(plan, state)
    best, best_d = None, None
    for c in candidates:
        theirs = plan_signature(override(plan, resolve(c["spec"], state)), state)
        d = abs(mine[0] - theirs[0]) + abs(mine[1] - theirs[1])
        if best_d is None or d < best_d:
            best, best_d = c["name"], d
    return best


class PlannerAgent:
    """A rule planner of the repo (`mimic`, `fast8`, `stance`, ... or `rule`, whose economy `mimic`
    reproduces): its own plan at the item's state, mapped to the nearest candidate. With a rebuilt game
    whose hero runs the same planner kind, a copy of the hero's own planner is asked (it keeps the
    game's history, e.g. stance's fast 8 commitment); otherwise a fresh planner of that kind."""

    NOTE_FIELDS = ("stance",)  # plan fields noted besides the economy ones

    def __init__(self, kind: str):
        self.kind = kind
        self.name = kind
        self.planner_kind = "mimic" if kind == "rule" else kind
        plan_kind(self.planner_kind)  # fail early on an unknown kind

    def choose(self, view: dict) -> str:
        hero = view.get("hero_policy")
        planner = getattr(hero, "planner", None)
        if planner is not None and getattr(planner, "name", None) == self.planner_kind:
            planner = copy.deepcopy(planner)
        else:
            planner = plan_kind(self.planner_kind)
        state = copy.deepcopy(view["state"])
        np_state, py_state = np.random.get_state(), _random.getstate()
        try:
            plan = planner.plan(state, view["comps"], view["comp_now"])
        finally:
            np.random.set_state(np_state)
            _random.setstate(py_state)
        view.setdefault("notes", {})["plan"] = {k: plan.get(k) for k in ECONOMY_FIELDS + self.NOTE_FIELDS if k in plan}
        return self.nearest(plan, view["state"], view["candidates"])

    @staticmethod
    def nearest(plan: dict, state: dict, candidates: list[dict]) -> str:
        """How a plan is mapped to a candidate (bank v2 overrides this, tfteval/bank2.py)."""
        return nearest_candidate(plan, state, candidates)


class CallableAgent:
    """Any function view -> candidate name (e.g. a model call)."""

    def __init__(self, fn, name: str = "callable"):
        self.fn, self.name = fn, name

    def choose(self, view: dict) -> str:
        return self.fn(view)


class FixedAgent:
    def __init__(self, name: str):
        self.target, self.name = name, f"always:{name}"

    def choose(self, view: dict) -> str:
        names = [c["name"] for c in view["candidates"]]
        return self.target if self.target in names else names[0]


class RandomAgent:
    """Uniform over the candidates, seeded by the item id (the same choices every run)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def choose(self, view: dict) -> str:
        h = int.from_bytes(hashlib.sha256(f"{self.seed}:{view['id']}".encode()).digest()[:8], "big")
        return view["candidates"][h % len(view["candidates"])]["name"]


class NoisyAgent:
    """Another agent's choice, replaced by a uniform pick with probability N% (by item id): a ladder of
    agents whose strength order is known in advance, as noisyN is for seat policies."""

    def __init__(self, inner, percent: int):
        self.inner, self.percent = inner, int(percent)
        self.name = f"noisy{self.percent}:{inner.name}"

    def choose(self, view: dict) -> str:
        h = int.from_bytes(hashlib.sha256(f"noisy:{view['id']}".encode()).digest()[:8], "big")
        if h % 100 < self.percent:
            return RandomAgent(seed=1).choose(view)
        return self.inner.choose(view)


class LabelAgent:
    """Sanity anchors that read the labels: `oracle` picks the discovery-best candidate (regret 0 by
    definition), `worst` the candidate with the highest label regret."""

    uses_labels = True

    def __init__(self, which: str):
        self.name = which

    def choose(self, view: dict) -> str:
        lab = view["labels"]
        if self.name == "oracle":
            return lab["best"]
        return max(lab["regret"], key=lambda c: lab["regret"][c])


CHOICE_PROMPT = """You are playing Teamfight Tactics (Set 4, 8 players, last one alive wins; lower placement is better).
It is the start of round {point}. Pick ONE of these commitments for the next {rounds} rounds; after them your
usual strategy takes over again.

{candidates}

State (JSON; you, and only the public part of the other players):
{state}

Answer with ONE JSON object: {{"choice": "<one of {names}>", "why": "<one short sentence>"}}"""


def choice_prompt(view: dict) -> str:
    cands = "\n".join(f"- {c['name']}: {c['desc']} This round: {json.dumps(c.get('now', {}))}"
                      for c in view["candidates"])
    return CHOICE_PROMPT.format(point=view["point"], rounds=view["candidates"][0]["rounds"], candidates=cands,
                                state=json.dumps(view["state"], separators=(",", ":")),
                                names=", ".join(c["name"] for c in view["candidates"]))


class CommandAgent:
    """A shell command that reads choice_prompt(view) on stdin and prints a JSON object with "choice"
    (a model, later). An answer without a valid choice is recorded as invalid."""

    def __init__(self, cmd: str, name: str | None = None, timeout: float = 300.0):
        self.cmd, self.name, self.timeout = cmd, name or "cmd", timeout

    def choose(self, view: dict) -> str:
        import re

        out = subprocess.run(self.cmd, shell=True, input=choice_prompt(view), capture_output=True, text=True,
                             timeout=self.timeout).stdout
        for raw in reversed(re.findall(r"\{[^{}]*\}", out)):
            try:
                choice = json.loads(raw).get("choice")
            except json.JSONDecodeError:
                continue
            if choice:
                return str(choice)
        return "invalid"


def make_agent(spec: str):
    """`oracle`, `worst`, `random`, `always:<candidate>`, `noisy<N>:<agent>`, `cmd:<shell command>`,
    `py:<module>:<function>` (a callable view -> name), or a plan-seat kind (`stance`, `mimic`, `fast8`,
    ...; `rule` = the rule bot's economy, as `mimic`)."""
    if spec in ("oracle", "worst"):
        return LabelAgent(spec)
    if spec == "random":
        return RandomAgent()
    if spec.startswith("always:"):
        return FixedAgent(spec.split(":", 1)[1])
    if spec.startswith("cmd:"):
        return CommandAgent(spec.split(":", 1)[1])
    if spec.startswith("py:"):  # py:package.module:function, a callable view -> name
        import importlib

        module, _, fn = spec[3:].rpartition(":")
        return CallableAgent(getattr(importlib.import_module(module), fn), name=spec)
    if spec.startswith("noisy") and ":" in spec:
        head, inner = spec.split(":", 1)
        return NoisyAgent(make_agent(inner), int(head[5:]))
    return PlannerAgent(spec)


# --------------------------------------------------------------------------- scoring

def load_items(path) -> list[dict]:
    """The items of a bank file; an item written again (extended with more label branches) replaces
    its earlier line, in the order items first appeared."""
    items: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["id"]] = item
    return list(items.values())


def usable(items: list[dict], sd_floor: float = SD_FLOOR) -> list[dict]:
    """Items with labels (recomputed from their branches), dropped ones left out."""
    out = []
    for it in items:
        if it.get("dropped") or not it.get("branches"):
            continue
        lab = label_item(it, sd_floor)
        if lab.get("dropped"):
            continue
        out.append({**it, "label": lab})
    return out


def views(items: list[dict], rebuild_states: bool = True):
    """(item, view) per item. With `rebuild_states`, each source game is played once through its
    decision points (continuing one game is the same as play_to each point) and every state is checked
    against the item's fingerprint; the view then carries the hero's policy. Otherwise the view is built
    from the stored public state."""
    by_game = defaultdict(list)
    for it in items:
        by_game[it["game"]].append(it)
    for game_items in by_game.values():
        game_items.sort(key=lambda it: it["recipe"]["round"])
        game = None
        for it in game_items:
            view = {"id": it["id"], "point": it["point"], "round": it["recipe"]["round"],
                    "candidates": [{k: c[k] for k in ("name", "desc", "rounds", "spec", "now") if k in c}
                                   for c in it["candidates"]],
                    "state": it["public_state"], "comp_now": it.get("comp_now"), "comps": None,
                    "hero_policy": None}
            if rebuild_states:
                recipe = it["recipe"]
                if game is None:
                    game = new_game(recipe)
                game.run(until_round=recipe["round"])
                seat = recipe["hero_seat"]
                if not hero_alive(game, recipe):
                    raise RuntimeError(f"{it['id']}: hero not alive on rebuild")
                got = fingerprint_hash(fingerprint(game, seat))
                if got != it["fingerprint_hash"]:
                    raise RuntimeError(f"{it['id']}: rebuilt state differs (fingerprint {got} != "
                                       f"{it['fingerprint_hash']})")
                state, comps, comp_now = hero_view(game, seat)
                view.update(state=state, comps=comps, comp_now=comp_now, hero_policy=game.seat_policies[seat])
            if view["comps"] is None:
                from tfteval.executor import COMPS

                view["comps"] = dict(COMPS)
            yield it, view


def choose_all(items: list[dict], agents: list, rebuild_states: bool = True) -> dict:
    """agent name -> item id -> {"choice", "notes"} for every usable item; each state is built once and
    shown to every agent (each gets its own copy of the view)."""
    out = {agent.name: {} for agent in agents}
    for it, view in views(items, rebuild_states):
        for agent in agents:
            mine = dict(view, state=copy.deepcopy(view["state"]))
            if getattr(agent, "uses_labels", False):
                mine["labels"] = it["label"]
            choice = agent.choose(mine)
            out[agent.name][it["id"]] = {"choice": choice, "notes": mine.get("notes")}
    return out


def clustered_mean_ci(values, clusters) -> dict:
    """Mean of item values with a 95% t interval clustered by source game (cluster-robust standard
    error of the mean, G - 1 degrees of freedom)."""
    values = np.asarray(values, float)
    n = len(values)
    if n == 0:
        return {"n": 0, "games": 0, "mean": None, "ci95": None}
    mean = float(values.mean())
    sums = defaultdict(float)
    for v, c in zip(values, clusters):
        sums[c] += v - mean
    g = len(sums)
    if g < 2:
        return {"n": n, "games": g, "mean": mean, "ci95": None}
    se = math.sqrt(sum(s * s for s in sums.values()) * g / (g - 1)) / n
    return {"n": n, "games": g, "mean": mean, "ci95": t975(g - 1) * se}


def score_rows(items: list[dict], choices: dict) -> list[dict]:
    rows = []
    for it in items:
        lab = it["label"]
        choice = choices[it["id"]]["choice"]
        names = [c["name"] for c in it["candidates"]]
        valid = choice in names
        regret = lab["regret"][choice] if valid else max(lab["regret"].values())
        rows.append({"id": it["id"], "game": it["game"], "split": it["split"], "point": it["point"],
                     "kind": lab["kind"], "choice": choice, "valid": valid, "best": lab["best"],
                     "regret": regret, "within_ci": bool(valid and lab["within_ci"][choice]),
                     "best_group": bool(valid and choice in lab["best_group"])})
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"items": 0}
    games = [r["game"] for r in rows]
    out = {"items": len(rows), "games": len(set(games)),
           "regret": clustered_mean_ci([r["regret"] for r in rows], games),
           "within_ci": clustered_mean_ci([float(r["within_ci"]) for r in rows], games),
           "best_choice": float(np.mean([r["best_group"] for r in rows])),
           "invalid": sum(not r["valid"] for r in rows)}
    out["choices"] = {c: sum(r["choice"] == c for r in rows) for c in sorted({r["choice"] for r in rows})}
    return out


def scorecard(items: list[dict], choices: dict, agent_name: str, bank: str | None = None) -> dict:
    """Mean regret and within-interval share: all / clear / ambiguous, per decision point, per split."""
    rows = score_rows(items, choices)
    card = {"agent": agent_name, "bank": bank, "items": len(rows), "splits": {}}
    for sp in ("all", "dev", "heldout"):
        part = [r for r in rows if sp == "all" or r["split"] == sp]
        entry = {kind: summarize_rows([r for r in part if kind == "all" or r["kind"] == kind])
                 for kind in ("all", "clear", "ambiguous")}
        entry["points"] = {p: summarize_rows([r for r in part if r["point"] == p])
                           for p in POINTS if any(r["point"] == p for r in part)}
        card["splits"][sp] = entry
    card["rows"] = rows
    return card


def paired(card_a: dict, card_b: dict) -> dict:
    """Regret of A minus regret of B on the same items, clustered by source game."""
    b = {r["id"]: r for r in card_b["rows"]}
    shared = [r for r in card_a["rows"] if r["id"] in b]
    return clustered_mean_ci([r["regret"] - b[r["id"]]["regret"] for r in shared], [r["game"] for r in shared])


def fmt_ci(entry: dict | None) -> str:
    if not entry or entry.get("mean") is None:
        return "-"
    ci = entry.get("ci95")
    return f"{entry['mean']:+.2f}" + (f" ±{ci:.2f}" if ci is not None else " (1 game)")


def table(card: dict) -> str:
    lines = [f"{card['agent']}: {card['items']} items",
             f"{'split':8} {'subset':10} {'items':>5} {'games':>5}  {'regret (95% CI by game)':24} "
             f"{'within CI':>9} {'best':>5}  choices"]
    for sp, entry in card["splits"].items():
        subsets = [(k, entry[k]) for k in ("all", "clear", "ambiguous")] + list(entry["points"].items())
        for name, s in subsets:
            if not s.get("items"):
                continue
            lines.append(f"{sp:8} {name:10} {s['items']:>5} {s['games']:>5}  {fmt_ci(s['regret']):24} "
                         f"{s['within_ci']['mean']:>9.0%} {s['best_choice']:>5.0%}  "
                         + " ".join(f"{c}:{n}" for c, n in s["choices"].items()))
    return "\n".join(lines)


def bank_summary(items: list[dict]) -> dict:
    """Counts of the bank itself: items built, dropped (by reason), clear / ambiguous, per point."""
    built = len(items)
    dropped = defaultdict(int)
    for it in items:
        if it.get("dropped"):
            dropped[it["dropped"].split(":")[0]] += 1
    good = usable(items)
    per_point = {p: {"clear": sum(it["point"] == p and it["label"]["kind"] == "clear" for it in good),
                     "ambiguous": sum(it["point"] == p and it["label"]["kind"] == "ambiguous" for it in good)}
                 for p in POINTS}
    secs = [b["seconds"] for it in items for b in it.get("branches", [])]
    return {"built": built, "dropped": dict(dropped), "usable": len(good),
            "clear": sum(it["label"]["kind"] == "clear" for it in good), "per_point": per_point,
            "games": len({it["game"] for it in good}),
            "splits": {sp: sum(it["split"] == sp for it in good) for sp in ("dev", "heldout")},
            "branch_seconds_mean": float(np.mean(secs)) if secs else None}

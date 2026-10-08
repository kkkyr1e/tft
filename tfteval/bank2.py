"""Decision bank v2 (docs/BANK_V2.md): triggered decision points, paired racing labels, empirical-Bayes
clear items, cross-fit anchors and a check that no simple rule solves a stratum.

What stays from v1 (tfteval/bank.py): an item is one hero seat of one source game at one decision point,
stored as a recipe that rebuilds the state (checked against a fingerprint), a menu of candidate
commitments (economy plan fields held for COMMIT_ROUNDS planning phases on top of the continuation
planner, `bank.CommitPlanner`), and branches: the game played on from the saved state with re-seed k
until the hero is out, the hero's place being the outcome. The same k is the same random future for
every candidate (common random numbers).

What is new:

Triggered decision points (STRATA). A stratum is a round window, a trigger (a predicate over the hero's
public state at the start of a planning phase, `trigger_features`: HP, gold, level, win / loss streak,
pairs on board and bench, losses to death) and a menu. The source game is played round by round through
the window and the item is made at the FIRST planning phase where the trigger fires; that round goes into
the recipe, so a rebuild is `bank.play_recipe` as in v1 (deterministic, fingerprint-checked). A game
where the trigger never fires (or the hero is out first) is still written, with `dropped` saying why and
the scanned rounds in `scan`, so hit rates can be counted from the bank file.

Paired racing labels (`race`). Every candidate gets k = 0 .. K_START-1. Then, at every look, the leader
is the candidate with the lowest mean place over the ks all active candidates share; a candidate whose
paired mean difference to the leader exceeds ELIM_Z paired standard errors is eliminated (standard error
max(SD, RACE_SD_FLOOR) / sqrt(n): with a few branches a sample SD of 0 happens by chance). The survivors
get K_STEP more ks each while the item's budget (BUDGET_PER_CAND branches per distinct candidate) allows,
until one is left or they reach K_MIN (and, budget permitting, at most K_MAX). Branches are k-prefixes per
candidate: an eliminated candidate keeps the ks it had. At the first look, a candidate whose hero actions
are identical to an earlier candidate's (menu order) in all its branches is merged into it: it gets no
more branches, its first ones do not count toward the budget (the item is raced as if the menu were its
distinct candidates), and it shares the earlier one's labels (`race.merged`; e.g. two roll depths that
a round's 15 actions cannot tell apart). The race is a deterministic function of the outcomes, so a
stored item can be re-raced (e.g. with a larger budget) by replaying its stored branches and playing the
missing ones.

Labels (`fit_stratum`, `label_items`), computed at scoring time from the stored branches:

* Noise. Per item, the per-branch variance of the paired difference between a candidate and the
  reference candidate (the one with the most ks) is pooled over the item's candidates, shrunk toward
  the stratum's pooled value with VAR_PRIOR_DF degrees of freedom and floored at SD_FLOOR^2. The model
  is place(c, k) = mu_c + a_k + e_ck with e iid (sigma^2 = sd_d^2 / 2), which gives the covariance of
  the paired contrast estimates d_c = mean over shared k of place(c, k) - place(ref, k).
* Prior (random effects, method of moments per stratum): mu_c = m + alpha_c + u_c, u_c ~ N(0, tau^2).
  alpha_c (the stratum's average effect of each candidate) and its sampling covariance come from the
  items' centred contrast vectors; tau^2 from the spread of the paired contrasts across items minus their
  sampling variance. With fewer than MIN_FIT_ITEMS items alpha is 0 and tau is TAU_FALLBACK.
* Posterior: Gaussian on the contrasts; P(candidate is truly best) by Monte Carlo (POSTERIOR_DRAWS, seeded
  by the item id). Duplicate candidates (identical hero actions in every branch, or merged at the race's
  first look) are one group and share its probability and its representative's regret. Clear item: max P >= CLEAR_P. Accuracy ceiling: the sum of max P over clear
  items; an agent's expected hits are the sum of P(its choice) over the clear items.
* Regret of a candidate: its paired mean difference to the posterior-best candidate over the ks both
  have (all branches; an eliminated candidate is compared on its own ks).
* Cross-fit anchors: the oracle picks the best candidate on the odd ks (paired contrasts) and is scored
  on the even ks, and the other way round, averaged; the cross-fit worst likewise picks the worst. No
  anchor reads the labels it is scored on.

Agents are v1's (`bank.make_agent`: plan-seat kinds via the nearest candidate, `random`, `always:<c>`,
`noisy<N>:<agent>`, `cmd:`, `py:`) except that plans are mapped with `nearest_candidate` below (fodder
counts) and the label-reading `oracle` / `worst` do not exist here.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from collections import defaultdict

import numpy as np

from tfteval import bank, stages

SCHEMA = 2

# --------------------------------------------------------------------------- candidates

COMMIT_ROUNDS = 2  # planning phases a commitment lasts (docs/BANK_V2.md)

CANDIDATES = {
    "save": dict(bank.CANDIDATES["save"]),
    "roll10": {"level": "hold", "roll_floor": 10,
               "desc": "Roll down: reroll down to 10 gold every round at the current level, no xp."},
    "level7": {"level": "7+", "keep": 0, "roll_floor": 999,
               "desc": "Level to 7: buy xp toward level 7 (one level up when already 7 or more) with all the "
                       "gold (partial levels allowed), no rerolls; then hold."},
    "level8": {"level": 8, "keep": 0, "roll_floor": 999, "then": "base",
               "desc": "Level to 8 now: buy xp toward level 8 with all the gold, no rerolls; once at 8 the "
                       "usual strategy decides whether to hold or roll."},
    "roll30": {"level": "hold", "roll_floor": 30,
               "desc": "Hold the level and reroll down to 30 gold every round, no xp."},
    "streak": {"level": "curve", "keep": 50, "roll_floor": 999, "fodder": True,
               "desc": "Keep the loss streak: field the weakest units (the strong ones stay on the bench), no "
                       "rerolls, xp only toward the lobby curve and only with the gold above 50."},
    "roll20": {"level": "hold", "roll_floor": 20,
               "desc": "Stabilize: reroll down to 20 gold every round at the current level, no xp."},
    "level_roll10": {"level": "+1", "keep": 10, "roll_floor": 10,
                     "desc": "Level, then roll: buy xp toward one level up with the gold above 10, then reroll "
                             "down to 10 gold."},
    "roll_all": {"level": "hold", "roll_floor": 0,
                 "desc": "All in: reroll at the current level as much as the round's actions allow (down to 0 "
                         "gold) every round, no xp."},
}

# --------------------------------------------------------------------------- strata (triggers)

PAIRS_WINDOW = ("3-2", "3-5")
PAIRS_MIN = 2                 # pairs (two 1-star copies of a unit) on board + bench
PAIRS_GOLD = (30, 50)
STREAK8_WINDOW = ("4-1", "4-2")
STREAK8_GOLD = 50
STREAK8_WINS = 2
STREAK8_MAX_LEVEL = 7         # "level to 8 now / hold 7": the menu assumes level 7 or below
LOSING_WINDOW = ("3-1", "3-3")
LOSING_LOSSES = 2             # streak <= -2
LOSING_HP = 60
# `streak` fields the weakest units; `save` (no rerolls, xp only toward the lobby curve with the gold above 50)
# is the same economy on the usual board, which is what stance's keep_streak does (docs/BANK_V2.md, interim
# results): without it, nearest_candidate maps that plan to `streak`. Added after the first 20 items.
LOSING_MENU = ("streak", "save", "roll20", "level_roll10")
LOWHP_WINDOW = ("4-1", "5-1")
LOWHP_HP = 50
LOWHP_GOLD = 40
REF_WINDOW = ("4-1", "4-1")
REF_SAMPLE_MOD = 3            # ref41 items only for source games with seed % REF_SAMPLE_MOD == 0

# Roll depths that the round's 15 actions (planner.ACTIONS_PER_ROUND) cannot tell apart are one decision:
# at 40 gold "roll to 0" and "roll to 20" both rolled 10 times in the pilot. The deep option is therefore
# "as much as the cap allows" (roll_all), the shallow one roll30 (5 rolls at 40 gold).
LOWHP_MENU = ("roll_all", "roll30", "level_roll10", "save")

# name -> round window (first, last stage label), trigger (feature_min / feature_max bounds over
# trigger_features), menu in tie-break order, commitment length, and sample_mod: only source games with
# seed % sample_mod == 0 get an item of the stratum (1: every game)
STRATA = {
    "pairs3": {"window": PAIRS_WINDOW, "when": {"pairs_min": PAIRS_MIN, "gold_min": PAIRS_GOLD[0],
                                                 "gold_max": PAIRS_GOLD[1]},
               "menu": ("roll10", "level7", "save"), "rounds": COMMIT_ROUNDS, "sample_mod": 1},
    "streak4": {"window": STREAK8_WINDOW, "when": {"gold_min": STREAK8_GOLD, "win_streak_min": STREAK8_WINS,
                                                   "level_max": STREAK8_MAX_LEVEL},
                "menu": ("level8", "roll30", "save"), "rounds": COMMIT_ROUNDS, "sample_mod": 1},
    "losing3": {"window": LOSING_WINDOW, "when": {"lose_streak_min": LOSING_LOSSES, "hp_min": LOSING_HP},
                "menu": LOSING_MENU, "rounds": COMMIT_ROUNDS, "sample_mod": 1},
    "lowhp4": {"window": LOWHP_WINDOW, "when": {"hp_max": LOWHP_HP, "gold_min": LOWHP_GOLD},
               "menu": LOWHP_MENU, "rounds": COMMIT_ROUNDS, "sample_mod": 1},
    "ref41": {"window": REF_WINDOW, "when": {}, "menu": LOWHP_MENU, "rounds": COMMIT_ROUNDS,
              "sample_mod": REF_SAMPLE_MOD},
}

TRIGGER_FEATURES = ("round", "hp", "gold", "level", "streak", "win_streak", "lose_streak", "pairs",
                    "losses_to_death")


def trigger_features(state: dict) -> dict:
    """The features triggers and rules read, from a describe() state (the hero's public state)."""
    from tfteval.stance import own_units

    streak = int(state["streak"])
    return {"round": int(state["round"]), "stage": state.get("stage") or stages.label(int(state["round"])),
            "hp": int(state["hp"]), "gold": int(state["gold"]), "level": int(state["level"]), "streak": streak,
            "win_streak": max(streak, 0), "lose_streak": max(-streak, 0), "pairs": int(own_units(state)["pairs"]),
            "losses_to_death": int(state["losses_to_death"])}


def fires(when: dict, features: dict) -> bool:
    """Every bound of a trigger holds: `<feature>_min` (>=) and `<feature>_max` (<=)."""
    for key, bound in when.items():
        name, _, side = key.rpartition("_")
        if side not in ("min", "max") or name not in TRIGGER_FEATURES:
            raise ValueError(f"bad trigger bound {key!r}")
        value = features[name]
        if (side == "min" and value < bound) or (side == "max" and value > bound):
            return False
    return True


def window_rounds(stratum: str) -> list[int]:
    first, last = STRATA[stratum]["window"]
    return list(range(stages.parse_label(first), stages.parse_label(last) + 1))


def trigger_spec(stratum: str) -> dict:
    """What an item records of its stratum's definition (a build refuses a file whose items differ)."""
    s = STRATA[stratum]
    return {"window": list(s["window"]), "when": dict(s["when"]), "menu": list(s["menu"]), "rounds": int(s["rounds"]),
            "sample_mod": int(s["sample_mod"])}


def sampled(stratum: str, seed: int) -> bool:
    """Whether the source game with this seed gets an item of the stratum (ref41: every REF_SAMPLE_MOD-th)."""
    return int(seed) % int(STRATA[stratum]["sample_mod"]) == 0


def menu(stratum: str) -> list[dict]:
    s = STRATA[stratum]
    return [{"name": name, "desc": CANDIDATES[name]["desc"], "rounds": int(s["rounds"]),
             "spec": {k: v for k, v in CANDIDATES[name].items() if k != "desc"}} for name in s["menu"]]


def extend_menu(item: dict) -> dict:
    """The item with its stratum's current menu when that only adds candidates to the stored one (same
    order): stored candidates and their branches kept, the new ones without branches (race them with
    rerace_item), the trigger spec updated. ValueError when the menus differ otherwise."""
    old = [c["name"] for c in item["candidates"]]
    new = list(STRATA[item["stratum"]]["menu"])
    if [c for c in new if c in old] != old:
        raise ValueError(f"{item['id']}: menu {old} is not part of the current {new}")
    if old == new:
        return item
    have = {c["name"]: c for c in item["candidates"]}
    cands = []
    for c in menu(item["stratum"]):
        if c["name"] in have:
            cands.append(have[c["name"]])
        else:
            if item.get("public_state") is not None:
                c["now"] = bank.resolve(c["spec"], item["public_state"])
            cands.append(c)
    out = {**item, "candidates": cands, "trigger": trigger_spec(item["stratum"])}
    out["menu_extended"] = list(item.get("menu_extended", [])) + [{"from": old, "to": new}]
    return out


def first_fire(rows: list[dict], stratum: str) -> int | None:
    """The first scanned round (trigger_features rows) inside the stratum's window where it fires."""
    rounds = set(window_rounds(stratum))
    for row in sorted(rows, key=lambda r: r["round"]):
        if row["round"] in rounds and fires(STRATA[stratum]["when"], row):
            return int(row["round"])
    return None


# --------------------------------------------------------------------------- agents

FODDER_GOLD = 2  # a fodder mismatch counts like this much gold in nearest_candidate (one reroll)


def plan_signature(plan: dict, state: dict) -> tuple[int, int, bool]:
    """bank.plan_signature (gold into xp, gold into rerolls this round) plus whether it fields fodder."""
    from tfteval.planner import compile_knobs

    xp, roll = bank.plan_signature(plan, state)
    return xp, roll, bool(compile_knobs(plan, state)["fodder"])


def nearest_candidate(plan: dict, state: dict, candidates: list[dict]) -> str:
    """bank.nearest_candidate with the v2 candidates: L1 distance of the gold into xp and into rerolls
    this round, plus FODDER_GOLD when one fields a fodder board and the other does not (the `streak`
    candidate). A candidate whose first round is the planner's own (`then: "base"` with the target level
    already reached) is the plan itself. Ties go to the earlier candidate in the menu."""
    mine = plan_signature(plan, state)
    best, best_d = None, None
    for c in candidates:
        fields = bank.resolve(c["spec"], state)
        theirs = plan_signature(plan if fields is None else bank.override(plan, fields), state)
        d = abs(mine[0] - theirs[0]) + abs(mine[1] - theirs[1]) + FODDER_GOLD * (mine[2] != theirs[2])
        if best_d is None or d < best_d:
            best, best_d = c["name"], d
    return best


class PlannerAgent(bank.PlannerAgent):
    """bank.PlannerAgent with the v2 mapping of a plan to a candidate."""

    NOTE_FIELDS = ("stance", "fodder")

    @staticmethod
    def nearest(plan: dict, state: dict, candidates: list[dict]) -> str:
        return nearest_candidate(plan, state, candidates)


CROSSFIT_AGENTS = ("xfit:oracle", "xfit:worst")


def make_agent(spec: str):
    """bank.make_agent for v2 items: the same specs, plan kinds mapped with the v2 nearest_candidate.
    The label-reading `oracle` and `worst` are refused (use the cross-fit anchors xfit:oracle and
    xfit:worst, which scorecards compute from the branches)."""
    if spec in ("oracle", "worst"):
        raise ValueError(f"{spec!r} reads the labels it is scored on; bank v2 reports xfit:{spec} instead")
    if spec in CROSSFIT_AGENTS:
        raise ValueError(f"{spec} is computed from the branches, not asked (scorecards add it)")
    if spec.startswith("noisy") and ":" in spec:
        head, inner = spec.split(":", 1)
        return bank.NoisyAgent(make_agent(inner), int(head[5:]))
    if spec == "random" or spec.startswith(("always:", "cmd:", "py:")):
        return bank.make_agent(spec)
    return PlannerAgent(spec)


# --------------------------------------------------------------------------- racing

K_START = 8         # branches every candidate gets first
K_STEP = 4          # branches added to each survivor per look
K_MIN = 24          # survivors stop here when the budget is spent (or one is left) ...
K_MAX = 32          # ... and never get more than this
BUDGET_PER_CAND = 16  # an item's budget is this times its distinct candidates (3 -> 48, 4 -> 64 branches)
ELIM_Z = 2.0        # eliminate at a paired mean difference > ELIM_Z standard errors behind the leader
RACE_SD_FLOOR = bank.SD_FLOOR  # floor of the paired SD in the standard error (places)

RACING = {"k_start": K_START, "k_step": K_STEP, "k_min": K_MIN, "k_max": K_MAX, "budget_per_cand": BUDGET_PER_CAND,
          "z": ELIM_Z, "sd_floor": RACE_SD_FLOOR}


def racing_config(**overrides) -> dict:
    cfg = dict(RACING)
    for key, value in overrides.items():
        if value is not None:
            if key not in cfg:
                raise ValueError(f"unknown racing setting {key!r}")
            cfg[key] = type(cfg[key])(value)
    if not 1 <= cfg["k_start"] <= cfg["k_max"] or cfg["k_min"] > cfg["k_max"] or cfg["k_step"] < 1:
        raise ValueError(f"inconsistent racing settings {cfg}")
    return cfg


def eliminate(places: dict, active: list[str], ks: list[int], z: float, sd_floor: float,
              order: list[str] | None = None) -> tuple[str, dict, list[str]]:
    """One look of the race. places: candidate -> k -> place. The leader has the lowest mean over `ks`
    (ties: `order`); every other active candidate with a paired mean difference to the leader greater
    than z * max(SD, sd_floor) / sqrt(n) is eliminated. Returns (leader, stats, eliminated)."""
    order = list(order or active)
    means = {c: float(np.mean([places[c][k] for k in ks])) for c in active}
    leader = min(active, key=lambda c: (means[c], order.index(c)))
    stats, out = {}, []
    for c in active:
        if c == leader:
            continue
        d = np.array([places[c][k] - places[leader][k] for k in ks], float)
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        se = max(sd, sd_floor) / math.sqrt(len(d))
        stats[c] = {"diff": float(d.mean()), "sd": sd, "se": se}
        if d.mean() > z * se:
            out.append(c)
    return leader, {"means": means, "vs_leader": stats}, out


def race(names: list[str], play, cfg: dict, cache: dict | None = None) -> dict:
    """Successive elimination over paired branches. `play(name, k)` returns a branch dict with "place";
    `cache` (name -> k -> branch) is used before playing (re-racing a stored item). At the first look,
    a candidate whose hero actions equal an earlier candidate's (menu order) in every branch so far is
    merged into it (`merged`: later -> earlier) and gets no more branches; the budget is
    budget_per_cand times the distinct candidates, the merged ones' first branches not counted. Returns
    the branches used (k-major order), the looks, why it stopped, the survivors, the ks per candidate
    and the merges."""
    have = {c: {} for c in names}
    used: dict[str, int] = {c: 0 for c in names}
    branches: list[dict] = []
    cache = cache or {}

    def get(c: str, k: int, look: int) -> dict:
        b = cache.get(c, {}).get(k)
        if b is None:
            b = play(c, k)
        b = {**b, "look": look}
        have[c][k] = b
        branches.append(b)
        return b

    active, looks, stop, merged = list(names), [], None, {}
    n = 0
    target = cfg["k_start"]
    while True:
        for k in range(n, target):
            for c in active:
                get(c, k, len(looks))
        for c in active:
            used[c] = target
        n = target
        places = {c: {k: b["place"] for k, b in have[c].items()} for c in names}
        if any(p is None for c in active for p in places[c].values()):
            stop = "unfinished branch"
            break
        first = not looks
        if first:  # merge duplicates (identical hero actions in every branch so far) into the earlier one
            groups = bank.duplicate_groups(branches, names)
            if len(groups) == 1:
                looks.append({"n": n, "active": list(active), "merged": {}, "leader": None, "eliminated": []})
                stop = "all candidates duplicate"
                break
            merged = {later: g[0] for g in groups for later in g[1:]}
            active = [c for c in active if c not in merged]
        budget = cfg["budget_per_cand"] * (len(names) - len(merged))
        leader, stats, out = eliminate(places, active, list(range(n)), cfg["z"], cfg["sd_floor"], names)
        looks.append({"n": n, "active": list(active), **({"merged": dict(merged)} if first else {}),
                      "leader": leader, **stats, "eliminated": out})
        active = [c for c in active if c not in out]
        if len(active) == 1:
            stop = "one left"
            break
        if n >= cfg["k_max"]:
            stop = "k_max"
            break
        step = min(cfg["k_step"], cfg["k_max"] - n)
        room = budget - sum(m for c, m in used.items() if c not in merged)
        if step * len(active) > room:
            if n >= cfg["k_min"]:
                stop = "k_min, budget spent"
                break
            step = room // len(active)
            if step <= 0:
                stop = "budget spent"
                break
        target = n + step
    return {"branches": branches, "looks": looks, "stop": stop, "survivors": active, "n": used, "merged": merged}


def replay_race(item: dict, cfg: dict | None = None) -> dict:
    """The race of a stored item replayed on its stored branches (KeyError when a branch it needs is
    missing, i.e. the config asks for more than was played)."""
    cache = defaultdict(dict)
    for b in item["branches"]:
        cache[b["cand"]][b["k"]] = b
    names = [c["name"] for c in item["candidates"]]

    def missing(c, k):
        raise KeyError(f"{item['id']}: no stored branch for {c} k={k}")

    return race(names, missing, cfg or item["racing"], cache)


# --------------------------------------------------------------------------- building

def item_id(lineup: str, seed: int, stratum: str) -> str:
    return f"{lineup}#{seed}@{stratum}"


def _alive(game, seat: str, rnd: int) -> bool:
    return not game.done and game.round == rnd and game.player(seat) is not None and seat not in game.placements


def scan_game(lineup: str, seed: int, settings: dict | None = None, last: str | int = "5-1",
              first: str | int = "3-1") -> dict:
    """Play a source game through the planning phases first..last and record trigger_features of the
    hero at each one (no branches): the trigger hit rates of every stratum, in one game. `fired` is the
    round an item would be made at, None for a stratum the seed is not sampled for (`sampled`)."""
    recipe = bank.make_recipe(lineup, seed, stages.label(stages.parse_label(first)), settings)
    game = bank.new_game(recipe)
    seat = recipe["hero_seat"]
    rows, out_at = [], None
    started = time.time()
    for rnd in range(stages.parse_label(first), stages.parse_label(last) + 1):
        game.run(until_round=rnd)
        if not _alive(game, seat, rnd):
            out_at = rnd
            break
        state, _, _ = bank.hero_view(game, seat)
        rows.append(trigger_features(state))
    fired = {s: first_fire(rows, s) if sampled(s, seed) else None for s in STRATA}
    return {"game": f"{lineup}#{seed}", "lineup": lineup, "seed": int(seed), "hero_seat": seat, "rows": rows,
            "out_at": out_at, "fired": fired, "sampled": {s: sampled(s, seed) for s in STRATA},
            "seconds": round(time.time() - started, 2),
            "settings": {k: recipe[k] for k in bank.SETTING_KEYS}}


def _play(snap, seat: str, cands: dict, continuation: str):
    def play(name: str, k: int) -> dict:
        return bank.play_branch(snap, seat, cands[name], k, continuation)
    return play


def build_item(lineup: str, seed: int, stratum: str, cfg: dict | None = None, continuation: str = "stance",
               settings: dict | None = None, harness: str | None = None, donors: list[dict] = ()) -> dict:
    """Play a source game through the stratum's window to the first round its trigger fires, save it
    and race the menu's candidates (see the module doc). `donors`: built items of the same game whose
    state and menu may be this one's (ref41 at 4-1 is lowhp4 when that fires at 4-1); a donor with the
    same round, fingerprint, menu, racing settings and continuation lends its branches (`copied_from`)."""
    from tfteval import simfixes
    from tfteval.branching import snapshot
    from tfteval.planner import PlanPolicy

    if stratum not in STRATA:
        raise ValueError(f"no stratum {stratum!r}; choose from {list(STRATA)}")
    if not sampled(stratum, seed):
        raise ValueError(f"seed {seed} is not sampled for {stratum} (seed % {STRATA[stratum]['sample_mod']} != 0)")
    cfg = dict(cfg or RACING)
    started = time.time()
    rounds = window_rounds(stratum)
    recipe = bank.make_recipe(lineup, seed, stages.label(rounds[0]), settings)
    recipe.update(sim_commit=None, harness_commit=harness)
    cands = menu(stratum)
    item = {"schema": SCHEMA, "id": item_id(lineup, seed, stratum), "game": f"{lineup}#{seed}", "lineup": lineup,
            "seed": int(seed), "stratum": stratum, "point": None, "split": bank.split(seed),
            "trigger": trigger_spec(stratum), "recipe": recipe, "continuation": continuation, "racing": cfg,
            "candidates": cands, "scan": [], "dropped": None}
    game = bank.new_game(recipe)
    recipe["sim_commit"], recipe["sim_fixes"] = simfixes.sim_commit(), list(game.sim_fixes)
    recipe["carousel_pickers"] = list(getattr(game, "carousel_pickers", []))
    if (game.sim, game.sim_options, game.rules) != (recipe["sim"], recipe["sim_options"], recipe["rules"]):
        raise RuntimeError(f"game played {game.sim} {game.sim_options} {game.rules}, the recipe says "
                           f"{recipe['sim']} {recipe['sim_options']} {recipe['rules']}")
    seat = recipe["hero_seat"]
    if not isinstance(game.seat_policies[seat], PlanPolicy):
        raise ValueError(f"the hero of {lineup!r} is not a plan seat; the continuation needs its executor state")
    fired = None
    for rnd in rounds:
        game.run(until_round=rnd)
        if not _alive(game, seat, rnd):
            item["dropped"] = "hero out before the trigger fired"
            break
        state, _, comp_now = bank.hero_view(game, seat)
        f = trigger_features(state)
        ok = fires(STRATA[stratum]["when"], f)
        item["scan"].append({**f, "fired": ok})
        if ok:
            fired = rnd
            break
    else:
        item["dropped"] = "trigger did not fire"
    scan_seconds = time.time() - started
    if fired is None:
        recipe["round"], recipe["point"] = None, None
        item["timing"] = {"scan_seconds": round(scan_seconds, 2)}
        return item
    recipe["round"], recipe["point"] = fired, stages.label(fired)
    item["point"] = stages.label(fired)
    fp = bank.fingerprint(game, seat)
    item.update(fingerprint=fp, fingerprint_hash=bank.fingerprint_hash(fp), public_state=state, comp_now=comp_now,
                features=f)
    for c in cands:
        c["now"] = bank.resolve(c["spec"], state)
    for donor in donors or ():
        if (donor.get("fingerprint_hash") == item["fingerprint_hash"] and donor["recipe"]["round"] == fired
                and donor["game"] == item["game"] and donor.get("racing") == cfg
                and donor.get("continuation") == continuation and not donor.get("dropped")
                and [c["name"] for c in donor["candidates"]] == [c["name"] for c in cands]
                and [c["spec"] for c in donor["candidates"]] == [c["spec"] for c in cands]):
            item.update(branches=copy.deepcopy(donor["branches"]), race=copy.deepcopy(donor["race"]),
                        copied_from=donor["id"])
            item["dropped"] = donor.get("dropped")
            item["timing"] = {"scan_seconds": round(scan_seconds, 2), "branches": 0, "copied": len(item["branches"]),
                              "item_seconds": round(time.time() - started, 2)}
            return item
    snap = snapshot(game)
    del game
    result = race([c["name"] for c in cands], _play(snap, seat, {c["name"]: c for c in cands}, continuation), cfg)
    _store_race(item, result)
    secs = [b["seconds"] for b in item["branches"]]
    item["timing"] = {"scan_seconds": round(scan_seconds, 2), "branches": len(secs),
                      "branch_seconds_mean": round(float(np.mean(secs)), 2) if secs else None,
                      "item_seconds": round(time.time() - started, 2)}
    return item


def _store_race(item: dict, result: dict) -> None:
    item["branches"] = result["branches"]
    item["race"] = {k: result[k] for k in ("looks", "stop", "survivors", "n", "merged")}
    if result["stop"] in ("all candidates duplicate", "unfinished branch"):
        item["dropped"] = result["stop"]


def rerace_item(item: dict, cfg: dict, harness: str | None = None) -> dict:
    """The item raced again under `cfg` (e.g. a larger budget): its stored branches are reused and the
    missing ones played from the state rebuilt from the recipe (checked against the fingerprint, on the
    simulator commit the item was built on). The result is what a fresh build with `cfg` would give."""
    from tfteval import simfixes
    from tfteval.branching import snapshot

    if item.get("fingerprint_hash") is None:
        raise ValueError(f"{item['id']} has no decision point ({item.get('dropped')}); nothing to race")
    if simfixes.sim_commit() != item["recipe"].get("sim_commit"):
        raise RuntimeError(f"{item['id']} was built on simulator {item['recipe'].get('sim_commit')}, "
                           f"this one is {simfixes.sim_commit()}")
    started = time.time()
    cache = defaultdict(dict)
    for b in item.get("branches", []):
        cache[b["cand"]][b["k"]] = {k: v for k, v in b.items() if k != "look"}
    seat = item["recipe"]["hero_seat"]
    holder = {}

    def play(name, k):  # the state is rebuilt only when a new branch is needed
        if "snap" not in holder:
            holder["snap"] = snapshot(bank.rebuild(item["recipe"], item["fingerprint_hash"]))
        return bank.play_branch(holder["snap"], seat, {c["name"]: c for c in item["candidates"]}[name], k,
                                item["continuation"])

    result = race([c["name"] for c in item["candidates"]], play, cfg, cache)
    new = {**item, "racing": dict(cfg), "dropped": None}
    new.pop("copied_from", None)
    _store_race(new, result)
    played = sum(1 for b in result["branches"] if b["k"] not in cache.get(b["cand"], {}))
    new["reraced"] = list(item.get("reraced", [])) + [{"from": item.get("racing"), "to": dict(cfg), "played": played,
                                                       "harness_commit": harness,
                                                       "seconds": round(time.time() - started, 2)}]
    return new


# --------------------------------------------------------------------------- labels

CLEAR_P = 0.85            # clear item: the posterior probability of the best candidate is at least this
VAR_PRIOR_DF = 8          # degrees of freedom of the stratum's pooled noise variance in an item's estimate
SD_FLOOR = bank.SD_FLOOR  # floor of the per-branch paired SD in the posterior (places)
MIN_FIT_ITEMS = 5         # a stratum with fewer usable items gets alpha = 0 and tau = TAU_FALLBACK
TAU_FALLBACK = 0.26       # v1: SD of a true two-candidate difference about 0.37 at 4-1 / 5-1 (tau = 0.37/sqrt 2)
POSTERIOR_DRAWS = 20000
PRIORS = ("candidate", "zero")  # prior mean: the stratum's candidate effects alpha, or 0 (exchangeable)


def places_of(item: dict) -> dict:
    out = {c["name"]: {} for c in item["candidates"]}
    for b in item.get("branches", []):
        out[b["cand"]][b["k"]] = b["place"]
    return out


def candidate_groups(item: dict) -> list[list[str]]:
    """The item's candidates as groups of one decision, in menu order (a group's first is its
    representative): the merges of the race's first look (`race.merged`, later -> earlier) and, as in v1,
    candidates whose hero actions are identical in every branch (bank.duplicate_groups)."""
    names = [c["name"] for c in item["candidates"]]
    merged = (item.get("race") or {}).get("merged") or {}
    groups = bank.duplicate_groups(item.get("branches", []), [c for c in names if c not in merged])
    for later, earlier in merged.items():
        next(g for g in groups if earlier in g).append(later)
    return [sorted(g, key=names.index) for g in groups]


def usable(items: list[dict]) -> list[dict]:
    """Items with a decision point and finished branches (dropped and all-duplicate ones left out)."""
    out = []
    for it in items:
        if it.get("dropped") or not it.get("branches"):
            continue
        places = places_of(it)
        if any(p is None for ks in places.values() for p in ks.values()) or any(not ks for ks in places.values()):
            continue
        if len(candidate_groups(it)) < 2:
            continue
        out.append(it)
    return out


def _item_data(item: dict) -> dict:
    """Per-item pieces of the posterior: duplicate groups, the reference, paired contrasts and their
    per-branch difference variance (pooled over the item's candidates)."""
    names = [c["name"] for c in item["candidates"]]
    places = places_of(item)
    groups = candidate_groups(item)
    reps = [g[0] for g in groups]
    group_of = {c: g for g in groups for c in g}
    ref = max(reps, key=lambda c: (len(places[c]), -names.index(c)))
    others = [c for c in reps if c != ref]
    shared = {c: sorted(set(places[c]) & set(places[ref])) for c in others}
    d = {c: float(np.mean([places[c][k] - places[ref][k] for k in shared[c]])) for c in others}
    ss, df = 0.0, 0
    for c in others:
        if len(shared[c]) > 1:
            diffs = np.array([places[c][k] - places[ref][k] for k in shared[c]], float)
            ss += float(((diffs - diffs.mean()) ** 2).sum())
            df += len(diffs) - 1
    return {"id": item["id"], "names": names, "places": places, "groups": groups, "reps": reps,
            "group_of": group_of, "ref": ref, "others": others, "shared": shared, "d": d,
            "var_d": ss / df if df else None, "df": df}


def _contrast_cov(data: dict, sigma2: float) -> np.ndarray:
    """Sampling covariance of the paired contrasts d_c (c in data["others"]) under the model."""
    others, shared = data["others"], data["shared"]
    m = len(others)
    v = np.zeros((m, m))
    for i, a in enumerate(others):
        for j, b in enumerate(others):
            na, nb = len(shared[a]), len(shared[b])
            both = len(set(shared[a]) & set(shared[b]))
            v[i, j] = sigma2 * ((1.0 / na if i == j else 0.0) + both / (na * nb))
    return v


def _centred(data: dict) -> dict:
    """The item's contrast estimates over the whole menu (a duplicate takes its group's), centred."""
    vals = {}
    for name in data["names"]:
        rep = data["group_of"][name][0]
        vals[name] = 0.0 if rep == data["ref"] else data["d"][rep]
    mean = float(np.mean(list(vals.values())))
    return {k: v - mean for k, v in vals.items()}


def _pair_var(data: dict, a: str, b: str, sigma2: float) -> float:
    """Sampling variance of contrast(a) - contrast(b) for menu names a, b in one item."""
    ra, rb = data["group_of"][a][0], data["group_of"][b][0]
    if ra == rb:
        return 0.0
    idx = {c: i for i, c in enumerate(data["others"])}
    v = _contrast_cov(data, sigma2)
    w = np.zeros(len(data["others"]))
    if ra in idx:
        w[idx[ra]] += 1
    if rb in idx:
        w[idx[rb]] -= 1
    return float(w @ v @ w)


def fit_stratum(items: list[dict], prior: str = "candidate", sd_floor: float = SD_FLOOR,
                df0: int = VAR_PRIOR_DF, min_items: int = MIN_FIT_ITEMS) -> dict:
    """The stratum-level prior: pooled noise, candidate effects alpha (with their sampling covariance) and
    tau^2 by the method of moments (see the module doc). `items` are usable items of one stratum."""
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}")
    datas = [_item_data(it) for it in items]
    names = list(datas[0]["names"]) if datas else []
    pooled_ss = sum(d["var_d"] * d["df"] for d in datas if d["var_d"] is not None)
    pooled_df = sum(d["df"] for d in datas if d["var_d"] is not None)
    pooled = pooled_ss / pooled_df if pooled_df else sd_floor ** 2
    sig = {}
    for d in datas:
        own = d["var_d"] if d["var_d"] is not None else pooled
        var_d = (d["df"] * own + df0 * pooled) / (d["df"] + df0)
        sig[d["id"]] = max(var_d, sd_floor ** 2) / 2.0  # sigma^2 = var of a paired difference / 2
    n = len(datas)
    fit = {"items": n, "prior": prior, "names": names, "pooled_sd_d": math.sqrt(pooled), "sigma2": sig,
           "alpha": {c: 0.0 for c in names}, "alpha_cov": [[0.0] * len(names) for _ in names]}
    if n < min_items:
        fit.update(tau2=TAU_FALLBACK ** 2, tau_source="fallback")
        return fit
    centred = [_centred(d) for d in datas]
    vecs = np.array([[cv[c] for c in names] for cv in centred])
    if prior == "candidate":
        alpha = vecs.mean(axis=0)
        fit["alpha"] = {c: float(a) for c, a in zip(names, alpha)}
        fit["alpha_cov"] = (np.cov(vecs, rowvar=False, ddof=1) / n).tolist()
    num, den = 0.0, 0.0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dd = np.array([cv[a] - cv[b] for cv in centred])
            w = np.array([_pair_var(d, a, b, sig[d["id"]]) for d in datas])
            if prior == "candidate":
                if n < 2:
                    continue
                est, weight = (float(dd.var(ddof=1)) - float(w.mean())) / 2.0, n - 1
            else:
                est, weight = (float((dd ** 2).mean()) - float(w.mean())) / 2.0, n
            num += weight * est
            den += weight
    fit.update(tau2=max(0.0, num / den) if den else TAU_FALLBACK ** 2, tau_source="moments")
    return fit


def posterior(item: dict, fit: dict, draws: int = POSTERIOR_DRAWS) -> dict:
    """P(each candidate is truly best) for one item under the stratum fit (Gaussian on the paired
    contrasts, Monte Carlo seeded by the item id); duplicates share their group's probability."""
    data = _item_data(item)
    names, others, ref, group_of = data["names"], data["others"], data["ref"], data["group_of"]
    sigma2 = fit["sigma2"].get(item["id"])
    if sigma2 is None:  # an item the fit did not see: the fit's pooled noise
        sigma2 = max(fit["pooled_sd_d"] ** 2, SD_FLOOR ** 2) / 2.0
    alpha = fit["alpha"]
    acov = np.array(fit["alpha_cov"], float)
    index = {c: i for i, c in enumerate(fit["names"])}

    def group_vec(rep: str) -> np.ndarray:
        v = np.zeros(len(fit["names"]))
        members = [c for c in group_of[rep] if c in index]
        for c in members:
            v[index[c]] = 1.0 / len(members)
        return v

    m = len(others)
    d = np.array([data["d"][c] for c in others])
    v = _contrast_cov(data, sigma2)
    tau2 = max(float(fit["tau2"]), 1e-10)
    lmat = np.array([group_vec(c) - group_vec(ref) for c in others]) if m else np.zeros((0, len(fit["names"])))
    m0 = lmat @ np.array([alpha.get(c, 0.0) for c in fit["names"]]) if m else np.zeros(0)
    p0 = tau2 * (np.eye(m) + np.ones((m, m))) + (lmat @ acov @ lmat.T if m else 0.0)
    gain = p0 @ np.linalg.inv(p0 + v)
    mean = m0 + gain @ (d - m0)
    cov = p0 - gain @ p0
    cov = (cov + cov.T) / 2.0
    seed = int.from_bytes(hashlib.sha256(f"bank2-posterior:{item['id']}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    sample = rng.multivariate_normal(mean, cov, size=draws, method="eigh") if m else np.zeros((draws, 0))
    full = np.concatenate([np.zeros((draws, 1)), sample], axis=1)  # column 0: the reference
    order = [ref] + others
    best_idx = full.argmin(axis=1)
    p_rep = {c: float(np.mean(best_idx == i)) for i, c in enumerate(order)}
    p = {c: p_rep[group_of[c][0]] for c in names}
    best = max(data["reps"], key=lambda c: (p_rep[c], -names.index(c)))
    return {"ref": ref, "P": p, "best": best, "best_group": list(group_of[best]), "pmax": p_rep[best],
            "post_mean": {c: float(x) for c, x in zip(order, [0.0] + list(mean))},
            "sigma_d": math.sqrt(2 * sigma2), "duplicates": data["groups"]}


def paired_mean(places: dict, a: str, b: str, ks=None) -> float | None:
    shared = sorted(set(places[a]) & set(places[b]) & (set(ks) if ks is not None else set(places[a])))
    if not shared:
        return None
    return float(np.mean([places[a][k] - places[b][k] for k in shared]))


def crossfit(places: dict, names: list[str], best: str, which: str = "oracle") -> dict:
    """Pick the best (`oracle`) or worst candidate by paired contrasts on the odd ks, score its paired
    difference to `best` on the even ks, and the other way round; the mean of the two halves."""
    ref = max(names, key=lambda c: (len(places[c]), -names.index(c)))
    picks, vals = [], []
    for parity in (1, 0):
        half = [k for k in places[ref] if k % 2 == parity]
        score = {}
        for c in names:
            s = 0.0 if c == ref else paired_mean(places, c, ref, half)
            if s is not None:
                score[c] = s
        if not score:
            continue
        key = (lambda c: (score[c], names.index(c))) if which == "oracle" else (lambda c: (-score[c], names.index(c)))
        pick = min(score, key=key)
        other = [k for k in places[pick] if k % 2 != parity]
        val = paired_mean(places, pick, best, other)
        if val is None:
            continue
        picks.append(pick)
        vals.append(val)
    return {"picks": picks, "regret": float(np.mean(vals)) if vals else None}


def label_items(items: list[dict], prior: str = "candidate", draws: int = POSTERIOR_DRAWS) -> tuple[dict, dict]:
    """Labels of usable items: (item id -> label, stratum -> fit). A label has the posterior (P, best,
    pmax, clear), regrets of every candidate against the posterior-best (paired, all shared ks), the
    candidates' mean places and branch counts, and the cross-fit oracle / worst. A duplicate or merged
    candidate has its group's representative's label (regret, mean, n); the cross-fit anchors pick
    among the representatives."""
    by_stratum = defaultdict(list)
    for it in items:
        by_stratum[it["stratum"]].append(it)
    labels, fits = {}, {}
    for stratum, part in by_stratum.items():
        fit = fit_stratum(part, prior)
        fits[stratum] = {k: v for k, v in fit.items() if k != "sigma2"}
        for it in part:
            post = posterior(it, fit, draws)
            places = places_of(it)
            names = [c["name"] for c in it["candidates"]]
            rep = {c: g[0] for g in post["duplicates"] for c in g}
            reps = [g[0] for g in post["duplicates"]]
            best = post["best"]
            regret = {c: paired_mean(places, rep[c], best) for c in names}
            labels[it["id"]] = {**post, "clear": post["pmax"] >= CLEAR_P, "regret": regret,
                                "mean": {c: float(np.mean(list(places[rep[c]].values()))) for c in names},
                                "n": {c: len(places[rep[c]]) for c in names},
                                "merged": dict((it.get("race") or {}).get("merged") or {}),
                                "xfit": {w: crossfit(places, reps, best, w) for w in ("oracle", "worst")}}
    return labels, fits


# --------------------------------------------------------------------------- scoring

def stratified_mean_ci(values, clusters, strata) -> dict:
    """The mean of the strata's means (each stratum weighs the same) with a 95% t interval clustered by
    source game (a game can have items in several strata): cluster-robust variance of the linearized
    estimator, G - 1 degrees of freedom. One stratum: bank.clustered_mean_ci. No interval when a stratum
    has items from one source game only (its variance cannot be estimated; the formula would give 0)."""
    values = np.asarray(values, float)
    n = len(values)
    if n == 0:
        return {"n": 0, "games": 0, "strata": 0, "mean": None, "ci95": None}
    count = defaultdict(int)
    total = defaultdict(float)
    games_in = defaultdict(set)
    for v, c, s in zip(values, clusters, strata):
        count[s] += 1
        total[s] += v
        games_in[s].add(c)
    means = {s: total[s] / count[s] for s in count}
    k = len(means)
    mean = float(np.mean(list(means.values())))
    contrib = defaultdict(float)
    for v, c, s in zip(values, clusters, strata):
        contrib[c] += (v - means[s]) / (k * count[s])
    g = len(contrib)
    if g < 2 or min(len(v) for v in games_in.values()) < 2:
        return {"n": n, "games": g, "strata": k, "mean": mean, "ci95": None}
    se = math.sqrt(sum(x * x for x in contrib.values()) * g / (g - 1))
    return {"n": n, "games": g, "strata": k, "mean": mean, "ci95": bank.t975(g - 1) * se}


def fmt_ci(entry: dict | None) -> str:
    """bank.fmt_ci, but an interval that cannot be estimated says so (one game, or a stratum with one)."""
    if entry and entry.get("mean") is not None and entry.get("ci95") is None:
        return f"{entry['mean']:+.2f} " + ("(1 game)" if entry.get("games", 0) < 2 else "(no CI)")
    return bank.fmt_ci(entry)


def score_rows(items: list[dict], labels: dict, choices: dict) -> list[dict]:
    """One row per item: the choice, its regret, P(choice is best), whether it is in the best group."""
    rows = []
    for it in items:
        lab = labels[it["id"]]
        choice = choices[it["id"]]["choice"]
        names = [c["name"] for c in it["candidates"]]
        valid = choice in names
        regret = lab["regret"][choice] if valid else max(lab["regret"].values())
        rows.append({"id": it["id"], "game": it["game"], "split": it["split"], "stratum": it["stratum"],
                     "point": it["point"], "clear": lab["clear"], "pmax": lab["pmax"], "choice": choice,
                     "valid": valid, "best": lab["best"], "regret": regret,
                     "p_choice": lab["P"][choice] if valid else 0.0,
                     "hit": float(valid and choice in lab["best_group"])})
    return rows


def crossfit_rows(items: list[dict], labels: dict, which: str) -> list[dict]:
    """Rows of the cross-fit anchor (`oracle` or `worst`): two picks per item (one per half), so its
    P(choice) and hit are the means over the two."""
    rows = []
    for it in items:
        lab = labels[it["id"]]
        xf = lab["xfit"][which]
        picks = xf["picks"] or [lab["best"]]
        rows.append({"id": it["id"], "game": it["game"], "split": it["split"], "stratum": it["stratum"],
                     "point": it["point"], "clear": lab["clear"], "pmax": lab["pmax"], "choice": "/".join(picks),
                     "valid": True, "best": lab["best"], "regret": xf["regret"] if xf["regret"] is not None else 0.0,
                     "p_choice": float(np.mean([lab["P"][p] for p in picks])),
                     "hit": float(np.mean([p in lab["best_group"] for p in picks]))})
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"items": 0}
    games = [r["game"] for r in rows]
    clear = [r for r in rows if r["clear"]]
    ceiling = float(sum(r["pmax"] for r in clear))
    expected = float(sum(r["p_choice"] for r in clear))
    out = {"items": len(rows), "games": len(set(games)), "regret": bank.clustered_mean_ci([r["regret"] for r in rows], games),
           "clear": len(clear), "ceiling": ceiling, "expected_hits": expected,
           "hits": float(sum(r["hit"] for r in clear)), "accuracy": expected / ceiling if ceiling else None,
           "invalid": sum(not r["valid"] for r in rows)}
    out["choices"] = {c: sum(r["choice"] == c for r in rows) for c in sorted({r["choice"] for r in rows})}
    return out


def summarize_equal(rows: list[dict]) -> dict:
    """All strata, each weighted the same (regret: stratified_mean_ci; accuracy: summed over strata)."""
    if not rows:
        return {"items": 0}
    out = summarize(rows)
    out["regret"] = stratified_mean_ci([r["regret"] for r in rows], [r["game"] for r in rows],
                                       [r["stratum"] for r in rows])
    return out


def scorecard(rows: list[dict], agent: str, bank_path: str | None = None, strata=None) -> dict:
    """Per stratum and equal-weighted over strata, for all source games and the dev / held-out splits."""
    strata = list(strata or STRATA)
    card = {"agent": agent, "bank": bank_path, "items": len(rows), "splits": {}}
    for sp in ("all", "dev", "heldout"):
        part = [r for r in rows if sp == "all" or r["split"] == sp]
        entry = {"equal": summarize_equal(part),
                 "strata": {s: summarize([r for r in part if r["stratum"] == s]) for s in strata
                            if any(r["stratum"] == s for r in part)}}
        card["splits"][sp] = entry
    card["rows"] = rows
    return card


def paired(card_a: dict, card_b: dict) -> dict:
    """Regret of A minus regret of B on the same items: per stratum (clustered by source game) and
    equal-weighted over strata."""
    b = {r["id"]: r for r in card_b["rows"]}
    shared = [r for r in card_a["rows"] if r["id"] in b]
    diffs = [r["regret"] - b[r["id"]]["regret"] for r in shared]
    out = {"equal": stratified_mean_ci(diffs, [r["game"] for r in shared], [r["stratum"] for r in shared]),
           "strata": {}}
    for s in sorted({r["stratum"] for r in shared}, key=lambda s: list(STRATA).index(s) if s in STRATA else 99):
        idx = [i for i, r in enumerate(shared) if r["stratum"] == s]
        out["strata"][s] = bank.clustered_mean_ci([diffs[i] for i in idx], [shared[i]["game"] for i in idx])
    return out


def table(card: dict, split: str = "all") -> str:
    entry = card["splits"][split]
    lines = [f"{card['agent']}: {card['items']} items ({split})",
             f"{'stratum':10} {'items':>5} {'games':>5}  {'regret (95% CI by game)':24} {'clear':>5} "
             f"{'ceiling':>7} {'exp.hits':>8} {'acc':>5}  choices"]
    rows = list(entry["strata"].items()) + [("equal", entry["equal"])]
    for name, s in rows:
        if not s.get("items"):
            continue
        acc = f"{s['accuracy']:>5.0%}" if s.get("accuracy") is not None else f"{'-':>5}"
        lines.append(f"{name:10} {s['items']:>5} {s['games']:>5}  {fmt_ci(s['regret']):24} {s['clear']:>5} "
                     f"{s['ceiling']:>7.2f} {s['expected_hits']:>8.2f} {acc}  "
                     + " ".join(f"{c}:{n}" for c, n in s["choices"].items()))
    return "\n".join(lines)


def bank_summary(items: list[dict]) -> dict:
    """Per stratum: items built, fired (and at which rounds), dropped by reason, usable; branch time."""
    out = {"built": len(items), "strata": {}}
    for s in STRATA:
        part = [it for it in items if it.get("stratum") == s]
        if not part:
            continue
        fired = [it for it in part if it.get("point")]
        dropped = defaultdict(int)
        for it in part:
            if it.get("dropped"):
                dropped[it["dropped"].split(":")[0]] += 1
        rounds = defaultdict(int)
        for it in fired:
            rounds[it["point"]] += 1
        out["strata"][s] = {"games": len(part), "fired": len(fired), "rounds": dict(sorted(rounds.items())),
                            "dropped": dict(dropped), "usable": len(usable(part)),
                            "branches": sum(len(it.get("branches", [])) for it in part if not it.get("copied_from")),
                            "copied": sum(1 for it in part if it.get("copied_from"))}
    secs = [b["seconds"] for it in items if not it.get("copied_from") for b in it.get("branches", [])]
    out["usable"] = sum(v["usable"] for v in out["strata"].values())
    out["games"] = len({it["game"] for it in items})
    out["branch_seconds_mean"] = float(np.mean(secs)) if secs else None
    return out


# --------------------------------------------------------------------------- rule check

RULE_FEATURES = ("hp", "gold", "pairs", "streak", "level", "losses_to_death")
RULE_FOLDS = 5          # cross-fitting folds over source games
MIN_LEAF = 3            # items per leaf of the depth-2 tree
RULE_HEADROOM = 0.1     # a stratum whose best rule is within this of the cross-fit oracle is a rule stratum


def item_features(item: dict) -> dict:
    return trigger_features(item["public_state"])


def game_folds(games, k: int = RULE_FOLDS) -> dict:
    """game -> fold, by a hash of the game (whole games per fold); k is capped by the number of games."""
    games = sorted(set(games), key=lambda g: hashlib.sha256(f"bank2-fold:{g}".encode()).hexdigest())
    k = max(1, min(k, len(games)))
    return {g: i % k for i, g in enumerate(games)}


def _rule_rows(items: list[dict], labels: dict) -> list[dict]:
    return [{"id": it["id"], "game": it["game"], "x": item_features(it), "R": dict(labels[it["id"]]["regret"]),
             "P": dict(labels[it["id"]]["P"]), "clear": labels[it["id"]]["clear"], "pmax": labels[it["id"]]["pmax"],
             "best_group": labels[it["id"]]["best_group"]} for it in items]


def _thresholds(values) -> list[float]:
    vals = sorted(set(values))
    return [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]


def fit_fixed(rows: list[dict], names: list[str]) -> dict:
    total = {c: sum(r["R"][c] for r in rows) for c in names}
    best = min(names, key=lambda c: (total[c], names.index(c)))
    return {"kind": "fixed", "choice": best}


def fit_threshold(rows: list[dict], names: list[str], feature: str) -> dict:
    """The rule `a if x[feature] <= t else b` with the lowest total regret on `rows`. The family contains
    the fixed rules (t None: always a); ties go to the fixed rule, then the lower t, then menu order."""
    fixed = fit_fixed(rows, names)["choice"]
    best = {"kind": "threshold", "feature": feature, "t": None, "a": fixed, "b": fixed}
    best_cost = sum(r["R"][fixed] for r in rows)
    for t in _thresholds([r["x"][feature] for r in rows]):
        low = [r for r in rows if r["x"][feature] <= t]
        high = [r for r in rows if r["x"][feature] > t]
        a = min(names, key=lambda c: (sum(r["R"][c] for r in low), names.index(c)))
        b = min(names, key=lambda c: (sum(r["R"][c] for r in high), names.index(c)))
        cost = sum(r["R"][a] for r in low) + sum(r["R"][b] for r in high)
        if cost < best_cost - 1e-12:
            best, best_cost = {"kind": "threshold", "feature": feature, "t": t, "a": a, "b": b}, cost
    return best


def fit_tree(rows: list[dict], names: list[str], features=RULE_FEATURES, depth: int = 2,
             min_leaf: int = MIN_LEAF) -> dict:
    """A greedy regret-minimizing tree: each leaf picks the candidate with the lowest total regret of its
    items, each split (feature <= t) the one that lowers the total most with at least min_leaf items a side."""
    leaf = min(names, key=lambda c: (sum(r["R"][c] for r in rows), names.index(c)))
    cost = sum(r["R"][leaf] for r in rows)
    node = {"leaf": leaf}
    if depth <= 0 or len(rows) < 2 * min_leaf:
        return node
    best = None
    for f in features:
        for t in _thresholds([r["x"][f] for r in rows]):
            low = [r for r in rows if r["x"][f] <= t]
            high = [r for r in rows if r["x"][f] > t]
            if len(low) < min_leaf or len(high) < min_leaf:
                continue
            c = min(sum(r["R"][n] for r in low) for n in names) + min(sum(r["R"][n] for r in high) for n in names)
            if best is None or c < best[0] - 1e-12:
                best = (c, f, t, low, high)
    if best is None or best[0] >= cost - 1e-12:
        return node
    _, f, t, low, high = best
    return {"feature": f, "t": t, "le": fit_tree(low, names, features, depth - 1, min_leaf),
            "gt": fit_tree(high, names, features, depth - 1, min_leaf)}


def apply_rule(rule: dict, x: dict) -> str:
    if rule.get("kind") == "fixed":
        return rule["choice"]
    if rule.get("kind") == "threshold":
        return rule["a"] if rule["t"] is None or x[rule["feature"]] <= rule["t"] else rule["b"]
    while "leaf" not in rule:
        rule = rule["le"] if x[rule["feature"]] <= rule["t"] else rule["gt"]
    return rule["leaf"]


def crossfit_rule(rows: list[dict], fit, folds: dict) -> list[str]:
    """Each row's choice by a rule fitted on the other folds' games (fit(train_rows) -> rule)."""
    choices = [None] * len(rows)
    for fold in sorted(set(folds.values())):
        train = [r for r in rows if folds[r["game"]] != fold]
        if not train:
            train = rows  # one game: nothing to hold out
        rule = fit(train)
        for i, r in enumerate(rows):
            if folds[r["game"]] == fold:
                choices[i] = apply_rule(rule, r["x"])
    return choices


def rule_check(items: list[dict], labels: dict, folds: int = RULE_FOLDS) -> dict:
    """Per stratum: regret of every fixed candidate, of the best fixed candidate chosen by cross-fitting,
    of every single-feature threshold rule (cross-fitted over source games), of a depth-2 tree (cross-
    validated by game), against the cross-fit oracle; headroom = best rule regret - oracle regret."""
    by_stratum = defaultdict(list)
    for it in items:
        by_stratum[it["stratum"]].append(it)
    out = {}
    for stratum, part in by_stratum.items():
        names = [c["name"] for c in part[0]["candidates"]]
        rows = _rule_rows(part, labels)
        games = [r["game"] for r in rows]
        fold_of = game_folds(games, folds)

        def stat(choices):
            regrets = [r["R"][c] for r, c in zip(rows, choices)]
            clear = [(r, c) for r, c in zip(rows, choices) if r["clear"]]
            ceiling = sum(r["pmax"] for r, _ in clear)
            exp = sum(r["P"][c] for r, c in clear)
            return {"regret": bank.clustered_mean_ci(regrets, games),
                    "accuracy": exp / ceiling if ceiling else None, "expected_hits": exp,
                    "choices": {n: sum(c == n for c in choices) for n in names if any(c == n for c in choices)}}

        entry = {"items": len(rows), "games": len(set(games)), "folds": len(set(fold_of.values())),
                 "clear": sum(r["clear"] for r in rows), "ceiling": sum(r["pmax"] for r in rows if r["clear"])}
        xo = [labels[it["id"]]["xfit"]["oracle"]["regret"] for it in part]
        xw = [labels[it["id"]]["xfit"]["worst"]["regret"] for it in part]
        entry["xfit_oracle"] = bank.clustered_mean_ci([v if v is not None else 0.0 for v in xo], games)
        entry["xfit_worst"] = bank.clustered_mean_ci([v if v is not None else 0.0 for v in xw], games)
        entry["fixed"] = {c: stat([c] * len(rows)) for c in names}
        entry["best_fixed"] = stat(crossfit_rule(rows, lambda tr: fit_fixed(tr, names), fold_of))
        entry["threshold"] = {}
        for f in RULE_FEATURES:
            res = stat(crossfit_rule(rows, lambda tr, f=f: fit_threshold(tr, names, f), fold_of))
            res["rule_on_all"] = fit_threshold(rows, names, f)
            entry["threshold"][f] = res
        entry["tree"] = stat(crossfit_rule(rows, lambda tr: fit_tree(tr, names), fold_of))
        entry["tree"]["rule_on_all"] = fit_tree(rows, names)
        contenders = {"best_fixed": entry["best_fixed"]["regret"]["mean"], "tree": entry["tree"]["regret"]["mean"],
                      **{f"threshold:{f}": entry["threshold"][f]["regret"]["mean"] for f in RULE_FEATURES}}
        best_rule = min(contenders, key=lambda k: contenders[k])
        entry["best_rule"] = best_rule
        entry["best_rule_regret"] = contenders[best_rule]
        entry["headroom"] = contenders[best_rule] - entry["xfit_oracle"]["mean"]
        entry["rule_stratum"] = entry["headroom"] < RULE_HEADROOM
        out[stratum] = entry
    return out


# --------------------------------------------------------------------------- held-out check

BRANCH_FOLDS = 4        # folds over an item's branch keys for the held-out oracle
VPI_DRAWS = 20000


def _centred_places(item: dict) -> dict:
    """{"ks": the keys every distinct candidate has, "cen": candidate -> k -> its place minus the mean
    place of the item's distinct candidates on that k} (duplicates take their group's places)."""
    places = places_of(item)
    groups = candidate_groups(item)
    reps = [g[0] for g in groups]
    group_of = {c: g for g in groups for c in g}
    ks = sorted(set.intersection(*(set(places[r]) for r in reps)))
    mean_k = {k: float(np.mean([places[r][k] for r in reps])) for k in ks}
    names = [c["name"] for c in item["candidates"]]
    return {"ks": ks, "reps": reps,
            "cen": {c: {k: places[group_of[c][0]][k] - mean_k[k] for k in ks} for c in names}}


def _mean_on(cen: dict, c: str, keys) -> float:
    return float(np.mean([cen[c][k] for k in keys]))


def heldout_oracle(data: dict, folds: int = BRANCH_FOLDS) -> dict:
    """The oracle that picks the best candidate on the item's other branch folds and is scored on the
    held-out one, averaged over folds (every branch is scored once, never by the pick it informed)."""
    ks, cen = data["ks"], data["cen"]
    fold = {k: i % folds for i, k in enumerate(ks)}
    picks, scores, weights = [], [], []
    for f in sorted(set(fold.values())):
        held = [k for k in ks if fold[k] == f]
        train = [k for k in ks if fold[k] != f]
        if not held or not train:
            continue
        pick = min(data["reps"], key=lambda c: (_mean_on(cen, c, train), data["reps"].index(c)))
        picks.append(pick)
        scores.append(_mean_on(cen, pick, held))
        weights.append(len(held))
    return {"picks": picks, "score": float(np.average(scores, weights=weights))}


def vpi(fit: dict, names: list[str], draws: int = VPI_DRAWS, seed: int = 0) -> float:
    """Value of perfect information under the stratum prior: E[min_c (alpha_c + u_c)] - min_c alpha_c with
    u_c ~ N(0, tau^2), in places; how much an agent that reads every state perfectly gains on average over
    the best fixed candidate (an upper bound for state reading on this menu)."""
    alpha = np.array([fit["alpha"].get(c, 0.0) for c in names])
    tau = math.sqrt(fit["tau2"])
    u = np.random.default_rng(seed).normal(0.0, tau, size=(draws, len(names)))
    return float(alpha.min() - (alpha + u).min(axis=1).mean())


def heldout_check(items: list[dict], fits: dict, choices: dict | None = None, folds: int = RULE_FOLDS,
                  branch_folds: int = BRANCH_FOLDS) -> dict:
    """Per stratum, every policy scored on the same branches by the place of its choice relative to the
    item's mean over candidates (lower is better; a uniformly random choice scores 0): the held-out oracle
    (heldout_oracle), every fixed candidate, the best fixed candidate, single-feature threshold rules and a
    depth-2 tree (each fitted on the other folds' source games, so nothing is scored on branches it was
    chosen with), and the agents in `choices` (agent -> item id -> choice). Differences to the oracle are
    paired per item (95% CI clustered by game). VPI (`vpi`) is the ceiling for state reading."""
    choices = choices or {}
    by_stratum = defaultdict(list)
    for it in items:
        by_stratum[it["stratum"]].append(it)
    out = {}
    for stratum, part in by_stratum.items():
        names = [c["name"] for c in part[0]["candidates"]]
        datas = [_centred_places(it) for it in part]
        games = [it["game"] for it in part]
        rows = [{"id": it["id"], "game": it["game"], "x": item_features(it),
                 "R": {c: _mean_on(d["cen"], c, d["ks"]) for c in names}} for it, d in zip(part, datas)]
        oracle = [heldout_oracle(d, branch_folds)["score"] for d in datas]
        fold_of = game_folds(games, folds)

        def stat(per_item):
            return {"score": bank.clustered_mean_ci(per_item, games),
                    "vs_oracle": bank.clustered_mean_ci([a - b for a, b in zip(per_item, oracle)], games),
                    "values": [float(v) for v in per_item]}

        def of(choice_list):
            return [r["R"][c] for r, c in zip(rows, choice_list)]

        entry = {"ids": [it["id"] for it in part], "game_of": games, "items": len(part), "games": len(set(games)), "branches_per_cand": float(np.mean([len(d["ks"]) for d in datas])),
                 "tau": math.sqrt(fits[stratum]["tau2"]), "alpha": fits[stratum]["alpha"],
                 "vpi": vpi(fits[stratum], names), "oracle": stat(oracle)}
        entry["fixed"] = {c: stat(of([c] * len(rows))) for c in names}
        entry["best_fixed"] = stat(of(crossfit_rule(rows, lambda tr: fit_fixed(tr, names), fold_of)))
        entry["threshold"] = {f: stat(of(crossfit_rule(rows, lambda tr, f=f: fit_threshold(tr, names, f), fold_of)))
                              for f in RULE_FEATURES}
        entry["tree"] = stat(of(crossfit_rule(rows, lambda tr: fit_tree(tr, names), fold_of)))
        entry["agents"] = {}
        for agent, picks in choices.items():
            if all(it["id"] in picks for it in part):
                entry["agents"][agent] = stat(of([picks[it["id"]] for it in part]))
        rules = {"best_fixed": entry["best_fixed"], "tree": entry["tree"],
                 **{f"threshold:{f}": v for f, v in entry["threshold"].items()}}
        best = min(rules, key=lambda k: rules[k]["score"]["mean"])
        entry["best_rule"], entry["best_rule_vs_oracle"] = best, rules[best]["vs_oracle"]
        out[stratum] = entry
    return out

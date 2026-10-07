"""Rubric v1 (tfteval/rubric.py) on hand-built rows: item logic, threshold edges, the rules profile's max
level, hard gates, aggregation and the within-state estimator. No games."""

import math

import numpy as np
import pytest

pytest.importorskip("Simulator")  # the rules profiles (max level, xp table) come from the simulator

from tfteval import rubric  # noqa: E402
from tfteval.rubric import (banked_without_leveling, died_with_gold, items_on_non_carries, max_level,  # noqa: E402
                            misplaced_items, plan_gates, rates, score_result, score_seat, within_state_association)

COMPS = {"mage": ["ahri", "annie", "lulu"], "divine": ["irelia", "wukong"]}


def row(rnd, gold=10, level=5, xp=0, end_gold=None, end_level=None, end_xp=None, **extra):
    out = {"round": rnd, "hp": 50, "gold": gold, "level": level, "xp": xp, "streak": 0, "actions": {}, "illegal": 0,
           "fallbacks": 0, "end": {"gold": gold if end_gold is None else end_gold,
                                   "level": level if end_level is None else end_level,
                                   "xp": xp if end_xp is None else end_xp}}
    out.update(extra)
    return out


def test_profile_max_level():
    assert max_level("set4") == 9 and max_level("set18") == 10


def test_components_match_the_simulator():
    from Simulator.battle.item_stats import basic_items, items

    assert rubric.COMPONENTS == frozenset(basic_items)
    completed = {i for i in items if rubric.is_completed(i)}
    assert "infinity_edge" in completed and "force_of_nature" in completed
    assert not completed & {"spatula", "bf_sword", "champion_duplicator", "reforger", "magnetic_remover"}


def test_died_with_gold_edge():
    assert died_with_gold([row(20, end_gold=21)], eliminated=True)["flag"]
    assert not died_with_gold([row(20, end_gold=20)], eliminated=True)["flag"]  # more than 20
    assert not died_with_gold([row(20, end_gold=80)], eliminated=False)["flag"]  # the winner is not eliminated
    assert died_with_gold([row(20, end_gold=80)], eliminated=False)["gold"] is None
    assert died_with_gold([row(20, end_gold=5)], eliminated=True, gold=4)["flag"]  # threshold is a parameter
    # the gold that counts is at the end of the last planning phase, not at its start
    assert not died_with_gold([row(20, gold=60, end_gold=3)], eliminated=True)["flag"]


def test_banked_without_leveling_runs_and_edges():
    rows = [row(10, end_gold=51, level=7), row(11, end_gold=51, level=7), row(12, end_gold=30, level=7),
            row(13, end_gold=60, level=7), row(14, end_gold=50, level=7), row(15, end_gold=70, level=7)]
    out = banked_without_leveling(rows, "set4")
    assert out["flag"] and out["rounds"] == 2 and out["longest"] == 2  # 50 is not "more than 50"
    assert not banked_without_leveling(rows[2:], "set4")["flag"]  # only runs of one round left
    assert banked_without_leveling(rows[2:], "set4", gold=49)["flag"]  # 13, 14, 15 now one run of three
    assert banked_without_leveling(rows[2:], "set4", gold=49)["rounds"] == 3
    assert not banked_without_leveling(rows, "set4", rounds=3)["flag"]
    # consecutive means consecutive planning phases
    assert not banked_without_leveling([row(10, end_gold=80), row(12, end_gold=80)], "set4")["flag"]


def test_banked_without_leveling_reads_the_profile_max_level():
    rows = [row(30, end_gold=90, level=9), row(31, end_gold=90, level=9)]
    assert not banked_without_leveling(rows, "set4")["flag"]  # level 9 is the top in set4
    assert banked_without_leveling(rows, "set18")["flag"]  # set18 goes to 10
    assert banked_without_leveling(rows, "set18")["max_level"] == 10
    # the level that counts is the one at the end of the planning phase
    levelled = [row(30, end_gold=60, level=8, end_level=9), row(31, end_gold=60, level=9)]
    assert not banked_without_leveling(levelled, "set4")["flag"]


def test_items_on_non_carries_plan_seat():
    board = [["ahri", 2, 4, ["infinity_edge"]], ["zed", 1, 2, ["deathblade"]], ["vayne", 1, 1, ["bf_sword"]],
             ["garen", 1, 1, ["warmogs_armor", "chain_vest"]], ["lulu", 1, 2, [], "mage"]]
    r = row(16, comp="mage", plan={"carry": "zed", "level_to": 7}, board=board)
    assert misplaced_items(r, COMPS) == 1  # garen's warmogs; zed is the carry, vayne's sword is a component
    out = items_on_non_carries([row(14, **{k: r[k] for k in ("comp", "plan", "board")}), r], COMPS)
    assert out == {"flag": True, "rounds": 1, "checked": 1, "items": 1}  # 14 is before 4-1 (15)
    assert items_on_non_carries([r], COMPS, start="5-1")["checked"] == 0
    clean = row(17, comp="mage", plan={"carry": None}, board=board[:1] + board[4:])
    assert not items_on_non_carries([clean], COMPS)["flag"]


def test_items_on_non_carries_without_a_plan_uses_the_top_two():
    board = [["a", 2, 4, ["infinity_edge"]], ["b", 1, 3, ["deathblade"]], ["c", 3, 1, ["bloodthirster"]],
             ["d", 1, 2, ["zephyr", "spatula"]], ["e", 1, 5]]
    r = row(20, comp="mage", board=board)  # a rule seat: its comp is recorded, but it has no plan
    # a (4x2=8) and e (5x1=5) are the two highest cost x stars; b, c and d hold one completed item each
    assert misplaced_items(r, COMPS) == 3
    assert misplaced_items(r, COMPS, fallback=3) == 2
    # a plan seat before it has a comp falls back too
    assert misplaced_items(row(20, plan={"carry": None}, board=board), COMPS) == 3
    # ties on cost x stars go to the higher cost: b (3x1) is kept over c (1x3)
    assert misplaced_items(row(20, board=board[1:4]), COMPS) == 1


def test_plan_gates_level_and_roll_floor():
    # level 6 -> 7 needs 36 xp in set4; 10 held, 26 missing = 7 buys = 28 gold
    short = row(12, end_gold=30, level=6, end_xp=10, knobs={"level_to": 7, "roll_floor": 10}, actions={"buy": 15})
    poor = row(13, end_gold=27, level=6, end_xp=10, knobs={"level_to": 7, "roll_floor": 10})
    rolled = row(14, end_gold=40, level=7, knobs={"level_to": 7, "roll_floor": 10}, roll_low=8)
    at_floor = row(15, end_gold=40, level=7, knobs={"level_to": 7, "roll_floor": 10}, roll_low=10)
    idle = row(16, end_gold=60, level=7, end_xp=0, knobs={"level_to": 8, "roll_floor": 999}, actions={"pass": 3})
    capped = row(17, end_gold=90, level=9, knobs={"level_to": 10, "roll_floor": 999})  # set4 tops at 9
    out = plan_gates([short, poor, rolled, at_floor, idle, capped], "set4")
    assert out == {"plan_rounds": 6, "level_short": 2, "level_short_idle": 1, "roll_below_floor": 1}
    assert plan_gates([capped], "set18")["level_short"] == 1  # set18: 10 is reachable
    assert plan_gates([row(12, end_gold=99)], "set4")["plan_rounds"] == 0  # not a plan seat


def test_score_seat_from_round_and_score_result():
    rows = [row(r, end_gold=60, level=6) for r in range(8, 12)] + [row(12, end_gold=25, level=6)]
    whole = score_seat(rows, rules="set4", place=5, eliminated_round=12)
    assert whole["flags"] == {"died_with_gold": True, "banked_without_leveling": True, "items_on_non_carries": False}
    late = score_seat(rows, rules="set4", place=5, eliminated_round=12, from_round=11)
    assert late["rounds"] == 2 and not late["flags"]["banked_without_leveling"]
    assert score_seat(rows, rules="set4", eliminated_round=12, thresholds={"died_gold": 30})["flags"][
        "died_with_gold"] is False
    result = {"seed": 1, "rules": "set4", "lobby": {"player_0": "hero", "player_1": "rule"},
              "placements": {"player_0": 1, "player_1": 2}, "eliminated": {"player_1": 12},
              "records": {"player_0": rows[:3], "player_1": rows}}
    scores = score_result(result, comps=COMPS)
    assert [(s["seat"], s["policy"], s["place"]) for s in scores] == [("player_0", "hero", 1), ("player_1", "rule", 2)]
    assert not scores[0]["flags"]["died_with_gold"] and scores[1]["flags"]["died_with_gold"]
    assert [s["seat"] for s in score_result(result, seats=["player_1"], comps=COMPS)] == ["player_1"]


def test_rates_by_policy():
    def score(policy, died, fallbacks=0):
        return {"policy": policy, "flags": {"died_with_gold": died, "banked_without_leveling": False,
                                            "items_on_non_carries": False},
                "detail": {"banked_rounds": 0, "items_rounds": 0, "items_misplaced": 0},
                "gates": {"fallbacks": fallbacks, "illegal": 0, "level_short": 0, "level_short_idle": 0,
                          "roll_below_floor": 0}}

    out = rates([score("a", True), score("a", False, 2), score("a", False), score("b", True)])
    assert out["a"]["n"] == 3 and out["a"]["items"]["died_with_gold"]["k"] == 1
    assert math.isclose(out["a"]["items"]["died_with_gold"]["rate"], 1 / 3)
    w = out["a"]["items"]["died_with_gold"]
    assert 0 < w["low"] < 1 / 3 < w["high"] < 1
    assert out["a"]["gates"]["fallbacks"] == {"total": 2, "per_seat": 2 / 3}
    assert out["a"]["gates"]["seats_with_any"] == 1
    assert out["b"]["items"]["died_with_gold"]["rate"] == 1.0


def test_wilson_edges():
    assert rubric.wilson(0, 0)["rate"] is None
    w = rubric.wilson(0, 10)
    assert w["rate"] == 0 and w["low"] == 0 and 0.2 < w["high"] < 0.35


def test_within_state_association_recovers_a_known_difference():
    # 12 source games x 2 states x 4 branches; the flagged branches place exactly 1.5 worse than the others
    # of their state, on top of large state effects; one state per game never fires (no contrast there)
    branches, rng = [], np.random.default_rng(1)
    for game in range(12):
        for state in range(2):
            base = rng.uniform(1, 7)
            for k in range(4):
                flag = state == 0 and k % 2 == 0
                branches.append({"state": (game, state), "cluster": game, "place": base + 1.5 * flag, "flag": flag})
    out = within_state_association(branches)
    assert out["states"] == 24 and out["states_mixed"] == 12 and out["flagged"] == 24
    assert math.isclose(out["fe_diff"], 1.5) and math.isclose(out["ci95"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["mean_state_diff"]["mean"], 1.5)
    assert "not a causal effect" in out["note"]
    # with noise the interval covers the truth, clustered by game
    noisy = [{**b, "place": b["place"] + rng.normal(0, 1)} for b in branches]
    out = within_state_association(noisy)
    assert out["clusters"] == 12 and out["low"] < 1.5 < out["high"]
    # pooling across states without demeaning would mix the state effects in; demeaning removes them
    shifted = [{**b, "place": b["place"] + (100 if b["flag"] and b["state"][1] == 1 else 0)} for b in branches]
    assert math.isclose(within_state_association(shifted)["fe_diff"], 1.5)


def test_within_state_association_without_contrast():
    branches = [{"state": s, "cluster": s, "place": 3, "flag": s % 2 == 0} for s in range(4)]
    out = within_state_association(branches)
    assert out["fe_diff"] is None and out["states_mixed"] == 0 and out["states"] == 4

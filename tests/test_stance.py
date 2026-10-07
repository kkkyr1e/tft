"""Rule stance planner: each stance's trigger, priorities, ablations, and the information boundary.

Version 1 (seat kind stance1) keeps its own tests unchanged, run on v1(); version 2 (seat kind stance)
has its own section below, run on v2() with a fixed stand-in win-probability model so that the triggers
do not move when the fitted model is refitted (tests/test_winprob.py tests the fitted model)."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("Simulator")

from Simulator.battle.champion import champion  # noqa: E402
from Simulator.battle.stats import COST  # noqa: E402
from Simulator.game import pool  # noqa: E402
from Simulator.game.player import Player  # noqa: E402
from Simulator.generators.default_agent import Default_Agent  # noqa: E402
from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS, TEAM_COMPS  # noqa: E402

from tfteval import make_policy, public, stages, winprob  # noqa: E402
from tfteval import stance as st  # noqa: E402
from tfteval.planner import ParamPlanner, PlanPolicy, compile_knobs, describe, level_costs  # noqa: E402

COMPS = dict(zip(TEAM_COMP_TRAITS, TEAM_COMPS))
EVEN = ["ahri*1", "annie*1", "lulu*1", "veigar*1", "thresh*1", "jax*1"]  # a middling 6-unit board
STRONG = ["ahri*2[bf_sword,chain_vest]", "annie*2", "lulu*2", "veigar*2", "thresh*1", "jax*1"]
WEAK = ["vayne*1", "fiora*1", "garen*1"]


def own(tag: str) -> dict:
    u = st.parse_tag(tag)
    u["cost"] = COST[u["name"]]
    return u


def opp(seat: str, board=EVEN, hp=80, level=6, streak=0, interest=3) -> dict:
    return {"seat": seat, "hp": hp, "level": level, "streak": streak, "interest": interest, "board": list(board)}


def make_state(idx=12, hp=80, gold=50, level=6, xp=0, streak=0, board=EVEN, bench=(), opponents=None,
               next_from=None, comp=None, rules="set4") -> dict:
    """A synthetic describe() state (test_synthetic_state_has_the_describe_keys checks the keys)."""
    sched = stages.schedule(idx)
    opponents = [opp(f"player_{i}") for i in range(1, 8)] if opponents is None else opponents
    state = {
        "round": idx, "rules": rules, "stage": sched["stage"], "pve": sched["pve"],
        "next": {"carousel": sched["to_carousel"], "pve": sched["to_pve"], "stage": sched["to_stage"]},
        "hp": hp, "gold": gold, "level": level, "xp": xp, "xp_needed": level_costs(rules)[level], "streak": streak,
        "dmg_per_loss": stages.damage_per_loss(idx, rules), "losses_to_death": stages.losses_to_death(hp, idx, rules),
        "board": [own(t) for t in board], "bench": [own(t) for t in bench], "item_bench": [], "shop": [],
        "active_traits": {},
        "hp_rank": 1 + sum(o["hp"] > hp for o in opponents), "alive": len(opponents) + 1,
        "opponents": opponents,
        "next_from": [o["seat"] for o in opponents[:3]] if next_from is None else next_from,
    }
    seen = st.seen_copies(state)
    state["contested_by_comp"] = {t: sum(seen[u] for u in units) for t, units in COMPS.items()}
    if comp:
        state["contested"] = {u: seen[u] for u in COMPS[comp]}
    return state


def run(planner, state, comp=None):
    return planner.plan(state, COMPS, comp)


def v1(enabled=(), **kw):
    """The version-1 planner, as seat kind stance1 builds it."""
    spec = st.STANCE_KINDS["stance1"]
    return st.StancePlanner("stance1", disabled=spec["off"], enabled=tuple(spec["on"]) + tuple(enabled),
                            opts=spec["opts"], **kw)


# p_win from the board score alone: EVEN vs EVEN 0.5 (close), STRONG vs EVEN 0.75 (stronger), WEAK vs EVEN 0.31
STAND_IN = winprob.Model({"terms": ["score"], "coef": {"score": 0.05}, "side": 0.0})


def v2(kind="stance", **kw):
    """A version-2 planner (seat kind `kind`) with the stand-in model."""
    spec = st.STANCE_KINDS[kind]
    planner = st.StancePlanner(kind, disabled=spec.get("off", ()), enabled=spec.get("on", ()), opts=spec.get("opts"),
                               **kw)
    planner.model = STAND_IN
    return planner


# --------------------------------------------------------------------------- features

def test_board_score_is_the_rule_bots_rank_comp_plus_items():
    p = Player(pool.pool(), 0)
    p.level = p.max_units = 6
    tags = ["ahri*2", "annie*1", "lulu*1", "veigar*1", "nami*1", "twistedfate*1"]
    for i, tag in enumerate(tags):
        name, _, stars = tag.partition("*")
        assert p.add_to_bench(champion(name, stars=int(stars)))
        assert p.move_bench_to_board(next(j for j, c in enumerate(p.bench) if c), i, 0)
    units = [st.parse_tag(public.unit_tag(u)) for u in public.board_units(p)]
    assert st.board_score(units) == Default_Agent().rank_comp(p.board) > 0
    units[0]["items"] = ["bf_sword", "chain_vest"]
    assert st.board_score(units) == Default_Agent().rank_comp(p.board) + 2 * st.ITEM_SCORE


def test_parse_tag_reads_public_unit_tags():
    assert st.parse_tag("ahri*2(mage)[bf_sword,chain_vest]") == \
        {"name": "ahri", "star": 2, "chosen": "mage", "items": ["bf_sword", "chain_vest"]}
    assert st.parse_tag("twistedfate*1") == {"name": "twistedfate", "star": 1}
    with pytest.raises(ValueError):
        st.parse_tag("ahri")


def test_features_compare_with_the_candidates_only():
    opponents = [opp("player_1", WEAK), opp("player_2", WEAK), opp("player_3", STRONG, level=8),
                 opp("player_4", STRONG, level=7, interest=5)]
    f = st.features(make_state(streak=-2, opponents=opponents, next_from=["player_1", "player_2"]), COMPS, None)
    assert f["ratio"] > 1 and f["score_rank"] == 3  # stronger than the two candidates, not than the field
    assert f["opp_near8"] == 2 and f["opp_level_max"] == 8 and f["opp_rich"] == 1
    assert f["streak_gold_loss"] == 1 and f["streak_gold_win"] == 1  # a third loss pays 1; a win ends it
    assert st.features(make_state(streak=4), COMPS, None)["streak_gold_win"] == 1 + 3


def test_projection_and_xp_cost():
    assert st.projected_gold(50, 2) == 50 + 10 + 10 and st.projected_gold(9, 1) == 9 + 5
    assert st.xp_gold(7, 0, 8) == 56 and st.xp_gold(6, 0, 8, rounds=4) == 84 and st.xp_gold(7, 55, 8) == 4


# --------------------------------------------------------------------------- stances

def test_stage_one_and_a_quiet_round_are_standard_econ():
    planner = v1()
    assert run(planner, make_state(idx=1, level=2))["stance"] == "standard"
    state = make_state(idx=10, level=5, gold=58)
    plan = run(v1(), state)
    assert plan["stance"] == "standard" and plan["field_comp"] is True
    base = ParamPlanner().plan(state, COMPS, None)
    assert {k: plan[k] for k in base} == base  # the rule bot's own economy
    assert not {"level_by", "spend", "survival"} & set(plan)
    assert plan["features"]["ratio"] == 1.0 and plan["why"].startswith("default")


def test_stabilize_when_three_losses_from_death():
    state = make_state(idx=16, hp=30, level=6, xp=30, gold=45)  # 4-2: ~11.7 a loss (set4)
    assert state["losses_to_death"] == 3
    plan = run(v1(), state)
    assert plan["stance"] == "stabilize" and plan["why"].startswith("dying")
    assert plan["roll_floor"] == st.STABILIZE_HARD_FLOOR and plan["survival"] == st.SURVIVAL_LTD
    assert plan["level_to"] == 7  # 6 xp short: cheap
    assert run(v1(), make_state(idx=16, hp=50))["stance"] == "standard"  # 4 to death


def test_stabilize_on_heavy_losses_from_stage_three():
    strong = [opp(f"player_{i}", STRONG) for i in range(1, 8)]
    plan = run(v1(), make_state(idx=12, hp=80, streak=-3, opponents=strong))
    assert plan["stance"] == "stabilize" and plan["why"].startswith("heavy")
    assert plan["roll_floor"] == st.STABILIZE_SOFT_FLOOR
    assert run(v1(), make_state(idx=6, hp=80, streak=-3, opponents=strong))["stance"] == "standard"
    assert run(v1(), make_state(idx=12, hp=80, streak=-2, opponents=strong))["stance"] == "standard"
    assert run(v1(), make_state(idx=12, hp=80, streak=-3))["stance"] == "standard"  # even boards


SLOW = dict(idx=9, level=5, gold=60, board=["nami*2", "ahri*1", "lulu*1", "twistedfate*1", "annie*1"],
            bench=["nami*1"])  # 4 copies of a 1-cost mage


def test_slow_roll_starts_on_four_copies_of_a_cheap_comp_unit():
    plan = run(v1(), make_state(**SLOW))
    assert plan["stance"] == "slow_roll" and plan["why"].startswith("start: nami")
    assert plan["level_to"] == 5 and plan["roll_floor"] == st.SLOWROLL_FLOOR and plan["carry"] == "nami"
    assert plan["comp"] == "mage"  # of mage / spirit / enlightened, the one holding most of the units
    assert run(v1(), make_state(**{**SLOW, "level": 4}))["level_to"] == 5  # one level up first


@pytest.mark.parametrize("change", [
    {"level": 6},  # above the 1-cost reroll level
    {"bench": []},  # 3 copies
    {"opponents": [opp("player_1", ["nami*2", "nami*1"])] + [opp(f"player_{i}") for i in range(2, 8)]},  # 4 seen
    {"hp": 45},
    {"idx": 16},  # past SLOWROLL_UNTIL for a 1-cost
])
def test_slow_roll_does_not_start(change):
    assert run(v1(), make_state(**{**SLOW, **change}))["stance"] != "slow_roll"


def test_slow_roll_only_inside_the_chosen_comp():
    assert run(v1(), make_state(**SLOW, comp="divine"), "divine")["stance"] == "standard"
    plan = run(v1(), make_state(**SLOW, comp="mage"), "mage")
    assert plan["stance"] == "slow_roll" and plan["comp"] is None  # keep the executor's comp


@pytest.mark.parametrize("later, reason", [
    ({}, None),
    ({"board": ["nami*3", "ahri*1", "lulu*1", "twistedfate*1", "annie*1"], "bench": []}, "3-star"),
    ({"hp": 54}, "bleeding"),
    ({"opponents": [opp("player_1", ["nami*2", "nami*1"])] + [opp(f"player_{i}") for i in range(2, 8)]},
     "contested"),
    ({"bench": []}, "copies sold"),
    ({"idx": 16}, "too late"),
])
def test_slow_roll_is_kept_until_a_stop_condition(later, reason):
    planner = v1()
    assert run(planner, make_state(**SLOW))["stance"] == "slow_roll"
    plan = run(planner, make_state(**{**SLOW, "idx": 10, **later}))
    if reason is None:
        assert plan["stance"] == "slow_roll" and plan["why"].startswith("keep: nami") and plan["level_to"] == 5
    else:
        assert plan["stance"] == "standard" and plan["features"]["slow_roll_end"] == f"nami: {reason}"
        again = make_state(**{**SLOW, "idx": plan["features"]["round"] + 1})
        assert run(planner, again)["stance"] != "slow_roll"  # not restarted on the same unit


FAST8 = dict(idx=12, level=6, xp=0, gold=75, hp=80)  # 3-5: 75 gold -> 115 by 4-2, 84 of it for level 8


def test_fast8_commits_levels_by_the_deadline_then_rolls_down():
    planner = v1()
    plan = run(planner, make_state(**FAST8))
    assert plan["stance"] == "fast8" and plan["why"].startswith("start: 8 by 4-2")
    assert plan["level_by"] == {"level": 8, "by": "4-2"} and plan["level_to"] == 6 and plan["roll_floor"] == 999
    assert compile_knobs(plan, make_state(**FAST8))["xp_priority"]
    keep = run(planner, make_state(**{**FAST8, "idx": 13, "gold": 40}))  # committed: no re-check of the gold
    assert keep["stance"] == "fast8" and keep["why"].startswith("keep") and keep["level_by"]["by"] == "4-2"
    landed = run(planner, make_state(**{**FAST8, "idx": 17, "level": 8, "gold": 30}))
    assert landed["stance"] == "fast8" and landed["spend"] == {"to": st.FAST8_ROLL_TO, "by": "4-5"}
    assert run(planner, make_state(**{**FAST8, "idx": 18, "level": 8, "gold": 25}))["stance"] == "fast8"
    assert run(planner, make_state(**{**FAST8, "idx": 19, "level": 8, "gold": 25}))["stance"] == "standard"


def test_fast8_dropped_after_a_missed_deadline_or_below_50_hp():
    planner = v1()
    run(planner, make_state(**FAST8))
    assert run(planner, make_state(**{**FAST8, "idx": 17, "level": 7}))["stance"] == "fast8"  # 1 round grace
    plan = run(planner, make_state(**{**FAST8, "idx": 18, "level": 7}))
    assert plan["stance"] == "standard" and plan["features"]["fast8_end"] == "missed the deadline"
    planner = v1()
    run(planner, make_state(**FAST8))
    assert run(planner, make_state(**{**FAST8, "idx": 13, "hp": 52}))["stance"] == "fast8"
    plan = run(planner, make_state(**{**FAST8, "idx": 14, "hp": 46}))  # 4 losses to death: not yet stabilize
    assert plan["stance"] == "standard" and plan["features"]["fast8_end"] == "hp 46"


@pytest.mark.parametrize("change", [{"gold": 40}, {"hp": 55}, {"idx": 11}, {"idx": 17},
                                    {"opponents": [opp(f"player_{i}", STRONG) for i in range(1, 8)]}])
def test_fast8_does_not_start(change):
    assert run(v1(), make_state(**{**FAST8, **change}))["stance"] != "fast8"


def test_fast8_goes_earlier_when_opponents_are_close_to_8():
    racing = [opp("player_1", level=8), opp("player_2", level=7, interest=5)] + \
        [opp(f"player_{i}") for i in range(3, 8)]
    plan = run(v1(), make_state(**{**FAST8, "gold": 90, "opponents": racing}))
    assert plan["stance"] == "fast8" and plan["level_by"]["by"] == "4-1"
    # 75 gold pays for 8 by 4-2 but not by 4-1
    assert run(v1(), make_state(**{**FAST8, "opponents": racing}))["stance"] != "fast8"


STREAK = dict(idx=6, level=4, xp=0, gold=30, streak=3, board=STRONG[:4],
              opponents=[opp(f"player_{i}", WEAK, level=4) for i in range(1, 8)])


def test_keep_streak_levels_and_rolls_the_excess():
    plan = run(v1(), make_state(**STREAK))
    assert plan["stance"] == "keep_streak" and plan["why"].startswith("streak: 3 wins")
    assert plan["level_to"] == 5 and plan["roll_floor"] == st.STREAK_ROLL_FLOOR  # 12 gold for 5, 18 left
    assert run(v1(), make_state(**{**STREAK, "level": 5}))["level_to"] == 5  # stage-2 cap
    assert run(v1(), make_state(**{**STREAK, "gold": 20}))["level_to"] == 4  # would leave 8


@pytest.mark.parametrize("change", [{"streak": 1}, {"opponents": [opp(f"player_{i}", STRONG) for i in range(1, 8)]}])
def test_keep_streak_needs_a_streak_and_the_stronger_board(change):
    assert run(v1(), make_state(**{**STREAK, **change}))["stance"] == "standard"


def test_priorities():
    dying_streak = make_state(**{**STREAK, "idx": 16, "hp": 30})
    assert run(v1(), dying_streak)["stance"] == "stabilize"
    assert run(v1(), make_state(**{**FAST8, "streak": 3}))["stance"] == "fast8"
    # a 2-cost with 4 copies at level 6 and the gold for a fast 8: the slow roll goes first ...
    planner = v1()
    slow_and_rich = make_state(**{**FAST8, "board": ["teemo*2", "ahri*1", "lulu*1", "veigar*1", "thresh*1"],
                                  "bench": ["teemo*1"]})  # teemo: a 2-cost no opponent fields
    assert run(planner, slow_and_rich)["stance"] == "slow_roll"
    # ... and blocks the fast 8 while it lasts
    assert run(planner, {**slow_and_rich, "round": 13, "stage": "3-6"})["stance"] == "slow_roll"


def test_stance_seat_kinds_are_registered():
    assert set(st.STANCE_KINDS) == {
        "stance", "stance1", "stance-nofast8", "stance-nostreak", "stance-nostabilize", "stance-nocapout",
        "stance-nolobbylevel", "stance-fast8v1", "stance-stabilizev1", "stance-nohprank", "stance-ratio",
        "stance+lossstreak", "stance+slowroll", "stance+hold"}
    plain = make_policy("stance").planner
    assert make_policy("stance").name == "stance" and not plain.disabled and plain.opts == st.V2
    assert not plain.on("loss_streak") and not plain.on("slow_roll") and plain.on("cap_out")
    old = make_policy("stance1").planner
    assert old.opts == st.V1 and old.on("slow_roll") and not old.on("cap_out") and not old.on("loss_streak")
    assert make_policy("stance+lossstreak").planner.on("loss_streak")
    assert make_policy("stance+slowroll").planner.on("slow_roll")
    assert make_policy("stance+hold").planner.opts == {**st.V2, "hold": True}
    for kind, spec in st.STANCE_KINDS.items():
        planner = make_policy(f"x={kind}").planner
        assert planner.disabled == set(spec.get("off", ())) and planner.enabled == set(spec.get("on", ()))
        assert planner.opts == {**st.V2, **spec.get("opts", {})}
    with pytest.raises(ValueError):
        st.StancePlanner(disabled=["standard"])
    with pytest.raises(ValueError):
        st.StancePlanner(enabled=["fodder"])
    with pytest.raises(ValueError):
        st.StancePlanner(opts={"slowroll": True})


# --------------------------------------------------------------------------- controlled loss streak (opt-in)

LOSS = dict(idx=4, level=3, gold=12, hp=92, streak=-1)  # 2-2, lost 2-1


def loss_planner():
    return v1(enabled=["loss_streak"])


def test_loss_streak_is_off_in_the_default_stance_policy():
    assert run(v1(), make_state(**LOSS))["stance"] == "standard"


def test_loss_streak_fields_fodder_while_losing_at_high_hp():
    planner = loss_planner()
    plan = run(planner, make_state(**LOSS))
    assert plan["stance"] == "loss_streak" and plan["fodder"] is True and plan["why"].startswith("start: 1 losses")
    assert compile_knobs(plan, make_state(**LOSS))["fodder"]
    keep = run(planner, make_state(**{**LOSS, "idx": 5, "hp": 84, "streak": -2}))
    assert keep["stance"] == "loss_streak" and keep["why"].startswith("keep: 2 losses")
    over = run(planner, make_state(**{**LOSS, "idx": 6, "hp": 76, "streak": -3}))  # below 80 HP
    assert over["stance"] == "standard" and "fodder" not in over and over["features"]["loss_streak_end"] == "hp 76"
    assert run(planner, make_state(**{**LOSS, "idx": 7, "hp": 90, "streak": -4}))["stance"] == "standard"  # once


def test_loss_streak_starts_on_a_weak_board_and_ends_when_the_streak_breaks():
    planner = loss_planner()
    strong = [opp(f"player_{i}", STRONG) for i in range(1, 8)]
    plan = run(planner, make_state(**{**LOSS, "idx": 3, "hp": 100, "streak": 0, "opponents": strong}))
    assert plan["stance"] == "loss_streak" and plan["why"].startswith("start: board")
    broken = run(planner, make_state(**{**LOSS, "idx": 4, "hp": 100, "streak": 1, "opponents": strong}))
    assert broken["stance"] == "standard" and broken["features"]["loss_streak_end"] == "streak broken"


@pytest.mark.parametrize("change", [{"hp": 78}, {"streak": 0}, {"streak": 2}, {"idx": 8}, {"idx": 9}])
def test_loss_streak_does_not_start(change):
    assert run(loss_planner(), make_state(**{**LOSS, **change}))["stance"] != "loss_streak"


def test_losses_on_purpose_do_not_count_for_the_heavy_loss_stabilize():
    """stance+lossstreak loses stage 2 on purpose; the streak it carries into stage 3 must not trigger
    stabilize's "heavy: loss streak" (which would roll away the gold the streak was for). Losses after
    the fodder board count again."""
    planner = v2("stance+lossstreak")
    strong = [opp(f"player_{i}", STRONG) for i in range(1, 8)]
    for idx, hp, streak in ((3, 100, 0), (4, 96, -1), (5, 92, -2), (6, 88, -3), (7, 84, -4)):  # 2-1 .. 2-6
        plan = run(planner, make_state(idx=idx, hp=hp, gold=20, level=4, streak=streak, opponents=strong))
        assert plan["stance"] == "loss_streak" and plan["fodder"] is True, (idx, plan["why"])
    run(planner, make_state(idx=8, hp=84, gold=30, level=5, streak=-5, opponents=strong))  # 2-7 PvE: over
    assert planner.fodder_rounds == {3, 4, 5, 6, 7}
    for idx, streak in ((9, -5), (10, -6), (11, -7)):  # 3-1: all five on purpose; then 1 and 2 real losses
        plan = run(planner, make_state(idx=idx, hp=84, gold=30, level=5, streak=streak, opponents=strong))
        assert plan["stance"] != "stabilize", (idx, plan["why"])
    plan = run(planner, make_state(idx=12, hp=84, gold=30, level=5, streak=-8, opponents=strong))
    assert plan["stance"] == "stabilize" and plan["why"].startswith("heavy: loss streak 3")
    # the same streak without a fodder board behind it is heavy at once
    assert run(v2(), make_state(idx=9, hp=84, gold=30, level=5, streak=-5, opponents=strong))["why"].startswith(
        "heavy: loss streak 5")


def test_economy_numbers_follow_the_rules_profile():
    assert [st.streak_bonus(n, "set4") for n in range(2, 7)] == [1, 1, 2, 3, 3]
    assert [st.streak_bonus(n, "set18") for n in range(2, 7)] == [1, 1, 1, 2, 3]
    assert st.projected_gold(20, 2, "set4") == st.projected_gold(20, 2, "set18") == 20 + 7 + 7
    assert st.xp_gold(8, 0, 9, rules="set18") == 68 and st.xp_gold(8, 0, 9, rules="set4") == 80
    assert st.xp_total(9, 0, "set18") == 200 and st.xp_total(9, 0, "set4") == 212
    f = st.features(make_state(streak=4, rules="set18"), COMPS, None)
    assert f["streak_gold_win"] == 1 + 2  # a fifth win: 2 under set18, 3 under set4
    assert st.features(make_state(streak=4), COMPS, None)["streak_gold_win"] == 1 + 3


def test_loss_streak_ends_with_stage_two():
    planner = loss_planner()
    assert run(planner, make_state(**{**LOSS, "idx": 7, "streak": -3}))["stance"] == "loss_streak"
    plan = run(planner, make_state(**{**LOSS, "idx": 8, "streak": -4}))  # 2-7 is PvE: no fodder there
    assert plan["stance"] == "standard" and plan["features"]["loss_streak_end"] == "end of stage 2"


def test_new_game_resets_the_commitments():
    planner = v1()
    run(planner, make_state(**FAST8))
    assert planner.fast8 is not None
    run(planner, make_state(idx=3, level=3, gold=5))
    assert planner.fast8 is None and planner.hp_seen == {3: 80}


def test_every_plan_compiles_and_logs(tmp_path):
    log = tmp_path / "stance.jsonl"
    planner = v1(enabled=["loss_streak"], log_path=str(log))
    planner.context = {"seed": 7, "seat": "player_3"}
    states = [make_state(idx=1, level=2), make_state(**SLOW), make_state(**FAST8), make_state(**STREAK),
              make_state(idx=16, hp=20), make_state(**LOSS)]
    for state in states:
        planner._new_game()
        plan = run(planner, state)
        compile_knobs(plan, state)
        json.dumps(plan)  # plain data
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["stance"] for r in rows] == ["standard", "slow_roll", "fast8", "keep_streak", "stabilize",
                                           "loss_streak"]
    assert rows[0]["seed"] == 7 and rows[0]["seat"] == "player_3" and "features" not in rows[0]["plan"]


# --------------------------------------------------------------------------- version 2

def test_strength_classes_from_p_win_and_from_the_ratio():
    f = {"p_mean": 0.6, "p_min": 0.5, "ratio": 1.0, "score": 10, "cand_max": 10}
    assert st.classify(f, True) == "stronger"
    assert st.classify({**f, "p_min": 0.49}, True) == "close"  # one candidate is too strong
    assert st.classify({**f, "p_mean": 0.59}, True) == "close"
    assert st.classify({**f, "p_mean": 0.39}, True) == "weaker"
    assert st.classify({**f, "p_mean": None, "p_min": None}, True) == "close"  # nobody to compare with
    assert st.classify(f, False) == "close"
    assert st.classify({**f, "ratio": 1.2, "score": 12}, False) == "stronger"
    assert st.classify({**f, "ratio": 1.2, "score": 12, "cand_max": 13}, False) == "close"  # 12 < 0.95 x 13
    assert st.classify({**f, "ratio": 0.94}, False) == "weaker"


def test_p_win_is_taken_against_the_candidates_only():
    opponents = [opp("player_1", WEAK), opp("player_2", EVEN), opp("player_3", STRONG)]
    f = run(v2(), make_state(opponents=opponents, next_from=["player_1", "player_2"]))["features"]
    assert f["p_mean"] == round((STAND_IN.p(*[winprob.view_features(v) for v in (
        {"board": EVEN, "level": 6, "hp": 80}, {"board": WEAK, "level": 6, "hp": 80})]) + 0.5) / 2, 3)
    assert f["p_min"] == 0.5 and f["strength"] == "close"
    assert run(v2(), make_state(board=STRONG))["features"]["strength"] == "stronger"
    assert run(v2(), make_state(board=WEAK))["features"]["strength"] == "weaker"


def test_own_units_count_pairs_and_stars():
    state = make_state(board=["ahri*2", "annie*1", "lulu*1"], bench=["annie*1", "lulu*1", "lulu*1", "jax*3"])
    assert st.own_units(state) == {"units_owned": 7, "pairs": 1, "two_stars": 1, "three_stars": 1}  # lulu x3: no pair


def test_v2_defaults_slow_roll_and_loss_streak_off():
    assert run(v2(), make_state(**SLOW))["stance"] != "slow_roll"
    assert run(v2("stance+slowroll"), make_state(**SLOW))["stance"] == "slow_roll"
    assert run(v2(), make_state(**LOSS))["stance"] == "standard"
    assert run(v2("stance+lossstreak"), make_state(**LOSS))["stance"] == "loss_streak"


# standard: lobby-relative levelling

def test_standard_levels_to_the_lobby_median_when_30_gold_is_left():
    calm = dict(idx=6, level=4, xp=0, hp=100)  # 2-5: no curve yet; the opponents are level 6
    plan = run(v2(), make_state(**calm, gold=45))  # level 5 costs 12: 33 left; level 6 costs 32: 13 left
    assert plan["stance"] == "standard" and plan["level_to"] == 5
    assert run(v2(), make_state(**calm, gold=41))["level_to"] == 4  # 29 would be left
    assert run(v2(), make_state(**calm, gold=62))["level_to"] == 6  # both levels, 30 left
    low = [opp(f"player_{i}", level=4) for i in range(1, 8)]
    assert run(v2(), make_state(**calm, gold=62, opponents=low))["level_to"] == 4  # already at the median
    assert run(v1(), make_state(**calm, gold=62))["level_to"] == 4  # v1: the rule bot's economy only
    assert run(v2("stance-nolobbylevel"), make_state(**calm, gold=62))["level_to"] == 4


@pytest.mark.parametrize("idx, level, xp, gold, want", [
    (9, 5, 0, 80, 5),    # 3-1: no curve, opponents level 4
    (10, 5, 0, 50, 6),   # 3-2: at least 6 (20 xp, 30 left)
    (10, 5, 0, 49, 5),   # ... not when 29 would be left
    (12, 5, 0, 90, 7),   # 3-5: at least 7, both levels (56 gold, 34 left)
    (12, 5, 0, 70, 6),   # ... one level when two would leave less than 30
    (16, 7, 40, 50, 8),  # 4-2: at least 8 (16 xp, 34 left)
    (16, 7, 40, 45, 7),  # ... not when 29 would be left
    (16, 8, 0, 60, 8),   # the curve stops at 8
])
def test_standard_follows_the_level_curve(idx, level, xp, gold, want):
    low = [opp(f"player_{i}", level=4) for i in range(1, 8)]
    plan = run(v2(), make_state(idx=idx, level=level, xp=xp, gold=gold, hp=100, opponents=low))
    assert plan["stance"] == "standard" and plan["level_to"] == want


# fast 8 v2

FAST8V2 = dict(idx=12, level=7, xp=0, gold=62, hp=80)  # 3-5: 62 -> 92 by 4-1, 52 of it for level 8


def test_fast8_v2_buys_the_xp_above_50_gold_and_lands_by_4_1():
    planner = v2()
    plan = run(planner, make_state(**FAST8V2))
    assert plan["stance"] == "fast8" and plan["why"].startswith("start: 8 by 4-1")
    assert plan["level_by"] == {"level": 8, "by": "4-1"} and plan["roll_floor"] == 999
    assert plan["xp_buys"] == 3  # 62 - 3 x 4 = 50
    knobs = compile_knobs(plan, make_state(**FAST8V2))
    assert knobs["xp_buys"] == 3 and knobs["xp_priority"]
    keep = run(planner, make_state(**{**FAST8V2, "idx": 13, "gold": 52, "xp": 14}))
    assert keep["stance"] == "fast8" and "xp_buys" not in keep  # 52: one buy would leave 48
    big = run(planner, make_state(**{**FAST8V2, "idx": 14, "gold": 200, "xp": 30}))
    assert big["stance"] == "fast8" and big["xp_buys"] == 7  # capped at the 26 xp still missing
    # (200 gold at 80 HP would stabilize, 6 losses to death <= 2 + 12, but not a fast 8 on its way)
    on = run(planner, make_state(**{**FAST8V2, "idx": 15, "gold": 60, "xp": 50}))
    assert compile_knobs(on, make_state(**{**FAST8V2, "idx": 15, "gold": 60, "xp": 50}))["level_to"] == 8


def test_fast8_v1_banks_until_the_deadline():
    plan = run(v1(), make_state(**{**FAST8V2, "gold": 80}))
    assert plan["stance"] == "fast8" and plan["level_by"]["by"] == "4-2" and "xp_buys" not in plan
    plan = run(v2("stance-fast8v1"), make_state(**{**FAST8V2, "gold": 80}))
    assert plan["stance"] == "fast8" and plan["level_by"]["by"] == "4-2" and "xp_buys" not in plan


def test_fast8_v2_rolls_to_30_after_landing_and_stops_once_stronger():
    planner = v2()
    run(planner, make_state(**FAST8V2))
    landed = run(planner, make_state(**{**FAST8V2, "idx": 15, "level": 8, "gold": 70}))
    assert landed["stance"] == "fast8" and landed["spend"] == {"to": st.FAST8_V2_ROLL_TO, "by": "4-2"}
    strong = run(planner, make_state(**{**FAST8V2, "idx": 16, "level": 8, "gold": 50, "board": STRONG}))
    assert strong["stance"] != "fast8" and strong["features"]["fast8_end"] == "stronger at 8"
    assert planner.fast8 is None


def test_fast8_v2_is_not_dropped_for_low_hp_once_xp_was_bought():
    planner = v2()
    run(planner, make_state(**FAST8V2))
    low = {**FAST8V2, "idx": 13, "hp": 45, "gold": 20}  # 4 losses to death; 20 gold: no stabilize
    plan = run(planner, make_state(**{**low, "xp": 2 + 12}))  # passive 2 + three buys
    assert plan["stance"] == "fast8" and plan["why"].startswith("keep")
    planner = v2()
    run(planner, make_state(**FAST8V2))
    plan = run(planner, make_state(**{**low, "xp": 2}))  # passive xp only
    assert plan["stance"] == "standard" and plan["features"]["fast8_end"] == "hp 45"


@pytest.mark.parametrize("change", [{"gold": 30}, {"hp": 55}, {"idx": 11}, {"idx": 16}, {"board": WEAK}])
def test_fast8_v2_does_not_start(change):
    assert run(v2(), make_state(**{**FAST8V2, **change}))["stance"] != "fast8"


# stabilize v2

@pytest.mark.parametrize("gold, fires", [(20, False), (21, True), (35, True), (36, True)])
def test_stabilize_v2_fires_when_the_gold_cannot_be_spent_in_time(gold, fires):
    state = make_state(idx=16, hp=30, level=6, xp=0, gold=gold)  # 4-2: 3 losses to death (set4)
    assert state["losses_to_death"] == 3  # within reach: 2 + ceil((gold - 20) / 15) >= 3 from 21 gold
    plan = run(v2(), state)
    assert (plan["stance"] == "stabilize") == fires
    assert run(v1(), state)["stance"] == "stabilize"  # v1: always at 3


def test_stabilize_v2_fires_early_with_a_big_bank():
    state = make_state(idx=12, hp=50, level=6, gold=80)  # 3-5: 5 losses to death, reach 2 + 4 = 6
    assert state["losses_to_death"] == 5
    plan = run(v2(), state)
    assert plan["stance"] == "stabilize" and plan["why"].startswith("in reach: 5 losses to death <= 6")
    assert run(v1(), state)["stance"] != "stabilize"
    assert run(v2("stance-stabilizev1"), state)["stance"] != "stabilize"
    assert run(v2(), make_state(idx=12, hp=50, level=6, gold=80, board=STRONG))["stance"] != "stabilize"


def test_stabilize_v2_before_3_2_keeps_the_v1_trigger():
    assert run(v2(), make_state(idx=9, hp=20, gold=10))["why"].startswith("dying: 3")
    assert run(v2(), make_state(idx=9, hp=30, gold=80))["stance"] != "stabilize"  # 4 to death


def test_stabilize_v2_roll_floor_follows_the_strength():
    weak = run(v2(), make_state(idx=16, hp=40, level=5, xp=0, gold=45, board=WEAK))
    assert weak["features"]["strength"] == "weaker" and weak["roll_floor"] == st.STABILIZE_WEAK_FLOOR
    assert weak["level_to"] == 5 and weak["survival"] == st.SURVIVAL_LTD  # level 6 is 20 xp: not cheap
    close = run(v2(), make_state(idx=16, hp=40, level=5, xp=0, gold=45))
    assert close["features"]["strength"] == "close" and close["roll_floor"] == st.STABILIZE_CLOSE_FLOOR
    assert close["level_to"] == 6  # xp first: 20 gold for level 6 leaves 25
    poor = run(v2(), make_state(idx=16, hp=40, level=5, xp=0, gold=36))
    assert poor["stance"] == "stabilize" and poor["level_to"] == 5  # 16 would be left
    eight = run(v2(), make_state(idx=16, hp=40, level=7, xp=40, gold=36))  # 16 xp = 4 buys to 8
    assert eight["level_to"] == 8
    assert run(v2(), make_state(idx=16, hp=40, level=7, xp=36, gold=36))["level_to"] == 7  # 5 buys


def test_stabilize_v2_when_last_late_in_the_game():
    three = [opp("player_1", hp=90), opp("player_2", hp=70), opp("player_3", hp=60)]
    plan = run(v2(), make_state(idx=21, hp=55, gold=30, level=8, opponents=three))
    assert plan["stance"] == "stabilize" and plan["why"].startswith("last: lowest HP of 4 alive")
    assert run(v2("stance-nohprank"), make_state(idx=21, hp=55, gold=30, level=8, opponents=three))["stance"] \
        != "stabilize"
    four = three + [opp("player_4", hp=58)]
    assert run(v2(), make_state(idx=21, hp=55, gold=30, level=8, opponents=four))["stance"] != "stabilize"
    strong = run(v2(), make_state(idx=21, hp=55, gold=30, level=8, opponents=three, board=STRONG))
    assert strong["stance"] == "stabilize" and "survival" not in strong  # stronger: survival mode without rolls
    assert strong["roll_floor"] == ParamPlanner().plan(make_state(idx=21, hp=55, gold=30, level=8), COMPS,
                                                       None)["roll_floor"]


# cap out

CAP = dict(idx=21, level=8, xp=0, gold=60, hp=70, board=STRONG)  # 5-1, clearly stronger


def test_cap_out_buys_toward_level_9_without_rolling():
    plan = run(v2(), make_state(**CAP))
    assert plan["stance"] == "cap_out" and plan["roll_floor"] == 999 and plan["level_to"] == 8
    assert plan["xp_buys"] == 12  # 60 - 12 x 4 = 12 >= CAPOUT_KEEP
    assert compile_knobs(plan, make_state(**CAP))["xp_buys"] == 12


@pytest.mark.parametrize("change", [{"gold": 49}, {"idx": 20}, {"level": 7}, {"board": EVEN}])
def test_cap_out_needs_stage_5_level_8_50_gold_and_a_stronger_board(change):
    assert run(v2(), make_state(**{**CAP, **change}))["stance"] != "cap_out"


def test_cap_out_when_first_late_in_the_game_and_never_in_v1():
    three = [opp("player_1", hp=50), opp("player_2", hp=40), opp("player_3", hp=30)]
    first = make_state(**{**CAP, "idx": 18, "gold": 30, "opponents": three})  # 4-5, first of 4 alive
    plan = run(v2(), first)
    assert plan["stance"] == "cap_out" and plan["why"].startswith("first") and plan["xp_buys"] == 5
    assert run(v2("stance-nohprank"), first)["stance"] != "cap_out"
    assert run(v2("stance-nocapout"), make_state(**CAP))["stance"] != "cap_out"
    assert run(v1(), make_state(**CAP))["stance"] != "cap_out"


# hold

def test_hold_is_planned_in_stage_2_for_the_hold_seat_only():
    for idx in range(3, 9):  # 2-1 .. 2-7
        plan = run(v2("stance+hold"), make_state(idx=idx, level=4, gold=10, hp=90))
        assert plan["hold"] is True and compile_knobs(plan, make_state(idx=idx, level=4, gold=10, hp=90))["hold"]
        assert "hold" not in run(v2(), make_state(idx=idx, level=4, gold=10, hp=90))
    assert "hold" not in run(v2("stance+hold"), make_state(idx=9, level=5, gold=10, hp=90))
    assert "hold" not in run(v2("stance+hold"), make_state(idx=2, level=3, gold=10, hp=100))
    assert "hold" not in run(v1(), make_state(idx=5, level=4, gold=10, hp=90))


def test_survival_overrides_hold():
    plan = {"comp": None, "level_to": 4, "roll_floor": 999, "carry": None, "hold": True, "survival": 2}
    assert compile_knobs(plan, {"round": 5, "gold": 10, "level": 4, "xp": 0, "hp": 90})["hold"]
    assert not compile_knobs(plan, {"round": 5, "gold": 10, "level": 4, "xp": 0, "hp": 8})["hold"]


# ablations and logging

@pytest.mark.parametrize("kind, state, stance", [
    ("stance-nostabilize", dict(idx=16, hp=40, level=6, gold=45), "stabilize"),
    ("stance-nofast8", FAST8V2, "fast8"),
    ("stance-nostreak", STREAK, "keep_streak"),
    ("stance-nocapout", CAP, "cap_out"),
])
def test_v2_ablations_fall_through_to_standard(kind, state, stance):
    assert run(v2(), make_state(**state))["stance"] == stance
    policy = make_policy(f"hero={kind}")
    assert isinstance(policy, PlanPolicy) and policy.name == "hero"
    policy.planner.model = STAND_IN
    assert run(policy.planner, make_state(**state))["stance"] == "standard"


def test_v2_plans_compile_and_log_the_unit_counts(tmp_path):
    log = tmp_path / "stance.jsonl"
    planner = v2(log_path=str(log))
    states = [make_state(idx=1, level=2), make_state(**FAST8V2), make_state(**STREAK), make_state(**CAP),
              make_state(idx=16, hp=40, level=6, gold=45), make_state(idx=10, level=5, gold=50, hp=100)]
    for state in states:
        planner._new_game()
        plan = run(planner, state)
        compile_knobs(plan, state)
        json.dumps(plan)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["stance"] for r in rows] == ["standard", "fast8", "keep_streak", "cap_out", "stabilize", "standard"]
    assert all({"units_owned", "pairs", "two_stars", "p_mean", "strength"} <= set(r["features"]) for r in rows)


# --------------------------------------------------------------------------- information boundary

@pytest.mark.parametrize("module, used", [(st, {"opponents", "next_from", "board", "bench"}),
                                          (winprob, {"board", "level", "hp", "streak"})])
def test_the_module_never_reaches_for_hidden_state(module, used):
    """No name, attribute or key in the planner's code or the win-probability model's (docstrings aside)
    that leads to the env or to another player's hidden state; they only read the describe() dict."""
    tree = ast.parse(Path(module.__file__).read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    words = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            words.add(node.id)
        elif isinstance(node, ast.Attribute):
            words.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            words.update(re.findall(r"\w+", node.value))
    hidden = {"env", "matchups", "game_round", "player_manager", "player_states", "unwrapped", "opponent_options",
              "possible_opponents", "health", "exp", "shop", "item_bench", "pool", "Player"}
    assert not words & hidden
    assert used <= words  # (the check does see the code)


class Trap:
    """Stands in for game_round: any attribute read fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"read game_round.{name}")


def make_player(num, board=(), bench=(), gold=0, level=4, hp=100):
    p = Player(pool.pool(), num)
    p.gold, p.level, p.max_units, p.health = gold, level, max(level, len(board)), hp
    for i, tag in enumerate(board):
        name, _, stars = tag.partition("*")
        champ = champion(name, stars=int(stars or 1))
        assert p.add_to_bench(champ)
        assert p.move_bench_to_board(p.bench.index(champ), i, 0)
    for tag in bench:
        name, _, stars = tag.partition("*")
        assert p.add_to_bench(champion(name, stars=int(stars or 1)))
    return p


@pytest.fixture
def lobby():
    me = make_player(0, board=["nami*2", "ahri", "lulu", "twistedfate"], bench=["nami"], gold=62, level=5, hp=80)
    rich = make_player(1, board=["ahri*2", "lulu", "veigar", "nami"], bench=["veigar", "veigar"], gold=57, level=6)
    poor = make_player(2, board=["vayne", "fiora"], gold=4, level=5, hp=35)
    players = {"player_0": me, "player_1": rich, "player_2": poor}
    me.opponent_options = {"player_1": 1, "player_2": 1}
    env = SimpleNamespace(unwrapped=SimpleNamespace(player_manager=SimpleNamespace(player_states=players),
                                                    game_round=Trap()))
    return me, players, env


# version 1, and version 2 with the slow roll on and the fitted win-probability model
PLANNERS = {"stance1": v1, "stance+slowroll": lambda: make_policy("stance+slowroll").planner}


def plan_for(me, env, idx=9, kind="stance1"):
    state = describe(me, ["zed", None], idx, env, seat="player_0", comps=COMPS, comp=None)
    return json.dumps(PLANNERS[kind]().plan(state, COMPS, None), sort_keys=True), state


def test_synthetic_state_has_the_describe_keys(lobby):
    me, _, env = lobby
    _, state = plan_for(me, env)
    assert set(make_state()) == set(state)


@pytest.mark.parametrize("kind", sorted(PLANNERS))
def test_plan_ignores_hidden_state_and_matchups(lobby, kind):
    me, players, env = lobby
    before, state = plan_for(me, env, kind=kind)
    assert json.loads(before)["stance"] == "slow_roll"  # 4 copies of nami, 1 seen
    assert kind == "stance1" or json.loads(before)["features"]["p_mean"] is not None
    rich = players["player_1"]
    rich.gold, rich.exp = 51, 17  # same interest bracket
    rich.bench = [champion("nami")] * 3 + [None] * 6  # many more namis, but out of sight
    rich.shop = ["nami"] * 5
    rich.item_bench = ["chain_vest"] + [None] * 9
    me.possible_opponents = {k: 99 for k in me.possible_opponents}
    env.unwrapped.game_round = SimpleNamespace(matchups=[("player_0", "player_2")])  # the "real" next opponent
    after, _ = plan_for(me, env, kind=kind)
    assert after == before
    # the same plan from a plain-data copy of the state: nothing live is reachable from it
    assert json.dumps(PLANNERS[kind]().plan(json.loads(json.dumps(state)), COMPS, None), sort_keys=True) == before
    # public information does move it: the poor seat fields three namis
    poor = players["player_2"]
    poor.board[0][0] = champion("nami", stars=2)
    poor.board[1][0] = champion("nami")
    assert json.loads(plan_for(me, env, kind=kind)[0])["stance"] != "slow_roll"

"""Executor capabilities: plan compilation, comp pivot, multi-round budgets, fodder board, survival."""

import numpy as np
import pytest

from tfteval.planner import (LLMPlanner, ParamPlanner, XP_BUY_CAP, compile_knobs, xp_to_level)

BASE = {"comp": None, "level_to": 5, "roll_floor": 999, "carry": None}


def state(idx, gold=50, level=5, xp=0, hp=100):
    return {"round": idx, "gold": gold, "level": level, "xp": xp, "hp": hp}


# --------------------------------------------------------------------------- compile_knobs

def test_plain_plan_compiles_to_itself():
    knobs = compile_knobs({**BASE, "roll_floor": 20, "carry": "ahri", "comp": "mage"}, state(12))
    assert knobs == {"comp": "mage", "level_to": 5, "roll_floor": 20, "carry": "ahri", "xp_buys": 0,
                     "xp_priority": False, "fodder": False, "survival": False}


def test_level_by_buys_as_late_as_the_action_cap_allows():
    plan = {**BASE, "level_to": 7, "level_by": {"level": 8, "by": "4-1"}}  # 4-1 is idx 15
    assert xp_to_level(7, 0, 8) == 56  # 14 buy-xp actions
    early = compile_knobs(plan, state(13, gold=60, level=7))  # two rounds before: 56 - 4 passive = 13 buys
    assert early["xp_buys"] == 0 and early["level_to"] == 7  # 13 <= 2 * XP_BUY_CAP: nothing yet
    before = compile_knobs(plan, state(14, gold=60, level=7))  # one round before: 54 xp -> 14 buys
    assert before["xp_buys"] == 14 - XP_BUY_CAP and before["xp_priority"]
    on = compile_knobs(plan, state(15, gold=60, level=7, xp=18))
    assert on["level_to"] == 8 and on["xp_buys"] == 0
    late = compile_knobs(plan, state(17, gold=60, level=7, xp=18))
    assert late["level_to"] == 8
    done = compile_knobs(plan, state(15, level=8))
    assert done["level_to"] == 7 and not done["xp_priority"]  # reached: the field is inert


def test_level_by_relative_rounds():
    knobs = compile_knobs({**BASE, "level_by": {"level": 9, "rounds": 2}}, state(20, level=7))
    # 56 + 80 xp, 2 passive -> 34 buys over this round and the next: 34 - 10 now
    assert knobs["xp_buys"] == 34 - XP_BUY_CAP


def test_spend_spreads_the_roll_down():
    plan = {**BASE, "spend": {"to": 10, "by": "4-3"}}  # idx 17
    assert compile_knobs(plan, state(15, gold=60))["roll_floor"] == 60 - 17  # 50 over 3 rounds
    assert compile_knobs(plan, state(16, gold=52))["roll_floor"] == 52 - 21
    assert compile_knobs(plan, state(17, gold=39))["roll_floor"] == 10
    assert compile_knobs(plan, state(18, gold=39))["roll_floor"] == 999  # expired
    assert compile_knobs(plan, state(15, gold=8))["roll_floor"] == 999  # already below
    front = compile_knobs({**BASE, "spend": {"to": 0, "rounds": 2}}, state(15, gold=90))
    assert front["roll_floor"] == 90 - 60  # 90 cannot be spent at 30 a round unless 60 goes now
    assert compile_knobs({**plan, "roll_floor": 5}, state(15, gold=60))["roll_floor"] == 5  # plan's own floor wins


def test_survival_override():
    plan = {**BASE, "level_to": 8, "roll_floor": 50, "fodder": True, "survival": 2,
            "spend": {"to": 40, "rounds": 3}, "level_by": {"level": 8, "rounds": 3}}
    calm = compile_knobs(plan, state(16, hp=60, level=6, xp=30))
    assert not calm["survival"] and calm["fodder"]
    knobs = compile_knobs(plan, state(16, hp=25, level=6, xp=30))  # 4-2 losses cost ~13.9: 2 to death
    assert knobs["survival"] and not knobs["fodder"] and knobs["roll_floor"] == 0 and knobs["xp_buys"] == 0
    assert knobs["level_to"] == 7  # 6 xp short: two buys, cheap enough
    assert compile_knobs(plan, state(16, hp=25, level=6, xp=0))["level_to"] == 6  # 36 xp short: roll instead
    assert not compile_knobs({**plan, "survival": None}, state(16, hp=5))["survival"]


def test_param_planner_defaults_add_no_fields():
    plan = ParamPlanner().plan({**state(12), "xp_needed": 20}, {}, None)
    assert set(plan) == {"comp", "level_to", "roll_floor", "carry"}
    fodder = ParamPlanner(fodder=("2-1", "2-6"))
    assert [fodder.plan({**state(i), "xp_needed": 20}, {}, None)["fodder"] for i in (2, 3, 7, 8)] == \
        [False, True, True, False]


def test_llm_parse_accepts_the_new_fields():
    text = ('{"comp": null, "level_to": 7, "roll_floor": 30, "carry": null, "fodder": true, "survival": 2,'
            ' "level_by": {"level": 8, "by": "4-1"}, "spend": {"to": 20, "rounds": 2}}')
    plan = LLMPlanner.parse(text, {"board": []}, {})
    assert plan["fodder"] is True and plan["survival"] == 2
    assert plan["level_by"] == {"level": 8, "by": "4-1"} and plan["spend"] == {"to": 20, "rounds": 2}
    bad = LLMPlanner.parse('{"level_to": 7, "roll_floor": 30, "level_by": {"level": 8, "by": "9-9"}}',
                           {"board": []}, {})
    assert bad is not None and "level_by" not in bad


# --------------------------------------------------------------------------- executor on a real player

sim = pytest.importorskip("Simulator")

from Simulator.battle.champion import champion  # noqa: E402
from Simulator.encoding.token.action import ActionToken  # noqa: E402
from Simulator.game import pool  # noqa: E402
from Simulator.game.player import Player  # noqa: E402
from Simulator.utils import decode_action  # noqa: E402

from tfteval.planner import _make_executor  # noqa: E402

PlanExecutor, COMPS, TRAITS = _make_executor()


def make_player(board=(), bench=(), gold=10, level=3, hp=100):
    p = Player(pool.pool(), 0)
    p.gold, p.level, p.max_units, p.health = gold, level, level, hp
    for i, name in enumerate(board):
        champ = champion(name)
        assert p.add_to_bench(champ)
        assert p.move_bench_to_board(p.bench.index(champ), i, 0)
    for name in bench:
        assert p.add_to_bench(champion(name))
    set_shop(p, [None] * 5)
    return p


def set_shop(p, names):
    p.shop = list(names)
    p.shop_champions = p.create_shop_champions()


def perform(p, command):
    kind, x1, x2 = decode_action([command])[0]
    {0: lambda: None, 1: p.buy_exp_action, 2: p.refresh_shop_action, 3: lambda: p.buy_shop_action(x1),
     4: lambda: p.sell_action(x1), 5: lambda: p.move_champ_action(x1, x2),
     6: lambda: p.move_item_action(x2, x1)}[int(kind)]()
    p.actions_remaining -= 1


def run_round(ex, p, idx, plan, actions=15):
    if ex.current_round == 0:  # join mid-game the way the rule bot would have reached this round
        ex.next_round = idx
    ex.begin_round(plan, compile_knobs(plan, {"round": idx, "gold": p.gold, "level": p.level, "xp": p.exp,
                                              "hp": p.health}), idx)
    p.actions_remaining = actions
    commands = []
    for _ in range(actions):
        mask = np.asarray(ActionToken(p).fetch_action_mask()).reshape(55, 38)
        command = ex.act(p, p.shop, idx, mask)
        perform(p, command)
        commands.append(command)
    return commands


def board_names(p):
    return sorted(u.name for col in p.board for u in col if u)


def bench_names(p):
    return sorted(u.name for u in p.bench if u)


STRONG, WEAK = ["ahri", "jinx", "annie"], ["nami", "vayne", "fiora"]  # costs 4, 3, 2 / 1, 1, 1


def test_fodder_fields_the_weakest_and_reverts():
    p = make_player(board=STRONG, bench=WEAK)
    ex = PlanExecutor()
    run_round(ex, p, 5, {**BASE, "level_to": 3, "fodder": True})
    assert board_names(p) == sorted(WEAK) and bench_names(p) == sorted(STRONG)
    p.end_turn_actions()  # autofill only tops up to max_units: nothing moves
    assert board_names(p) == sorted(WEAK)
    run_round(ex, p, 6, {**BASE, "level_to": 3})
    assert board_names(p) == sorted(STRONG) and ex.stats["restore_moves"] == 3


def test_fodder_buys_one_costs_when_the_bench_has_no_weak_units():
    p = make_player(board=STRONG, gold=6)
    ex = PlanExecutor()
    for idx, name in zip((4, 5, 6), WEAK):
        set_shop(p, [name, "ahri", "jinx", "ahri", "jinx"])
        commands = run_round(ex, p, idx, {**BASE, "level_to": 3, "fodder": True})
        assert commands[0] == "3_0"  # one buy per shop: the buy mask closes once a slot is empty
    assert ex.stats["fodder_buys"] == 3
    assert board_names(p) == sorted(WEAK) and bench_names(p) == sorted(STRONG)
    assert p.gold == 3  # the three 1-costs still sell for 3


def test_fodder_keeps_the_interest_bracket():
    for gold, buys in ((20, 0), (21, 1)):
        p = make_player(board=STRONG, gold=gold)
        set_shop(p, ["vayne", "fiora", "nami", "fiora", "nami"])
        ex = PlanExecutor()
        run_round(ex, p, 4, {**BASE, "level_to": 3, "fodder": True})
        assert ex.stats["fodder_buys"] == buys  # (the rule bot itself may still buy)


def test_fodder_filter_blocks_rule_bot_undoing_it():
    p = make_player(board=WEAK, bench=STRONG)
    ex = PlanExecutor()
    ex.begin_round({**BASE, "fodder": True}, compile_knobs({**BASE, "fodder": True}, state(5)))
    kept = 28 + [u.name for u in p.bench if u].index("ahri")
    assert ex.fodder_filter("4_" + str(kept), p) == "0"  # selling a kept unit
    assert ex.fodder_filter("5_28_3", p) == "0"  # fielding a benched unit
    assert ex.fodder_filter("6_0_1", p) == "0"  # items on a fodder unit
    assert ex.fodder_filter("5_0_4", p) == "5_0_4"  # board to board is fine
    assert ex.fodder_filter("3_2", p) == "3_2"


def test_pivot_sells_off_comp_bench_units_and_keeps_playing():
    mage, divine = COMPS["mage"], COMPS["divine"]
    p = make_player(board=mage[:4], bench=["fiora", "jax", "lux", "nami", "nami"], gold=20, level=4)
    ex = PlanExecutor()
    ex.round_11_clean_up = False
    run_round(ex, p, 13, {**BASE, "level_to": 4, "comp": "mage"})
    assert ex.comp_number == TRAITS.index("mage") and ex.stats["pivots"] == 0
    ex.update_pairs_list(p)
    run_round(ex, p, 14, {**BASE, "level_to": 4, "comp": "divine"})
    assert ex.comp_number == TRAITS.index("divine") and ex.stats["pivots"] == 1
    left = bench_names(p) + board_names(p)
    assert "fiora" not in left  # off-comp, not a pair: sold
    assert left.count("nami") == 2  # a pair is kept, as decide_comp does
    assert {"jax", "lux"} <= set(left) and set(divine) & set(board_names(p))  # divine units fielded
    assert not ex.pivot_pending


def test_level_by_preempts_the_rule_bot_on_the_deadline_round():
    p = make_player(board=COMPS["mage"][:7], gold=70, level=7)
    set_shop(p, COMPS["mage"][:5])  # the rule bot wants to buy every one of these
    ex = PlanExecutor()
    ex.comp_number, ex.round_11_clean_up = TRAITS.index("mage"), False
    commands = run_round(ex, p, 15, {**BASE, "level_to": 7, "level_by": {"level": 8, "by": "4-1"}})
    assert p.level == 8 and commands.count("1") == 14 and ex.stats["xp_preempt"] > 0


def test_without_level_by_the_same_round_does_not_reach_8():
    p = make_player(board=COMPS["mage"][:7], gold=70, level=7)
    set_shop(p, COMPS["mage"][:5])
    ex = PlanExecutor()
    ex.comp_number, ex.round_11_clean_up = TRAITS.index("mage"), False
    run_round(ex, p, 15, {**BASE, "level_to": 8})
    assert p.level == 7  # buying units first leaves too few actions: why level_by schedules ahead


def test_survival_rolls_down():
    p = make_player(board=COMPS["mage"][:5], gold=30, level=5, hp=12)
    ex = PlanExecutor()
    ex.comp_number, ex.round_11_clean_up = TRAITS.index("mage"), False
    commands = run_round(ex, p, 16, {**BASE, "level_to": 5, "roll_floor": 50, "survival": 2})
    assert ex.stats["survival_rounds"] == 1 and "2" in commands and p.gold < 10

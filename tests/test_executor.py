"""Executor capabilities: plan compilation, comp pivot, multi-round budgets, fodder board, survival.

The executor tests run on the simulator fork (README, "对模拟器的修正"). Two of its fixes matter here:
the rule bot's bench-to-board swap check works (on upstream it never fired, so the rule bot only
filled empty board slots), and a shop slot stays buyable after another one was bought (on upstream
the buy mask closed after one purchase per refresh).
"""

from contextlib import contextmanager

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
                     "xp_priority": False, "fodder": False, "field_comp": False, "survival": False, "hold": False}


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
            ' "field_comp": true, "level_by": {"level": 8, "by": "4-1"}, "spend": {"to": 20, "rounds": 2}}')
    plan = LLMPlanner.parse(text, {"board": []}, {})
    assert plan["fodder"] is True and plan["survival"] == 2 and plan["field_comp"] is True
    assert plan["level_by"] == {"level": 8, "by": "4-1"} and plan["spend"] == {"to": 20, "rounds": 2}
    bad = LLMPlanner.parse('{"level_to": 7, "roll_floor": 30, "level_by": {"level": 8, "by": "9-9"}}',
                           {"board": []}, {})
    assert bad is not None and "level_by" not in bad


# --------------------------------------------------------------------------- executor on a real player

sim = pytest.importorskip("Simulator")

from Simulator.battle.champion import champion  # noqa: E402
from Simulator.battle.combat_context import CombatContext  # noqa: E402
from Simulator.encoding.token.action import ActionToken  # noqa: E402
from Simulator.game import pool  # noqa: E402
from Simulator.game.player import Player  # noqa: E402
from Simulator.rng import EnvRNG  # noqa: E402
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


def mask_of(p):
    return np.asarray(ActionToken(p).fetch_action_mask()).reshape(55, 38)


@contextmanager
def seeded(seed):
    """Fixed shop draws: the pool draws from the bound context's generator (a random one otherwise)."""
    with CombatContext(rng=EnvRNG.from_episode_seed(seed)).bind():
        yield


def run_round(ex, p, idx, plan, actions=15, golds=None):
    if ex.current_round == 0:  # join mid-game the way the rule bot would have reached this round
        ex.next_round = idx
    ex.begin_round(plan, compile_knobs(plan, {"round": idx, "gold": p.gold, "level": p.level, "xp": p.exp,
                                              "hp": p.health}), idx)
    p.actions_remaining = actions
    commands = []
    for _ in range(actions):
        if golds is not None:
            golds.append(p.gold)
        command = ex.act(p, p.shop, idx, mask_of(p))
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
    # the rule bot's swap check would field a strong unit again: dropped once, then done for the round
    assert ex.stats["swaps_dropped"] == 1
    p.end_turn_actions()  # autofill only tops up to max_units: nothing moves
    assert board_names(p) == sorted(WEAK)
    run_round(ex, p, 6, {**BASE, "level_to": 3})
    assert ex.stats["restore_moves"] == 3  # the strongest units go back on ...
    plain = make_player(board=STRONG, bench=WEAK)
    run_round(PlanExecutor(), plain, 6, {**BASE, "level_to": 3})
    # ... and the rule bot plays on as a seat that never fielded fodder: its swap check prefers a
    # third mage (nami) to jinx
    assert board_names(p) == board_names(plain) == ["ahri", "annie", "nami"]


def owned(p):
    return sorted(board_names(p) + bench_names(p))


def weakest_fielded(p):
    from tfteval.executor import strength

    units = [u for col in p.board for u in col if u] + [u for u in p.bench if u]
    fielded = sorted(strength(u) for col in p.board for u in col if u)
    return fielded == sorted(strength(u) for u in units)[:len(fielded)]


# 1-costs with no trait in common with STRONG. The rule bot decides on the strong view, which fields
# the strongest units by strength(); on the simulator fork the base economy's own swap check fields by
# its comp score, so a weak unit that completes a trait (nami: a third mage, vayne: a second
# sharpshooter) goes onto the base board and the two boards, and the decisions after, part ways.
WEAK_NO_TRAIT = ["fiora", "garen", "maokai"]


@pytest.mark.parametrize("board,bench,shop,gold,items", [
    # the base economy buys a second jinx, fields it in place of annie and sells annie and a 1-cost to
    # get back to 10 gold (the rest of the shop stays buyable on the fork, nothing else is worth it):
    # in fodder mode the units sold stand on the fodder board, so they are sold from there
    (STRONG, WEAK_NO_TRAIT, ["jinx", "garen", "fiora", "maokai", "garen"], 10, ()),
    (STRONG, WEAK_NO_TRAIT, ["jinx", "garen", "fiora", "maokai", "garen"], 10, ("bf_sword",)),  # item left unplaced
    # nothing the base economy wants: a rule bot looking at the fodder board would buy the pair of
    # the fodder fiora
    (STRONG, WEAK, ["fiora"] * 5, 20, ()),
    # no weak units at all: the fodder board is the weakest of what is owned, nothing is bought for it
    (STRONG, (), ["vayne", "fiora", "nami", "fiora", "nami"], 6, ()),
])
def test_fodder_plays_the_base_economy(board, bench, shop, gold, items):
    def play(fodder):
        p = make_player(board=board, bench=bench, gold=gold)
        for i, item in enumerate(items):
            p.item_bench[i] = item
        set_shop(p, shop)
        ex = PlanExecutor()
        commands = run_round(ex, p, 4, {**BASE, "level_to": 3, "fodder": fodder})
        return p, ex, [c for c in commands if c.startswith("3_")]

    base, _, base_buys = play(False)
    p, ex, buys = play(True)
    assert buys == base_buys and p.gold == base.gold and owned(p) == owned(base)
    assert weakest_fielded(p) and ex.stats["fodder_rounds"] == 1 and "fodder_buys" not in ex.stats
    assert [i for i in p.item_bench if i] == list(items)  # no items on a fodder board
    if items:
        assert not [i for i in base.item_bench if i]


def test_fodder_board_without_weak_units_stays_as_it_is():
    p = make_player(board=STRONG, gold=0)
    set_shop(p, ["vayne", "fiora", "nami", "fiora", "nami"])
    ex = PlanExecutor()
    commands = run_round(ex, p, 4, {**BASE, "level_to": 3, "fodder": True})
    assert not [c for c in commands if c[0] in "1346"]  # no buys, sales, xp or items (moves are fine)
    assert board_names(p) == sorted(STRONG) and p.gold == 0 and ex.stats["fodder_moves"] == 0


def test_strong_view_and_translation():
    p = make_player(board=WEAK, bench=STRONG)
    ex = PlanExecutor()
    view, swaps = ex.strong_view(p)
    assert sorted(u.name for col in view.board for u in col if u) == sorted(STRONG)
    assert sorted(u.name for u in view.bench if u) == sorted(WEAK) and not view.bench_full()
    assert view.gold == p.gold and view.num_units_in_play == 3  # the rest comes from the player
    assert len(swaps) == 6 and all(swaps[swaps[loc]] == loc for loc in swaps)
    assert board_names(p) == sorted(WEAK)  # the player itself is untouched
    slot = 28 + next(i for i, u in enumerate(view.bench) if u and u.name == "nami")
    cell = swaps[slot]  # where nami really stands: on the fodder board
    x, y = cell // 4, cell % 4
    assert p.board[x][y].name == "nami" and ex.translate("4_" + str(slot), swaps) == "4_" + str(cell)
    assert ex.translate("3_2", swaps) == "3_2" and ex.translate("1", swaps) == "1"


def test_fodder_filter():
    p = make_player(board=WEAK, bench=STRONG)
    ex = PlanExecutor()
    ex.begin_round({**BASE, "fodder": True}, compile_knobs({**BASE, "fodder": True}, state(5)))
    p.actions_remaining = 5
    assert ex.fodder_filter("5_28_3", p) == "0"  # fielding a benched unit
    assert ex.fodder_filter("5_3_28", p) == "0"  # benching a fielded one
    assert ex.fodder_filter("6_0_1", p) == "0"  # items on a fodder unit
    assert ex.fodder_filter("5_0_4", p) == "5_0_4"  # board to board is fine
    assert ex.fodder_filter("3_2", p) == "3_2"
    assert ex.fodder_filter("4_0", p) == "4_0" and ex.fodder_filter("4_28", p) == "4_28"
    p.actions_remaining = 1  # a board sale on the last action would let autofill field a strong unit
    assert ex.fodder_filter("4_0", p) == "0" and ex.fodder_filter("4_28", p) == "4_28"


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


def test_field_comp_swaps_comp_units_onto_a_full_board():
    mage = COMPS["mage"]
    off, bench = ["fiora", "vayne", "jax"], ["ahri", "veigar", "thresh"]  # costs 1, 1, 2 / 4, 3, 2
    assert not set(off) & set(mage) and set(bench) <= set(mage)

    def setup():
        p = make_player(board=off, bench=bench, gold=0, level=3)
        ex = PlanExecutor()
        ex.comp_number, ex.round_11_clean_up = TRAITS.index("mage"), False
        return p, ex

    p, ex = setup()
    run_round(ex, p, 13, {**BASE, "level_to": 3, "comp": "mage"})
    # the rule bot's own swap check (live in the simulator fork) fields comp units too ...
    assert board_names(p) == sorted(bench) and ex.stats["comp_swaps"] == 0
    p, ex = setup()
    run_round(ex, p, 13, {**BASE, "level_to": 3, "comp": "mage", "field_comp": True})
    # ... with field_comp the executor makes those swaps, strongest comp unit first
    assert board_names(p) == sorted(bench) and ex.stats["comp_swaps"] == 3


def test_field_comp_does_not_field_a_weaker_comp_unit():
    boards = {}
    for field_comp in (False, True):
        p = make_player(board=["jax"], bench=["nami"], gold=0, level=1)  # off-comp 2-cost / mage 1-cost
        ex = PlanExecutor()
        ex.comp_number, ex.round_11_clean_up = TRAITS.index("mage"), False
        run_round(ex, p, 13, {**BASE, "level_to": 1, "comp": "mage", "field_comp": field_comp})
        boards[field_comp] = board_names(p)
        assert ex.stats["comp_swaps"] == 0 and ex.stats["swaps_dropped"] == int(field_comp)
    # the rule bot puts any comp unit in for an off-comp one; field_comp owns those swaps and drops it
    assert boards == {False: ["nami"], True: ["jax"]}


def test_owns_swap():
    ex = PlanExecutor()
    ex.comp_number = TRAITS.index("mage")
    p = make_player(board=["jax", "ahri"], bench=["nami", "fiora"], level=2)  # jax at 0, ahri at 4
    nami, fiora = 28 + [u.name for u in p.bench if u].index("nami"), 28 + [u.name for u in p.bench if u].index("fiora")
    for plan, idx, command, owned in (
            ({**BASE}, 13, f"5_0_{nami}", False),  # no knob: the rule bot's swap goes through
            ({**BASE, "fodder": True}, 5, f"5_0_{nami}", True),
            ({**BASE, "hold": True}, 5, f"5_4_{fiora}", True),
            ({**BASE, "hold": True}, 5, "5_4_8", False),  # board to board
            ({**BASE, "hold": True}, 5, f"5_{nami}_9", False),  # a bench unit into an empty slot
            ({**BASE, "field_comp": True}, 13, f"5_0_{nami}", True),  # comp unit in for an off-comp one
            ({**BASE, "field_comp": True}, 13, f"5_4_{fiora}", False),  # takes a comp unit off: not field_comp's
            ({**BASE, "field_comp": True}, 13, f"5_0_{fiora}", False),  # off-comp for off-comp
            ({**BASE, "field_comp": True}, 9, f"5_0_{nami}", False)):  # before round 11 there is no comp
        ex.begin_round(plan, compile_knobs(plan, state(idx)))
        assert ex.owns_swap(command, p, idx) is owned, (plan, command)


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
    for seed in range(4):  # the shops the rolls find
        for survival in (None, 2):
            p = make_player(board=COMPS["mage"][:5], gold=30, level=5, hp=12)
            ex = PlanExecutor()
            ex.comp_number, ex.round_11_clean_up = TRAITS.index("mage"), False
            golds = []
            with seeded(seed):
                commands = run_round(ex, p, 16, {**BASE, "level_to": 5, "roll_floor": 50, "survival": survival},
                                     golds=golds)
            if survival is None:  # the plan's own floor: no roll at all
                assert "2" not in commands and p.gold == 30
                continue
            assert ex.stats["survival_rounds"] == 1 and "2" in commands
            # every action the rule bot leaves free is a roll while 2 gold are left. The rule bot's
            # buys (several per shop) and swaps take actions too, so 15 actions do not always reach
            # 0 gold (on upstream, one buy per shop left more actions for rolls: gold < 10 there).
            assert all(g < 2 for c, g in zip(commands, golds) if c == "0")
            assert p.gold <= 15


def test_overlay_planner_adds_fields_inside_windows():
    from tfteval.planner import OverlayPlanner

    planner = OverlayPlanner(ParamPlanner(), [{"from": "3-3", "to": "3-7", "fields": {"comp": "mage"}},
                                              {"from": "4-1", "fields": {"comp": "divine", "survival": 2}}])
    comps = [planner.plan({**state(i), "xp_needed": 20}, {}, None).get("comp") for i in (10, 11, 14, 15, 30)]
    assert comps == [None, "mage", "mage", "divine", "divine"]
    assert "survival" not in planner.plan({**state(12), "xp_needed": 20}, {}, None)

"""Executor knob `hold` and plan field `xp_buys` (tfteval/executor.py, tfteval/planner.py).

`hold` keeps units only: the rule bot's bench sales are dropped while the bench has room (its sales
for interest), and everything else is the rule bot's own play: it fields its own trait-aware board
(its swap check by comp score) and places items. Without the field the executor plays as before
(tests/test_regression.py). A dropped sale marks the rule bot's check done for the round, so it does
not propose the same sale on every action.
"""

import pytest

pytest.importorskip("Simulator")

from test_executor import BASE, COMPS, STRONG, TRAITS, WEAK, PlanExecutor, bench_names, board_names, make_player, run_round, set_shop, state  # noqa: E402,E501

from tfteval.planner import compile_knobs  # noqa: E402

HOLD = {**BASE, "level_to": 3, "hold": True}
PLAIN = {**BASE, "level_to": 3}


def test_hold_fields_the_rule_bots_own_board():
    # the rule bot's swap check fields a third mage (nami) rather than jinx, with or without hold
    boards = {}
    for name, plan in (("plain", PLAIN), ("hold", HOLD)):
        p = make_player(board=WEAK, bench=STRONG)
        ex = PlanExecutor()
        run_round(ex, p, 5, plan)
        boards[name] = board_names(p)
        assert ex.stats["swaps_dropped"] == 0 and "hold_moves" not in ex.stats
    assert boards == {"plain": ["ahri", "annie", "nami"], "hold": ["ahri", "annie", "nami"]}


def test_hold_keeps_the_units_the_rule_bot_sells_for_interest():
    sold, ex = {}, None
    for name, plan in (("plain", PLAIN), ("hold", HOLD)):
        p = make_player(board=STRONG, bench=WEAK, gold=18)  # selling two 1-costs reaches 20 gold
        ex = PlanExecutor()
        run_round(ex, p, 5, plan)
        sold[name] = 6 - len(bench_names(p) + board_names(p))
        assert board_names(p) == ["ahri", "annie", "nami"]  # the rule bot's own board either way
    assert sold == {"plain": 2, "hold": 0}
    # the sale was dropped once and its check marked done, not proposed again on every action
    assert ex.stats["hold_filtered"] == 1 and ex.round_3_10_checks[4] is False


def test_hold_places_items():
    placed = {}
    for name, plan in (("plain", PLAIN), ("hold", HOLD)):
        p = make_player(board=STRONG, bench=WEAK)
        p.item_bench[0] = "bf_sword"
        run_round(PlanExecutor(), p, 5, plan)
        placed[name] = p.item_bench[0] is None
    assert placed == {"plain": True, "hold": True}


def test_hold_sells_to_make_room_on_a_full_bench():
    bench = ["nami", "vayne", "fiora", "garen", "diana", "elise", "lissandra", "maokai", "nidalee"]
    p = make_player(board=STRONG, bench=bench, gold=0)
    set_shop(p, ["ahri", None, None, None, None])  # a pair for the board: the rule bot wants it
    assert p.bench_full()
    ex = PlanExecutor()
    commands = run_round(ex, p, 5, HOLD)
    assert commands[0].startswith("4_2") and not p.bench_full()  # sell_bench_full goes through


def test_hold_drops_round_11_clean_up_sales_once():
    mage = COMPS["mage"]
    p = make_player(board=mage[:3], bench=["fiora", "garen", "jax"], gold=0, level=3)
    ex = PlanExecutor()
    commands = run_round(ex, p, 11, {**HOLD, "comp": "mage"})
    assert ex.comp_number == TRAITS.index("mage")  # decide_comp still picks the comp ...
    assert bench_names(p) == ["fiora", "garen", "jax"] and not [c for c in commands if c.startswith("4_")]
    # ... its clean-up sales are dropped once, and the rule bot moves on to round_11_end
    assert ex.stats["hold_filtered"] == 1 and ex.round_11_clean_up is False


def test_hold_filter():
    p = make_player(board=STRONG, bench=WEAK)
    assert PlanExecutor.hold_filter("4_28", p) == "0"  # bench sale with room on the bench
    assert PlanExecutor.hold_filter("4_3", p) == "4_3"  # a board sale is the rule bot's clean-up: kept
    assert PlanExecutor.hold_filter("6_0_1", p) == "6_0_1"  # items go on
    assert PlanExecutor.hold_filter("3_2", p) == "3_2" and PlanExecutor.hold_filter("5_3_28", p) == "5_3_28"
    full = make_player(board=STRONG, bench=["nami", "vayne", "fiora", "garen", "diana", "elise", "lissandra",
                                            "maokai", "nidalee"])
    assert full.bench_full() and PlanExecutor.hold_filter("4_28", full) == "4_28"  # making room


def test_dropped_swap_marks_the_check_done():
    # field_comp owns a comp-for-off-comp swap; the rule bot is asked again and its later checks run
    p = make_player(board=["jax"], bench=["nami"], gold=0, level=1)  # off-comp 2-cost / mage 1-cost
    p.item_bench[0] = "bf_sword"
    ex = PlanExecutor()
    ex.comp_number, ex.round_11_clean_up = TRAITS.index("mage"), False
    commands = run_round(ex, p, 13, {**BASE, "level_to": 1, "comp": "mage", "field_comp": True})
    assert board_names(p) == ["jax"] and ex.stats["swaps_dropped"] == 1
    assert ex.round_11_end_checks[2] is False and p.item_bench[0] is None  # the item still went on
    assert commands.count("0") >= 12


def test_xp_buys_plan_field():
    knobs = compile_knobs({**BASE, "xp_buys": 3}, state(12, gold=62, level=7))
    assert knobs["xp_buys"] == 3 and knobs["level_to"] == 5 and not knobs["xp_priority"]
    deadline = {**BASE, "level_to": 7, "level_by": {"level": 8, "by": "4-1"}}
    assert compile_knobs(deadline, state(14, gold=62, level=7))["xp_buys"] == 4  # (56 - 2) / 4 = 14, 10 next round
    for extra, want in ((3, 4), (6, 6)):  # the larger of the two
        knobs = compile_knobs({**deadline, "xp_buys": extra}, state(14, gold=62, level=7))
        assert knobs["xp_buys"] == want and knobs["xp_priority"]
    p = make_player(board=STRONG, gold=30, level=3)
    p.exp = 0
    ex = PlanExecutor()
    run_round(ex, p, 5, {**BASE, "level_to": 3, "xp_buys": 2})
    assert ex.stats["xp"] == 2 and p.gold == 22

"""Executor knob `hold` and plan field `xp_buys` (tfteval/executor.py, tfteval/planner.py).

`hold` keeps the fodder board's filters (no bench sales while the bench has room, no item placement)
but fields the strongest units. Without the field the executor plays as before (tests/test_regression.py).
"""

import pytest

pytest.importorskip("Simulator")

from test_executor import BASE, STRONG, WEAK, PlanExecutor, bench_names, board_names, make_player, run_round, state  # noqa: E402,E501

from tfteval.planner import compile_knobs  # noqa: E402

HOLD = {**BASE, "level_to": 3, "hold": True}
PLAIN = {**BASE, "level_to": 3}


def test_hold_fields_the_strongest_units():
    for plan, board in ((HOLD, STRONG), (PLAIN, WEAK)):  # the rule bot only fills empty slots
        p = make_player(board=WEAK, bench=STRONG)
        ex = PlanExecutor()
        run_round(ex, p, 5, plan)
        assert board_names(p) == sorted(board)
    assert ex.stats["hold_moves"] == 0


def test_hold_keeps_the_units_the_rule_bot_sells_for_interest():
    sold = {}
    for name, plan in (("plain", PLAIN), ("hold", HOLD)):
        p = make_player(board=STRONG, bench=WEAK, gold=18)  # selling two 1-costs reaches 20 gold
        ex = PlanExecutor()
        run_round(ex, p, 5, plan)
        sold[name] = 3 - len(bench_names(p))
        if name == "hold":
            assert ex.stats["hold_filtered"] >= 1 and board_names(p) == sorted(STRONG)
    assert sold == {"plain": 2, "hold": 0}


def test_hold_places_no_items():
    placed = {}
    for name, plan in (("plain", PLAIN), ("hold", HOLD)):
        p = make_player(board=STRONG, bench=WEAK)
        p.item_bench[0] = "bf_sword"
        run_round(PlanExecutor(), p, 5, plan)
        placed[name] = p.item_bench[0] is None
    assert placed == {"plain": True, "hold": False}


def test_hold_filter():
    p = make_player(board=STRONG, bench=WEAK)
    assert PlanExecutor.hold_filter("6_0_1", p) == "0"  # item placement
    assert PlanExecutor.hold_filter("4_28", p) == "0"  # bench sale with room on the bench
    assert PlanExecutor.hold_filter("4_3", p) == "4_3"  # a board sale is the rule bot's clean-up: kept
    assert PlanExecutor.hold_filter("3_2", p) == "3_2" and PlanExecutor.hold_filter("5_28_3", p) == "5_28_3"
    full = make_player(board=STRONG, bench=["nami", "vayne", "fiora", "garen", "diana", "elise", "lissandra",
                                            "maokai", "nidalee"])
    assert full.bench_full() and PlanExecutor.hold_filter("4_28", full) == "4_28"  # making room


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

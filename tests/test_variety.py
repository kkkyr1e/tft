"""Strategic variety metrics (tfteval/variety.py) on toy data. No games, no simulator."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval.variety import (board_comp, comp_distribution, early_signals, entropy, final_comp,  # noqa: E402
                             mutual_information, noninferiority, responsiveness)

COMPS = {"mage": ["ahri", "annie", "lulu"], "divine": ["irelia", "wukong", "lux"], "elderwood": ["ashe", "lulu"]}


def test_entropy():
    assert entropy([5]) == 0.0
    assert math.isclose(entropy([3, 3]), 1.0)
    assert math.isclose(entropy({"a": 1, "b": 1, "c": 1, "d": 1}), 2.0)
    assert entropy([]) == 0.0 and entropy([0, 4]) == 0.0


def test_comp_distribution():
    out = comp_distribution(["mage", "mage", "divine", None, "elderwood"], universe=4)
    assert out["n"] == 4 and out["distinct"] == 3 and out["counts"] == {"mage": 2, "divine": 1, "elderwood": 1}
    assert math.isclose(out["entropy_bits"], 1.5) and math.isclose(out["max_bits"], 2.0)
    assert math.isclose(out["normalized"], 0.75) and math.isclose(out["effective_comps"], 2 ** 1.5)
    one = comp_distribution(["mage"] * 5, universe=13)
    assert one["entropy_bits"] == 0.0 and one["effective_comps"] == 1.0


def test_mutual_information():
    xs = ["a", "b", "c", "d"] * 10
    same = mutual_information(xs, [x.upper() for x in xs], permutations=200)
    assert math.isclose(same["mi_bits"], 2.0) and math.isclose(same["normalized"], 1.0)
    assert math.isclose(same["mi_miller_madow"], 2.0 + (4 + 4 - 4 - 1) / (2 * 40 * math.log(2)))
    assert same["p_value"] < 0.01 and same["null_mean"] < 0.5
    indep = mutual_information(["a", "b"] * 20, ["x"] * 20 + ["y"] * 20, permutations=200)
    assert math.isclose(indep["mi_bits"], 0.0, abs_tol=1e-12) and indep["p_value"] > 0.5
    # a 2 x 2 table that fills all four cells: Miller-Madow lowers the estimate (the plug-in's upward bias)
    noise = mutual_information(["a", "b", "a", "a", "b"] * 6, ["x", "x", "y", "x", "y"] * 6, permutations=0)
    assert noise["mi_miller_madow"] < noise["mi_bits"] and noise["p_value"] is None
    # deterministic: the null uses a fixed seed
    assert mutual_information(xs, xs[::-1], permutations=50) == mutual_information(xs, xs[::-1], permutations=50)
    assert mutual_information([], [])["mi_bits"] is None


def row(rnd, **extra):
    return {"round": rnd, **extra}


def test_early_signals_and_final_comp():
    r = row(9, board=[["ahri", 1, 4, ["bf_sword", "infinity_edge"]], ["ashe", 1, 2, [], "elderwood"]],
            bench=[["lux", 1, 3, ["chain_vest"]]], items=["chain_vest", "tear_of_the_goddess"],
            contested={"mage": 4, "divine": 9})
    assert early_signals([row(8), r], "3-1") == {"item": "chain_vest", "chosen": "elderwood", "contested": "divine"}
    assert early_signals([row(8)], "3-1") is None
    bare = row(9, board=[], bench=[], items=[], contested={})
    assert early_signals([bare]) == {"item": "none", "chosen": "none", "contested": "none"}
    # ties: the first name wins
    tie = row(9, board=[], items=["tear_of_the_goddess", "bf_sword"], contested={"mage": 2, "divine": 2})
    assert early_signals([tie]) == {"item": "bf_sword", "chosen": "none", "contested": "divine"}

    assert final_comp([row(20, comp="mage", board=[]), row(21, comp="divine", board=[])], COMPS) == "divine"
    rule_less = [row(21, board=[["irelia", 2, 3], ["wukong", 1, 1], ["ahri", 1, 4]])]
    assert final_comp(rule_less, COMPS) == "divine"
    assert board_comp([["ahri", 1, 4], ["ashe", 1, 2]], COMPS) == "mage"  # tie: first listed
    assert board_comp([["zed", 1, 4]], COMPS) is None and final_comp([], COMPS) is None


def test_responsiveness_over_games():
    games = [{"comp": c, "signals": {"item": "x", "chosen": "none", "contested": c}} for c in ["mage", "divine"] * 15]
    games.append({"comp": None, "signals": {"item": "x", "chosen": "none", "contested": "mage"}})  # dropped
    out = responsiveness(games, permutations=100)
    assert out["n"] == 30
    assert math.isclose(out["contested"]["mi_bits"], 1.0) and out["contested"]["p_value"] < 0.05
    assert out["item"]["mi_bits"] == 0.0 and out["chosen"]["mi_bits"] == 0.0


def test_noninferiority_margin():
    assert noninferiority({"mean": 0.1, "ci95": 0.1, "n": 300}, 0.25)["noninferior"] is True
    worse = noninferiority({"mean": 0.2, "ci95": 0.1, "n": 300}, 0.25)
    assert worse["noninferior"] is False and "not reported as a plus" in worse["note"]
    assert noninferiority({"mean": -0.5, "ci95": 0.75, "n": 300}, 0.25)["noninferior"] is True  # upper 0.25
    assert noninferiority(None, 0.25)["noninferior"] is None
    assert noninferiority({"mean": 0.0, "ci95": None, "n": 1}, 0.25)["noninferior"] is False

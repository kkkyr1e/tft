import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval.stats import games_needed, paired_diff, summarize  # noqa: E402


def game(seed, places, names=("a", "a", "a", "a", "b", "b", "b", "b"), finished=True):
    seats = [f"player_{i}" for i in range(8)]
    return {"seed": seed, "lobby": dict(zip(seats, names)), "placements": dict(zip(seats, places)), "finished": finished}


def test_summarize_perfect_split():
    results = [game(i, [1, 2, 3, 4, 5, 6, 7, 8]) for i in range(5)]
    a, b = summarize(results, "a"), summarize(results, "b")
    assert a["mean"] == 2.5 and b["mean"] == 6.5
    assert a["top4_rate"] == 1.0 and b["top4_rate"] == 0.0
    assert a["ci95"] == 0.0
    assert a["histogram"] == [5, 5, 5, 5, 0, 0, 0, 0]


def test_lobby_average_is_always_4_5():
    results = [game(0, [3, 8, 1, 6, 2, 7, 5, 4]), game(1, [8, 7, 6, 5, 4, 3, 2, 1])]
    assert math.isclose(summarize(results, "a")["mean"] + summarize(results, "b")["mean"], 9.0)


def test_unfinished_games_are_dropped():
    results = [game(0, [1, 2, 3, 4, 5, 6, 7, 8]), game(1, [8, 7, 6, 5, 4, 3, 2, 1], finished=False)]
    assert summarize(results, "a")["n"] == 1


def test_interval_uses_games_not_seats():
    results = [game(0, [1, 2, 3, 4, 5, 6, 7, 8]), game(1, [5, 6, 7, 8, 1, 2, 3, 4])]
    s = summarize(results, "a")
    assert s["n"] == 2
    assert math.isclose(s["ci95"], 1.959964 * math.sqrt(8) / math.sqrt(2), rel_tol=1e-4)


def test_paired_diff_cancels_shared_luck():
    one = ("x",) + ("o",) * 7
    a = [game(s, [p, 0, 0, 0, 0, 0, 0, 0], names=one) for s, p in enumerate([2, 5, 7, 3])]
    b = [game(s, [p + 1, 0, 0, 0, 0, 0, 0, 0], names=one) for s, p in enumerate([2, 5, 7, 3])]
    d = paired_diff(a, b, "x", "x")
    assert d["mean"] == -1.0 and d["ci95"] == 0.0
    assert d["unpaired_ci95"] > 2.0


def test_games_needed_matches_plan_figure():
    assert games_needed(sd_per_game=math.sqrt(63 / 12), half_width=0.2) == 505


if __name__ == "__main__":  # lets the suite run without pytest installed
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

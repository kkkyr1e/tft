"""Single-fight win probability: public features, the fitted model, the fitting helpers and the fight recorder."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Simulator")

from Simulator.battle.champion import champion  # noqa: E402
from Simulator.game import pool  # noqa: E402
from Simulator.game.player import Player  # noqa: E402

from tfteval import make_policy, public, winprob  # noqa: E402
from tfteval.fightlog import FightRecorder  # noqa: E402
from tfteval.planner import _unit  # noqa: E402
from tfteval.runner import Game  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fit_script():
    spec = importlib.util.spec_from_file_location("fit_winprob", ROOT / "scripts" / "fit_winprob.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def view(board, level=5, hp=70, streak=0):
    return {"board": list(board), "level": level, "hp": hp, "streak": streak}


# --------------------------------------------------------------------------- features and p_win

def test_features_are_the_same_from_tags_and_from_own_unit_dicts():
    p = Player(pool.pool(), 0)
    p.level = p.max_units = 5
    for i, (name, stars, items) in enumerate([("ahri", 2, ["bf_sword"]), ("annie", 1, []), ("lulu", 3, []),
                                              ("veigar", 1, ["chain_vest", "sparring_gloves"])]):
        champ = champion(name, stars=stars, itemlist=items)
        assert p.add_to_bench(champ)
        assert p.move_bench_to_board(p.bench.index(champ), i, 0)
    seen = public.opponent_view("player_0", p)  # what the others see: tags
    own = {"board": [_unit(u) for u in public.board_units(p)], "level": p.level, "hp": p.health, "streak": 0}
    assert winprob.view_features(seen) == winprob.view_features(own)
    f = winprob.view_features(seen)
    assert (f["units"], f["star2"], f["star3"], f["items"], f["level"]) == (4, 1, 1, 3, 5)
    assert f["score"] == f["value"] + f["traits"] + 2 * f["items"]


def test_p_win_is_a_probability_of_one_of_two_outcomes():
    even = view(["ahri*1", "annie*1", "lulu*1", "veigar*1"])
    assert winprob.p_win(even, even) == pytest.approx(0.5)
    strong = view(["ahri*2[bf_sword,chain_vest]", "annie*2", "lulu*2", "veigar*2", "jinx*1", "nami*1", "zed*1"],
                  level=7, hp=80)
    weak = view(["vayne*1", "fiora*1", "garen*1"], level=4, hp=40)
    for model in ("start", "fight"):
        p = winprob.p_win(strong, weak, model)
        assert p > 0.8 and p + winprob.p_win(weak, strong, model) == pytest.approx(1.0)
    one = winprob.load("start")
    fa, fb = winprob.view_features(strong), winprob.view_features(weak)
    assert one.p(fa, fb, side=1) + one.p(fb, fa, side=-1) == pytest.approx(1.0)
    assert one.p(fa, fb) == pytest.approx(0.5 * (one.p(fa, fb, side=1) + one.p(fa, fb, side=-1)))


def test_the_fitted_model_file():
    spec = json.loads(winprob.MODEL_PATH.read_text())
    assert spec["data"]["games"] >= 50 and spec["data"]["test_seeds"]
    names = set(winprob.derived(winprob.view_features(view(["ahri*1"]))))
    for when in ("start", "fight"):
        model = spec["models"][when]
        assert set(model["terms"]) <= names and set(model["coef"]) == set(model["terms"])
        held = model["held_out"]
        assert held["n"] > 500 and held["log_loss"] < 0.65 and held["auc"] > 0.7  # a coin flip is 0.693 / 0.5
        assert sum(c["n"] for c in model["calibration"]) == held["n"]


# --------------------------------------------------------------------------- fitting helpers

def test_auc_log_loss_and_split():
    fw = fit_script()
    y = np.array([0, 0, 1, 1.0])
    assert fw.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert fw.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert fw.auc(y, np.full(4, 0.5)) == 0.5
    assert fw.log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))
    train, test = fw.split_seeds(list(range(100, 160)))
    assert len(test) == 12 and not set(train) & set(test) and len(train) + len(test) == 60


def test_fit_recovers_known_coefficients():
    fw = fit_script()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 2))
    X = np.column_stack([np.ones(len(x)), x])
    w_true = np.array([0.3, 1.0, -0.5])
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-(X @ w_true)))).astype(float)
    w = fw.fit(X, y)
    assert np.allclose(w, w_true, atol=0.08)


# --------------------------------------------------------------------------- fight recorder

def partial_game(seed: int, record: bool, until: int = 11):
    lobby = {f"player_{i}": make_policy("rule") for i in range(8)}
    game = Game(lobby, seed)
    if not record:
        return game.run(until_round=until), None
    with FightRecorder({"seed": seed}) as rec:
        game.run(until_round=until)
    return game, rec


def test_recorder_reads_exact_results_and_changes_nothing():
    plain, _ = partial_game(7100, record=False)
    game, rec = partial_game(7100, record=True)
    assert game.health == plain.health and game.actions == plain.actions and game.round == plain.round == 11
    assert [r["round"] for r in rec.records] == [3, 4, 5, 6, 7, 9, 10]  # PvP rounds 2-1 .. 3-2 (2-7 is PvE)
    keys = {"seat", "hp", "level", "streak", "interest", "board"}
    checked = 0
    for r, nxt in zip(rec.records, rec.records[1:]):  # only PvE rounds (no HP lost) lie in between
        assert all(set(v) == keys for views in (r["start"], r["fight"]) for v in views.values())
        assert len(r["fights"]) == 4 and not any(f["ghost"] for f in r["fights"])  # 8 alive: four pairs
        for fight in r["fights"]:
            a, b, won, dmg = fight["a"], fight["b"], fight["won"], fight["damage"]
            hp = {s: r["fight"][s]["hp"] for s in (a, b)}
            new = {s: nxt["start"][s]["hp"] for s in (a, b)}
            if won == 1:
                assert new == {a: hp[a], b: hp[b] - dmg}
            elif won == 2:
                assert new == {a: hp[a] - dmg, b: hp[b]}
            else:
                assert new == {a: hp[a] - dmg, b: hp[b] - dmg}
            checked += 1
    assert checked == 24
    # the wrappers are gone once the recorder exits
    from Simulator.battle import champion as champion_module
    from Simulator.game import game_round

    assert game_round.Game_Round.combat_phase.__qualname__ == "Game_Round.combat_phase"
    assert champion_module.run.__module__ == "Simulator.battle.champion"

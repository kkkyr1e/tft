"""The public observation shows what a real player sees, and nothing hidden."""

import copy
import json

import pytest

pytest.importorskip("Simulator")

from Simulator.battle.champion import champion  # noqa: E402
from Simulator.game import pool  # noqa: E402
from Simulator.game.player import Player  # noqa: E402

from tfteval import public, stages  # noqa: E402
from tfteval.planner import describe  # noqa: E402


def unit(tag):
    name, _, stars = tag.partition("*")
    return champion(name, stars=int(stars or 1))


def make_player(num, board=(), bench=(), gold=0, level=4, hp=100):
    """A simulator Player with units placed through the simulator's own bench and move functions."""
    p = Player(pool.pool(), num)
    p.gold, p.level, p.max_units, p.health = gold, level, max(level, len(board)), hp
    for i, tag in enumerate(board):
        champ = unit(tag)
        assert p.add_to_bench(champ)
        assert p.move_bench_to_board(p.bench.index(champ), i, 0)
    for tag in bench:
        assert p.add_to_bench(unit(tag))
    return p


@pytest.fixture
def lobby():
    me = make_player(0, board=["ahri", "annie"], bench=["nami"], gold=23)
    rich = make_player(1, board=["ahri*2", "lulu"], bench=["veigar", "veigar"], gold=57, level=6, hp=80)
    poor = make_player(2, board=["vayne"], gold=4, level=3, hp=35)
    rich.win_streak, poor.loss_streak = 3, 4
    rich.board[0][0].items = ["bf_sword"]
    players = {"player_0": me, "player_1": rich, "player_2": poor}
    me.opponent_options = {"player_0": 1, "player_1": 1, "player_2": 0, "player_player_1": 0, "player_9": 1}
    return me, players


def test_opponent_view_has_only_public_fields(lobby):
    me, players = lobby
    view = public.public_view("player_0", me, players)
    rich = next(o for o in view["opponents"] if o["seat"] == "player_1")
    assert rich == {"seat": "player_1", "hp": 80, "level": 6, "streak": 3, "interest": 5,
                    "board": ["ahri*2[bf_sword]", "lulu*1"]}
    poor = next(o for o in view["opponents"] if o["seat"] == "player_2")
    assert poor["streak"] == -4 and poor["interest"] == 0
    assert [o["seat"] for o in view["opponents"]] == ["player_1", "player_2"]  # by HP, self excluded
    assert view["hp_rank"] == 1 and view["alive"] == 3


def test_candidates_come_from_opponent_options_only(lobby):
    me, players = lobby
    # own seat (the simulator can flag it) and seats that are not alive players are dropped
    assert public.public_view("player_0", me, players)["next_from"] == ["player_1"]


def test_hidden_next_opponent_candidates_come_from_the_env_info(lobby):
    """With hide_next_opponent the env's info["opponent_candidates"] is the candidate set; own seat and
    seats that are not alive players are dropped, and [] (before a PvE round) stays empty."""
    me, players = lobby
    view = public.public_view("player_0", me, players, candidates=["player_0", "player_2", "player_9"])
    assert view["next_from"] == ["player_2"]
    assert public.public_view("player_0", me, players, candidates=[])["next_from"] == []


def test_hidden_state_does_not_reach_the_view(lobby):
    me, players = lobby
    before = json.dumps(public.public_view("player_0", me, players, {"mage": ["ahri", "lulu"]}, "mage"))
    rich = players["player_1"]
    rich.gold = 51  # same interest bracket
    rich.exp = 17
    rich.bench = [None] * 9
    rich.item_bench = ["chain_vest"] + [None] * 9
    rich.shop = ["zed", "zed", "zed", "zed", "zed"]
    rich.possible_opponents = {k: 99 for k in rich.possible_opponents}
    after = json.dumps(public.public_view("player_0", me, players, {"mage": ["ahri", "lulu"]}, "mage"))
    assert before == after
    rich.health = 79  # public: must show
    assert json.dumps(public.public_view("player_0", me, players)) != before


def test_contested_counts_copies_on_other_boards(lobby):
    me, players = lobby
    comps = {"mage": ["ahri", "annie", "lulu", "veigar"], "duelist": ["fiora"]}
    view = public.public_view("player_0", me, players, comps, "mage")
    # ahri*2 on player_1 is 3 copies; my own ahri and the benched veigars are not counted
    assert view["contested"] == {"ahri": 3, "annie": 0, "lulu": 1, "veigar": 0}
    assert view["contested_by_comp"] == {"mage": 4, "duelist": 0}


def test_describe_adds_stage_features_and_stays_compact(lobby):
    me, players = lobby

    class Env:  # stands in for the env: only the player states are read
        class unwrapped:
            class player_manager:
                player_states = players

    state = describe(me, ["zed", None], 10, Env, seat="player_0", comps={"mage": ["ahri"]}, comp="mage")
    assert state["stage"] == "3-2" and state["next"] == {"carousel": 2, "pve": 4, "stage": 5}
    assert state["streak"] == 0 and state["dmg_per_loss"] == stages.damage_per_loss(10, me.rules)
    assert state["losses_to_death"] == stages.losses_to_death(me.health, 10, me.rules)
    assert state["shop"] == ["zed"] and [u["name"] for u in state["bench"]] == ["nami"]
    assert state["contested"] == {"ahri": 3}
    assert len(json.dumps(state, separators=(",", ":"))) < 1200
    copy.deepcopy(state)  # plain data

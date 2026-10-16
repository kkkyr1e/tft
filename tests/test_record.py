"""Trajectory recorder (tfteval/record.py): action-mask indexing, seat selection, and the recorder in real
games: a recorded game plays exactly as an unrecorded one, and recording goes through a snapshot. Plays two
short full games (a cheap lobby, mostly random seats) and two partial ones: about a minute."""

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("Simulator")

from tfteval import make_policy  # noqa: E402
from tfteval.record import ACTION_KINDS, action_index, legal, record_seats  # noqa: E402
from tfteval.runner import Game, play_game, sim_options  # noqa: E402


def test_action_index_inverts_the_simulators_action_space():
    from Simulator.encoding.token.action import ActionToken

    for index in range(55 * 38):
        action = ActionToken.action_space_to_action(index)
        back = action_index(action)
        assert ActionToken.action_space_to_action(back) == action
        if index // 38 in range(47) or index % 38 == 0:  # unit/item rows, and column 0 of the others
            assert back == index
    mask = np.zeros(55 * 38)
    mask[54 * 38] = 1
    assert legal([2, 0, 0], mask) and not legal([1, 0, 0], mask) and legal([1, 0, 0], None)
    assert ACTION_KINDS[2] == "roll"


def test_record_seats(monkeypatch):
    seats = {f"player_{i}": make_policy(name) for i, name in enumerate(["hero=mimic", "rule", "rule", "random"])}
    monkeypatch.delenv("TFT_RECORD", raising=False)
    assert record_seats(None, seats) == [] and record_seats(False, seats) == []
    assert record_seats(True, seats) == sorted(seats)
    assert record_seats("hero", seats) == ["player_0"]
    assert record_seats("rule,player_3", seats) == ["player_1", "player_2", "player_3"]
    assert record_seats(["player_2"], seats) == ["player_2"]
    monkeypatch.setenv("TFT_RECORD", "1")
    assert record_seats(None, seats) == sorted(seats)
    monkeypatch.setenv("TFT_RECORD", "0")
    assert record_seats(None, seats) == []
    with pytest.raises(ValueError):
        record_seats("nobody", seats)


class Hash:
    """Wraps a seat policy and hashes every action it picks."""

    def __init__(self, policy):
        self.policy, self.name, self.digest = policy, policy.name, hashlib.sha256()

    def reset(self, seed):
        self.policy.reset(seed)

    def carousel_picker(self):
        return self.policy.carousel_picker() if hasattr(self.policy, "carousel_picker") else None

    def act(self, observation, info, agent, env):
        action = self.policy.act(observation, info, agent, env)
        self.digest.update(repr(list(map(int, action))).encode())
        return action


CHEAP = ["hero=stance", "random", "mimic", "random", "random", "random", "random", "random"]


def lobby(hashed=True):
    return {f"player_{i}": Hash(make_policy(name)) if hashed else make_policy(name) for i, name in enumerate(CHEAP)}


def test_recording_changes_nothing_and_rows_are_complete(monkeypatch):
    monkeypatch.delenv("TFT_RECORD", raising=False)
    plain_seats, rec_seats = lobby(), lobby()
    plain = play_game(plain_seats, 31001)
    rec = play_game(rec_seats, 31001, record=True)
    assert plain.finished and rec.finished
    assert rec.placements == plain.placements and rec.actions == plain.actions and rec.steps == plain.steps
    assert rec.fallbacks == plain.fallbacks and rec.eliminated == plain.eliminated
    assert {s: w.digest.hexdigest() for s, w in rec_seats.items()} == \
        {s: w.digest.hexdigest() for s, w in plain_seats.items()}
    assert plain.records == {} and set(rec.records) == set(rec.placements)
    winner = next(s for s, p in rec.placements.items() if p == 1)
    assert winner not in rec.eliminated and len(rec.eliminated) == 7

    for seat, rows in rec.records.items():
        rounds = [r["round"] for r in rows]
        assert rounds == list(range(rounds[0], rounds[0] + len(rows)))  # one row per planning phase
        assert sum(sum(r["actions"].values()) for r in rows) == rec.actions[seat]
        budget = sim_options()[1].get("max_actions_per_round", 15)  # the default profile's actions per phase
        assert all(sum(r["actions"].values()) == budget for r in rows)
        assert sum(r["fallbacks"] for r in rows) == rec.fallbacks[seat]
        for r in rows:
            assert {"end", "board", "bench", "items", "fight", "hp_lost"} <= set(r)
            assert r["fight"] in ("W", "L", "pve") and r["hp_lost"] >= 0
            assert all(len(u) in (3, 4, 5) for u in r["board"] + r["bench"])
        for before, after in zip(rows, rows[1:]):
            assert after["hp"] == before["hp"] - before["hp_lost"]
            if before["fight"] == "W":
                assert after["streak"] == max(before["streak"], 0) + 1
            elif before["fight"] == "L":
                assert after["streak"] == min(before["streak"], 0) - 1
        if seat in rec.eliminated:
            assert rows[-1]["round"] == rec.eliminated[seat] and rows[-1]["hp"] - rows[-1]["hp_lost"] <= 0
    hero = rec.records["player_0"]
    assert all({"plan", "knobs", "stance", "why"} <= set(r) for r in hero)
    assert any(r.get("comp") for r in hero) and "plan" not in rec.records["player_1"][0]
    assert len(json.dumps(rec.records["player_0"])) < 1500 * len(hero)  # rows stay small


def test_records_go_through_a_snapshot(monkeypatch):
    from tfteval.branching import branch, play_to, snapshot

    monkeypatch.delenv("TFT_RECORD", raising=False)
    straight = Game(lobby(False), 31002, record=["hero"]).run(until_round=9).result()
    snap = snapshot(play_to(lobby(False), 31002, 6, record=["hero"]))
    resumed = branch(snap).run(until_round=9).result()
    assert list(resumed.records) == ["player_0"]
    assert resumed.records == straight.records and resumed.actions == straight.actions

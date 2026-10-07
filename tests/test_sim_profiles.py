"""Simulator profiles (tfteval.runner.SIM_PROFILES) in real games: the options reach the env, the
plan seats pick on the carousel, the hidden next opponent stays hidden from planners, and the
runner's keyed-stream speed-up (light_board_copies) changes nothing. Plays partial games: ~1 minute."""

import contextlib
import hashlib

import pytest

pytest.importorskip("Simulator")

from tfteval import make_policy, runner  # noqa: E402
from tfteval.runner import SIM_PROFILES, Game, sim_options  # noqa: E402


def test_sim_options_parse_profiles_and_overrides(monkeypatch):
    assert sim_options("default") == ("default", {})
    assert sim_options("realistic")[1] == {"pve_damage": True, "fortune_orbs": True, "carousel_fixes": True,
                                           "hide_next_opponent": True, "rng_streams": "keyed"}
    spec, opts = sim_options("realistic, rng_streams=shared")
    assert opts == {**SIM_PROFILES["realistic"], "rng_streams": "shared"}
    assert sim_options("default,hide_next_opponent=true")[1] == {"hide_next_opponent": True}
    monkeypatch.delenv("TFT_SIM", raising=False)
    assert sim_options()[0] == runner.DEFAULT_SIM == "realistic"
    monkeypatch.setenv("TFT_SIM", "default")
    assert sim_options() == ("default", {})
    with pytest.raises(ValueError):
        sim_options("realistc")
    with pytest.raises(ValueError):
        sim_options("realistic,pve_damage")


class Watch:
    """Wraps a seat policy: hashes its actions, and for a plan seat checks at every new round that the
    planner's candidate set is the env's and that the pairings are not drawn yet."""

    def __init__(self, policy):
        self.policy, self.name, self.trace, self.round, self.checked = policy, policy.name, hashlib.sha256(), None, 0

    def reset(self, seed):
        self.policy.reset(seed)

    def carousel_picker(self):
        return self.policy.carousel_picker() if hasattr(self.policy, "carousel_picker") else None

    def act(self, observation, info, agent, env):
        action = self.policy.act(observation, info, agent, env)
        self.trace.update(repr(list(map(int, action))).encode())
        rnd = info.get("game_round")
        state = getattr(self.policy, "last_state", None)
        if rnd != self.round and state is not None and state["round"] == rnd and "opponent_candidates" in info:
            self.round = rnd
            assert env.unwrapped.game_round.matchups == []  # drawn at combat time
            assert state["next_from"] == [s for s in info["opponent_candidates"] if s != agent]
            assert state["next_from"] == [s for s, f in sorted(info["player"].opponent_options.items()) if f == 1]
            if state["pve"]:
                assert state["next_from"] == []  # no player combat before a PvE round
            self.checked += 1
        return action


LOBBY = ["mimic", "stance", "rule", "rule", "stance+lossstreak", "rule", "rule", "rule"]


def partial(sim, until, seed=7100, light=True, monkeypatch=None):
    seats = {f"player_{i}": Watch(make_policy(name)) for i, name in enumerate(LOBBY)}
    if not light:
        monkeypatch.setattr(runner, "light_board_copies", lambda env: contextlib.nullcontext())
    game = Game(seats, seed, sim=sim).run(until_round=until)
    return game, seats


def test_realistic_profile_reaches_the_env_and_the_plan_seats_pick():
    game, seats = partial("realistic", 13)
    config = game.env.unwrapped.config
    assert all(getattr(config, k) == v for k, v in SIM_PROFILES["realistic"].items())
    result = game.result()
    assert result.sim == "realistic" and result.sim_options == SIM_PROFILES["realistic"]
    plan_seats = [s for s, w in seats.items() if hasattr(w.policy, "executor")]
    assert result.carousel_pickers == plan_seats == ["player_0", "player_1", "player_4"]
    for seat in plan_seats:
        executor = seats[seat].policy.executor
        assert [e["round"] for e in executor.carousel_log] == [0, 6, 12]  # 1-1 (inside reset), 2-4, 3-4
        assert executor.stats["carousel_picks"] == 3
        assert seats[seat].checked >= 10  # every planning round from 1-3 on
        assert executor.rng is game.player(seat).default_agent.rng  # keyed: the seat's own bot generator
    rule_seat = next(s for s in seats if s not in plan_seats)
    assert game.player(rule_seat).default_agent.rng is not None


def test_light_board_copies_change_nothing(monkeypatch):
    fast, fast_seats = partial("realistic", 9)
    slow, slow_seats = partial("realistic", 9, light=False, monkeypatch=monkeypatch)
    assert fast.health == slow.health and fast.actions == slow.actions
    assert {s: w.trace.hexdigest() for s, w in fast_seats.items()} == \
        {s: w.trace.hexdigest() for s, w in slow_seats.items()}

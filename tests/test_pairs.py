"""Contrastive pairs (tfteval/pairs.py): the edit, the paired contrasts and the pooling.

Everything is pure Python on fakes and synthetic data."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfteval import bank2, pairs  # noqa: E402


class FakePlayer:
    def __init__(self, health=60, gold=50):
        self.health, self.gold = health, gold


class FakeStance:
    def __init__(self, hp_seen):
        self.hp_seen = dict(hp_seen)


class FakePolicy:
    def __init__(self, planner):
        self.planner = planner


class FakeGame:
    def __init__(self, player, policy):
        self._player, self.seat_policies = player, {"player_0": policy}

    def player(self, seat):
        return self._player


def test_hp_edit_shifts_the_stance_history_so_recent_loss_is_unchanged():
    game = FakeGame(FakePlayer(health=60), FakePolicy(FakeStance({20: 80, 21: 70, 22: 60})))
    changed = pairs.apply_edit(game, "player_0", {"hp": 35})
    assert game.player("player_0").health == 35 and changed == {"hp": [60, 35]}
    seen = game.seat_policies["player_0"].planner.hp_seen
    assert seen == {20: 55, 21: 45, 22: 35}
    assert max(seen.values()) - 35 == 80 - 60  # the base state's bleeding, not 80 - 35


def test_hp_edit_through_commit_planner_and_without_history():
    class Commit:
        def __init__(self, base):
            self.base = base

    stance = FakeStance({5: 90})
    game = FakeGame(FakePlayer(health=90), FakePolicy(Commit(stance)))
    pairs.apply_edit(game, "player_0", {"hp": 50})
    assert stance.hp_seen == {5: 50}
    game = FakeGame(FakePlayer(health=90), FakePolicy(object()))  # a planner with no hp history
    assert pairs.apply_edit(game, "player_0", {"hp": 50}) == {"hp": [90, 50]}


def test_edit_rejects_bad_values_and_axes():
    game = FakeGame(FakePlayer(), FakePolicy(FakeStance({})))
    with pytest.raises(ValueError):
        pairs.apply_edit(game, "player_0", {"hp": 0})
    with pytest.raises(ValueError):
        pairs.apply_edit(game, "player_0", {"hp": 101})
    with pytest.raises(ValueError):
        pairs.apply_edit(game, "player_0", {"gold": -1})
    with pytest.raises(ValueError):
        pairs.apply_edit(game, "player_0", {"level": 8})
    assert pairs.apply_edit(game, "player_0", {"gold": 30}) == {"gold": [50, 30]}


def test_candidate_matches_the_bank_menu_entry():
    menu = {c["name"]: c for c in bank2.menu("ref41")}
    assert pairs.candidate("roll_all") == menu["roll_all"]


def synthetic_pair(effect0, effect1, k=40, seed=0, noise=1.5):
    """Branches whose A - B is effect0 on side 0 and effect1 on side 1, with a shared per-k term."""
    rng = np.random.default_rng(seed)
    rows = []
    for kk in range(k):
        common = rng.normal(0, 2)
        for side, eff in ((0, effect0), (1, effect1)):
            rows.append({"side": side, "cand": "A", "k": kk, "place": 4.5 + common + eff / 2 + rng.normal(0, noise)})
            rows.append({"side": side, "cand": "B", "k": kk, "place": 4.5 + common - eff / 2 + rng.normal(0, noise)})
    return {"cands": ["A", "B"], "branches": rows}


def test_contrasts_recover_the_interaction_paired_by_k():
    pair = synthetic_pair(-1.0, 0.5, k=400)
    c = pairs.contrasts(pair)
    assert c["ks"] == 400
    assert c["side"][0]["mean"] == pytest.approx(-1.0, abs=0.3)
    assert c["side"][1]["mean"] == pytest.approx(0.5, abs=0.3)
    assert c["interaction"]["mean"] == pytest.approx(-1.5, abs=0.4)
    # the shared per-k term cancels: the interaction SD is about 2 * noise, not inflated by it
    assert c["interaction"]["sd"] == pytest.approx(2 * 1.5, rel=0.15)


def test_contrasts_use_only_ks_all_four_cells_have():
    pair = synthetic_pair(0, 0, k=10)
    pair["branches"] = [b for b in pair["branches"] if not (b["side"] == 1 and b["cand"] == "B" and b["k"] >= 7)]
    pair["branches"].append({"side": 0, "cand": "A", "k": 3, "place": None})  # an unfinished branch is skipped
    assert pairs.contrasts(pair)["ks"] == 7


def test_pooled_mean_and_random_effects():
    ests = [{"mean": m, "se": 0.3} for m in (-0.6, -0.4, -0.5)]
    p = pairs.pooled(ests)
    assert p["pairs"] == 3 and p["mean"] == pytest.approx(-0.5)
    assert p["fixed"] == pytest.approx(-0.5) and p["tau2"] == 0.0
    assert p["fixed_se"] == pytest.approx(0.3 / math.sqrt(3))
    spread = pairs.pooled([{"mean": m, "se": 0.1} for m in (-2.0, 0.0, 2.0)])
    assert spread["tau2"] > 1 and spread["random_se"] > spread["fixed_se"]


def test_assemble_refuses_sides_from_different_states():
    spec = {"lineup": "stance@rule:7", "seed": 1, "point": "4-1", "axis": "hp", "values": [35, 85],
            "cands": ["roll_all", "save"], "k": 1, "continuation": "stance"}
    side = {"recipe": {}, "base": {"fingerprint_hash": "x"}, "branches": [], "side": 0}
    other = {**side, "side": 1, "base": {"fingerprint_hash": "y"}}
    with pytest.raises(RuntimeError):
        pairs.assemble(spec, [side, other])
    pair = pairs.assemble(spec, [{**other, "base": {"fingerprint_hash": "x"}}, side])
    assert [s["side"] for s in pair["sides"]] == [0, 1] and pair["id"] == "stance@rule:7#1@4-1:hp=35|85"


def test_set_action_budget_raises_every_seat_from_now_on():
    class Config:
        max_actions_per_round = 15

    class Seat:
        actions_remaining, actions_per_round = 15, 15

    class Manager:
        config = Config()
        player_states = {"player_0": Seat(), "player_1": None, "player_2": Seat()}

    class Env:
        max_actions_per_round = 15
        player_manager = Manager()

    class Wrapped:
        unwrapped = Env()

    class Game:
        env = Wrapped()

    game = Game()
    pairs.set_action_budget(game, 40)
    env = game.env.unwrapped
    assert env.max_actions_per_round == 40 and env.player_manager.config.max_actions_per_round == 40
    seats = [p for p in env.player_manager.player_states.values() if p is not None]
    assert all(p.actions_remaining == 40 and p.actions_per_round == 40 for p in seats)
    assert pairs.pair_id("l", 1, "4-1", "hp", [35, 85], 40) == "l#1@4-1:hp=35|85@actions40"
    assert pairs.pair_id("l", 1, "4-1", "hp", [35, 85]) == "l#1@4-1:hp=35|85"

import json
from pathlib import Path

import pytest

from tfteval import stages

ROOT = Path(__file__).resolve().parents[1]


def test_labels_follow_the_simulator_schedule():
    assert [stages.label(i) for i in range(10)] == ["1-2", "1-3", "1-4", "2-1", "2-2", "2-3", "2-5", "2-6", "2-7",
                                                   "3-1"]
    assert stages.label(11) == "3-3"  # the rule bot's "round 11"
    assert stages.label(14) == "3-7" and stages.label(15) == "4-1" and stages.label(32) == "6-7"


def test_parse_label_inverts_label():
    for idx in range(1, 44):
        assert stages.parse_label(stages.label(idx)) == idx
    assert stages.parse_label("2-4") == stages.parse_label("2-5") == 6  # carousel merged into 2-5
    assert stages.parse_label(17) == 17
    with pytest.raises(ValueError):
        stages.parse_label("2-8")


def test_round_kinds_match_game_round_table(monkeypatch):
    from Simulator import config
    from Simulator.game.game_round import Game_Round

    monkeypatch.setattr(config, "LOGMESSAGES", False)  # else the constructor writes log.txt
    rounds = Game_Round({}, None, None)
    for idx, steps in enumerate(rounds.game_rounds):
        names = [s.__name__ for s in steps]
        assert stages.is_carousel(idx) == ("carousel_round" in names or "round_1" in names), idx
        assert stages.is_pve(idx) == ("minion_round" in names or "round_1" in names), idx


def test_rounds_until_next_events():
    s = stages.schedule(3)  # 2-1
    assert (s["to_carousel"], s["to_pve"], s["to_stage"]) == (3, 5, 6)
    s = stages.schedule(8)  # 2-7 krugs
    assert s["pve"] and (s["to_carousel"], s["to_pve"], s["to_stage"]) == (4, 0, 1)
    assert stages.schedule(6)["carousel"] and stages.schedule(6)["to_carousel"] == 0


# --------------------------------------------------------------------------- damage

PROFILES = ("set4", "set18")


def _game_round(monkeypatch, name):
    from Simulator import config
    from Simulator.game import pool
    from Simulator.game.game_round import Game_Round

    monkeypatch.setattr(config, "LOGMESSAGES", False)
    return Game_Round({}, pool.pool(rules=name), None)


@pytest.mark.parametrize("name", PROFILES)
def test_base_damage_is_the_simulators_table(monkeypatch, name):
    rounds = _game_round(monkeypatch, name)  # the table combat_phase charges, for a game on this profile
    assert rounds.rules.name == name
    for idx in range(0, 60):
        tier = next(d for last, d in rounds.ROUND_DAMAGE if idx <= last)  # combat_phase's own lookup
        assert stages.base_damage(idx, name) == tier, idx


def test_base_damage_by_stage():
    first = [stages.parse_label(f"{s}-1") for s in range(2, 9)]  # 2-1 .. 8-1
    assert [stages.base_damage(i, "set4") for i in first] == [0, 2, 3, 5, 8, 15, 15]  # patch 10.24
    assert [stages.base_damage(i, "set18") for i in first] == [2, 6, 7, 10, 12, 17, 150]
    # the old simulator charged each tier a stage early (2 from 2-2); the fork charges 0 all of stage 2
    assert {stages.base_damage(i, "set4") for i in range(3, 9)} == {0}


def test_unit_damage_is_the_profiles_table():
    from Simulator.battle.stats import DAMAGE_PER_UNIT

    assert [stages.unit_damage(n, "set4") for n in range(len(DAMAGE_PER_UNIT))] == list(DAMAGE_PER_UNIT)
    assert [stages.unit_damage(n, "set18") for n in range(12)] == list(range(12))
    assert stages.max_damage(19, 6, "set4") == 3 + 11 and stages.max_damage(19, 6, "set18") == 7 + 6  # 4-6
    assert stages.max_damage(20, 6) == 0  # PvE


def test_profile_comes_from_tft_rules_like_the_runner(monkeypatch):
    from Simulator.game.rules import SET4, SET18

    monkeypatch.delenv("TFT_RULES", raising=False)
    assert stages.rules_profile() is SET4
    monkeypatch.setenv("TFT_RULES", "set18")
    assert stages.rules_profile() is SET18 and stages.base_damage(3) == 2
    assert stages.rules_profile(SET4) is SET4 and stages.rules_profile("SET4") is SET4  # explicit wins
    with pytest.raises(ValueError):
        stages.rules_profile("set99")


@pytest.mark.parametrize("name", PROFILES)
def test_damage_per_loss_is_base_plus_priced_survivors(name):
    for idx in range(3, 44):
        if stages.is_pve(idx):
            assert stages.damage_per_loss(idx, name) == 0
            continue
        counts = stages.survivors_hist(idx)
        unit = sum(c * stages.unit_damage(n, name) for n, c in enumerate(counts)) / sum(counts)
        assert stages.damage_per_loss(idx, name) == pytest.approx(stages.base_damage(idx, name) + unit, abs=0.01)
    # the same survivors cost more per unit under set4 (2 each up to 5) than under set18 (1 each)
    assert stages.expected_unit_damage(3, "set4") > stages.expected_unit_damage(3, "set18")


@pytest.mark.parametrize("name", PROFILES)
def test_damage_per_loss_matches_the_measurement(name):
    """scripts/measure_damage.py on the fork (8 rule bots): per stage, the model's expected damage per
    loss, averaged over the measured losses, is within 1 HP (and 10%) of the measured mean."""
    path = ROOT / "results" / "damage" / f"{name}.json"
    measured = json.loads(path.read_text())
    assert measured["rules"] == name
    checked = 0
    for stage, row in measured["by_stage"].items():
        if row["n"] < 50:
            continue
        raw = [(i, d) for i, d, _ in measured["raw"] if stages.stage_round(i)[0] == int(stage)]
        model = sum(stages.damage_per_loss(i, name) for i, _ in raw) / len(raw)
        assert model == pytest.approx(row["mean"], abs=max(1.0, 0.1 * row["mean"])), (stage, model, row)
        checked += 1
    assert checked >= 4


def test_losses_to_death_walks_the_upcoming_pvp_rounds():
    for name in PROFILES:
        d = [stages.damage_per_loss(i, name) for i in range(44)]
        assert stages.losses_to_death(d[3] / 2, 3, name) == 1
        # 2-6 then 2-7 (PvE, free) then 3-1
        assert stages.losses_to_death(d[7] + d[9] - 0.5, 7, name) == 2
        assert stages.losses_to_death(100, 3, name) > stages.losses_to_death(100, 20, name)
        assert stages.losses_to_death(0, 10, name) == 0
    # set18's stage 8 (150 base) ends any game
    assert stages.losses_to_death(1000, stages.parse_label("8-1"), "set18") == 7


def test_describe_uses_the_players_profile(monkeypatch):
    from Simulator.game import pool
    from Simulator.game.player import Player

    from tfteval.planner import compile_knobs, describe

    monkeypatch.delenv("TFT_RULES", raising=False)
    for name in PROFILES:
        p = Player(pool.pool(rules=name), 0)
        p.health = 30
        state = describe(p, [None] * 5, 10)
        assert state["dmg_per_loss"] == stages.damage_per_loss(10, name)
        assert state["losses_to_death"] == stages.losses_to_death(30, 10, name)
        knobs = compile_knobs({"comp": None, "level_to": 1, "roll_floor": 999, "carry": None,
                               "survival": state["losses_to_death"]}, state)
        assert knobs["survival"]  # compile_knobs reads describe's losses_to_death

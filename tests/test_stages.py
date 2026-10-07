import pytest

from tfteval import stages


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


def test_damage_table_matches_simulator_constants():
    from Simulator.battle.stats import DAMAGE_PER_UNIT

    assert tuple(DAMAGE_PER_UNIT) == stages.DAMAGE_PER_UNIT
    assert [stages.base_damage(i) for i in (3, 4, 9, 10, 16, 22, 28)] == [0, 2, 2, 3, 5, 8, 15]
    assert stages.damage_per_loss(8) == 0  # PvE: no HP lost
    assert stages.max_damage(19, 6) == 5 + 11 and stages.max_damage(20, 6) == 0


def test_losses_to_death_walks_the_upcoming_pvp_rounds():
    assert stages.losses_to_death(4, 3) == 1  # 2-1 costs ~4.4
    assert stages.losses_to_death(13, 6) == 2  # 2-5, 2-6 cost ~8 each
    # 2-6 (8.0), 2-7 is PvE, 3-1 (8.0), 3-2 (10.6): 20 HP survive two losses, not three
    assert stages.losses_to_death(20, 7) == 3
    assert stages.losses_to_death(100, 3) > stages.losses_to_death(100, 20)
    assert stages.losses_to_death(0, 10) == 0

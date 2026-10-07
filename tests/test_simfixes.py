import random
from types import SimpleNamespace

from tfteval.simfixes import carousel_order


def _players(hps):
    return [SimpleNamespace(name=f"p{i}", health=hp) for i, hp in enumerate(hps)]


def test_every_living_player_gets_a_pick():
    players = _players([80, 90, 70, 60, 100, 50, 0, 95]) + [None]
    order = carousel_order(players, 6, random.Random(0))
    assert sorted(p.name for p in order) == ["p0", "p1", "p2", "p3", "p4", "p5", "p7"]


def test_released_in_pairs_from_lowest_hp():
    players = _players([80, 90, 70, 60, 100, 50, 30, 95])
    for seed in range(20):
        order = carousel_order(players, 12, random.Random(seed))
        pairs = [sorted(p.health for p in order[i:i + 2]) for i in range(0, 8, 2)]
        assert pairs == [[30, 50], [60, 70], [80, 90], [95, 100]]


def test_ties_and_pair_order_are_random_but_seeded():
    players = _players([50] * 8)
    a = [p.name for p in carousel_order(players, 18, random.Random(1))]
    b = [p.name for p in carousel_order(players, 18, random.Random(1))]
    c = [p.name for p in carousel_order(players, 18, random.Random(2))]
    assert a == b and a != c


def test_first_carousel_is_everyone_at_once():
    players = _players([100] * 8)
    order = carousel_order(players, 0, random.Random(3))
    assert len(order) == 8


def test_no_runtime_patch_when_the_simulator_has_the_fix(monkeypatch):
    from Simulator.game import carousel as carousel_module

    from tfteval import simfixes

    monkeypatch.setattr(carousel_module, "carousel_order", lambda players, r: list(players), raising=False)
    assert simfixes.needed() == ()

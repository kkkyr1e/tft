"""Save-state branching. Needs the simulator (skipped without it) and plays real games: ~2-3 minutes.

Run with PYTHONHASHSEED pinned like every other game here; the comparisons are within one process,
so they hold for any fixed hash seed."""

import pickle

import pytest

pytest.importorskip("Simulator")

from tfteval import make_policy, play_game  # noqa: E402
from tfteval.branching import branch, play_to, restore, snapshot, switch_planner  # noqa: E402


def lobby(seed):
    names = ["hero=mimic"] + ["rule"] * 7
    shift = seed % 8
    order = names[shift:] + names[:shift]
    return {f"player_{i}": make_policy(name) for i, name in enumerate(order)}


def hero_seat(seed):
    return f"player_{(8 - seed % 8) % 8}"


def outcome(result):
    return result.placements, result.steps, result.actions


# Round 9 is before the rule bots draw their comp from numpy's global generator (round 11),
# so it also checks that the global generator is saved and restored.
@pytest.mark.parametrize("seed,rnd", [(7100, 9), (7101, 13), (7102, 17)])
def test_resume_without_reseed_replays_the_original_game(seed, rnd):
    original = play_game(lobby(seed), seed)
    game = play_to(lobby(seed), seed, rnd)
    assert game.round == rnd and not game.done
    snap = snapshot(game)
    for _ in range(2):  # restoring twice from the same snapshot gives the same game both times
        assert outcome(branch(snap).run().result()) == outcome(original)


@pytest.fixture(scope="module")
def snap13():
    return snapshot(play_to(lobby(7103), 7103, 13))


def test_reseeded_branches_differ_and_are_themselves_reproducible(snap13):
    plain = branch(snap13).run().result().placements
    futures = [branch(snap13, reseed=k).run().result().placements for k in range(3)]
    assert all(f != plain for f in futures)
    assert len({tuple(sorted(f.items())) for f in futures}) == 3
    assert branch(snap13, reseed=1).run().result().placements == futures[1]


def test_switch_planner_keeps_the_executor_state(snap13):
    seat = hero_seat(7103)
    kept = restore(snap13).seat_policies[seat].executor
    policy = branch(snap13, switch={seat: switch_planner("fast8")}).seat_policies[seat]
    assert policy.planner.level_gold == 34 and policy.name == "hero"
    assert kept.comp_number >= 0  # the comp chosen at round 11 ...
    assert (policy.executor.comp_number, policy.executor.pairs) == (kept.comp_number, kept.pairs)  # ... survives


def test_snapshot_is_independent_of_the_running_game(snap13):
    game = restore(snap13)
    game.run(until_round=14)
    again = restore(snap13)
    assert again.round == 13 and game.round == 14
    assert pickle.loads(snap13.blob)["game"].round == 13


def test_all_seats_share_one_pool_and_one_shop_stream(snap13):
    """One seat's reroll changes the shop another seat rolls next."""
    seat = hero_seat(7103)
    other = next(s for s in lobby(7103) if s != seat)

    def other_shop(reroll_first):
        env = restore(snap13).env.unwrapped
        players = env.player_manager.player_states
        assert all(p.pool_obj is env.pool_obj for p in players.values())
        with env.combat_ctx.bind():
            if reroll_first:
                players[seat].refresh_shop()
            players[other].refresh_shop()
        return list(players[other].shop)

    assert other_shop(False) == other_shop(False)
    assert other_shop(True) != other_shop(False)

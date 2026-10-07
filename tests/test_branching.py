"""Save-state branching. Needs the simulator (skipped without it) and plays real games: ~3-4 minutes.

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


# One seat runs the executor's optional plan fields in windows around the snapshot. The fodder board
# (2-3 to 2-7) is still on when the game is saved at the start of round 9 (3-1), and the executor only
# knows to field its strongest units again in 3-1 from the knobs it carries over; level_by is
# scheduled across the snapshot and field_comp starts after it. Executor state (knobs, tracked pairs,
# the rule bot's round checks, stats) has to come through the pickle for the resumed game to match.
# With PYTHONHASHSEED=0 every capability acts in this game: fodder buys and moves, restore moves,
# comp swaps and xp bought ahead of the rule bot.
OVERLAY = [
    {"from": "2-3", "to": "2-7", "fields": {"fodder": True}},
    {"from": "2-5", "to": "4-1", "fields": {"level_by": {"level": 8, "by": "4-1"}}},
    {"from": "3-3", "fields": {"field_comp": True}},
]


def overlay_lobby(seed):
    from tfteval.planner import OverlayPlanner, ParamPlanner, PlanPolicy

    seats = {f"player_{i}": make_policy("rule") for i in range(8)}
    seats[hero_seat(seed)] = PlanPolicy(OverlayPlanner(ParamPlanner(), OVERLAY, name="overlay"), name="hero")
    return seats


def test_resume_replays_a_game_with_overlay_plan_fields():
    seed, seat = 7110, hero_seat(7110)
    seats = overlay_lobby(seed)
    original = play_game(seats, seed)
    stats = dict(seats[seat].executor.stats)
    assert stats["fodder_rounds"] and stats["restore_moves"] and stats["xp"]  # the windows did act

    game = play_to(overlay_lobby(seed), seed, 9)
    assert game.round == 9 and not game.done
    assert game.seat_policies[seat].executor.knobs["fodder"]  # saved with the fodder board still on
    snap = snapshot(game)
    for _ in range(2):
        resumed = branch(snap).run()
        assert outcome(resumed.result()) == outcome(original)
        assert dict(resumed.seat_policies[seat].executor.stats) == stats


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

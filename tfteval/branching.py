"""Save a game in progress and resume it, possibly many times with different futures.

    from tfteval import make_policy
    from tfteval.branching import play_to, snapshot, branch, switch_planner

    lobby = {f"player_{i}": make_policy("rule") for i in range(8)}
    lobby["player_3"] = make_policy("hero=mimic")
    game = play_to(lobby, seed=7000, round=13)       # stops when the planning phase of round 13 begins
    snap = snapshot(game)                            # bytes-backed, ~ms; safe to send to another process
    same = branch(snap).run().result()               # no reseed: replays the original game exactly
    alt = branch(snap, reseed=1,                     # an independent future from the same state,
                 switch={"player_3": switch_planner("fast8")}).run().result()  # with the hero on another economy

What a snapshot holds: the whole `runner.Game` (env with its EnvRNG and combat context, every seat
policy, the fallback policy, the runner's bookkeeping: placements so far, health, step and action
counts), plus the two process-global generators the game reads: numpy's global generator (the
upstream rule bot picks its comp with it) and Python's `random` module. It is a pickle, so it can be
written to disk or sent to a worker. A worker must run with the same PYTHONHASHSEED (the simulator
iterates sets of seat names) and gets the same simulator fixes installed on restore.

Re-seeding (`reseed=k`) replaces every random stream a game reads: the env's Python and numpy
generators (shops, carousel ties, loot orbs, matchmaking, combat), numpy's global generator, the
`random` module, and every numpy / Python generator found inside the seat policies and the fallback
policy. Seeds come from SeedSequence([game seed, branch round, k]), so branch k of a state is the same
future whichever policy is switched in: branches with the same k share their random numbers until
the policies' actions first differ (the env has one stream for all seats, so the first different
action shifts everyone's later draws).
"""

from __future__ import annotations

import copy
import os
import pickle
import random
import time
from dataclasses import dataclass

import numpy as np

from tfteval.runner import Game

PROTOCOL = pickle.HIGHEST_PROTOCOL


@dataclass
class Snapshot:
    blob: bytes  # pickle of {"game", "np_global", "py_random", "hashseed"}
    seed: int  # the game's seed
    round: int  # the round whose planning phase had just begun
    seconds: float  # time taken to make it

    @property
    def size(self) -> int:
        return len(self.blob)


def play_to(seat_policies, seed: int, round: int, **game_kwargs) -> Game:
    """Start a game and play until the planning phase of `round` begins (or the game ends first;
    check `game.done` and `game.round`)."""
    return Game(seat_policies, seed, **game_kwargs).run(until_round=round)


def snapshot(game: Game) -> Snapshot:
    started = time.perf_counter()
    state = {
        "game": game,
        "np_global": np.random.get_state(),
        "py_random": random.getstate(),
        "hashseed": os.environ.get("PYTHONHASHSEED"),
        "players_order": list(game.env.unwrapped.player_manager.players),
    }
    blob = pickle.dumps(state, protocol=PROTOCOL)
    return Snapshot(blob, game.seed, game.round, time.perf_counter() - started)


def _restore_seat_set_order(game: Game, want: list) -> None:
    """PlayerManager.players is a set of seat names, and the simulator iterates it to refresh the
    shops at the start of every round. All shops draw from one generator, so the iteration order
    decides who gets which shop. Pickle (and deepcopy) rebuild a set by inserting its items in
    iteration order, and when two names collide in the hash table that lands them in different slots
    than the original insertion order (player_0 .. player_7) did: the restored game then deals the
    same random numbers to different seats and diverges from the next round on (measured on seed 7000).
    Rebuilding the set the way the simulator builds it restores the original order."""
    manager = game.env.unwrapped.player_manager
    rebuilt = {"player_" + str(i) for i in range(len(want))}  # PlayerManager.__init__'s insertion order
    if list(rebuilt) != want:
        raise RuntimeError(f"cannot restore the seat iteration order {want}; rebuilt set gives {list(rebuilt)}")
    manager.players = rebuilt


def restore(snap: Snapshot) -> Game:
    """A new, independent Game in exactly the snapshot's state; also restores the global generators."""
    state = pickle.loads(snap.blob)
    if state["hashseed"] != os.environ.get("PYTHONHASHSEED"):
        raise RuntimeError(f"snapshot made with PYTHONHASHSEED={state['hashseed']!r}, this process has "
                           f"{os.environ.get('PYTHONHASHSEED')!r}; the resumed game would not match")
    game: Game = state["game"]
    _restore_seat_set_order(game, state["players_order"])
    if game.sim_fixes:
        from tfteval import simfixes

        if set(game.sim_fixes) - set(simfixes.apply()):
            raise RuntimeError(f"snapshot uses sim fixes {game.sim_fixes} this version does not have")
    np.random.set_state(state["np_global"])
    random.setstate(state["py_random"])
    return game


def branch(snap: Snapshot, reseed: int | None = None, switch: dict | None = None) -> Game:
    """Restore a snapshot, optionally swap seat policies and re-seed all randomness.

    `switch` maps a seat to a function old_policy -> new_policy (see `switch_planner`) or to a
    ready policy object. A switched-in policy is NOT reset, so it can carry state over.
    `reseed=None` keeps the original random streams: with no switch, the game replays exactly.
    """
    game = restore(snap)
    for seat, new in (switch or {}).items():
        game.seat_policies[seat] = new(game.seat_policies[seat]) if callable(new) else new
    if reseed is not None:
        reseed_game(game, np.random.SeedSequence([snap.seed, snap.round, reseed]))
    return game


def reseed_game(game: Game, seq: np.random.SeedSequence) -> None:
    env_py, env_np, np_global, py_global, policies = seq.spawn(5)
    rng = game.env.unwrapped.rng  # the same EnvRNG object the combat context holds
    if game.env.unwrapped.combat_ctx.rng is not rng:
        raise RuntimeError("env.rng and combat_ctx.rng are different objects; re-seeding one would miss the other")
    rng.py.seed(int(env_py.generate_state(1, np.uint64)[0]))
    rng.np.bit_generator.state = np.random.default_rng(env_np).bit_generator.state  # np_api wraps this Generator
    np.random.seed(env_seed32(np_global))
    random.seed(int(py_global.generate_state(1, np.uint64)[0]))
    found = []
    _collect_generators(game.seat_policies, found, set())
    _collect_generators(game.fallback_policy, found, set())
    for child, gen in zip(policies.spawn(len(found)), found):
        if isinstance(gen, np.random.Generator):
            gen.bit_generator.state = np.random.default_rng(child).bit_generator.state
        elif isinstance(gen, np.random.RandomState):
            gen.seed(env_seed32(child))
        else:
            gen.seed(int(child.generate_state(1, np.uint64)[0]))


def env_seed32(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1)[0])


def _collect_generators(obj, found: list, seen: set, depth: int = 0) -> None:
    """Every numpy Generator / RandomState / random.Random reachable from a policy, in a stable order."""
    if depth > 6 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (np.random.Generator, np.random.RandomState, random.Random)):
        found.append(obj)
        return
    if isinstance(obj, dict):
        for key in sorted(obj, key=str):
            _collect_generators(obj[key], found, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_generators(item, found, seen, depth + 1)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for key in sorted(vars(obj)):
            _collect_generators(vars(obj)[key], found, seen, depth + 1)


def switch_planner(planner):
    """A `switch` entry that gives a plan-executor seat another planner and keeps its executor
    state: target comp, tracked pairs, per-round checks. A fresh PlanPolicy would forget its comp
    and pairs mid-game, which is a different change. `planner` is a planner object or a name that
    `make_plan_policy` knows (`mimic`, `fast8`, ... or `llm`)."""
    from tfteval.planner import PlanPolicy, make_plan_policy

    def apply(old):
        if not isinstance(old, PlanPolicy):
            raise TypeError(f"switch_planner needs a plan-executor seat, got {type(old).__name__}")
        new = copy.copy(old)  # shares the executor and its state; the old policy is discarded
        # the seat keeps its name (e.g. its `hero` alias)
        new.planner = make_plan_policy(planner).planner if isinstance(planner, str) else planner
        new.round = None  # ask the new planner at once, even mid-round
        return new

    return apply

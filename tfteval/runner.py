"""Play one full 8-player game and return finishing places."""

from __future__ import annotations

import contextlib
import io
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from tfteval.policies import Policy, RandomPolicy


@dataclass
class GameResult:
    seed: int
    lobby: dict  # seat -> policy name
    placements: dict  # seat -> place, 1 is best
    steps: int
    finished: bool
    seconds: float
    actions: dict = field(default_factory=dict)  # seat -> number of actions the policy chose
    fallbacks: dict = field(default_factory=dict)  # seat -> actions replaced because the policy raised
    reproducible: bool = False  # True only when PYTHONHASHSEED was pinned for this process
    sim_fixes: list = field(default_factory=list)  # tfteval.simfixes patched in at runtime; [] = simulator as checked out
    sim_commit: str | None = None  # git commit of the simulator checkout (scripts/setup_sim.sh pins it)
    rules: str = "set4"  # simulator economy profile

    def to_json(self) -> dict:
        return asdict(self)


def _seat_seed(game_seed: int, seat_index: int) -> int:
    return int(np.random.SeedSequence([game_seed, seat_index]).generate_state(1)[0])


def play_game(seat_policies: dict[str, Policy], seed: int, max_steps: int = 20000, quiet: bool = True,
              sim_fixes: bool | None = None, rules: str | None = None) -> GameResult:
    """Run one game. `seat_policies` maps "player_0".."player_7" to a policy.

    The same seed replays the same game only if the interpreter was started with a fixed
    PYTHONHASHSEED; without it the same seed gives a different game in every new process
    (measured; the likely cause is hash-ordered iteration inside the simulator). scripts/run_lobby.py pins it for you.

    Places are handed out from 8 upward as seats are eliminated; seats eliminated in the
    same step are ordered by remaining health, lower health taking the worse place.
    If a policy raises, a random legal action is substituted and counted in `fallbacks`.

    `sim_fixes` installs tfteval.simfixes (default on; TFT_SIM_FIXES=0 turns it off, e.g. to
    replay runs made before the fixes). A process cannot switch back once they are installed.

    `rules` picks the simulator's economy profile: "set4" (default) or "set18" (current-set shop
    odds, xp, pool sizes, streak gold and player damage on the Set 4 roster; needs the fork).
    Default from TFT_RULES.
    """
    from Simulator.simulators.tft_simulator import TFTConfig, parallel_env

    from tfteval import simfixes

    if sim_fixes is None:
        sim_fixes = os.environ.get("TFT_SIM_FIXES", "1") != "0"
    active = list(simfixes.apply()) if sim_fixes else []
    rules = rules or os.environ.get("TFT_RULES", "set4")
    config = {"rules": rules} if rules != "set4" else {}  # older simulators have no rules option

    started = time.time()
    np.random.seed(seed % (2**32))  # the upstream rule bot uses numpy's global generator
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if quiet else _Passthrough()):
        env = parallel_env(TFTConfig(num_players=len(seat_policies), **config))
        observations, infos = env.reset(seed=seed)
        seats = list(env.possible_agents)
        if set(seats) != set(seat_policies):
            raise ValueError(f"lobby seats {sorted(seat_policies)} do not match env seats {seats}")

        fallback_policy = RandomPolicy()
        fallback_policy.reset(_seat_seed(seed, 10_000))
        for index, seat in enumerate(seats):
            seat_policies[seat].reset(_seat_seed(seed, index))

        placements: dict[str, int] = {}
        actions = {seat: 0 for seat in seats}
        fallbacks = {seat: 0 for seat in seats}
        health = {seat: 100.0 for seat in seats}
        next_place, steps = len(seats), 0

        while len(placements) < len(seats) and steps < max_steps and env.agents:
            chosen = {}
            for seat in env.agents:
                if seat in placements:
                    continue
                obs, info = observations.get(seat), infos.get(seat, {})
                try:
                    chosen[seat] = seat_policies[seat].act(obs, info, seat, env)
                except NotImplementedError:
                    raise
                except Exception:
                    fallbacks[seat] += 1
                    chosen[seat] = fallback_policy.act(obs, info, seat, env)
                actions[seat] += 1
            observations, _, terminated, _, infos = env.step(chosen)

            for seat in seats:
                player = infos.get(seat, {}).get("player")
                if player is not None:
                    health[seat] = float(getattr(player, "health", health[seat]))
            out_now = sorted(
                (seat for seat in seats if terminated.get(seat, False) and seat not in placements),
                key=lambda seat: health[seat],
            )
            for seat in out_now:
                placements[seat] = next_place
                next_place -= 1
            steps += 1
        env.close()

    return GameResult(
        seed=seed,
        lobby={seat: seat_policies[seat].name for seat in seats},
        placements=placements,
        steps=steps,
        finished=len(placements) == len(seats),
        seconds=round(time.time() - started, 2),
        actions=actions,
        fallbacks=fallbacks,
        reproducible=os.environ.get("PYTHONHASHSEED", "random").isdigit(),
        sim_fixes=active,
        sim_commit=simfixes.sim_commit(),
        rules=rules,
    )


class _Passthrough(io.TextIOBase):
    def write(self, text):
        import sys

        return sys.__stdout__.write(text)

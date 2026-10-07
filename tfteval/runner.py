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


class Game:
    """One game in progress. `play_game` runs it to the end; tfteval.branching saves and resumes it.

    Everything a game needs to continue lives on this object (env, seat policies, the runner's
    bookkeeping) except two process-global generators: numpy's global generator, which the
    upstream rule bot draws from, and Python's `random` module (unused as far as we know, saved anyway).
    """

    def __init__(self, seat_policies: dict[str, Policy], seed: int, max_steps: int = 20000, quiet: bool = True,
                 sim_fixes: bool | None = None, rules: str | None = None):
        from Simulator.simulators.tft_simulator import TFTConfig, parallel_env

        from tfteval import simfixes

        if sim_fixes is None:
            sim_fixes = os.environ.get("TFT_SIM_FIXES", "1") != "0"
        self.sim_fixes = list(simfixes.apply()) if sim_fixes else []
        self.rules = rules or os.environ.get("TFT_RULES", "set4")
        config = {"rules": self.rules} if self.rules != "set4" else {}  # older simulators have no rules option
        self.seed, self.max_steps, self.quiet = seed, max_steps, quiet
        self.seat_policies = seat_policies

        started = time.time()
        np.random.seed(seed % (2**32))  # the upstream rule bot uses numpy's global generator
        with self._stdout():
            self.env = parallel_env(TFTConfig(num_players=len(seat_policies), **config))
            self.observations, self.infos = self.env.reset(seed=seed)
        self.seats = list(self.env.possible_agents)
        if set(self.seats) != set(seat_policies):
            raise ValueError(f"lobby seats {sorted(seat_policies)} do not match env seats {self.seats}")

        self.fallback_policy = RandomPolicy()
        self.fallback_policy.reset(_seat_seed(seed, 10_000))
        for index, seat in enumerate(self.seats):
            seat_policies[seat].reset(_seat_seed(seed, index))

        self.placements: dict[str, int] = {}
        self.actions = {seat: 0 for seat in self.seats}
        self.fallbacks = {seat: 0 for seat in self.seats}
        self.health = {seat: 100.0 for seat in self.seats}
        self.next_place, self.steps = len(self.seats), 0
        self.closed = False
        self.seconds = time.time() - started

    # ------------------------------------------------------------------ state

    @property
    def round(self) -> int:
        """The round whose planning phase is under way (the `game_round` policies see)."""
        return int(self.env.unwrapped.game_round.current_round)

    @property
    def done(self) -> bool:
        return len(self.placements) >= len(self.seats) or self.steps >= self.max_steps or not self.env.agents

    def player(self, seat: str):
        """The simulator's player object for a seat, or None once it is out."""
        return self.env.unwrapped.player_manager.player_states.get(seat)

    # ------------------------------------------------------------------ play

    def _stdout(self):
        return contextlib.redirect_stdout(io.StringIO() if self.quiet else _Passthrough())

    def step(self) -> None:
        """One env step: every seat still in the game picks one action."""
        chosen = {}
        for seat in self.env.agents:
            if seat in self.placements:
                continue
            obs, info = self.observations.get(seat), self.infos.get(seat, {})
            try:
                chosen[seat] = self.seat_policies[seat].act(obs, info, seat, self.env)
            except NotImplementedError:
                raise
            except Exception:
                self.fallbacks[seat] += 1
                chosen[seat] = self.fallback_policy.act(obs, info, seat, self.env)
            self.actions[seat] += 1
        self.observations, _, terminated, _, self.infos = self.env.step(chosen)

        for seat in self.seats:
            player = self.infos.get(seat, {}).get("player")
            if player is not None:
                self.health[seat] = float(getattr(player, "health", self.health[seat]))
        out_now = sorted(
            (seat for seat in self.seats if terminated.get(seat, False) and seat not in self.placements),
            key=lambda seat: self.health[seat],
        )
        for seat in out_now:
            self.placements[seat] = self.next_place
            self.next_place -= 1
        self.steps += 1

    def run(self, until_round: int | None = None) -> "Game":
        """Play until the game ends, or until the planning phase of `until_round` begins."""
        started = time.time()
        with self._stdout():
            while not self.done and (until_round is None or self.round < until_round):
                self.step()
            if self.done and not self.closed:
                self.env.close()
                self.closed = True
        self.seconds += time.time() - started
        return self

    def result(self) -> GameResult:
        from tfteval import simfixes

        return GameResult(
            seed=self.seed,
            lobby={seat: self.seat_policies[seat].name for seat in self.seats},
            placements=dict(self.placements),
            steps=self.steps,
            finished=len(self.placements) == len(self.seats),
            seconds=round(self.seconds, 2),
            actions=dict(self.actions),
            fallbacks=dict(self.fallbacks),
            reproducible=os.environ.get("PYTHONHASHSEED", "random").isdigit(),
            sim_fixes=list(self.sim_fixes),
            sim_commit=simfixes.sim_commit(),
            rules=self.rules,
        )


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
    return Game(seat_policies, seed, max_steps, quiet, sim_fixes, rules).run().result()


class _Passthrough(io.TextIOBase):
    def write(self, text):
        import sys

        return sys.__stdout__.write(text)

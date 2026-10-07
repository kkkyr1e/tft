"""Play one full 8-player game and return finishing places."""

from __future__ import annotations

import contextlib
import copy
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
    sim: str = "default"  # simulator profile (SIM_PROFILES, with any overrides), see sim_options()
    sim_options: dict = field(default_factory=dict)  # the exact TFTConfig options it set (besides rules)
    carousel_pickers: list = field(default_factory=list)  # seats that picked on the carousel with their own picker

    def to_json(self) -> dict:
        return asdict(self)


def _seat_seed(game_seed: int, seat_index: int) -> int:
    return int(np.random.SeedSequence([game_seed, seat_index]).generate_state(1)[0])


# Simulator profiles: TFTConfig options of the fork (FORK_NOTES.md, "Realism options"), all off by
# default in the simulator. "realistic" turns on every rule fix and the keyed random streams:
# PvE losses cost HP, Fortune pays loot orbs, the 5-4 carousel table and random item-to-unit
# pairing, the next opponent drawn at combat time (planners see only the candidates), and one
# random stream per event (common random numbers for branch comparisons).
SIM_PROFILES = {
    "default": {},
    "realistic": {"pve_damage": True, "fortune_orbs": True, "carousel_fixes": True, "hide_next_opponent": True,
                  "rng_streams": "keyed"},
}
DEFAULT_SIM = "realistic"


def _option_value(text: str):
    low = text.strip().lower()
    if low in ("true", "on", "yes"):
        return True
    if low in ("false", "off", "no"):
        return False
    try:
        return int(low)
    except ValueError:
        return text.strip()


def sim_options(spec: str | None = None) -> tuple[str, dict]:
    """(profile spec, TFTConfig options) for a simulator profile: "realistic" or "default", optionally
    with overrides, e.g. "realistic,rng_streams=shared". None reads TFT_SIM (default DEFAULT_SIM)."""
    spec = (spec or os.environ.get("TFT_SIM") or DEFAULT_SIM).strip()
    name, *overrides = [part.strip() for part in spec.split(",") if part.strip()]
    if name not in SIM_PROFILES:
        raise ValueError(f"unknown simulator profile {name!r}; choose from {sorted(SIM_PROFILES)}")
    options = dict(SIM_PROFILES[name])
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"bad simulator option {item!r} in {spec!r}; expected key=value")
        options[key.strip()] = _option_value(value)
    return spec, options


def seat_picker(policy):
    """The carousel picker a seat policy exposes (`policy.carousel_picker()`), or None."""
    make = getattr(policy, "carousel_picker", None)
    return make() if callable(make) else None


class Game:
    """One game in progress. `play_game` runs it to the end; tfteval.branching saves and resumes it.

    Everything a game needs to continue lives on this object (env, seat policies, the runner's
    bookkeeping) except two process-global generators: numpy's global generator, which the rule bot
    draws its comp from with rng_streams="shared" (with "keyed" every seat's rule bot has its own
    generator inside the env), and Python's `random` module (unused as far as we know, saved anyway).
    """

    def __init__(self, seat_policies: dict[str, Policy], seed: int, max_steps: int = 20000, quiet: bool = True,
                 sim_fixes: bool | None = None, rules: str | None = None, sim: str | None = None,
                 pickers: bool | None = None):
        from Simulator.simulators.tft_simulator import TFTConfig, parallel_env

        from tfteval import simfixes

        if sim_fixes is None:
            sim_fixes = os.environ.get("TFT_SIM_FIXES", "1") != "0"
        self.sim_fixes = list(simfixes.apply()) if sim_fixes else []
        self.rules = rules or os.environ.get("TFT_RULES", "set4")
        self.sim, self.sim_options = sim_options(sim)
        config = {"rules": self.rules} if self.rules != "set4" else {}  # older simulators have no rules option
        config.update(self.sim_options)
        self.seed, self.max_steps, self.quiet = seed, max_steps, quiet
        self.seat_policies = seat_policies

        started = time.time()
        # Seat policies are reset before the env exists: the 1-1 carousel is picked inside env.reset,
        # so a seat's carousel picker (and the policy state it reads) must be ready by then. No reset
        # draws from numpy's global generator or `random`, so the order does not change a game.
        self.seats = [f"player_{i}" for i in range(len(seat_policies))]  # TFT_Simulator.possible_agents
        if set(self.seats) != set(seat_policies):
            raise ValueError(f"lobby seats {sorted(seat_policies)} do not match env seats {self.seats}")
        self.fallback_policy = RandomPolicy()
        self.fallback_policy.reset(_seat_seed(seed, 10_000))
        for index, seat in enumerate(self.seats):
            seat_policies[seat].reset(_seat_seed(seed, index))
        if pickers is None:
            pickers = os.environ.get("TFT_PICKERS", "1") != "0"
        chosen = {seat: seat_picker(seat_policies[seat]) for seat in self.seats} if pickers else {}
        chosen = {seat: picker for seat, picker in chosen.items() if picker is not None}
        self.carousel_pickers, self.pickers_on = sorted(chosen), bool(pickers)
        if chosen:
            config["carousel_pickers"] = chosen

        np.random.seed(seed % (2**32))  # the rule bot uses numpy's global generator (unless rng_streams="keyed")
        with self._stdout():
            self.env = parallel_env(TFTConfig(num_players=len(seat_policies), **config))
            self.observations, self.infos = self.env.reset(seed=seed)
        if list(self.env.possible_agents) != self.seats:
            raise RuntimeError(f"env seats {self.env.possible_agents} differ from {self.seats}")

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
        with light_board_copies(self.env):
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
            sim=getattr(self, "sim", "default"),  # games saved before the profiles existed ran the defaults
            sim_options=dict(getattr(self, "sim_options", {})),
            carousel_pickers=list(getattr(self, "carousel_pickers", [])),
        )


def play_game(seat_policies: dict[str, Policy], seed: int, max_steps: int = 20000, quiet: bool = True,
              sim_fixes: bool | None = None, rules: str | None = None, sim: str | None = None,
              pickers: bool | None = None) -> GameResult:
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

    `sim` picks the simulator profile (SIM_PROFILES): "realistic" (the default for new runs: PvE
    damage, Fortune orbs, carousel fixes, hidden next opponent, keyed random streams) or "default"
    (every fork option off, as the fork played before them), optionally with overrides such as
    "realistic,rng_streams=shared". Default from TFT_SIM. The result records the profile and the
    exact options (`sim`, `sim_options`).

    `pickers`: seats whose policy exposes a carousel picker (`carousel_picker()`, the plan-executor
    seats) pick their own carousel unit; the others take the most expensive. Default on;
    TFT_PICKERS=0 turns it off (e.g. to replay games made before the pickers).
    """
    return Game(seat_policies, seed, max_steps, quiet, sim_fixes, rules, sim, pickers).run().result()


def _share(obj, memo):
    return obj


@contextlib.contextmanager
def light_board_copies(env):
    """While the seats choose their actions, deep copies share the env's combat context.

    The rule bot scores candidate buys and swaps on copies of the board (`deepcopy(player.board)` in
    Default_Agent.round_3_10 and round_11_end, once per bench unit and board cell; about 100 000
    copies a game with 7 rule bots), and every champion holds the env's combat context
    (`champion.ctx`), so each copy also copied the context: its random generators and combat lists,
    and with rng_streams="keyed" the KeyedStreams, which holds the Game_Round (`streams.game_round`)
    and through it every player. A board copy at 2-3 took 12 ms under keyed streams instead of 1 ms;
    a game of rule bots ran about 6x slower than with shared streams. The copies are only scored
    (`rank_comp` reads names, costs, stars and traits), nothing is drawn from or written to their
    context, and choosing an action draws nothing from the env's streams (the env performs the
    actions in env.step, outside this block), so sharing the context for that time changes nothing
    in a game (tests/test_sim_profiles.py replays a game with and without it) and makes each copy
    about 3x cheaper with either kind of streams. Implemented as a deepcopy dispatch entry for the
    context's class while the seats choose (nothing in the simulator is edited)."""
    ctx = getattr(getattr(env, "unwrapped", env), "combat_ctx", None)
    cls = type(ctx) if ctx is not None else None
    if cls is None or cls in copy._deepcopy_dispatch or hasattr(cls, "__deepcopy__"):
        yield
        return
    copy._deepcopy_dispatch[cls] = _share
    try:
        yield
    finally:
        copy._deepcopy_dispatch.pop(cls, None)


class _Passthrough(io.TextIOBase):
    def write(self, text):
        import sys

        return sys.__stdout__.write(text)

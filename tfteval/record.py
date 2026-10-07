"""Per-seat, per-round trajectory records: what a seat had, did and got in every round of a game.

Opt-in: `Game(..., record=True)` (or a list of seats or policy names), `play_game(..., record=...)`,
`scripts/run_lobby.py --record`, or TFT_RECORD in the environment, which the runner reads in every
worker when `record` is not given: `1` / `all` records every seat, `0` or empty none, otherwise a comma
list of seats (`player_3`) or policy names (`hero`). The rows end up in GameResult.records as
`{seat: [row, ...]}`, one row per planning phase the seat played, in round order; the round of
elimination is in GameResult.eliminated and the placement in GameResult.placements.

Row (keys absent when they do not apply; a unit is `[name, stars, cost]`, with its `[items]` appended
when it holds any, and the chosen trait after them on a chosen unit; unit_items / unit_chosen read them):

    round       planning phase index (tfteval.stages: 9 = 3-1, 15 = 4-1, ...)
    hp, gold, level, xp, streak
                at the start of the planning phase (streak: +n wins, -n losses, tfteval.public)
    end         {"gold", "level", "xp"} at the end of the planning phase, just before the round is played
    board, bench, items
                fielded units, bench units and the item bench at the end of the planning phase (before the
                simulator's end-of-turn autofill)
    actions     the seat's actions in the round by type (pass, xp, roll, buy, sell, move, item), nonzero only
    illegal     actions the observation's action mask did not allow (the simulator ignores or rejects them)
    fallbacks   actions the runner replaced with a random legal one because the policy raised
    roll_low    the lowest gold left right after one of this round's (legal) rolls
    fight       "W" or "L" for the PvP fight that followed (a draw is a loss for both, as in the simulator),
                "pve" for a round against monsters
    hp_lost     HP lost in that round (PvE losses cost HP only with the `pve_damage` option)
    comp        the target comp (trait) at the end of the planning phase: the plan executor's for a plan
                seat, the seat's own rule bot's otherwise (none before it picks one, at 3-3)
    contested   copies of each comp's units on the other boards at the start of the round, nonzero comps
                only (tfteval.public's contested_by_comp, what a planner sees)
    plan        plan-executor seats: the round's plan fields (tfteval.planner PLAN_KEYS and EXTRA_KEYS)
    knobs       plan-executor seats: what the executor was told after compile_knobs (level_to, roll_floor,
                xp_buys, survival)
    stance, why stance seats: the stance picked and its reason

Recording reads the game and draws no random numbers, so a recorded game plays exactly as an
unrecorded one (tests/test_record.py replays one both ways). The end-of-planning state and the fight
are read by wrapping the env's `game_round.play_game_round` for the duration of one env.step (an
attribute on that one object, removed again before the step returns, so snapshots never hold it);
nothing in the simulator is edited.
"""

from __future__ import annotations

import contextlib
import os

ACTION_KINDS = ("pass", "xp", "roll", "buy", "sell", "move", "item")  # simulator action type 0..6
MASK_COLUMNS = 38


def action_index(action) -> int:
    """Index of a simulator action [type, a, b] in the flat 55x38 action mask
    (Simulator/encoding/token/action.py, fetch_action_mask and action_space_to_action)."""
    kind, a, b = (int(x) for x in list(action)[:3])
    if kind == 0:
        return 52 * MASK_COLUMNS
    if kind == 1:
        return 53 * MASK_COLUMNS
    if kind == 2:
        return 54 * MASK_COLUMNS
    if kind == 3:
        return (47 + a) * MASK_COLUMNS
    if kind == 4:
        return a * MASK_COLUMNS + 37
    if kind == 5:
        return a * MASK_COLUMNS + b
    if kind == 6:  # [6, unit location, item bench slot]
        return (37 + b) * MASK_COLUMNS + a
    raise ValueError(f"unknown action type {kind}")


def legal(action, mask) -> bool:
    """Whether the observation's action mask allowed `action` (no mask: assumed legal)."""
    if mask is None:
        return True
    import numpy as np

    flat = np.asarray(mask).reshape(-1)
    try:
        index = action_index(action)
    except (ValueError, TypeError):
        return False
    return 0 <= index < flat.size and bool(flat[index] > 0)


def unit_entry(champ) -> list:
    """[name, stars, cost], then [items] when it holds any or is chosen, then the chosen trait."""
    entry = [champ.name, int(champ.stars), int(champ.cost)]
    if champ.items or champ.chosen:
        entry.append([str(i) for i in champ.items])
    if champ.chosen:
        entry.append(str(champ.chosen))
    return entry


def unit_items(entry) -> list:
    """Items of a recorded unit entry."""
    return list(entry[3]) if len(entry) > 3 else []


def unit_chosen(entry):
    """Chosen trait of a recorded unit entry, or None."""
    return entry[4] if len(entry) > 4 else None


def record_seats(spec, seat_policies: dict) -> list:
    """Seats to record. `spec`: None reads TFT_RECORD; True / "1" / "all" every seat; False / "" / "0"
    none; else seats or policy names, as an iterable or a comma-separated string."""
    if spec is None:
        spec = os.environ.get("TFT_RECORD", "")
    if spec is True:
        return sorted(seat_policies)
    if spec is False or spec is None:
        return []
    if isinstance(spec, str):
        text = spec.strip()
        if text in ("", "0", "false", "off", "no"):
            return []
        if text in ("1", "all", "true", "on", "yes"):
            return sorted(seat_policies)
        spec = [part.strip() for part in text.split(",") if part.strip()]
    wanted = set(spec)
    seats = sorted(seat for seat, policy in seat_policies.items()
                   if seat in wanted or getattr(policy, "name", None) in wanted)
    unknown = wanted - set(seats) - {getattr(seat_policies[s], "name", None) for s in seats}
    if unknown:
        raise ValueError(f"record: no seat or policy named {sorted(unknown)} in {sorted(seat_policies)}")
    return seats


def _plan_policy(policy):
    """The plan-executor policy behind a seat (also through a wrapper with a `.policy`), or None."""
    for obj in (policy, getattr(policy, "policy", None)):
        if obj is not None and getattr(obj, "executor", None) is not None and hasattr(obj, "traits"):
            return obj
    return None


def _comps():
    from tfteval.executor import COMPS

    return COMPS


class Recorder:
    """Collects the rows of the recorded seats while a Game runs (tfteval.runner.Game.step calls it).
    Holds plain data only, so a game with a recorder can be snapshotted (tfteval.branching)."""

    def __init__(self, seats):
        self.seats = list(seats)
        self.rows = {seat: [] for seat in self.seats}
        self._before = {}  # seat -> (hp, len(match_history)) just before the round is played

    # ------------------------------------------------------------------ planning phase
    def observe(self, game, seat: str, info: dict) -> None:
        """Before the seat picks an action: open a row when its planning phase is new."""
        rows = self.rows[seat]
        idx = int(info.get("game_round", game.round))
        if rows and rows[-1]["round"] == idx:
            return
        player = game.player(seat)
        if player is None:
            return
        from tfteval import public

        players = public.alive_players(game.env)
        seen = public.seen_copies(players, seat)
        contested = {trait: n for trait, units in _comps().items() if (n := sum(seen[u] for u in units))}
        rows.append({"round": idx, "hp": int(player.health), "gold": int(player.gold), "level": int(player.level),
                     "xp": int(player.exp), "streak": public.streak(player), "actions": {}, "illegal": 0,
                     "fallbacks": 0, "contested": contested})

    def acted(self, game, seat: str, action, observation, fallback: bool) -> None:
        """After the seat picked `action` (before the env performs it)."""
        rows = self.rows[seat]
        if not rows:
            return
        row = rows[-1]
        kind = ACTION_KINDS[int(action[0])] if 0 <= int(action[0]) < len(ACTION_KINDS) else str(int(action[0]))
        row["actions"][kind] = row["actions"].get(kind, 0) + 1
        mask = observation.get("action_mask") if isinstance(observation, dict) else None
        ok = legal(action, mask)
        if not ok:
            row["illegal"] += 1
        if fallback:
            row["fallbacks"] += 1
        player = game.player(seat)
        if kind == "roll" and ok and player is not None:
            left = int(player.gold) - int(getattr(player, "refresh_cost", 2))
            row["roll_low"] = min(row.get("roll_low", left), left)
        if "plan" not in row:
            self._plan(game.seat_policies[seat], row)

    @staticmethod
    def _plan(policy, row: dict) -> None:
        plan_seat = _plan_policy(policy)
        if plan_seat is None or plan_seat.round != row["round"]:
            return
        from tfteval.planner import EXTRA_KEYS, PLAN_KEYS

        executor = plan_seat.executor
        plan, knobs = executor.plan or {}, executor.knobs or {}
        row["plan"] = {k: plan[k] for k in PLAN_KEYS + EXTRA_KEYS if k in plan}
        row["knobs"] = {k: knobs[k] for k in ("level_to", "roll_floor", "xp_buys", "survival") if k in knobs}
        for key in ("stance", "why"):
            if plan.get(key) is not None:
                row[key] = plan[key]

    # ------------------------------------------------------------------ the round itself
    @contextlib.contextmanager
    def combat(self, game):
        """Around one env.step: if the step plays the round, note every recorded seat's end-of-planning
        state just before, and its fight and HP loss just after."""
        gr = getattr(game.env.unwrapped, "game_round", None)
        if gr is None or "play_game_round" in vars(gr):
            yield
            return
        original = gr.play_game_round
        recorder = self

        def play_game_round(*args, **kwargs):
            recorder._end_of_planning(game, int(gr.current_round))
            out = original(*args, **kwargs)
            recorder._after_round(game)
            return out

        gr.play_game_round = play_game_round
        try:
            yield
        finally:
            vars(gr).pop("play_game_round", None)

    def _alive(self, game):
        states = game.env.unwrapped.player_manager.player_states
        for seat in self.seats:
            player = states.get(seat)
            rows = self.rows[seat]
            if player is not None and rows:
                yield seat, player, rows[-1]

    def _end_of_planning(self, game, idx: int) -> None:
        from tfteval import public

        self._before = {}
        for seat, player, row in self._alive(game):
            if row["round"] != idx or float(player.health) <= 0:
                continue
            row["end"] = {"gold": int(player.gold), "level": int(player.level), "xp": int(player.exp)}
            row["board"] = [unit_entry(u) for u in public.board_units(player)]
            row["bench"] = [unit_entry(u) for u in player.bench if public.is_unit(u)]
            row["items"] = [str(i) for i in player.item_bench if i]
            comp = self._comp(game, seat, player)
            if comp:
                row["comp"] = comp
            self._before[seat] = (float(player.health), len(getattr(player, "match_history", ())))

    def _after_round(self, game) -> None:
        for seat, player, row in self._alive(game):
            if seat not in self._before:
                continue
            hp, fights = self._before[seat]
            history = getattr(player, "match_history", [])
            if len(history) > fights:
                row["fight"] = "W" if history[-1] == 1 else "L"
            else:
                row["fight"] = "pve"
            row["hp_lost"] = int(round(hp - float(player.health)))
        self._before = {}

    @staticmethod
    def _comp(game, seat: str, player) -> str | None:
        plan_seat = _plan_policy(game.seat_policies[seat])
        if plan_seat is not None:
            number = plan_seat.executor.comp_number
            return plan_seat.traits[number] if number >= 0 else None
        agent = getattr(player, "default_agent", None)
        number = getattr(agent, "comp_number", -1)
        if number is None or number < 0:
            return None
        from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS

        return TEAM_COMP_TRAITS[number]

    # ------------------------------------------------------------------ result
    def records(self) -> dict:
        return {seat: [dict(row) for row in rows] for seat, rows in self.rows.items()}

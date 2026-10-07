"""The plan executor (see tfteval/planner.py for what it does with each plan field). A module of its
own because it subclasses the simulator's rule bot, so importing it needs the simulator on the path;
planner.py imports it lazily. It lives at module level (not inside a function) so that a game holding
one can be pickled, which tfteval/branching.py relies on."""

from __future__ import annotations

import math
from collections import Counter

from Simulator.game.pool_stats import cost_star_values
from Simulator.generators.default_agent import Default_Agent
from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS, TEAM_COMPS
from Simulator.utils import coord_to_x_y, x_y_to_1d_coord

from tfteval import public
from tfteval.planner import ACTIONS_PER_ROUND, compile_knobs, xp_to_level


def strength(unit) -> int:
    return cost_star_values[unit.cost - 1][unit.stars - 1] + 2 * len(unit.items) + (1 if unit.chosen else 0)


class PlanExecutor(Default_Agent):
    def __init__(self):
        super().__init__()
        self.plan = {"comp": None, "level_to": 1, "roll_floor": 999, "carry": None}
        self.knobs = compile_knobs(self.plan, {"round": 1, "gold": 0, "level": 1, "xp": 0, "hp": 100})
        self.pivot_pending = False
        self.restore_pending = False
        self.xp_bought = 0
        self.rule_calls, self.took_over = 0, False  # this round: rule-bot calls, executor-only actions
        self.stats = Counter()

    def begin_round(self, plan: dict, knobs: dict, game_round: int | None = None) -> None:
        """New round: the plan and its compiled knobs (compile_knobs)."""
        if game_round is not None and self.took_over and not self.rule_calls:
            # The executor used every action of the last round, so the rule bot never saw it and
            # would not notice that this one is new (it resets its checks on current == next).
            self.next_round = game_round
        self.rule_calls, self.took_over = 0, False
        was_fodder = self.knobs["fodder"]
        self.plan, self.knobs, self.xp_bought = plan, knobs, 0
        if was_fodder and not self.knobs["fodder"]:
            self.restore_pending = True
        if self.knobs["fodder"]:
            self.restore_pending = False
            self.stats["fodder_rounds"] += 1
        if self.knobs["survival"]:
            self.stats["survival_rounds"] += 1

    # ---- comp
    def set_comp(self, trait):
        if trait not in TEAM_COMP_TRAITS:
            return
        number = TEAM_COMP_TRAITS.index(trait)
        if self.comp_number == -1:
            self.comp_number = number
        elif number != self.comp_number:  # pivot
            self.comp_number = number
            self.pivot_pending = True
            self.round_11_end_checks = [True for _ in range(5)]
            self.stats["pivots"] += 1

    def pivot_cleanup(self, player):
        """One action of the switch to the new comp, None when done: decide_comp's clean-up
        (sell off-comp bench units that are not pairs, and chosen units of another trait;
        decide_comp compares the chosen trait with the unit list, so it sells every chosen unit,
        here only off-trait ones and on the board only off-comp ones), then field_comp_swap."""
        units, trait = TEAM_COMPS[self.comp_number], TEAM_COMP_TRAITS[self.comp_number]
        for i, unit in enumerate(player.bench):
            if unit and unit.name not in units and (unit.name + "_" + str(unit.stars)) not in self.pairs:
                return "4_" + str(i + 28)
            if unit and unit.chosen and unit.chosen != trait:
                return "4_" + str(i + 28)
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                unit = player.board[x][y]
                if unit and unit.chosen and unit.chosen != trait and unit.name not in units:
                    return "4_" + str(x_y_to_1d_coord(x, y))
        return self.field_comp_swap(player)

    def field_comp_swap(self, player):
        """Swap the weakest off-comp board unit for the strongest comp unit on the bench, if that
        one is no weaker; None when there is nothing to swap. The rule bot's own bench-to-board
        check (round_11_end) puts the first comp unit it finds on the bench in place of the first
        off-comp board unit, however weak the comp unit is; with field_comp those swaps are dropped
        (owns_swap) and made here instead. On the upstream simulator that check never fired (it
        tested `champion in BASE_CHAMPION_LIST`, an object against names), so there the rule bot
        fielded units only into empty slots; the fork fixed it."""
        units = TEAM_COMPS[self.comp_number]
        board = []
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                unit = player.board[x][y]
                if public.is_unit(unit) and unit.name not in units:
                    board.append((strength(unit), x_y_to_1d_coord(x, y)))
        bench = [(strength(u), 28 + i) for i, u in enumerate(player.bench) if public.is_unit(u) and u.name in units]
        if board and bench:
            (weak, out), (strong, into) = min(board), max(bench)
            if strong >= weak:
                return "5_" + str(out) + "_" + str(into)
        return None

    # ---- fodder board
    @staticmethod
    def _units(player):
        out = []
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                unit = player.board[x][y]
                if public.is_unit(unit):
                    out.append((strength(unit), 0, x_y_to_1d_coord(x, y)))
        for i, unit in enumerate(player.bench):
            if public.is_unit(unit):
                out.append((strength(unit), 1, 28 + i))
        return out

    def _wanted(self, player, weakest: bool):
        """Locations of the units that should be on the board: the weakest (or strongest)
        max_units, preferring units already on the board on ties."""
        units = self._units(player)
        k = min(player.max_units, len(units))
        order = sorted(units, key=lambda t: (t[0] if weakest else -t[0], t[1], t[2]))
        return units, {loc for _, _, loc in order[:k]}

    def arrange(self, player, weakest: bool):
        units, want = self._wanted(player, weakest)
        sign = 1 if weakest else -1
        out = sorted((t for t in units if t[2] < 28 and t[2] not in want), key=lambda t: -sign * t[0])
        into = sorted((t for t in units if t[2] >= 28 and t[2] in want), key=lambda t: sign * t[0])
        if out and into:
            return "5_" + str(out[0][2]) + "_" + str(into[0][2])  # swap
        if into and player.num_units_in_play < player.max_units:
            loc = into[0][2]
            command = self.move_bench_to_empty_board(player, loc, player.bench[loc - 28].name)
            if command != "0":
                return command
            for x in range(len(player.board)):
                for y in range(len(player.board[x])):
                    if player.board[x][y] is None:
                        return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(loc)
        return None

    def strong_view(self, player):
        """The player as the rule bot would see it with its strongest units fielded, and the
        location swaps that turn the real arrangement into that one (a dict, both directions).

        In fodder mode the rule bot decides on this view, so it buys, sells and levels as it would
        with its normal board: it does not buy pairs of fodder units or units that only improve
        the fodder board, and its interest sales take the units it would have on its bench."""
        units = self._units(player)
        _, keep = self._wanted(player, weakest=False)
        out = sorted((t for t in units if t[2] < 28 and t[2] not in keep), key=lambda t: (t[0], t[2]))
        into = sorted((t for t in units if t[2] >= 28 and t[2] in keep), key=lambda t: (-t[0], t[2]))
        board, bench, swaps = [list(col) for col in player.board], list(player.bench), {}
        for (_, _, cell), (_, _, slot) in zip(out, into):
            x, y = coord_to_x_y(cell)
            board[x][y], bench[slot - 28] = player.bench[slot - 28], player.board[x][y]
            swaps[cell], swaps[slot] = slot, cell
        return StrongView(player, board, bench), swaps

    @staticmethod
    def translate(command: str, swaps: dict) -> str:
        """A rule-bot command on the strong view, as a command on the real board and bench."""
        kind, *args = command.split("_")
        if kind == "4" and len(args) == 1:
            return "4_" + str(swaps.get(int(args[0]), int(args[0])))
        if kind == "5" and len(args) == 2:
            return "5_" + "_".join(str(swaps.get(int(a), int(a))) for a in args)
        if kind == "6" and len(args) == 2:
            return "6_" + str(swaps.get(int(args[0]), int(args[0]))) + "_" + args[1]
        return command

    def fodder_filter(self, command: str, player) -> str:
        """In fodder mode, drop commands (already translated to the real board) that would field a
        benched unit or bench a fielded one, put an item on a unit, or sell a fielded unit on the
        round's last action (the end-of-round autofill would field a bench unit in its place).
        (The rule bot's swaps never reach this: fodder_rule_command drops them first, owns_swap.)"""
        kind, *args = command.split("_")
        if kind == "6":
            return "0"
        if kind == "5" and len(args) == 2 and max(int(args[0]), int(args[1])) >= 28:
            return "0"
        if kind == "4" and len(args) == 1 and int(args[0]) < 28 \
                and getattr(player, "actions_remaining", ACTIONS_PER_ROUND) <= 1:
            return "0"
        return command

    def fodder_rule_command(self, player, shop, game_round, mask) -> str:
        """The rule bot's command in fodder mode: decided on the strong view, translated, filtered.
        When a command is dropped, the rule bot's step that produced it (swapping a bench unit in,
        placing items, fixing positions) is marked done for the round, as if it had been carried
        out, and the rule bot is asked again, so the economy steps after it (interest sales) still
        run. The swap is caught before translation: on the strong view it can translate into a
        board-to-board move that would pass the filter and repeat on every action."""
        for _ in range(5):
            command = "0"
            view, swaps = self.strong_view(player)
            raw = self.policy(view, shop, game_round, mask)
            if self.owns_swap(raw, view, game_round):
                self.stats["swaps_dropped"] += 1
                self.round_3_10_checks[2] = self.round_11_end_checks[2] = False
                continue
            command = self.fodder_filter(self.translate(raw, swaps), player)
            if command != "0" or raw in ("0", "1", "2"):
                return command
            self.stats["filtered"] += 1
            if raw.startswith("6_") and len(self.round_3_10_checks) > 5:
                self.round_3_10_checks[5] = False
            elif raw.startswith("5_"):
                self.round_3_10_checks[3] = self.round_11_end_checks[3] = False
            else:
                break
        return command

    # ---- the rule bot's bench-to-board swap
    def owns_swap(self, command: str, player, game_round: int) -> bool:
        """Whether `command` is a rule-bot swap of a bench unit onto the board ("5_<board>_<bench>",
        its round_3_10 / round_11_end swap check) that a knob owns. fodder and hold field the units
        themselves (arrange), so every such swap is theirs. field_comp owns the swaps of a comp unit
        for an off-comp unit (field_comp_swap makes them, strongest first and never a weaker unit);
        the rule bot's other swaps (better comp score, see rank_comp) stay its own."""
        kind, *args = command.split("_")
        if kind != "5" or len(args) != 2:
            return False
        board, bench = int(args[0]), int(args[1])
        if not board < 28 <= bench:
            return False  # board to board, or a bench unit into an empty slot (bench first)
        if self.knobs["fodder"] or self.knobs.get("hold"):
            return True
        if self.knobs["field_comp"] and game_round >= 11 and self.comp_number >= 0:
            units = TEAM_COMPS[self.comp_number]
            x, y = coord_to_x_y(board)
            out, into = player.board[x][y], player.bench[bench - 28]
            return bool(out and into and into.name in units and out.name not in units)
        return False

    # ---- hold (plan field `hold`): keep units and items back, field the strongest units
    @staticmethod
    def hold_filter(command: str, player) -> str:
        """Drop rule-bot item placements, and bench sales while the bench has room (the sales for
        interest); a sale that makes room on a full bench goes through. (The rule bot's swaps are
        dropped before this, in act(): hold fields the units itself, see owns_swap.)"""
        kind, *args = command.split("_")
        if kind == "6":
            return "0"
        if kind == "4" and args and int(args[0]) >= 28 and not player.bench_full():
            return "0"
        return command

    # ---- economy
    def _xp_owed(self, player) -> int:
        need = 0
        if player.level < min(self.knobs["level_to"], player.max_level):
            need = math.ceil(xp_to_level(player.level, player.exp, self.knobs["level_to"]) / 4)
        return max(need, self.knobs["xp_buys"] - self.xp_bought)

    def wants_xp(self, player, mask) -> bool:
        if player.gold < 4 or not mask[53][0]:
            return False
        if self.knobs["fodder"] and getattr(player, "actions_remaining", ACTIONS_PER_ROUND) <= 1:
            return False  # a level-up on the last action would let autofill field a strong unit
        return (player.level < min(self.knobs["level_to"], player.max_level)
                or self.xp_bought < self.knobs["xp_buys"])

    def take_over(self, player, shop, game_round, mask):
        """Actions of the optional plan fields that go before the rule bot: pivot clean-up,
        fodder (or restore) moves, and xp the deadline round cannot wait for."""
        knobs = self.knobs
        if self.pivot_pending and game_round >= 11 and not (game_round == 11 and self.round_11_clean_up):
            command = self.pivot_cleanup(player)
            if command:
                self.stats["pivot_sells"] += 1
                return command
            self.pivot_pending = False
        if knobs["fodder"] or self.restore_pending:
            command = self.arrange(player, weakest=knobs["fodder"])
            if command:
                self.stats["fodder_moves" if knobs["fodder"] else "restore_moves"] += 1
                return command
            self.restore_pending = False
        if knobs.get("hold") and not knobs["fodder"]:
            command = self.arrange(player, weakest=False)
            if command:
                self.stats["hold_moves"] += 1
                return command
        if knobs["field_comp"] and not knobs["fodder"] and game_round >= 11 and self.comp_number >= 0:
            command = self.field_comp_swap(player)
            if command:
                self.stats["comp_swaps"] += 1
                return command
        if knobs["xp_priority"] and self.wants_xp(player, mask) \
                and self._xp_owed(player) >= getattr(player, "actions_remaining", ACTIONS_PER_ROUND):
            self.xp_bought += 1
            self.stats["xp"] += 1
            self.stats["xp_preempt"] += 1
            return "1"
        return None

    def act(self, player, shop, game_round, mask):
        knobs = self.knobs
        if game_round >= 11 and knobs.get("comp"):
            self.set_comp(knobs["comp"])
        command = self.take_over(player, shop, game_round, mask)
        if command:
            self.took_over = True
            return command
        self.rule_calls += 1
        if knobs["fodder"]:
            command = self.fodder_rule_command(player, shop, game_round, mask)
        else:
            command = self.policy(player, shop, game_round, mask)
            if self.owns_swap(command, player, game_round):
                # A knob owns this swap. The rule bot's swap check is done for this round, as if it
                # had found nothing to swap (it would propose the same swap on every call), and the
                # bot is asked again; a roll re-opens the check, as it re-opens every other check.
                self.stats["swaps_dropped"] += 1
                self.round_3_10_checks[2] = self.round_11_end_checks[2] = False
                command = self.policy(player, shop, game_round, mask)
            if knobs.get("hold"):
                filtered = self.hold_filter(command, player)
                if filtered != command:
                    self.stats["hold_filtered"] += 1
                    command = filtered
        if command not in ("0", "1", "2"):
            return command
        item = self.place_item(player, mask) if game_round > 2 and not knobs["fodder"] else None
        if item:
            return item
        if self.wants_xp(player, mask):
            self.xp_bought += 1
            self.stats["xp"] += 1
            return "1"
        if player.gold - 2 >= knobs["roll_floor"] and mask[54][0]:
            self.stats["rolls"] += 1
            self.round_11_end_checks = [True for _ in range(5)]
            self.round_3_10_checks = [True for _ in range(6)]
            return "2"
        return "0"

    def place_item(self, player, mask):
        if self.knobs.get("hold"):
            return None  # hold: the items wait until the hold ends
        items = [(i, it) for i, it in enumerate(player.item_bench)
                 if it is not None and it not in ("champion_duplicator", "spatula")]
        if not items:
            return None
        champs = []
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                unit = player.board[x][y]
                if unit and unit.name != "sandguard":
                    first = 1 if unit.name == self.knobs.get("carry") else 0
                    champs.append(((first, unit.cost * unit.stars), x_y_to_1d_coord(x, y)))
        champs.sort(reverse=True)
        for _, coord in champs:
            for idx, _item in items:
                if mask[37 + idx][coord]:
                    return "6_" + str(coord) + "_" + str(idx)
        return None


class StrongView:
    """A stand-in for the player with another arrangement of the same units (see
    PlanExecutor.strong_view): the rule bot reads board and bench from here, everything else from
    the player. Only lives for one rule-bot call; nothing is changed through it."""

    def __init__(self, player, board, bench):
        self._player, self.board, self.bench = player, board, bench

    def __getattr__(self, name):
        return getattr(self._player, name)

    def bench_full(self):
        return all(self.bench)

    @property
    def team_tiers(self):  # decide_comp picks a comp from the traits of the fielded units
        chosen = next((u.chosen for col in self.board for u in col if u and u.chosen), "")
        return Default_Agent.update_team_tiers(None, self.board, chosen)[1]


COMPS = dict(zip(TEAM_COMP_TRAITS, TEAM_COMPS))

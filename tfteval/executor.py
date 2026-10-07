"""The plan executor (see tfteval/planner.py for what it does with each plan field). A module of its
own because it subclasses the simulator's rule bot, so importing it needs the simulator on the path;
planner.py imports it lazily. It lives at module level (not inside a function) so that a game holding
one can be pickled, which tfteval/branching.py relies on."""

from __future__ import annotations

import math
from collections import Counter

from Simulator.battle.stats import COST
from Simulator.game.pool_stats import cost_star_values
from Simulator.generators.default_agent import Default_Agent
from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS, TEAM_COMPS
from Simulator.utils import x_y_to_1d_coord

from tfteval import public
from tfteval.planner import ACTIONS_PER_ROUND, FODDER_STRENGTH, compile_knobs, xp_to_level


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
        one is no weaker; None when there is nothing to swap. This is what the rule bot's own
        bench-to-board check means to do; it never fires, because it tests
        `champion in BASE_CHAMPION_LIST` (an object against names), so the rule bot fields units
        only into empty slots and its comp decides what it buys, not what it fields."""
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

    def fodder_buy(self, player, shop, mask):
        """Buy a 1-cost unit to stand in for a stronger unit that would otherwise be fielded.
        A 1-star sells back at cost, so this only ties gold up for a while. It never crosses an
        interest bracket, never buys a third copy (no 2-star) and keeps two bench slots free."""
        units, want = self._wanted(player, weakest=True)
        if all(t[0] <= FODDER_STRENGTH for t in units if t[2] in want):
            return None
        if player.bench.count(None) < 2:
            return None
        owned = Counter(u.name for u in public.board_units(player) + [c for c in player.bench if c]
                        if u.stars == 1)
        bracket = min(player.gold // 10, 5)
        for i, name in enumerate(shop):
            if not name or name.endswith("_c") or not mask[47 + i][0] or COST.get(name) != 1:
                continue
            if owned[name] >= 2 or player.gold < 1 or min((player.gold - 1) // 10, 5) < bracket:
                continue
            return "3_" + str(i)
        return None

    def fodder_filter(self, command: str, player) -> str:
        """In fodder mode, drop rule-bot commands that would field a benched unit or put an item
        on a unit, and keep the units that would be fielded otherwise: their sale is dropped, or
        redirected to the weakest other bench unit when the bench is full."""
        kind, *args = command.split("_")
        if kind == "6":
            return "0"
        if kind == "5" and len(args) == 2 and max(int(args[0]), int(args[1])) >= 28:
            return "0"
        if kind == "4" and args and int(args[0]) >= 28:
            units, keep = self._wanted(player, weakest=False)
            if int(args[0]) in keep:
                if not player.bench_full():
                    return "0"
                spare = sorted(t for t in units if t[2] >= 28 and t[2] not in keep)
                return "4_" + str(spare[0][2]) if spare else command
        return command

    # ---- hold (plan field `hold`): the fodder board's filters, with the strongest units fielded
    @staticmethod
    def hold_filter(command: str, player) -> str:
        """Drop rule-bot item placements, and bench sales while the bench has room (the sales for
        interest); a sale that makes room on a full bench goes through."""
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
        fodder (or restore) moves and buys, and xp the deadline round cannot wait for."""
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
            command = self.fodder_buy(player, shop, mask) if knobs["fodder"] else None
            if command:
                self.stats["fodder_buys"] += 1
                return command
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
        command = self.policy(player, shop, game_round, mask)
        if knobs["fodder"]:
            filtered = self.fodder_filter(command, player)
            if filtered != command:
                self.stats["filtered"] += 1
                command = filtered
        elif knobs.get("hold"):
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


COMPS = dict(zip(TEAM_COMP_TRAITS, TEAM_COMPS))

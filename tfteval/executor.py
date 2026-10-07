"""The plan executor (see tfteval/planner.py). A module of its own because it subclasses the
simulator's rule bot, so importing it needs the simulator on the path; planner.py imports it lazily.
It lives at module level (not inside a function) so that a game holding one can be pickled,
which tfteval/branching.py relies on."""

from __future__ import annotations

from Simulator.generators.default_agent import Default_Agent
from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS, TEAM_COMPS
from Simulator.utils import x_y_to_1d_coord


class PlanExecutor(Default_Agent):
    def __init__(self):
        super().__init__()
        self.plan = {"comp": None, "level_to": 1, "roll_floor": 999, "carry": None}

    def set_comp(self, trait):
        if trait in TEAM_COMP_TRAITS and self.comp_number == -1:
            self.comp_number = TEAM_COMP_TRAITS.index(trait)

    def place_item(self, player, mask):
        items = [(i, it) for i, it in enumerate(player.item_bench)
                 if it is not None and it not in ("champion_duplicator", "spatula")]
        if not items:
            return None
        champs = []
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                unit = player.board[x][y]
                if unit and unit.name != "sandguard":
                    first = 1 if unit.name == self.plan.get("carry") else 0
                    champs.append(((first, unit.cost * unit.stars), x_y_to_1d_coord(x, y)))
        champs.sort(reverse=True)
        for _, coord in champs:
            for idx, _item in items:
                if mask[37 + idx][coord]:
                    return "6_" + str(coord) + "_" + str(idx)
        return None

    def act(self, player, shop, game_round, mask):
        if game_round >= 11 and self.plan.get("comp"):
            self.set_comp(self.plan["comp"])
        command = self.policy(player, shop, game_round, mask)
        if command not in ("0", "1", "2"):
            return command
        item = self.place_item(player, mask) if game_round > 2 else None
        if item:
            return item
        plan = self.plan
        if player.level < min(plan["level_to"], player.max_level) and player.gold >= 4 and mask[53][0]:
            return "1"
        if player.gold - 2 >= plan["roll_floor"] and mask[54][0]:
            self.round_11_end_checks = [True for _ in range(5)]
            self.round_3_10_checks = [True for _ in range(6)]
            return "2"
        return "0"


COMPS = dict(zip(TEAM_COMP_TRAITS, TEAM_COMPS))

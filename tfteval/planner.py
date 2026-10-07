"""Plan-and-execute seat: one typed plan per round, a rule executor turns it into atomic actions.

See docs/PLAN.md section 5. The executor is the simulator's rule bot with three changes:

* its own levelling and rerolling are switched off; the plan decides them (`level_to`, `roll_floor`);
* the target comp comes from the plan (`comp`) instead of a random pick at round 11;
* items are placed in every round (the rule bot only places them before round 11), on the
  plan's `carry` first.

Planners produce the plan:

* `MimicPlanner` reproduces the rule bot's own economy, so `plan:mimic` against `rule`
  measures the executor changes alone (mainly late item placement).
* `LLMPlanner` asks a model once per round. The model is any shell command that reads the
  prompt on stdin and prints an answer containing one JSON object; every call is logged and
  cached by prompt hash so a game can be replayed without calling the model again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from tfteval.policies import MASK_SHAPE

PLAN_KEYS = ("comp", "level_to", "roll_floor", "carry")


# --------------------------------------------------------------------------- state

def _unit(champ) -> dict:
    out = {"name": champ.name, "star": int(champ.stars), "cost": int(champ.cost)}
    if champ.items:
        out["items"] = list(champ.items)
    if champ.chosen:
        out["chosen"] = champ.chosen
    return out


def describe(player, shop, game_round: int, env) -> dict:
    """What a human player can see at the start of a round, as plain data."""
    board = []
    for x in range(len(player.board)):
        for y in range(len(player.board[x])):
            if player.board[x][y] and player.board[x][y].name != "sandguard":
                board.append(_unit(player.board[x][y]))
    others = []
    manager = getattr(getattr(env, "unwrapped", env), "player_manager", None)
    if manager is not None:
        for pid, other in manager.player_states.items():
            if other is not player and other.health > 0:
                others.append({"hp": int(other.health), "level": int(other.level)})
    traits = {k: int(v) for k, v in getattr(player, "team_tiers", {}).items() if v}
    return {
        "round": int(game_round),
        "hp": int(player.health),
        "gold": int(player.gold),
        "level": int(player.level),
        "xp": int(player.exp),
        "xp_needed": int(player.level_costs[player.level]) if player.level < len(player.level_costs) else 0,
        "streak": {"win": int(player.win_streak), "loss": int(player.loss_streak)},
        "board": board,
        "bench": [_unit(c) for c in player.bench if c],
        "item_bench": [i for i in player.item_bench if i],
        "shop": [s for s in shop if s],
        "active_traits": traits,
        "opponents": sorted(others, key=lambda o: -o["hp"]),
    }


# --------------------------------------------------------------------------- planners

class MimicPlanner:
    """The rule bot's own economy, written as a plan."""

    name = "mimic"

    def plan(self, state: dict, comps: dict, comp_now: str | None) -> dict:
        hp, gold, level = state["hp"], state["gold"], state["level"]
        desperate = hp < 30
        if state["round"] < 11:  # the rule bot levels when it is 4 xp short
            level_to = level + 1 if state["xp"] == state["xp_needed"] - 4 else level
            return {"comp": None, "level_to": level_to, "roll_floor": 999, "carry": None}
        level_to = level + 1 if level < 8 and (gold >= 54 or (desperate and gold > 4)) else level
        if desperate:
            roll_floor = 4
        elif level == 8:
            roll_floor = 52
        else:
            roll_floor = 999
        return {"comp": None, "level_to": level_to, "roll_floor": roll_floor, "carry": None}


PROMPT = """You are playing Teamfight Tactics (Set 4, 8 players, last one alive wins; you want the best placement).
Once per round you set a plan. A fixed executor carries it out with at most 15 actions this round:
it buys units for your target comp and pairs, fields a full board, sells excess bench units,
places items on your carry first, levels up while level < level_to, and rerolls the shop while gold - 2 >= roll_floor.
Leveling costs 4 gold per 4 xp; current xp and xp needed for the next level are in the state. A reroll costs 2 gold.
Interest: +1 gold per 10 gold held, max +5 at 50 gold. Win/loss streaks of 2+ pay extra gold.

Target comps you can pick (trait: units):
{comps}

Current target comp: {comp_now}

State (JSON):
{state}

Answer with ONE JSON object and nothing else:
{{"comp": <one trait name from the list, or null to let the executor keep/choose>,
  "level_to": <int, the level you want to reach this round>,
  "roll_floor": <int, keep at least this much gold; 999 means do not reroll>,
  "carry": <a unit name on your board that should get items, or null>,
  "why": <one short sentence>}}"""


class LLMPlanner:
    def __init__(self, cmd: str, log_path: str | None = None, cache_dir: str | None = None,
                 timeout: float = 300.0, name: str = "llm"):
        self.cmd, self.timeout, self.name = cmd, timeout, name
        self.log_path = Path(log_path) if log_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fallback = MimicPlanner()
        self.calls = self.invalid = 0
        self.context: dict = {}

    def _ask(self, prompt: str) -> tuple[str, float, bool]:
        key = hashlib.sha256((self.cmd + "\n" + prompt).encode()).hexdigest()[:24]
        cached = self.cache_dir / f"{key}.txt" if self.cache_dir else None
        if cached and cached.exists():
            return cached.read_text(), 0.0, True
        started = time.time()
        proc = subprocess.run(self.cmd, shell=True, input=prompt, capture_output=True, text=True,
                              timeout=self.timeout)
        out = proc.stdout
        if cached and proc.returncode == 0:
            cached.write_text(out)
        return out, time.time() - started, False

    @staticmethod
    def parse(text: str, state: dict, comps: dict) -> dict | None:
        objs = re.findall(r"\{[^{}]*\}", text, flags=re.S)
        for raw in reversed(objs):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not all(k in obj for k in ("level_to", "roll_floor")):
                continue
            comp = obj.get("comp")
            if comp is not None and comp not in comps:
                return None
            names = {u["name"] for u in state["board"]}
            carry = obj.get("carry")
            if carry is not None and carry not in names:
                carry = None
            try:
                level_to = max(1, min(9, int(obj["level_to"])))
                roll_floor = max(0, int(obj["roll_floor"]))
            except (TypeError, ValueError):
                return None
            return {"comp": comp, "level_to": level_to, "roll_floor": roll_floor, "carry": carry,
                    "why": str(obj.get("why", ""))[:300]}
        return None

    def plan(self, state: dict, comps: dict, comp_now: str | None) -> dict:
        prompt = PROMPT.format(
            comps="\n".join(f"- {t}: {', '.join(u)}" for t, u in comps.items()),
            comp_now=comp_now or "none yet (the executor picks one at round 11 unless you do)",
            state=json.dumps(state, ensure_ascii=False),
        )
        self.calls += 1
        error, plan = None, None
        try:
            text, seconds, cached = self._ask(prompt)
            plan = self.parse(text, state, comps)
        except Exception as exc:  # timeout, missing binary: the round falls back to the rule economy
            text, seconds, cached, error = "", 0.0, False, repr(exc)
        valid = plan is not None
        if not valid:
            self.invalid += 1
            plan = self.fallback.plan(state, comps, comp_now)
        if self.log_path:
            with self.log_path.open("a") as fh:
                fh.write(json.dumps({**self.context, "round": state["round"], "valid": valid, "cached": cached,
                                     "seconds": round(seconds, 2), "error": error, "plan": plan,
                                     "state": state, "answer": text[-2000:]}, ensure_ascii=False) + "\n")
        return plan


# --------------------------------------------------------------------------- executor

def _make_executor():
    from Simulator.generators.default_agent import Default_Agent
    from Simulator.generators.default_agent_stats import TEAM_COMPS, TEAM_COMP_TRAITS
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

    return PlanExecutor, dict(zip(TEAM_COMP_TRAITS, TEAM_COMPS)), TEAM_COMP_TRAITS


class PlanPolicy:
    """Seat policy: new plan whenever the round changes, executor acts on every step."""

    def __init__(self, planner, name: str | None = None):
        self.planner = planner
        self.name = name or f"plan:{planner.name}"
        self.executor = None
        self.round = None

    def reset(self, seed: int) -> None:
        cls, self.comps, self.traits = _make_executor()
        self.executor = cls()
        self.round = None
        if hasattr(self.planner, "context"):
            self.planner.context = {"seed": seed}

    def act(self, observation, info, agent, env):
        from Simulator.utils import decode_action

        player, shop, game_round = info["player"], info.get("shop"), info.get("game_round", 1)
        if game_round != self.round:
            self.round = game_round
            comp_now = self.traits[self.executor.comp_number] if self.executor.comp_number >= 0 else None
            state = describe(player, shop, game_round, env)
            if hasattr(self.planner, "context"):
                self.planner.context["seat"] = agent
            self.executor.plan = self.planner.plan(state, self.comps, comp_now)
        mask = np.asarray(observation["action_mask"])
        if mask.ndim == 1 and mask.size == MASK_SHAPE[0] * MASK_SHAPE[1]:
            mask = mask.reshape(MASK_SHAPE)
        command = self.executor.act(player, shop, game_round, mask)
        return decode_action([command])[0]


def make_plan_policy(kind: str) -> PlanPolicy:
    """`mimic`, or `llm` configured by env vars TFT_LLM_CMD, TFT_LLM_LOG, TFT_LLM_CACHE, TFT_LLM_NAME."""
    if kind == "mimic":
        return PlanPolicy(MimicPlanner())
    if kind == "llm":
        cmd = os.environ.get("TFT_LLM_CMD")
        if not cmd:
            raise ValueError("set TFT_LLM_CMD to a shell command that reads the prompt on stdin")
        name = os.environ.get("TFT_LLM_NAME", "llm")
        return PlanPolicy(LLMPlanner(cmd, os.environ.get("TFT_LLM_LOG"), os.environ.get("TFT_LLM_CACHE"),
                                     float(os.environ.get("TFT_LLM_TIMEOUT", "300")), name), name=f"plan:{name}")
    raise ValueError(f"unknown planner {kind!r}")

"""Plan-and-execute seat: one typed plan per round, a rule executor turns it into atomic actions.

See docs/PLAN.md section 5 and docs/STRATEGY.md section 3. The executor (tfteval/executor.py) is
the simulator's rule bot with these changes:

* its own levelling and rerolling are switched off; the plan decides them (`level_to`, `roll_floor`);
* the target comp comes from the plan (`comp`) instead of a random pick at round 11;
* items are placed in every round (the rule bot only places them before round 11), on the
  plan's `carry` first.

Optional plan fields (absent = the executor behaves exactly as before them):

* `comp` changed after the first choice pivots: off-comp bench units are sold as at round 11 and
  comp units on the bench are swapped onto the board (once, when the pivot is made).
* `field_comp` true: every round, swap the weakest off-comp board unit for the strongest comp unit
  on the bench while that one is no weaker, and drop the rule bot's own swaps of a comp unit for an
  off-comp one. On the simulator fork the rule bot's swap check works: from round 11 it puts any
  comp unit on the bench in place of any off-comp board unit, however weak, so without this field
  the comp already decides what is fielded; field_comp only makes those swaps strongest first and
  never fields a weaker unit. (On upstream the rule bot's swap check never fired, and without this
  field the comp decided what was bought but not what was fielded.)
* `level_by` {"level": N, "by": "4-1"} (or "rounds": K): reach level N by that round. Xp is bought
  as late as the action cap allows (at most XP_BUY_CAP buys in each earlier round, the rest on the
  deadline round, where the executor takes action slots from the rule bot if it has to).
* `spend` {"to": G, "by": "4-2"} (or "rounds": K, counted from the round the plan is issued):
  reroll down to G gold, spread evenly over the rounds up to the deadline.
* `fodder` true: field the weakest units it owns (board and bench) up to max_units, keep the strongest
  on the bench, place no items; the round after it goes back to false the strongest units are fielded
  again, then the rule bot plays on (its own swaps included). It buys nothing for the fodder board:
  the rule bot keeps deciding its buys, sales and levels as if its strongest units were fielded
  (PlanExecutor.strong_view); its bench-to-board swaps are dropped.
* `survival` N: when the expected losses to death (tfteval.stages.losses_to_death) are <= N, ignore
  the economy fields and fodder and roll for board strength.
* `xp_buys` N: buy xp at least N times this round (on top of `level_to` and `level_by`), whether or
  not that completes a level; used to spend the gold above a bank on xp.
* `hold` true: keep units and items back, field the strongest: the rule bot's bench sales (unless the
  bench is full), item placements and bench-to-board swaps are dropped, the executor places no
  items, and the strongest units (by strength(): cost, stars, items; not the rule bot's trait-aware
  comp score) are fielded every round (what the first fodder board's filters did, without the
  fodder).

`compile_knobs` turns a plan into the per-round knobs the executor reads (`PlanExecutor.begin_round`).

Planners produce the plan:

* `ParamPlanner` / `MimicPlanner` reproduces the rule bot's own economy, so `plan:mimic` against
  `rule` measures the executor changes alone (mainly late item placement). Its variants (VARIANTS)
  change a few knobs.
* `LLMPlanner` asks a model once per round. The model is any shell command that reads the
  prompt on stdin and prints an answer containing one JSON object; every call is logged and
  cached by prompt hash so a game can be replayed without calling the model again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from tfteval import public, stages
from tfteval.policies import MASK_SHAPE

PLAN_KEYS = ("comp", "level_to", "roll_floor", "carry")
EXTRA_KEYS = ("level_by", "spend", "fodder", "field_comp", "survival", "xp_buys", "hold")

LEVEL_COSTS = (0, 2, 2, 6, 10, 20, 36, 56, 80, 100)  # xp for the next level, Player.level_costs
ACTIONS_PER_ROUND = 15
XP_BUY_CAP = 10  # buy-xp actions scheduled in a round before the deadline round
SPEND_CAP = 30  # gold one round can spend on rerolls and the units they find


# --------------------------------------------------------------------------- state

def _unit(champ) -> dict:
    out = {"name": champ.name, "star": int(champ.stars), "cost": int(champ.cost)}
    if champ.items:
        out["items"] = list(champ.items)
    if champ.chosen:
        out["chosen"] = champ.chosen
    return out


def describe(player, shop, game_round: int, env=None, seat: str | None = None,
             comps: dict | None = None, comp: str | None = None) -> dict:
    """What a human player can see at the start of a round, as plain data: everything about
    yourself, and only the public part of the others (tfteval.public)."""
    hp, idx = int(player.health), int(game_round)
    sched = stages.schedule(idx)
    traits = {k: int(v) for k, v in getattr(player, "team_tiers", {}).items() if v}
    state = {
        "round": idx,
        "stage": sched["stage"],
        "pve": sched["pve"],
        "next": {"carousel": sched["to_carousel"], "pve": sched["to_pve"], "stage": sched["to_stage"]},
        "hp": hp,
        "gold": int(player.gold),
        "level": int(player.level),
        "xp": int(player.exp),
        "xp_needed": int(player.level_costs[player.level]) if player.level < len(player.level_costs) else 0,
        "streak": public.streak(player),
        "dmg_per_loss": stages.damage_per_loss(idx),
        "losses_to_death": stages.losses_to_death(hp, idx),
        "board": [_unit(u) for u in public.board_units(player)],
        "bench": [_unit(c) for c in player.bench if c],
        "item_bench": [i for i in player.item_bench if i],
        "shop": [s for s in shop if s],
        "active_traits": traits,
    }
    players = public.alive_players(env) if env is not None else {}
    if seat is None:
        seat = next((s for s, p in players.items() if p is player), None)
    state.update(public.public_view(seat, player, players, comps, comp))
    return state


# --------------------------------------------------------------------------- plan -> knobs

def _deadline(spec: dict, idx: int) -> int:
    if "by" in spec:
        return stages.parse_label(spec["by"])
    return idx + max(1, int(spec.get("rounds", 1))) - 1


def xp_to_level(level: int, xp: int, target: int) -> int:
    """Xp still missing to go from (level, xp) to `target`."""
    return max(0, sum(LEVEL_COSTS[lv] for lv in range(level, min(target, len(LEVEL_COSTS)))) - xp)


def compile_knobs(plan: dict, state: dict) -> dict:
    """Per-round executor knobs from a plan and the round-start state (describe() output).

    Without the optional fields the knobs are the plan's own comp / level_to / roll_floor / carry.
    """
    knobs = {"comp": plan.get("comp"), "level_to": int(plan["level_to"]), "roll_floor": int(plan["roll_floor"]),
             "carry": plan.get("carry"), "xp_buys": 0, "xp_priority": False, "fodder": bool(plan.get("fodder")),
             "field_comp": bool(plan.get("field_comp")), "survival": False, "hold": bool(plan.get("hold"))}
    idx, gold, level, xp, hp = state["round"], state["gold"], state["level"], state["xp"], state["hp"]

    spec = plan.get("level_by")
    if spec and level < int(spec["level"]):
        target, deadline = int(spec["level"]), _deadline(spec, idx)
        knobs["xp_priority"] = True
        if idx >= deadline:  # deadline round, or late: level all the way now
            knobs["level_to"] = max(knobs["level_to"], target)
        else:
            later = deadline - idx  # planning phases after this one, up to the deadline
            buys = math.ceil(max(0, xp_to_level(level, xp, target) - 2 * later) / 4)  # +2 xp each round start
            knobs["xp_buys"] = max(0, buys - XP_BUY_CAP * later)

    if plan.get("xp_buys"):
        knobs["xp_buys"] = max(knobs["xp_buys"], int(plan["xp_buys"]))

    spec = plan.get("spend")
    if spec:
        target, deadline = int(spec["to"]), _deadline(spec, idx)
        excess = gold - target
        if idx <= deadline and excess > 0:
            left = deadline - idx + 1
            share = max(math.ceil(excess / left), excess - SPEND_CAP * (left - 1))
            knobs["roll_floor"] = min(knobs["roll_floor"], max(target, gold - share))

    threshold = plan.get("survival")
    if threshold is not None and stages.losses_to_death(hp, idx) <= int(threshold):
        knobs.update(survival=True, fodder=False, hold=False, xp_buys=0, xp_priority=False, roll_floor=0)
        cheap_level = level < 8 and xp_to_level(level, xp, level + 1) <= 8  # at most two buys
        knobs["level_to"] = level + 1 if cheap_level else level
    return knobs


# --------------------------------------------------------------------------- planners

class ParamPlanner:
    """A fixed economy with a few knobs. The defaults are the rule bot's own economy.

    From round 11: level while below `max_level` and gold >= `level_gold`; at `max_level` reroll
    down to `top_floor`; below `desperate_hp` level and reroll down to `desperate_floor`.
    Optional: `fodder` = (first, last) stage labels for a fodder board, `survival` = threshold
    for the survival override, `field_comp` = swap comp units onto the board every round.
    Left at None/False they add nothing to the plan.
    """

    def __init__(self, name="mimic", level_gold=54, max_level=8, top_floor=52, desperate_hp=30, desperate_floor=4,
                 fodder=None, survival=None, field_comp=False):
        self.name, self.level_gold, self.max_level = name, level_gold, max_level
        self.top_floor, self.desperate_hp, self.desperate_floor = top_floor, desperate_hp, desperate_floor
        self.fodder = tuple(stages.parse_label(x) for x in fodder) if fodder else None
        self.survival = survival
        self.field_comp = field_comp

    def plan(self, state: dict, comps: dict, comp_now: str | None) -> dict:
        plan = self._economy(state)
        if self.fodder:
            plan["fodder"] = self.fodder[0] <= state["round"] <= self.fodder[1]
        if self.survival is not None:
            plan["survival"] = self.survival
        if self.field_comp:
            plan["field_comp"] = True
        return plan

    def _economy(self, state: dict) -> dict:
        hp, gold, level = state["hp"], state["gold"], state["level"]
        desperate = hp < self.desperate_hp
        if state["round"] < 11:  # the rule bot levels when it is 4 xp short
            level_to = level + 1 if state["xp"] == state["xp_needed"] - 4 else level
            return {"comp": None, "level_to": level_to, "roll_floor": 999, "carry": None}
        wants_level = gold >= self.level_gold or (desperate and gold > self.desperate_floor)
        level_to = level + 1 if level < self.max_level and wants_level else level
        if desperate:
            roll_floor = self.desperate_floor
        elif level >= self.max_level:
            roll_floor = self.top_floor
        else:
            roll_floor = 999
        return {"comp": None, "level_to": level_to, "roll_floor": roll_floor, "carry": None}


MimicPlanner = ParamPlanner


class OverlayPlanner:
    """A base planner plus fixed plan fields inside windows of rounds, for smoke runs and ablations
    of one executor capability at a time, e.g.

        OverlayPlanner(ParamPlanner(), [{"from": "3-3", "to": "3-7", "fields": {"comp": "mage"}},
                                         {"from": "4-1", "fields": {"comp": "divine"}}])
    """

    def __init__(self, base, windows: list[dict], name: str | None = None):
        self.base, self.name = base, name or f"{base.name}+overlay"
        self.windows = [(stages.parse_label(w.get("from", 1)), stages.parse_label(w.get("to", 10_000)),
                         dict(w["fields"])) for w in windows]

    def plan(self, state: dict, comps: dict, comp_now: str | None) -> dict:
        plan = self.base.plan(state, comps, comp_now)
        for first, last, fields in self.windows:
            if first <= state["round"] <= last:
                plan.update(fields)
        return plan

# Economy variants for tuning the executor without a model (name -> knobs).
VARIANTS = {
    "mimic": {},
    "fast8": {"level_gold": 34},  # level whenever 30 gold is left after buying xp
    "rolldown8": {"top_floor": 10},  # at level 8 spend down to 10 gold
    "hp50": {"desperate_hp": 50},  # start the desperate roll-down earlier
    "fast8roll": {"level_gold": 34, "top_floor": 10},
    "fodder2": {"fodder": ("2-1", "2-6")},  # mimic, but field the weakest units from 2-1 to 2-6
    "mimicfc": {"field_comp": True},  # mimic, plus swap comp units onto the board every round
    "fast8fc": {"level_gold": 34, "field_comp": True},
}


PROMPT = """You are playing Teamfight Tactics (Set 4, 8 players, last one alive wins; you want the best placement).
Once per round you set a plan. A fixed executor carries it out with at most 15 actions this round:
it buys units for your target comp and pairs, fields a full board, sells excess bench units,
places items on your carry first, levels up while level < level_to, and rerolls the shop while gold - 2 >= roll_floor.
Leveling costs 4 gold per 4 xp; current xp and xp needed for the next level are in the state. A reroll costs 2 gold.
Interest: +1 gold per 10 gold held, max +5 at 50 gold. Win/loss streaks of 2+ pay extra gold.
Losing a PvP fight costs HP (dmg_per_loss is the usual amount this stage); losing to monsters (PvE) costs none.
The state shows each opponent's board, HP, level, streak (+wins/-losses) and interest bracket, the seats you can
meet next (next_from), and how many copies of each comp's units the opponents hold (contested_by_comp).

Target comps you can pick (trait: units):
{comps}

Current target comp: {comp_now}

State (JSON):
{state}

Answer with ONE JSON object and nothing else:
{{"comp": <one trait name from the list, or null to let the executor keep/choose; a different comp than the current one pivots and sells off-comp bench units>,
  "level_to": <int, the level you want to reach this round>,
  "roll_floor": <int, keep at least this much gold; 999 means do not reroll>,
  "carry": <a unit name on your board that should get items, or null>,
  "why": <one short sentence>}}
Optional fields you may add:
  "level_by": {{"level": N, "by": "<stage-round>"}}  reach level N by that round, buying xp as late as the action cap allows
  "spend": {{"to": G, "by": "<stage-round>"}}  reroll down to G gold, spread evenly over the rounds up to that one
  "fodder": true  field your weakest units and keep the strongest on the bench (to lose on purpose); false fields the best again
  "field_comp": true  swap comp units from the bench onto the board in place of weaker off-comp units, every round
  "survival": N  when you can afford at most N more losses, ignore the economy and roll for board strength"""


def _parse_extras(obj: dict) -> dict:
    """Validated optional fields; an invalid one is dropped, not the whole plan."""
    out = {}
    try:
        for key in ("fodder", "field_comp"):
            if isinstance(obj.get(key), bool):
                out[key] = obj[key]
        if obj.get("survival") is not None:
            out["survival"] = max(0, min(10, int(obj["survival"])))
        for key, field in (("level_by", "level"), ("spend", "to")):
            spec = obj.get(key)
            if isinstance(spec, dict) and field in spec and ("by" in spec or "rounds" in spec):
                clean = {field: int(spec[field])}
                if "by" in spec:
                    stages.parse_label(spec["by"])
                    clean["by"] = spec["by"]
                else:
                    clean["rounds"] = max(1, int(spec["rounds"]))
                out[key] = clean
    except (TypeError, ValueError):
        pass
    if "level_by" in out:
        out["level_by"]["level"] = max(2, min(9, out["level_by"]["level"]))
    if "spend" in out:
        out["spend"]["to"] = max(0, out["spend"]["to"])
    return out


class LLMPlanner:
    def __init__(self, cmd: str, log_path: str | None = None, cache_dir: str | None = None,
                 timeout: float = 300.0, name: str = "llm"):
        self.cmd, self.timeout, self.name = cmd, timeout, name
        self.log_path = Path(log_path) if log_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fallback = ParamPlanner()
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
        objs = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.S)
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
                    "why": str(obj.get("why", ""))[:300], **_parse_extras(obj)}
        return None

    def plan(self, state: dict, comps: dict, comp_now: str | None) -> dict:
        prompt = PROMPT.format(
            comps="\n".join(f"- {t}: {', '.join(u)}" for t, u in comps.items()),
            comp_now=comp_now or "none yet (the executor picks one at round 11 unless you do)",
            state=json.dumps(state, ensure_ascii=False, separators=(",", ":")),
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
    from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS

    from tfteval.executor import COMPS, PlanExecutor  # module level so games can be pickled (branching.py)

    return PlanExecutor, dict(COMPS), TEAM_COMP_TRAITS


class PlanPolicy:
    """Seat policy: new plan whenever the round changes, executor acts on every step."""

    def __init__(self, planner, name: str | None = None):
        self.planner = planner
        self.name = name or f"plan:{planner.name}"
        self.executor = None
        self.round = None
        self.last_state = None

    def reset(self, seed: int) -> None:
        cls, self.comps, self.traits = _make_executor()
        self.executor = cls()
        self.round = None
        self.last_state = None
        if hasattr(self.planner, "context"):
            self.planner.context = {"seed": seed}

    def act(self, observation, info, agent, env):
        from Simulator.utils import decode_action

        player, shop, game_round = info["player"], info.get("shop"), info.get("game_round", 1)
        if game_round != self.round:
            self.round = game_round
            comp_now = self.traits[self.executor.comp_number] if self.executor.comp_number >= 0 else None
            state = describe(player, shop, game_round, env, seat=agent, comps=self.comps, comp=comp_now)
            if hasattr(self.planner, "context"):
                self.planner.context["seat"] = agent
            plan = self.planner.plan(state, self.comps, comp_now)
            self.executor.begin_round(plan, compile_knobs(plan, state), game_round)
            self.last_state = state
        mask = np.asarray(observation["action_mask"])
        if mask.ndim == 1 and mask.size == MASK_SHAPE[0] * MASK_SHAPE[1]:
            mask = mask.reshape(MASK_SHAPE)
        command = self.executor.act(player, shop, game_round, mask)
        return decode_action([command])[0]


def make_plan_policy(kind: str) -> PlanPolicy:
    """An economy variant from VARIANTS (`mimic`, `fast8`, ...), or `llm` configured by env vars
    TFT_LLM_CMD, TFT_LLM_LOG, TFT_LLM_CACHE, TFT_LLM_NAME."""
    if kind in VARIANTS:
        return PlanPolicy(ParamPlanner(kind, **VARIANTS[kind]), name=kind)
    if kind == "llm":
        cmd = os.environ.get("TFT_LLM_CMD")
        if not cmd:
            raise ValueError("set TFT_LLM_CMD to a shell command that reads the prompt on stdin")
        name = os.environ.get("TFT_LLM_NAME", "llm")
        return PlanPolicy(LLMPlanner(cmd, os.environ.get("TFT_LLM_LOG"), os.environ.get("TFT_LLM_CACHE"),
                                     float(os.environ.get("TFT_LLM_TIMEOUT", "300")), name), name=f"plan:{name}")
    raise ValueError(f"unknown planner {kind!r}")

"""Rule stance policy (experiment E4): round-start features -> one stance -> plan fields.

docs/STRATEGY.md section 3 in code: the features are computed here, not by a model; one stance is
picked per round from a small set; the stance is compiled into the plan fields the executor already
reads (tfteval/planner.py). This is the baseline a model planner has to beat.

Stances, in priority order (the first that applies is played):

* `stabilize`   losses to death <= STABILIZE_LTD, or (from 3-1) a loss streak of
                STABILIZE_LOSS_STREAK with a board clearly weaker than the possible next opponents'.
                Roll down (to STABILIZE_HARD_FLOOR when dying, else STABILIZE_SOFT_FLOOR), level when
                two xp buys or fewer are missing, and `survival` (roll to 0) at SURVIVAL_LTD.
* `loss_streak` off by default (seat kind `stance+lossstreak`): in stage 2's PvP rounds at HP >=
                LOSSSTREAK_MIN_HP, when the last fight was lost or the board is clearly weaker than the
                candidates': `fodder` (field the weakest units, keep the strong ones on the bench).
                Re-checked every round; ends for the game once the streak breaks or HP drops below
                the threshold. (README: in stage 2 a fodder board cost +16.5 HP for +1.0 gold, but
                ended 0.72 +-1.11 places better than mimic on the same 32 seeds; not settled.)
* `slow_roll`   a 1-3 cost unit of the target comp with SLOWROLL_COPIES copies held, few copies on
                other boards, level at most SLOWROLL_LEVEL[cost]: go to that level, hold it, roll only
                the gold above SLOWROLL_FLOOR, items on that unit. Kept until it is 3-star, contested,
                too late or the seat bleeds.
* `fast8`       HP and board fine and the gold, projected to the deadline, pays for level 8 with
                FAST8_GOLD_LEFT to spare: `level_by` 8 (xp bought as late as the action cap allows),
                no rolls; once at 8, roll down to FAST8_ROLL_TO over FAST8_ROLL_ROUNDS rounds. Kept
                until then unless HP drops below FAST8_KEEP_HP or the deadline is missed.
* `keep_streak` win streak of STREAK_MIN or more and a board at least as strong as the possible
                next opponents': level when it leaves STREAK_KEEP_GOLD, roll the gold above
                STREAK_ROLL_FLOOR.
* `standard`    the rule bot's own economy (ParamPlanner defaults): save, level at 54 gold, roll only
                at level 8 or below 30 HP.

`field_comp` is on in every stance. A stance switched off by a seat kind (STANCE_KINDS) is never
picked, so its rounds fall through to the next applicable stance, in practice `standard`.

Information boundary: `plan()` gets the describe() state (tfteval.public) and nothing else. Every
feature below comes from it: own state, opponents' boards / HP / level / streak / interest bracket,
the candidate set `next_from` and copies seen on other boards. It never sees the env, so it cannot
read `game_round.matchups`, other players' bench, shop, gold, xp or the pool.

Board strength is the rule bot's own comp score, `Default_Agent.rank_comp` (unit value by cost
and star + trait count x trait tier), plus ITEM_SCORE per item as in the executor's strength(); it is
computed on the public board (unit, star, chosen trait, items) of every player.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from tfteval import stages
from tfteval.planner import ParamPlanner, PlanPolicy, xp_to_level

# --------------------------------------------------------------------------- thresholds
# [doc] the trigger suggested in the stance table of docs/STRATEGY.md section 3; [sim] chosen from a
# measurement in this simulator (see the note); [guess] not measured, a starting point to be tuned.

STANCE_FROM = 3                # [sim] 2-1: stage 1 is PvE only, every stance before it is standard
CHEAP_XP = 8                   # [sim] "cheap level": at most two xp buys short, as compile_knobs' survival

STABILIZE_LTD = 3              # [doc] losses to death (stages.losses_to_death) <= 3
STABILIZE_HARD_FLOOR = 10      # [doc] roll to 0-10 only when life is at stake
STABILIZE_SOFT_FLOOR = 20      # [guess] losing heavily but not dying: roll down to 20, keep 2 interest
SURVIVAL_LTD = 2               # [guess] plan field `survival`: roll to 0 at <= 2 losses to death
STABILIZE_HEAVY_FROM = "3-1"   # [sim] a stage-2 loss costs ~8 HP: losing streaks there are not "heavy"
STABILIZE_LOSS_STREAK = 3      # [guess] "losing heavily" = this many losses in a row ...
STABILIZE_WEAK_RATIO = 0.85    # [guess] ... with board score < 0.85 x the candidates' mean

LOSSSTREAK_STAGE = 2           # [doc] only in stage 2, and only its PvP rounds (2-1 to 2-6): a PvE
                               # round lost on purpose loses its loot and keeps the streak anyway
LOSSSTREAK_MIN_HP = 80         # [doc] HP >= 80 to start and to go on
LOSSSTREAK_MIN_LOSSES = 1      # [guess] "already losing": lost at least the last fight ...
LOSSSTREAK_WEAK_RATIO = 0.85   # [guess] ... or own board < 0.85 x the candidates' mean (start only)
                               # the doc's "<= 6 HP a loss" is left out: the measured stage-2 loss costs 8.0
                               # (stages.DAMAGE_PER_LOSS), so it would never hold

SLOWROLL_COPIES = 4            # [sim] the doc says ~6; the mimic seat never holds 5 copies of a 1-3 cost comp
                               # unit before stage 5 (64 games, results/smoke_*.json); 4 = a 2-star + 1
SLOWROLL_LEVEL = {1: 5, 2: 6, 3: 7}  # [doc] reroll level by unit cost; start at most one level below
SLOWROLL_FLOOR = 50            # [doc] keep 50, roll only the gold above it
SLOWROLL_MAX_SEEN = 3          # [guess] "nobody contests": at most 3 copies on other boards
SLOWROLL_MIN_HP = 50           # [guess] no new slow roll below 50 HP
SLOWROLL_BLEED = 25            # [guess] stop when this much HP went in the last SLOWROLL_BLEED_ROUNDS
SLOWROLL_BLEED_ROUNDS = 3
SLOWROLL_UNTIL = {1: "4-1", 2: "4-2", 3: "4-5"}  # [guess] give up a slow roll after this round

FAST8_FROM = "3-5"             # [guess] first round a fast 8 can be committed
FAST8_BY = "4-2"               # [sim] the doc: 8 at the end of stage 4 / start of stage 5 (the real game's
                               # curve); this lobby's rule bots reach 8 around 4-6 (mimic seat: level 7.4
                               # at 4-1, 8.0 at 4-7), so 4-5 would not be fast here
FAST8_EARLY_BY = "4-1"         # [doc] earlier when opponents are close to 8
FAST8_RACE = 2                 # [guess] "close": opponents at 8, or at 7 with 50+ gold
FAST8_MIN_HP = 60              # [doc] HP 60+
FAST8_MIN_RATIO = 0.9          # [guess] "board stable": score >= 0.9 x the candidates' mean
FAST8_GOLD_LEFT = 20           # [doc] 20-30 gold left after levelling (projected, before unit buys)
FAST8_ROLL_TO = 10             # [guess] at 8, roll down to this
FAST8_ROLL_ROUNDS = 2          # [guess] ... spread over this many rounds
FAST8_GRACE = 1                # [guess] rounds past the deadline before a missed fast 8 is dropped
FAST8_KEEP_HP = 50             # [guess] a committed fast 8 is dropped below this HP before it reaches 8
                               # (its gold is not re-checked once committed: unit buys eat into the projection)

STREAK_MIN = 2                 # [doc] win streak of 2+
STREAK_RATIO = 1.0             # [doc] "stronger than the candidates": score >= their mean
STREAK_KEEP_GOLD = 10          # [guess] level when at least this much gold is left
STREAK_LEVEL_CAP = {2: 5, 3: 7}  # [guess] by stage, else 8: one level ahead of this lobby's curve
STREAK_ROLL_FLOOR = 50         # [guess] roll the gold above 50 for upgrades

ITEM_SCORE = 2                 # [sim] the executor's strength(): +2 per item

STANCES = ("stabilize", "loss_streak", "slow_roll", "fast8", "keep_streak", "standard")  # priority order
OFF_BY_DEFAULT = frozenset({"loss_streak"})

# seat kind -> stances switched off (their rounds fall through, in practice to standard) / switched on
STANCE_KINDS = {
    "stance": {},
    "stance-nofast8": {"off": ("fast8",)},
    "stance-noslowroll": {"off": ("slow_roll",)},
    "stance-nostreak": {"off": ("keep_streak",)},
    "stance-nostabilize": {"off": ("stabilize",)},
    "stance+lossstreak": {"on": ("loss_streak",)},
}


# --------------------------------------------------------------------------- board strength

_TAG = re.compile(r"^(?P<name>[a-z_]+)\*(?P<star>\d)(?:\((?P<chosen>[^)]*)\))?(?:\[(?P<items>[^\]]*)\])?$")
_RULE = None


def parse_tag(tag: str) -> dict:
    """A public unit tag (tfteval.public.unit_tag, e.g. "ahri*2(mage)[bf_sword]") as a unit dict."""
    m = _TAG.match(tag)
    if not m:
        raise ValueError(f"bad unit tag {tag!r}")
    unit = {"name": m["name"], "star": int(m["star"])}
    if m["chosen"]:
        unit["chosen"] = m["chosen"]
    if m["items"]:
        unit["items"] = m["items"].split(",")
    return unit


def _rule_tables():
    global _RULE
    if _RULE is None:
        from Simulator.battle.origin_class_stats import origin_class
        from Simulator.battle.stats import COST
        from Simulator.generators.default_agent import Default_Agent

        _RULE = Default_Agent(), origin_class, COST
    return _RULE


def board_score(units: list[dict]) -> int:
    """Default_Agent.rank_comp on a board rebuilt from unit dicts, plus ITEM_SCORE per item."""
    agent, origins, cost = _rule_tables()
    grid = [[None] * 4 for _ in range(7)]
    items, k = 0, 0
    for u in units:
        if u["name"] not in cost or u["name"] not in origins or k >= 28:
            continue
        grid[k % 7][k // 7] = SimpleNamespace(name=u["name"], cost=cost[u["name"]], stars=int(u["star"]),
                                              chosen=u.get("chosen") or False, origin=origins[u["name"]])
        items += len(u.get("items", ()))
        k += 1
    return int(agent.rank_comp(grid)) + ITEM_SCORE * items


def copies(unit: dict) -> int:
    return 3 ** (int(unit["star"]) - 1)


# --------------------------------------------------------------------------- features

def streak_bonus(n: int) -> int:
    """Streak gold in the next income for an n-game streak (Player.gold_income)."""
    return 0 if n < 2 else 1 if n <= 3 else 2 if n == 4 else 3


def projected_gold(gold: int, rounds: int) -> int:
    """Gold after `rounds` incomes of 5 + interest, spending nothing (streaks left out)."""
    for _ in range(max(0, rounds)):
        gold += 5 + min(gold // 10, 5)
    return gold


def xp_gold(level: int, xp: int, target: int, rounds: int = 0) -> int:
    """Gold for the xp buys still missing to reach `target` after `rounds` passive +2 xp."""
    need = xp_to_level(level, xp, target) - 2 * max(0, rounds)
    return 4 * math.ceil(max(0, need) / 4)


def features(state: dict, comps: dict, comp_now: str | None, hp_seen: dict | None = None) -> dict:
    """Everything a stance decision reads, from the describe() state only."""
    idx, hp, gold, level, streak = state["round"], state["hp"], state["gold"], state["level"], state["streak"]
    opponents = state.get("opponents", [])
    scores = {o["seat"]: board_score([parse_tag(t) for t in o["board"]]) for o in opponents}
    mine = board_score(state["board"])
    cands = [s for s in state.get("next_from", []) if s in scores] or list(scores)
    cand = [scores[s] for s in cands]
    cand_mean = mean(cand) if cand else 0.0
    seen = seen_copies(state)
    near8 = sum(o["level"] >= 8 or (o["level"] == 7 and o["interest"] >= 5) for o in opponents)
    recent = [hp_seen[i] for i in range(idx - SLOWROLL_BLEED_ROUNDS, idx) if hp_seen and i in hp_seen]
    out = {
        "stage": state["stage"], "round": idx, "pve": state.get("pve"), "next": state.get("next"),
        "hp": hp, "hp_rank": state.get("hp_rank"), "alive": state.get("alive"),
        "ltd": state["losses_to_death"], "dmg": state["dmg_per_loss"],
        "gold": gold, "level": level, "xp": state["xp"], "streak": streak,
        "streak_gold_win": 1 + streak_bonus(max(streak, 0) + 1),  # +1 for the win itself
        "streak_gold_loss": streak_bonus(max(-streak, 0) + 1),
        "score": mine, "cand_score": round(cand_mean, 1), "cand_max": max(cand, default=0),
        "ratio": round(mine / max(cand_mean, 1.0), 2),
        "score_rank": 1 + sum(s > mine for s in scores.values()),
        "comp": comp_now,
        "contested": sum(state["contested"].values()) if state.get("contested") else None,
        "opp_level_max": max((o["level"] for o in opponents), default=0),
        "opp_near8": near8,
        "opp_rich": sum(o["interest"] >= 5 for o in opponents),
        "hp_lost_recent": max(recent) - hp if recent else 0,
        "carry": slow_roll_candidate(state, comps, comp_now, seen),
    }
    if level < 8:
        by = stages.parse_label(FAST8_BY)
        rounds = max(0, by - idx)
        out["fast8_left"] = projected_gold(gold, rounds) - xp_gold(level, state["xp"], 8, rounds)
    return out


def seen_copies(state: dict) -> Counter:
    """Copies of each unit on the other players' (public) boards."""
    seen = Counter()
    for o in state.get("opponents", []):
        for t in o["board"]:
            u = parse_tag(t)
            seen[u["name"]] += copies(u)
    return seen


def held_copies(state: dict) -> tuple[Counter, dict, set]:
    """Own copies per unit on board and bench, unit costs, and units already 3-star."""
    held, cost, three = Counter(), {}, set()
    for u in list(state["board"]) + list(state["bench"]):
        held[u["name"]] += copies(u)
        cost[u["name"]] = int(u["cost"])
        if int(u["star"]) >= 3:
            three.add(u["name"])
    return held, cost, three


def slow_roll_candidate(state: dict, comps: dict, comp_now: str | None, seen: Counter) -> dict | None:
    """The 1-3 cost comp unit with the most copies held (ties: cheaper), not yet 3-star."""
    allowed = set(comps[comp_now]) if comp_now in comps else {u for units in comps.values() for u in units}
    held, cost, three = held_copies(state)
    best = None
    for name, n in held.items():
        if name in allowed and name not in three and cost[name] <= 3:
            key = (n, -cost[name], name)
            if best is None or key > best[0]:
                best = (key, {"unit": name, "cost": cost[name], "copies": n, "seen": seen[name]})
    return best[1] if best else None


# --------------------------------------------------------------------------- planner

class StancePlanner:
    """Same interface as ParamPlanner: plan(state, comps, comp_now) -> plan dict. The plan carries
    `stance`, `why` and `features` besides the executor's fields; with TFT_STANCE_LOG (or log_path)
    every round is appended to a JSONL file."""

    def __init__(self, name: str = "stance", disabled=(), enabled=(), log_path: str | None = None):
        unknown = (set(disabled) | set(enabled)) - set(STANCES[:-1])
        if unknown:
            raise ValueError(f"unknown stances {sorted(unknown)}")
        self.name, self.disabled, self.enabled = name, frozenset(disabled), frozenset(enabled)
        self.log_path = Path(log_path) if log_path else None
        self.standard = ParamPlanner("standard", field_comp=True)
        self.context: dict = {}
        self._new_game()

    def _new_game(self) -> None:
        self.last_round = -1
        self.hp_seen: dict[int, int] = {}
        self.slow: dict | None = None  # the unit being slow-rolled
        self.dropped: set[str] = set()  # units a slow roll was given up on (or finished)
        self.comp: str | None = None  # comp a slow roll picked before the executor had one
        self.fast8: dict | None = None  # {"by": idx, "landed": idx or None}
        self.loss = None  # None: not started this game; True: on; False: over for this game

    def on(self, stance: str) -> bool:
        if stance in OFF_BY_DEFAULT and stance not in self.enabled:
            return False
        return stance not in self.disabled

    # ---- choice
    def choose(self, state: dict, f: dict, comps: dict) -> tuple[str, str]:
        idx, level = state["round"], state["level"]
        if idx < STANCE_FROM:
            return "standard", "stage 1: PvE only"

        if self.on("stabilize"):
            why = None
            if f["ltd"] <= STABILIZE_LTD:
                why = f"dying: {f['ltd']} losses to death"
            elif (idx >= stages.parse_label(STABILIZE_HEAVY_FROM) and f["streak"] <= -STABILIZE_LOSS_STREAK
                  and f["ratio"] < STABILIZE_WEAK_RATIO):
                why = f"heavy: loss streak {-f['streak']}, board {f['ratio']} of candidates' mean"
            if why:
                if self.slow:
                    self.dropped.add(self.slow["unit"])
                    f["slow_roll_end"] = f"{self.slow['unit']}: stabilize"
                self.slow = self.fast8 = None
                return "stabilize", why

        if self.on("loss_streak") and self.loss is not False:
            why = self._loss_streak(state, f)
            if why.startswith(("start", "keep")):
                self.loss = True
                return "loss_streak", why
            if self.loss:
                self.loss = False
                f["loss_streak_end"] = why

        if self.slow:
            unit = self.slow["unit"]
            stop = self._slow_roll_stop(state, f)
            if stop is None:
                return "slow_roll", f"keep: {unit}, {self.slow['copies']} copies, {self.slow['seen']} seen"
            self.dropped.add(unit)
            self.slow = None
            f["slow_roll_end"] = f"{unit}: {stop}"
        elif self.on("slow_roll") and not self.fast8:
            c = f["carry"]
            if c and c["unit"] not in self.dropped and c["copies"] >= SLOWROLL_COPIES \
                    and c["seen"] <= SLOWROLL_MAX_SEEN and level <= SLOWROLL_LEVEL[c["cost"]] <= level + 1 \
                    and idx <= stages.parse_label(SLOWROLL_UNTIL[c["cost"]]) and f["hp"] >= SLOWROLL_MIN_HP:
                self.slow = dict(c)
                if f["comp"] is None:
                    self.comp = self.comp or self._comp_for(c["unit"], state, comps)
                return "slow_roll", f"start: {c['unit']}, {c['copies']} copies, {c['seen']} seen"

        if self.fast8:
            if level >= 8 and self.fast8["landed"] is None:
                self.fast8["landed"] = idx
            landed = self.fast8["landed"]
            if landed is not None and idx < landed + FAST8_ROLL_ROUNDS:
                return "fast8", f"roll: at 8 since {stages.label(landed)}"
            if landed is None and idx <= self.fast8["by"] + FAST8_GRACE and f["hp"] >= FAST8_KEEP_HP:
                return "fast8", f"keep: level 8 by {stages.label(self.fast8['by'])}"
            if landed is None:
                f["fast8_end"] = f"hp {f['hp']}" if f["hp"] < FAST8_KEEP_HP else "missed the deadline"
            self.fast8 = None
        elif self.on("fast8") and level < 8 and idx >= stages.parse_label(FAST8_FROM):
            early = stages.parse_label(FAST8_EARLY_BY)
            by = early if f["opp_near8"] >= FAST8_RACE and idx <= early else stages.parse_label(FAST8_BY)
            left = projected_gold(f["gold"], by - idx) - xp_gold(level, f["xp"], 8, by - idx)
            if idx <= by and f["hp"] >= FAST8_MIN_HP and f["ratio"] >= FAST8_MIN_RATIO and left >= FAST8_GOLD_LEFT:
                self.fast8 = {"by": by, "landed": None}
                return "fast8", f"start: 8 by {stages.label(by)}, {left} gold left, {f['opp_near8']} near 8"

        if self.on("keep_streak") and f["streak"] >= STREAK_MIN and f["ratio"] >= STREAK_RATIO:
            return "keep_streak", f"streak: {f['streak']} wins, board {f['ratio']} of candidates' mean"
        return "standard", "default: no other stance applies"

    def _loss_streak(self, state: dict, f: dict) -> str:
        """A reason starting with "start" or "keep" to field a fodder board this round, else why not."""
        if int(state["stage"].split("-")[0]) != LOSSSTREAK_STAGE or state["pve"]:
            return "end of stage 2" if self.loss else "not stage 2 PvP"
        if f["hp"] < LOSSSTREAK_MIN_HP:
            return f"hp {f['hp']}"
        if self.loss:
            if f["streak"] <= -LOSSSTREAK_MIN_LOSSES:
                return f"keep: {-f['streak']} losses, hp {f['hp']}"
            return "streak broken"
        if f["streak"] <= -LOSSSTREAK_MIN_LOSSES:
            return f"start: {-f['streak']} losses"
        if f["ratio"] < LOSSSTREAK_WEAK_RATIO:
            return f"start: board {f['ratio']} of candidates' mean"
        return "not losing"

    def _slow_roll_stop(self, state: dict, f: dict) -> str | None:
        """Why the slow roll ends this round, or None to keep rolling. Updates its copies / seen."""
        unit = self.slow["unit"]
        held, _, three = held_copies(state)
        self.slow.update(copies=held[unit], seen=seen_copies(state)[unit])
        if unit in three:
            return "3-star"
        if held[unit] < SLOWROLL_COPIES:
            return "copies sold"
        if self.slow["seen"] > SLOWROLL_MAX_SEEN:
            return "contested"
        if state["round"] > stages.parse_label(SLOWROLL_UNTIL[self.slow["cost"]]):
            return "too late"
        if f["hp_lost_recent"] >= SLOWROLL_BLEED:
            return "bleeding"
        return None

    @staticmethod
    def _comp_for(unit: str, state: dict, comps: dict) -> str | None:
        """The comp containing `unit` with the most of the seat's units (the first listed on ties)."""
        owned = {u["name"] for u in list(state["board"]) + list(state["bench"])}
        options = [(sum(u in owned for u in units), -i, trait) for i, (trait, units) in enumerate(comps.items())
                   if unit in units]
        return max(options)[2] if options else None

    # ---- compile
    def compile(self, stance: str, state: dict, f: dict, comps: dict, comp_now: str | None) -> dict:
        plan = self.standard.plan(state, comps, comp_now)  # comp/level_to/roll_floor/carry + field_comp
        level, xp, gold = state["level"], state["xp"], state["gold"]
        if comp_now is None and self.comp:
            plan["comp"] = self.comp
        if stance == "stabilize":
            floor = STABILIZE_HARD_FLOOR if f["ltd"] <= STABILIZE_LTD else STABILIZE_SOFT_FLOOR
            plan["roll_floor"] = min(plan["roll_floor"], floor)
            if level < 8 and xp_to_level(level, xp, level + 1) <= CHEAP_XP:
                plan["level_to"] = max(plan["level_to"], level + 1)
            plan["survival"] = SURVIVAL_LTD
        elif stance == "loss_streak":
            plan["fodder"] = True
        elif stance == "slow_roll":
            plan["level_to"] = max(level, SLOWROLL_LEVEL[self.slow["cost"]])
            plan["roll_floor"] = SLOWROLL_FLOOR
            plan["carry"] = self.slow["unit"]
        elif stance == "fast8":
            if level < 8:
                plan.update(level_to=level, roll_floor=999, level_by={"level": 8, "by": stages.label(self.fast8["by"])})
            else:
                plan["spend"] = {"to": FAST8_ROLL_TO, "by": stages.label(self.fast8["landed"] + FAST8_ROLL_ROUNDS - 1)}
        elif stance == "keep_streak":
            cap = STREAK_LEVEL_CAP.get(int(state["stage"].split("-")[0]), 8)
            if level < cap and gold - xp_gold(level, xp, level + 1) >= STREAK_KEEP_GOLD:
                plan["level_to"] = max(plan["level_to"], level + 1)
            plan["roll_floor"] = min(plan["roll_floor"], STREAK_ROLL_FLOOR)
        return plan

    # ---- entry point
    def plan(self, state: dict, comps: dict, comp_now: str | None) -> dict:
        idx = state["round"]
        if idx <= self.last_round:  # a new game on the same planner
            self._new_game()
        self.last_round = idx
        f = features(state, comps, comp_now, self.hp_seen)
        self.hp_seen[idx] = state["hp"]
        stance, why = self.choose(state, f, comps)
        plan = self.compile(stance, state, f, comps, comp_now)
        plan.update(stance=stance, why=why, features=f)
        if self.log_path:
            row = {**self.context, "policy": self.name, "round": idx, "stage": state["stage"], "stance": stance,
                   "why": why, "plan": {k: v for k, v in plan.items() if k not in ("features", "stance", "why")},
                   "features": f}
            with self.log_path.open("a") as fh:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        return plan


def make_stance_policy(kind: str) -> PlanPolicy:
    """`stance` or one of its ablations (STANCE_KINDS) as a plan seat. TFT_STANCE_LOG names a JSONL
    file that gets one row per round (scripts/stance_report.py reads it)."""
    if kind not in STANCE_KINDS:
        raise ValueError(f"unknown stance policy {kind!r}; choose from {sorted(STANCE_KINDS)}")
    spec = STANCE_KINDS[kind]
    planner = StancePlanner(kind, disabled=spec.get("off", ()), enabled=spec.get("on", ()),
                            log_path=os.environ.get("TFT_STANCE_LOG"))
    return PlanPolicy(planner, name=kind)

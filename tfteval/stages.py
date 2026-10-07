"""Stage labels, the round schedule and the player damage model of the simulator.

The env reports a round index `idx` (info["game_round"]), not a stage label. A planning phase
with index `idx` is followed by the round `game_round.game_rounds[idx]`:

    idx 0       1-1 carousel + 1-2 PvE (both run inside env.reset, no planning phase before them)
    idx 1, 2    1-3, 1-4 PvE
    idx 3..8    2-1, 2-2, 2-3, 2-5 (the 2-4 carousel runs right after this planning phase), 2-6, 2-7 PvE
    then 6 indices per stage the same way: idx 9 = 3-1, idx 14 = 3-7, idx 15 = 4-1, ...

PvE losses cost no HP and do not touch streaks (the fork still leaves PvE damage out, FORK_NOTES.md
"Other findings"); 6-7 (idx 32) is a PvE round on the fork.

Damage. Losing a PvP fight costs the base damage of the stage plus damage for the winner's surviving
units. Both numbers come from the simulator's economy rules profile (Simulator/game/rules.py on the
fork: `stage_damage(idx)`, `unit_damage(survivors)`; "set4" or "set18"), never from a table here.
The profile is the one the runner configures: `rules=None` means TFT_RULES (default "set4"), as
tfteval.runner.Game reads it; describe() passes the player's own profile (Player.rules, set from
Game.rules). A simulator without rules.py (upstream) has one table, read from Game_Round.

The expected damage of a loss adds to the base the mean unit damage of the surviving enemy units, from
SURVIVORS: the measured number of surviving enemy units per PvP loss, by stage. Survivors are a fact
about fights, not about the economy, so one histogram serves both profiles; each profile prices it with
its own unit table (Set 4: 2 per unit up to 5, then 1; Set 18: 1 per unit). Measured on the fork
(develop a2840a1) with `python scripts/measure_damage.py --rules set4 --seed 9000 --games 30` (8 rule
bots); the same script under `--rules set18` checks the model out of sample (README, "扣血表";
tests/test_stages.py compares damage_per_loss with both measurements, results/damage/).
"""

from __future__ import annotations

import os
from functools import lru_cache

LAST_IDX = 43  # game_round.game_rounds has 44 entries
_ROUND_IN_STAGE = (1, 2, 3, 5, 6, 7)  # position k = (idx - 3) % 6 inside a stage from stage 2 on

# Surviving enemy units per PvP loss (draws count as a loss with 0 survivors), stage -> counts for
# 0, 1, 2, ... survivors; the last key covers its stage and every later one (few games reach stage 7).
# scripts/measure_damage.py prints this table as "survivors_table" (from results/damage/set4.json:
# 30 games of 8 rule bots, seeds 9000-9029, set4; 2395 losses).
SURVIVORS = {
    2: (2, 91, 170, 185, 122, 27, 4),
    3: (2, 41, 90, 107, 138, 132, 78, 12, 1),
    4: (18, 31, 50, 69, 94, 95, 114, 76, 40, 6, 2),
    5: (25, 27, 28, 39, 55, 69, 71, 69, 41, 7, 4),
    6: (23, 15, 17, 17, 26, 27, 16, 12, 9, 1),
}


def stage_round(idx: int) -> tuple[int, int]:
    """(stage, round) of the fight played after planning phase `idx`."""
    if idx < 0:
        raise ValueError(f"negative round index {idx}")
    if idx < 3:
        return 1, idx + 2
    return 2 + (idx - 3) // 6, _ROUND_IN_STAGE[(idx - 3) % 6]


def label(idx: int) -> str:
    stage, rnd = stage_round(idx)
    return f"{stage}-{rnd}"


def parse_label(text) -> int:
    """Index of a stage label such as "4-1". Accepts an int index unchanged. "2-4" (carousel) maps to
    the index whose fight is 2-5, because the simulator merges the carousel into that round."""
    if isinstance(text, int):
        return text
    stage, _, rnd = str(text).strip().partition("-")
    stage, rnd = int(stage), int(rnd)
    if stage == 1:
        if not 1 <= rnd <= 4:
            raise ValueError(f"no round {text}")
        return max(0, rnd - 2)
    if not 1 <= rnd <= 7:
        raise ValueError(f"no round {text}")
    k = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5}[rnd]
    return 3 + (stage - 2) * 6 + k


def is_carousel(idx: int) -> bool:
    return idx % 6 == 0


def is_pve(idx: int) -> bool:
    return idx <= 2 or (idx >= 8 and (idx - 8) % 6 == 0)


def is_pvp(idx: int) -> bool:
    return not is_pve(idx)


def _rounds_until(idx: int, test) -> int | None:
    for n in range(idx, LAST_IDX + 1):
        if test(n):
            return n - idx
    return None


def schedule(idx: int) -> dict:
    """Where planning phase `idx` sits: label, what this round is, and rounds until the next
    carousel / PvE / stage change (0 = this round; None = none left in the schedule)."""
    return {
        "stage": label(idx),
        "idx": idx,
        "pve": is_pve(idx),
        "carousel": is_carousel(idx),  # runs after this planning phase, before the fight
        "to_carousel": _rounds_until(idx, is_carousel),
        "to_pve": _rounds_until(idx, is_pve),
        "to_stage": _rounds_until(idx, lambda n: n > idx and n >= 3 and (n - 3) % 6 == 0),
    }


# --------------------------------------------------------------------------- damage

class _TableRules:
    """A stand-in profile for a simulator without Simulator/game/rules.py (upstream): its one base
    damage table (Game_Round.ROUND_DAMAGE) and battle/stats.py DAMAGE_PER_UNIT."""

    name = "set4"

    def __init__(self):
        from Simulator import config
        from Simulator.battle.stats import DAMAGE_PER_UNIT
        from Simulator.game.game_round import Game_Round

        log, config.LOGMESSAGES = config.LOGMESSAGES, False  # else the constructor writes log.txt
        try:
            self.round_damage = tuple(tuple(row) for row in Game_Round({}, None, None).ROUND_DAMAGE)
        finally:
            config.LOGMESSAGES = log
        self.unit_damage_table = tuple(DAMAGE_PER_UNIT)

    def stage_damage(self, round_index):
        return next(d for last, d in self.round_damage if round_index <= last)

    def unit_damage(self, survivors):
        table = self.unit_damage_table
        if survivors < len(table):
            return table[survivors]
        return table[-1] + (survivors - len(table) + 1) * (table[-1] - table[-2])


@lru_cache(maxsize=None)
def _profile_by_name(name: str):
    try:
        from Simulator.game.rules import get_rules
    except ImportError:  # upstream simulator: Set 4 only
        if name != "set4":
            raise ValueError(f"rules profile {name!r} needs the simulator fork (Simulator/game/rules.py)") from None
        return _TableRules()
    return get_rules(name)


def rules_profile(rules=None):
    """The simulator's economy rules profile: a profile object (Player.rules) as it is, a name
    ("set4", "set18"), or None for TFT_RULES (default "set4"), the way tfteval.runner.Game picks it."""
    if rules is not None and not isinstance(rules, str):
        return rules
    return _profile_by_name(str(rules or os.environ.get("TFT_RULES", "set4")).lower())


def base_damage(idx: int, rules=None) -> int:
    """Base player damage for losing the fight after planning phase `idx` (the profile's stage damage)."""
    return int(rules_profile(rules).stage_damage(idx))


def unit_damage(survivors: int, rules=None) -> int:
    """Player damage for `survivors` surviving enemy units under the profile."""
    return int(rules_profile(rules).unit_damage(int(survivors)))


def survivors_hist(idx: int) -> tuple:
    """SURVIVORS for the stage of round `idx` (stages after the last listed use the last one)."""
    stage = stage_round(idx)[0]
    return SURVIVORS[min(max(stage, min(SURVIVORS)), max(SURVIVORS))]


@lru_cache(maxsize=None)
def _mean_unit_damage(unit_table: tuple, counts: tuple) -> float:
    return sum(c * unit_table[n] for n, c in enumerate(counts)) / sum(counts)


def expected_unit_damage(idx: int, rules=None) -> float:
    """Mean damage of the surviving enemy units in a loss at round `idx`: SURVIVORS priced by the
    profile's unit_damage."""
    rules = rules_profile(rules)
    counts = survivors_hist(idx)
    return _mean_unit_damage(tuple(rules.unit_damage(n) for n in range(len(counts))), counts)


def damage_per_loss(idx: int, rules=None) -> float:
    """Expected HP lost if the PvP fight after planning phase `idx` is lost (0 for PvE)."""
    if is_pve(idx):
        return 0.0
    return round(base_damage(idx, rules) + expected_unit_damage(idx, rules), 2)


def max_damage(idx: int, enemy_units: int, rules=None) -> int:
    """Worst case for a loss: every enemy unit survives."""
    if is_pve(idx):
        return 0
    return base_damage(idx, rules) + unit_damage(enemy_units, rules)


def losses_to_death(hp: float, idx: int, rules=None) -> int:
    """How many PvP losses in a row, starting with this round's fight, would take `hp` to 0 or below,
    at the expected damage per loss of each upcoming round. 1 means the next loss is expected to kill."""
    rules = rules_profile(rules)
    left, n, r = float(hp), 0, idx
    while left > 0:  # past the schedule the last stage keeps charging, so this ends
        dmg = damage_per_loss(r, rules)
        if dmg > 0:
            n += 1
            left -= dmg
        r += 1
    return n

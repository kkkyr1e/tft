"""Stage labels, the round schedule and the damage table of the Set 4 simulator.

The env reports a round index `idx` (info["game_round"]), not a stage label. A planning phase
with index `idx` is followed by the round `game_round.game_rounds[idx]`:

    idx 0       1-1 carousel + 1-2 PvE (both run inside env.reset, no planning phase before them)
    idx 1, 2    1-3, 1-4 PvE
    idx 3..8    2-1, 2-2, 2-3, 2-5 (the 2-4 carousel runs right after this planning phase), 2-6, 2-7 PvE
    then 6 indices per stage the same way: idx 9 = 3-1, idx 14 = 3-7, idx 15 = 4-1, ...

PvE losses cost no HP and do not touch streaks; the 6-7 PvE (idx 32) is skipped by the simulator
(battle/minion.py only fights idx >= 33 there), so it is a round with no fight at all.

Damage for losing a PvP fight is base(idx) + DAMAGE_PER_UNIT[surviving enemy units]
(game/game_round.py ROUND_DAMAGE, battle/stats.py DAMAGE_PER_UNIT). The expected damage per loss
below adds the measured surviving-unit part to the base: 24 rule-bot games with the sim fixes,
seeds 9000-9023, `python scripts/measure_damage.py --seed 9000 --games 24`.
"""

from __future__ import annotations

# (last idx of the bucket, base damage) as in game_round.ROUND_DAMAGE
BASE_DAMAGE = ((3, 0), (9, 2), (15, 3), (21, 5), (27, 8), (10_000, 15))
DAMAGE_PER_UNIT = (0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
# (last idx of the bucket, mean HP lost per PvP loss), same buckets as BASE_DAMAGE.
# Losses measured per bucket: n = 96, 480, 480, 416, 196, 28.
DAMAGE_PER_LOSS = ((3, 4.4), (9, 8.0), (15, 10.6), (21, 13.9), (27, 16.5), (10_000, 23.0))

LAST_IDX = 43  # game_round.game_rounds has 44 entries
_ROUND_IN_STAGE = (1, 2, 3, 5, 6, 7)  # position k = (idx - 3) % 6 inside a stage from stage 2 on


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


def _bucket(table, idx: int) -> float:
    for last, value in table:
        if idx <= last:
            return value
    return table[-1][1]


def base_damage(idx: int) -> int:
    return int(_bucket(BASE_DAMAGE, idx))


def damage_per_loss(idx: int) -> float:
    """Expected HP lost if the PvP fight after planning phase `idx` is lost (0 for PvE)."""
    return 0.0 if is_pve(idx) else _bucket(DAMAGE_PER_LOSS, idx)


def max_damage(idx: int, enemy_units: int) -> int:
    """Worst case for a loss: every enemy unit survives."""
    if is_pve(idx):
        return 0
    return base_damage(idx) + DAMAGE_PER_UNIT[min(enemy_units, len(DAMAGE_PER_UNIT) - 1)]


def losses_to_death(hp: float, idx: int) -> int:
    """How many PvP losses in a row, starting with this round's fight, would take `hp` to 0 or below,
    at the expected damage per loss of each upcoming round. 1 means the next loss is expected to kill."""
    left, n, r = float(hp), 0, idx
    while left > 0:  # past the schedule the last bucket keeps charging, so this ends
        dmg = damage_per_loss(r)
        if dmg > 0:
            n += 1
            left -= dmg
        r += 1
    return n

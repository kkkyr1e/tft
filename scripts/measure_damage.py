"""Measure HP lost per PvP loss by stage under a rules profile (the data behind tfteval/stages.py).

    python scripts/measure_damage.py --rules set4 --seed 9000 --games 36 --workers 3 --out results/damage/set4.json
    python scripts/measure_damage.py --rules set18 --seed 9000 --games 36 --workers 3 --out results/damage/set18.json

Every PvP loss (draws included: both sides lose the base damage) is recorded with its round index and
damage. A loss costs the profile's base damage for the stage plus its damage for the winner's surviving
units (Simulator/game/rules.py: stage_damage, unit_damage); the survivors are read back from the damage
by inverting the profile's unit table. PvE losses cost no HP in the simulator and are not recorded.

Printed (and written to --out): per stage the number of losses, the mean damage, base and unit part,
the histogram of surviving enemy units, and what tfteval.stages.damage_per_loss predicts for that
profile from its stored survivor histograms; and the same per round index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def survivors_of(rules, idx: int, damage: float) -> int:
    """Surviving enemy units behind a loss of `damage` at round `idx` (inverse of rules.unit_damage)."""
    part = int(round(damage)) - rules.stage_damage(idx)
    for n in range(64):
        if rules.unit_damage(n) == part:
            return n
    raise ValueError(f"damage {damage} at idx {idx} is not base + unit damage under {rules.name}")


def play(job):
    seed, lobby, rules_name = job
    from Simulator.game import player as player_mod

    from tfteval import make_policy, play_game
    from tfteval.stages import rules_profile

    rules = rules_profile(rules_name)
    records = []
    original = player_mod.Player.loss_round

    def loss_round(self, damage):
        if not self.combat:  # the simulator ignores a second call in the same fight
            idx = int(self.round)
            records.append((idx, float(damage), survivors_of(rules, idx, damage)))
        return original(self, damage)

    player_mod.Player.loss_round = loss_round
    try:
        result = play_game({f"player_{i}": make_policy(name) for i, name in enumerate(lobby)}, seed, rules=rules_name)
    finally:
        player_mod.Player.loss_round = original
    assert result.rules == rules_name
    return records


def summary(rows: list[tuple], rules_name: str) -> dict:
    from tfteval import stages

    rules = stages.rules_profile(rules_name)
    out = {}
    for key, group in rows:
        dmg = [d for _, d, _ in group]
        surv = Counter(n for _, _, n in group)
        idxs = sorted({i for i, _, _ in group})
        base = [rules.stage_damage(i) for i, _, _ in group]
        out[key] = {
            "n": len(group), "rounds": f"{idxs[0]}-{idxs[-1]}",
            "mean": round(sum(dmg) / len(dmg), 2),
            "base": round(sum(base) / len(base), 2),
            "unit_part": round((sum(dmg) - sum(base)) / len(dmg), 2),
            "survivors_mean": round(sum(n for _, _, n in group) / len(group), 2),
            "survivors": [surv.get(n, 0) for n in range(max(surv) + 1)],
            # what stages.py predicts, averaged over the same losses (rounds weighted as they occur)
            "stages_py": round(sum(stages.damage_per_loss(i, rules) for i, _, _ in group) / len(group), 2),
        }
    return out


def report(records: list, rules_name: str, meta: dict) -> dict:
    from tfteval import stages

    by_stage, by_idx = defaultdict(list), defaultdict(list)
    for rec in records:
        by_stage[stages.stage_round(rec[0])[0]].append(rec)
        by_idx[rec[0]].append(rec)
    pooled = defaultdict(Counter)  # tfteval.stages.SURVIVORS: stages from the last key on pooled
    last = max(stages.SURVIVORS)
    for idx, _, n in records:
        pooled[min(max(stages.stage_round(idx)[0], min(stages.SURVIVORS)), last)][n] += 1
    return {"rules": rules_name, **meta, "losses": len(records),
            "survivors_table": {s: tuple(c.get(n, 0) for n in range(max(c) + 1)) for s, c in sorted(pooled.items())},
            "by_stage": summary([(str(s), g) for s, g in sorted(by_stage.items())], rules_name),
            "by_idx": summary([(str(i), g) for i, g in sorted(by_idx.items())], rules_name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lobby", default="rule:8")
    parser.add_argument("--rules", default=os.environ.get("TFT_RULES", "set4"), choices=["set4", "set18"])
    parser.add_argument("--out", help="also write the summary and the raw losses to this JSON file")
    parser.add_argument("--summarize", metavar="FILE",
                        help="no games: recompute the summary of a file written by --out (after stages.py changed)")
    args = parser.parse_args()

    if args.summarize:
        saved = json.loads(Path(args.summarize).read_text())
        records = [tuple(r) for r in saved["raw"]]
        out = report(records, saved["rules"], {"lobby": saved["lobby"], "seeds": saved["seeds"]})
        Path(args.summarize).write_text(json.dumps({**out, "raw": records}, separators=(",", ":")) + "\n")
    else:
        lobby = [name for part in args.lobby.split(",") for name, _, n in [part.partition(":")]
                 for _ in range(int(n or 1))]
        records = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for game in pool.map(play, [(args.seed + i, lobby, args.rules) for i in range(args.games)]):
                records.extend(game)
        out = report(records, args.rules, {"lobby": args.lobby, "seeds": [args.seed, args.seed + args.games - 1]})
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({**out, "raw": records}, separators=(",", ":")) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Measure HP lost per PvP loss by round index (source of tfteval.stages.DAMAGE_PER_LOSS).

    python scripts/measure_damage.py --seed 9000 --games 24 --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def play(job):
    seed, lobby = job
    from Simulator.game import player as player_mod

    from tfteval import make_policy, play_game

    records = []
    original = player_mod.Player.loss_round

    def loss_round(self, damage):
        if not self.combat:  # the simulator ignores a second call in the same fight
            records.append((int(self.round), float(damage)))
        return original(self, damage)

    player_mod.Player.loss_round = loss_round
    try:
        play_game({f"player_{i}": make_policy(name) for i, name in enumerate(lobby)}, seed)
    finally:
        player_mod.Player.loss_round = original
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lobby", default="rule:8")
    args = parser.parse_args()

    from tfteval.stages import BASE_DAMAGE

    lobby = [name for part in args.lobby.split(",") for name, _, n in [part.partition(":")] for _ in range(int(n or 1))]
    by_idx = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for records in pool.map(play, [(args.seed + i, lobby) for i in range(args.games)]):
            for idx, dmg in records:
                by_idx[idx].append(dmg)
    buckets = []
    lo = 0
    for last, base in BASE_DAMAGE:
        vals = [d for idx, ds in by_idx.items() if lo <= idx <= last for d in ds]
        if vals:
            buckets.append({"idx": f"{lo}-{last}", "base": base, "n": len(vals), "mean": round(sum(vals) / len(vals), 1)})
        lo = last + 1
    per_idx = {idx: {"n": len(ds), "mean": round(sum(ds) / len(ds), 2)} for idx, ds in sorted(by_idx.items())}
    print(json.dumps({"buckets": buckets, "per_idx": per_idx}, indent=1))


if __name__ == "__main__":
    main()

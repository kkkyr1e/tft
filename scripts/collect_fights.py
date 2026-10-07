"""Play games with every PvP fight recorded (tfteval/fightlog.py), as data for the win-probability model.

    python scripts/collect_fights.py --lobby rule:4,mimicfc:2,stance:2 --games 60 --seed 9600 \
        --out results/fights/mixed_9600

Writes <out>.jsonl.gz (one record per round with fights, see tfteval/fightlog.py) and <out>.json
(the games' results, as scripts/run_lobby.py writes them). Games run one after the other in this
process; seats rotate with the seed as in run_lobby.py. Both files are rewritten after every game,
so an interrupted run keeps the games it finished.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import make_policy, play_game  # noqa: E402
from tfteval.fightlog import FightRecorder  # noqa: E402


def parse_lobby(text: str) -> list[str]:
    names = []
    for part in text.split(","):
        name, _, count = part.partition(":")
        names.extend([name.strip()] * int(count or 1))
    if len(names) != 8:
        raise SystemExit(f"a lobby needs 8 seats, got {len(names)}: {names}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lobby", required=True)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=9600)
    parser.add_argument("--out", required=True, help="path prefix: <out>.jsonl.gz and <out>.json")
    args = parser.parse_args()

    names = parse_lobby(args.lobby)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records, results = [], []
    for i in range(args.games):
        seed = args.seed + i
        shift = seed % 8
        order = names[shift:] + names[:shift]
        lobby = {f"player_{k}": make_policy(name) for k, name in enumerate(order)}
        with FightRecorder({"seed": seed, "lobby": {s: p.name for s, p in lobby.items()}}) as rec:
            result = play_game(lobby, seed)
        records.extend(rec.records)
        results.append(result.to_json())
        with gzip.open(f"{out}.jsonl.gz", "wt") as fh:
            for row in records:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        Path(f"{out}.json").write_text(json.dumps(results))
        fights = sum(len(r["fights"]) for r in rec.records)
        print(f"game {i + 1}/{args.games} seed={seed} {result.seconds}s rounds={len(rec.records)} fights={fights}",
              flush=True)


if __name__ == "__main__":
    main()

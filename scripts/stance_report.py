"""Stance distribution of a run with stance seats (tfteval/stance.py).

    TFT_STANCE_LOG=results/e4/smoke_stance_log.jsonl python scripts/run_lobby.py \
        --lobby hero=stance:1,rule:7 --games 16 --seed 9500 --workers 1 --out results/e4/smoke_stance.json
    python scripts/stance_report.py results/e4/smoke_stance_log.jsonl --results results/e4/smoke_stance.json

Prints the share of planning rounds in each stance by stage, the games each stance fired in and its
logged reasons, and with --results the seats' placements, overall and split by whether a stance
fired in the game (descriptive only: which games a stance fires in is not random).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import stages  # noqa: E402
from tfteval.runner import _seat_seed  # noqa: E402
from tfteval.stance import STANCES  # noqa: E402


def load(path: str) -> dict:
    """(seat, seat seed) -> that seat's log rows in round order."""
    games = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        games[(row.get("seat"), row.get("seed"))].append(row)
    return {k: sorted(v, key=lambda r: r["round"]) for k, v in games.items()}


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:3.0f}%" if total else "   -"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="JSONL written through TFT_STANCE_LOG")
    parser.add_argument("--results", help="run_lobby.py output, for placements")
    parser.add_argument("--by-round", action="store_true", help="one line per round label instead of per stage")
    args = parser.parse_args()

    games = load(args.log)
    rows = [r for g in games.values() for r in g]
    key = (lambda r: r["stage"]) if args.by_round else (lambda r: r["stage"].split("-")[0])
    order = sorted({key(r) for r in rows}, key=lambda s: [int(x) for x in s.split("-")])
    print(f"{len(games)} seat-games, {len(rows)} planning rounds\n")
    print("stage   rounds  " + "  ".join(f"{s:>11}" for s in STANCES))
    for label in order:
        here = Counter(r["stance"] for r in rows if key(r) == label)
        total = sum(here.values())
        print(f"{label:<7} {total:6d}  " + "  ".join(f"{here[s]:5d} {_pct(here[s], total)}" for s in STANCES))
    total = Counter(r["stance"] for r in rows)
    print(f"{'all':<7} {len(rows):6d}  " + "  ".join(f"{total[s]:5d} {_pct(total[s], len(rows))}" for s in STANCES))

    print("\nstance       games fired  first fired (median)  reasons (top 4)")
    fired = {s: {k for k, g in games.items() if any(r["stance"] == s for r in g)} for s in STANCES}
    for s in STANCES:
        firsts = sorted(next(r["round"] for r in g if r["stance"] == s) for k, g in games.items() if k in fired[s])
        first = stages.label(firsts[(len(firsts) - 1) // 2]) if firsts else "-"
        why = Counter(r["why"].split(":")[0] for r in rows if r["stance"] == s and r["why"])
        print(f"{s:<12} {len(fired[s]):3d}/{len(games):<3d}     {first:<22} "
              + "; ".join(f"{w} x{n}" for w, n in why.most_common(4)))
    for field, label in (("slow_roll_end", "slow roll ends"), ("fast8_end", "fast 8 dropped"),
                         ("loss_streak_end", "loss streak ends")):
        ends = Counter(r["features"][field].split(": ")[-1] for r in rows if field in r["features"])
        if ends:
            print(f"{label}: " + ", ".join(f"{k} x{n}" for k, n in ends.most_common()))

    if args.results:
        place = {}
        for result in json.loads(Path(args.results).read_text()):
            for seat, p in result["placements"].items():
                place[(seat, _seat_seed(result["seed"], int(seat.split("_")[1])))] = p
        mine = {k: place[k] for k in games if k in place}
        if not mine:
            raise SystemExit("no log rows match the results file (different seeds?)")
        print(f"\nplacement over {len(mine)} games: {statistics.mean(mine.values()):.2f}  "
              f"top4 {_pct(sum(p <= 4 for p in mine.values()), len(mine))}  "
              f"places {sorted(Counter(mine.values()).items())}")
        for s in STANCES[:-1]:
            yes = [p for k, p in mine.items() if k in fired[s]]
            no = [p for k, p in mine.items() if k not in fired[s]]
            fmt = lambda v: f"{statistics.mean(v):.2f} (n={len(v)})" if v else "- (n=0)"  # noqa: E731
            print(f"  {s:<12} fired {fmt(yes):<14} not fired {fmt(no)}")


if __name__ == "__main__":
    main()

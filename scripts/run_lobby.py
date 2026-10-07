"""Run a batch of games for one lobby composition.

    python scripts/run_lobby.py --lobby rule:4,random:4 --games 28 --seed 1000 --workers 2 --out results/rule4_random4.json
    python scripts/run_lobby.py --lobby hero=noisy20:1,rule:7 --games 24 --seed 2000 --out results/hero_noisy20.json

Seats are `policy:count`, comma separated, 8 in total. `alias=policy` reports a seat under its
own name so the seat under test can be told apart from identical opponents.

`--sim` picks the simulator profile (tfteval.runner.SIM_PROFILES): `realistic` (the default: PvE
damage, Fortune orbs, carousel fixes, hidden next opponent, keyed random streams) or `default` (the
fork's options all off), optionally with overrides, e.g. `realistic,rng_streams=shared`. Every
result records the profile and the exact options (`sim`, `sim_options`).

`--record` keeps per-round trajectory rows (tfteval/record.py) of every seat in each result's `records`
(`--record hero` only the seats of that policy name, or a comma list of seats); tfteval/rubric.py scores
them. Recording does not change the games.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # without this a seed does not replay across processes
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import make_policy, play_game, summarize  # noqa: E402


def parse_lobby(text: str) -> list[str]:
    names = []
    for part in text.split(","):
        name, _, count = part.partition(":")
        names.extend([name.strip()] * int(count or 1))
    if len(names) != 8:
        raise SystemExit(f"a lobby needs 8 seats, got {len(names)}: {names}")
    return names


def run_one(job):
    names, seed, rotate = job
    shift = seed % 8 if rotate else 0  # rotate seats so no policy is tied to a seat index
    order = names[shift:] + names[:shift]
    lobby = {f"player_{i}": make_policy(name) for i, name in enumerate(order)}
    return play_game(lobby, seed).to_json()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lobby", required=True, help="e.g. rule:4,random:4 or llm:1,rule:7")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="first game seed; game i uses seed+i")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-rotate", action="store_true", help="keep policies on fixed seats")
    parser.add_argument("--rules", choices=["set4", "set18"], help="economy profile (default set4, or TFT_RULES)")
    parser.add_argument("--sim", help="simulator profile: realistic (default) or default, optionally with overrides "
                                      "such as realistic,rng_streams=shared (or TFT_SIM)")
    parser.add_argument("--record", nargs="?", const="all", metavar="WHO",
                        help="record per-round rows: all seats (no value), or seats / policy names, comma separated")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.record:
        os.environ["TFT_RECORD"] = args.record  # read by the runner in every worker
    if args.rules:
        os.environ["TFT_RULES"] = args.rules  # read by play_game in every worker
    if args.sim:
        from tfteval.runner import sim_options

        sim_options(args.sim)  # fail early on a bad profile
        os.environ["TFT_SIM"] = args.sim  # read by play_game in every worker

    names = parse_lobby(args.lobby)
    jobs = [(names, args.seed + i, not args.no_rotate) for i in range(args.games)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_one, jobs):
            results.append(result)
            out.write_text(json.dumps(results))
            print(f"game {len(results)}/{args.games} seed={result['seed']} {result['seconds']}s", flush=True)

    for policy in sorted({name for result in results for name in result["lobby"].values()}):
        s = summarize(results, policy)
        ci = f"±{s['ci95']:.2f}" if s["ci95"] is not None else "n/a"
        print(f"{policy:>8}: avg place {s['mean']:.2f} {ci}  top4 {s['top4_rate']:.0%}  n={s['n']} games")
    unfinished = sum(not r["finished"] for r in results)
    fallbacks = sum(sum(r["fallbacks"].values()) for r in results)
    print(f"unfinished games: {unfinished}   policy errors replaced by random actions: {fallbacks}")


if __name__ == "__main__":
    main()

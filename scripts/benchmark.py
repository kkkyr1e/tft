"""Benchmark v1: one agent against sampled opponent lobbies under the frozen configuration benchmarks/v1.json.

    # the agent on the dev pool, Set 4 economy, with the hero's rounds recorded for the rubric and variety
    python scripts/benchmark.py --agent stance --track set4 --record --workers 3 --out results/bench/v1_set4_dev_stance.json

    # the scorecard of existing results (several files of one run are merged), with a comparison agent
    python scripts/benchmark.py --scorecard results/bench/v1_set4_dev_stance.json \\
        --compare results/bench/v1_set4_dev_mimic.json

    # milestones only: the held-out pool and seeds
    python scripts/benchmark.py --agent stance --track set4 --heldout --record --out results/bench/v1_set4_heldout_stance.json

Every game: the agent on seat player_(seed mod 8), reported as `hero`, and 7 opponents drawn for that seed
from the pool with replacement (tfteval/benchmark.py). Dev pool {rule, mimic, mimicfc, fast8, stance,
stance+hold}, seeds 30000 on; held-out pool {stance1, stance+lossstreak, noisy20}, seeds 39000 on, never
used for tuning. Simulator profile `realistic`; rules from the track (set4, or set18 as the second track).

The number of games comes from tfteval.stats.games_needed for the configured interval (±0.25 places at an
SD of 2.25 per game: 312 games), printed before anything runs; --sd gives a measured SD instead and
--games overrides the count (smoke runs; the card then shows the shortfall). The output is written after
every game and a rerun continues where it stopped; it refuses a file made under another configuration hash,
agent, track, pool, recording setting or simulator commit. At the end the scorecard is printed (Markdown)
and written next to --out as .scorecard.json and .scorecard.md.

Decision-bank regret is a hook: with --bank and --bank-scorer, and if scripts/score_bank.py exists (it comes
with the decision bank), the card includes the JSON it writes (see tfteval.benchmark.bank_regret).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # without this a seed does not replay across processes
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import benchmark as bm  # noqa: E402

HELDOUT_WARNING = ("WARNING: held-out pool and seeds. For milestones only: do not tune on these results, "
                   "and do not run them after every change.")


def card_paths(out: Path) -> tuple[Path, Path]:
    stem = str(out.with_suffix("")) if out.suffix == ".json" else str(out)
    return Path(stem + ".scorecard.json"), Path(stem + ".scorecard.md")


def write_card(doc: dict, config: dict, out: Path, args, compare: dict | None) -> None:
    meta = doc["meta"]
    bank = bm.bank_regret(meta["agent"], meta["rules"], args.bank, args.bank_scorer, args.bank_cmd)
    card = bm.scorecard(doc, config, compare=compare, bank=bank)
    text = bm.scorecard_markdown(card)
    print(text)
    js, md = card_paths(out)
    js.write_text(json.dumps(card, indent=1))
    md.write_text(text + "\n")
    print(f"\nscorecard: {js} and {md}")


def run(args, config: dict) -> None:
    from tfteval import simfixes

    pool = bm.pool_name(args.heldout)
    rules = bm.track_rules(config, args.track)
    if args.heldout:
        print(HELDOUT_WARNING, flush=True)
    planned = bm.games_planned(config, args.sd, args.half_width)
    print(f"benchmark {config['name']}-v{config['version']} (config {bm.config_hash(config)}): agent {args.agent}, "
          f"track {args.track} ({rules}), {pool} pool, sim {bm.config_sim(config)}", flush=True)
    print(f"sample size: {bm.needed_note(config, args.sd, args.half_width)}", flush=True)
    games = args.games or planned
    if args.games and args.games != planned:
        print(f"--games {args.games} overrides the planned {planned}; the interval will not reach the target",
              flush=True)

    out = Path(args.out)
    meta = bm.new_meta(config, args.agent, args.track, pool, args.record, planned, simfixes.sim_commit(),
                       bm.harness_commit())
    if out.exists():
        doc = bm.load_results(out)
        for warning in bm.check_meta(doc["meta"], meta):
            print(f"note: {warning}", flush=True)
        sd = bm.measured_sd(doc)
        if sd is not None:
            print(f"measured SD so far {sd:.2f} over {len(doc['games'])} games: "
                  f"{bm.needed_note(config, sd, args.half_width)} (the planned count stays as set before the run)",
                  flush=True)
    else:
        doc = {"meta": meta, "games": []}
    done = {g["seed"] for g in doc["games"]}
    todo = [s for s in bm.seeds(config, pool, games) if s not in done]
    print(f"{len(done)} games already in {out}, {len(todo)} to play, {args.workers} workers", flush=True)

    # play_one passes rules, sim, pickers and recording explicitly; these cover code that reads the defaults
    os.environ["TFT_RULES"], os.environ["TFT_SIM"] = rules, bm.config_sim(config)
    harness = meta["harness_commit"]
    started = time.time()
    jobs = [(config, args.agent, args.track, pool, seed, args.record) for seed in todo]
    with ProcessPoolExecutor(max_workers=args.workers) as pool_ex:
        pending = set()
        while jobs or pending:
            while jobs and len(pending) < args.workers:
                pending.add(pool_ex.submit(bm.play_one, jobs.pop(0)))
            finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            for fut in finished:
                game = fut.result()
                game["harness_commit"] = harness
                doc["games"].append(game)
                doc["games"].sort(key=lambda g: g["seed"])
                if not doc["meta"].get("sim_commit"):
                    doc["meta"]["sim_commit"] = game["result"].get("sim_commit")
                bm.save_results(out, doc)
                place = bm.hero_place(game)
                print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} seed {game['seed']}: hero {place} "
                      f"({game['stance_opponents']} stance-family opponents, {game['result']['seconds']}s; "
                      f"{len(jobs) + len(pending)} left, {time.time() - started:.0f}s so far)", flush=True)
    write_card(doc, config, out, args, load_compare(args, doc))


def load_compare(args, doc: dict) -> dict | None:
    if not args.compare:
        return None
    other = bm.load_results(args.compare)
    bm.check_meta(other["meta"], doc["meta"], keys=("config_hash", "track", "pool"))
    return other


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", help="the agent's policy spec (make_policy), e.g. stance, mimic, llm")
    parser.add_argument("--track", help="economy track from the config: set4 (default) or set18")
    parser.add_argument("--heldout", action="store_true", help="held-out pool and seeds: milestones only")
    parser.add_argument("--record", action="store_true", help="record the hero's rounds (rubric and variety)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--games", type=int, help="override the planned number of games (smoke runs)")
    parser.add_argument("--half-width", type=float, help="target 95%% half width in places (default from config)")
    parser.add_argument("--sd", type=float, help="measured SD of placement per game (default from config)")
    parser.add_argument("--config", default=str(bm.CONFIG), help="benchmark configuration (default benchmarks/v2.json)")
    parser.add_argument("--out", help="results JSON (resumable)")
    parser.add_argument("--scorecard", nargs="+", metavar="RESULTS", help="print the card of existing results")
    parser.add_argument("--compare", help="results of a comparison agent (same config, track and pool)")
    parser.add_argument("--bank", help="decision bank JSONL (hook: needs scripts/score_bank.py)")
    parser.add_argument("--bank-scorer", help="agent spec scripts/score_bank.py scores with --agent (e.g. stance)")
    parser.add_argument("--bank-cmd", default=bm.BANK_CMD, help="command template of the bank hook")
    args = parser.parse_args()
    config = bm.load_config(args.config)

    if args.scorecard:
        doc = bm.merge_results([bm.load_results(p) for p in args.scorecard], config)
        out = Path(args.out) if args.out else Path(args.scorecard[0])
        if len(args.scorecard) > 1 and not args.out:
            raise SystemExit("several results files: give --out for the merged scorecard's name")
        if doc["meta"].get("pool") == "heldout":
            print(HELDOUT_WARNING)
        write_card(doc, config, out, args, load_compare(args, doc))
        return
    if not args.agent or not args.out:
        raise SystemExit("give --agent and --out to run, or --scorecard RESULTS")
    args.track = args.track or config["default_track"]
    bm.track_rules(config, args.track)  # fail early on an unknown track
    from tfteval import make_policy

    make_policy(args.agent)  # fail early on an unknown policy
    run(args, config)


if __name__ == "__main__":
    main()

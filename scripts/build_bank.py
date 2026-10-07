"""Build decision-bank items: saved decision points, three candidate commitments, labels from branches.

    # pilot: 2 source games x 2 decision points x 3 candidates x (2 discovery + 2 label) branches
    python scripts/build_bank.py --lineups stance@rule:7 --seeds 9100:2 --points 3-2 4-1 --kd 2 --kl 2 \\
        --workers 2 --out results/bank/pilot.jsonl

A source game is a lineup (`HERO@OPPONENTS`: the hero's policy up to the decision point, a plan-seat kind
such as `stance` or `mimic`, and the 7 other seats) and a seed; the hero's seat rotates with the seed.
For every source game and decision point (3-2, 3-5, 4-1, 4-5, 5-1) the game is played to the planning
phase of that round and saved; each candidate of the point's menu (tfteval.bank.POINTS) is then played
from the save with re-seeds 0 .. Kd+Kl-1, the hero committed to the candidate for a few rounds and then
playing --continuation (default `stance`), until the hero is out. One item per line in --out (JSONL; the
schema is in tfteval/bank.py and README). A rerun skips the items already in --out, so a build can be
stopped and resumed; items whose hero is out before the point, or whose candidates all played the same
actions, are written too (with `dropped`) so they are not rebuilt.

A larger --kl than an existing item has extends it: its state is rebuilt from the recipe (checked against
the fingerprint, on the same simulator commit), the missing label branches are played and the item is
written again with new labels (a later line replaces an earlier one with the same id).
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

if os.environ.get("PYTHONHASHSEED") != "0":  # recipes only replay with a pinned hash seed
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import bank  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    """`9100:30` (first:count), `9100-9129` or `9100,9105,9111`."""
    if ":" in text:
        first, count = text.split(":")
        return list(range(int(first), int(first) + int(count)))
    if "-" in text:
        first, last = text.split("-")
        return list(range(int(first), int(last) + 1))
    return [int(s) for s in text.split(",")]


def run_job(job):
    if job[0] == "extend":
        _, item, config = job
        return bank.extend_item(item, config["kl"], config["harness"])
    _, lineup, seed, point, config = job
    return bank.build_item(lineup, seed, point, config["kd"], config["kl"], config["continuation"],
                           config["settings"], config["harness"], config["early_drop"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lineups", nargs="+", default=["stance@rule:7"],
                        help="source lineups HERO@OPPONENTS, e.g. stance@rule:7 mimic@rule:4,stance:3")
    parser.add_argument("--seeds", required=True, help="first:count, first-last or a comma list")
    parser.add_argument("--points", nargs="+", default=list(bank.POINTS), choices=list(bank.POINTS))
    parser.add_argument("--kd", type=int, default=2, help="discovery branches per candidate (pick the best)")
    parser.add_argument("--kl", type=int, default=2, help="label branches per candidate (regrets)")
    parser.add_argument("--continuation", default="stance", help="the hero's planner after the commitment")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rules", choices=["set4", "set18"], help="economy profile (default set4, or TFT_RULES)")
    parser.add_argument("--early-drop", action="store_true",
                        help="skip the label branches when every candidate repeated the same actions in all "
                             "discovery branches (the item is then dropped as all duplicates)")
    parser.add_argument("--out", required=True, help="bank JSONL, appended to")
    args = parser.parse_args()
    if args.rules:
        os.environ["TFT_RULES"] = args.rules  # inherited by the workers
    if args.workers > 2:
        print("note: more than 2 workers; other jobs share this machine", flush=True)

    for lineup in args.lineups:
        hero, _ = bank.parse_lineup(lineup)
        bank.plan_kind(hero)  # the hero must be a plan seat: the continuation keeps its executor state
    bank.plan_kind(args.continuation)

    out = Path(args.out)
    items = bank.load_items(out) if out.exists() else []
    settings = bank.sim_settings()
    config = {"kd": args.kd, "kl": args.kl, "continuation": args.continuation, "settings": settings,
              "harness": bank.harness_commit(), "early_drop": args.early_drop}
    for it in items:
        mismatch = [key for key, mine in (("kd", args.kd), ("continuation", args.continuation)) if it.get(key) != mine]
        mismatch += [key for key in ("rules", "sim_profile", "sim_options")
                     if (it["recipe"].get(key) or None) != (settings.get(key) or None)]
        if mismatch:
            raise SystemExit(f"{out} holds {it['id']} built with other settings ({', '.join(mismatch)}); "
                             f"use another --out")
    done = {it["id"]: it for it in items}
    wanted = [(lineup, seed, point) for lineup in args.lineups for seed in parse_seeds(args.seeds)
              for point in args.points]
    jobs = [("build", lineup, seed, point, config) for lineup, seed, point in wanted
            if bank.item_id(lineup, seed, point) not in done]
    branches = sum(len(bank.POINTS[j[3]]["menu"]) for j in jobs) * (args.kd + args.kl)
    extend = [done[bank.item_id(*w)] for w in wanted if bank.item_id(*w) in done]
    extend = [it for it in extend if not it.get("dropped") and int(it["kl"]) < args.kl]
    jobs += [("extend", it, config) for it in extend]
    branches += sum(len(it["candidates"]) * (args.kl - int(it["kl"])) for it in extend)
    print(f"{len(jobs) - len(extend)} items to build, {len(extend)} to extend to kl={args.kl} ({len(done)} already "
          f"in {out}), up to {branches} branches, {args.workers} workers; harness {config['harness']}, "
          f"rules {settings['rules']}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
        pending = set()
        while jobs or pending:
            while jobs and len(pending) < args.workers:
                pending.add(pool.submit(run_job, jobs.pop(0)))
            finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            for fut in finished:
                item = fut.result()
                fh.write(json.dumps(item) + "\n")
                fh.flush()
                items = [it for it in items if it["id"] != item["id"]] + [item]
                lab = item.get("label") or {}
                what = item["dropped"] or (f"{lab['kind']}, best {lab['best']}, regrets "
                                           + " ".join(f"{c}:{r:+.2f}" for c, r in lab["regret"].items()))
                t = item.get("timing", {})
                if item.get("extended"):
                    took = f"extended to kl={item['kl']} in {item['extended'][-1]['seconds']}s"
                else:
                    took = f"{t.get('item_seconds', t.get('rebuild_seconds'))}s, "
                    took += f"{t.get('branch_seconds_mean')}s a branch"
                print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {item['id']}: {what} ({took}; "
                      f"{len(jobs) + len(pending)} left, {time.time() - started:.0f}s so far)", flush=True)
    print(json.dumps(bank.bank_summary(items), indent=1))


if __name__ == "__main__":
    main()

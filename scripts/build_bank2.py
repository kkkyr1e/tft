"""Build decision-bank v2 items: triggered decision points, candidates raced on paired branches.

    # trigger hit rates only (no branches): one line per source game
    python scripts/build_bank2.py --lineups stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1 \\
        --seeds 9400:20 --scan-only --out results/bank2/pilot.scan.jsonl
    # items: every stratum of every source game, the default racing (8 branches each, then elimination)
    python scripts/build_bank2.py --lineups stance@rule:7 --seeds 9400:30 --out results/bank2/bank.jsonl
    # small pilot budgets
    python scripts/build_bank2.py ... --k-start 2 --k-step 2 --k-min 4 --k-max 4 --budget-per-cand 4

A source game is a lineup (`HERO@OPPONENTS`, as in build_bank.py) and a seed. For every source game and
stratum (tfteval.bank2.STRATA: pairs3, streak4, losing3, lowhp4, ref41) the game is played round by round
through the stratum's window; the item is made at the first planning phase where the trigger fires
(that round goes into the recipe). The menu's candidates are then raced (tfteval.bank2.race): K_START
branches each, candidates identical to an earlier one merged into it, then the ones clearly behind the
leader are dropped and the survivors get more branches up to the budget (BUDGET_PER_CAND per distinct
candidate). A game whose trigger never fires (or whose hero is out first) is written too, with
`dropped` and the scanned rounds, so the hit rates can be counted and the item is not tried again.
ref41 is a reference stratum and only every REF_SAMPLE_MOD-th seed (seed % 3 == 0) gets an item of it.

A rerun skips the items already in --out, so a build can be stopped and resumed. The settings (racing,
continuation, simulator, strata definitions) must match the items already in the file, or the build is
refused; with --extend, items raced with other racing settings are raced again with these (their
branches are reused, the missing ones played from the rebuilt state; a later line replaces an earlier
one with the same id). ref41 at 4-1 is the same state and menu as lowhp4 when that fires at 4-1: it then
copies lowhp4's branches (`copied_from`) instead of playing them again (jobs run lowhp4 first).

Simulator settings as in build_bank.py (--sim, --rules, TFT_PICKERS). The CPU is shared: --workers
defaults to 1; run it under `nice -n 19` when other jobs are running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # recipes only replay with a pinned hash seed
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import bank, bank2, stages  # noqa: E402

# lowhp4 before ref41: ref41 copies lowhp4's branches when both are the same 4-1 state
JOB_ORDER = ("pairs3", "losing3", "lowhp4", "streak4", "ref41")


def parse_seeds(text: str) -> list[int]:
    if ":" in text:
        first, count = text.split(":")
        return list(range(int(first), int(first) + int(count)))
    if "-" in text:
        first, last = text.split("-")
        return list(range(int(first), int(last) + 1))
    return [int(s) for s in text.split(",")]


def run_job(job):
    kind = job[0]
    if kind == "scan":
        _, lineup, seed, config = job
        return bank2.scan_game(lineup, seed, config["settings"], last=config["last"], first=config["first"])
    if kind == "rerace":
        _, item, config = job
        return bank2.rerace_item(item, config["racing"], config["harness"])
    _, lineup, seed, stratum, config, donors = job
    return bank2.build_item(lineup, seed, stratum, config["racing"], config["continuation"], config["settings"],
                            config["harness"], donors)


def scan_summary(rows: list[dict]) -> dict:
    """Per stratum: the source games sampled for it (bank2.sampled), the ones where the trigger fires and
    at which rounds (`rate`: items per source game scanned); of the other sampled games, how many lost
    the hero before the window ended and how many reached the window but never met the trigger."""
    out = {"games": len(rows), "strata": {}}
    for s in bank2.STRATA:
        last = max(bank2.window_rounds(s))
        mine = [r for r in rows if bank2.sampled(s, r["seed"])]
        fired = [r["fired"][s] for r in mine if r["fired"].get(s) is not None]
        missed = [r for r in mine if r["fired"].get(s) is None]
        out_first = sum(1 for r in missed if r["out_at"] is not None and r["out_at"] <= last)
        out["strata"][s] = {"sampled": len(mine), "fired": len(fired), "rate": len(fired) / len(rows) if rows else None,
                            "rounds": dict(sorted(Counter(stages.label(x) for x in fired).items(),
                                                  key=lambda kv: stages.parse_label(kv[0]))),
                            "hero_out": out_first, "no_fire": len(missed) - out_first}
    secs = [r["seconds"] for r in rows]
    out["seconds_per_game"] = sum(secs) / len(secs) if secs else None
    return out


def print_scan(summary: dict) -> None:
    print(f"{summary['games']} source games scanned ({summary['seconds_per_game']:.0f}s a game)")
    for s, e in summary["strata"].items():
        rounds = " ".join(f"{k}:{v}" for k, v in e["rounds"].items())
        print(f"  {s:8} fired in {e['fired']:3} of {e['sampled']} sampled games ({e['rate']:.0%} of all)  "
              f"rounds {rounds or '-'}  (not fired: {e['no_fire']} reached the window, {e['hero_out']} hero out first)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lineups", nargs="+", default=["stance@rule:7"])
    parser.add_argument("--seeds", required=True, help="first:count, first-last or a comma list")
    parser.add_argument("--strata", nargs="+", default=list(bank2.STRATA), choices=list(bank2.STRATA))
    parser.add_argument("--continuation", default="stance", help="the hero's planner after the commitment")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rules", choices=["set4", "set18"], help="economy profile (default set4, or TFT_RULES)")
    parser.add_argument("--sim", help="simulator profile (default realistic, or TFT_SIM)")
    for key in ("k_start", "k_step", "k_min", "k_max", "budget_per_cand"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, help=f"racing: {key} (default {bank2.RACING[key]})")
    parser.add_argument("--z", type=float, help=f"racing: elimination at z paired SEs (default {bank2.ELIM_Z})")
    parser.add_argument("--extend", action="store_true",
                        help="race items built with other racing settings again with these (reusing branches)")
    parser.add_argument("--scan-only", action="store_true",
                        help="only play the source games through the windows and record the trigger features")
    parser.add_argument("--out", required=True, help="JSONL, appended to")
    args = parser.parse_args()
    settings = bank.sim_settings(args.sim, args.rules)
    os.environ["TFT_SIM"], os.environ["TFT_RULES"] = settings["sim"], settings["rules"]
    os.environ["TFT_PICKERS"] = "1" if settings["pickers"] else "0"
    for lineup in args.lineups:
        hero, _ = bank.parse_lineup(lineup)
        bank.plan_kind(hero)
    bank.plan_kind(args.continuation)
    racing = bank2.racing_config(k_start=args.k_start, k_step=args.k_step, k_min=args.k_min, k_max=args.k_max,
                                 budget_per_cand=args.budget_per_cand, z=args.z)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    harness = bank.harness_commit()
    if args.workers > 1:
        print("note: more than 1 worker; other jobs share this machine", flush=True)

    if args.scan_only:
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
        for r in rows:
            if r["settings"] != {k: settings[k] for k in bank.SETTING_KEYS}:
                raise SystemExit(f"{out} holds {r['game']} scanned with other settings; use another --out")
        done = {r["game"] for r in rows}
        first = min(stages.parse_label(bank2.STRATA[s]["window"][0]) for s in args.strata)
        last = max(stages.parse_label(bank2.STRATA[s]["window"][1]) for s in args.strata)
        config = {"settings": settings, "first": first, "last": last}
        jobs = [("scan", lineup, seed, config) for lineup in args.lineups for seed in seeds
                if f"{lineup}#{seed}" not in done]
        print(f"{len(jobs)} source games to scan ({len(done)} already in {out}), rounds "
              f"{stages.label(first)}..{stages.label(last)}, {args.workers} workers; harness {harness}", flush=True)
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
            pending = set()
            while jobs or pending:
                while jobs and len(pending) < args.workers:
                    pending.add(pool.submit(run_job, jobs.pop(0)))
                finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                for fut in finished:
                    row = fut.result()
                    row["harness_commit"] = harness
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
                    fired = " ".join(f"{s}@{stages.label(r)}" for s, r in row["fired"].items() if r is not None)
                    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {row['game']}: {fired or 'nothing fired'}"
                          f"{'; hero out at ' + stages.label(row['out_at']) if row['out_at'] else ''} "
                          f"({row['seconds']:.0f}s; {len(jobs) + len(pending)} left, {time.time() - started:.0f}s so far)",
                          flush=True)
        summary = scan_summary(rows)
        print_scan(summary)
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
        return

    items = bank.load_items(out) if out.exists() else []
    stale = []
    for it in items:
        mismatch = [] if it.get("continuation") == args.continuation else ["continuation"]
        built = bank.recipe_settings(it["recipe"])
        mismatch += [key for key in bank.SETTING_KEYS if built[key] != settings[key]]
        if it.get("trigger") != bank2.trigger_spec(it["stratum"]):
            mismatch.append(f"stratum {it['stratum']} definition")
        if mismatch:
            raise SystemExit(f"{out} holds {it['id']} built with other settings ({', '.join(mismatch)}); "
                             f"use another --out")
        if it.get("racing") != racing and it.get("fingerprint_hash"):
            stale.append(it)
    if stale and not args.extend:
        raise SystemExit(f"{out} holds {len(stale)} items raced with other settings (e.g. {stale[0]['racing']}); "
                         f"give the same racing settings, or --extend to race them again with these")
    done = {it["id"]: it for it in items}
    order = {s: i for i, s in enumerate(JOB_ORDER)}
    wanted = [(lineup, seed, s) for lineup in args.lineups for seed in seeds
              for s in sorted(args.strata, key=lambda s: order.get(s, 99)) if bank2.sampled(s, seed)]
    config = {"racing": racing, "continuation": args.continuation, "settings": settings, "harness": harness}
    todo = [w for w in wanted if bank2.item_id(*w) not in done]
    stale_ids = {it["id"] for it in stale}
    rerace = [done[bank2.item_id(*w)] for w in wanted if bank2.item_id(*w) in stale_ids]
    print(f"{len(todo)} items to build, {len(rerace)} to race again ({len(done)} already in {out}); racing {racing}; "
          f"{args.workers} workers; harness {harness}, rules {settings['rules']}, sim {settings['sim']} "
          f"{settings['sim_options']}, pickers {'on' if settings['pickers'] else 'off'}", flush=True)
    started = time.time()
    by_game = defaultdict(list)
    for it in items:
        by_game[it["game"]].append(it)
    queue = [("build",) + w for w in todo] + [("rerace", it) for it in rerace]
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
        pending = set()
        while queue or pending:
            while queue and len(pending) < args.workers:
                head = queue.pop(0)
                if head[0] == "build":
                    _, lineup, seed, stratum = head
                    names = list(bank2.STRATA[stratum]["menu"])
                    donors = [it for it in by_game[f"{lineup}#{seed}"]
                              if [c["name"] for c in it["candidates"]] == names and it.get("branches")]
                    job = ("build", lineup, seed, stratum, config, donors)
                else:
                    job = ("rerace", head[1], config)
                pending.add(pool.submit(run_job, job))
            finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            for fut in finished:
                item = fut.result()
                fh.write(json.dumps(item) + "\n")
                fh.flush()
                items = [it for it in items if it["id"] != item["id"]] + [item]
                by_game[item["game"]] = [it for it in by_game[item["game"]] if it["id"] != item["id"]] + [item]
                if item.get("dropped"):
                    what = item["dropped"]
                    if item.get("scan"):
                        what += " (scanned " + ", ".join(f"{stages.label(r['round'])}" for r in item["scan"]) + ")"
                else:
                    race = item["race"]
                    what = (f"{item['point']}: {race['stop']}, survivors {','.join(race['survivors'])}, "
                            f"branches " + " ".join(f"{c}:{n}" for c, n in race["n"].items()))
                    if item.get("copied_from"):
                        what += f" (copied from {item['copied_from']})"
                t = item.get("timing", {})
                took = f"{t.get('item_seconds', t.get('scan_seconds'))}s"
                if t.get("branch_seconds_mean"):
                    took += f", {t['branch_seconds_mean']}s a branch"
                if item.get("reraced"):
                    took = f"raced again in {item['reraced'][-1]['seconds']}s, {item['reraced'][-1]['played']} new"
                print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {item['id']}: {what} ({took}; "
                      f"{len(queue) + len(pending)} left, {time.time() - started:.0f}s so far)", flush=True)
    print(json.dumps(bank2.bank_summary(items), indent=1))


if __name__ == "__main__":
    main()

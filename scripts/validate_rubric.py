"""Within-state validation of rubric v1: from the same saved state, does a branch where an item fires end
worse than a branch where it does not?

    python scripts/validate_rubric.py --hero stance --seeds 36000:40 --rounds 3-1 4-1 --branches 4 --workers 3 \\
        --out results/rubric/validate_stance.jsonl
    python scripts/validate_rubric.py --summarize --out results/rubric/validate_stance.jsonl

For every seed the lobby is the benchmark's (tfteval/benchmark.py: the hero on seat player_(seed mod 8) and
7 opponents drawn from the dev pool for that seed; --opponents rule:7 and the like give fixed opponents)
with the hero's rounds recorded (tfteval/record.py). The game is played to the planning phase of each
--rounds entry and saved there (tfteval/branching.py); from each saved state K re-seeded branches
(reseed=k) are played until the hero is out or the game ends. Every branch is scored with rubric v1 on the
hero's rows from the branch round on (--whole-game: from the first round; rounds before the branch are the
same in every branch of a state, so they only shift a flag for the whole state) and labelled with the
hero's placement. Default seeds 36000 on: not the benchmark's dev (30000 on) or held-out (39000 on) seeds.

Summary per item (tfteval.rubric.within_state_association): placement and flag are demeaned within each
state (state fixed effects); `fe_diff` is the within-state difference in mean placement, flagged minus not
flagged (positive: the flagged branches placed worse), with a 95% interval clustered by source game (the
states of one game share its first rounds); `mean_state_diff` averages the per-state differences; and the
number of states that have both flagged and unflagged branches (only those carry information). This is an
association within state, not a causal effect: what made the item fire in one branch (a worse shop, a lost
fight) can itself cost placement. Hard gates are counted, not related to placement.

One JSON line per branch in --out; a rerun skips branches already there and refuses rows of another
configuration. The summary goes to --summary (default: --out with .summary.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # snapshots and seeds only replay with a pinned hash seed
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import stages  # noqa: E402
from tfteval.rubric import GATES, ITEMS, NOTE, within_state_association  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    """`36000:30` (first:count), `36000-36029` or `36000,36005`."""
    if ":" in text:
        first, count = text.split(":")
        return list(range(int(first), int(first) + int(count)))
    if "-" in text:
        first, last = text.split("-")
        return list(range(int(first), int(last) + 1))
    return [int(s) for s in text.split(",")]


def lobby_specs(seed: int, hero: str, opponents: str) -> tuple[dict, str]:
    """seat -> policy spec and the hero's seat: the benchmark's dev lobby, or fixed opponents."""
    from tfteval import benchmark as bm

    if opponents == "dev":
        lobby = bm.sample_lobby(bm.load_config(), "dev", seed, hero)
        return lobby["seats"], lobby["hero_seat"]
    names = []
    for part in opponents.split(","):
        name, _, count = part.partition(":")
        names.extend([name.strip()] * int(count or 1))
    if len(names) != 7:
        raise SystemExit(f"--opponents needs 7 seats, got {len(names)}")
    seat = seed % 8
    specs = names[:seat] + [f"hero={bm.agent_kind(hero)}"] + names[seat:]
    return {f"player_{i}": s for i, s in enumerate(specs)}, f"player_{seat}"


def run_state(job):
    """One saved state: (seed, round, config, [k, ...]) -> one row per branch."""
    from tfteval import make_policy
    from tfteval.branching import branch, play_to, snapshot
    from tfteval.rubric import score_result

    seed, rnd, config, todo = job
    specs, seat = lobby_specs(seed, config["hero"], config["opponents"])
    t0 = time.time()
    game = play_to({s: make_policy(spec) for s, spec in specs.items()}, seed, rnd, sim=config["sim"],
                   rules=config["rules"], record=[seat])
    base = {"config": config, "seed": seed, "round": rnd, "label": stages.label(rnd), "hero_seat": seat}
    if game.done or game.round != rnd or game.player(seat) is None:
        return [{**base, "k": k, "skipped": "hero out before the branch round"} for k in todo]
    snap = snapshot(game)
    make_state = round(time.time() - t0, 2)
    rows = []
    for k in todo:
        started = time.time()
        g = branch(snap, reseed=k)
        while not g.done and seat not in g.placements:  # the hero's place is final once it is out
            g.run(until_round=g.round + 1)
        result = g.result().to_json()
        score = score_result(result, seats=[seat], comps=None, thresholds=config.get("thresholds"),
                             from_round=None if config["whole_game"] else rnd)[0]
        rows.append({**base, "k": k, "place": result["placements"].get(seat), "flags": score["flags"],
                     "detail": score["detail"], "gates": score["gates"], "rounds_scored": score["rounds"],
                     "make_state_seconds": make_state, "branch_seconds": round(time.time() - started, 2)})
    return rows


def summarize(rows: list[dict]) -> dict:
    skipped = sorted({(r["seed"], r["round"]) for r in rows if r.get("skipped")})
    rows = [r for r in rows if not r.get("skipped") and r.get("place") is not None]
    out = {"config": rows[0]["config"] if rows else None, "note": NOTE, "branches": len(rows),
           "states": len({(r["seed"], r["round"]) for r in rows}), "source_games": len({r["seed"] for r in rows}),
           "skipped_states": [list(s) for s in skipped], "items": {}, "by_round": {}}
    for item in ITEMS:
        branches = [{"state": (r["seed"], r["round"]), "cluster": r["seed"], "place": r["place"],
                     "flag": r["flags"][item]} for r in rows]
        out["items"][item] = within_state_association(branches)
        by_round = defaultdict(list)
        for b, r in zip(branches, rows):
            by_round[r["label"]].append(b)
        out["by_round"][item] = {label: within_state_association(bs) for label, bs in sorted(by_round.items())}
    out["gates"] = {g: int(sum(r["gates"].get(g, 0) for r in rows)) for g in GATES}
    out["gates"]["branches_with_any"] = sum(any(r["gates"].get(g, 0) for g in GATES) for r in rows)
    return out


def report(s: dict) -> str:
    lines = [f"rubric v1 within-state validation: {s['states']} states from {s['source_games']} games, "
             f"{s['branches']} branches ({len(s['skipped_states'])} states skipped: hero out before the branch)",
             f"NOTE: {s['note']}"]
    for item, a in s["items"].items():
        head = f"  {item}: flagged in {a['flagged']}/{a['branches']} branches; states with both flagged and " \
               f"unflagged branches: {a['states_mixed']}/{a['states']}"
        if a["fe_diff"] is None:
            lines.append(head + "; no within-state contrast")
            continue
        ci = (f"±{a['ci95']:.2f} (clustered by source game, {a['clusters']} games)" if a["ci95"] is not None
              else "(no interval: one source game)")
        m = a["mean_state_diff"]
        mci = f" ±{m['ci95']:.2f}" if m.get("ci95") is not None else ""
        lines.append(head + f"\n    placement flagged - not flagged, within state: {a['fe_diff']:+.2f} {ci}; "
                     f"mean of per-state differences {m['mean']:+.2f}{mci}")
    g = s["gates"]
    lines.append("  hard gates (counts, not related to placement): " + ", ".join(f"{k} {v}" for k, v in g.items()))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hero", default="stance", help="hero policy (make_policy spec)")
    parser.add_argument("--opponents", default="dev",
                        help="dev (the benchmark's dev-pool lobby for the seed) or 7 fixed seats, e.g. rule:7")
    parser.add_argument("--seeds", default="36000:2", help="first:count, first-last or a comma list")
    parser.add_argument("--rounds", nargs="+", default=["4-1"], help="branch rounds: stage labels or indices")
    parser.add_argument("--branches", type=int, default=4, help="re-seeded branches per state")
    parser.add_argument("--whole-game", action="store_true", help="score the hero's rows from the first round")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sim", default="realistic", help="simulator profile (default realistic, as the benchmark)")
    parser.add_argument("--rules", choices=["set4", "set18"], default="set4")
    parser.add_argument("--out", required=True, help="raw JSONL, appended to")
    parser.add_argument("--summary", help="summary JSON (default: --out with .summary.json)")
    parser.add_argument("--summarize", action="store_true", help="only summarise --out")
    args = parser.parse_args()

    out = Path(args.out)
    summary_path = Path(args.summary) if args.summary else out.with_suffix(".summary.json")
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    if not args.summarize:
        from tfteval import make_policy
        from tfteval.benchmark import load_config
        from tfteval.runner import sim_options

        make_policy(args.hero)  # fail early on an unknown policy
        sim, _ = sim_options(args.sim)
        os.environ["TFT_SIM"], os.environ["TFT_RULES"] = sim, args.rules  # read by code that uses the defaults
        config = {"hero": args.hero, "opponents": args.opponents, "sim": sim, "rules": args.rules,
                  "whole_game": args.whole_game, "thresholds": load_config()["rubric"]}
        if rows and any(r["config"] != config for r in rows):
            raise SystemExit(f"{out} holds rows of another configuration: {rows[0]['config']}")
        done = {(r["seed"], r["round"], r["k"]) for r in rows}
        jobs = []
        for seed in parse_seeds(args.seeds):
            for label in args.rounds:
                rnd = stages.parse_label(label)
                todo = [k for k in range(args.branches) if (seed, rnd, k) not in done]
                if todo:
                    jobs.append((seed, rnd, config, todo))
        print(f"{len(jobs)} states, {sum(len(j[3]) for j in jobs)} branches to run, {args.workers} workers", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
            pending = set()
            while jobs or pending:
                while jobs and len(pending) < args.workers:
                    pending.add(pool.submit(run_state, jobs.pop(0)))
                finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                for fut in finished:
                    new = fut.result()
                    for row in new:
                        fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows += new
                    flags = [{k for k, v in r.get("flags", {}).items() if v} for r in new]
                    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} seed {new[0]['seed']} {new[0]['label']}: "
                          f"places {[r.get('place') for r in new]} flags {[sorted(f) for f in flags]} "
                          f"({len(jobs) + len(pending)} states left, {time.time() - started:.0f}s so far)", flush=True)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=1, default=list))
    print(report(summary))


if __name__ == "__main__":
    main()

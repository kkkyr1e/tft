"""Build contrastive pairs (tfteval.pairs) and report their interactions.

    # pilot: three 4-1 states, hero HP 35 vs 85, roll_all vs save, 48 branches per cell
    python scripts/build_pairs.py --lineup stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1 \\
        --states 9501@4-1 9510@4-1 9517@4-1 --axis hp --values 35 85 --cands roll_all save --k 48 \\
        --workers 3 --fresh-workers --out results/bank2/pairs/hp41.jsonl
    # report only (any number of files)
    python scripts/build_pairs.py --report results/bank2/pairs/hp41.jsonl

Each pair is two jobs, one per side (rebuild, edit, play every candidate on k = 0..K-1); a pair is
written when both of its sides are done. A rerun skips the pairs already in --out. The report prints,
per pair, the mean places of the four cells, the side contrasts A - B and the interaction
(A - B on side 0) - (A - B on side 1), all paired by k, and the interaction pooled over the pairs.
"""

from __future__ import annotations

import argparse
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

from tfteval import bank, bank2, pairs  # noqa: E402


def make_pool(args) -> ProcessPoolExecutor:
    if args.fresh_workers:
        import multiprocessing

        return ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                   max_tasks_per_child=1)
    return ProcessPoolExecutor(max_workers=args.workers)


def run_side(job):
    spec, side = job
    return pairs.play_side(spec, side)


def load(paths) -> list[dict]:
    rows = []
    for path in paths:
        if Path(path).exists():
            rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    latest = {r["id"]: r for r in rows}  # a later line replaces an earlier one with the same id
    return list(latest.values())


def fmt(s: dict) -> str:
    return f"{s['mean']:+.2f} ±{1.96 * s['se']:.2f}"


def report(rows: list[dict]) -> dict:
    out = {}
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["axis"], tuple(r["values"]), tuple(r["cands"]), r["point"], r["continuation"]), []).append(r)
    for (axis, values, cands, point, cont), rs in groups.items():
        a, b = cands
        key = f"{point} {axis} {values[0]} vs {values[1]}, {a} - {b} ({cont})"
        print(f"== {key}: {len(rs)} pairs")
        print(f"  {'pair':10} {'base':>9} {'ks':>3}  {f'side 0 ({axis} {values[0]})':>22}  "
              f"{f'side 1 ({axis} {values[1]})':>22}  {'interaction':>14}  corr")
        ests = []
        for r in sorted(rs, key=lambda r: r["seed"]):
            c = pairs.contrasts(r)
            base = r["base"]["features"]
            ests.append(c["interaction"])
            mp = c["mean_place"]
            corr = f"{c['side_corr']:+.2f}" if c["side_corr"] is not None else "-"
            print(f"  #{r['seed']:<9} {f'hp{base['hp']} g{base['gold']}':>9} {c['ks']:>3}  "
                  f"{fmt(c['side'][0]):>13} ({mp[f'0:{a}']:.2f}/{mp[f'0:{b}']:.2f})  "
                  f"{fmt(c['side'][1]):>13} ({mp[f'1:{a}']:.2f}/{mp[f'1:{b}']:.2f})  "
                  f"{fmt(c['interaction']):>14}  {corr}")
        p = pairs.pooled(ests)
        line = f"  pooled interaction: mean {p['mean']:+.2f} (between-pair SE {p['se_between']:.2f})"
        if "fixed" in p:
            line += (f"; inverse-variance {p['fixed']:+.2f} ±{1.96 * p['fixed_se']:.2f}, "
                     f"random effects {p['random']:+.2f} ±{1.96 * p['random_se']:.2f} (tau {p['tau2'] ** 0.5:.2f})")
        print(line)
        out[key] = {"pairs": [{"id": r["id"], **pairs.contrasts(r)} for r in rs], "pooled": p}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lineup", default="stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1")
    parser.add_argument("--states", nargs="+", help="SEED@POINT, e.g. 9501@4-1")
    parser.add_argument("--axis", choices=pairs.AXES)
    parser.add_argument("--values", nargs=2, type=int, help="the edited value on side 0 and on side 1")
    parser.add_argument("--cands", nargs=2, choices=list(bank2.CANDIDATES), help="candidates A and B")
    parser.add_argument("--k", type=int, default=32, help="branches per cell (side x candidate)")
    parser.add_argument("--continuation", default="stance")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fresh-workers", action="store_true", help="every side job in a new (spawned) process")
    parser.add_argument("--rules", choices=["set4", "set18"])
    parser.add_argument("--sim")
    parser.add_argument("--out")
    parser.add_argument("--report", nargs="+", help="only report these pair files")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()

    if args.report:
        result = report(load(args.report))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=1, default=float) + "\n")
        return
    if not (args.states and args.axis and args.values and args.cands and args.out):
        parser.error("--states, --axis, --values, --cands and --out are needed to build")
    settings = bank.sim_settings(args.sim, args.rules)
    harness = bank.harness_commit()
    done = {r["id"] for r in load([args.out])}
    specs = []
    for text in args.states:
        seed, point = text.split("@")
        spec = {"lineup": args.lineup, "seed": int(seed), "point": point, "axis": args.axis, "values": list(args.values),
                "cands": list(args.cands), "k": args.k, "continuation": args.continuation, "settings": settings,
                "harness": harness}
        if pairs.pair_id(args.lineup, int(seed), point, args.axis, args.values) not in done:
            specs.append(spec)
    print(f"{len(specs)} pairs to build ({len(done)} already in {args.out}); {args.axis} {args.values[0]} vs "
          f"{args.values[1]}, {args.cands[0]} vs {args.cands[1]}, {args.k} branches per cell, {args.workers} workers, "
          f"harness {harness}, sim {settings}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sides: dict = {}
    with make_pool(args) as pool:
        pending = {pool.submit(run_side, (spec, side)): (i, side) for i, spec in enumerate(specs) for side in (0, 1)}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                i, side = pending.pop(fut)
                spec = specs[i]
                try:
                    sides.setdefault(i, []).append(fut.result())
                except Exception as err:  # noqa: BLE001 - report and go on with the other pairs
                    print(f"  #{spec['seed']}@{spec['point']} side {side} failed: {err!r}", flush=True)
                    sides.setdefault(i, []).append(None)
                    continue
                print(f"  #{spec['seed']}@{spec['point']} side {side} done in {sides[i][-1]['seconds']:.0f}s "
                      f"({(time.time() - started) / 60:.0f} min)", flush=True)
                if len(sides[i]) == 2 and None not in sides[i]:
                    pair = pairs.assemble(spec, sides[i])
                    with open(args.out, "a") as fh:
                        fh.write(json.dumps(pair) + "\n")
    report(load([args.out]))


if __name__ == "__main__":
    main()

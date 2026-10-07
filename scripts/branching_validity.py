"""Branching-validity experiment: do short branches from a saved state carry signal?

Hero = plan executor with the rule bot's economy (`mimic`) against 7 rule bots, seats rotated by seed.
For each branch round R in --rounds, --states games are played to the planning phase of round R
(a seed whose hero is already out is replaced by the next candidate seed), the state is saved, and
from it every policy in --policies plays --branches re-seeded futures to the end. Branch k uses the
same re-seed for every policy (common random numbers).

    python scripts/branching_validity.py --out results/branching/raw.jsonl
    python scripts/analyze_branching.py results/branching/raw.jsonl

One JSON line per branch; rerunning skips branches already in --out, so the run can be stopped and
resumed. --max-workers caps busy processes; --until-utc HH:MM:--then N raises the cap at that time.
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

if os.environ.get("PYTHONHASHSEED") != "0":  # snapshots and seeds only replay with a pinned hash seed
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import make_policy  # noqa: E402
from tfteval.branching import branch, play_to, snapshot, switch_planner  # noqa: E402
from tfteval.planner import VARIANTS, ParamPlanner  # noqa: E402

SEED_BASE = {9: 7200, 13: 7300, 17: 7400}
HORIZON = 8  # record the hero every round up to R + HORIZON


def lobby(seed):
    names = ["hero=mimic"] + ["rule"] * 7
    shift = seed % 8
    order = names[shift:] + names[:shift]
    return {f"player_{i}": make_policy(name) for i, name in enumerate(order)}


def hero_seat(seed):
    return f"player_{(8 - seed % 8) % 8}"


def hero_state(game, seat):
    p = game.player(seat)
    if p is None:
        return None
    return {"hp": float(p.health), "gold": int(p.gold), "level": int(p.level), "xp": int(p.exp),
            "win_streak": int(p.win_streak), "loss_streak": int(p.loss_streak)}


class TracingPlanner(ParamPlanner):
    """A variant planner that also notes the rounds where its plan differs from mimic's."""

    def __init__(self, kind):
        super().__init__(kind, **VARIANTS[kind])
        self.reference = ParamPlanner("mimic")
        self.differs = []

    def plan(self, state, comps, comp_now):
        mine = super().plan(state, comps, comp_now)
        if mine != self.reference.plan(state, comps, comp_now):
            self.differs.append(state["round"])
        return mine


def make_state(rnd, index):
    """Play candidate seeds until the hero is still in at round `rnd`; return (seed, game)."""
    for attempt in range(20):
        seed = SEED_BASE[rnd] + index + 100 * attempt
        game = play_to(lobby(seed), seed, rnd)
        if not game.done and game.player(hero_seat(seed)) is not None:
            return seed, game, attempt
    raise RuntimeError(f"no candidate seed for round {rnd} index {index}")


def run_job(job):
    rnd, index, policy, ks = job
    t0 = time.time()
    seed, game, skipped = make_state(rnd, index)
    seat = hero_seat(seed)
    snap = snapshot(game)
    at_branch = hero_state(game, seat)
    others = sorted((float(p.health) for s, p in game.env.unwrapped.player_manager.player_states.items()
                     if p is not None and s != seat), reverse=True)
    state_info = {"round": rnd, "index": index, "seed": seed, "hero_seat": seat, "skipped_seeds": skipped,
                  "hero": at_branch, "others_hp": others, "steps": game.steps,
                  "comp_number": game.seat_policies[seat].executor.comp_number,
                  "snapshot_seconds": round(snap.seconds, 4), "snapshot_bytes": snap.size,
                  "make_state_seconds": round(time.time() - t0, 2)}
    rows = []
    for k in ks:
        started = time.time()
        planner = TracingPlanner(policy)
        t_restore = time.perf_counter()
        g = branch(snap, reseed=k, switch={seat: switch_planner(planner)})
        restore_seconds = time.perf_counter() - t_restore
        trajectory = []
        for h in range(1, HORIZON + 1):
            g.run(until_round=rnd + h)
            trajectory.append(hero_state(g, seat) if not g.done and g.round == rnd + h else None)
        result = g.run().result()
        rows.append({**state_info, "policy": policy, "k": k, "place": result.placements.get(seat),
                     "placements": result.placements, "finished": result.finished,
                     "trajectory": trajectory, "plan_differs_from_mimic": planner.differs,
                     "fallbacks": result.fallbacks.get(seat, 0), "restore_seconds": round(restore_seconds, 4),
                     "branch_seconds": round(time.time() - started, 2)})
    return rows


def cap(args):
    if args.until_utc and dt.datetime.now(dt.timezone.utc).strftime("%H:%M") >= args.until_utc:
        return args.then
    return args.max_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", default="9,13,17")
    parser.add_argument("--states", type=int, default=10, help="states per round")
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--policies", default="mimic,fast8")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--until-utc", default=None, help="HH:MM (UTC) after which --then workers may run")
    parser.add_argument("--then", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["round"], row["index"], row["policy"], row["k"]))

    jobs = []
    for index in range(args.states):  # interleave rounds so a partial run covers all of them
        for rnd in [int(r) for r in args.rounds.split(",")]:
            for policy in args.policies.split(","):
                ks = [k for k in range(args.branches) if (rnd, index, policy, k) not in done]
                if ks:
                    jobs.append((rnd, index, policy, ks))
    print(f"{len(jobs)} jobs, {sum(len(j[3]) for j in jobs)} branches to run", flush=True)

    pending = set()
    with ProcessPoolExecutor(max_workers=max(args.max_workers, args.then)) as pool, out.open("a") as fh:
        while jobs or pending:
            while jobs and len(pending) < cap(args):
                pending.add(pool.submit(run_job, jobs.pop(0)))
            finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for fut in finished:
                rows = fut.result()
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                r = rows[0]
                print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} R={r['round']} idx={r['index']} seed={r['seed']} "
                      f"{r['policy']}: places {[x['place'] for x in rows]} "
                      f"({sum(x['branch_seconds'] for x in rows):.0f}s, {len(jobs)} jobs left)", flush=True)


if __name__ == "__main__":
    main()

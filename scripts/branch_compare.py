"""Compare two hero policies on the same saved states, with paired re-seeded branches.

    # does field_comp help? branch where it starts acting (3-3), 30 states x 5 branches per policy
    python scripts/branch_compare.py --a mimic --b mimicfc --round 3-3 --seeds 9700:30 --branches 5 \\
        --out results/branching/mimicfc_3-3.jsonl

    # a controlled loss streak in stage 2, with the state checkpoints and the fight window reported
    python scripts/branch_compare.py --a mimic --b fodder2 --round 2-1 --seeds 9800:40 --branches 5 \\
        --at 3-1 4-1 --window 2-1 2-6 --out results/branching/fodder2_2-1.jsonl

For every seed the hero plays policy A (on a seat rotated by the seed, against --opponents) to the
planning phase of --round, and the game is saved there. From that one snapshot, branch k of A and
branch k of B are played with the same re-seed k (common random numbers: they share every random draw
until the hero's actions first differ) until the hero is out or the game ends; the label of a branch
is the hero's final placement. A plan-executor hero gets B's planner through `switch_planner`, so it
keeps its executor state (comp, pairs, round checks); another kind of policy is switched in fresh.

The simulator profile (--sim, default realistic: keyed random streams among others) and the economy
rules (--rules) are part of the configuration. Under rng_streams="keyed" every event draws from its own
stream, so the two arms of branch k keep sharing the random numbers of the events they have in common
after the hero's actions differ; with "shared" streams the first different action shifts every later
draw of every seat.

One JSON line per branch in --out; a rerun skips branches already there, so a run can be stopped and
resumed (with the same --a/--b/--round/--opponents/--sim/--rules). At the end, or with --summarize alone, the
summary goes to --summary (default: --out with .summary.json): the mean over states of the paired
placement difference B - A with a 95% t interval over states (a state's branches share its past, so
the state is the independent unit), the share of states where B changed any hero action, per-state
differences and their correlation with the state, hero state at the --at checkpoints, and the
fights of the --window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # snapshots and seeds only replay with a pinned hash seed
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # statistics of tiny subsets come out as NaN

from tfteval import stages  # noqa: E402
from tfteval.stats import cluster_mean_ci  # noqa: E402

ARMS = ("A", "B")


# --------------------------------------------------------------------------- playing

def parse_seats(text: str) -> list[str]:
    names = []
    for part in text.split(","):
        name, _, count = part.partition(":")
        names.extend([name.strip()] * int(count or 1))
    return names


def parse_seeds(text: str) -> list[int]:
    """`9700:30` (first:count), `9700-9729` or `9700,9705,9711`."""
    if ":" in text:
        first, count = text.split(":")
        return list(range(int(first), int(first) + int(count)))
    if "-" in text:
        first, last = text.split("-")
        return list(range(int(first), int(last) + 1))
    return [int(s) for s in text.split(",")]


def hero_seat(seed: int) -> str:
    return f"player_{(8 - seed % 8) % 8}"


def lobby(seed: int, hero: str, opponents: list[str]) -> dict:
    from tfteval import make_policy

    names = [f"hero={hero}"] + opponents
    shift = seed % 8  # hero on seat (8 - shift) % 8, as in run_lobby.py and branching_validity.py
    order = names[shift:] + names[:shift]
    return {f"player_{i}": make_policy(name) for i, name in enumerate(order)}


def unit_value(units) -> int:
    from Simulator.game.pool_stats import cost_star_values

    return sum(cost_star_values[u.cost - 1][u.stars - 1] for u in units)


def player_state(p) -> dict:
    from tfteval import public
    from tfteval.executor import strength

    board = public.board_units(p)
    bench = [c for c in p.bench if public.is_unit(c)]
    return {"hp": float(p.health), "gold": int(p.gold), "level": int(p.level), "xp": int(p.exp),
            "win_streak": int(p.win_streak), "loss_streak": int(p.loss_streak),
            "board_units": len(board), "bench_units": len(bench), "unit_value": unit_value(board + bench),
            "board_strength": sum(strength(u) for u in board)}


def rank(value: float, others: list[float]) -> float:
    """1 = highest; ties share the average rank."""
    above = sum(o > value for o in others)
    tied = sum(o == value for o in others)
    return 1 + above + tied / 2


def lobby_state(game, seat: str) -> dict:
    """The hero at the branch point and where it stands in the lobby."""
    players = {s: p for s, p in game.env.unwrapped.player_manager.player_states.items() if p is not None}
    me = player_state(players[seat])
    others = [player_state(p) for s, p in sorted(players.items()) if s != seat]
    return {"hero": me, "alive": len(players),
            "hp_rank": rank(me["hp"], [o["hp"] for o in others]),
            "strength_rank": rank(me["board_strength"], [o["board_strength"] for o in others]),
            "others_hp": sorted((o["hp"] for o in others), reverse=True),
            "others_strength": sorted((o["board_strength"] for o in others), reverse=True)}


class ActionLog:
    """Wraps the hero's policy and hashes its actions round by round, so two branches with the same
    re-seed can be checked for whether the hero ever acted differently."""

    def __init__(self, policy):
        self.policy, self.name, self.rounds = policy, policy.name, {}

    def reset(self, seed):
        self.policy.reset(seed)

    def act(self, observation, info, agent, env):
        action = self.policy.act(observation, info, agent, env)
        rnd = int(info.get("game_round", 1))
        self.rounds.setdefault(rnd, hashlib.sha256()).update(repr(list(map(int, action))).encode())
        return action

    def digests(self) -> dict:
        return {str(r): h.hexdigest()[:12] for r, h in sorted(self.rounds.items())}


class TracingPlanner:
    """A planner that also notes the rounds where its plan differs from a reference planner's
    plan for the same state (both must be stateless, e.g. ParamPlanner variants)."""

    def __init__(self, planner, reference=None):
        self.planner, self.reference, self.name = planner, reference, planner.name
        self.differs = []

    def plan(self, state, comps, comp_now):
        mine = self.planner.plan(state, comps, comp_now)
        if self.reference is not None and mine != self.reference.plan(state, comps, comp_now):
            self.differs.append(state["round"])
        return mine


def switch_for(kind: str, reference: str, current, seed: int):
    """What branch() should switch the hero seat to for policy `kind`."""
    from tfteval import make_policy
    from tfteval.branching import switch_planner
    from tfteval.planner import VARIANTS, ParamPlanner, PlanPolicy, make_plan_policy

    is_plan = kind in VARIANTS or kind == "llm"
    if is_plan and isinstance(current, PlanPolicy):
        if kind in VARIANTS:
            ref = ParamPlanner(reference, **VARIANTS[reference]) if reference in VARIANTS and reference != kind else None
            return switch_planner(TracingPlanner(ParamPlanner(kind, **VARIANTS[kind]), ref))
        return switch_planner(make_plan_policy(kind).planner)
    fresh = make_policy(f"hero={kind}")  # a different kind of seat: starts fresh mid-game
    fresh.reset(seed)
    return fresh


def run_job(job):
    """All missing branches of one seed: (seed, config, [(arm, k), ...]) -> rows."""
    from tfteval.branching import branch, play_to, snapshot

    seed, config, todo = job
    seat, rnd = hero_seat(seed), config["round"]
    t0 = time.time()
    game = play_to(lobby(seed, config["a"], config["opponents"]), seed, rnd, sim=config.get("sim"),
                   rules=config.get("rules"))
    base = {"config": config, "seed": seed, "hero_seat": seat, "round": rnd, "label": stages.label(rnd)}
    if game.done or game.round != rnd or game.player(seat) is None:
        return [{**base, "arm": arm, "k": k, "skipped": "hero out before the branch round"} for arm, k in todo]
    snap = snapshot(game)
    state = lobby_state(game, seat)
    hero = game.seat_policies[seat]
    executor = getattr(hero, "executor", None)
    stats0 = Counter(executor.stats) if executor is not None else Counter()
    state.update(steps=game.steps, comp_number=getattr(executor, "comp_number", None),
                 make_state_seconds=round(time.time() - t0, 2), snapshot_bytes=snap.size)
    rows = []
    for arm, k in todo:
        started = time.time()
        kind = config["a"] if arm == "A" else config["b"]
        g = branch(snap, reseed=k, switch={seat: switch_for(kind, config["a"], hero, seed)})
        log = g.seat_policies[seat] = ActionLog(g.seat_policies[seat])
        trajectory = []
        while not g.done and seat not in g.placements:  # the hero's place is final once it is out
            p = g.player(seat)
            if p is not None:
                trajectory.append({"round": g.round, **player_state(p)})
            g.run(until_round=g.round + 1)
        policy = log.policy
        planner = getattr(policy, "planner", None)
        ex = getattr(policy, "executor", None)
        stats = Counter(ex.stats) - stats0 if ex is not None else Counter()
        rows.append({**base, "arm": arm, "policy": kind, "k": k, "place": g.placements.get(seat),
                     "steps": g.steps, "state": state, "trajectory": trajectory, "actions": log.digests(),
                     "plan_differs": getattr(planner, "differs", None), "executor_stats": dict(stats),
                     "fallbacks": g.fallbacks.get(seat, 0), "branch_seconds": round(time.time() - started, 2)})
    return rows


# --------------------------------------------------------------------------- summary

def streak_bonus(win: int, loss: int) -> int:
    s = max(win, loss)
    return 0 if s < 2 else 1 if s <= 3 else 2 if s == 4 else 3


def window_stats(row: dict, first: int, last: int) -> dict | None:
    """PvP fights after planning phases first..last, from the round-start states (a draw counts as
    a loss). Fight gold: +1 per win plus the streak gold of the incomes these fights feed."""
    by = {t["round"]: t for t in row["trajectory"]}
    fights = [i for i in range(first, last + 1) if stages.is_pvp(i)]
    if not all(i in by and i + 1 in by for i in fights):
        return None  # out before the window ended (or the window starts before the branch)
    lost = sum(by[i + 1]["loss_streak"] > by[i]["loss_streak"] for i in fights)
    won = sum(by[i + 1]["win_streak"] > by[i]["win_streak"] for i in fights)
    stop = next((i for i in range(last + 1, last + 8) if stages.is_pvp(i)), last + 1)
    incomes = [i for i in range(first + 1, stop + 1) if i >= 5 and i in by]
    streak_gold = sum(streak_bonus(by[i]["win_streak"], by[i]["loss_streak"]) for i in incomes)
    return {"fights": len(fights), "lost": lost, "won": won, "hp_lost": by[first]["hp"] - by[last + 1]["hp"],
            "streak_gold": streak_gold, "fight_gold": streak_gold + won}


def at_round(row: dict, idx: int) -> dict | None:
    return next((t for t in row["trajectory"] if t["round"] == idx), None)


def corr(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float | None:
    def ranks(v):
        v = np.asarray(v, float)
        out = np.empty(len(v))
        out[v.argsort(kind="mergesort")] = np.arange(len(v))
        for value in np.unique(v):
            out[v == value] = out[v == value].mean()
        return out

    return corr(ranks(x), ranks(y)) if len(x) >= 3 else None


def r2(x) -> float | None:
    return None if x is None else round(x, 3)


def paired(states: dict, metric) -> dict:
    """Mean over states of mean_k(metric(B, k) - metric(A, k)); pairs where metric is None drop out."""
    per_state = []
    for arms in states.values():
        diffs = [metric(arms["B"][k]) - metric(arms["A"][k]) for k in sorted(set(arms["A"]) & set(arms["B"]))
                 if metric(arms["A"][k]) is not None and metric(arms["B"][k]) is not None]
        if diffs:
            per_state.append(float(np.mean(diffs)))
    return cluster_mean_ci(per_state) if per_state else {"n": 0}


def summarize(rows: list[dict], at: list[str], window: list[str] | None) -> dict:
    skipped = sorted({r["seed"] for r in rows if r.get("skipped")})
    rows = [r for r in rows if not r.get("skipped")]
    config = rows[0]["config"]
    rnd = config["round"]
    states = defaultdict(lambda: {"A": {}, "B": {}})
    for r in rows:
        states[r["seed"]][r["arm"]][r["k"]] = r
    states = {s: v for s, v in sorted(states.items()) if set(v["A"]) & set(v["B"])}
    out = {"config": config, "label": stages.label(rnd), "states": len(states), "branches": len(rows),
           "skipped_seeds": skipped}

    # main result: paired placement difference, the state as the unit
    per_state, crn_a, crn_b, changed_pairs, first_diff = [], [], [], [], []
    for seed, arms in states.items():
        ks = sorted(set(arms["A"]) & set(arms["B"]))
        pa = np.array([arms["A"][k]["place"] for k in ks], float)
        pb = np.array([arms["B"][k]["place"] for k in ks], float)
        changed = [arms["A"][k]["actions"] != arms["B"][k]["actions"] for k in ks]
        changed_pairs += changed
        for k in ks:
            da, db = arms["A"][k]["actions"], arms["B"][k]["actions"]
            diff_rounds = [int(x) for x in sorted(set(da) | set(db), key=int) if da.get(x) != db.get(x)]
            if diff_rounds:
                first_diff.append(diff_rounds[0] - rnd)
        crn_a += list(pa - pa.mean())
        crn_b += list(pb - pb.mean())
        st = arms["A"][ks[0]]["state"]
        per_state.append({"seed": seed, "k": len(ks), "diff": float(pb.mean() - pa.mean()),
                          "place_a": float(pa.mean()), "place_b": float(pb.mean()), "changed": any(changed),
                          "hp": st["hero"]["hp"], "gold": st["hero"]["gold"], "level": st["hero"]["level"],
                          "unit_value": st["hero"]["unit_value"], "board_strength": st["hero"]["board_strength"],
                          "hp_rank": st["hp_rank"], "strength_rank": st["strength_rank"]})
    d = np.array([s["diff"] for s in per_state])
    main = cluster_mean_ci(d)
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, size=len(d)).mean() for _ in range(5000)]
    main["bootstrap95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    main["place_a"] = float(np.mean([s["place_a"] for s in per_state]))
    main["place_b"] = float(np.mean([s["place_b"] for s in per_state]))
    changed = np.array([s["changed"] for s in per_state])
    main["share_of_states_changed"] = float(changed.mean())
    main["share_of_branch_pairs_changed"] = float(np.mean(changed_pairs))
    main["first_changed_round_after_branch"] = (
        {"median": float(np.median(first_diff)), "max": int(max(first_diff))} if first_diff else None)
    main["where_changed"] = cluster_mean_ci(d[changed]) if changed.any() else None
    main["crn_correlation"] = corr(crn_a, crn_b)  # same-k branches of A and B, within a state
    # does the effect differ between states? (permutation of the paired branch differences)
    e = [np.array([arms["B"][k]["place"] - arms["A"][k]["place"] for k in sorted(set(arms["A"]) & set(arms["B"]))], float)
         for arms in states.values()]
    sizes, pooled = [len(x) for x in e], np.concatenate(e)
    observed, hits = np.var([x.mean() for x in e]), 0
    for _ in range(2000):
        perm = rng.permutation(pooled)
        hits += np.var([c.mean() for c in np.split(perm, np.cumsum(sizes)[:-1])]) >= observed - 1e-12
    v_pair = float(np.mean([x.var(ddof=1) for x in e if len(x) > 1]))
    K = float(np.mean(sizes))
    main["paired_branch_sd"] = math.sqrt(v_pair)
    main["effect_sd_across_states"] = math.sqrt(max(0.0, float(d.var(ddof=1)) - v_pair / K))
    main["effect_heterogeneity_p"] = (hits + 1) / 2001
    out["placement_b_minus_a"] = main

    # does the effect depend on the state at the branch?
    out["diff_vs_state"] = {
        key: {"pearson": r2(corr([s[key] for s in per_state], d)), "spearman": r2(spearman([s[key] for s in per_state], d))}
        for key in ("hp", "hp_rank", "gold", "level", "unit_value", "board_strength", "strength_rank")}
    out["per_state"] = per_state

    # hero state at checkpoints (alive heroes only; paired differences over pairs where both are alive)
    checkpoints = {}
    for label in at:
        idx = stages.parse_label(label)
        if idx <= rnd:
            continue
        entry = {}
        for arm in ARMS:
            ts = [at_round(r, idx) for arms in states.values() for r in arms[arm].values()]
            alive = [t for t in ts if t is not None]
            entry[arm] = {"alive_share": len(alive) / len(ts), **{
                key: float(np.mean([t[key] for t in alive])) if alive else None
                for key in ("hp", "gold", "level", "unit_value", "board_units", "bench_units", "board_strength")}}
            entry[arm]["gold_plus_unit_value"] = (entry[arm]["gold"] + entry[arm]["unit_value"]) if alive else None
        entry["b_minus_a"] = {
            key: paired(states, lambda r, key=key, idx=idx: (at_round(r, idx) or {}).get(key))
            for key in ("hp", "gold", "level", "unit_value", "board_strength")}
        checkpoints[label] = entry
    out["checkpoints"] = checkpoints

    if window:
        first, last = (stages.parse_label(x) for x in window)
        entry = {"window": window}
        for arm in ARMS:
            ws = [window_stats(r, first, last) for arms in states.values() for r in arms[arm].values()]
            ws = [w for w in ws if w is not None]
            entry[arm] = {key: float(np.mean([w[key] for w in ws])) for key in ws[0]} if ws else {}
            entry[arm]["n"] = len(ws)
            if ws:
                entry[arm]["lost_histogram"] = np.bincount([w["lost"] for w in ws], minlength=ws[0]["fights"] + 1).tolist()
        entry["b_minus_a"] = {key: paired(states, lambda r, key=key: (window_stats(r, first, last) or {}).get(key))
                              for key in ("lost", "hp_lost", "fight_gold")}
        out["window"] = entry

    stats = {arm: Counter() for arm in ARMS}
    for arms in states.values():
        for arm in ARMS:
            for r in arms[arm].values():
                stats[arm].update(r["executor_stats"])
    n = {arm: sum(len(arms[arm]) for arms in states.values()) for arm in ARMS}
    out["executor_stats_per_branch"] = {arm: {k: round(v / n[arm], 2) for k, v in sorted(stats[arm].items())}
                                        for arm in ARMS}
    secs = [r["branch_seconds"] for r in rows]
    out["timing"] = {"branch_seconds_mean": float(np.mean(secs)), "branch_seconds_total": float(np.sum(secs)),
                     "make_state_seconds_mean": float(np.mean([r["state"]["make_state_seconds"] for r in rows]))}
    return out


def report(s: dict) -> str:
    m = s["placement_b_minus_a"]
    c = s["config"]
    ci = f"±{m['ci95']:.3f}" if m["ci95"] is not None else "(no interval: one state)"
    lines = [f"{c['b']} - {c['a']} from {s['label']}: {s['states']} states, {s['branches']} branches",
             f"  placement B - A: {m['mean']:+.3f} {ci} (t over states; bootstrap "
             f"[{m['bootstrap95'][0]:+.3f}, {m['bootstrap95'][1]:+.3f}]); A {m['place_a']:.2f}, B {m['place_b']:.2f}",
             f"  B changed a hero action in {m['share_of_states_changed']:.0%} of states, "
             f"{m['share_of_branch_pairs_changed']:.0%} of branch pairs; same-k correlation {m['crn_correlation']}",
             f"  effect SD across states {m['effect_sd_across_states']:.2f}, heterogeneity p={m['effect_heterogeneity_p']:.3f}"]
    for label, e in s["checkpoints"].items():
        lines.append(f"  {label}: " + "  ".join(
            f"{arm} hp {e[arm]['hp']:.1f} gold {e[arm]['gold']:.1f} lvl {e[arm]['level']:.2f}" for arm in ARMS
            if e[arm]["hp"] is not None))
    if "window" in s:
        w = s["window"]
        lines.append(f"  window {w['window']}: lost A {w['A'].get('lost', 0):.2f} B {w['B'].get('lost', 0):.2f}, "
                     f"hp lost A {w['A'].get('hp_lost', 0):.1f} B {w['B'].get('hp_lost', 0):.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="hero policy before the branch and in arm A (make_policy kind)")
    parser.add_argument("--b", required=True, help="hero policy of arm B")
    parser.add_argument("--opponents", default="rule:7", help="the other 7 seats, e.g. rule:7 or rule:4,noisy20:3")
    parser.add_argument("--seeds", required=True, help="first:count, first-last or a comma list")
    parser.add_argument("--round", required=True, help="branch at this planning phase: a stage label (2-1) or index")
    parser.add_argument("--branches", type=int, default=5, help="re-seeded branches per state and policy")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--at", nargs="*", default=["3-1", "4-1", "5-1"], help="checkpoints for the hero state")
    parser.add_argument("--window", nargs=2, help="first and last planning phase of a fight window, e.g. 2-1 2-6")
    parser.add_argument("--sim", help="simulator profile: realistic (default) or default, optionally with overrides "
                                      "such as realistic,rng_streams=shared (or TFT_SIM)")
    parser.add_argument("--rules", choices=["set4", "set18"], help="economy profile (default set4, or TFT_RULES)")
    parser.add_argument("--out", required=True, help="raw JSONL, appended to")
    parser.add_argument("--summary", help="summary JSON (default: --out with .summary.json)")
    parser.add_argument("--summarize", action="store_true", help="only summarise --out")
    args = parser.parse_args()

    out = Path(args.out)
    summary_path = Path(args.summary) if args.summary else out.with_suffix(".summary.json")
    opponents = parse_seats(args.opponents)
    if len(opponents) != 7:
        raise SystemExit(f"--opponents needs 7 seats, got {len(opponents)}")
    from tfteval.runner import sim_options

    sim, _ = sim_options(args.sim)  # default: TFT_SIM, else realistic
    rules = args.rules or os.environ.get("TFT_RULES", "set4")
    os.environ["TFT_SIM"], os.environ["TFT_RULES"] = sim, rules  # read by the runner in every worker
    config = {"a": args.a, "b": args.b, "round": stages.parse_label(args.round), "opponents": opponents,
              "sim": sim, "rules": rules}
    if args.a == args.b:
        print("A and B are the same policy: an A/A check, every difference is noise", flush=True)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    if not args.summarize:
        if rows and any({"sim": "default", "rules": "set4", **r["config"]} != config for r in rows):  # older rows
            raise SystemExit(f"{out} holds rows of another configuration: {rows[0]['config']}")
        done = {(r["seed"], r["arm"], r["k"]) for r in rows}
        jobs = []
        for seed in parse_seeds(args.seeds):
            todo = [(arm, k) for k in range(args.branches) for arm in ARMS if (seed, arm, k) not in done]
            if todo:
                jobs.append((seed, config, todo))
        print(f"{len(jobs)} states, {sum(len(j[2]) for j in jobs)} branches to run, {args.workers} workers", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
            pending = set()
            while jobs or pending:
                while jobs and len(pending) < args.workers:
                    pending.add(pool.submit(run_job, jobs.pop(0)))
                finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                for fut in finished:
                    new = fut.result()
                    for row in new:
                        fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows += new
                    by_arm = {arm: [r.get("place") for r in new if r["arm"] == arm] for arm in ARMS}
                    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} seed {new[0]['seed']}: A {by_arm['A']} "
                          f"B {by_arm['B']} ({sum(r.get('branch_seconds', 0) for r in new):.0f}s; "
                          f"{len(jobs) + len(pending)} states left, {time.time() - started:.0f}s so far)", flush=True)
    summary = summarize(rows, args.at, args.window)
    summary_path.write_text(json.dumps(summary, indent=1))
    print(report(summary))


if __name__ == "__main__":
    main()

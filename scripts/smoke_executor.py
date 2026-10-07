"""Trace one seat round by round and compare it with a control seat on the same seeds.

    # fodder board from 2-1 to 2-6 against plain mimic
    python scripts/smoke_executor.py --hero fodder2 --games 32 --seed 7300 --window 2-1 2-6

    # one capability at a time: mimic plus plan fields inside windows of rounds
    python scripts/smoke_executor.py --hero mimic --name lvl8 --games 8 \
        --overlay '[{"from": "3-5", "fields": {"level_by": {"level": 8, "by": "4-1"}}}]'

Each seed is played twice: once with the hero policy on the rotating hero seat and once with the
control policy there; the other seven seats are rule bots. Reported per policy (means over games):

* for the PvP fights in --window: fights lost, HP lost, and the gold the fights paid: win gold
  (+1 per win) plus streak gold in the incomes they feed (the income at the start of a round with
  idx >= 5 adds a bonus for the streak standing after the previous fight; PvE leaves streaks alone);
* HP, gold, level, share of games at level >= 8 and share of fielded units that belong to the
  executor's target comp, at the start of each --at round;
* the executor's counters (fodder moves and buys, pivots, xp taken from the rule bot, ...);
* final placement (noisy at these sample sizes).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import stages  # noqa: E402


def streak_bonus(win: int, loss: int) -> int:
    s = max(win, loss)
    return 0 if s < 2 else 1 if s <= 3 else 2 if s == 4 else 3


class Tracer:
    """Wraps a policy; records the seat's state at the start of every round."""

    def __init__(self, policy):
        self.policy, self.name, self.rows, self.round = policy, policy.name, [], None

    def reset(self, seed):
        self.policy.reset(seed)

    def act(self, observation, info, agent, env):
        player, idx = info["player"], info.get("game_round", 1)
        if idx != self.round:
            self.round = idx
            from tfteval.public import board_units

            executor = getattr(self.policy, "executor", None)
            comp = None
            if executor is not None and executor.comp_number >= 0:
                comp = self.policy.traits[executor.comp_number]
            board = [u.name for u in board_units(player)]
            on_comp = None
            if comp and board:
                on_comp = sum(name in self.policy.comps[comp] for name in board) / len(board)
            self.rows.append({
                "idx": idx, "hp": float(player.health), "gold": int(player.gold), "level": int(player.level),
                "win": int(player.win_streak), "loss": int(player.loss_streak), "comp": comp, "on_comp": on_comp,
                "board": sorted(f"{u.name}*{u.stars}" for u in board_units(player)),
                "bench": sorted(f"{c.name}*{c.stars}" for c in player.bench if c),
            })
        return self.policy.act(observation, info, agent, env)


def window_stats(rows: list, first: int, last: int) -> dict:
    """Fights first..last (PvP only). Each row is the state at the start of planning phase row['idx']."""
    by_idx = {r["idx"]: r for r in rows}
    fights = [i for i in range(first, last + 1) if stages.is_pvp(i) and i in by_idx and i + 1 in by_idx]
    lost = won = 0
    for i in fights:
        before, after = by_idx[i], by_idx[i + 1]
        if after["loss"] > before["loss"]:  # a draw counts as a loss for both
            lost += 1
        elif after["win"] > before["win"]:
            won += 1
    hp_lost = by_idx[first]["hp"] - by_idx[last + 1]["hp"] if first in by_idx and last + 1 in by_idx else None
    stop = next((i for i in range(last + 1, last + 8) if stages.is_pvp(i)), last + 1)
    incomes = [i for i in range(first + 1, stop + 1) if i >= 5 and i in by_idx]
    streak_gold = sum(streak_bonus(by_idx[i]["win"], by_idx[i]["loss"]) for i in incomes)
    return {"fights": len(fights), "lost": lost, "won": won, "hp_lost": hp_lost, "streak_gold": streak_gold,
            "win_gold": won, "gold_from_fights": streak_gold + won}


def make_hero(spec: dict):
    from tfteval import make_policy
    from tfteval.planner import VARIANTS, OverlayPlanner, ParamPlanner, PlanPolicy

    if not spec.get("overlay"):
        return make_policy(spec["policy"])
    base = ParamPlanner(spec["policy"], **VARIANTS[spec["policy"]])
    return PlanPolicy(OverlayPlanner(base, spec["overlay"], spec["name"]), name=spec["name"])


def play(job):
    from tfteval import make_policy, play_game

    seed, spec = job
    shift = seed % 8
    hero_seat = f"player_{(8 - shift) % 8}"
    lobby = {f"player_{i}": make_policy("rule") for i in range(8)}
    tracer = lobby[hero_seat] = Tracer(make_hero(spec))
    result = play_game(lobby, seed)
    executor = getattr(tracer.policy, "executor", None)
    return {"seed": seed, "policy": spec["name"], "seat": hero_seat, "place": result.placements[hero_seat],
            "rows": tracer.rows, "executor": dict(executor.stats) if executor is not None else {}}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def summarise(games, first, last, checkpoints):
    windows = [window_stats(g["rows"], first, last) for g in games]
    out = {"n": len(games), "place": _mean([g["place"] for g in games])}
    out["window"] = {k: _mean([w[k] for w in windows]) for k in windows[0]} if windows else {}
    at = {}
    for label in checkpoints:
        idx = stages.parse_label(label)
        rows = [next((r for r in g["rows"] if r["idx"] == idx), None) for g in games]
        alive = [r for r in rows if r is not None]
        at[label] = {"alive": len(alive), "hp": _mean([r["hp"] for r in alive]),
                     "gold": _mean([r["gold"] for r in alive]), "level": _mean([r["level"] for r in alive]),
                     "lv8": _mean([float(r["level"] >= 8) for r in alive]),
                     "on_comp": _mean([r.get("on_comp") for r in alive])}
    out["at_start_of"] = at
    totals = Counter()
    for g in games:
        totals.update(g["executor"])
    out["executor_per_game"] = {k: round(v / len(games), 2) for k, v in sorted(totals.items())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hero", required=True, help="a policy name; with --overlay a VARIANTS name")
    parser.add_argument("--overlay", help="JSON list of {from, to, fields} windows added to the hero's plans")
    parser.add_argument("--name", help="label for the hero (default: --hero)")
    parser.add_argument("--control", default="mimic")
    parser.add_argument("--control-overlay", help="like --overlay, for the control seat")
    parser.add_argument("--control-name", help="label for the control (default: --control)")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--window", nargs=2, default=["2-1", "2-6"])
    parser.add_argument("--at", nargs="*", default=["3-1", "4-1", "4-2", "4-3", "5-1"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out")
    args = parser.parse_args()

    hero = {"policy": args.hero, "name": args.name or args.hero,
            "overlay": json.loads(args.overlay) if args.overlay else None}
    control = {"policy": args.control, "name": args.control_name or args.control,
               "overlay": json.loads(args.control_overlay) if args.control_overlay else None}
    if hero["name"] == control["name"]:
        raise SystemExit("hero and control need different names (--name)")
    first, last = (stages.parse_label(x) for x in args.window)
    jobs = [(args.seed + i, spec) for i in range(args.games) for spec in (hero, control)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        games = list(pool.map(play, jobs))
    report = {"hero": hero, "control": control, "window": args.window,
              "seeds": [args.seed, args.seed + args.games - 1]}
    for spec in (hero, control):
        report[spec["name"]] = summarise([g for g in games if g["policy"] == spec["name"]], first, last, args.at)
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"report": report, "games": games}))


if __name__ == "__main__":
    main()

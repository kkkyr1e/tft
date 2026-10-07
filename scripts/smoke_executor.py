"""Trace one seat round by round and summarise a window of rounds against a control seat.

    python scripts/smoke_executor.py --hero fodder2 --control mimic --games 16 --seed 7300 \
        --window 2-1 2-6 --out results/smoke_fodder2.json

Each game is played twice on the same seed: once with the hero policy on the rotating hero seat and
once with the control policy there; the other seven seats are rule bots. For the PvP fights in the
window (inclusive) it reports fights lost, HP lost, and the gold the fights paid: win gold (+1 per
win) and streak gold in the incomes they feed (the income at the start of round idx >= 5 adds a
streak bonus for the streak standing after the previous fight; PvE rounds do not touch streaks).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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
    """Wraps a policy; records the seat's public and private scalars at the start of every round."""

    def __init__(self, policy):
        self.policy, self.name, self.rows, self.round = policy, policy.name, [], None

    def reset(self, seed):
        self.policy.reset(seed)

    def act(self, observation, info, agent, env):
        player, idx = info["player"], info.get("game_round", 1)
        if idx != self.round:
            self.round = idx
            from tfteval.public import board_units
            self.rows.append({
                "idx": idx, "hp": float(player.health), "gold": int(player.gold), "level": int(player.level),
                "win": int(player.win_streak), "loss": int(player.loss_streak),
                "board": sorted(f"{u.name}*{u.stars}" for u in board_units(player)),
                "bench": sorted(f"{c.name}*{c.stars}" for c in player.bench if c),
            })
        return self.policy.act(observation, info, agent, env)


def window_stats(rows: list, first: int, last: int) -> dict:
    """Fights first..last (PvP only). rows[i] is the state at the start of planning phase rows[i]['idx']."""
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
    # incomes fed by these fights: start of every round after a window fight, through the first PvP after it
    stop = next((i for i in range(last + 1, last + 8) if stages.is_pvp(i)), last + 1)
    incomes = [i for i in range(first + 1, stop + 1) if i >= 5 and i in by_idx]
    streak_gold = sum(streak_bonus(by_idx[i]["win"], by_idx[i]["loss"]) for i in incomes)
    return {"fights": len(fights), "lost": lost, "won": won, "hp_lost": hp_lost, "streak_gold": streak_gold,
            "win_gold": won, "gold_from_fights": streak_gold + won}


def play(job):
    from tfteval import make_policy, play_game

    seed, hero, n_rule = job
    seats = [hero] + ["rule"] * n_rule
    shift = seed % 8
    order = seats[shift:] + seats[:shift]
    lobby, tracer = {}, None
    for i, name in enumerate(order):
        policy = make_policy(name)
        if i == (8 - shift) % 8 and tracer is None:
            policy = tracer = Tracer(policy)
        lobby[f"player_{i}"] = policy
    hero_seat = f"player_{(8 - shift) % 8}"
    result = play_game(lobby, seed)
    stats = dict(tracer.policy.executor.stats) if hasattr(tracer.policy, "executor") else {}
    return {"seed": seed, "policy": hero, "seat": hero_seat, "place": result.placements[hero_seat],
            "rows": tracer.rows, "executor": stats}


def summarise(games, first, last):
    rows = [dict(window_stats(g["rows"], first, last), place=g["place"]) for g in games]
    keys = ("fights", "lost", "won", "hp_lost", "streak_gold", "win_gold", "gold_from_fights", "place")
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = round(statistics.mean(vals), 2) if vals else None
    hp_at = {}
    for label in ("3-1", "4-1", "5-1"):
        idx = stages.parse_label(label)
        vals = [next((r["hp"] for r in g["rows"] if r["idx"] == idx), 0.0) for g in games]
        hp_at[label] = round(statistics.mean(vals), 1)
    out["hp_at_start_of"] = hp_at
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hero", required=True)
    parser.add_argument("--control", default="mimic")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--window", nargs=2, default=["2-1", "2-6"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out")
    args = parser.parse_args()

    first, last = (stages.parse_label(x) for x in args.window)
    jobs = [(args.seed + i, p, 7) for i in range(args.games) for p in (args.hero, args.control)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        games = list(pool.map(play, jobs))
    report = {"window": args.window, "seeds": [args.seed, args.seed + args.games - 1]}
    for name in (args.hero, args.control):
        report[name] = summarise([g for g in games if g["policy"] == name], first, last)
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"report": report, "games": games}))


if __name__ == "__main__":
    main()

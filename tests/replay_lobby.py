"""Play a few fixed games and print a fingerprint of every seat's actions (used by test_regression.py).

    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 --sim realistic \
        > tests/data/executor_reference_realistic.json
    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 --sim default --no-pickers \
        > tests/data/executor_reference.json

The fingerprint is the placements plus a hash of the exact action sequence of each seat, so any
change in what a seat does (not only in where it finishes) shows up. `--sim` is the simulator profile
(tfteval.runner.SIM_PROFILES) and `--no-pickers` leaves every carousel pick to the simulator's default
(the plan seats' own picker is on otherwise); both are recorded in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOBBY = ["mimic", "fast8", "rolldown8", "hp50", "fast8roll", "rule", "rule", "rule"]


class Recording:
    def __init__(self, policy):
        self.policy, self.name, self.trace = policy, policy.name, hashlib.sha256()

    def reset(self, seed):
        self.policy.reset(seed)

    def carousel_picker(self):
        make = getattr(self.policy, "carousel_picker", None)
        return make() if callable(make) else None

    def act(self, observation, info, agent, env):
        action = self.policy.act(observation, info, agent, env)
        self.trace.update(repr(list(map(int, action))).encode())
        return action


def play(job) -> dict:
    from tfteval import make_policy, play_game

    seed, sim, pickers = job
    seats = {f"player_{i}": Recording(make_policy(name)) for i, name in enumerate(LOBBY)}
    result = play_game(seats, seed, sim=sim, pickers=pickers)
    return {"seed": seed, "lobby": result.lobby, "placements": dict(sorted(result.placements.items())),
            "actions": result.actions, "sim_fixes": result.sim_fixes, "sim_commit": result.sim_commit,
            "rules": result.rules, "sim": result.sim, "sim_options": result.sim_options,
            "carousel_pickers": result.carousel_pickers, "seconds": result.seconds,
            "digest": {seat: p.trace.hexdigest()[:16] for seat, p in seats.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sim", required=True, help="simulator profile, e.g. realistic or default")
    parser.add_argument("--no-pickers", action="store_true", help="every seat takes the default carousel pick")
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        games = list(pool.map(play, [(seed, args.sim, not args.no_pickers) for seed in args.seeds]))
    print(json.dumps({"lobby": LOBBY, "sim": args.sim, "pickers": not args.no_pickers, "games": games}, indent=1))


if __name__ == "__main__":
    main()

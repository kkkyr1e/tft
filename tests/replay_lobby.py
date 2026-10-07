"""Play a few fixed games and print a fingerprint of every seat's actions (used by test_regression.py).

    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 > tests/data/executor_reference.json

The fingerprint is the placements plus a hash of the exact action sequence of each seat, so any
change in what a seat does (not only in where it finishes) shows up.
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

    def act(self, observation, info, agent, env):
        action = self.policy.act(observation, info, agent, env)
        self.trace.update(repr(list(map(int, action))).encode())
        return action


def play(seed: int) -> dict:
    from tfteval import make_policy, play_game

    seats = {f"player_{i}": Recording(make_policy(name)) for i, name in enumerate(LOBBY)}
    result = play_game(seats, seed)
    return {"seed": seed, "lobby": result.lobby, "placements": dict(sorted(result.placements.items())),
            "actions": result.actions, "sim_fixes": result.sim_fixes, "sim_commit": result.sim_commit,
            "digest": {seat: p.trace.hexdigest()[:16] for seat, p in seats.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        games = list(pool.map(play, args.seeds))
    print(json.dumps({"lobby": LOBBY, "games": games}, indent=1))


if __name__ == "__main__":
    main()

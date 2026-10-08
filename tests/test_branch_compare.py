"""scripts/branch_compare.py: the summary on hand-made rows, and an A/A run on the simulator."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfteval.stats import cluster_mean_ci  # noqa: E402

SCRIPT = ROOT / "scripts/branch_compare.py"
CONFIG = {"a": "mimic", "b": "fodder2", "round": 3, "opponents": ["rule"] * 7}


def run(*args, timeout=60):
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    out = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=ROOT,
                         timeout=timeout)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def trajectory(results):
    """Round-start states for rounds 3..9 after the 2-1..2-6 fights end in `results` ('W'/'L')."""
    rows, hp, win, loss = [], 100.0, 0, 0
    for i, rnd in enumerate(range(3, 10)):
        rows.append({"round": rnd, "hp": hp, "gold": 10 + i, "level": 4, "xp": 0, "win_streak": win,
                     "loss_streak": loss, "board_units": 4, "bench_units": 2, "unit_value": 8, "board_strength": 8})
        if i < len(results):
            if results[i] == "L":
                hp, win, loss = hp - 5, 0, loss + 1
            else:
                win, loss = win + 1, 0
    return rows


def row(seed, arm, k, place, actions, results):
    state = {"hero": {"hp": 100.0, "gold": 10 + seed, "level": 4, "unit_value": 8, "board_strength": seed},
             "hp_rank": 4.5, "strength_rank": float(seed), "make_state_seconds": 1.0}
    return {"config": CONFIG, "seed": seed, "hero_seat": "player_0", "round": 3, "label": "2-1", "arm": arm,
            "policy": CONFIG["a" if arm == "A" else "b"], "k": k, "place": place, "steps": 100, "state": state,
            "trajectory": trajectory(results), "actions": actions, "plan_differs": [], "executor_stats": {},
            "fallbacks": 0, "branch_seconds": 2.0}


def test_summary_pairs_branches_within_states(tmp_path):
    same = {"3": "aaa"}
    rows = []
    places = {1: ([4, 6], [3, 5]), 2: ([2, 2], [2, 2]), 3: ([5, 7], [3, 3]), 4: ([8, 1], [6, 2])}
    for seed, (pa, pb) in places.items():
        for k in range(2):
            rows.append(row(seed, "A", k, pa[k], same, "WWLWL"))
            rows.append(row(seed, "B", k, pb[k], same if seed == 2 else {"3": "bbb"}, "LLLLL"))
    rows.append({**row(5, "A", 0, None, same, ""), "skipped": "hero out before the branch round"})
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run("--a", "mimic", "--b", "fodder2", "--round", "2-1", "--seeds", "1:4", "--out", str(raw), "--summarize",
        "--window", "2-1", "2-6", "--at", "3-1")
    s = json.loads((tmp_path / "raw.summary.json").read_text())
    m = s["placement_b_minus_a"]
    d = [-1.0, 0.0, -3.0, -0.5]  # per state: mean over k of B - A
    assert s["states"] == 4 and [p["diff"] for p in s["per_state"]] == d
    assert m["mean"] == pytest.approx(np.mean(d)) and m["ci95"] == pytest.approx(cluster_mean_ci(d)["ci95"])
    assert m["share_of_states_changed"] == 0.75 and m["share_of_branch_pairs_changed"] == 0.75
    assert m["place_a"] == pytest.approx(35 / 8) and m["place_b"] == pytest.approx(26 / 8)
    w = s["window"]
    assert w["A"]["lost"] == 2 and w["B"]["lost"] == 5 and w["B"]["hp_lost"] == 25
    assert w["b_minus_a"]["lost"]["mean"] == 3
    assert w["B"]["streak_gold"] == 1 + 1 + 2 + 3 + 3  # loss streaks 2, 3, 4, 5, 5 at 2-3, 2-5, 2-6, 2-7, 3-1
    assert s["skipped_seeds"] == [5]
    assert s["checkpoints"]["3-1"]["A"]["gold"] == 16
    assert s["diff_vs_state"]["board_strength"]["spearman"] is not None


def test_identical_policies_give_identical_branches(tmp_path):
    pytest.importorskip("Simulator")
    raw = tmp_path / "aa.jsonl"
    args = ["--a", "mimic", "--b", "mimic", "--round", "5-1", "--seeds", "7100:1", "--branches", "2",
            "--workers", "1", "--out", str(raw)]
    run(*args, timeout=600)
    s = json.loads((tmp_path / "aa.summary.json").read_text())
    m = s["placement_b_minus_a"]
    assert s["branches"] == 4 and m["mean"] == 0 and m["share_of_branch_pairs_changed"] == 0
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len({json.dumps(r["actions"]) for r in rows}) == 2  # the two re-seeds differ, the arms do not
    assert "0 states, 0 branches to run" in run(*args, timeout=600)  # resumed: nothing left to do

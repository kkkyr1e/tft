"""mimic and the economy variants play exactly as when the references were recorded.

Each reference in tests/data/ holds the placements and a hash of every seat's full action sequence
for three seeds, under a pinned simulator profile (tfteval.runner.SIM_PROFILES) and carousel-picker
setting:

    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 --workers 2 --sim default \\
        --no-pickers > tests/data/executor_reference.json
    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 --workers 2 --sim realistic15 \\
        > tests/data/executor_reference_realistic.json
    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 --workers 2 --sim realistic \\
        > tests/data/executor_reference_realistic60.json

executor_reference.json was first generated on commit 57ef4d7, before tfteval/stages.py,
tfteval/public.py and the optional plan fields existed, on the upstream simulator; the
stance-executor changes replayed it exactly. It was regenerated once, when the project moved to the
simulator fork (README, "对模拟器的修正"), because the simulator changed every game. It was recorded
before the fork had options, so it is the `default` profile (every option off) with every carousel
pick left to the simulator; the fork's develop at 2ba01d5, the simulator profiles, the plan seats'
carousel picker (off here) and the keyed-stream generator of the executor (unused with shared
streams) replay it unchanged.

executor_reference_realistic.json is the `realistic` profile as it was until 2026-10-09 (15 actions
per planning phase, now `realistic15`), with the plan seats' carousel picker. Recorded on 2ba01d5
when the profile was introduced. executor_reference_realistic60.json is the same lobby as games run
now: `realistic` with 60 actions per planning phase, recorded on ff4db16 when the budget was raised.

Each game records the simulator commit it was played on (sim_commit). Replaying the same seeds must
give the same numbers: executor code paths for the optional plan fields may only act when a plan
uses them.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("Simulator")

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "executor_reference.json": ["--sim", "default", "--no-pickers"],
    "executor_reference_realistic.json": ["--sim", "realistic15"],
    "executor_reference_realistic60.json": ["--sim", "realistic"],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_mimic_and_variants_replay_the_reference(name):
    reference = json.loads((ROOT / "tests/data" / name).read_text())
    seeds = [str(g["seed"]) for g in reference["games"]]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    env.pop("TFT_SIM", None)  # the case pins its profile
    env.pop("TFT_RULES", None)
    workers = os.environ.get("TFT_TEST_WORKERS", "2")  # 1 on a machine that is busy with other runs
    out = subprocess.run([sys.executable, str(ROOT / "tests/replay_lobby.py"), "--seeds", *seeds,
                          "--workers", workers, *CASES[name]],
                         capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
    now = json.loads(out.stdout)
    assert now["lobby"] == reference["lobby"]
    for old, new in zip(reference["games"], now["games"]):
        where = (old["seed"], "reference simulator", old.get("sim_commit"), "now", new.get("sim_commit"))
        assert new["placements"] == old["placements"], where
        assert new["digest"] == old["digest"], where
        if "sim_options" in old:
            assert new["sim_options"] == old["sim_options"], where

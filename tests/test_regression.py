"""mimic and the economy variants play exactly as when the reference was recorded.

tests/data/executor_reference.json holds the placements and a hash of every seat's full action
sequence for three seeds:

    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 --workers 2 \
        > tests/data/executor_reference.json

It was first generated on commit 57ef4d7, before tfteval/stages.py, tfteval/public.py and the
optional plan fields existed, on the upstream simulator; the stance-executor changes replayed it
exactly. It was regenerated once, when the project moved to the simulator fork (README,
"对模拟器的修正"), because the simulator changed every game; each game records the simulator commit
it was played on (sim_commit). Replaying the same seeds must give the same numbers: executor code
paths for the optional plan fields may only act when a plan uses them.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("Simulator")

ROOT = Path(__file__).resolve().parents[1]


def test_mimic_and_variants_replay_the_reference():
    reference = json.loads((ROOT / "tests/data/executor_reference.json").read_text())
    seeds = [str(g["seed"]) for g in reference["games"]]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    workers = os.environ.get("TFT_TEST_WORKERS", "2")  # 1 on a machine that is busy with other runs
    out = subprocess.run([sys.executable, str(ROOT / "tests/replay_lobby.py"), "--seeds", *seeds, "--workers", workers],
                         capture_output=True, text=True, env=env, cwd=ROOT, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    now = json.loads(out.stdout)
    assert now["lobby"] == reference["lobby"]
    for old, new in zip(reference["games"], now["games"]):
        where = (old["seed"], "reference simulator", old.get("sim_commit"), "now", new.get("sim_commit"))
        assert new["placements"] == old["placements"], where
        assert new["digest"] == old["digest"], where

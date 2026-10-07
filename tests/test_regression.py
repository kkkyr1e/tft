"""mimic and the economy variants play exactly as before the stance-executor changes.

tests/data/executor_reference.json was generated on commit 57ef4d7, before tfteval/stages.py,
tfteval/public.py and the new plan fields existed:

    PYTHONHASHSEED=0 python tests/replay_lobby.py --seeds 7100 7101 7102 --workers 2

It holds the placements and a hash of every seat's full action sequence. Replaying the same seeds
now must give the same numbers: the new code paths may only act when a plan uses the new fields.
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
    out = subprocess.run([sys.executable, str(ROOT / "tests/replay_lobby.py"), "--seeds", *seeds, "--workers", "2"],
                         capture_output=True, text=True, env=env, cwd=ROOT, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    now = json.loads(out.stdout)
    assert now["lobby"] == reference["lobby"]
    for old, new in zip(reference["games"], now["games"]):
        assert new["placements"] == old["placements"], old["seed"]
        assert new["digest"] == old["digest"], old["seed"]

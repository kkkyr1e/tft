"""Stand-in model for testing the LLM planner path without a model: reads the prompt on stdin
and answers with the rule bot's own economy (MimicPlanner), wrapped in chatter like a real model.

    TFT_LLM_CMD="python scripts/mock_llm.py" python scripts/run_lobby.py --lobby llm:1,rule:7 ...
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tfteval.planner import MimicPlanner  # noqa: E402

prompt = sys.stdin.read()
state = json.loads(prompt.split("State (JSON):\n", 1)[1].split("\n\nAnswer with", 1)[0])
plan = MimicPlanner().plan(state, {}, None)
print("Sure, here is my plan:\n" + json.dumps({**plan, "why": "mock"}))

#!/bin/bash
# Benchmark v2 (60 actions), set4 dev pool, one agent, recorded; $1 agent, $2 workers. Resumable.
cd /home/claude/tft-v2
export PYTHONPATH=$PWD/third_party/TFTMuZeroAgent:$PWD
OUT=/tmp/claude-0/-home-claude/c10acc4e-9037-548a-8e3c-11ca43d76ee0/scratchpad/bench2
mkdir -p $OUT
echo "== $1 $(date -u +%H:%M:%S)" >> $OUT/bench.log
nice -n 19 timeout 6900 python scripts/benchmark.py --agent "$1" --track set4 --record --workers $2 \
  --out $OUT/v2_set4_dev_$1.json >> $OUT/bench.log 2>&1
echo "== $1 exit $? $(date -u +%H:%M:%S)" >> $OUT/bench.log
tail -4 $OUT/bench.log

#!/bin/bash
# Benchmark v2 (60 actions), set4 dev pool, one agent, recorded; $1 agent, $2 workers. Resumable.
# Writes $OUT/v2_set4_dev_<agent>.json and appends to $OUT/bench.log; OUT defaults to this folder
# (relative paths are from the repo root).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD/third_party/TFTMuZeroAgent:$PWD
OUT=${OUT:-results/bench}
mkdir -p "$OUT"
echo "== $1 $(date -u +%H:%M:%S)" >> "$OUT/bench.log"
nice -n 19 timeout 6900 python scripts/benchmark.py --agent "$1" --track set4 --record --workers "$2" \
  --out "$OUT/v2_set4_dev_$1.json" >> "$OUT/bench.log" 2>&1
echo "== $1 exit $? $(date -u +%H:%M:%S)" >> "$OUT/bench.log"
tail -4 "$OUT/bench.log"

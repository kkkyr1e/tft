#!/bin/bash
# Benchmark v2 (60 actions), set18 track (Set 18.5 economy, Set 4 content), dev pool, recorded.
# Runs stance, stance+hold, mimic in turn; $1 workers. Resumable: rerun to continue.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD/third_party/TFTMuZeroAgent:$PWD
OUT=${OUT:-results/bench}
mkdir -p "$OUT"
for a in stance stance+hold mimic; do
  echo "== $a $(date -u +%H:%M:%S)" >> "$OUT/bench_set18.log"
  nice -n 19 python scripts/benchmark.py --agent "$a" --track set18 --record --workers "$1" \
    --out "$OUT/v2_set18_dev_$a.json" >> "$OUT/bench_set18.log" 2>&1
  echo "== $a exit $? $(date -u +%H:%M:%S)" >> "$OUT/bench_set18.log"
done

#!/bin/bash
# A contrastive-pair pilot: $1 name, $2 workers, then build_pairs.py arguments. Resumable (pairs in the out file are skipped).
# Writes $OUT/<name>.jsonl and $OUT/<name>.log; OUT defaults to this folder (relative paths are from the repo root).
# SIM (default realistic, 60 actions) is the simulator profile; the pilots built before 2026-10-09 are realistic15.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=$PWD/third_party/TFTMuZeroAgent:$PWD
OUT=${OUT:-results/bank2/pairs}
SIM=${SIM:-realistic}
name=$1; workers=$2; shift 2
mkdir -p "$OUT"
echo "== $name $(date -u +%H:%M:%S)" >> "$OUT/$name.log"
nice -n 19 timeout 6900 python scripts/build_pairs.py --sim "$SIM" --rules set4 --workers "$workers" --fresh-workers \
  --out "$OUT/$name.jsonl" "$@" >> "$OUT/$name.log" 2>&1
echo "== $name exit $? $(date -u +%H:%M:%S)" >> "$OUT/$name.log"
tail -12 "$OUT/$name.log"

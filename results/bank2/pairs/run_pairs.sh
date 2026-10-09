#!/bin/bash
# A contrastive-pair pilot: $1 name, $2 workers, then build_pairs.py arguments. Resumable (pairs in the out file are skipped).
cd /home/claude/tft-v2
export PYTHONPATH=$PWD/third_party/TFTMuZeroAgent:$PWD
OUT=/tmp/claude-0/-home-claude/c10acc4e-9037-548a-8e3c-11ca43d76ee0/scratchpad/pairs
name=$1; workers=$2; shift 2
echo "== $name $(date -u +%H:%M:%S)" >> $OUT/$name.log
nice -n 19 timeout 6900 python scripts/build_pairs.py --sim realistic --rules set4 --workers $workers --fresh-workers \
  --out $OUT/$name.jsonl "$@" >> $OUT/$name.log 2>&1
echo "== $name exit $? $(date -u +%H:%M:%S)" >> $OUT/$name.log
tail -12 $OUT/$name.log

"""Compare two runs that used the same seeds.

    python scripts/compare.py results/hero_rule.json results/hero_noisy20.json --policy hero
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import paired_diff, summarize  # noqa: E402
from tfteval.stats import games_needed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--policy", required=True, help="name of the seat under test in both runs")
    parser.add_argument("--gap", type=float, default=0.3, help="placement gap you want to be able to detect")
    args = parser.parse_args()

    a, b = json.loads(Path(args.run_a).read_text()), json.loads(Path(args.run_b).read_text())
    for label, run in (("A", a), ("B", b)):
        s = summarize(run, args.policy)
        print(f"{label}: avg place {s['mean']:.2f} ±{s['ci95']:.2f}  top4 {s['top4_rate']:.0%}  n={s['n']} games")
    d = paired_diff(a, b, args.policy, args.policy)
    corr = "n/a" if d["correlation"] is None else f"{d['correlation']:.2f}"
    print(f"A - B over {d['n']} shared seeds: {d['mean']:+.2f} ±{d['ci95']:.2f} (unpaired ±{d['unpaired_ci95']:.2f})")
    print(f"correlation between the two runs on the same seed: {corr}")
    sd_diff = d["ci95"] / 1.959964 * d["n"] ** 0.5
    print(f"games per version to pin the gap to ±{args.gap}: {games_needed(sd_diff, args.gap)}")


if __name__ == "__main__":
    main()

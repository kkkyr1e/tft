"""Held-out check of a decision bank v2: can anything beat the best fixed candidate? (no simulation)

    python scripts/heldout_bank2.py results/bank2/bank.jsonl
    python scripts/heldout_bank2.py results/bank2/bank.jsonl --choices results/bank2/bank.sanity.json

Every policy is scored on the same branches, by the place of its choice relative to the item's mean over
its candidates (lower is better; a uniformly random choice scores 0), and nothing is scored on branches
it was chosen with (tfteval.bank2.heldout_check):

* oracle: picks the best candidate on 3 of the item's 4 branch folds, scored on the 4th, rotated;
* always:<c>, the best fixed candidate, single-feature threshold rules and a depth-2 tree: fitted on the
  other folds' source games (all their branches), scored on all of this item's branches;
* agents: the choices in a score_bank2.py scorecard (--choices), e.g. stance and mimic.

"vs oracle" is the paired per-item difference (95% CI clustered by game). VPI is the value of perfect
information under the stratum prior (alpha, tau): how much perfect state reading could gain over the best
fixed candidate on average. When the oracle cannot beat the best fixed candidate, the branches cannot tell
the item's answer from the stratum's default, and the stratum only measures defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import bank, bank2  # noqa: E402


def load_choices(path: str) -> dict:
    doc = json.loads(Path(path).read_text())
    return {agent: {row["id"]: row["choice"] for row in card["rows"]}
            for agent, card in doc["cards"].items() if not agent.startswith(("xfit:", "always:"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bank", help="bank JSONL from scripts/build_bank2.py")
    parser.add_argument("--choices", help="scorecard JSON from score_bank2.py --out (agents' choices)")
    parser.add_argument("--prior", choices=bank2.PRIORS, default="candidate")
    parser.add_argument("--out", help="JSON (default <bank>.heldout.json)")
    args = parser.parse_args()
    path = Path(args.bank)
    good = bank2.usable(bank.load_items(path))
    if not good:
        raise SystemExit("no usable items")
    _, fits = bank2.label_items(good, args.prior)
    choices = load_choices(args.choices) if args.choices else {}
    report = bank2.heldout_check(good, fits, choices)
    for stratum, e in report.items():
        print(f"{stratum}: {e['items']} items, {e['games']} games, {e['branches_per_cand']:.1f} shared branches "
              f"per candidate; tau {e['tau']:.2f}; VPI {e['vpi']:.2f}")
        print(f"  {'policy':24} {'score':>16} {'vs oracle':>16}")
        rows = [("oracle (held-out)", e["oracle"]), ("best fixed (cross-fit)", e["best_fixed"]),
                ("tree depth 2 (CV)", e["tree"])]
        rows += [(f"threshold:{f}", v) for f, v in e["threshold"].items()]
        rows += [(f"always:{c}", v) for c, v in e["fixed"].items()]
        rows += [(a, v) for a, v in e["agents"].items()]
        for name, v in rows:
            vs = "" if name.startswith("oracle") else bank2.fmt_ci(v["vs_oracle"])
            print(f"  {name:24} {bank2.fmt_ci(v['score']):>16} {vs:>16}")
        print(f"  best rule {e['best_rule']}: vs oracle {bank2.fmt_ci(e['best_rule_vs_oracle'])}")
    # equal-weighted over strata, for the policies every stratum has
    print("equal-weighted over strata:")
    strata = list(report)
    policies = {"oracle (held-out)": lambda e: e["oracle"], "best fixed (cross-fit)": lambda e: e["best_fixed"],
                "tree depth 2 (CV)": lambda e: e["tree"]}
    for agent in set.intersection(*(set(e["agents"]) for e in report.values())) if report else ():
        policies[agent] = lambda e, a=agent: e["agents"][a]
    summary = {}
    for name, get in policies.items():
        vals, diffs, games, strata_of = [], [], [], []
        for s in strata:
            e = report[s]
            v = get(e)
            vals += v["values"]
            diffs += [a - b for a, b in zip(v["values"], e["oracle"]["values"])]
            games += e["game_of"]
            strata_of += [s] * len(v["values"])
        summary[name] = {"score": bank2.stratified_mean_ci(vals, games, strata_of),
                         "vs_oracle": bank2.stratified_mean_ci(diffs, games, strata_of)}
        vs = "" if name.startswith("oracle") else bank2.fmt_ci(summary[name]["vs_oracle"])
        print(f"  {name:24} {bank2.fmt_ci(summary[name]['score']):>16} {vs:>16}")
    mean_vpi = sum(report[s]["vpi"] for s in strata) / len(strata)
    print(f"  mean VPI {mean_vpi:.2f}")
    out = Path(args.out) if args.out else path.with_suffix(".heldout.json")
    out.write_text(json.dumps({"bank": str(path), "prior": args.prior, "strata": report, "equal": summary,
                               "mean_vpi": mean_vpi}, indent=1))
    print(f"report: {out}")


if __name__ == "__main__":
    main()

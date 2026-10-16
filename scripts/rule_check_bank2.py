"""Is a stratum of a decision bank v2 solved by a simple rule? (docs/BANK_V2.md section 6; no simulation)

    python scripts/rule_check_bank2.py results/bank2/bank.jsonl

Per stratum, from the stored branches (labels as in score_bank2.py): the regret of every fixed candidate,
of the best fixed candidate chosen on the other folds' source games, of every single-feature threshold
rule (`a if feature <= t else b`, feature in HP, gold, pairs, streak, level, losses to death; a, b and t
chosen on the other folds, RULE_FOLDS folds by source game) and of a depth-2 regret-minimizing tree
(cross-validated by game), each with its accuracy on clear items against the ceiling; and the cross-fit
oracle and worst. Headroom = best rule regret - cross-fit oracle regret; a stratum with headroom below
0.1 places is a rule stratum (its answers can be had without reading the state beyond one feature).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import bank, bank2  # noqa: E402


def describe_rule(rule: dict) -> str:
    if rule.get("kind") == "threshold":
        if rule["t"] is None:
            return f"always {rule['a']}"
        return f"{rule['a']} if {rule['feature']} <= {rule['t']:g} else {rule['b']}"
    if "leaf" in rule:
        return rule["leaf"]
    return f"({rule['feature']} <= {rule['t']:g} ? {describe_rule(rule['le'])} : {describe_rule(rule['gt'])})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bank", help="bank JSONL from scripts/build_bank2.py")
    parser.add_argument("--prior", choices=bank2.PRIORS, default="candidate")
    parser.add_argument("--folds", type=int, default=bank2.RULE_FOLDS)
    parser.add_argument("--out", help="JSON (default <bank>.rules.json)")
    args = parser.parse_args()
    path = Path(args.bank)
    good = bank2.usable(bank.load_items(path))
    if not good:
        raise SystemExit("no usable items")
    labels, fits = bank2.label_items(good, args.prior)
    report = bank2.rule_check(good, labels, args.folds)
    for stratum, e in report.items():
        print(f"{stratum}: {e['items']} items, {e['games']} games, {e['folds']} folds, {e['clear']} clear "
              f"(ceiling {e['ceiling']:.2f}); tau {fits[stratum]['tau2'] ** 0.5:.2f} ({fits[stratum]['tau_source']})")
        print(f"  {'xfit:oracle':28} {bank2.fmt_ci(e['xfit_oracle']):>16}")
        print(f"  {'xfit:worst':28} {bank2.fmt_ci(e['xfit_worst']):>16}")
        for c, s in e["fixed"].items():
            acc = "-" if s["accuracy"] is None else f"{s['accuracy']:.0%}"
            print(f"  {'always:' + c:28} {bank2.fmt_ci(s['regret']):>16}  acc {acc:>4}")
        rows = [("best fixed (cross-fit)", e["best_fixed"], None)]
        rows += [(f"threshold:{f}", s, s["rule_on_all"]) for f, s in e["threshold"].items()]
        rows += [("tree depth 2 (CV)", e["tree"], e["tree"]["rule_on_all"])]
        for name, s, rule in rows:
            acc = "-" if s["accuracy"] is None else f"{s['accuracy']:.0%}"
            print(f"  {name:28} {bank2.fmt_ci(s['regret']):>16}  acc {acc:>4}"
                  + (f"  on all items: {describe_rule(rule)}" if rule else ""))
        print(f"  best rule {e['best_rule']} {e['best_rule_regret']:+.2f}; headroom {e['headroom']:+.2f} -> "
              f"{'RULE STRATUM' if e['rule_stratum'] else 'needs more than one feature'}")
    out = Path(args.out) if args.out else path.with_suffix(".rules.json")
    out.write_text(json.dumps({"bank": str(path), "prior": args.prior, "fits": fits, "strata": report}, indent=1))
    print(f"report: {out}")


if __name__ == "__main__":
    main()

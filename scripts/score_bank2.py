"""Score agents on a decision bank v2: regret against the posterior-best candidate, accuracy on clear
items against the Bayesian ceiling, per stratum and equal-weighted over strata.

    python scripts/score_bank2.py results/bank2/pilot.jsonl --agent stance mimic random --stored
    python scripts/score_bank2.py results/bank2/bank.jsonl --sanity

Agents (tfteval.bank2.make_agent): a plan-seat kind (`stance`, `mimic`, `fast8`, ...; `rule` = `mimic`)
mapped to the nearest candidate (bank2.nearest_candidate: gold into xp and rerolls this round, fodder);
`random`; `always:<candidate>` (scored on the items whose menu has the candidate only);
`noisy<N>:<agent>`; `cmd:<shell command>` (reads bank.choice_prompt on stdin, answers {"choice": ...});
`py:<module>:<function>`. The cross-fit anchors `xfit:oracle` and `xfit:worst` are always reported
(picked on half the branches, scored on the other half; tfteval/bank2.py). There is no label-reading
oracle or worst.

Labels are recomputed from the stored branches: per stratum the random-effects prior (tau^2, candidate
effects; --prior zero for an exchangeable prior), then per item P(each candidate is best), clear items
(max P >= 0.85) and regrets against the posterior-best candidate. States are rebuilt from the recipes
and checked against the fingerprints unless --stored (then the stored public state is used and a
stateful planner starts fresh), as in score_bank.py; --sim / --rules only check the bank's settings.

--sanity scores stance, mimic, noisy50:stance, random and always:<each candidate> and checks the orders
xfit:oracle <= stance <= noisy50:stance <= random <= xfit:worst and stance <= mimic (a report, not a test).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # recipes only replay with a pinned hash seed
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import bank, bank2  # noqa: E402

SANITY_AGENTS = ["stance", "mimic", "noisy50:stance", "random"]
SANITY_ORDER = [("xfit:oracle", "stance", "the cross-fit oracle should not be beaten by a rule policy"),
                ("stance", "noisy50:stance", "half of stance's choices made at random should not help"),
                ("noisy50:stance", "random", "a rule policy should beat uniform choice"),
                ("random", "xfit:worst", "uniform choice should beat the cross-fit worst"),
                ("stance", "mimic", "stance placed better than mimic in the round robin and in bank v1")]


def verdict(d: dict) -> str:
    if d.get("mean") is None:
        return "no items"
    if d["mean"] == 0:
        return "tie"
    sig = d.get("ci95") is not None and abs(d["mean"]) > d["ci95"]
    return ("VIOLATED" if d["mean"] > 0 else "holds") + (" (significant)" if sig else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bank", help="bank JSONL from scripts/build_bank2.py")
    parser.add_argument("--agent", nargs="+", default=[], help="agents to score (see above)")
    parser.add_argument("--sanity", action="store_true", help="score the sanity ladder and check its order")
    parser.add_argument("--stored", action="store_true", help="use the stored public states, no rebuild")
    parser.add_argument("--prior", choices=bank2.PRIORS, default="candidate",
                        help="prior mean of the candidates' effects: the stratum's (default) or zero")
    parser.add_argument("--draws", type=int, default=bank2.POSTERIOR_DRAWS, help="Monte Carlo draws per item")
    parser.add_argument("--sim", help="check the bank was built with this simulator profile")
    parser.add_argument("--rules", choices=["set4", "set18"], help="check the bank was built with this economy")
    parser.add_argument("--out", help="scorecard JSON (default <bank>.score.json, or .sanity.json)")
    args = parser.parse_args()

    path = Path(args.bank)
    items = bank.load_items(path)
    found = {json.dumps(bank.recipe_settings(it["recipe"]), sort_keys=True) for it in items}
    if len(found) > 1:
        raise SystemExit(f"{path} mixes simulator settings: {sorted(found)}")
    if found and (args.sim or args.rules):
        built = json.loads(found.pop())
        want = bank.sim_settings(args.sim or built["sim"], args.rules or built["rules"], built["pickers"])
        if want != built:
            raise SystemExit(f"{path} was built with {built}, not {want}")
    summary = bank2.bank_summary(items)
    good = bank2.usable(items)
    print(f"{path}: {summary['built']} items from {summary['games']} source games, {summary['usable']} usable", flush=True)
    for s, e in summary["strata"].items():
        print(f"  {s:8} fired {e['fired']}/{e['games']} {e['rounds']}  usable {e['usable']}  dropped {e['dropped']}")
    if not good:
        raise SystemExit("no usable items")
    labels, fits = bank2.label_items(good, args.prior, args.draws)
    for s, fit in fits.items():
        print(f"  {s:8} prior: tau {fit['tau2'] ** 0.5:.2f} ({fit['tau_source']}, {fit['items']} items), pooled paired "
              f"SD {fit['pooled_sd_d']:.2f}, alpha " + " ".join(f"{c}:{a:+.2f}" for c, a in fit["alpha"].items()))
    menus = {c["name"] for it in good for c in it["candidates"]}
    names = list(dict.fromkeys(args.agent + (SANITY_AGENTS + [f"always:{c}" for c in bank2.CANDIDATES if c in menus]
                                             if args.sanity else [])))
    agents = [bank2.make_agent(n) for n in names]
    choices = bank.choose_all(good, agents, rebuild_states=not args.stored) if agents else {}
    cards = {}
    for which in ("oracle", "worst"):
        name = f"xfit:{which}"
        cards[name] = bank2.scorecard(bank2.crossfit_rows(good, labels, which), name, str(path))
    for agent in agents:
        part = good
        if agent.name.startswith("always:"):  # only the strata whose menu has the candidate
            part = [it for it in good if any(c["name"] == agent.target for c in it["candidates"])]
        rows = bank2.score_rows(part, labels, choices[agent.name])
        for row in rows:
            row["notes"] = choices[agent.name][row["id"]].get("notes")
        cards[agent.name] = bank2.scorecard(rows, agent.name, str(path))
    for card in cards.values():
        print(bank2.table(card), flush=True)
    result = {"bank": str(path), "summary": summary, "rebuilt": not args.stored, "prior": args.prior,
              "fits": fits, "labels": labels, "cards": cards}
    pairs = []
    if args.sanity:
        pairs = [(a, b, why) for a, b, why in SANITY_ORDER if a in cards and b in cards]
    elif len(names) > 1:
        pairs = [(a, b, "") for i, a in enumerate(names) for b in names[i + 1:]]
    if pairs:
        print("differences A - B (regret, 95% CI by source game; equal-weighted over strata, then per stratum):")
        result["differences"] = []
        for a, b, why in pairs:
            d = bank2.paired(cards[a], cards[b])
            result["differences"].append({"a": a, "b": b, "why": why, "diff": d, "verdict": verdict(d["equal"])})
            per = "  ".join(f"{s} {bank.fmt_ci(e)}" for s, e in d["strata"].items())
            print(f"  {a:>15} vs {b:<15} {bank.fmt_ci(d['equal']):>16}  {verdict(d['equal']):22} {per}")
    out = Path(args.out) if args.out else path.with_suffix(".sanity.json" if args.sanity else ".score.json")
    out.write_text(json.dumps(result, indent=1))
    print(f"scorecard: {out}")


if __name__ == "__main__":
    main()

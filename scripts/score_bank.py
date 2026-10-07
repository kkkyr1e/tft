"""Score agents on a decision bank: mean regret, share within the label interval, per point and split.

    python scripts/score_bank.py results/bank/pilot.jsonl --agent stance
    python scripts/score_bank.py results/bank/pilot.jsonl --agent stance mimic random --stored
    python scripts/score_bank.py results/bank/pilot.jsonl --sanity

Agents (tfteval.bank.make_agent): a plan-seat kind (`stance`, `mimic`, `fast8`, ...; `rule` is the rule
bot's economy, which `mimic` reproduces), whose own plan at the item's state is mapped to the nearest
candidate; `random`; `always:<candidate>`; `noisy<N>:<agent>` (N% of the choices replaced by a uniform
pick); `cmd:<shell command>` (reads a prompt on stdin, answers {"choice": ...}); `py:<module>:<function>`
(any callable view -> name, e.g. a model planner later; bank.CallableAgent from Python); and the
label-reading anchors `oracle` (the discovery-best candidate, regret 0) and `worst`.

By default every state is rebuilt from its recipe (each source game played once through its decision
points) and checked against the item's fingerprint; the agent then gets the rebuilt public state and,
for a planner of the hero's own kind, a copy of the hero's planner with its history. --stored skips the
rebuild and uses the public state stored in the item (fast; a stateful planner then starts fresh).

Labels are recomputed from the stored branches (--sd-floor changes the interval). The scorecard JSON
(--out, default next to the bank: <bank>.<agent>.score.json) has the regret with a 95% interval
clustered by source game and the within-interval share, for all / clear / ambiguous items and per
decision point, on all source games and on the dev and held-out splits, plus every choice.

--sanity scores oracle, stance, mimic, rule, noisy50:stance, random and worst, and checks the orders
that must hold if the bank measures decision quality: oracle <= stance <= noisy50:stance <= random <=
worst, and stance <= mimic (stance placed better than mimic in the fork smoke runs). It is a report,
not a test: each pair's regret difference is given with its interval clustered by source game.
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

from tfteval import bank  # noqa: E402

SANITY_AGENTS = ["oracle", "stance", "mimic", "rule", "noisy50:stance", "random", "worst"]
SANITY_ORDER = [("oracle", "stance", "by construction: the oracle has regret 0"),
                ("stance", "noisy50:stance", "half of stance's choices made at random should not help"),
                ("noisy50:stance", "random", "a rule policy should beat uniform choice"),
                ("random", "worst", "by construction"),
                ("stance", "mimic", "stance placed better than mimic in the fork smoke (2.25 vs 4.38, 8 games)")]


def sanity_report(cards: dict) -> tuple[list[dict], str]:
    rows, lines = [], ["sanity: lower regret expected for the first of each pair (difference A - B, CI by game)"]
    for a, b, why in SANITY_ORDER:
        d = bank.paired(cards[a], cards[b])
        if d.get("mean") is None:
            verdict = "no items"
        elif d["mean"] == 0:
            verdict = "tie"
        elif d["mean"] > 0:
            verdict = "VIOLATED" + (" (significant)" if d["ci95"] is not None and d["mean"] - d["ci95"] > 0 else "")
        else:
            verdict = "holds" + (" (significant)" if d["ci95"] is not None and d["mean"] + d["ci95"] < 0 else "")
        rows.append({"a": a, "b": b, "why": why, "diff": d, "verdict": verdict})
        lines.append(f"  {a:>15} vs {b:<15} {bank.fmt_ci(d):>16}  {verdict:24} {why}")
    return rows, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bank", help="bank JSONL from scripts/build_bank.py")
    parser.add_argument("--agent", nargs="+", default=[], help="agents to score (see above)")
    parser.add_argument("--sanity", action="store_true", help="score the sanity ladder and check its order")
    parser.add_argument("--stored", action="store_true", help="use the stored public states, no rebuild")
    parser.add_argument("--sd-floor", type=float, default=bank.SD_FLOOR, help="floor of the label interval's SD")
    parser.add_argument("--out", help="scorecard JSON (default <bank>.<agent>.score.json, or .sanity.json)")
    args = parser.parse_args()
    names = list(dict.fromkeys(args.agent + (SANITY_AGENTS if args.sanity else [])))
    if not names:
        raise SystemExit("give --agent or --sanity")

    path = Path(args.bank)
    items = bank.load_items(path)
    summary = bank.bank_summary(items)
    good = bank.usable(items, args.sd_floor)
    print(f"{path}: {summary['built']} items built, {summary['usable']} usable ({summary['clear']} clear) from "
          f"{summary['games']} source games; items dev {summary['splits']['dev']}, held-out {summary['splits']['heldout']}; "
          f"dropped {summary['dropped'] or 0}", flush=True)
    if not good:
        raise SystemExit("no usable items")
    agents = [bank.make_agent(n) for n in names]
    choices = bank.choose_all(good, agents, rebuild_states=not args.stored)
    cards = {}
    for agent in agents:
        card = bank.scorecard(good, choices[agent.name], agent.name, str(path))
        for row in card["rows"]:
            row["notes"] = choices[agent.name][row["id"]].get("notes")
        cards[agent.name] = card
        print(bank.table(card), flush=True)
    result = {"bank": str(path), "summary": summary, "rebuilt": not args.stored, "sd_floor": args.sd_floor,
              "cards": cards}
    if args.sanity:
        rows, text = sanity_report(cards)
        result["sanity"] = rows
        print(text)
    out = Path(args.out) if args.out else path.with_suffix(
        ".sanity.json" if args.sanity else f".{'_'.join(n.replace(':', '-') for n in names)}.score.json")
    out.write_text(json.dumps(result if len(cards) > 1 or args.sanity else cards[names[0]], indent=1))
    print(f"scorecard: {out}")


if __name__ == "__main__":
    main()

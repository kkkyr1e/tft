"""Round-robin analysis: one seat per policy per game, so policies are paired within each game.

    python scripts/round_robin_report.py results/m3/round_robin_set4_realistic.json [more.json ...]
"""
import json
import math
import sys
from collections import defaultdict


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n), n


def main(paths):
    for path in paths:
        games = json.load(open(path))
        games = [g for g in games if g.get("finished", True)]
        per = defaultdict(list)  # policy -> placements
        by_game = []
        rules = {g.get("rules") for g in games}
        for g in games:
            row = {}
            for seat, pol in g["lobby"].items():
                row[pol] = g["placements"][seat]
                per[pol].append(g["placements"][seat])
            by_game.append(row)
        pols = sorted(per, key=lambda p: sum(per[p]) / len(per[p]))
        secs = [g.get("seconds", 0) for g in games]
        print(f"\n## {path}  games={len(games)} rules={rules} mean_sec={sum(secs)/max(1,len(secs)):.1f}")
        print(f"{'policy':22s} {'mean':>6s} {'ci95':>6s} {'top4':>6s} {'win':>6s}")
        for p in pols:
            m, ci, n = mean_ci(per[p])
            top4 = sum(x <= 4 for x in per[p]) / n
            win = sum(x == 1 for x in per[p]) / n
            print(f"{p:22s} {m:6.2f} {ci:6.2f} {top4:6.1%} {win:6.1%}")
        print("\npairwise: row minus column, mean placement diff within game (negative = row better); * = CI excludes 0")
        print(" " * 22 + "".join(f"{p[:9]:>11s}" for p in pols))
        for a in pols:
            cells = []
            for b in pols:
                if a == b:
                    cells.append(f"{'':>11s}")
                    continue
                d = [r[a] - r[b] for r in by_game if a in r and b in r]
                m, ci, _ = mean_ci(d)
                star = "*" if abs(m) > ci else " "
                cells.append(f"{m:+5.2f}±{ci:.2f}{star}".rjust(11))
            print(f"{a:22s}" + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])

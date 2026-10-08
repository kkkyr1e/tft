"""Rebuild every usable bank v2 item in this (fresh) process and compare with the stored fingerprint."""
import json, sys, collections
from tfteval import bank
rows = [json.loads(l) for l in open(sys.argv[1])]
items = [r for r in rows if 'branches' in r and not r.get('dropped')]
by_game = collections.defaultdict(list)
for it in items:
    by_game[it['game']].append(it)
out = []
def diff(x, y, path=''):
    if isinstance(x, dict) and isinstance(y, dict):
        return [d for k in sorted(set(x) | set(y)) for d in diff(x.get(k), y.get(k), path + '/' + str(k))]
    return [] if x == y else [f"{path}: stored={x} rebuilt={y}"]
for g, its in by_game.items():
    its.sort(key=lambda it: it['recipe']['round'])
    game = None
    for it in its:
        if game is None:
            game = bank.new_game(it['recipe'])
        game.run(until_round=it['recipe']['round'])
        seat = it['recipe']['hero_seat']
        fp = bank.fingerprint(game, seat)
        d = diff(it['fingerprint'], fp)
        out.append({"id": it['id'], "stratum": it['stratum'], "round": it['recipe']['round'],
                    "harness": it['recipe'].get('harness_commit', '')[:7], "match": not d, "diff": d})
        print(("ok  " if not d else "DIFF"), it['id'].split('#')[1], it['recipe']['round'], d[:3], flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print("mismatch", sum(not o['match'] for o in out), "of", len(out))

"""Replay stored branches of rebuildable bank v2 items in this process: same place, same action digests?
usage: branch_replay.py BANK REBUILD_JSON PART NPARTS OUT"""
import json, sys, random
from tfteval import bank
from tfteval.branching import snapshot
bank_path, rebuild_path, part, nparts, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ok_ids = {r['id'] for r in json.load(open(rebuild_path)) if r['match']}
rows = [json.loads(l) for l in open(bank_path)]
items = [r for r in rows if r['id'] in ok_ids and 'branches' in r and not r.get('copied_from')]
random.Random(7).shuffle(items)
items = items[:24][part::nparts]
out = []
for it in items:
    seat = it['recipe']['hero_seat']
    snap = snapshot(bank.rebuild(it['recipe'], it['fingerprint_hash']))
    cands = {c['name']: c for c in it['candidates']}
    picks = random.Random(it['seed']).sample(it['branches'], 2)
    for b in picks:
        r = bank.play_branch(snap, seat, cands[b['cand']], b['k'], it['continuation'])
        first = next((rnd for rnd in sorted(b['actions'], key=int) if r['actions'].get(rnd) != b['actions'][rnd]), None)
        rec = {"id": it['id'], "cand": b['cand'], "k": b['k'], "stored": b['place'], "replay": r['place'],
               "same_place": b['place'] == r['place'], "same_actions": r['actions'] == b['actions'], "first_diff_round": first}
        out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(out_path, 'w'), indent=1)

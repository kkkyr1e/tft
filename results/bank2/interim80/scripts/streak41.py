"""Q3 check: on the losing3 items, a 'streak through stage 3, then level and roll at 4-1' candidate,
played on the same branch keys as the stored ones (paired). usage: streak41.py BANK REBUILD_JSON PART NPARTS OUT"""
import json, sys, time
from tfteval import bank, bank2
from tfteval.bank import CommitPlanner, ActionLog, plan_kind
from tfteval.branching import snapshot, branch, switch_planner

FOUR_ONE = 15

def spec(name):
    c = {"name": name, **{k: v for k, v in bank2.CANDIDATES[name].items() if k != "desc"}}
    return c

def make_switch(fired, continuation):
    def apply(old):
        planner = getattr(old, "planner", None)
        base = planner if getattr(planner, "name", None) == continuation else plan_kind(continuation)
        inner = CommitPlanner(base, spec("level_roll10"), FOUR_ONE + 2 - fired)   # level and roll at 4-1, 4-2
        outer = CommitPlanner(inner, spec("streak"), FOUR_ONE - fired)            # streak up to 3-7
        return switch_planner(outer)(old)
    return apply

def play(snap, seat, fired, k, continuation):
    started = time.time()
    g = branch(snap, reseed=k, switch={seat: make_switch(fired, continuation)})
    log = g.seat_policies[seat] = ActionLog(g.seat_policies[seat])
    while not g.done and seat not in g.placements:
        g.run(until_round=g.round + 1)
    outer = log.policy.planner
    return {"cand": "streak41", "k": k, "place": g.placements.get(seat), "applied_streak": list(outer.applied),
            "applied_roll": list(outer.base.applied), "seconds": round(time.time() - started, 2)}

bank_path, rebuild_path, part, nparts, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ok_ids = {r['id'] for r in json.load(open(rebuild_path)) if r['match']}
rows = [json.loads(l) for l in open(bank_path)]
items = sorted([r for r in rows if r.get('stratum') == 'losing3' and 'branches' in r and r['id'] in ok_ids], key=lambda r: r['seed'])
items = items[part::nparts]
LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 else None
jobs = [(it, k) for it in items for k in sorted({b['k'] for b in it['branches'] if b['cand'] == 'streak'})][:LIMIT]
out, snaps = [], {}
for it, k in jobs:
    if it['id'] not in snaps:
        snaps.clear()
        snaps[it['id']] = snapshot(bank.rebuild(it['recipe'], it['fingerprint_hash']))
    r = play(snaps[it['id']], it['recipe']['hero_seat'], it['recipe']['round'], k, it['continuation'])
    r['id'] = it['id']
    out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(out_path, 'w'))

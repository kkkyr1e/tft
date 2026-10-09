"""How the hero spends its per-round actions in one branch: action type counts per round.
usage: action_mix.py SEED POINT HP CAP K CAND"""
import collections, sys
sys.path.insert(0, "/home/claude/tft-v2")
from tfteval import bank, pairs
from tfteval.bank import commit_switch
from tfteval.branching import branch, snapshot

seed, point, hp, cap, k, name = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
lineup = "stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1"
recipe, game = pairs.base_recipe(lineup, seed, point, bank.sim_settings("realistic", "set4"))
seat = recipe["hero_seat"]
pairs.apply_edit(game, seat, {"hp": hp})
if cap:
    pairs.set_action_budget(game, cap)
snap = snapshot(game)
c = pairs.candidate(name)
g = branch(snap, reseed=k, switch={seat: commit_switch(c["spec"] | {"name": name}, c["rounds"], "stance")})
counts = collections.defaultdict(collections.Counter)
inner = g.seat_policies[seat]
orig = inner.act
NAMES = {0: "pass", 1: "xp", 2: "roll", 3: "buy?", 4: "sell?", 5: "move?", 6: "item", 7: "7"}
def act(observation, info, agent, env):
    a = orig(observation, info, agent, env)
    counts[int(info.get("game_round", 0))][NAMES.get(int(a[0]), str(a[0]))] += 1
    return a
inner.act = act
start = g.round
while not g.done and seat not in g.placements and g.round < start + 4:
    p = g.player(seat)
    gold = int(p.gold)
    g.run(until_round=g.round + 1)
    print(g.round - 1, "gold at start", gold, dict(counts[g.round - 1]), flush=True)

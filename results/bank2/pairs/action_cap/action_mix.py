"""How the hero spends its per-round actions in one branch: action type counts per round, with moves split
into board<->board (repositioning) and board<->bench (a unit swapped in or out).
usage: action_mix.py SEED POINT HP CAP K CAND [PROFILE]   (CAP 0: the profile's budget; PROFILE: realistic15)"""
import collections, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tfteval import bank, pairs
from tfteval.bank import commit_switch
from tfteval.branching import branch, snapshot

seed, point, hp, cap, k, name = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
profile = sys.argv[7] if len(sys.argv) > 7 else "realistic15"
lineup = "stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1"
recipe, game = pairs.base_recipe(lineup, seed, point, bank.sim_settings(profile, "set4"))
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
# action types of Simulator/game/step_function.py; a move's slots are 0..27 board, 28..36 bench (player.move_champ_action)
NAMES = {0: "pass", 1: "xp", 2: "roll", 3: "buy", 4: "sell", 5: "move", 6: "item"}


def where(slot):
    return "board" if slot < 28 else "bench"


def act(observation, info, agent, env):
    a = orig(observation, info, agent, env)
    kind = NAMES.get(int(a[0]), str(a[0]))
    if int(a[0]) == 5:
        x1, x2 = sorted((int(a[1]), int(a[2])))
        kind = f"move {where(x1)}<->{where(x2)}"
    counts[int(info.get("game_round", 0))][kind] += 1
    return a


inner.act = act
start = g.round
while not g.done and seat not in g.placements and g.round < start + 4:
    p = g.player(seat)
    gold = int(p.gold)
    g.run(until_round=g.round + 1)
    print(g.round - 1, "gold at start", gold, dict(counts[g.round - 1]), flush=True)

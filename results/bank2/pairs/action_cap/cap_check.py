"""Does the 15-action cap make roll_all = roll30 and flatten save vs roll_all at high HP?
One 4-1 state, hero HP edited, the lobby's per-round action budget raised from the decision point on.
usage: cap_check.py SEED POINT HP CAP K CAND... > out"""
import json, sys
sys.path.insert(0, "/home/claude/tft-v2")
from tfteval import bank, pairs
from tfteval.bank import ActionLog, commit_switch
from tfteval.branching import branch, snapshot

seed, point, hp, cap, K = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
cands = sys.argv[6:]
lineup = "stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1"
recipe, game = pairs.base_recipe(lineup, seed, point, bank.sim_settings("realistic", "set4"))
seat = recipe["hero_seat"]
pairs.apply_edit(game, seat, {"hp": hp})
env = game.env.unwrapped
if cap:
    env.max_actions_per_round = cap
    pm = env.player_manager
    pm.config.max_actions_per_round = cap
    for p in pm.player_states.values():
        if p is not None:
            p.actions_remaining = cap
            p.actions_per_round = cap
snap = snapshot(game)
start = game.round
for k in range(K):
    for name in cands:
        c = pairs.candidate(name)
        g = branch(snap, reseed=k, switch={seat: commit_switch(c["spec"] | {"name": name}, c["rounds"], "stance")})
        log = g.seat_policies[seat] = ActionLog(g.seat_policies[seat])
        traj = []
        while not g.done and seat not in g.placements:
            p = g.player(seat)
            if p is not None and g.round <= start + 3:
                traj.append([int(g.round), int(p.gold), int(p.level), int(p.health)])
            g.run(until_round=g.round + 1)
        print(json.dumps({"k": k, "cand": name, "cap": cap, "hp": hp, "place": g.placements.get(seat),
                          "actions": log.digests(), "traj": traj}), flush=True)

"""Full games from the start at a given per-round action budget: time, steps, the hero's place,
and how many actions each seat type uses per planning phase. usage: cap_games.py CAP SEED..."""
import collections, json, sys, time
sys.path.insert(0, "/home/claude/tft-v2")
from tfteval import bank

cap, seeds = int(sys.argv[1]), [int(s) for s in sys.argv[2:]]
lineup = "stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1"
sim = "realistic" + (f",max_actions_per_round={cap}" if cap != 15 else "")
for seed in seeds:
    recipe = bank.make_recipe(lineup, seed, "5-1", bank.sim_settings(sim, "set4"))
    game = bank.new_game(recipe)
    env = game.env.unwrapped
    used = collections.defaultdict(list)
    started = time.time()
    while not game.done:
        rnd = game.round
        game.run(until_round=rnd + 1)
    kinds = {seat: spec for seat, spec in recipe["lobby"].items()}
    print(json.dumps({"cap": cap, "seed": seed, "seconds": round(time.time() - started, 1), "steps": game.steps,
                      "rounds": game.round, "placements": {kinds[s]: p for s, p in game.placements.items()}
                      if len(set(kinds.values())) == len(kinds) else game.placements,
                      "lobby": kinds}), flush=True)

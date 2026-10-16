"""Full games from the start at a given per-round action budget: time, steps, rounds and placements.
usage: cap_games.py CAP SEED..."""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tfteval import bank

cap, seeds = int(sys.argv[1]), [int(s) for s in sys.argv[2:]]
lineup = "stance@rule:2,mimic:1,mimicfc:1,fast8:1,stance:1,stance+hold:1"
sim = "realistic15" if cap == 15 else f"realistic,max_actions_per_round={cap}"
for seed in seeds:
    recipe = bank.make_recipe(lineup, seed, "5-1", bank.sim_settings(sim, "set4"))
    game = bank.new_game(recipe)
    started = time.time()
    while not game.done:
        game.run(until_round=game.round + 1)
    kinds = {seat: spec for seat, spec in recipe["lobby"].items()}
    print(json.dumps({"cap": cap, "seed": seed, "seconds": round(time.time() - started, 1), "steps": game.steps,
                      "rounds": game.round, "placements": {kinds[s]: p for s, p in game.placements.items()}
                      if len(set(kinds.values())) == len(kinds) else game.placements,
                      "lobby": kinds}), flush=True)

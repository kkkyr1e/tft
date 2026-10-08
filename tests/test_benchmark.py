"""Benchmark v1 (tfteval/benchmark.py, benchmarks/v1.json): lobby sampling, the frozen configuration and its
hash, the refusal to merge results across configurations, the scorecard on synthetic results and the
decision-bank hook. No games."""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import benchmark as bm  # noqa: E402

CONFIG = bm.load_config()
V1_HASH = "3bc9bb108cc1c077"  # change only on purpose: a new hash is a new benchmark


def test_the_v1_configuration_is_frozen():
    assert bm.config_hash(CONFIG) == V1_HASH
    assert CONFIG["sim"] == "realistic" and CONFIG["default_track"] == "set4"
    assert {t: c["rules"] for t, c in CONFIG["tracks"].items()} == {"set4": "set4", "set18": "set18"}
    assert CONFIG["pools"]["dev"] == {"policies": ["rule", "mimic", "mimicfc", "fast8", "stance", "stance+hold"],
                                      "first_seed": 30000}
    assert CONFIG["pools"]["heldout"] == {"policies": ["stance1", "stance+lossstreak", "noisy20"], "first_seed": 39000}
    assert not set(CONFIG["pools"]["dev"]["policies"]) & set(CONFIG["pools"]["heldout"]["policies"])
    assert bm.games_planned(CONFIG) == 312  # (1.96 x 2.25 / 0.25)^2
    assert bm.games_planned(CONFIG, sd=2.0, half_width=0.5) == 62


def test_config_hash_ignores_doc_keys_only():
    edited = copy.deepcopy(CONFIG)
    edited["_doc"] = "reworded"
    assert bm.config_hash(edited) == V1_HASH
    edited["pools"]["dev"]["policies"].append("noisy50")
    assert bm.config_hash(edited) != V1_HASH
    reordered = json.loads(json.dumps(CONFIG, sort_keys=True))
    assert bm.config_hash(reordered) == V1_HASH  # key order does not matter


def test_every_pool_policy_is_a_valid_spec():
    from tfteval import make_policy
    from tfteval.policies import NoisyRulePolicy

    for pool in CONFIG["pools"].values():
        for spec in pool["policies"]:
            assert make_policy(spec).name == spec
    noisy = make_policy("noisy20")
    assert isinstance(noisy, NoisyRulePolicy) and noisy.epsilon == 0.2


def test_lobby_sampling_is_deterministic_and_the_hero_rotates():
    a = bm.sample_lobby(CONFIG, "dev", 30005, "stance")
    assert a == bm.sample_lobby(CONFIG, "dev", 30005, "stance")
    assert a["hero_seat"] == "player_5" and a["seats"]["player_5"] == "hero=stance"
    assert [s for s in a["seats"] if s != a["hero_seat"]] == [f"player_{i}" for i in range(8) if i != 5]
    assert [a["seats"][s] for s in a["seats"] if s != a["hero_seat"]] == a["opponents"]
    assert a["stance_opponents"] == sum(p.startswith("stance") for p in a["opponents"])
    # the same lobby for every agent (pairing by seed); an alias on the agent spec is dropped
    b = bm.sample_lobby(CONFIG, "dev", 30005, "x=mimic")
    assert b["opponents"] == a["opponents"] and b["seats"]["player_5"] == "hero=mimic"
    seats = {bm.sample_lobby(CONFIG, "dev", s, "stance")["hero_seat"] for s in range(30000, 30008)}
    assert seats == {f"player_{i}" for i in range(8)}


def test_lobby_mixes_vary_and_draw_with_replacement():
    lobbies = [bm.sample_lobby(CONFIG, "dev", s, "stance")["opponents"] for s in range(30000, 30200)]
    assert len({tuple(sorted(o)) for o in lobbies}) > 100  # mixes vary from game to game
    assert all(len(set(o)) < 7 for o in lobbies)  # 7 draws from 6 policies always repeat one
    drawn = Counter(p for o in lobbies for p in o)
    assert set(drawn) == set(CONFIG["pools"]["dev"]["policies"])
    assert max(drawn.values()) / min(drawn.values()) < 1.4  # roughly uniform
    held = bm.sample_lobby(CONFIG, "heldout", 39000, "stance")
    assert set(held["opponents"]) <= set(CONFIG["pools"]["heldout"]["policies"])
    assert bm.seeds(CONFIG, "heldout", 3) == [39000, 39001, 39002] and bm.seeds(CONFIG, "dev", 2) == [30000, 30001]


def meta(**changes):
    out = bm.new_meta(CONFIG, "stance", "set4", "dev", True, 312, "simA", "harness1")
    out.update(changes)
    return out


def test_results_of_another_configuration_are_refused():
    assert bm.check_meta(meta(), meta()) == []
    with pytest.raises(bm.ConfigMismatch, match="config_hash"):
        bm.check_meta(meta(config_hash="0000"), meta())
    for key, value in (("agent", "mimic"), ("track", "set18"), ("pool", "heldout"), ("record", False)):
        with pytest.raises(bm.ConfigMismatch, match=key):
            bm.check_meta(meta(**{key: value}), meta())
    with pytest.raises(bm.ConfigMismatch, match="simulator commit"):
        bm.check_meta(meta(sim_commit="simB"), meta())
    assert "harness commit differs" in bm.check_meta(meta(harness_commit="harness2"), meta())[0]
    docs = [{"meta": meta(), "games": [{"seed": 30000}]}, {"meta": meta(config_hash="0000"), "games": []}]
    with pytest.raises(bm.ConfigMismatch):
        bm.merge_results(docs)
    with pytest.raises(bm.ConfigMismatch):
        bm.merge_results(docs[1:], CONFIG)  # made under another configuration than this one
    later = {"meta": meta(harness_commit="h2"), "games": [{"seed": 30001}, {"seed": 30000}]}
    merged = bm.merge_results([docs[0], later], CONFIG)
    assert [g["seed"] for g in merged["games"]] == [30000, 30001] and merged["warnings"]


def test_results_file_round_trip(tmp_path):
    doc = {"meta": meta(), "games": [{"seed": 30000}]}
    path = tmp_path / "r.json"
    bm.save_results(path, doc)
    assert bm.load_results(path) == doc
    path.write_text("{}")
    with pytest.raises(ValueError):
        bm.load_results(path)


COMPS = {"mage": ["ahri", "annie"], "divine": ["irelia", "wukong"], "elderwood": ["ashe"]}


def fake_game(seed, place, comp, record=True, stance=2, contested="mage"):
    hero = f"player_{seed % 8}"
    rows = [{"round": 9, "gold": 30, "level": 5, "xp": 0, "end": {"gold": 30, "level": 5, "xp": 0},
             "board": [["ahri", 1, 4, ["bf_sword"]]], "bench": [], "items": [], "contested": {contested: 3},
             "actions": {"buy": 3}, "illegal": 0, "fallbacks": 0},
            {"round": 20, "gold": 60, "level": 7, "xp": 0, "end": {"gold": 60, "level": 7, "xp": 0},
             "board": [["irelia", 2, 3, ["infinity_edge"]]], "bench": [], "items": [], "comp": comp,
             "actions": {"pass": 15}, "illegal": 0, "fallbacks": 0}]
    placements = {f"player_{i}": i + 1 for i in range(8)}
    other = next(s for s, p in placements.items() if p == place)
    placements[other], placements[hero] = placements[hero], place
    lobby = {f"player_{i}": "rule" for i in range(8)}
    lobby[hero] = "hero"
    result = {"seed": seed, "lobby": lobby, "placements": placements, "finished": True, "fallbacks": {hero: 0},
              "rules": "set4", "sim_commit": "simA", "eliminated": {hero: 20} if place > 1 else {},
              "records": {hero: rows} if record else {}}
    return {"seed": seed, "hero_seat": hero, "lobby": lobby, "stance_opponents": stance, "harness_commit": "h1",
            "result": result}


def test_scorecard_on_synthetic_results():
    games = [fake_game(30000 + i, place, comp, stance=i % 3)
             for i, (place, comp) in enumerate([(1, "mage"), (2, "divine"), (3, "mage"), (4, "elderwood"), (6, "mage"),
                                                (8, "divine"), (2, "mage"), (5, "divine")])]
    doc = {"meta": meta(planned_games=8), "games": games}
    card = bm.scorecard(doc, CONFIG, comps=COMPS)
    s = card["strength"]
    assert s["n"] == 8 and s["mean"] == pytest.approx(31 / 8) and s["histogram"] == [1, 2, 1, 1, 1, 1, 0, 1]
    assert s["top4"]["k"] == 5 and s["win"]["k"] == 1
    assert set(card["by_stance_opponents"]) == {"0", "1", "2"}
    assert sum(v["n"] for v in card["by_stance_opponents"].values()) == 8
    assert card["recorded_games"] == 8 and card["rubric"]["n"] == 8
    assert card["rubric"]["items"]["died_with_gold"]["k"] == 7  # 60 gold at elimination, every place but 1st
    v = card["variety"]
    assert v["top4_games"] == 5 and v["final_comps"]["counts"] == {"mage": 3, "divine": 1, "elderwood": 1}
    assert v["noninferiority"]["noninferior"] is None  # no comparison agent
    md = bm.scorecard_markdown(card)
    assert "Mean placement" in md and "died_with_gold" in md and "Strategic variety" in md

    # a comparison agent on the same seeds, placing 1 worse: the hero is non-inferior, and more varied
    worse = [fake_game(g["seed"], min(8, bm.hero_place(g) + 1), "mage") for g in games]
    compare = {"meta": meta(agent="mimic", planned_games=8), "games": worse}
    ni = bm.scorecard(doc, CONFIG, compare=compare, comps=COMPS)["variety"]["noninferiority"]
    assert ni["noninferior"] is True and ni["variety_plus"] is True and ni["diff"] < 0
    with pytest.raises(bm.ConfigMismatch):
        bm.scorecard(doc, CONFIG, compare={"meta": meta(agent="mimic", pool="heldout"), "games": worse}, comps=COMPS)
    with pytest.raises(bm.ConfigMismatch):
        bm.scorecard({"meta": meta(config_hash="0000"), "games": games}, CONFIG, comps=COMPS)

    unrecorded = {"meta": meta(record=False), "games": [fake_game(30000, 3, "mage", record=False)]}
    card = bm.scorecard(unrecorded, CONFIG, comps=COMPS)
    assert card["rubric"] is None and card["variety"] is None
    assert "not recorded" in bm.scorecard_markdown(card)
    held = bm.scorecard({"meta": meta(pool="heldout"), "games": games[:2]}, CONFIG, comps=COMPS)
    assert any("milestones only" in w for w in held["warnings"])


def test_bank_regret_hook(tmp_path, monkeypatch):
    assert bm.bank_regret("stance", "set4", None, None)["status"] == "not run"
    monkeypatch.setattr(bm, "BANK_SCRIPT", tmp_path / "missing.py")
    assert bm.bank_regret("stance", "set4", "bank.jsonl", "s")["status"] == "unavailable"
    script = tmp_path / "score_bank.py"
    script.write_text("import argparse, json\np = argparse.ArgumentParser()\np.add_argument('bank')\n"
                      "for a in ('--agent', '--rules', '--out'): p.add_argument(a)\n"
                      "a = p.parse_args()\n"
                      "json.dump({'regret_mean': 0.4, 'n': 3, 'policy': a.agent, 'bank': a.bank}, open(a.out, 'w'))\n")
    monkeypatch.setattr(bm, "BANK_SCRIPT", script)
    out = bm.bank_regret("mimic", "set4", "my bank.jsonl", "stance")
    assert out["status"] == "ok" and out["result"] == {"regret_mean": 0.4, "n": 3, "policy": "stance",
                                                       "bank": "my bank.jsonl"}
    script.write_text("raise SystemExit(3)\n")
    assert bm.bank_regret("stance", "set4", "b", "s")["status"] == "error"

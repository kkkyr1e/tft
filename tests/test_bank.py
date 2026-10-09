"""Decision bank (tfteval/bank.py): candidates, CommitPlanner, labels, duplicates, scoring, rebuild.

Everything here but the last two tests is pure Python; those play the first rounds of one game (~10 s)."""

import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfteval import bank, stages  # noqa: E402
from tfteval.planner import ParamPlanner, PlanPolicy, compile_knobs  # noqa: E402


def state(**kw):
    s = {"round": stages.parse_label("3-2"), "gold": 50, "level": 5, "xp": 2, "hp": 70, "losses_to_death": 6}
    s.update(kw)
    return s


# a stance-like plan with every economy field set: the overrides must replace all of them
BASE_PLAN = {"comp": "mage", "level_to": 6, "roll_floor": 20, "carry": None, "field_comp": True, "survival": 2,
             "spend": {"to": 10, "rounds": 2}, "xp_buys": 3, "level_by": {"level": 8, "by": "4-1"},
             "stance": "stabilize"}


def knobs_for(name, st, anchor=None):
    spec = bank.CANDIDATES[name]
    return compile_knobs(bank.override(BASE_PLAN, bank.resolve(spec, st, anchor)), st)


def test_points_and_menus():
    assert [stages.parse_label(p) for p in bank.POINTS] == [10, 12, 15, 18, 21]
    for point, spec in bank.POINTS.items():
        assert len(spec["menu"]) == 3 and 2 <= spec["rounds"] <= 3
        assert all(c["name"] in bank.CANDIDATES and c["desc"] for c in bank.menu(point))


def test_candidate_overrides_compile_to_the_intended_knobs():
    st = state()  # 3-2: level 5, 2 xp, 50 gold; the lobby curve asks for 6 (18 xp = 5 buys)
    save, level, roll = knobs_for("save", st), knobs_for("level", st), knobs_for("roll", st)
    for k in (save, level, roll):  # the base's economy is gone, the rest is kept
        assert k["comp"] == "mage" and k["field_comp"] and not k["survival"] and not k["xp_priority"]
        assert k["level_to"] == 5
    assert save["roll_floor"] == 999 and save["xp_buys"] == 0  # xp only above 50 gold: none
    assert level["roll_floor"] == 999 and level["xp_buys"] == 5  # 20 gold of xp leaves 30
    assert roll["roll_floor"] == 15 and roll["xp_buys"] == 0
    assert knobs_for("level", state(gold=40))["xp_buys"] == 2  # partial: only the gold above 30
    assert knobs_for("save", state(gold=62))["xp_buys"] == 3  # the gold above 50, toward the curve
    assert knobs_for("save", state(level=6, xp=0, gold=80))["xp_buys"] == 0  # already on the curve

    st = state(round=stages.parse_label("4-1"), level=7, xp=10, gold=60)
    fast8 = knobs_for("fast8", st)
    assert fast8["xp_buys"] == 12 and fast8["roll_floor"] == 999  # 46 xp to 8, 50 gold above 10
    landed = knobs_for("fast8", state(round=16, level=8, xp=0, gold=44), anchor=7)
    assert landed["xp_buys"] == 0 and landed["roll_floor"] == 20  # at 8: roll down to 20
    at8 = knobs_for("fast8", state(round=18, level=8, xp=0, gold=44))  # already 8 at the decision: toward 9
    assert at8["xp_buys"] == 8 and at8["roll_floor"] == 999
    cap = knobs_for("cap_out", state(round=21, level=8, xp=0, gold=50))
    assert cap["xp_buys"] == 10 and cap["roll_floor"] == 999

    dying = state(losses_to_death=1)  # the base's survival would roll to 0; the commitment holds
    assert compile_knobs(BASE_PLAN, dying)["survival"]
    assert not knobs_for("save", dying)["survival"] and knobs_for("save", dying)["roll_floor"] == 999


class Base:
    name = "base"

    def __init__(self):
        self.calls = []

    def plan(self, st, comps, comp_now):
        self.calls.append(st["round"])
        return {"comp": "mage", "level_to": st["level"], "roll_floor": 999, "carry": None, "field_comp": True}


def test_commit_planner_hands_over_after_k_rounds_and_pickles():
    planner = bank.CommitPlanner(Base(), "roll", rounds=2)
    p10 = planner.plan(state(round=10), {}, None)
    assert p10["roll_floor"] == 15 and p10["commit"] == "roll" and p10["comp"] == "mage"
    planner = pickle.loads(pickle.dumps(planner))  # mid-commitment, e.g. inside a snapshot
    assert planner.plan(state(round=11), {}, None)["roll_floor"] == 15
    p12 = planner.plan(state(round=12), {}, None)
    assert p12 == Base().plan(state(round=12), {}, None)  # handed over: the base's plan, untouched
    assert planner.plan(state(round=13), {}, None)["roll_floor"] == 999
    assert planner.base.calls == [10, 11, 12, 13]  # the base is asked every round
    assert planner.applied == [10, 11] and planner.changed == [10, 11]
    assert planner.anchor["level"] == 5 and planner.start == 10
    # a commitment that equals the base's own plan changes nothing
    same = bank.CommitPlanner(Base(), "save", rounds=2)
    same.plan(state(round=10, gold=40), {}, None)
    assert same.applied == [10] and same.changed == []


def test_commit_switch_keeps_the_executor_and_the_continuation_planner():
    old = PlanPolicy(ParamPlanner(), name="hero")  # a mimic seat
    old.executor, old.round = object(), 9
    new = bank.commit_switch("level", 2, continuation="mimic")(old)
    assert isinstance(new.planner, bank.CommitPlanner) and new.planner.base is old.planner  # same kind: kept
    assert new.executor is old.executor and new.round is None and new.name == "hero"
    other = bank.commit_switch("level", 2, continuation="fast8")(old)
    assert other.planner.base is not old.planner and other.planner.base.level_gold == 34  # a fresh fast8
    pickle.loads(pickle.dumps(new.planner))
    with pytest.raises(TypeError):
        bank.commit_switch("roll", 2)(object())


def make_item(places, actions=None, kd=2, kl=3, game="g#1", point="3-2", seed=1):
    """places: name -> list of placements for k = 0 .. kd+kl-1."""
    names = list(places)
    branches = []
    for name in names:
        for k, p in enumerate(places[name]):
            acts = (actions or {}).get(name, {}).get(k, {"10": f"{name}{k}"})
            branches.append({"cand": name, "k": k, "place": p, "actions": acts,
                             "set": "discovery" if k < kd else "label"})
    return {"id": f"{game}@{point}", "game": game, "seed": seed, "point": point, "split": bank.split(seed),
            "kd": kd, "kl": kl, "candidates": [{"name": n, "desc": n, "rounds": 2, "spec": {}} for n in names],
            "branches": branches, "dropped": None}


def test_regret_sign_and_the_discovery_label_split():
    places = {"save": [2, 2, 3, 3, 4], "level": [5, 5, 6, 7, 6], "roll": [4, 4, 3, 4, 4]}
    lab = bank.label_item(make_item(places))
    assert lab["best"] == "save"  # discovery means 2 / 5 / 4
    assert lab["regret"]["save"] == 0 and lab["regret"]["level"] == pytest.approx(3.0)  # worse: positive
    assert lab["regret"]["roll"] == pytest.approx(1 / 3)
    assert lab["label_mean"] == {"save": pytest.approx(10 / 3), "level": pytest.approx(19 / 3),
                                 "roll": pytest.approx(11 / 3)}
    # pooled paired SD sqrt((1 + 1/3) / 2) < the floor 1.0; df 4; half width 2.776 / sqrt(3)
    assert lab["sd"] == 1.0 and lab["df"] == 4 and lab["ci95"] == pytest.approx(2.776 / math.sqrt(3))
    assert lab["within_ci"] == {"save": True, "level": False, "roll": True} and lab["kind"] == "ambiguous"

    # the label set never picks the best: making save look bad there changes the regrets, not the pick
    worse = dict(places, save=[2, 2, 7, 7, 7])
    lab2 = bank.label_item(make_item(worse))
    assert lab2["best"] == "save" and lab2["regret"]["roll"] < 0  # unclipped: the discovery pick was wrong
    # and the discovery set alone decides it
    better_roll = dict(places, roll=[1, 1, 3, 4, 4])
    assert bank.label_item(make_item(better_roll))["best"] == "roll"
    # ties on discovery go to the menu order
    assert bank.label_item(make_item(dict(places, roll=[2, 2, 3, 4, 4])))["best"] == "save"

    clear = bank.label_item(make_item(dict(places, roll=[4, 4, 6, 6, 7])))
    assert clear["kind"] == "clear" and clear["within_ci"]["roll"] is False


def test_duplicates_are_grouped_and_an_all_duplicate_item_is_dropped():
    same = {k: {"10": "x", "11": f"y{k}"} for k in range(5)}
    places = {"save": [3, 3, 4, 2, 5], "level": [3, 3, 4, 2, 5], "roll": [6, 6, 7, 7, 7]}
    lab = bank.label_item(make_item(places, {"save": same, "level": same}))
    assert lab["duplicates"] == [["save", "level"], ["roll"]] and not lab["dropped"]
    assert lab["best_group"] == ["save", "level"] and lab["regret"]["level"] == 0 and lab["within_ci"]["level"]

    # identical on discovery but not in one label branch: not duplicates
    late = {**same, 4: {"10": "x", "11": "z"}}
    lab = bank.label_item(make_item(places, {"save": same, "level": late}))
    assert lab["duplicates"] == [["save"], ["level"], ["roll"]]

    lab = bank.label_item(make_item(dict(places, roll=places["save"]), {n: same for n in places}))
    assert lab["dropped"] == "all candidates duplicate" and lab["duplicates"] == [["save", "level", "roll"]]
    assert bank.usable([make_item(dict(places, roll=places["save"]), {n: same for n in places})]) == []

    unfinished = make_item(dict(places, roll=[6, 6, 7, None, 7]))
    assert bank.label_item(unfinished)["dropped"].startswith("unfinished")


def test_plans_map_to_the_nearest_candidate():
    st, cands = state(), bank.menu("3-2")
    plan = {"comp": None, "level_to": 5, "roll_floor": 999, "carry": None}
    assert bank.nearest_candidate(plan, st, cands) == "save"  # saves everything
    assert bank.nearest_candidate(dict(plan, level_to=6), st, cands) == "level"  # 20 gold of xp
    assert bank.nearest_candidate(dict(plan, roll_floor=10), st, cands) == "roll"
    assert bank.nearest_candidate(dict(plan, survival=6), st, cands) == "roll"  # survival: roll to 0
    assert bank.nearest_candidate(dict(plan, level_by={"level": 8, "by": "4-1"}), st, cands) == "level"
    assert bank.nearest_candidate(dict(plan, spend={"to": 20, "rounds": 2}), st, cands) == "roll"
    st41, cands41 = state(round=15, level=7, xp=10, gold=60), bank.menu("4-1")
    assert bank.nearest_candidate(dict(plan, level_to=8), st41, cands41) == "fast8"
    assert bank.plan_signature(dict(plan, roll_floor=0), state(gold=90)) == (0, 30)  # one round's cap


def test_scorecard_regret_within_share_and_game_clusters():
    a = make_item({"save": [2, 2, 3, 3, 4], "level": [5, 5, 6, 7, 6], "roll": [4, 4, 3, 4, 4]}, game="g#1", seed=1)
    b = make_item({"save": [5, 5, 6, 6, 6], "level": [5, 5, 5, 5, 5], "roll": [1, 1, 2, 2, 2]},
                  game="g#2", seed=2, point="4-1")
    c = make_item({"save": [3, 3, 3, 3, 3], "level": [4, 4, 5, 5, 5], "roll": [4, 4, 5, 5, 4]},
                  game="g#2", seed=2, point="3-2")
    items = bank.usable([a, b, c])
    choices = {a["id"]: {"choice": "roll"}, b["id"]: {"choice": "save"}, c["id"]: {"choice": "nonsense"}}
    card = bank.scorecard(items, choices, "test")
    rows = {r["id"]: r for r in card["rows"]}
    assert rows[a["id"]]["regret"] == pytest.approx(1 / 3) and rows[a["id"]]["within_ci"]
    assert rows[b["id"]]["regret"] == pytest.approx(4.0) and not rows[b["id"]]["within_ci"]
    assert rows[c["id"]]["valid"] is False and rows[c["id"]]["regret"] == pytest.approx(2.0)  # the worst
    allr = card["splits"]["all"]["all"]
    regrets = [1 / 3, 4.0, 2.0]
    assert allr["items"] == 3 and allr["games"] == 2 and allr["invalid"] == 1
    assert allr["regret"]["mean"] == pytest.approx(np.mean(regrets))
    m = np.mean(regrets)  # cluster-robust: per-game sums of deviations
    s1, s2 = regrets[0] - m, regrets[1] - m + regrets[2] - m
    se = math.sqrt((s1 ** 2 + s2 ** 2) * 2 / 1) / 3
    assert allr["regret"]["ci95"] == pytest.approx(12.706 * se)
    assert set(card["splits"]["all"]["points"]) == {"3-2", "4-1"}
    assert card["splits"]["all"]["points"]["4-1"]["items"] == 1
    oracle = bank.scorecard(items, {i["id"]: {"choice": i["label"]["best"]} for i in items}, "oracle")
    assert oracle["splits"]["all"]["all"]["regret"]["mean"] == 0
    diff = bank.paired(card, oracle)
    assert diff["mean"] == pytest.approx(np.mean(regrets))


def test_label_agents_and_baselines_choose_valid_candidates():
    item = bank.usable([make_item({"save": [2, 2, 3, 3, 4], "level": [5, 5, 6, 7, 6], "roll": [4, 4, 3, 4, 4]})])[0]
    view = {"id": item["id"], "candidates": item["candidates"], "labels": item["label"]}
    assert bank.make_agent("oracle").choose(view) == "save"
    assert bank.make_agent("worst").choose(view) == "level"
    assert bank.make_agent("always:roll").choose(view) == "roll"
    assert bank.make_agent("random").choose(view) == bank.make_agent("random").choose(view)  # by item id
    assert bank.make_agent("noisy100:always:roll").choose(view) in {"save", "level", "roll"}
    assert bank.make_agent("noisy0:always:roll").choose(view) == "roll"
    assert bank.make_agent("py:os.path:basename").choose("x/roll") == "roll"  # any callable view -> name
    with pytest.raises(ValueError):
        bank.make_agent("no-such-planner")


def test_dev_heldout_split_is_by_seed_and_about_70_30():
    seeds = range(10_000, 11_000)
    dev = sum(bank.split(s) == "dev" for s in seeds) / 1000
    assert 0.65 < dev < 0.75 and bank.split(9100) == bank.split(9100)


def test_recipes_record_the_simulator_settings(monkeypatch):
    monkeypatch.setenv("TFT_SIM", "realistic,rng_streams=shared")
    monkeypatch.setenv("TFT_RULES", "set18")
    monkeypatch.setenv("TFT_PICKERS", "0")
    st = bank.sim_settings()
    assert st == {"rules": "set18", "sim": "realistic,rng_streams=shared", "pickers": False,
                  "sim_options": {"pve_damage": True, "fortune_orbs": True, "carousel_fixes": True,
                                  "hide_next_opponent": True, "rng_streams": "shared", "max_actions_per_round": 60}}
    assert bank.sim_settings("default", "set4", True) == {"rules": "set4", "sim": "default", "sim_options": {},
                                                           "pickers": True}
    monkeypatch.delenv("TFT_SIM")
    monkeypatch.delenv("TFT_PICKERS")
    assert bank.sim_settings()["sim"] == "realistic" and bank.sim_settings()["pickers"]  # the defaults
    with pytest.raises(ValueError):
        bank.sim_settings("unreal")

    recipe = bank.make_recipe("stance@rule:7", 9100, "3-2", bank.sim_settings("realistic", "set4"))
    assert {k: recipe[k] for k in bank.SETTING_KEYS} == bank.sim_settings("realistic", "set4")
    assert bank.game_kwargs(recipe) == {"rules": "set4", "sim": "realistic", "pickers": True}
    stale = dict(recipe, sim_options={**recipe["sim_options"], "pve_damage": False})  # the profile changed since
    with pytest.raises(RuntimeError):
        bank.game_kwargs(stale)
    # a recipe from before the profiles (the first pilot): every fork option off, no carousel pickers
    legacy = {k: v for k, v in recipe.items() if k not in ("sim", "pickers")}
    legacy.update(sim_profile=None, sim_options={})
    assert bank.recipe_settings(legacy) == {"rules": "set4", "sim": "default", "sim_options": {}, "pickers": False}
    assert bank.game_kwargs(legacy) == {"rules": "set4", "sim": "default", "pickers": False}


def test_levels_follow_the_rules_profile():
    pytest.importorskip("Simulator")  # the profiles live in the simulator's rules.py (no game is played)
    s4 = state(round=21, level=8, xp=0, gold=100, rules="set4")
    s18 = dict(s4, rules="set18")
    assert bank.resolve(bank.CANDIDATES["cap_out"], s4)["xp_buys"] == 20  # 80 xp from 8 to 9
    assert bank.resolve(bank.CANDIDATES["cap_out"], s18)["xp_buys"] == 17  # 68 under set18
    assert bank.max_level(s4) == 9 and bank.max_level(s18) == 10
    at9 = dict(s4, level=9)
    assert "xp_buys" not in bank.resolve(bank.CANDIDATES["fast8"], at9)  # nothing above 9 under set4
    assert bank.resolve(bank.CANDIDATES["fast8"], dict(at9, rules="set18"))["xp_buys"] == 17  # 68 xp, 9 -> 10


# --------------------------------------------------------------------------- on the simulator

def test_recipe_rebuild_matches_the_fingerprint():
    pytest.importorskip("Simulator")
    recipe = bank.make_recipe("mimic@rule:7", 7100, "2-2")  # a short game: 4 planning phases
    assert recipe["lobby"][recipe["hero_seat"]] == "hero=mimic" and recipe["round"] == 4
    game = bank.play_recipe(recipe)
    assert bank.hero_alive(game, recipe)
    fp = bank.fingerprint(game, recipe["hero_seat"])
    expect = bank.fingerprint_hash(fp)
    assert fp["round"] == 4 and fp["hero"]["hp"] > 0 and len(fp["lobby_hp"]) == 8
    assert bank.fingerprint_hash(bank.fingerprint(bank.rebuild(recipe, expect), recipe["hero_seat"])) == expect
    with pytest.raises(RuntimeError):
        bank.rebuild(recipe, "0" * 16)
    other = bank.make_recipe("mimic@rule:7", 7101, "2-2")
    assert bank.fingerprint_hash(bank.fingerprint(bank.play_recipe(other), other["hero_seat"])) != expect
    # another process (same PYTHONHASHSEED) rebuilds the same state
    code = ("import json, sys; from tfteval import bank; r = json.loads(sys.argv[1]); "
            "g = bank.play_recipe(r); print(bank.fingerprint_hash(bank.fingerprint(g, r['hero_seat'])))")
    out = subprocess.run([sys.executable, "-c", code, json.dumps(recipe)], capture_output=True, text=True,
                         env={**os.environ, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0")}, cwd=ROOT,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == expect


def test_hero_view_is_the_planners_state():
    pytest.importorskip("Simulator")
    recipe = bank.make_recipe("stance@rule:7", 7100, "2-2")
    game = bank.play_recipe(recipe)
    st, comps, comp_now = bank.hero_view(game, recipe["hero_seat"])
    assert st["round"] == 4 and st["gold"] == bank.fingerprint(game, recipe["hero_seat"])["hero"]["gold"]
    assert "opponents" in st and "next_from" in st and comps and comp_now is None
    assert bank.make_agent("stance").choose({"id": "x", "state": st, "comps": comps, "comp_now": comp_now,
                                             "candidates": bank.menu("3-2"),
                                             "hero_policy": game.seat_policies[recipe["hero_seat"]]}) \
        in {"save", "level", "roll"}


def test_a_later_line_replaces_an_item(tmp_path):
    a, b = make_item({"save": [1] * 5, "roll": [2] * 5}, game="g#1"), make_item({"save": [1] * 5}, game="g#2")
    longer = {**a, "kl": 5, "note": "extended"}
    path = tmp_path / "bank.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in (a, b, longer)))
    items = bank.load_items(path)
    assert [it["id"] for it in items] == [a["id"], b["id"]] and items[0]["note"] == "extended"

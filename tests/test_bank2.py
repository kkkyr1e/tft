"""Decision bank v2 (tfteval/bank2.py): triggers, candidates, racing, posterior, cross-fit anchors,
scoring, the rule check, and v1 scoring left unchanged.

Everything but the tests marked "on the simulator" is pure Python on synthetic data."""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfteval import bank, bank2, stages  # noqa: E402
from tfteval.planner import compile_knobs  # noqa: E402

L = stages.parse_label


def unit(name, star=1, cost=1):
    return {"name": name, "star": star, "cost": cost}


def state(**kw):
    """A describe()-like state: board / bench lists of unit dicts and the scalar fields."""
    s = {"round": L("3-2"), "stage": "3-2", "hp": 80, "gold": 40, "level": 6, "xp": 4, "streak": 0,
         "losses_to_death": 8, "board": [unit("a"), unit("b"), unit("c", 2)], "bench": [unit("a"), unit("b")]}
    s.update(kw)
    return s


# --------------------------------------------------------------------------- triggers

def test_trigger_features_count_pairs_and_split_the_streak():
    f = bank2.trigger_features(state(streak=-3))
    assert f["pairs"] == 2 and f["lose_streak"] == 3 and f["win_streak"] == 0 and f["streak"] == -3
    assert bank2.trigger_features(state(bench=[unit("a"), unit("a")]))["pairs"] == 0  # three copies: not a pair
    assert bank2.trigger_features(state(bench=[unit("c", 2)]))["pairs"] == 0  # 2-stars are not pairs
    assert bank2.trigger_features(state(streak=4))["win_streak"] == 4


@pytest.mark.parametrize("stratum,fire,miss", [
    ("pairs3", dict(gold=40), dict(gold=51)),
    ("pairs3", dict(gold=30), dict(bench=[unit("a")])),  # one pair only
    ("streak4", dict(round=L("4-1"), gold=55, streak=2, level=7), dict(round=L("4-1"), gold=49, streak=3, level=7)),
    ("streak4", dict(round=L("4-2"), gold=60, streak=5, level=6), dict(round=L("4-1"), gold=60, streak=5, level=8)),
    ("streak4", dict(round=L("4-1"), gold=60, streak=2, level=7), dict(round=L("4-1"), gold=60, streak=-2, level=7)),
    ("losing3", dict(round=L("3-1"), streak=-2, hp=60), dict(round=L("3-1"), streak=-1, hp=90)),
    ("losing3", dict(round=L("3-3"), streak=-5, hp=88), dict(round=L("3-3"), streak=-5, hp=59)),
    ("lowhp4", dict(round=L("4-5"), hp=50, gold=40), dict(round=L("4-5"), hp=51, gold=80)),
    ("lowhp4", dict(round=L("5-1"), hp=12, gold=45), dict(round=L("5-1"), hp=12, gold=39)),
    ("ref41", dict(round=L("4-1")), None),
])
def test_triggers_fire_and_do_not(stratum, fire, miss):
    when = bank2.STRATA[stratum]["when"]
    assert bank2.fires(when, bank2.trigger_features(state(**fire)))
    if miss is not None:
        assert not bank2.fires(when, bank2.trigger_features(state(**miss)))


def test_first_fire_takes_the_first_round_inside_the_window():
    rows = [bank2.trigger_features(state(round=r, gold=g)) for r, g in ((9, 40), (10, 60), (11, 45), (12, 40))]
    assert bank2.first_fire(rows, "pairs3") == 11  # 3-1 fires but is outside 3-2..3-5; 3-2 has 60 gold
    assert bank2.first_fire(rows[:3], "pairs3") == 11 and bank2.first_fire(rows[:2], "pairs3") is None
    assert bank2.window_rounds("lowhp4") == list(range(L("4-1"), L("5-1") + 1))
    assert bank2.window_rounds("ref41") == [15] and bank2.window_rounds("losing3") == [9, 10, 11]
    with pytest.raises(ValueError):
        bank2.fires({"mana_min": 3}, rows[0])


def test_strata_menus_and_specs():
    for name, s in bank2.STRATA.items():
        assert 3 <= len(s["menu"]) <= 4 and s["rounds"] == 2
        assert all(c in bank2.CANDIDATES and bank2.CANDIDATES[c]["desc"] for c in s["menu"])
        assert bank2.trigger_spec(name)["menu"] == list(s["menu"])
        assert bank2.trigger_spec(name)["sample_mod"] == (bank2.REF_SAMPLE_MOD if name == "ref41" else 1)
    assert bank2.STRATA["ref41"]["menu"] == bank2.STRATA["lowhp4"]["menu"]
    assert bank2.STRATA["lowhp4"]["menu"] == ("roll_all", "roll30", "level_roll10", "save")
    assert bank2.CANDIDATES["save"] == bank.CANDIDATES["save"]


def test_ref41_is_sampled_every_third_seed():
    assert bank2.REF_SAMPLE_MOD == 3
    assert [s for s in range(9400, 9410) if bank2.sampled("ref41", s)] == [9402, 9405, 9408]
    assert all(bank2.sampled(s, 9401) for s in bank2.STRATA if s != "ref41")
    with pytest.raises(ValueError):
        bank2.build_item("stance@rule:7", 9401, "ref41")


# --------------------------------------------------------------------------- candidates

BASE_PLAN = {"comp": "mage", "level_to": 6, "roll_floor": 999, "carry": None, "field_comp": True}


def knobs(name, st, anchor=None, plan=BASE_PLAN):
    fields = bank.resolve(bank2.CANDIDATES[name], st, anchor)
    return compile_knobs(plan if fields is None else bank.override(plan, fields), st), fields


def test_v2_candidates_compile_to_their_knobs():
    st = state(level=6, xp=4, gold=40)  # 6 -> 7 needs 32 xp: 8 buys
    k, _ = knobs("roll10", st)
    assert k["roll_floor"] == 10 and k["xp_buys"] == 0 and not k["fodder"]
    k, _ = knobs("level7", st)
    assert k["xp_buys"] == 8 and k["roll_floor"] == 999  # keep 0: all the gold toward 7
    k, _ = knobs("level7", state(level=7, xp=0, gold=40))  # "7+" at 7: toward 8 with all 40 gold
    assert k["xp_buys"] == 10
    k, _ = knobs("streak", state(round=L("3-2"), level=5, xp=0, gold=62))
    assert k["fodder"] and k["roll_floor"] == 999 and k["xp_buys"] == 3  # the gold above 50 toward the curve (6)
    assert knobs("streak", state(round=L("3-1"), level=5, xp=0, gold=62))[0]["xp_buys"] == 0  # no curve before 3-2
    k, _ = knobs("level_roll10", state(round=L("4-1"), level=7, xp=20, gold=50))
    assert k["xp_buys"] == 9 and k["roll_floor"] == 10  # 36 xp to 8 with the gold above 10, then roll to 10
    k, _ = knobs("roll_all", state(round=L("4-1"), level=7, gold=50))
    assert k["roll_floor"] == 0 and k["xp_buys"] == 0
    st8 = state(round=L("4-1"), level=7, xp=20, gold=60)
    k, fields = knobs("level8", st8)
    assert k["xp_buys"] == 9 and k["roll_floor"] == 999
    k, fields = knobs("level8", state(round=L("4-2"), level=8, xp=0, gold=30), anchor=7)
    assert fields is None and k == compile_knobs(BASE_PLAN, state(round=L("4-2"), level=8, xp=0, gold=30))
    assert bank.resolve(bank.CANDIDATES["fast8"], state(level=8, xp=0, gold=60, round=18))["xp_buys"] == 12  # v1 "8+": toward 9, gold above 10


class Base:
    name = "base"

    def plan(self, st, comps, comp_now):
        return {"comp": "mage", "level_to": st["level"], "roll_floor": 40, "carry": None}


def test_commit_planner_hands_a_reached_level8_round_to_the_base():
    spec = {"name": "level8", **{k: v for k, v in bank2.CANDIDATES["level8"].items() if k != "desc"}}
    planner = bank.CommitPlanner(Base(), spec, rounds=2)
    first = planner.plan(state(round=15, level=7, xp=20, gold=60), {}, None)
    assert first["xp_buys"] == 9 and first["roll_floor"] == 999 and first["commit"] == "level8"
    second = planner.plan(state(round=16, level=8, xp=2, gold=20), {}, None)
    assert second == Base().plan(state(round=16, level=8), {}, None)  # at 8: the base's own round
    assert planner.applied == [15]


def test_nearest_candidate_v2_maps_plans_with_fodder():
    st = state(round=L("3-1"), level=5, xp=0, gold=40)
    cands = bank2.menu("losing3")
    plan = {"comp": None, "level_to": 5, "roll_floor": 999, "carry": None}
    assert bank2.nearest_candidate(plan, st, cands) == "streak"  # saves: closest to keeping the streak
    assert bank2.nearest_candidate(dict(plan, fodder=True), st, cands) == "streak"
    assert bank2.nearest_candidate(dict(plan, roll_floor=20), st, cands) == "roll20"
    assert bank2.nearest_candidate(dict(plan, roll_floor=10, xp_buys=5), st, cands) == "level_roll10"
    st41 = state(round=L("4-1"), level=7, xp=20, gold=60)
    cands41 = bank2.menu("streak4")
    assert bank2.nearest_candidate(dict(plan, level_to=8), st41, cands41) == "level8"
    assert bank2.nearest_candidate(dict(plan, roll_floor=30), st41, cands41) == "roll30"
    assert bank2.nearest_candidate(plan, st41, cands41) == "save"
    cands4 = bank2.menu("lowhp4")
    st45 = state(round=L("4-5"), level=7, gold=40)
    assert bank2.nearest_candidate(dict(plan, survival=9), st45, cands4) == "roll_all"
    assert bank2.nearest_candidate(dict(plan, roll_floor=30), st45, cands4) == "roll30"


def test_agents_on_v2_menus():
    view = {"id": "x", "candidates": bank2.menu("lowhp4")}
    assert bank2.make_agent("always:roll30").choose(view) == "roll30"
    assert bank2.make_agent("always:level8").choose(view) == "roll_all"  # not on the menu: the first one
    assert bank2.make_agent("random").choose(view) in bank2.STRATA["lowhp4"]["menu"]
    assert isinstance(bank2.make_agent("noisy50:stance").inner, bank2.PlannerAgent)
    assert isinstance(bank2.make_agent("mimic"), bank2.PlannerAgent)
    for bad in ("oracle", "worst", "xfit:oracle"):
        with pytest.raises(ValueError):
            bank2.make_agent(bad)


# --------------------------------------------------------------------------- racing

def test_eliminate_uses_paired_differences_with_a_variance_floor():
    places = {"a": {k: 3 for k in range(8)}, "b": {k: 4 for k in range(8)}, "c": {k: 3 + (k % 2) * 4 for k in range(8)}}
    leader, stats, out = bank2.eliminate(places, ["a", "b", "c"], list(range(8)), 2.0, 1.0)
    # b is 1 place behind on every k: SD 0 would eliminate anything; the floor gives SE 1/sqrt(8) = 0.354
    assert leader == "a" and stats["vs_leader"]["b"]["se"] == pytest.approx(1 / math.sqrt(8))
    # c: mean +2, SD 2.14 -> SE 0.756; 2 > 1.51: out as well
    assert stats["vs_leader"]["c"]["diff"] == 2 and stats["vs_leader"]["c"]["se"] == pytest.approx(2.138 / math.sqrt(8), abs=1e-3)
    assert out == ["b", "c"]
    noisy = {"a": {k: 3 for k in range(8)}, "b": {k: 6 if k % 4 == 3 else 3 for k in range(8)}}  # +0.75, SE 0.49
    assert bank2.eliminate(noisy, ["a", "b"], list(range(8)), 2.0, 1.0)[2] == []
    close = {"a": {k: 3 for k in range(2)}, "b": {k: 3.5 for k in range(2)}}
    assert bank2.eliminate(close, ["a", "b"], [0, 1], 2.0, 1.0)[2] == []  # 0.5 < 2 * 1/sqrt(2)
    tie = {"a": {0: 4, 1: 4}, "b": {0: 4, 1: 4}}
    assert bank2.eliminate(tie, ["b", "a"], [0, 1], 2.0, 1.0, order=["a", "b"])[0] == "a"  # ties: menu order


def fake_play(true, log=None):
    """Outcomes that depend only on (candidate, k), like branches of one saved state."""
    shock = np.random.default_rng(7).normal(0, 1, 200)

    def play(c, k):
        if log is not None:
            log.append((c, k))
        own = np.random.default_rng([ord(ch) for ch in c] + [k]).normal(0, 0.3)
        return {"cand": c, "k": k, "place": float(true[c] + shock[k] + own), "actions": {"15": f"{c}{k}"},
                "seconds": 1.0}
    return play


def test_race_budget_elimination_and_stopping():
    cfg = bank2.racing_config()
    assert cfg == bank2.RACING and cfg["budget_per_cand"] == 16
    # four equal candidates: nobody is eliminated, the budget (4 x 16 = 64) stops the race at 16 each
    even = {c: 4.0 for c in "abcd"}
    r = bank2.race(list(even), fake_play(even), cfg)
    assert r["stop"] == "budget spent" and set(r["survivors"]) == set("abcd") and r["merged"] == {}
    assert set(r["n"].values()) == {16}
    assert [b["k"] for b in r["branches"][:8]] == [0, 0, 0, 0, 1, 1, 1, 1]  # k-major
    # four candidates, two far behind: out at the first look, the two others reach K_MIN (8 x 4 + 2 x 16 = 64)
    four = {"a": 3.0, "b": 3.1, "c": 7.0, "d": 7.5}
    r = bank2.race(list(four), fake_play(four), cfg)
    assert r["looks"][0]["eliminated"] == ["c", "d"] and r["n"] == {"a": 24, "b": 24, "c": 8, "d": 8}
    assert r["stop"] == "k_min, budget spent" and sum(r["n"].values()) == 64
    # three candidates (48 branches), one far behind: the two others get 20 each
    three = {"a": 3.0, "b": 3.1, "c": 7.0}
    r = bank2.race(list(three), fake_play(three), cfg)
    assert r["looks"][0]["eliminated"] == ["c"] and r["n"] == {"a": 20, "b": 20, "c": 8}
    assert r["stop"] == "budget spent" and sum(r["n"].values()) == 48
    # one clearly best: everyone else out, the race stops with one left
    solo = {"a": 1.0, "b": 6.0, "c": 6.5}
    r = bank2.race(list(solo), fake_play(solo), cfg)
    assert r["stop"] == "one left" and r["survivors"] == ["a"] and r["n"] == {"a": 8, "b": 8, "c": 8}
    # a larger budget lets two survivors reach K_MAX
    r = bank2.race(list(three), fake_play(three), bank2.racing_config(budget_per_cand=100))
    assert r["stop"] == "k_max" and r["n"] == {"a": 32, "b": 32, "c": 8}
    with pytest.raises(ValueError):
        bank2.racing_config(k_min=40)


def test_race_drops_all_duplicates_and_replays():
    same = {c: 4.0 for c in "abc"}

    def dup_play(c, k):
        return {"cand": c, "k": k, "place": 4, "actions": {"15": f"x{k}"}, "seconds": 0.0}
    r = bank2.race(list(same), dup_play, bank2.RACING)
    assert r["stop"] == "all candidates duplicate" and r["n"] == {"a": 8, "b": 8, "c": 8}

    three = {"a": 3.0, "b": 3.1, "c": 7.0}
    small = bank2.racing_config(k_start=2, k_step=2, k_min=4, k_max=4, budget_per_cand=6)
    r = bank2.race(list(three), fake_play(three), small)
    item = {"id": "x", "candidates": [{"name": c} for c in three], "branches": r["branches"], "racing": small}
    again = bank2.replay_race(item)
    assert again["looks"] == r["looks"] and again["n"] == r["n"] and again["stop"] == r["stop"]
    with pytest.raises(KeyError):  # asks for branches never played
        bank2.replay_race(item, bank2.RACING)
    # re-racing with a cache plays only the missing branches and gives what a fresh race gives
    log = []
    cache = {}
    for b in r["branches"]:
        cache.setdefault(b["cand"], {})[b["k"]] = b
    fresh = bank2.race(list(three), fake_play(three), bank2.RACING)
    again = bank2.race(list(three), fake_play(three, log=log), bank2.RACING, cache)
    assert again["n"] == fresh["n"] and [b["place"] for b in again["branches"]] == [b["place"] for b in fresh["branches"]]
    assert not set(log) & {(b["cand"], b["k"]) for b in r["branches"]}


def merged_play(true, same):
    """fake_play where candidate `later` plays exactly like `earlier` (same: later -> earlier)."""
    inner = fake_play(true)

    def play(c, k):
        b = inner(same.get(c, c), k)
        return {**b, "cand": c}
    return play


def test_race_merges_duplicates_at_the_first_look_and_labels_share():
    true = {"a": 4.0, "b": 4.0, "c": 4.0, "d": 4.0}
    play = merged_play(true, {"b": "a"})
    r = bank2.race(list(true), play, bank2.RACING)
    # b repeats a: merged at the first look, no more branches; the budget is 3 x 16 for a, c, d
    assert r["merged"] == {"b": "a"} and r["looks"][0]["merged"] == {"b": "a"}
    assert r["looks"][0]["active"] == ["a", "c", "d"] and "merged" not in r["looks"][1]
    assert r["n"] == {"a": 16, "b": 8, "c": 16, "d": 16} and r["stop"] == "budget spent"
    item = make_item({c: {} for c in true})
    item.update(branches=r["branches"], racing=dict(bank2.RACING),
                race={k: r[k] for k in ("looks", "stop", "survivors", "n", "merged")})
    again = bank2.replay_race(item)
    assert again["looks"] == r["looks"] and again["merged"] == r["merged"]
    # one decision: the groups, P, regret and the cross-fit picks treat b as a
    assert bank2.candidate_groups(item) == [["a", "b"], ["c"], ["d"]]
    assert bank.duplicate_groups(item["branches"], list(true)) != bank2.candidate_groups(item)  # b lacks a's ks
    labels, _ = bank2.label_items(bank2.usable([item]), draws=500)
    lab = labels[item["id"]]
    assert lab["P"]["b"] == lab["P"]["a"] and lab["regret"]["b"] == lab["regret"]["a"] and lab["n"]["b"] == 16
    assert lab["merged"] == {"b": "a"} and sum(lab["P"][g[0]] for g in lab["duplicates"]) == pytest.approx(1.0)
    assert "b" not in lab["xfit"]["oracle"]["picks"] + lab["xfit"]["worst"]["picks"]
    rows = bank2.score_rows([item], labels, {item["id"]: {"choice": "b"}})
    assert rows[0]["regret"] == lab["regret"]["a"] and rows[0]["hit"] == float("a" in lab["best_group"])
    # a merge that leaves two distinct candidates still races them
    r = bank2.race(["a", "b", "c"], merged_play({"a": 3.0, "b": 3.0, "c": 3.2}, {"b": "a"}), bank2.RACING)
    assert r["merged"] == {"b": "a"} and r["n"]["b"] == 8 and r["n"]["a"] == r["n"]["c"] == 16


# --------------------------------------------------------------------------- labels

def make_item(places, stratum="lowhp4", game="g#1", seed=1, actions=None, state_kw=None):
    """places: candidate -> {k: place} (or a list for k = 0..)."""
    names = list(places)
    branches = []
    for c in names:
        ks = places[c] if isinstance(places[c], dict) else dict(enumerate(places[c]))
        for k, p in ks.items():
            acts = (actions or {}).get(c, {}).get(k, {"15": f"{c}{k}"})
            branches.append({"cand": c, "k": k, "place": p, "actions": acts, "seconds": 1.0})
    return {"id": f"{game}@{stratum}", "game": game, "seed": seed, "stratum": stratum, "point": "4-1",
            "split": bank.split(seed), "candidates": [{"name": c, "desc": c, "rounds": 2, "spec": {}} for c in names],
            "branches": branches, "dropped": None, "public_state": state(**(state_kw or {}))}


def test_crossfit_oracle_and_worst_by_hand():
    places = {"a": {0: 2, 1: 6, 2: 2, 3: 6}, "b": {0: 4, 1: 4, 2: 4, 3: 4}, "c": {0: 7, 1: 7}}
    # odd ks {1, 3}: b best (4 vs a 6, c 7) -> scored on even {0, 2} against best a: 4 - 2 = +2
    # even ks {0, 2}: a best -> scored on odd {1, 3} against a: 0
    xf = bank2.crossfit(places, ["a", "b", "c"], "a", "oracle")
    assert xf["picks"] == ["b", "a"] and xf["regret"] == pytest.approx(1.0)
    # worst: c on odd (7), scored on even k 0 only (c has no k 2): 7 - 2 = 5; on even: c again, odd k 1: 7 - 6 = 1
    xw = bank2.crossfit(places, ["a", "b", "c"], "a", "worst")
    assert xw["picks"] == ["c", "c"] and xw["regret"] == pytest.approx(3.0)


def test_posterior_clear_unclear_and_duplicates():
    rng = np.random.default_rng(0)
    fit = {"names": ["a", "b", "c"], "alpha": {"a": 0.0, "b": 0.0, "c": 0.0}, "alpha_cov": np.zeros((3, 3)).tolist(),
           "tau2": 1.0, "sigma2": {}, "pooled_sd_d": 1.7}
    shock = rng.normal(0, 1, 32)
    clear = make_item({"a": [2 + shock[k] + rng.normal(0, 1) for k in range(24)],
                       "b": [5 + shock[k] + rng.normal(0, 1) for k in range(24)],
                       "c": [6 + shock[k] + rng.normal(0, 1) for k in range(8)]})
    post = bank2.posterior(clear, fit, 4000)
    assert post["best"] == "a" and post["pmax"] > 0.99 and sum(post["P"].values()) == pytest.approx(1.0)
    same = make_item({c: [4 + shock[k] + rng.normal(0, 1.2) for k in range(24)] for c in "abc"})
    p = bank2.posterior(same, {**fit, "tau2": 0.01}, 4000)
    assert p["pmax"] < 0.6  # nothing in the data or the prior separates them
    # duplicates share their group's probability; the group's representative is the first in menu order
    dup_actions = {"a": {k: {"15": f"x{k}"} for k in range(16)}, "b": {k: {"15": f"x{k}"} for k in range(16)}}
    pl = [3 + shock[k] for k in range(16)]
    dup = make_item({"a": pl, "b": list(pl), "c": [6 + shock[k] for k in range(16)]}, actions=dup_actions)
    p = bank2.posterior(dup, fit, 4000)
    assert p["duplicates"] == [["a", "b"], ["c"]] and p["P"]["a"] == p["P"]["b"] and p["best_group"] == ["a", "b"]
    assert p["P"]["a"] + p["P"]["c"] == pytest.approx(1.0)
    # the prior mean pulls a weak item toward the stratum's better candidate
    weak = make_item({c: [4 + shock[k] + rng.normal(0, 1.5) for k in range(8)] for c in "abc"})
    tilt = {**fit, "alpha": {"a": 0.0, "b": -1.0, "c": 1.0}, "tau2": 0.04}
    assert bank2.posterior(weak, tilt, 4000)["best"] == "b"


def synthetic_stratum(n_games=60, tau=0.5, alpha=None, noise=1.2, stratum="lowhp4", seed=0, rule=None):
    """Items of one stratum with known true effects; `rule(features, rng) -> {cand: shift}` adds a
    feature-dependent effect. Returns (items, truth: id -> true best)."""
    rng = np.random.default_rng(seed)
    names = list(bank2.STRATA[stratum]["menu"])
    alpha = alpha or {c: 0.0 for c in names}
    items, truth = [], {}
    for g in range(n_games):
        hp = int(rng.integers(10, 51))
        true = {c: alpha[c] + rng.normal(0, tau) for c in names}
        if rule:
            for c, v in rule({"hp": hp}).items():
                true[c] += v
        shock = rng.normal(0, 1, 40)
        ns = {c: (24 if i < 2 else 8) for i, c in enumerate(names)}
        places = {c: {k: float(4.5 + true[c] + shock[k] + rng.normal(0, noise)) for k in range(ns[c])} for c in names}
        it = make_item(places, stratum, game=f"g#{g}", seed=g, state_kw={"round": 15, "hp": hp, "gold": 45})
        items.append(it)
        truth[it["id"]] = min(names, key=lambda c: true[c])
    return items, truth


def test_stratum_fit_recovers_tau_and_alpha_and_the_posterior_is_calibrated():
    alpha = {"roll_all": 0.3, "roll30": -0.3, "level_roll10": 0.0, "save": 0.0}
    items, truth = synthetic_stratum(n_games=300, tau=0.6, alpha=alpha)
    labels, fits = bank2.label_items(items, draws=2000)
    fit = fits["lowhp4"]
    assert fit["tau_source"] == "moments" and abs(math.sqrt(fit["tau2"]) - 0.6) < 0.12
    assert fit["alpha"]["roll30"] < fit["alpha"]["roll_all"] - 0.3
    assert fit["pooled_sd_d"] == pytest.approx(1.2 * math.sqrt(2), abs=0.15)
    clear = [i for i, lab in labels.items() if lab["clear"]]
    hits = np.mean([labels[i]["best"] == truth[i] for i in clear])
    assert len(clear) > 30 and hits > 0.85  # clear items are right about as often as P >= 0.85 says
    pmax = np.mean([labels[i]["pmax"] for i in truth])  # calibration of the posterior-best pick
    assert abs(pmax - np.mean([labels[i]["best"] == truth[i] for i in truth])) < 0.06
    # a tiny stratum falls back to v1's tau and no candidate effects
    few_labels, few = bank2.label_items(items[:3], draws=500)
    assert few["lowhp4"]["tau_source"] == "fallback" and few["lowhp4"]["tau2"] == pytest.approx(bank2.TAU_FALLBACK ** 2)
    assert set(few["lowhp4"]["alpha"].values()) == {0.0}
    zero = bank2.fit_stratum(items, prior="zero")
    assert set(zero["alpha"].values()) == {0.0} and zero["tau2"] > fit["tau2"]  # the effects go into tau


def test_scoring_regret_ceiling_and_equal_weights():
    a, _ = synthetic_stratum(n_games=12, stratum="lowhp4", seed=1)
    b, _ = synthetic_stratum(n_games=8, stratum="pairs3", seed=2)
    items = bank2.usable(a + b)
    labels, _ = bank2.label_items(items, draws=1000)
    best = {it["id"]: {"choice": labels[it["id"]]["best"]} for it in items}
    rows = bank2.score_rows(items, labels, best)
    card = bank2.scorecard(rows, "posterior-best")
    for s, e in card["splits"]["all"]["strata"].items():
        assert e["regret"]["mean"] == 0 and e["accuracy"] in (None, pytest.approx(1.0))
    first = {it["id"]: {"choice": it["candidates"][0]["name"]} for it in items}
    first[items[0]["id"]] = {"choice": "nonsense"}
    rows = bank2.score_rows(items, labels, first)
    lab0 = labels[items[0]["id"]]
    assert rows[0]["valid"] is False and rows[0]["regret"] == max(lab0["regret"].values()) and rows[0]["p_choice"] == 0
    assert rows[1]["regret"] == pytest.approx(labels[items[1]["id"]]["regret"][items[1]["candidates"][0]["name"]])
    card_first = bank2.scorecard(rows, "first")
    eq = card_first["splits"]["all"]["equal"]["regret"]
    per = card_first["splits"]["all"]["strata"]
    assert eq["mean"] == pytest.approx((per["lowhp4"]["regret"]["mean"] + per["pairs3"]["regret"]["mean"]) / 2)
    diff = bank2.paired(card_first, card)
    assert diff["equal"]["mean"] == pytest.approx(eq["mean"])
    xo = bank2.crossfit_rows(items, labels, "oracle")
    assert len(xo) == len(items) and all(r["valid"] for r in xo)


def test_stratified_ci_reduces_to_the_clustered_ci_and_weights_strata_equally():
    vals, games = [1.0, 2.0, 4.0, 3.0, 0.0], ["g1", "g1", "g2", "g3", "g3"]
    one = bank2.stratified_mean_ci(vals, games, ["s"] * 5)
    ref = bank.clustered_mean_ci(vals, games)
    assert one["mean"] == pytest.approx(ref["mean"]) and one["ci95"] == pytest.approx(ref["ci95"])
    two = bank2.stratified_mean_ci(vals, ["g1", "g1", "g2", "g3", "g1"], ["s", "s", "s", "t", "t"])
    assert two["mean"] == pytest.approx((7 / 3 + 1.5) / 2)
    # by hand: contributions (x - m_s) / (2 n_s) summed per game (g1 has items in both strata)
    c = {"g1": (1 - 7 / 3) / 6 + (2 - 7 / 3) / 6 + (0 - 1.5) / 4, "g2": (4 - 7 / 3) / 6, "g3": (3 - 1.5) / 4}
    se = math.sqrt(sum(v * v for v in c.values()) * 3 / 2)
    assert two["ci95"] == pytest.approx(bank.t975(2) * se)
    # a stratum with one source game: its variance is unknown, so no interval (the formula would give 0)
    lone = bank2.stratified_mean_ci([1.0, 2.0, 4.0], ["g1", "g2", "g3"], ["s", "s", "t"])
    assert lone["mean"] == pytest.approx((1.5 + 4.0) / 2) and lone["ci95"] is None
    assert bank2.fmt_ci(lone) == "+2.75 (no CI)" and bank2.fmt_ci({"mean": 1.0, "ci95": None, "games": 1}) == "+1.00 (1 game)"


# --------------------------------------------------------------------------- rule check

def test_rule_check_finds_a_single_feature_rule():
    # roll_all is 1.5 places better below 30 HP, save 1.5 better above: one HP threshold solves the stratum
    items, truth = synthetic_stratum(n_games=80, tau=0.1, noise=0.8, seed=4,
                                     rule=lambda f: {"roll_all": -1.5} if f["hp"] <= 30 else {"save": -1.5})
    labels, _ = bank2.label_items(items, draws=1000)
    rep = bank2.rule_check(items, labels)["lowhp4"]
    thr = rep["threshold"]["hp"]
    assert thr["rule_on_all"]["a"] == "roll_all" and thr["rule_on_all"]["b"] == "save"
    assert 25 <= thr["rule_on_all"]["t"] <= 35
    assert thr["regret"]["mean"] < rep["best_fixed"]["regret"]["mean"] - 0.5
    assert rep["fixed"]["roll30"]["regret"]["mean"] > thr["regret"]["mean"]
    assert rep["headroom"] < bank2.RULE_HEADROOM and rep["rule_stratum"]
    assert rep["tree"]["regret"]["mean"] < rep["best_fixed"]["regret"]["mean"]
    assert bank2.apply_rule(rep["tree"]["rule_on_all"], {"hp": 12, "gold": 45, "pairs": 2, "streak": 0, "level": 6,
                                                         "losses_to_death": 8}) == "roll_all"
    # gold carries no information here: its cross-fitted rule is no better than the best fixed one
    assert rep["threshold"]["gold"]["regret"]["mean"] >= rep["best_fixed"]["regret"]["mean"] - 0.1


def test_game_folds_keep_games_whole():
    folds = bank2.game_folds([f"g{i}" for i in range(12)] * 2, 5)
    assert set(folds.values()) == set(range(5)) and len(folds) == 12
    assert bank2.game_folds(["a", "b"], 5) in ({"a": 0, "b": 1}, {"a": 1, "b": 0})


# --------------------------------------------------------------------------- v1 unchanged

V1_BANK = ROOT / "results" / "bank" / "realistic_set4.jsonl"
V1_SANITY = ROOT / "results" / "bank" / "realistic_set4.sanity.json"


@pytest.mark.skipif(not V1_BANK.exists(), reason="no v1 bank")
def test_v1_labels_are_unchanged():
    items = bank.load_items(V1_BANK)
    good = bank.usable(items)
    assert len(good) == 172
    stored = {it["id"]: it["label"] for it in items if it.get("label")}
    for it in good:
        assert it["label"] == stored[it["id"]]


@pytest.mark.skipif(not V1_SANITY.exists(), reason="no v1 sanity scorecard")
def test_v1_scoring_is_unchanged():
    pytest.importorskip("Simulator")  # the stance planner scores boards with the rule bot's tables
    sanity = json.loads(V1_SANITY.read_text())
    assert sanity["rebuilt"] is False  # it was made with --stored, so this replays it exactly
    good = bank.usable(bank.load_items(V1_BANK), sanity["sd_floor"])
    agents = [bank.make_agent(n) for n in sanity["cards"]]
    choices = bank.choose_all(good, agents, rebuild_states=False)
    for agent in agents:
        card = bank.scorecard(good, choices[agent.name], agent.name, sanity["bank"])
        old = sanity["cards"][agent.name]
        assert [(r["id"], r["choice"], r["regret"]) for r in card["rows"]] == \
               [(r["id"], r["choice"], r["regret"]) for r in old["rows"]], agent.name
        assert card["splits"]["all"]["all"]["regret"] == old["splits"]["all"]["all"]["regret"]


# --------------------------------------------------------------------------- on the simulator

def test_stepping_round_by_round_rebuilds_the_same_state():
    """build_item finds the trigger round by playing round by round and reading the hero's state in
    between; the recipe then rebuilds the state with play_to in one go. Both must be the same game."""
    pytest.importorskip("Simulator")
    recipe = bank.make_recipe("stance@rule:7", 7100, "2-1")
    game = bank.new_game(recipe)
    seat = recipe["hero_seat"]
    rows = []
    for rnd in range(L("2-1"), L("2-3") + 1):
        game.run(until_round=rnd)
        st, _, _ = bank.hero_view(game, seat)
        rows.append(bank2.trigger_features(st))
    assert [r["round"] for r in rows] == [3, 4, 5]
    stepped = bank.fingerprint_hash(bank.fingerprint(game, seat))
    once = bank.play_recipe(dict(recipe, round=L("2-3")))
    assert bank.fingerprint_hash(bank.fingerprint(once, seat)) == stepped


PILOT = ROOT / "results" / "bank2" / "pilot.jsonl"


@pytest.mark.skipif(not PILOT.exists(), reason="no bank v2 pilot")
def test_pilot_items_replay_their_races_and_label():
    items = bank.load_items(PILOT)
    raced = [it for it in items if it.get("race") and not it.get("copied_from")]
    assert raced
    for it in raced:
        again = bank2.replay_race(it)
        assert again["looks"] == it["race"]["looks"] and again["n"] == it["race"]["n"]
        assert again["merged"] == it["race"]["merged"]
    good = bank2.usable(items)
    if good:
        labels, _ = bank2.label_items(good, draws=500)
        assert all(abs(sum(lab["P"][g[0]] for g in lab["duplicates"]) - 1) < 1e-9 for lab in labels.values())

"""Rule stance planner: each stance's trigger, priorities, ablations, and the information boundary."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("Simulator")

from Simulator.battle.champion import champion  # noqa: E402
from Simulator.battle.stats import COST  # noqa: E402
from Simulator.game import pool  # noqa: E402
from Simulator.game.player import Player  # noqa: E402
from Simulator.generators.default_agent import Default_Agent  # noqa: E402
from Simulator.generators.default_agent_stats import TEAM_COMP_TRAITS, TEAM_COMPS  # noqa: E402

from tfteval import make_policy, public, stages  # noqa: E402
from tfteval import stance as st  # noqa: E402
from tfteval.planner import LEVEL_COSTS, ParamPlanner, PlanPolicy, compile_knobs, describe  # noqa: E402

COMPS = dict(zip(TEAM_COMP_TRAITS, TEAM_COMPS))
EVEN = ["ahri*1", "annie*1", "lulu*1", "veigar*1", "thresh*1", "jax*1"]  # a middling 6-unit board
STRONG = ["ahri*2[bf_sword,chain_vest]", "annie*2", "lulu*2", "veigar*2", "thresh*1", "jax*1"]
WEAK = ["vayne*1", "fiora*1", "garen*1"]


def own(tag: str) -> dict:
    u = st.parse_tag(tag)
    u["cost"] = COST[u["name"]]
    return u


def opp(seat: str, board=EVEN, hp=80, level=6, streak=0, interest=3) -> dict:
    return {"seat": seat, "hp": hp, "level": level, "streak": streak, "interest": interest, "board": list(board)}


def make_state(idx=12, hp=80, gold=50, level=6, xp=0, streak=0, board=EVEN, bench=(), opponents=None,
               next_from=None, comp=None) -> dict:
    """A synthetic describe() state (test_synthetic_state_has_the_describe_keys checks the keys)."""
    sched = stages.schedule(idx)
    opponents = [opp(f"player_{i}") for i in range(1, 8)] if opponents is None else opponents
    state = {
        "round": idx, "stage": sched["stage"], "pve": sched["pve"],
        "next": {"carousel": sched["to_carousel"], "pve": sched["to_pve"], "stage": sched["to_stage"]},
        "hp": hp, "gold": gold, "level": level, "xp": xp, "xp_needed": LEVEL_COSTS[level], "streak": streak,
        "dmg_per_loss": stages.damage_per_loss(idx), "losses_to_death": stages.losses_to_death(hp, idx),
        "board": [own(t) for t in board], "bench": [own(t) for t in bench], "item_bench": [], "shop": [],
        "active_traits": {},
        "hp_rank": 1 + sum(o["hp"] > hp for o in opponents), "alive": len(opponents) + 1,
        "opponents": opponents,
        "next_from": [o["seat"] for o in opponents[:3]] if next_from is None else next_from,
    }
    seen = st.seen_copies(state)
    state["contested_by_comp"] = {t: sum(seen[u] for u in units) for t, units in COMPS.items()}
    if comp:
        state["contested"] = {u: seen[u] for u in COMPS[comp]}
    return state


def run(planner, state, comp=None):
    return planner.plan(state, COMPS, comp)


# --------------------------------------------------------------------------- features

def test_board_score_is_the_rule_bots_rank_comp_plus_items():
    p = Player(pool.pool(), 0)
    p.level = p.max_units = 6
    tags = ["ahri*2", "annie*1", "lulu*1", "veigar*1", "nami*1", "twistedfate*1"]
    for i, tag in enumerate(tags):
        name, _, stars = tag.partition("*")
        assert p.add_to_bench(champion(name, stars=int(stars)))
        assert p.move_bench_to_board(next(j for j, c in enumerate(p.bench) if c), i, 0)
    units = [st.parse_tag(public.unit_tag(u)) for u in public.board_units(p)]
    assert st.board_score(units) == Default_Agent().rank_comp(p.board) > 0
    units[0]["items"] = ["bf_sword", "chain_vest"]
    assert st.board_score(units) == Default_Agent().rank_comp(p.board) + 2 * st.ITEM_SCORE


def test_parse_tag_reads_public_unit_tags():
    assert st.parse_tag("ahri*2(mage)[bf_sword,chain_vest]") == \
        {"name": "ahri", "star": 2, "chosen": "mage", "items": ["bf_sword", "chain_vest"]}
    assert st.parse_tag("twistedfate*1") == {"name": "twistedfate", "star": 1}
    with pytest.raises(ValueError):
        st.parse_tag("ahri")


def test_features_compare_with_the_candidates_only():
    opponents = [opp("player_1", WEAK), opp("player_2", WEAK), opp("player_3", STRONG, level=8),
                 opp("player_4", STRONG, level=7, interest=5)]
    f = st.features(make_state(streak=-2, opponents=opponents, next_from=["player_1", "player_2"]), COMPS, None)
    assert f["ratio"] > 1 and f["score_rank"] == 3  # stronger than the two candidates, not than the field
    assert f["opp_near8"] == 2 and f["opp_level_max"] == 8 and f["opp_rich"] == 1
    assert f["streak_gold_loss"] == 1 and f["streak_gold_win"] == 1  # a third loss pays 1; a win ends it
    assert st.features(make_state(streak=4), COMPS, None)["streak_gold_win"] == 1 + 3


def test_projection_and_xp_cost():
    assert st.projected_gold(50, 2) == 50 + 10 + 10 and st.projected_gold(9, 1) == 9 + 5
    assert st.xp_gold(7, 0, 8) == 56 and st.xp_gold(6, 0, 8, rounds=4) == 84 and st.xp_gold(7, 55, 8) == 4


# --------------------------------------------------------------------------- stances

def test_stage_one_and_a_quiet_round_are_standard_econ():
    planner = st.StancePlanner()
    assert run(planner, make_state(idx=1, level=2))["stance"] == "standard"
    state = make_state(idx=10, level=5, gold=58)
    plan = run(st.StancePlanner(), state)
    assert plan["stance"] == "standard" and plan["field_comp"] is True
    base = ParamPlanner().plan(state, COMPS, None)
    assert {k: plan[k] for k in base} == base  # the rule bot's own economy
    assert not {"level_by", "spend", "survival"} & set(plan)
    assert plan["features"]["ratio"] == 1.0 and plan["why"].startswith("default")


def test_stabilize_when_three_losses_from_death():
    state = make_state(idx=16, hp=40, level=6, xp=30, gold=45)  # 4-2: ~13.9 a loss
    assert state["losses_to_death"] == 3
    plan = run(st.StancePlanner(), state)
    assert plan["stance"] == "stabilize" and plan["why"].startswith("dying")
    assert plan["roll_floor"] == st.STABILIZE_HARD_FLOOR and plan["survival"] == st.SURVIVAL_LTD
    assert plan["level_to"] == 7  # 6 xp short: cheap
    assert run(st.StancePlanner(), make_state(idx=16, hp=50))["stance"] == "standard"  # 4 to death


def test_stabilize_on_heavy_losses_from_stage_three():
    strong = [opp(f"player_{i}", STRONG) for i in range(1, 8)]
    plan = run(st.StancePlanner(), make_state(idx=12, hp=80, streak=-3, opponents=strong))
    assert plan["stance"] == "stabilize" and plan["why"].startswith("heavy")
    assert plan["roll_floor"] == st.STABILIZE_SOFT_FLOOR
    assert run(st.StancePlanner(), make_state(idx=6, hp=80, streak=-3, opponents=strong))["stance"] == "standard"
    assert run(st.StancePlanner(), make_state(idx=12, hp=80, streak=-2, opponents=strong))["stance"] == "standard"
    assert run(st.StancePlanner(), make_state(idx=12, hp=80, streak=-3))["stance"] == "standard"  # even boards


SLOW = dict(idx=9, level=5, gold=60, board=["nami*2", "ahri*1", "lulu*1", "twistedfate*1", "annie*1"],
            bench=["nami*1"])  # 4 copies of a 1-cost mage


def test_slow_roll_starts_on_four_copies_of_a_cheap_comp_unit():
    plan = run(st.StancePlanner(), make_state(**SLOW))
    assert plan["stance"] == "slow_roll" and plan["why"].startswith("start: nami")
    assert plan["level_to"] == 5 and plan["roll_floor"] == st.SLOWROLL_FLOOR and plan["carry"] == "nami"
    assert plan["comp"] == "mage"  # of mage / spirit / enlightened, the one holding most of the units
    assert run(st.StancePlanner(), make_state(**{**SLOW, "level": 4}))["level_to"] == 5  # one level up first


@pytest.mark.parametrize("change", [
    {"level": 6},  # above the 1-cost reroll level
    {"bench": []},  # 3 copies
    {"opponents": [opp("player_1", ["nami*2", "nami*1"])] + [opp(f"player_{i}") for i in range(2, 8)]},  # 4 seen
    {"hp": 45},
    {"idx": 16},  # past SLOWROLL_UNTIL for a 1-cost
])
def test_slow_roll_does_not_start(change):
    assert run(st.StancePlanner(), make_state(**{**SLOW, **change}))["stance"] != "slow_roll"


def test_slow_roll_only_inside_the_chosen_comp():
    assert run(st.StancePlanner(), make_state(**SLOW, comp="divine"), "divine")["stance"] == "standard"
    plan = run(st.StancePlanner(), make_state(**SLOW, comp="mage"), "mage")
    assert plan["stance"] == "slow_roll" and plan["comp"] is None  # keep the executor's comp


@pytest.mark.parametrize("later, reason", [
    ({}, None),
    ({"board": ["nami*3", "ahri*1", "lulu*1", "twistedfate*1", "annie*1"], "bench": []}, "3-star"),
    ({"hp": 54}, "bleeding"),
    ({"opponents": [opp("player_1", ["nami*2", "nami*1"])] + [opp(f"player_{i}") for i in range(2, 8)]},
     "contested"),
    ({"bench": []}, "copies sold"),
    ({"idx": 16}, "too late"),
])
def test_slow_roll_is_kept_until_a_stop_condition(later, reason):
    planner = st.StancePlanner()
    assert run(planner, make_state(**SLOW))["stance"] == "slow_roll"
    plan = run(planner, make_state(**{**SLOW, "idx": 10, **later}))
    if reason is None:
        assert plan["stance"] == "slow_roll" and plan["why"].startswith("keep: nami") and plan["level_to"] == 5
    else:
        assert plan["stance"] == "standard" and plan["features"]["slow_roll_end"] == f"nami: {reason}"
        again = make_state(**{**SLOW, "idx": plan["features"]["round"] + 1})
        assert run(planner, again)["stance"] != "slow_roll"  # not restarted on the same unit


FAST8 = dict(idx=12, level=6, xp=0, gold=75, hp=80)  # 3-5: 75 gold -> 115 by 4-2, 84 of it for level 8


def test_fast8_commits_levels_by_the_deadline_then_rolls_down():
    planner = st.StancePlanner()
    plan = run(planner, make_state(**FAST8))
    assert plan["stance"] == "fast8" and plan["why"].startswith("start: 8 by 4-2")
    assert plan["level_by"] == {"level": 8, "by": "4-2"} and plan["level_to"] == 6 and plan["roll_floor"] == 999
    assert compile_knobs(plan, make_state(**FAST8))["xp_priority"]
    keep = run(planner, make_state(**{**FAST8, "idx": 13, "gold": 40}))  # committed: no re-check of the gold
    assert keep["stance"] == "fast8" and keep["why"].startswith("keep") and keep["level_by"]["by"] == "4-2"
    landed = run(planner, make_state(**{**FAST8, "idx": 17, "level": 8, "gold": 30}))
    assert landed["stance"] == "fast8" and landed["spend"] == {"to": st.FAST8_ROLL_TO, "by": "4-5"}
    assert run(planner, make_state(**{**FAST8, "idx": 18, "level": 8, "gold": 25}))["stance"] == "fast8"
    assert run(planner, make_state(**{**FAST8, "idx": 19, "level": 8, "gold": 25}))["stance"] == "standard"


def test_fast8_dropped_after_a_missed_deadline_or_below_50_hp():
    planner = st.StancePlanner()
    run(planner, make_state(**FAST8))
    assert run(planner, make_state(**{**FAST8, "idx": 17, "level": 7}))["stance"] == "fast8"  # 1 round grace
    plan = run(planner, make_state(**{**FAST8, "idx": 18, "level": 7}))
    assert plan["stance"] == "standard" and plan["features"]["fast8_end"] == "missed the deadline"
    planner = st.StancePlanner()
    run(planner, make_state(**FAST8))
    assert run(planner, make_state(**{**FAST8, "idx": 13, "hp": 52}))["stance"] == "fast8"
    plan = run(planner, make_state(**{**FAST8, "idx": 14, "hp": 46}))  # 4 losses to death: not yet stabilize
    assert plan["stance"] == "standard" and plan["features"]["fast8_end"] == "hp 46"


@pytest.mark.parametrize("change", [{"gold": 40}, {"hp": 55}, {"idx": 11}, {"idx": 17},
                                    {"opponents": [opp(f"player_{i}", STRONG) for i in range(1, 8)]}])
def test_fast8_does_not_start(change):
    assert run(st.StancePlanner(), make_state(**{**FAST8, **change}))["stance"] != "fast8"


def test_fast8_goes_earlier_when_opponents_are_close_to_8():
    racing = [opp("player_1", level=8), opp("player_2", level=7, interest=5)] + \
        [opp(f"player_{i}") for i in range(3, 8)]
    plan = run(st.StancePlanner(), make_state(**{**FAST8, "gold": 90, "opponents": racing}))
    assert plan["stance"] == "fast8" and plan["level_by"]["by"] == "4-1"
    # 75 gold pays for 8 by 4-2 but not by 4-1
    assert run(st.StancePlanner(), make_state(**{**FAST8, "opponents": racing}))["stance"] != "fast8"


STREAK = dict(idx=6, level=4, xp=0, gold=30, streak=3, board=STRONG[:4],
              opponents=[opp(f"player_{i}", WEAK, level=4) for i in range(1, 8)])


def test_keep_streak_levels_and_rolls_the_excess():
    plan = run(st.StancePlanner(), make_state(**STREAK))
    assert plan["stance"] == "keep_streak" and plan["why"].startswith("streak: 3 wins")
    assert plan["level_to"] == 5 and plan["roll_floor"] == st.STREAK_ROLL_FLOOR  # 12 gold for 5, 18 left
    assert run(st.StancePlanner(), make_state(**{**STREAK, "level": 5}))["level_to"] == 5  # stage-2 cap
    assert run(st.StancePlanner(), make_state(**{**STREAK, "gold": 20}))["level_to"] == 4  # would leave 8


@pytest.mark.parametrize("change", [{"streak": 1}, {"opponents": [opp(f"player_{i}", STRONG) for i in range(1, 8)]}])
def test_keep_streak_needs_a_streak_and_the_stronger_board(change):
    assert run(st.StancePlanner(), make_state(**{**STREAK, **change}))["stance"] == "standard"


def test_priorities():
    dying_streak = make_state(**{**STREAK, "idx": 16, "hp": 30})
    assert run(st.StancePlanner(), dying_streak)["stance"] == "stabilize"
    assert run(st.StancePlanner(), make_state(**{**FAST8, "streak": 3}))["stance"] == "fast8"
    # a 2-cost with 4 copies at level 6 and the gold for a fast 8: the slow roll goes first ...
    planner = st.StancePlanner()
    slow_and_rich = make_state(**{**FAST8, "board": ["teemo*2", "ahri*1", "lulu*1", "veigar*1", "thresh*1"],
                                  "bench": ["teemo*1"]})  # teemo: a 2-cost no opponent fields
    assert run(planner, slow_and_rich)["stance"] == "slow_roll"
    # ... and blocks the fast 8 while it lasts
    assert run(planner, {**slow_and_rich, "round": 13, "stage": "3-6"})["stance"] == "slow_roll"


@pytest.mark.parametrize("kind, state", [
    ("stance-nostabilize", make_state(idx=16, hp=40)),
    ("stance-noslowroll", make_state(**SLOW)),
    ("stance-nofast8", make_state(**FAST8)),
    ("stance-nostreak", make_state(**STREAK)),
])
def test_ablations_replace_their_stance_with_standard_econ(kind, state):
    assert run(st.StancePlanner(), state)["stance"] != "standard"
    policy = make_policy(f"hero={kind}")
    assert isinstance(policy, PlanPolicy) and policy.name == "hero"
    assert policy.planner.disabled == set(st.STANCE_KINDS[kind]["off"]) and not policy.planner.enabled
    plan = run(policy.planner, state)
    base = ParamPlanner().plan(state, COMPS, None)
    assert plan["stance"] == "standard" and {k: plan[k] for k in base} == base


def test_stance_seat_kinds_are_registered():
    assert set(st.STANCE_KINDS) == {"stance", "stance-nofast8", "stance-noslowroll", "stance-nostreak",
                                    "stance-nostabilize", "stance+lossstreak"}
    plain = make_policy("stance").planner
    assert make_policy("stance").name == "stance" and not plain.disabled and not plain.on("loss_streak")
    assert make_policy("stance+lossstreak").planner.on("loss_streak")
    with pytest.raises(ValueError):
        st.StancePlanner(disabled=["standard"])
    with pytest.raises(ValueError):
        st.StancePlanner(enabled=["fodder"])


# --------------------------------------------------------------------------- controlled loss streak (opt-in)

LOSS = dict(idx=4, level=3, gold=12, hp=92, streak=-1)  # 2-2, lost 2-1


def loss_planner():
    return st.StancePlanner(enabled=["loss_streak"])


def test_loss_streak_is_off_in_the_default_stance_policy():
    assert run(st.StancePlanner(), make_state(**LOSS))["stance"] == "standard"


def test_loss_streak_fields_fodder_while_losing_at_high_hp():
    planner = loss_planner()
    plan = run(planner, make_state(**LOSS))
    assert plan["stance"] == "loss_streak" and plan["fodder"] is True and plan["why"].startswith("start: 1 losses")
    assert compile_knobs(plan, make_state(**LOSS))["fodder"]
    keep = run(planner, make_state(**{**LOSS, "idx": 5, "hp": 84, "streak": -2}))
    assert keep["stance"] == "loss_streak" and keep["why"].startswith("keep: 2 losses")
    over = run(planner, make_state(**{**LOSS, "idx": 6, "hp": 76, "streak": -3}))  # below 80 HP
    assert over["stance"] == "standard" and "fodder" not in over and over["features"]["loss_streak_end"] == "hp 76"
    assert run(planner, make_state(**{**LOSS, "idx": 7, "hp": 90, "streak": -4}))["stance"] == "standard"  # once


def test_loss_streak_starts_on_a_weak_board_and_ends_when_the_streak_breaks():
    planner = loss_planner()
    strong = [opp(f"player_{i}", STRONG) for i in range(1, 8)]
    plan = run(planner, make_state(**{**LOSS, "idx": 3, "hp": 100, "streak": 0, "opponents": strong}))
    assert plan["stance"] == "loss_streak" and plan["why"].startswith("start: board")
    broken = run(planner, make_state(**{**LOSS, "idx": 4, "hp": 100, "streak": 1, "opponents": strong}))
    assert broken["stance"] == "standard" and broken["features"]["loss_streak_end"] == "streak broken"


@pytest.mark.parametrize("change", [{"hp": 78}, {"streak": 0}, {"streak": 2}, {"idx": 8}, {"idx": 9}])
def test_loss_streak_does_not_start(change):
    assert run(loss_planner(), make_state(**{**LOSS, **change}))["stance"] != "loss_streak"


def test_loss_streak_ends_with_stage_two():
    planner = loss_planner()
    assert run(planner, make_state(**{**LOSS, "idx": 7, "streak": -3}))["stance"] == "loss_streak"
    plan = run(planner, make_state(**{**LOSS, "idx": 8, "streak": -4}))  # 2-7 is PvE: no fodder there
    assert plan["stance"] == "standard" and plan["features"]["loss_streak_end"] == "end of stage 2"


def test_new_game_resets_the_commitments():
    planner = st.StancePlanner()
    run(planner, make_state(**FAST8))
    assert planner.fast8 is not None
    run(planner, make_state(idx=3, level=3, gold=5))
    assert planner.fast8 is None and planner.hp_seen == {3: 80}


def test_every_plan_compiles_and_logs(tmp_path):
    log = tmp_path / "stance.jsonl"
    planner = st.StancePlanner(log_path=str(log))
    planner.context = {"seed": 7, "seat": "player_3"}
    planner.enabled = frozenset({"loss_streak"})
    states = [make_state(idx=1, level=2), make_state(**SLOW), make_state(**FAST8), make_state(**STREAK),
              make_state(idx=16, hp=20), make_state(**LOSS)]
    for state in states:
        planner._new_game()
        plan = run(planner, state)
        compile_knobs(plan, state)
        json.dumps(plan)  # plain data
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["stance"] for r in rows] == ["standard", "slow_roll", "fast8", "keep_streak", "stabilize",
                                           "loss_streak"]
    assert rows[0]["seed"] == 7 and rows[0]["seat"] == "player_3" and "features" not in rows[0]["plan"]


# --------------------------------------------------------------------------- information boundary

def test_the_module_never_reaches_for_hidden_state():
    """No name, attribute or key in the planner's code (docstrings aside) that leads to the env or to
    another player's hidden state; it only reads the describe() dict it is given."""
    tree = ast.parse(Path(st.__file__).read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    words = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            words.add(node.id)
        elif isinstance(node, ast.Attribute):
            words.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            words.update(re.findall(r"\w+", node.value))
    hidden = {"env", "matchups", "game_round", "player_manager", "player_states", "unwrapped", "opponent_options",
              "possible_opponents", "health", "exp", "shop", "item_bench", "pool", "Player"}
    assert not words & hidden
    assert {"opponents", "next_from", "board", "bench"} <= words  # (the check does see the code)


class Trap:
    """Stands in for game_round: any attribute read fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"read game_round.{name}")


def make_player(num, board=(), bench=(), gold=0, level=4, hp=100):
    p = Player(pool.pool(), num)
    p.gold, p.level, p.max_units, p.health = gold, level, max(level, len(board)), hp
    for i, tag in enumerate(board):
        name, _, stars = tag.partition("*")
        champ = champion(name, stars=int(stars or 1))
        assert p.add_to_bench(champ)
        assert p.move_bench_to_board(p.bench.index(champ), i, 0)
    for tag in bench:
        name, _, stars = tag.partition("*")
        assert p.add_to_bench(champion(name, stars=int(stars or 1)))
    return p


@pytest.fixture
def lobby():
    me = make_player(0, board=["nami*2", "ahri", "lulu", "twistedfate"], bench=["nami"], gold=62, level=5, hp=80)
    rich = make_player(1, board=["ahri*2", "lulu", "veigar", "nami"], bench=["veigar", "veigar"], gold=57, level=6)
    poor = make_player(2, board=["vayne", "fiora"], gold=4, level=5, hp=35)
    players = {"player_0": me, "player_1": rich, "player_2": poor}
    me.opponent_options = {"player_1": 1, "player_2": 1}
    env = SimpleNamespace(unwrapped=SimpleNamespace(player_manager=SimpleNamespace(player_states=players),
                                                    game_round=Trap()))
    return me, players, env


def plan_for(me, env, idx=9):
    state = describe(me, ["zed", None], idx, env, seat="player_0", comps=COMPS, comp=None)
    return json.dumps(st.StancePlanner().plan(state, COMPS, None), sort_keys=True), state


def test_synthetic_state_has_the_describe_keys(lobby):
    me, _, env = lobby
    _, state = plan_for(me, env)
    assert set(make_state()) == set(state)


def test_plan_ignores_hidden_state_and_matchups(lobby):
    me, players, env = lobby
    before, state = plan_for(me, env)
    assert json.loads(before)["stance"] == "slow_roll"  # 4 copies of nami, 1 seen
    rich = players["player_1"]
    rich.gold, rich.exp = 51, 17  # same interest bracket
    rich.bench = [champion("nami")] * 3 + [None] * 6  # many more namis, but out of sight
    rich.shop = ["nami"] * 5
    rich.item_bench = ["chain_vest"] + [None] * 9
    me.possible_opponents = {k: 99 for k in me.possible_opponents}
    env.unwrapped.game_round = SimpleNamespace(matchups=[("player_0", "player_2")])  # the "real" next opponent
    after, _ = plan_for(me, env)
    assert after == before
    # the same plan from a plain-data copy of the state: nothing live is reachable from it
    assert json.dumps(st.StancePlanner().plan(json.loads(json.dumps(state)), COMPS, None), sort_keys=True) == before
    # public information does move it: the poor seat fields three namis
    poor = players["player_2"]
    poor.board[0][0] = champion("nami", stars=2)
    poor.board[1][0] = champion("nami")
    assert json.loads(plan_for(me, env)[0])["stance"] != "slow_roll"

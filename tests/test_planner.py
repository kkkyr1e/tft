from tfteval.planner import LLMPlanner, MimicPlanner

COMPS = {"mage": ["ahri", "annie"], "divine": ["irelia", "jax"]}
STATE = {"round": 12, "hp": 60, "gold": 30, "level": 6, "xp": 2, "xp_needed": 36,
         "board": [{"name": "ahri", "star": 2, "cost": 2}], "bench": [], "item_bench": [], "shop": []}


def test_parse_takes_last_valid_object():
    text = 'thinking {"note": 1}\n{"comp": "mage", "level_to": 7, "roll_floor": 20, "carry": "ahri", "why": "x"}'
    plan = LLMPlanner.parse(text, STATE, COMPS)
    assert plan["comp"] == "mage" and plan["level_to"] == 7 and plan["roll_floor"] == 20 and plan["carry"] == "ahri"


def test_parse_rejects_unknown_comp_and_drops_unknown_carry():
    assert LLMPlanner.parse('{"comp": "void", "level_to": 7, "roll_floor": 20}', STATE, COMPS) is None
    plan = LLMPlanner.parse('{"comp": null, "level_to": 12, "roll_floor": -3, "carry": "zed"}', STATE, COMPS)
    assert plan["carry"] is None and plan["level_to"] == 9 and plan["roll_floor"] == 0


def test_parse_no_json():
    assert LLMPlanner.parse("I would roll down now.", STATE, COMPS) is None


def test_mimic_levels_when_four_short_before_round_11():
    early = {**STATE, "round": 5, "level": 4, "xp": 6, "xp_needed": 10}
    assert MimicPlanner().plan(early, COMPS, None)["level_to"] == 5


def test_field_comp_knob_only_when_set():
    assert "field_comp" not in MimicPlanner().plan(STATE, COMPS, None)
    assert MimicPlanner(field_comp=True).plan(STATE, COMPS, None)["field_comp"] is True

"""Single-fight win probability from public information: the project's first learned model.

`p_win(self_view, opp_view)` is the probability that the first player beats the second in the next
PvP fight. A view is what anyone in the lobby can see of a player: the board (unit tags as in
tfteval.public.unit_tag, or unit dicts as in describe()'s own `board`), level, HP and streak. Use
`self_view(state)` for the seat's own describe() state and the entries of `state["opponents"]` for
the others.

The model is a logistic regression on differences of per-player features (`view_features`), fitted by
scripts/fit_winprob.py on fights recorded by scripts/collect_fights.py (tfteval/fightlog.py).
Being a function of differences with no intercept, p_win(a, b) + p_win(b, a) = 1. The simulator's
blue side (the first seat of a matchup) is fitted as its own term, and p_win averages over the two
sides because a planner does not know which side it will get.

Two coefficient sets are kept in tfteval/data/winprob.json: `start` uses the views at the start of
the planning phase (what a planner sees; the default), `fight` the boards that actually fight
(after the round's buys and the autofill).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "data" / "winprob.json"

# per-player features: board_parts (stance.py), unit counts by star, items, level, HP, streak
BASE = ("score", "value", "traits", "active", "items", "units", "star2", "star3", "chosen", "level", "hp", "streak")


def view_features(view: dict) -> dict:
    """Per-player features of a public view. Board entries may be tags or unit dicts."""
    from tfteval.stance import board_parts, parse_tag

    units = [parse_tag(u) if isinstance(u, str) else u for u in view["board"]]
    f = board_parts(units)
    stars = [int(u["star"]) for u in units]
    f.update(units=len(units), star2=sum(s == 2 for s in stars), star3=sum(s >= 3 for s in stars),
             chosen=sum(bool(u.get("chosen")) for u in units), level=int(view["level"]), hp=float(view["hp"]),
             streak=int(view.get("streak", 0)))
    return f


def derived(f: dict) -> dict:
    """Features plus the transforms a model may use (a term is the difference of one of these)."""
    out = dict(f)
    out["log_score"] = math.log1p(max(0, f["score"]))
    out["log_value"] = math.log1p(max(0, f["value"]))
    out["win_streak"] = max(0, f["streak"])
    out["loss_streak"] = max(0, -f["streak"])
    return out


def self_view(state: dict) -> dict:
    """The seat's own public view from its describe() state."""
    return {"board": state["board"], "level": state["level"], "hp": state["hp"], "streak": state.get("streak", 0)}


def _sigmoid(z: float) -> float:
    return 0.5 * (1.0 + math.tanh(0.5 * z))


class Model:
    def __init__(self, spec: dict):
        self.terms = list(spec["terms"])
        self.coef = [float(spec["coef"][t]) for t in self.terms]
        self.side = float(spec.get("side", 0.0))

    def logit(self, fa: dict, fb: dict) -> float:
        da, db = derived(fa), derived(fb)
        return sum(c * (da[t] - db[t]) for t, c in zip(self.terms, self.coef))

    def p(self, fa: dict, fb: dict, side: int = 0) -> float:
        """P(a beats b). side=+1: a is blue, -1: a is red, 0: unknown (the average of both)."""
        z = self.logit(fa, fb)
        if side:
            return _sigmoid(z + side * self.side)
        return 0.5 * (_sigmoid(z + self.side) + _sigmoid(z - self.side))


@lru_cache(maxsize=None)
def load(which: str = "start", path: str | None = None) -> Model:
    spec = json.loads(Path(path or MODEL_PATH).read_text())
    return Model(spec["models"][which])


def p_win(self_view: dict, opp_view: dict, model: str = "start") -> float:
    """P(self_view's player beats opp_view's player in their next fight)."""
    return load(model).p(view_features(self_view), view_features(opp_view))

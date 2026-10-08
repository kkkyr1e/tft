"""Record every PvP fight of a game with both players' public views, from our side of the simulator.

The simulator resolves a round's fights in `Game_Round.combat_phase` (game/game_round.py): for each
pair in `game_round.matchups` it calls `champion.run(champion.champion, blue, red, base_damage)`,
which returns `(index_won, damage)` (0 draw, 1 blue won, 2 red won; damage is base damage plus the
winner's surviving units), and applies won_round / loss_round and the HP loss. The last unpaired
player fights a "ghost": the board of a living player, with only the unpaired player's result
counted. `combat_round` autofills every board (`end_turn_actions`) just before `combat_phase`,
so the boards seen when `combat_phase` starts are the boards that fight.

`FightRecorder` wraps three simulator functions while it is installed (nothing in third_party is
edited): `Game_Round.start_round` to keep each player's public view at the start of the planning
phase (what a planner sees), `Game_Round.combat_phase` to take the view again just before the fights,
and `champion.run` (only inside `combat_phase`) to read each fight's exact result. Only public
fields are recorded (tfteval.public.opponent_view: board units with stars, items and chosen trait,
HP, level, streak, interest bracket). The wrappers draw no random numbers, so a recorded game plays
exactly as an unrecorded one (tests/test_winprob.py checks this).

One record per round with at least one PvP fight:

    {"seed", "round", "stage", "lobby": {seat: policy},
     "start": {seat: view}, "fight": {seat: view},
     "fights": [{"a": seat, "b": seat, "ghost": bool, "won": 0/1/2, "damage": int}]}

`a` is the blue side (the matchup's first seat), `won` is the simulator's index_won from a's side
(1 = a won, 2 = b won, 0 = draw). In a ghost fight `b` is the seat whose board is copied.
"""

from __future__ import annotations

from tfteval import public, stages


def views(players: dict) -> dict:
    """seat -> public view of every living player."""
    return {seat: public.opponent_view(seat, p) for seat, p in sorted(players.items()) if p is not None}


class FightRecorder:
    """Context manager: `with FightRecorder() as rec: play_game(...)`, then `rec.records`."""

    def __init__(self, meta: dict | None = None):
        self.meta = dict(meta or {})
        self.records: list[dict] = []
        self._start: tuple[int, dict] | None = None
        self._saved = None

    # ------------------------------------------------------------------ install
    def __enter__(self) -> "FightRecorder":
        from Simulator.battle import champion
        from Simulator.game import game_round

        cls = game_round.Game_Round
        self._saved = (cls.start_round, cls.combat_phase, champion.run)
        orig_start, orig_combat, orig_run = self._saved
        recorder = self

        def start_round(gr):
            orig_start(gr)
            recorder._start = (int(gr.current_round), views(gr.PLAYERS))

        def combat_phase(gr, players, player_round):
            idx = int(gr.current_round)
            before = views(players)
            results = []

            def run(champion_q, blue, red, round_damage=0):
                out = orig_run(champion_q, blue, red, round_damage)
                results.append((blue, red, out))
                return out

            champion.run = run
            try:
                return orig_combat(gr, players, player_round)
            finally:
                champion.run = orig_run
                recorder._record(idx, gr.matchups, players, before, results)

        cls.start_round, cls.combat_phase = start_round, combat_phase
        return self

    def __exit__(self, *exc) -> None:
        from Simulator.battle import champion
        from Simulator.game import game_round

        cls = game_round.Game_Round
        cls.start_round, cls.combat_phase, champion.run = self._saved
        self._saved = None

    # ------------------------------------------------------------------ record
    def _record(self, idx: int, matchups, players: dict, before: dict, results: list) -> None:
        seat_of = {id(p): seat for seat, p in players.items() if p is not None}
        fights = []
        for match, (blue, red, (won, damage)) in zip(matchups, results):
            ghost = len(match) == 3 and match[1] == "ghost"
            a, b = match[0], (match[2] if ghost else match[1])
            if seat_of.get(id(blue)) != a or seat_of.get(id(red)) != b:
                raise RuntimeError(f"fight {idx}: results out of order with the matchups {matchups}")
            fights.append({"a": a, "b": b, "ghost": ghost, "won": int(won), "damage": int(damage)})
        if len(results) != len(matchups):
            raise RuntimeError(f"round {idx}: {len(results)} fights for {len(matchups)} matchups")
        if not fights:
            return
        start = self._start[1] if self._start and self._start[0] == idx else {}
        self.records.append({**self.meta, "round": idx, "stage": stages.label(idx), "start": start,
                             "fight": before, "fights": fights})

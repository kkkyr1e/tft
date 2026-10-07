"""The public part of the game: what a real player can see about the other players.

A player in the real game sees every opponent's board (units, stars, items, chosen), HP, level,
win/loss streak and interest bracket, plus the set of players they can meet next. They do not
see an opponent's bench, shop, exact gold or xp, the pool's exact counts, or the exact next
opponent. The simulator's observation encodes the same public fields per opponent
(encoding/token/basic_observation.py, `create_public_scalars` and `create_board_vector`); this
module reads only those fields from the player objects, so nothing hidden can reach a planner.
The candidate set comes from `player.opponent_options`, never from `game_round.matchups`.
"""

from __future__ import annotations

from collections import Counter


def is_unit(champ) -> bool:
    return bool(champ) and champ.name != "sandguard" and not getattr(champ, "target_dummy", False)


def board_units(player) -> list:
    return [u for col in player.board for u in col if is_unit(u)]


def copies(champ) -> int:
    """Copies of a unit taken out of the pool: a 2-star is 3 copies, a 3-star 9."""
    return 3 ** (int(champ.stars) - 1)


def unit_tag(champ) -> str:
    """Compact text for a unit: name*stars, then (chosen trait) and [items] when present."""
    tag = f"{champ.name}*{int(champ.stars)}"
    if champ.chosen:
        tag += f"({champ.chosen})"
    if champ.items:
        tag += "[" + ",".join(champ.items) + "]"
    return tag


def streak(player) -> int:
    """Signed streak: +n for an n-game win streak, -n for a loss streak."""
    return int(player.win_streak) if player.win_streak else -int(player.loss_streak)


def interest(player) -> int:
    return min(int(player.gold) // 10, 5)


def opponent_view(seat: str, player) -> dict:
    return {
        "seat": seat,
        "hp": int(player.health),
        "level": int(player.level),
        "streak": streak(player),
        "interest": interest(player),
        "board": [unit_tag(u) for u in board_units(player)],
    }


def alive_players(env) -> dict:
    """seat -> Player for living players. The only place the env is touched."""
    manager = getattr(getattr(env, "unwrapped", env), "player_manager", None)
    if manager is None:
        return {}
    return {seat: p for seat, p in sorted(manager.player_states.items()) if p is not None and p.health > 0}


def next_candidates(seat: str, player, players: dict) -> list[str]:
    """Seats the player can meet next round (public in the real game; always holds the real one)."""
    options = getattr(player, "opponent_options", {}) or {}
    return sorted(s for s, flag in options.items() if flag == 1 and s in players and s != seat)


def seen_copies(players: dict, seat: str) -> Counter:
    """Copies of each unit on the other players' boards."""
    seen = Counter()
    for other_seat, other in players.items():
        if other_seat != seat:
            for u in board_units(other):
                seen[u.name] += copies(u)
    return seen


def public_view(seat: str, player, players: dict, comps: dict | None = None, comp: str | None = None) -> dict:
    """Opponents, the next-opponent candidates, HP rank and how contested the comps are."""
    others = {s: p for s, p in players.items() if s != seat and p is not player}
    opponents = sorted((opponent_view(s, p) for s, p in others.items()), key=lambda o: (-o["hp"], o["seat"]))
    hps = [int(p.health) for p in others.values()]
    out = {
        "hp_rank": 1 + sum(hp > int(player.health) for hp in hps),
        "alive": len(others) + 1,
        "opponents": opponents,
        "next_from": next_candidates(seat, player, players),
    }
    if comps:
        seen = seen_copies(players, seat)
        out["contested_by_comp"] = {trait: sum(seen[u] for u in units) for trait, units in comps.items()}
        if comp in comps:
            out["contested"] = {u: seen[u] for u in comps[comp]}
    return out

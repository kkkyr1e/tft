"""Fixes to the pinned simulator, applied at runtime so no file in third_party is edited.

Each fix is a function that patches one simulator function. `apply()` installs the ones the
simulator still needs (idempotent); `play_game` calls it unless sim fixes are switched off, and
records which fixes were active in the game result, so runs made before and after a fix are
never mixed silently.

Since the switch to our simulator fork (scripts/setup_sim.sh), the fork carries these fixes
itself and `apply()` installs nothing; the patches stay for runs against the original upstream
commit. Which simulator a game ran on is recorded separately (`sim_commit` in the result).

carousel_order
    Upstream builds the pick order by walking the players once and inserting a player at the
    front only when their HP is <= the current front's HP (game/carousel.py, the loop at the
    top of `carousel`). Anyone with more HP than the lowest seen so far is never added and gets
    no carousel unit at all; the players who are added pick in reverse seat order. In the real
    game every living player gets one pick: in the first carousel everyone is released at once,
    later carousels release players two at a time from lowest HP up, ties broken at random.
    The replacement does that. Within a pair (or the opening free-for-all) the order is random,
    standing in for who reaches the unit first. Each player still takes the highest-cost unit,
    as upstream does; choosing by unit and item is a policy question, not part of this fix.
"""

from __future__ import annotations

FIXES = ("carousel_order",)
_applied = False
_commit = None


def carousel_order(players, r, rng):
    """Pick order for the carousel at round index `r`: list of living players, first pick first."""
    alive = [p for p in players if p and p.health > 0]
    rng.shuffle(alive)  # random tie-break among equal HP
    if r == 0:
        return alive
    alive.sort(key=lambda p: p.health)  # stable, so ties keep the shuffled order
    order = []
    for i in range(0, len(alive), 2):
        pair = alive[i:i + 2]
        rng.shuffle(pair)
        order.extend(pair)
    return order


def _fixed_carousel(players, r, pool_obj):
    from Simulator.game import carousel as upstream

    order = carousel_order(players, r, upstream.random)
    champions = upstream.generateChampions(r, pool_obj)
    items = upstream.generateHeldItems(r)
    for i, champ in enumerate(champions):
        champ.add_item(items[i])
    for player in order:
        if not champions:
            break
        current = champions[0]
        for champ in champions:
            if champ.cost > current.cost:
                current = champ
        player.add_to_bench(current, from_carousel=True)
        champions.remove(current)
        pool_obj.update_pool(current, -1)


def needed() -> tuple[str, ...]:
    """The fixes this simulator still lacks: none once it ships its own `carousel_order`."""
    from Simulator.game import carousel as carousel_module

    return () if hasattr(carousel_module, "carousel_order") else FIXES


def apply() -> tuple[str, ...]:
    global _applied
    fixes = needed()
    if fixes and not _applied:
        from Simulator.game import carousel as carousel_module
        from Simulator.game import game_round

        carousel_module.carousel = _fixed_carousel
        game_round.carousel = _fixed_carousel  # game_round imported the name directly
        _applied = True
    return FIXES if _applied else ()


def sim_commit() -> str | None:
    """Git commit of the simulator on the path, or None if it is not a git checkout."""
    global _commit
    if _commit is None:
        import pathlib
        import subprocess

        import Simulator

        root = pathlib.Path(Simulator.__file__).resolve().parent.parent
        try:
            out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                 timeout=10)
            _commit = out.stdout.strip() if out.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            _commit = ""
    return _commit or None

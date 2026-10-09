"""Benchmark v1: the frozen lobby configuration, its seeds and sample size, results files and the scorecard.

benchmarks/v1.json holds the configuration; scripts/benchmark.py is the command line. Kept here so the
tests can check the parts that must not drift: lobby sampling, the config hash and the refusal to merge
results across configurations.

A game: one hero seat (the agent under test, reported as `hero`) and 7 opponents drawn for that seed
from a pool, uniformly with replacement (so the mix varies from game to game), the hero on seat
`player_(seed mod 8)`. The draw is sha256("<salt>:<seed>:<slot>") mod the pool size, which no library
version can change. Game i of a run uses seed first_seed + i, so every agent meets the same lobbies on
the same seats, and two agents' results pair by seed. The simulator profile, the economy rules of the
track and the carousel pickers come from the configuration, never from the environment.

Two pools: `dev` for everyday comparisons and tuning, `heldout` (other opponents, other seeds) for
milestones only. A results file records the configuration hash, agent, track, pool, whether the hero was
recorded, the simulator commit and the harness commit; results are only merged (resumed, or combined
for a scorecard) when the configuration hash, agent, track, pool, recording and simulator commit agree.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "benchmarks" / "v2.json"
# a configuration frozen while a simulator profile had another definition plays the profile that keeps it:
# v1 ran "realistic" with 15 actions per planning phase, now "realistic15" (tfteval/runner.py SIM_PROFILES)
LEGACY_SIM = {1: {"realistic": "realistic15"}}
HERO = "hero"
BANK_SCRIPT = ROOT / "scripts" / "score_bank.py"
BANK_CMD = "{python} {script} {bank} --agent {scorer} --rules {rules} --out {out}"  # scripts/score_bank.py's CLI


class ConfigMismatch(ValueError):
    """Results made under another configuration (or agent, track, pool, recording, simulator)."""


# --------------------------------------------------------------------------- configuration

def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def config_sim(config: dict) -> str:
    """The simulator profile a configuration's games are played under (LEGACY_SIM for old versions)."""
    return LEGACY_SIM.get(config.get("version"), {}).get(config["sim"], config["sim"])


def config_hash(config: dict) -> str:
    """sha256 of the configuration as canonical JSON, top-level keys starting with "_" left out."""
    body = {k: v for k, v in config.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def pool_name(heldout: bool) -> str:
    return "heldout" if heldout else "dev"


def agent_kind(agent: str) -> str:
    """The policy kind of an agent spec, any `alias=` prefix dropped (the seat is always `hero`)."""
    return agent.rpartition("=")[2]


def track_rules(config: dict, track: str) -> str:
    if track not in config["tracks"]:
        raise ValueError(f"unknown track {track!r}; choose from {sorted(config['tracks'])}")
    return config["tracks"][track]["rules"]


def seeds(config: dict, pool: str, games: int) -> list:
    first = int(config["pools"][pool]["first_seed"])
    return list(range(first, first + games))


def games_planned(config: dict, sd: float | None = None, half_width: float | None = None) -> int:
    """Games for a 95% interval of the configured half width (tfteval.stats.games_needed)."""
    from tfteval.stats import games_needed

    size = config["sample_size"]
    return games_needed(sd or size["sd_per_game"], half_width or size["half_width"])


def _draw(salt: str, seed: int, slot: int, n: int) -> int:
    digest = hashlib.sha256(f"{salt}:{seed}:{slot}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n


def sample_lobby(config: dict, pool: str, seed: int, agent: str) -> dict:
    """The lobby of one game: {"seats": {seat: policy spec}, "hero_seat", "opponents", "stance_opponents"}."""
    policies = list(config["pools"][pool]["policies"])
    lobby = config["lobby"]
    n = int(lobby["opponents"])
    picks = [policies[_draw(lobby["salt"], seed, slot, len(policies))] for slot in range(n)]
    hero = seed % (n + 1)
    specs = picks[:hero] + [f"{HERO}={agent_kind(agent)}"] + picks[hero:]
    prefix = config.get("stance_family_prefix", "stance")
    return {"seats": {f"player_{i}": spec for i, spec in enumerate(specs)}, "hero_seat": f"player_{hero}",
            "opponents": picks, "stance_opponents": sum(p.startswith(prefix) for p in picks)}


def harness_commit() -> str | None:
    """The harness checkout's commit, with "-dirty" when tracked code or configuration has local edits."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--",
                                "tfteval", "scripts", "benchmarks"], capture_output=True, text=True, timeout=10)
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- results files

STRICT = ("config_hash", "agent", "track", "pool", "record")


def new_meta(config: dict, agent: str, track: str, pool: str, record: bool, planned: int,
             sim_commit: str | None, harness: str | None) -> dict:
    return {"benchmark": f"{config['name']}-v{config['version']}", "config_hash": config_hash(config),
            "config": {k: v for k, v in config.items() if not k.startswith("_")}, "agent": agent_kind(agent),
            "track": track, "rules": track_rules(config, track), "pool": pool, "sim": config_sim(config),
            "record": bool(record), "planned_games": planned, "first_seed": config["pools"][pool]["first_seed"],
            "sim_commit": sim_commit, "harness_commit": harness}


def check_meta(have: dict, want: dict, keys=STRICT) -> list:
    """Raise ConfigMismatch when `have` (an existing results file) and `want` differ in a strict key or in
    the simulator commit; return warnings for a different harness commit."""
    for key in keys:
        if have.get(key) != want.get(key):
            raise ConfigMismatch(f"refusing to merge: {key} differs ({have.get(key)!r} vs {want.get(key)!r})")
    if have.get("sim_commit") and want.get("sim_commit") and have["sim_commit"] != want["sim_commit"]:
        raise ConfigMismatch(f"refusing to merge: simulator commit differs ({have['sim_commit']} vs "
                             f"{want['sim_commit']}); games on another simulator are another benchmark")
    warnings = []
    if have.get("harness_commit") != want.get("harness_commit"):
        warnings.append(f"harness commit differs ({have.get('harness_commit')} vs {want.get('harness_commit')}); "
                        "every game records its own")
    return warnings


def load_results(path) -> dict:
    doc = json.loads(Path(path).read_text())
    if "meta" not in doc or "games" not in doc:
        raise ValueError(f"{path} is not a benchmark results file")
    return doc


def save_results(path, doc: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    os.replace(tmp, path)


def merge_results(docs: list, config: dict | None = None) -> dict:
    """One results document from several (the same benchmark run split up, e.g. resumed elsewhere). Refuses
    documents of another configuration hash (or agent, track, pool, recording, simulator commit); with
    `config`, also documents not made under that configuration. A seed present twice is kept once."""
    if not docs:
        raise ValueError("no results to merge")
    base = docs[0]["meta"]
    if config is not None and base["config_hash"] != config_hash(config):
        raise ConfigMismatch(f"refusing results of config hash {base['config_hash']}; this configuration is "
                             f"{config_hash(config)}")
    warnings = []
    for doc in docs[1:]:
        warnings += check_meta(doc["meta"], base)
    games, seen = [], set()
    for doc in docs:
        for game in doc["games"]:
            if game["seed"] not in seen:
                seen.add(game["seed"])
                games.append(game)
    games.sort(key=lambda g: g["seed"])
    return {"meta": dict(base), "games": games, "warnings": warnings}


# --------------------------------------------------------------------------- one game

def play_one(job: tuple) -> dict:
    """Worker: (config, agent, track, pool, seed, record) -> one game record."""
    config, agent, track, pool, seed, record = job
    from tfteval import make_policy, play_game

    lobby = sample_lobby(config, pool, seed, agent)
    seats = {seat: make_policy(spec) for seat, spec in lobby["seats"].items()}
    result = play_game(seats, seed, rules=track_rules(config, track), sim=config_sim(config),
                       pickers=bool(config.get("carousel_pickers", True)),
                       record=[lobby["hero_seat"]] if record else False)
    return {"seed": seed, "hero_seat": lobby["hero_seat"], "lobby": lobby["seats"],
            "stance_opponents": lobby["stance_opponents"], "result": result.to_json()}


# --------------------------------------------------------------------------- scorecard

def _mean_ci(values) -> dict:
    from tfteval.stats import _mean_ci as mean_ci

    return mean_ci(np.asarray(values, dtype=float)) if len(values) else {"n": 0, "mean": None, "ci95": None}


def hero_place(game: dict) -> int | None:
    return game["result"]["placements"].get(game["hero_seat"])


def strength(games: list) -> dict:
    """Mean placement with a 95% interval over games, top-4 and win rates with Wilson intervals, histogram."""
    from tfteval.rubric import wilson

    places = [hero_place(g) for g in games if hero_place(g) is not None]
    out = _mean_ci(places)
    out.update(top4=wilson(sum(p <= 4 for p in places), len(places)), win=wilson(sum(p == 1 for p in places),
                                                                               len(places)),
               histogram=np.bincount(places, minlength=9)[1:].tolist() if places else [0] * 8,
               sd=float(np.std(places, ddof=1)) if len(places) > 1 else None)
    return out


def by_stance_opponents(games: list) -> dict:
    groups = defaultdict(list)
    for g in games:
        groups[g["stance_opponents"]].append(g)
    return {str(k): strength(v) for k, v in sorted(groups.items())}


def hero_scores(games: list, thresholds: dict | None = None, comps: dict | None = None) -> list:
    from tfteval.rubric import score_result

    out = []
    for g in games:
        if g["result"].get("records", {}).get(g["hero_seat"]):
            out += score_result(g["result"], seats=[g["hero_seat"]], comps=comps, thresholds=thresholds)
    return out


def variety_games(games: list, signal_round: str, comps: dict | None = None) -> list:
    """Top-4 games of the hero with its final comp and early signals (recorded games only)."""
    from tfteval.variety import early_signals, final_comp

    out = []
    for g in games:
        rows = g["result"].get("records", {}).get(g["hero_seat"])
        place = hero_place(g)
        if not rows or place is None or place > 4:
            continue
        out.append({"seed": g["seed"], "place": place, "comp": final_comp(rows, comps),
                    "signals": early_signals(rows, signal_round)})
    return out


def variety(games: list, config: dict, compare: list | None = None, comps: dict | None = None) -> dict:
    from tfteval.stats import paired_diff
    from tfteval.variety import comp_distribution, noninferiority, responsiveness

    if comps is None:
        from tfteval.rubric import default_comps

        comps = default_comps()
    cfg = config["variety"]
    top = variety_games(games, cfg["signal_round"], comps)
    out = {"top4_games": len(top), "final_comps": comp_distribution([g["comp"] for g in top], len(comps)),
           "responsiveness": responsiveness(top, cfg.get("permutations", 1000))}
    diff = None
    if compare:
        try:
            diff = paired_diff([g["result"] for g in games], [g["result"] for g in compare], HERO, HERO)
        except ValueError:
            diff = None
        other = variety_games(compare, cfg["signal_round"], comps)
        if other:
            out["comparison_final_comps"] = comp_distribution([g["comp"] for g in other], len(comps))
    ni = noninferiority(diff, cfg["noninferiority_margin"])
    if compare and ni["noninferior"] and "comparison_final_comps" in out:
        more = out["final_comps"]["entropy_bits"] > out["comparison_final_comps"]["entropy_bits"]
        ni["variety_plus"] = bool(more)
        ni["note"] += "; more varied than the comparison" if more else "; but not more varied than the comparison"
    else:
        ni["variety_plus"] = False if compare else None
    out["noninferiority"] = ni
    return out


def bank_regret(agent: str, rules: str, bank: str | None, scorer: str | None, cmd: str = BANK_CMD) -> dict:
    """Hook for the decision bank (tfteval/bank.py, scored by scripts/score_bank.py). If the script exists,
    run `cmd` (placeholders {python} {script} {bank} {scorer} {agent} {rules} {out}; by default `scorer`
    is the agent spec score_bank.py takes with --agent, e.g. `stance`) and include the JSON object the
    script writes to {out} as it is. Nothing is imported from it."""
    if not bank or not scorer:
        return {"status": "not run", "note": "give --bank and --bank-scorer to add decision-bank regret"}
    if not BANK_SCRIPT.exists():
        return {"status": "unavailable", "note": "scripts/score_bank.py not found (it comes with the decision bank)"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "regret.json"
        command = cmd.format(python=shlex.quote(sys.executable), script=shlex.quote(str(BANK_SCRIPT)),
                             bank=shlex.quote(str(bank)), scorer=shlex.quote(str(scorer)),
                             agent=shlex.quote(agent), rules=shlex.quote(rules), out=shlex.quote(str(out)))
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True, cwd=str(ROOT))
        if proc.returncode != 0 or not out.exists():
            return {"status": "error", "command": command, "returncode": proc.returncode,
                    "stderr": proc.stderr[-2000:]}
        return {"status": "ok", "command": command, "result": json.loads(out.read_text())}


def scorecard(doc: dict, config: dict, compare: dict | None = None, bank: dict | None = None,
              comps: dict | None = None) -> dict:
    """The scorecard of a results document (merge_results / load_results). `compare`: another agent's
    results document under the same configuration, track and pool (for the variety non-inferiority note);
    `bank`: the output of bank_regret."""
    meta = doc["meta"]
    if meta["config_hash"] != config_hash(config):
        raise ConfigMismatch(f"results are of config hash {meta['config_hash']}, not {config_hash(config)}")
    if compare is not None:
        check_meta(compare["meta"], meta, keys=("config_hash", "track", "pool"))
    games = [g for g in doc["games"] if g["result"]["finished"]]
    played = [g["seed"] for g in doc["games"]]
    card = {"meta": {k: meta[k] for k in ("benchmark", "config_hash", "agent", "track", "rules", "pool", "sim",
                                          "record", "planned_games", "first_seed")},
            "games": {"finished": len(games), "unfinished": len(doc["games"]) - len(games),
                      "planned": meta["planned_games"],
                      "seeds": [min(played, default=None), max(played, default=None)]},
            "sim_commits": sorted({str(g["result"].get("sim_commit")) for g in doc["games"]}),
            "harness_commits": sorted({str(g.get("harness_commit")) for g in doc["games"]}),
            "warnings": list(doc.get("warnings", []))}
    card["strength"] = strength(games)
    card["by_stance_opponents"] = by_stance_opponents(games)
    card["gates"] = {"hero_fallbacks": int(sum(g["result"]["fallbacks"].get(g["hero_seat"], 0) for g in games)),
                     "all_seat_fallbacks": int(sum(sum(g["result"]["fallbacks"].values()) for g in games))}
    recorded = [g for g in games if g["result"].get("records", {}).get(g["hero_seat"])]
    card["recorded_games"] = len(recorded)
    if recorded:
        from tfteval.rubric import rates

        scores = hero_scores(recorded, config.get("rubric"), comps)
        card["rubric"] = rates(scores, key="policy").get(HERO)
        card["variety"] = variety(recorded, config, compare["games"] if compare else None, comps)
    else:
        card["rubric"] = card["variety"] = None
    card["regret"] = bank or {"status": "not run"}
    if meta.get("pool") == "heldout":
        card["warnings"].append("held-out pool: for milestones only, never for tuning")
    return card


def _pct(w: dict) -> str:
    if not w or w.get("rate") is None:
        return "n/a"
    return f"{w['rate']:.0%} [{w['low']:.0%}, {w['high']:.0%}]"


def _place(s: dict) -> str:
    if s.get("mean") is None:
        return "n/a"
    ci = f" ±{s['ci95']:.2f}" if s.get("ci95") is not None else ""
    return f"{s['mean']:.2f}{ci}"


def scorecard_markdown(card: dict) -> str:
    m, s, n, gates = card["meta"], card["strength"], card["games"], card["gates"]
    sd = f", SD {s['sd']:.2f}" if s.get("sd") is not None else ""
    lines = [f"### {m['benchmark']} · {m['agent']} · {m['track']} · {m['pool']} pool",
             "",
             f"config {m['config_hash']}, sim {m['sim']}, rules {m['rules']}; {n['finished']}/{n['planned']} games "
             f"(seeds {n['seeds'][0]}-{n['seeds'][1]}), {n['unfinished']} unfinished; "
             f"simulator {', '.join(card['sim_commits'])}; harness {', '.join(card['harness_commits'])}",
             "",
             "| Strength | Value |", "|---|---|",
             f"| Mean placement (95% CI) | {_place(s)} (n={s['n']}{sd}) |",
             f"| Top-4 rate | {_pct(s['top4'])} |",
             f"| Win rate | {_pct(s['win'])} |",
             f"| Placement histogram 1..8 | {' '.join(map(str, s['histogram']))} |",
             f"| Policy errors (hero / all seats) | {gates['hero_fallbacks']} / {gates['all_seat_fallbacks']} |",
             "",
             "| Stance-family opponents | Games | Mean placement | Top-4 | Win |", "|---|---|---|---|---|"]
    for k, v in card["by_stance_opponents"].items():
        lines.append(f"| {k} | {v['n']} | {_place(v)} | {_pct(v['top4'])} | {_pct(v['win'])} |")
    r = card.get("rubric")
    lines.append("")
    if r:
        lines += ["| Rubric v1 (hero, per game) | Rate |", "|---|---|"]
        for item, w in r["items"].items():
            lines.append(f"| {item} | {_pct(w)} |")
        g = r["gates"]
        lines.append("| hard gates, totals: fallbacks / illegal / level_short (idle) / roll_below_floor | "
                     f"{g['fallbacks']['total']} / {g['illegal']['total']} / {g['level_short']['total']} "
                     f"({g['level_short_idle']['total']}) / {g['roll_below_floor']['total']} |")
    else:
        lines.append("Rubric and variety: not recorded (run with --record).")
    v = card.get("variety")
    if v:
        fc = v["final_comps"]
        comps = ", ".join(f"{c} {k}" for c, k in fc["counts"].items())
        lines += ["", f"Strategic variety (reported apart from strength; top-4 finishes only, n={v['top4_games']}):",
                  f"- final comps: {fc['distinct']} distinct, entropy {fc['entropy_bits']:.2f} of "
                  f"{fc['max_bits']:.2f} bits (effective {fc['effective_comps'] or 0:.1f} comps): {comps}"]
        resp = v["responsiveness"]
        for name in ("item", "chosen", "contested"):
            mi = resp.get(name) or {}
            if mi.get("mi_bits") is None:
                continue
            null = (f"; shuffled {mi['null_mean']:.3f}, p={mi['p_value']:.3f}" if mi.get("null_mean") is not None
                    else "")
            lines.append(f"- responsiveness to {name} at the signal round: MI {mi['mi_bits']:.3f} bits "
                         f"(Miller-Madow {mi['mi_miller_madow']:.3f}{null})")
        lines.append(f"- {v['noninferiority']['note']}")
    reg = card.get("regret") or {}
    lines += ["", f"Decision-bank regret: {reg.get('status')}"
              + (f" ({json.dumps(reg.get('result'))[:200]})" if reg.get("status") == "ok" else "")]
    for w in card.get("warnings", []):
        lines.append(f"\nWARNING: {w}")
    return "\n".join(lines)


def measured_sd(doc: dict) -> float | None:
    places = [hero_place(g) for g in doc["games"] if g["result"]["finished"] and hero_place(g) is not None]
    return float(np.std(places, ddof=1)) if len(places) > 1 else None


def needed_note(config: dict, sd: float | None, half_width: float | None) -> str:
    size = config["sample_size"]
    sd, hw = sd or size["sd_per_game"], half_width or size["half_width"]
    n = games_planned(config, sd, hw)
    return f"{n} games for a 95% interval of ±{hw} places at SD {sd} per game (tfteval.stats.games_needed)"

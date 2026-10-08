"""Fit the single-fight win-probability model (tfteval/winprob.py) on recorded fights.

    python scripts/fit_winprob.py results/fights/mixed_9600.jsonl.gz --out tfteval/data/winprob.json

Rows: every recorded PvP fight that was not a draw (draws are counted and left out), seen from the
blue side (the matchup's first seat); y = 1 when blue won. x = the difference blue - red of the
per-player features (winprob.derived); a constant column is the blue-side term. Two models are fitted:
`start` on the views at the start of the planning phase, `fight` on the boards that fought.

Split by game, never by fight: TEST_SHARE of the games (a fixed shuffle of the seeds) are held out.
The term set of each model is chosen by grouped CV_FOLDS-fold cross-validation on the training games
(log-loss); the chosen set is refitted on all training games and scored once on the held-out games:
log-loss, AUC, Brier score, accuracy and a calibration table, with p averaged over the two sides as
p_win does. Logistic regression by Newton's method with a small ridge (RIDGE, on standardized columns);
numpy only.

`--compare MODEL.json` also scores a saved model (e.g. the current tfteval/data/winprob.json, read
before --out overwrites it) on the same held-out games, and on all games (it was fitted elsewhere);
`--replace-if-better` writes --out only when the new fit's held-out log-loss is no worse than the
compared model's for both `start` and `fight`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfteval import stages, winprob  # noqa: E402

TEST_SHARE = 0.2
CV_FOLDS = 5
SPLIT_SEED = 0
RIDGE = 1.0
TERM_SETS = {
    "score": ["score"],
    "log_score": ["log_score"],
    "score+level": ["score", "level"],
    "review": ["score", "level", "star2", "items"],  # the second opinion's list
    "board": ["score", "level", "units", "star2", "star3", "items"],
    "board+hp": ["score", "level", "units", "star2", "star3", "items", "hp"],
    "log_board+hp": ["log_score", "level", "units", "star2", "star3", "items", "hp"],
    "parts+hp": ["value", "traits", "active", "level", "units", "star2", "star3", "items", "chosen", "hp"],
    "all": ["value", "traits", "active", "level", "units", "star2", "star3", "items", "chosen", "hp",
            "win_streak", "loss_streak", "log_score"],
}


def load(paths: list[str]) -> list[dict]:
    """One row per non-draw fight: seed, round, ghost, y, derived features of both sides at start and fight."""
    rows, draws, cache = [], 0, {}

    def feats(view):
        key = json.dumps(view, sort_keys=True)
        if key not in cache:
            cache[key] = winprob.derived(winprob.view_features(view))
        return cache[key]

    for path in paths:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                for fight in rec["fights"]:
                    if fight["won"] == 0:
                        draws += 1
                        continue
                    a, b = fight["a"], fight["b"]
                    row = {"seed": rec["seed"], "round": rec["round"], "ghost": fight["ghost"],
                           "y": 1 if fight["won"] == 1 else 0, "damage": fight["damage"],
                           "fight": (feats(rec["fight"][a]), feats(rec["fight"][b]))}
                    if a in rec["start"] and b in rec["start"]:
                        row["start"] = (feats(rec["start"][a]), feats(rec["start"][b]))
                    rows.append(row)
    return rows, draws


def design(rows: list[dict], terms: list[str], when: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[r[when][0][t] - r[when][1][t] for t in terms] for r in rows], dtype=float)
    return np.column_stack([np.ones(len(rows)), x]), np.array([r["y"] for r in rows], dtype=float)


def fit(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE, iters: int = 50) -> np.ndarray:
    """Logistic regression; column 0 (the side term) is not penalized. Returns coefficients on X's scale."""
    scale = X.std(axis=0)
    scale[0] = 1.0
    scale[scale == 0] = 1.0
    Z = X / scale
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], ridge)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        grad = Z.T @ (p - y) + pen * w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w / scale


def predict(X: np.ndarray, w: np.ndarray, side_known: bool = False) -> np.ndarray:
    """P(blue wins). Unless side_known, averaged over the two sides (what p_win returns)."""
    z = X[:, 1:] @ w[1:]
    if side_known:
        return 1.0 / (1.0 + np.exp(-(z + w[0])))
    return 0.5 * (1.0 / (1.0 + np.exp(-(z + w[0]))) + 1.0 / (1.0 + np.exp(-(z - w[0]))))


def log_loss(y, p) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p) -> float:
    """Mann-Whitney AUC with ties counted half."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration(y, p, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "p_mean": round(float(p[m].mean()), 3),
                        "win_rate": round(float(y[m].mean()), 3)})
    return out


def metrics(y, p) -> dict:
    return {"n": int(len(y)), "log_loss": round(log_loss(y, p), 4), "auc": round(auc(y, p), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4), "accuracy": round(float(np.mean((p >= 0.5) == y)), 4)}


def split_seeds(seeds: list[int]) -> tuple[list[int], list[int]]:
    seeds = sorted(set(seeds))
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = [seeds[i] for i in rng.permutation(len(seeds))]
    n_test = max(1, round(TEST_SHARE * len(seeds)))
    return sorted(shuffled[n_test:]), sorted(shuffled[:n_test])


def cv_loss(rows: list[dict], terms: list[str], when: str, train_seeds: list[int]) -> float:
    folds = [train_seeds[k::CV_FOLDS] for k in range(CV_FOLDS)]
    losses, total = 0.0, 0
    for fold in folds:
        test = set(fold)
        tr = [r for r in rows if r["seed"] not in test]
        te = [r for r in rows if r["seed"] in test]
        if not te:
            continue
        w = fit(*design(tr, terms, when))
        X, y = design(te, terms, when)
        losses += log_loss(y, predict(X, w)) * len(te)
        total += len(te)
    return losses / total


def predict_spec(rows: list[dict], spec: dict, when: str) -> np.ndarray:
    """P(blue wins) under a saved model spec (winprob.json's models[when]), averaged over sides."""
    terms, side = spec["terms"], float(spec.get("side", 0.0))
    coef = np.array([float(spec["coef"][t]) for t in terms])
    z = np.array([[r[when][0][t] - r[when][1][t] for t in terms] for r in rows], dtype=float) @ coef
    return 0.5 * (1.0 / (1.0 + np.exp(-(z + side))) + 1.0 / (1.0 + np.exp(-(z - side))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="+", help="fight records (.jsonl.gz) from scripts/collect_fights.py")
    parser.add_argument("--out", help="write the models here (tfteval/data/winprob.json)")
    parser.add_argument("--compare", help="score this saved model on the same held-out games")
    parser.add_argument("--replace-if-better", action="store_true",
                        help="write --out only if the new held-out log-loss is <= the compared model's")
    args = parser.parse_args()
    old = json.loads(Path(args.compare).read_text()) if args.compare else None

    rows, draws = load(args.data)
    seeds = sorted({r["seed"] for r in rows})
    train_seeds, test_seeds = split_seeds(seeds)
    test_set = set(test_seeds)
    print(f"{len(rows)} fights ({sum(r['ghost'] for r in rows)} against a ghost board), {draws} draws left out, "
          f"{len(seeds)} games: {len(train_seeds)} train / {len(test_seeds)} held out")
    print(f"blue side won {np.mean([r['y'] for r in rows]):.3f} of the fights")

    out = {"version": 1, "data": {"files": [Path(p).as_posix() for p in args.data], "games": len(seeds),
                                  "fights": len(rows), "draws_left_out": draws, "test_seeds": test_seeds},
           "models": {}}
    for when in ("start", "fight"):
        rs = [r for r in rows if when in r]
        train = [r for r in rs if r["seed"] not in test_set]
        test = [r for r in rs if r["seed"] in test_set]
        print(f"\n== {when}: {len(train)} train fights, {len(test)} held-out fights")
        cv = {name: cv_loss(train, terms, when, train_seeds) for name, terms in TERM_SETS.items()}
        for name, loss in sorted(cv.items(), key=lambda kv: kv[1]):
            print(f"  cv log-loss {loss:.4f}  {name}: {', '.join(TERM_SETS[name])}")
        best = min(cv, key=cv.get)
        terms = TERM_SETS[best]
        w = fit(*design(train, terms, when))
        X, y = design(test, terms, when)
        p = predict(X, w)
        held = metrics(y, p)
        held_side = metrics(y, predict(X, w, side_known=True))
        base = {}
        for name in ("score", "log_score"):
            wb = fit(*design(train, TERM_SETS[name], when))
            Xb, _ = design(test, TERM_SETS[name], when)
            base[name] = metrics(y, predict(Xb, wb))
        by_stage = {}
        stage_of = np.array([int(stages.label(r["round"]).split("-")[0]) for r in test])
        for s in sorted(set(stage_of.tolist())):
            m = stage_of == s
            if m.sum() >= 20 and 0 < y[m].mean() < 1:
                by_stage[str(s)] = metrics(y[m], p[m])
        cal = calibration(y, p)
        print(f"  chosen: {best}; held-out {held}")
        print(f"  side known: {held_side}")
        for name, mt in base.items():
            print(f"  baseline {name}: {mt}")
        print("  by stage: " + "; ".join(f"{s}: n={m['n']} ll={m['log_loss']} auc={m['auc']}" for s, m in by_stage.items()))
        print("  calibration (held out):")
        for c in cal:
            print(f"    p {c['bin']}  n={c['n']:4d}  mean p {c['p_mean']:.3f}  won {c['win_rate']:.3f}")
        compared = None
        if old is not None and when in old["models"]:
            spec = old["models"][when]
            y_all = np.array([r["y"] for r in rs], dtype=float)
            compared = {"file": args.compare, "fitted_on": old.get("data", {}).get("files"), "terms": spec["terms"],
                        "held_out": metrics(y, predict_spec(test, spec, when)),
                        "all_games": metrics(y_all, predict_spec(rs, spec, when)),
                        "held_out_calibration": calibration(y, predict_spec(test, spec, when))}
            print(f"  compared model {args.compare}: held-out {compared['held_out']}")
            print(f"    all {len(rs)} fights: {compared['all_games']}")
            print("    calibration (held out): " + "; ".join(
                f"{c['bin']} n={c['n']} p={c['p_mean']:.2f} won={c['win_rate']:.2f}"
                for c in compared["held_out_calibration"]))
        coef = {t: float(c) for t, c in zip(terms, w[1:])}
        print("  coefficients: side " + f"{w[0]:+.4f}  " + "  ".join(f"{t} {c:+.4f}" for t, c in coef.items()))
        out["models"][when] = {
            "terms": terms, "coef": {t: round(c, 6) for t, c in coef.items()}, "side": round(float(w[0]), 6),
            "chosen_by_cv": best, "cv_log_loss": {k: round(v, 4) for k, v in cv.items()},
            "held_out": held, "held_out_side_known": held_side, "held_out_baselines": base,
            "held_out_by_stage": by_stage, "calibration": cal,
            "train_fights": len(train), "ridge": RIDGE,
        }
        if compared is not None:
            out["models"][when]["compared"] = compared
    if args.out and args.replace_if_better and old is not None:
        worse = [w for w, m in out["models"].items() if "compared" in m
                 and m["held_out"]["log_loss"] > m["compared"]["held_out"]["log_loss"]]
        if worse:
            print(f"\nnot written: the new fit is worse than {args.compare} on held-out log-loss ({', '.join(worse)})")
            return
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

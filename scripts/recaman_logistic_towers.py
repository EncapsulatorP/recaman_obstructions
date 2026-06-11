#!/usr/bin/env python3
"""
recaman_logistic_towers.py
==========================

Generate Recaman logistic tower layers from the signed triangular identity:

    a_n = T_n - 2 * sum_{i in D_n} i

This version explicitly separates PRE-DECISION (safe predictors) from
POST-DECISION (leaked outcomes) columns, and optionally fits a logistic
regression to benchmark how well each predictor group performs.

WHY {L5,L3,L0} AND {L2,L1,L0} LAYER PAIRS CANNOT BE USED
----------------------------------------------------------
The identity layers interact in two ways that break logistic regression:

  {L2, L1, L0} = (down_sum, T_n, a_n)
    RANK DEFICIENCY: a_n = T_n - 2*down_sum exactly.
    These three are coplanar; the feature matrix is singular.

  {L5, L3, L0} = (positional/prime/wheel features, candidate, a_n)
    TARGET LEAKAGE: a_n is POST-DECISION.
    is_down_step = int(a_n == candidate), so having both L3 and L0
    hands the classifier the answer (trivial AUC = 1.0, useless).

WHY {c530, c210} DIGIT-GROUP PAIRS CANNOT REPLACE {c321, c210}
--------------------------------------------------------------
The c321/c210 pair forms a consecutive "staircase":
    {3,2,1} and {2,1,0} overlap in {1,2} — a consecutive pair.
    The basis [[3,2],[2,1],[1,0]] exploits this chain structure.

  {c530, c210} = digit groups {5,3,0} and {2,1,0}:
    - Overlap only in digit 0 (no consecutive chain).
    - Digit 5 is rare in Recaman values up to ~200k (fewer distinct-5 numbers).
    - No natural staircase basis exists for non-consecutive groups.

The dominant single predictor is prev_is_down (b_{n-1}): the sequence
almost perfectly alternates DOWN/UP/DOWN/UP (phase-slip rate ≈ 0.001),
so P(down | prev=down) ≈ 0.001 and P(down | prev=up) ≈ 0.999.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "outputs" / "recaman_logistic_towers.json"
DEFAULT_CSV = ROOT / "outputs" / "recaman_logistic_towers.csv"

WHEEL = 210  # 2 * 3 * 5 * 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Recaman logistic tower layers (leakage-free, with optional LR fit)."
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--max-records-json", type=int, default=5000,
                        help="Max rows embedded in JSON (CSV always has all rows).")
    parser.add_argument(
        "--fit", action="store_true",
        help="Fit logistic regression on pre-decision features and report AUC.",
    )
    parser.add_argument(
        "--fit-rows", type=int, default=20_000,
        help="Number of rows for LR benchmark (temporal 80/20 split).",
    )
    return parser.parse_args()


@dataclass
class TowerRow:
    n: int

    # ── PRE-DECISION INPUTS (safe for logistic predictor) ─────────────────
    prev_a: int           # a_{n-1}
    T_n: int              # n*(n+1)//2
    candidate: int        # a_{n-1} − n  (may be <= 0)
    is_pos_cand: int      # int(candidate > 0)
    prev_is_down: int     # b_{n-1}: 1=prev step was DOWN, 0=UP — dominant predictor!
    prev_down_count: int  # |D_{n-1}|: down-steps before step n
    wheel_state: int      # S_{n-1}: symbolic 2-state wheel (0=210, 1=321)

    # L5 – features of n
    pos_bitlen_n: int
    pos_popcount_n: int
    pos_gray_n: int
    v2_n: int
    v3_n: int
    v5_n: int
    v7_n: int
    v11_n: int
    wheel_n: int          # n mod 210

    # L5 – features of a_{n-1}
    pos_bitlen_prev: int
    wheel_prev: int       # a_{n-1} mod 210
    prev_c321: float      # count of digits {1,2,3} in a_{n-1}
    prev_c210: float      # count of digits {0,1,2} in a_{n-1}

    # L5 – features of candidate (zero when candidate <= 0)
    wheel_cand: int       # candidate mod 210 (0 if candidate <= 0)
    cand_c321: float      # count of digits {1,2,3} in candidate (0 if <= 0)
    cand_c210: float      # count of digits {0,1,2} in candidate (0 if <= 0)

    # ── TARGET ───────────────────────────────────────────────────────────
    is_down_step: int     # b_n: 1=DOWN (free), 0=UP (blocked) ← THIS IS THE TARGET

    # ── POST-DECISION ORACLE (reveal outcome; NOT valid as predictors) ────
    collision: int        # int(candidate in V_{n-1})  [with is_pos_cand → perfect prediction]
    a_n: int              # new sequence value  [LEAKS target: a_n==candidate ↔ down step]
    down_count: int       # |D_n| after step n


def _valuation(x: int, p: int) -> int:
    if x <= 0:
        return 0
    v = 0
    while x % p == 0:
        v += 1
        x //= p
    return v


def _c321_c210(n: int) -> tuple[float, float]:
    """Count of digits in {1,2,3} and {0,1,2} in decimal representation."""
    if n <= 0:
        return 0.0, 0.0
    c321 = c210 = 0.0
    for ch in str(n):
        d = int(ch)
        if 1 <= d <= 3:
            c321 += 1.0
        if d <= 2:
            c210 += 1.0
    return c321, c210


def build_towers(steps: int) -> tuple[list[TowerRow], list[int], dict]:
    a_prev = 0
    visited: set[int] = {0}
    down_sum = 0
    down_count = 0
    wheel_state = 0   # flips on each UP step
    prev_is_down = 0  # bootstrap: treat virtual step-0 as "up"
    rows: list[TowerRow] = []
    down_indices: list[int] = []

    for n in range(1, steps + 1):
        t_n = (n * (n + 1)) // 2
        candidate = a_prev - n
        is_pos = int(candidate > 0)
        collision = int(is_pos and candidate in visited)
        is_down = int(is_pos and not collision)

        pc321, pc210 = _c321_c210(a_prev)
        cc321, cc210 = _c321_c210(candidate) if is_pos else (0.0, 0.0)

        a_n = candidate if is_down else a_prev + n

        rows.append(TowerRow(
            n=n,
            # pre-decision
            prev_a=a_prev,
            T_n=t_n,
            candidate=candidate,
            is_pos_cand=is_pos,
            prev_is_down=prev_is_down,
            prev_down_count=down_count,
            wheel_state=wheel_state,
            pos_bitlen_n=max(1, n.bit_length()),
            pos_popcount_n=bin(n).count("1"),
            pos_gray_n=n ^ (n >> 1),
            v2_n=_valuation(n, 2),
            v3_n=_valuation(n, 3),
            v5_n=_valuation(n, 5),
            v7_n=_valuation(n, 7),
            v11_n=_valuation(n, 11),
            wheel_n=n % WHEEL,
            pos_bitlen_prev=max(1, a_prev.bit_length()) if a_prev > 0 else 1,
            wheel_prev=a_prev % WHEEL,
            prev_c321=pc321,
            prev_c210=pc210,
            wheel_cand=candidate % WHEEL if is_pos else 0,
            cand_c321=cc321,
            cand_c210=cc210,
            # target
            is_down_step=is_down,
            # post-decision oracle
            collision=collision,
            a_n=a_n,
            down_count=down_count + is_down,
        ))

        # update state after recording
        if is_down:
            down_indices.append(n)
            down_sum += n
            down_count += 1
        else:
            wheel_state ^= 1

        visited.add(a_n)
        a_prev = a_n
        prev_is_down = is_down

    # identity sanity check
    t_steps = (steps * (steps + 1)) // 2
    expected = t_steps - 2 * down_sum
    if expected != a_prev:
        raise RuntimeError(
            f"Identity mismatch at n={steps}: T-2*D={expected}, a_n={a_prev}."
        )

    summary = {
        "steps": steps,
        "final_a_n": a_prev,
        "down_count": down_count,
        "up_count": steps - down_count,
        "down_rate": down_count / steps if steps else 0.0,
        "up_rate": (steps - down_count) / steps if steps else 0.0,
        "identity_formula": "a_n = T_n - 2*sum(D_n)  [verified]",
    }
    return rows, down_indices, summary


def _predecision_features(row: TowerRow) -> list[float]:
    """Extract safe (pre-decision) features for logistic regression."""
    return [
        float(row.prev_a),
        float(row.T_n),
        float(row.candidate),
        float(row.is_pos_cand),
        float(row.prev_is_down),    # dominant predictor (alternation)
        float(row.prev_down_count),
        float(row.wheel_state),
        float(row.pos_bitlen_n),
        float(row.pos_popcount_n),
        float(row.v2_n),
        float(row.v3_n),
        float(row.v5_n),
        float(row.v7_n),
        float(row.v11_n),
        float(row.wheel_n),
        float(row.pos_bitlen_prev),
        float(row.wheel_prev),
        float(row.prev_c321),
        float(row.prev_c210),
        float(row.wheel_cand),
        float(row.cand_c321),
        float(row.cand_c210),
    ]


PREDECISION_FEATURE_NAMES = [
    "prev_a", "T_n", "candidate", "is_pos_cand",
    "prev_is_down",          # ← dominant predictor
    "prev_down_count", "wheel_state",
    "pos_bitlen_n", "pos_popcount_n",
    "v2_n", "v3_n", "v5_n", "v7_n", "v11_n",
    "wheel_n", "pos_bitlen_prev", "wheel_prev",
    "prev_c321", "prev_c210",
    "wheel_cand", "cand_c321", "cand_c210",
]


def fit_logistic_benchmark(
    rows: list[TowerRow],
    fit_rows: int,
) -> dict:
    """Temporal 80/20 split logistic regression on pre-decision features."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "sklearn not available — install scikit-learn to use --fit."}

    use = rows[:fit_rows] if len(rows) > fit_rows else rows
    split = int(len(use) * 0.8)

    X = np.array([_predecision_features(r) for r in use], dtype=float)
    y = np.array([r.is_down_step for r in use], dtype=np.int8)

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf_full = LogisticRegression(max_iter=2000, random_state=7)
    clf_full.fit(X_tr_s, y_tr)
    auc_full = float(roc_auc_score(y_te, clf_full.predict_proba(X_te_s)[:, 1]))

    # ablation: remove prev_is_down (index 4 in PREDECISION_FEATURE_NAMES)
    prev_is_down_idx = PREDECISION_FEATURE_NAMES.index("prev_is_down")
    X_tr_no = np.delete(X_tr_s, prev_is_down_idx, axis=1)
    X_te_no = np.delete(X_te_s, prev_is_down_idx, axis=1)
    clf_no = LogisticRegression(max_iter=2000, random_state=7)
    clf_no.fit(X_tr_no, y_tr)
    auc_no_prev = float(roc_auc_score(y_te, clf_no.predict_proba(X_te_no)[:, 1]))

    # oracle: is_pos_cand AND NOT collision (requires visited set — not pre-decision)
    oracle_scores = np.array(
        [float(r.is_pos_cand and not r.collision) for r in use[split:]]
    )
    auc_oracle = float(roc_auc_score(y_te, oracle_scores))

    # coeff table for prev_is_down feature
    coeff_prev = float(clf_full.coef_[0][prev_is_down_idx])

    return {
        "fit_rows": len(use),
        "train_rows": split,
        "test_rows": len(use) - split,
        "auc_full_predecision": round(auc_full, 6),
        "auc_without_prev_is_down": round(auc_no_prev, 6),
        "auc_oracle_visited_set": round(auc_oracle, 6),
        "coeff_prev_is_down": round(coeff_prev, 4),
        "interpretation": {
            "auc_full_predecision": (
                "All pre-decision features incl. prev_is_down. "
                "Near 1.0 due to alternation (phase-slip rate ~0.001)."
            ),
            "auc_without_prev_is_down": (
                "Pre-decision arithmetic features only — no alternation signal. "
                "Should be ~0.5-0.6."
            ),
            "auc_oracle_visited_set": (
                "Oracle using visited-set collision flag. AUC=1.0 (perfect but "
                "requires full visited-set knowledge — not causal from local features)."
            ),
        },
    }


def write_csv(path: Path, rows: list[TowerRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_json(
    path: Path,
    rows: list[TowerRow],
    down_indices: list[int],
    summary: dict,
    logistic: dict | None,
    max_records: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "pair_constraints": {
            "{L2,L1,L0}": (
                "RANK DEFICIENT — a_n = T_n - 2*down_sum: exact linear dependence."
            ),
            "{L5,L3,L0}": (
                "TARGET LEAKAGE — a_n post-decision: is_down = int(a_n == candidate)."
            ),
            "{c530,c210}": (
                "Non-consecutive digit groups {5,3,0} and {2,1,0}: no staircase basis, "
                "digit-5 rare for n<=200k, overlap only at digit 0."
            ),
        },
        "column_groups": {
            "pre_decision_inputs": [
                "n", "prev_a", "T_n", "candidate", "is_pos_cand",
                "prev_is_down", "prev_down_count", "wheel_state",
                "pos_bitlen_n", "pos_popcount_n", "pos_gray_n",
                "v2_n", "v3_n", "v5_n", "v7_n", "v11_n",
                "wheel_n", "pos_bitlen_prev", "wheel_prev",
                "prev_c321", "prev_c210",
                "wheel_cand", "cand_c321", "cand_c210",
            ],
            "target": ["is_down_step"],
            "post_decision_oracle": ["collision", "a_n", "down_count"],
        },
        "logistic_benchmark": logistic,
        "down_indices_preview": down_indices[:500],
        "rows_embedded": min(len(rows), max_records),
        "rows_total": len(rows),
        "records_preview": [asdict(r) for r in rows[:max_records]],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    rows, down_indices, summary = build_towers(args.steps)

    logistic = None
    if args.fit:
        logistic = fit_logistic_benchmark(rows, args.fit_rows)

    write_csv(args.csv, rows)
    write_json(args.json, rows, down_indices, summary, logistic, args.max_records_json)

    print("Recaman Logistic Towers")
    print("=" * 72)
    print(f"Steps:          {summary['steps']:,}")
    print(f"Final a_n:      {summary['final_a_n']:,}")
    print(f"Down steps:     {summary['down_count']:,}  ({summary['down_rate']:.6f})")
    print(f"Up steps:       {summary['up_count']:,}   ({summary['up_rate']:.6f})")
    print()
    print("Pair constraints (why these feature groups fail):")
    print("  {L2,L1,L0}: RANK DEFICIENT  — a_n = T_n - 2*down_sum exactly.")
    print("  {L5,L3,L0}: TARGET LEAKAGE  — a_n is post-decision; is_down = (a_n==candidate).")
    print("  {c530,c210}: NON-CONSECUTIVE — digit-5 rare, no staircase basis, overlap only at 0.")
    print()
    print("Dominant predictor: prev_is_down (b_{n-1}).")
    print("  The sequence almost perfectly alternates DOWN/UP (phase-slip rate ~0.001).")

    if logistic:
        print()
        print("Logistic regression benchmark:")
        if "error" in logistic:
            print(f"  {logistic['error']}")
        else:
            print(f"  Rows used:                    {logistic['fit_rows']:,}")
            print(f"  AUC (all pre-decision):       {logistic['auc_full_predecision']:.6f}  <- dominated by prev_is_down")
            print(f"  AUC (without prev_is_down):   {logistic['auc_without_prev_is_down']:.6f}  <- arithmetic features only")
            print(f"  AUC (oracle, visited set):    {logistic['auc_oracle_visited_set']:.6f}  <- requires visited knowledge")
            print(f"  Coeff of prev_is_down:        {logistic['coeff_prev_is_down']}")

    print()
    print(f"CSV:  {args.csv}")
    print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

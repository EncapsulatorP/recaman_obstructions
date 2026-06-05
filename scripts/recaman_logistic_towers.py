#!/usr/bin/env python3
"""
recaman_logistic_towers.py
==========================

Generate Recaman logistic tower layers from the exact signed triangular identity:

    a_n = T_n - 2 * sum_{i in D_n} i

where:
    T_n = n(n + 1)/2,
    D_n = set of down-step indices up to n.

Layers emitted per step n:
  L0 = a_n
  L1 = T_n
  L2 = D_n (tracked via cumulative size/sum plus down-step flag)
  L3 = a_{n-1} - n
  L4 = 1[a_{n-1} - n in V_{n-1}] (plus full blocked flag for Recaman legality)
  L5 = positional encoding + prime tower + residue wheel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "outputs" / "recaman_logistic_towers.json"
DEFAULT_CSV = ROOT / "outputs" / "recaman_logistic_towers.csv"


@dataclass
class TowerRow:
    n: int
    L0_a_n: int
    L1_T_n: int
    L2_down_count: int
    L2_down_sum: int
    L2_is_down_step: int
    L3_candidate: int
    L4_collision_in_visited: int
    L4_blocked: int
    L5_pos_bitlen_n: int
    L5_pos_popcount_n: int
    L5_pos_gray_n: int
    L5_pos_bitlen_aprev: int
    L5_prime_v2_n: int
    L5_prime_v3_n: int
    L5_prime_v5_n: int
    L5_prime_v7_n: int
    L5_prime_v11_n: int
    L5_wheel_n_mod_210: int
    L5_wheel_aprev_mod_210: int
    L5_wheel_candidate_mod_210: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Recaman logistic tower layers from signed triangular identity."
    )
    parser.add_argument("--steps", type=int, default=200_000, help="Number of Recaman steps (n).")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="JSON output path.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV output path.")
    parser.add_argument(
        "--max-records-json",
        type=int,
        default=5000,
        help="Max number of rows to embed in JSON (CSV always contains all rows).",
    )
    return parser.parse_args()


def triangular(n: int) -> int:
    return (n * (n + 1)) // 2


def valuation(x: int, p: int) -> int:
    if x <= 0:
        return 0
    v = 0
    while x % p == 0:
        v += 1
        x //= p
    return v


def popcount(x: int) -> int:
    if x < 0:
        return 0
    return x.bit_count()


def bitlen(x: int) -> int:
    if x <= 0:
        return 1
    return x.bit_length()


def build_towers(steps: int) -> tuple[list[TowerRow], list[int], dict[str, int]]:
    a_prev = 0
    visited = {0}
    down_indices: list[int] = []
    down_sum = 0
    rows: list[TowerRow] = []

    for n in range(1, steps + 1):
        t_n = triangular(n)
        candidate = a_prev - n
        collision = int(candidate in visited)
        blocked = int(candidate <= 0 or collision == 1)
        is_down = int(blocked == 0)

        if is_down:
            down_indices.append(n)
            down_sum += n

        # Identity-first value construction.
        a_n = t_n - 2 * down_sum

        # Defensive consistency check against the direct rule target.
        if is_down:
            expected = candidate
        else:
            expected = a_prev + n
        if a_n != expected:
            raise RuntimeError(
                f"Identity mismatch at n={n}: identity={a_n}, rule={expected}."
            )

        rows.append(
            TowerRow(
                n=n,
                L0_a_n=a_n,
                L1_T_n=t_n,
                L2_down_count=len(down_indices),
                L2_down_sum=down_sum,
                L2_is_down_step=is_down,
                L3_candidate=candidate,
                L4_collision_in_visited=collision,
                L4_blocked=blocked,
                L5_pos_bitlen_n=bitlen(n),
                L5_pos_popcount_n=popcount(n),
                L5_pos_gray_n=n ^ (n >> 1),
                L5_pos_bitlen_aprev=bitlen(a_prev),
                L5_prime_v2_n=valuation(n, 2),
                L5_prime_v3_n=valuation(n, 3),
                L5_prime_v5_n=valuation(n, 5),
                L5_prime_v7_n=valuation(n, 7),
                L5_prime_v11_n=valuation(n, 11),
                L5_wheel_n_mod_210=n % 210,
                L5_wheel_aprev_mod_210=a_prev % 210,
                L5_wheel_candidate_mod_210=(candidate % 210),
            )
        )

        visited.add(a_n)
        a_prev = a_n

    summary = {
        "steps": steps,
        "final_a_n": a_prev,
        "down_count": len(down_indices),
        "down_sum": down_sum,
        "up_count": steps - len(down_indices),
        "blocked_rate": (steps - len(down_indices)) / steps if steps else 0,
        "identity_formula": "a_n = T_n - 2*sum_{i in D_n} i",
    }
    return rows, down_indices, summary


def write_csv(path: Path, rows: list[TowerRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_json(
    path: Path,
    rows: list[TowerRow],
    down_indices: list[int],
    summary: dict[str, int],
    max_records_json: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = [row.__dict__ for row in rows[: max(0, max_records_json)]]
    payload = {
        "summary": summary,
        "layers": {
            "L0": "a_n",
            "L1": "T_n",
            "L2": "D_n (down-step index set)",
            "L3": "a_{n-1} - n",
            "L4": "1[a_{n-1} - n in V_{n-1}]",
            "L5": "positional encoding + prime tower + residue wheel",
        },
        "down_indices": down_indices,
        "rows_embedded": len(preview),
        "rows_total": len(rows),
        "records_preview": preview,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    rows, down_indices, summary = build_towers(args.steps)
    write_csv(args.csv, rows)
    write_json(args.json, rows, down_indices, summary, args.max_records_json)

    print("Recaman Logistic Towers")
    print("=" * 72)
    print(f"Steps:               {summary['steps']:,}")
    print(f"Final a_n:           {summary['final_a_n']:,}")
    print(f"Down-step count:     {summary['down_count']:,}")
    print(f"Blocked (up) count:  {summary['up_count']:,}")
    print(f"Blocked rate:        {summary['blocked_rate']:.6f}")
    print(f"CSV:                 {args.csv}")
    print(f"JSON:                {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

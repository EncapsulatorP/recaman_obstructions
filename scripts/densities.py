#!/usr/bin/env python3
"""
Density summaries for the true Recaman obstruction process.

This script keeps several notions separate:

1. obstruction density:
   blocked steps / total steps
2. point-hit density:
   blocked steps where the backward candidate was already visited
3. boundary-block density:
   blocked steps where the backward candidate was <= 0
4. fill / hole density:
   how much of [0, max(a_n)] has been visited so far

The old draft mixed unrelated experiments into this file and reran the
generator from scratch for each scale. This version does one pass up to
the largest requested checkpoint and snapshots the statistics on the way.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class DensitySnapshot:
    step: int
    value: int
    max_value: int
    visited_points: int
    obstructions: int
    free_moves: int
    revisit_hits: int
    boundary_blocks: int
    obstruction_density: float
    free_density: float
    revisit_hit_density: float
    boundary_block_density: float
    fill_density: float
    hole_density: float
    balance_gap_vs_half: float


@dataclass(frozen=True)
class ValueBandDensity:
    start: int
    end: int
    width: int
    visited_points: int
    fill_density: float
    hole_density: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Recaman obstruction density at selected checkpoints."
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
        help="Checkpoint steps to report. Default: 100 1000 10000 100000",
    )
    parser.add_argument(
        "--value-bands",
        type=int,
        default=20,
        help="Approximate number of overlapping value-space windows at the final checkpoint.",
    )
    parser.add_argument(
        "--band-report-count",
        type=int,
        default=5,
        help="How many densest and sparsest value bands to print.",
    )
    return parser.parse_args()


def validate_steps(raw_steps: list[int]) -> list[int]:
    steps = sorted(set(raw_steps))
    if not steps:
        raise ValueError("At least one checkpoint is required.")
    if steps[0] <= 0:
        raise ValueError("All checkpoints must be positive integers.")
    return steps


def collect_snapshots(checkpoints: list[int]) -> tuple[list[DensitySnapshot], list[int]]:
    visited = {0}
    current = 0
    max_value = 0

    obstructions = 0
    free_moves = 0
    revisit_hits = 0
    boundary_blocks = 0

    snapshots: list[DensitySnapshot] = []
    checkpoint_idx = 0
    max_step = checkpoints[-1]

    for n in range(1, max_step + 1):
        backward = current - n
        if backward > 0 and backward not in visited:
            current = backward
            free_moves += 1
        else:
            obstructions += 1
            if backward <= 0:
                boundary_blocks += 1
            else:
                revisit_hits += 1
            current = current + n

        visited.add(current)
        if current > max_value:
            max_value = current

        if n != checkpoints[checkpoint_idx]:
            continue

        span_size = max_value + 1
        visited_points = len(visited)
        obstruction_density = obstructions / n
        free_density = free_moves / n
        revisit_hit_density = revisit_hits / n
        boundary_block_density = boundary_blocks / n
        fill_density = visited_points / span_size if span_size > 0 else 0.0
        hole_density = 1.0 - fill_density

        snapshots.append(
            DensitySnapshot(
                step=n,
                value=current,
                max_value=max_value,
                visited_points=visited_points,
                obstructions=obstructions,
                free_moves=free_moves,
                revisit_hits=revisit_hits,
                boundary_blocks=boundary_blocks,
                obstruction_density=obstruction_density,
                free_density=free_density,
                revisit_hit_density=revisit_hit_density,
                boundary_block_density=boundary_block_density,
                fill_density=fill_density,
                hole_density=hole_density,
                balance_gap_vs_half=obstruction_density - 0.5,
            )
        )

        checkpoint_idx += 1
        if checkpoint_idx == len(checkpoints):
            break

    return snapshots, sorted(visited)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_section_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def print_blocked_vs_free(snapshots: list[DensitySnapshot]) -> None:
    print_section_header("1. Blocked vs free moves")
    print(
        f"{'Steps':>10} | {'Obstructions':>12} | {'Free':>8} | "
        f"{'Obs density':>11} | {'Free density':>12} | {'Gap vs 50%':>10}"
    )
    print("-" * 79)
    for row in snapshots:
        print(
            f"{row.step:>10,} | {row.obstructions:>12,} | {row.free_moves:>8,} | "
            f"{pct(row.obstruction_density):>11} | {pct(row.free_density):>12} | "
            f"{pct(row.balance_gap_vs_half):>10}"
        )
    print()


def print_block_reason_breakdown(snapshots: list[DensitySnapshot]) -> None:
    print_section_header("2. Why steps were blocked")
    print(
        f"{'Steps':>10} | {'Point hits':>10} | {'Boundary':>10} | "
        f"{'Hit density':>11} | {'Boundary density':>16}"
    )
    print("-" * 70)
    for row in snapshots:
        print(
            f"{row.step:>10,} | {row.revisit_hits:>10,} | {row.boundary_blocks:>10,} | "
            f"{pct(row.revisit_hit_density):>11} | {pct(row.boundary_block_density):>16}"
        )
    print()


def print_span_coverage(snapshots: list[DensitySnapshot]) -> None:
    print_section_header("3. Coverage inside [0, max(a_n)]")
    print(
        f"{'Steps':>10} | {'a_n':>10} | {'max(a_n)':>10} | {'Visited':>10} | "
        f"{'Fill density':>12} | {'Hole density':>12}"
    )
    print("-" * 78)
    for row in snapshots:
        print(
            f"{row.step:>10,} | {row.value:>10,} | {row.max_value:>10,} | {row.visited_points:>10,} | "
            f"{pct(row.fill_density):>12} | {pct(row.hole_density):>12}"
        )
    print()


def build_value_bands(
    visited_points: list[int],
    max_value: int,
    approx_band_count: int,
) -> list[ValueBandDensity]:
    span_size = max_value + 1
    band_count = max(1, approx_band_count)
    width = max(1, (span_size + band_count - 1) // band_count)
    stride = max(1, width // 2)

    bands: list[ValueBandDensity] = []
    start = 0
    while start <= max_value:
        end = min(start + width - 1, max_value)
        left = bisect_left(visited_points, start)
        right = bisect_right(visited_points, end)
        visited_count = right - left
        actual_width = end - start + 1
        fill_density = visited_count / actual_width
        bands.append(
            ValueBandDensity(
                start=start,
                end=end,
                width=actual_width,
                visited_points=visited_count,
                fill_density=fill_density,
                hole_density=1.0 - fill_density,
            )
        )
        if end == max_value:
            break
        start += stride

    return bands


def print_value_band_structure(
    final_snapshot: DensitySnapshot,
    visited_points: list[int],
    approx_band_count: int,
    report_count: int,
) -> None:
    bands = build_value_bands(
        visited_points=visited_points,
        max_value=final_snapshot.max_value,
        approx_band_count=approx_band_count,
    )
    report_size = max(1, min(report_count, len(bands)))
    sparsest = sorted(bands, key=lambda band: (band.fill_density, band.start))[:report_size]
    densest = sorted(bands, key=lambda band: (-band.fill_density, band.start))[:report_size]
    min_fill = min(band.fill_density for band in bands)
    max_fill = max(band.fill_density for band in bands)

    print_section_header(f"4. Local hole density by value band at step {final_snapshot.step:,}")
    print(
        f"Using {len(bands)} overlapping windows across [0, {final_snapshot.max_value:,}] "
        f"with width about {bands[0].width:,}."
    )
    print(
        f"Local fill density ranges from {pct(min_fill)} to {pct(max_fill)}; "
        f"large spread suggests banding rather than uniform thinning."
    )
    print()

    print("Sparsest bands")
    print(
        f"{'Range start':>12} | {'Range end':>12} | {'Visited':>10} | "
        f"{'Fill density':>12} | {'Hole density':>12}"
    )
    print("-" * 70)
    for band in sparsest:
        print(
            f"{band.start:>12,} | {band.end:>12,} | {band.visited_points:>10,} | "
            f"{pct(band.fill_density):>12} | {pct(band.hole_density):>12}"
        )
    print()

    print("Densest bands")
    print(
        f"{'Range start':>12} | {'Range end':>12} | {'Visited':>10} | "
        f"{'Fill density':>12} | {'Hole density':>12}"
    )
    print("-" * 70)
    for band in densest:
        print(
            f"{band.start:>12,} | {band.end:>12,} | {band.visited_points:>10,} | "
            f"{pct(band.fill_density):>12} | {pct(band.hole_density):>12}"
        )
    print()


def main() -> None:
    args = parse_args()
    checkpoints = validate_steps(args.steps)
    snapshots, visited_points = collect_snapshots(checkpoints)

    print("Recaman obstruction density summary")
    print("==================================")
    print()
    print(
        "Definitions: obstruction = blocked backward move; "
        "point hit = blocked by a previously visited positive point."
    )
    print()

    print_blocked_vs_free(snapshots)
    print_block_reason_breakdown(snapshots)
    print_span_coverage(snapshots)
    print_value_band_structure(
        final_snapshot=snapshots[-1],
        visited_points=visited_points,
        approx_band_count=args.value_bands,
        report_count=args.band_report_count,
    )


if __name__ == "__main__":
    main()

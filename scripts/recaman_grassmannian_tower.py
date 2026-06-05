#!/usr/bin/env python3
"""
recaman_grassmannian_tower.py
=============================

Ranked prime tower / Grassmannian analysis of the Recaman obstruction stream.

Pipeline (from ranked_prime_tower_grassmannian_closure_final.pdf §2):
    [Recaman obstruction bits] → [Tower vectors over F_p] → [rank r(k) / Grassmannian φ_k]

Three analyses
--------------
1. 2-KERNEL RANK
   The 2-kernel of obstruction stream b_n = { b_{2^j·n+r} : j≥0, 0≤r<2^j }.
   Represent each arithmetic subsequence as a vector in F_2^D.
   Track cumulative F_2-rank as scale j increases.
   Stable rank ↔ finite 2-kernel ↔ b_n is 2-automatic.
   (Lemma 1, ranked_prime_tower_grassmannian_closure_final.pdf)

2. BRANCH GRASSMANNIAN  (w|0|w alignment)
   At each step n the Recaman rule chooses between:
     left  branch: candidate = a_{n-1} − n  (chosen when unblocked)
     right branch: forward  = a_{n-1} + n   (chosen when blocked)
   Recentred at a_{n-1}: (−n | 0 | +n) — the w|0|w structure.
   Build feature subspaces W_down and W_up from the taken/blocked candidate values.
   Compute principal angles (chordal Grassmannian distance).
   Compare to shuffled-label null (mandatory per Proposition 4.1,
   grassmannian_shadows.pdf).

3. SLIDING-WINDOW SHADOW RANK
   For window size w, form all length-w windows of the obstruction stream as
   rows of a binary matrix M and track rank(M) over F_2 as more windows are added.
   Fast saturation at low rank ↔ the stream is "shadow-collapsed" (few independent
   window patterns — expected for near-alternating streams).
   Shadow threshold analogy: §9.5 of ranked_prime_tower_grassmannian_closure_final.pdf.

Limitations (Principle 5.1, grassmannian_shadows.pdf)
------------------------------------------------------
  • The Grassmannian is order-blind: it detects rank/subspace/dependency, NOT
    cycle membership or orbit structure.
  • Small branch distance = shared coarse subspace geometry — evidence, not proof.
  • All results must be compared to null controls before making structural claims.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "outputs" / "recaman_grassmannian_tower.json"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recaman Grassmannian tower: 2-kernel rank, branch alignment, shadow collapse."
    )
    p.add_argument("--steps",    type=int, default=50_000, help="Recaman steps to generate.")
    p.add_argument("--vec-dim",  type=int, default=256,    help="F_2 vector length for kernel rows.")
    p.add_argument("--j-max",    type=int, default=12,     help="Max 2-kernel scale level j.")
    p.add_argument("--win-size", type=int, default=16,     help="Sliding-window size for shadow rank.")
    p.add_argument("--pca-k",    type=int, default=8,      help="Principal components for branch alignment.")
    p.add_argument("--seed",     type=int, default=7)
    p.add_argument("--json",     type=Path, default=DEFAULT_JSON)
    return p.parse_args()


# ── Recaman generator ──────────────────────────────────────────────────────────

def recaman_generate(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b) arrays of length N+1.
    a[n] = Recaman value, b[n] = 0 free/down, 1 blocked/up."""
    a = np.zeros(N + 1, dtype=np.int64)
    b = np.zeros(N + 1, dtype=np.int8)
    visited: set[int] = {0}
    for n in range(1, N + 1):
        cand = int(a[n - 1]) - n
        if cand > 0 and cand not in visited:
            a[n] = cand
            b[n] = 0
        else:
            a[n] = a[n - 1] + n
            b[n] = 1
        visited.add(int(a[n]))
    return a, b


# ── F_2 incremental rank tracker ───────────────────────────────────────────────

class F2Basis:
    """Online reduced row-echelon basis over F_2 (GF(2) Gaussian elimination)."""

    def __init__(self, ncols: int) -> None:
        self.ncols = ncols
        self._basis: list[np.ndarray] = []
        self._pivots: list[int] = []

    def add(self, row: np.ndarray) -> bool:
        """Try to add row. Returns True if rank increased (row was independent)."""
        r = row.astype(bool, copy=True)
        for i, p in enumerate(self._pivots):
            if r[p]:
                r ^= self._basis[i]
        nz = np.flatnonzero(r)
        if nz.size == 0:
            return False
        p = int(nz[0])
        # back-reduce existing basis rows with the new pivot
        for i in range(len(self._basis)):
            if self._basis[i][p]:
                self._basis[i] ^= r
        self._basis.append(r.copy())
        self._pivots.append(p)
        return True

    @property
    def rank(self) -> int:
        return len(self._basis)


# ── Feature encoder for real-valued branch analysis ───────────────────────────

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)
_WHEEL  = 210  # 2·3·5·7


def _encode(vals: np.ndarray) -> np.ndarray:
    """
    Encode integer array as (N × 14) float feature matrix.
    Features: log1p, 8 small-prime residues, wheel residue, bit_length,
              c321 (count of digits 1-3), c210 (count of digits 0-2).
    """
    N = len(vals)
    X = np.empty((N, 14), dtype=float)
    X[:, 0]  = np.log1p(vals)
    for i, p in enumerate(_PRIMES):
        X[:, 1 + i] = vals % p
    X[:, 9]  = vals % _WHEEL
    X[:, 10] = np.where(vals > 0, np.floor(np.log2(np.maximum(vals, 1))).astype(int) + 1, 1)
    # digit group counts (vectorised over common small values)
    c321 = np.zeros(N, dtype=float)
    c210 = np.zeros(N, dtype=float)
    for v_idx, v in enumerate(vals.tolist()):
        s = str(int(v))
        c321[v_idx] = sum(1 for ch in s if ch in "123")
        c210[v_idx] = sum(1 for ch in s if ch in "012")
    X[:, 11] = c321
    X[:, 12] = c210
    X[:, 13] = vals % 10  # last digit
    return X


# ── Analysis 1: 2-kernel rank ──────────────────────────────────────────────────

def two_kernel_analysis(
    b_stream: np.ndarray,
    j_max: int,
    D: int,
    rng: np.random.Generator,
) -> dict:
    """
    Build the 2-kernel of b_stream and track F_2-rank at each scale level j.

    At level j we add the 2^j arithmetic subsequences b[r::2^j] for r=0..2^j-1,
    each represented as its first D bits (zero-padded if shorter).

    Null control: same procedure on a random binary sequence with the same density.
    """
    N = len(b_stream)

    def _kernel_profile(seq: np.ndarray) -> list[dict]:
        tracker = F2Basis(D)
        profile = []
        for j in range(j_max + 1):
            step = 1 << j
            new_indep = 0
            for r in range(step):
                sub = seq[r::step]
                vec = np.zeros(D, dtype=bool)
                take = min(D, len(sub))
                vec[:take] = sub[:take].astype(bool)
                if tracker.add(vec):
                    new_indep += 1
            profile.append({
                "level": j,
                "subseqs_added": step,
                "cumul_subseqs": (1 << (j + 1)) - 1,
                "rank": tracker.rank,
                "new_independent": new_indep,
            })
        return profile

    real_profile  = _kernel_profile(b_stream)
    null_b        = (rng.random(N) < float(b_stream.mean())).astype(np.int8)
    null_profile  = _kernel_profile(null_b)

    # Check alternation rank: (0,1,0,1,...) stream
    alt_stream = np.arange(N, dtype=np.int8) % 2
    alt_profile = _kernel_profile(alt_stream)

    return {
        "vec_dim": D,
        "j_max": j_max,
        "stream_length": int(N),
        "obstruction_rate": float(b_stream.mean()),
        "real": real_profile,
        "null_random": null_profile,
        "pure_alternation": alt_profile,
        "interpretation": (
            "Pure alternation has 2-kernel rank 2 (two constant subsequences). "
            "Recaman b_n has phase-slip rate ~0.001, so expect rank slightly > 2. "
            "Random has rank ~ min(D, cumulative_subseqs) — grows fast. "
            "Rank stabilisation → finite 2-kernel → stream is 2-automatic. "
            "ARTIFACT WARNING: at scale j > floor(log2(N/D)) subsequences are shorter "
            "than D, so zero-padding forces artificial rank saturation that is NOT "
            "evidence of 2-automaticity. Trust only the rank values at j ≤ log2(N/D)."
        ),
    }


# ── Analysis 2: Branch Grassmannian (w|0|w) ───────────────────────────────────

def branch_grassmannian(
    a: np.ndarray,
    b: np.ndarray,
    pca_k: int,
    rng: np.random.Generator,
) -> dict:
    """
    Compare Grassmannian subspaces of down-branch vs up-branch candidate values.

    At each step n, the w|0|w structure is:
        (a_{n-1} - n) | a_{n-1} | (a_{n-1} + n)  →  recentred: (-n | 0 | +n)

    We encode the TAKEN candidates (down steps: a_n = a_{n-1}-n) and
    BLOCKED candidates (up steps where a_{n-1}-n > 0 but was in visited),
    then compare their feature subspaces.
    """
    N = len(b) - 1
    taken, blocked = [], []
    for n in range(1, N + 1):
        cand = int(a[n - 1]) - n
        if b[n] == 0:
            taken.append(cand)          # down: candidate was taken
        elif cand > 0:
            blocked.append(cand)        # up: candidate existed but was blocked

    taken   = np.array(taken,   dtype=np.int64)
    blocked = np.array(blocked, dtype=np.int64)

    min_n = min(len(taken), len(blocked))
    idx_t = rng.choice(len(taken),   min_n, replace=False)
    idx_b = rng.choice(len(blocked), min_n, replace=False)

    X_taken   = _encode(taken[idx_t])
    X_blocked = _encode(blocked[idx_b])

    def _principal_angles(X: np.ndarray, Y: np.ndarray, k: int) -> dict:
        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)
        _, _, Vhx = np.linalg.svd(Xc, full_matrices=False)
        _, _, Vhy = np.linalg.svd(Yc, full_matrices=False)
        k_eff = min(k, Vhx.shape[0], Vhy.shape[0])
        Qx = Vhx[:k_eff].T   # (d, k_eff)
        Qy = Vhy[:k_eff].T
        svals = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
        svals = np.clip(svals, 0.0, 1.0)
        angles = np.arccos(svals)
        chordal = float(np.sqrt(np.sum(np.sin(angles) ** 2)))
        return {
            "chordal_distance": round(chordal, 6),
            "principal_angles_deg": [round(float(a * 180 / np.pi), 3) for a in angles],
            "cosines": [round(float(s), 6) for s in svals],
            "k_used": k_eff,
        }

    result_tb = _principal_angles(X_taken, X_blocked, pca_k)

    # Null: shuffle candidate labels
    all_cands = np.concatenate([taken[idx_t], blocked[idx_b]])
    rng.shuffle(all_cands)
    X_null_a = _encode(all_cands[:min_n])
    X_null_b = _encode(all_cands[min_n:])
    result_null = _principal_angles(X_null_a, X_null_b, pca_k)

    # Also compare down-step values vs up-step values directly
    down_vals = a[1:][b[1:] == 0]
    up_vals   = a[1:][b[1:] == 1]
    min_v = min(len(down_vals), len(up_vals))
    X_down = _encode(down_vals[rng.choice(len(down_vals), min_v, replace=False)])
    X_up   = _encode(up_vals[rng.choice(len(up_vals),   min_v, replace=False)])
    result_du = _principal_angles(X_down, X_up, pca_k)

    return {
        "taken_count":   int(len(taken)),
        "blocked_count": int(len(blocked)),
        "sample_size":   int(min_n),
        "pca_k":         pca_k,
        "taken_vs_blocked_candidates": result_tb,
        "down_vs_up_values":           result_du,
        "null_shuffled_labels":        result_null,
        "interpretation": (
            "Small chordal distance = taken and blocked candidates live in similar "
            "feature subspaces (shared coarse geometry). "
            "Compare to null_shuffled_labels: if real distance ≈ null, no subspace "
            "separation. Subspace proximity is evidence only, not proof (Prop 2.3). "
            "The w|0|w structure: candidate = a_{n-1}-n recentred at 0 gives ±n pairs."
        ),
    }


# ── Analysis 3: Sliding-window shadow rank ─────────────────────────────────────

def shadow_rank_analysis(
    b_stream: np.ndarray,
    win_size: int,
    rng: np.random.Generator,
) -> dict:
    """
    Build all length-win_size windows of the obstruction stream as rows of a
    binary matrix over F_2 and track rank growth.

    For a pure alternating stream, only 2 distinct windows exist → rank = 2.
    Phase slips create new patterns → rank grows, but slowly.

    Null: same analysis on a random iid binary stream.
    """
    N = len(b_stream)
    n_windows = N - win_size + 1
    if n_windows <= 0:
        return {"error": "stream too short for window size"}

    # Checkpoints at which to record rank
    checkpoints = sorted({
        10, 50, 100, 500, 1000, 5000,
        10_000, 50_000, n_windows,
    } & set(range(1, n_windows + 1)))

    def _window_rank_profile(seq: np.ndarray) -> dict:
        tracker = F2Basis(win_size)
        ranks: dict[int, int] = {}
        for i in range(n_windows):
            window = seq[i: i + win_size].astype(bool)
            tracker.add(window)
            if i + 1 in checkpoints:
                ranks[i + 1] = tracker.rank
        return {
            "final_rank": tracker.rank,
            "max_possible_rank": win_size,
            "ranks_at_checkpoints": {str(k): v for k, v in sorted(ranks.items())},
        }

    real_profile = _window_rank_profile(b_stream)

    null_stream  = (rng.random(N) < 0.5).astype(np.int8)
    null_profile = _window_rank_profile(null_stream)

    alt_stream   = (np.arange(N) % 2).astype(np.int8)
    alt_profile  = _window_rank_profile(alt_stream)

    return {
        "win_size":         win_size,
        "n_windows":        n_windows,
        "real":             real_profile,
        "null_random":      null_profile,
        "pure_alternation": alt_profile,
        "interpretation": (
            f"Window size {win_size}: pure alternation has rank 2 (two distinct windows). "
            "Recaman b_n is near-alternating (phase-slip ~0.001), so expect rank just above 2. "
            f"Random iid stream reaches rank min({win_size}, n_windows) quickly. "
            "Low final rank → shadow collapse (stream lives on a low-dim F_2 subspace)."
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    print("Recaman Grassmannian Tower")
    print("=" * 72)
    t0 = time.perf_counter()
    print(f"Generating {args.steps:,} Recaman steps...")
    a, b = recaman_generate(args.steps)
    b_stream = b[1:]  # b[1..N]
    down_n = int((b_stream == 0).sum())
    up_n   = int((b_stream == 1).sum())
    print(f"  Down (free): {down_n:,}  Up (blocked): {up_n:,}  "
          f"Block rate: {up_n/args.steps:.6f}")
    print(f"  Time: {time.perf_counter()-t0:.2f}s")
    print()

    # ── Analysis 1 ──────────────────────────────────────────────────────────
    print(f"Analysis 1: 2-kernel rank  (D={args.vec_dim}, j_max={args.j_max})")
    t1 = time.perf_counter()
    kernel = two_kernel_analysis(b_stream, args.j_max, args.vec_dim, rng)
    print(f"  {'j':>3} | {'rank (real)':>11} | {'rank (alt)':>10} | {'rank (null)':>11} | new_indep")
    print(f"  {'---':>3}-+-{'----------':>11}-+-{'----------':>10}-+-{'----------':>11}-+----------")
    for rr, ra, rn in zip(kernel["real"], kernel["pure_alternation"], kernel["null_random"]):
        print(
            f"  {rr['level']:>3} | {rr['rank']:>11} | {ra['rank']:>10} | "
            f"{rn['rank']:>11} | {rr['new_independent']}"
        )
    import math
    j_valid = max(0, int(math.log2(args.steps / args.vec_dim)) - 1)
    print(f"  NOTE: valid range j≤{j_valid} (subseq length > D={args.vec_dim}).")
    print(f"  At j>{j_valid} zero-padding forces artificial rank saturation.")
    print(f"  Time: {time.perf_counter()-t1:.2f}s")
    print()

    # ── Analysis 2 ──────────────────────────────────────────────────────────
    print(f"Analysis 2: Branch Grassmannian  (w|0|w, pca_k={args.pca_k})")
    t2 = time.perf_counter()
    branch = branch_grassmannian(a, b, args.pca_k, rng)
    tb = branch["taken_vs_blocked_candidates"]
    du = branch["down_vs_up_values"]
    nl = branch["null_shuffled_labels"]
    print(f"  Taken vs blocked candidates — chordal: {tb['chordal_distance']:.6f}")
    print(f"    principal angles (deg): {tb['principal_angles_deg']}")
    print(f"  Down vs up step values   — chordal: {du['chordal_distance']:.6f}")
    print(f"    principal angles (deg): {du['principal_angles_deg']}")
    print(f"  Null (shuffled labels)   — chordal: {nl['chordal_distance']:.6f}")
    signal_tb = tb["chordal_distance"] - nl["chordal_distance"]
    # positive signal = real branches MORE separated than random shuffle → structural difference
    print(f"  Signal (taken/blocked - null): {signal_tb:+.6f}  "
          f"{'[BRANCH SEPARATION detected]' if signal_tb > 0.05 else '[weak or no branch separation]'}")
    print(f"  Time: {time.perf_counter()-t2:.2f}s")
    print()

    # ── Analysis 3 ──────────────────────────────────────────────────────────
    print(f"Analysis 3: Sliding-window shadow rank  (win={args.win_size})")
    t3 = time.perf_counter()
    shadow = shadow_rank_analysis(b_stream, args.win_size, rng)
    rr, ra, rn = shadow["real"], shadow["pure_alternation"], shadow["null_random"]
    print(f"  Max possible rank: {args.win_size}")
    print(f"  {'windows':>10} | {'rank (real)':>11} | {'rank (alt)':>10} | {'rank (null)':>11}")
    print(f"  {'-------':>10}-+-{'----------':>11}-+-{'----------':>10}-+-{'----------':>11}")
    all_cps = sorted(
        set(rr["ranks_at_checkpoints"]) |
        set(ra["ranks_at_checkpoints"]) |
        set(rn["ranks_at_checkpoints"]),
        key=int,
    )
    for cp in all_cps:
        r_r = rr["ranks_at_checkpoints"].get(cp, "—")
        r_a = ra["ranks_at_checkpoints"].get(cp, "—")
        r_n = rn["ranks_at_checkpoints"].get(cp, "—")
        print(f"  {int(cp):>10,} | {str(r_r):>11} | {str(r_a):>10} | {str(r_n):>11}")
    print(f"  Final rank — real: {rr['final_rank']}  alt: {ra['final_rank']}  "
          f"null: {rn['final_rank']}")
    print(f"  Time: {time.perf_counter()-t3:.2f}s")
    print()

    # ── Save ────────────────────────────────────────────────────────────────
    payload = {
        "config": {
            "steps":    args.steps,
            "vec_dim":  args.vec_dim,
            "j_max":    args.j_max,
            "win_size": args.win_size,
            "pca_k":    args.pca_k,
            "seed":     args.seed,
        },
        "two_kernel":          kernel,
        "branch_grassmannian": branch,
        "shadow_rank":         shadow,
        "grassmannian_principle": (
            "A Grassmannian shadow detects rank/subspace/dependency of a chosen "
            "representation (Lemma 2.2). It cannot detect an unrelated invariant "
            "(Lemma 2.1) unless null-controlled coupling is demonstrated (Prop 4.1). "
            "Ref: grassmannian_shadows.pdf + ranked_prime_tower_grassmannian_closure_final.pdf"
        ),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Total time: {time.perf_counter()-t0:.2f}s")
    print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

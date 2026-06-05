# Recaman Obstructions — Tower Analysis Results

**Scripts**: `scripts/recaman_logistic_towers.py` · `scripts/recaman_grassmannian_tower.py`
**Date**: 2026-06-05

---

## Background

The Recaman sequence is defined by:

```
a_0 = 0
a_n = a_{n-1} - n  if  > 0 and not yet visited  (DOWN / free)
    = a_{n-1} + n  otherwise                      (UP  / blocked)
```

The **obstruction bit** b_n = 0 (down) or 1 (up).  
Exact identity: `a_n = T_n − 2·∑_{i∈D_n} i` where T_n = n(n+1)/2.

**Key structural fact**: b_n almost perfectly alternates (0,1,0,1,...).  
Phase-slip rate ≈ 0.001 — fewer than 1 in 1000 steps breaks the alternation.

---

## Part 1 — Logistic Tower (`recaman_logistic_towers.py`)

**Run**: 200,000 steps · logistic regression benchmark on first 20,000 rows (80/20 temporal split)

### Why certain feature pairs cannot be used

| Pair | Problem |
|------|---------|
| `{L2, L1, L0}` — (a_n, T_n, down_sum) | **Rank deficient**: `a_n = T_n − 2·down_sum` is an exact linear identity → singular feature matrix |
| `{L5, L3, L0}` — (a_n, candidate, n) | **Target leakage**: `a_n` is post-decision; `is_down ≡ int(a_n == candidate)` trivially reveals the target |
| `{c530, c210}` — digit groups {5,3,0} vs {2,1,0} | **Non-consecutive**: no staircase basis; digit 5 is rare for n ≤ 200K; overlap only at digit 0 |

The valid encoding uses **c321** (digits 1,2,3) and **c210** (digits 0,1,2): consecutive overlap {1,2} gives a well-conditioned staircase basis [[3,2],[2,1],[1,0]].

### Feature groups (200,000 rows × 28 columns)

| Group | Columns |
|-------|---------|
| Pre-decision inputs (24) | n, prev_a, T_n, candidate, is_pos_cand, prev_is_down, prev_down_count, wheel_state, pos_bitlen_n, pos_popcount_n, pos_gray_n, v2_n … v11_n, wheel_n, pos_bitlen_prev, wheel_prev, prev_c321, prev_c210, wheel_cand, cand_c321, cand_c210 |
| Target (1) | is_down_step |
| Post-decision oracle (3) | collision, a_n, down_count |

### Logistic regression benchmark

| Model | AUC |
|-------|-----|
| All pre-decision features (incl. prev_is_down) | **0.9907** |
| Pre-decision arithmetic only (excl. prev_is_down) | **0.6791** |
| Oracle (visited-set collision flag) | **1.0000** |

**Dominant predictor**: `prev_is_down` with coefficient −3.4247.  
Removing it drops AUC from 0.991 → 0.679 — the alternation signal alone explains ~97% of predictability.

**Interpretation**: arithmetic features (prime valuations, wheel residues, digit counts, triangular number) contribute a non-trivial AUC of 0.679, meaningfully above 0.5, showing mild arithmetic structure in the obstruction stream beyond pure alternation. The oracle AUC of 1.0 confirms that the visited-set membership rule is the exact decision boundary — but it requires global state, not local arithmetic features.

---

## Part 2 — Grassmannian Tower (`recaman_grassmannian_tower.py`)

**Run**: 50,000 steps · D=256 · j_max=12 · win=16 · pca_k=8 · seed=7

Three null-controlled analyses (Proposition 4.1, grassmannian_shadows.pdf).

---

### Analysis 1 — 2-Kernel Rank over GF(2)

Each arithmetic subsequence `b[r::2^j]` is encoded as a vector in F_2^256.  
Rank is tracked cumulatively using online GF(2) Gaussian elimination.

**Valid range: j ≤ 7** (subseq length > D=256). At j ≥ 8 subsequences are zero-padded → artificial rank saturation.

| j | Recaman | Null random | Pure alt | new_indep (real) |
|---|---------|-------------|----------|-----------------|
| 0 | 1       | 1           | 1        | 1               |
| 1 | 3       | 3           | 2        | 2               |
| 2 | 7       | 7           | 2        | 4               |
| 3 | 15      | 15          | 2        | 8               |
| 4 | 31      | 31          | 2        | 16              |
| 5 | 63      | 63          | 2        | 32              |
| 6 | 127     | 127         | 2        | 64              |
| **7** | **221** | **255** | **2**  | **94**          |
| *(8)*  | *(251)* | *(256)* | *(4)* | *(30)*          |
| *(9+)* | *(253)* | *(256)* | *(6+)* | *(2, 0, 0, ...)* |

**Key finding at j=7** (last artifact-free scale):
- Recaman rank = **221** vs null = **255** vs alternation = **2**
- Gap from null: **−34 dimensions** (14% fewer independent directions)

The Recaman stream sits between pure alternation (rank 2 = minimal structure) and random noise (rank 255 = maximal). The 34-dimension deficit reflects long coherent runs with rare phase slips — many arithmetic subsequences share GF(2) linear dependencies that random sequences do not.

---

### Analysis 2 — Branch Grassmannian (w|0|w alignment)

At each step n the Recaman rule faces the triple:

```
(-n | 0 | +n)  centred at a_{n-1}
```

Feature subspaces are built for **taken** (down, candidate accepted) and **blocked** (up, candidate was already visited) candidates, then compared via principal angles / chordal Grassmannian distance.

| Comparison | Chordal distance | Max principal angle |
|------------|-----------------|---------------------|
| Taken vs blocked candidates | **0.0999** | 5.69° |
| Down vs up step values | **0.0721** | 4.11° |
| Null (shuffled labels) | **0.0246** | 1.09° |

**Signal (taken/blocked − null) = +0.075 → BRANCH SEPARATION detected**

The taken and blocked candidates live in measurably more different subspaces (4× the null distance). This shows the obstruction mechanism creates distributional structure: values that were safe to use as backward steps have different arithmetic characteristics (digit patterns, prime residues, wheel position) from values that were already occupied.

*Note*: all principal angles except the largest are < 0.6°, meaning the two subspaces share 7 of 8 principal directions almost exactly. The separation is driven by a single divergent direction (5.69°).

---

### Analysis 3 — Sliding-Window Shadow Rank

All length-16 windows of the obstruction stream are added as rows of a binary matrix; F_2 rank is tracked cumulatively.

| Windows seen | Recaman | Null random | Pure alternation |
|-------------|---------|-------------|-----------------|
| 10          | 10      | 10          | 2               |
| 50          | **16**  | **16**      | 2               |
| 100 – 49985 | 16      | 16          | 2               |

**Final rank**: Recaman = **16** (full) · Null = **16** · Pure alt = **2**

No shadow collapse: the Recaman obstruction stream saturates the full F_2^16 window space by the 50th window, identical to random noise. Pure alternation stays at rank 2 permanently.

The near-alternating structure (phase-slip rate 0.001) is invisible at window size 16: even ~50 phase slips across 50K steps create enough diverse window patterns to span the full space. To observe shadow collapse one would need `--win-size 4` or smaller, where a few hundred slips cannot fill all 2^4 = 16 patterns.

---

## Summary

| Property | Value | Interpretation |
|----------|-------|---------------|
| Down/up rate | 49.995% / 50.005% | Near-perfect symmetry |
| Phase-slip rate | ~0.001 | Dominant structure: alternation |
| AUC (all pre-decision) | 0.9907 | Alternation nearly determines outcome |
| AUC (arithmetic only) | 0.6791 | Mild arithmetic signal beyond alternation |
| prev_is_down coefficient | −3.4247 | Alternation is the strongest single predictor |
| 2-kernel rank at j=7 | 221 vs 255 (null) | −34 dims: GF(2) coherence from long alternating runs |
| Branch chordal distance | 0.100 vs 0.025 (null) | 4× null → distributional separation of taken/blocked |
| Shadow rank (win=16) | 16 (full) | No collapse; phase slips diversify window patterns |

**The obstruction stream is not random**: it is arithmetic-structured (alternation, phase slips, digit/wheel residues) but its local window patterns are diverse enough to span full binary subspaces. The GF(2) kernel analysis is the clearest structural signal — 34 fewer independent directions than random at scale j=7.

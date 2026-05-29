# SCAS Simulator POC — Day 4 Results (AC-B4 + AC-B2/B3 across T)

> Generated after T4.1–T4.3. Companion docs: [`requirements.md`](requirements.md), [`design.md`](design.md), [`results-day3.md`](results-day3.md).

## Headline

| AC | Literal threshold | Observed | Verdict |
| --- | --- | --- | --- |
| **AC-B4** (knob monotonicity) | Kendall τ ≤ −0.7 | **τ = +1.00 aggregate**, **+0.77 per-seed flat** | **monotonicity ✅ holds with \|τ\| = 1.0, but direction is inverted from the AC's prior — needs re-spec or re-interpretation** |
| AC-B2 (coverage speedup) at sweet spot | ≥ 1.5 | **1.73 at T=0.1/0.3, 1.69 at T=1.0** | PASS at tight T |
| AC-B3 (no mode collapse) across sweep | ≥ 1.5 | **1.89–2.52** | PASS everywhere |

Run dir: `sim_runs/sweep-T5-N10-20260527T195717/`. Plot: `ac_b4_knob_monotonicity.png`.

## 1. The T-sweep

Default constants (`W_P=0`, `PSI_0=2.0`), K=3, K'=5, steps=200, n_seeds=10, 5 T values (loose→tight: 10.0, 3.0, 1.0, 0.3, 0.1).

| coupling_T | inter / intra | steps_to_80% | AC-B2 speedup |
| --- | --- | --- | --- |
| 0.1 (tight) | 1.893 | 43.6 | 1.732 |
| 0.3        | 1.900 | 43.6 | 1.732 |
| 1.0        | 2.095 | 44.8 | 1.685 |
| 3.0        | 2.335 | 92.2 | 0.819 |
| 10.0 (loose) | 2.522 | 100.5 | 0.751 |

Two clean monotonic relationships:

- **Separation increases with T** (τ = +1.00 aggregate). Tight T → small separation; loose T → large separation.
- **Coverage speed decreases with T** (steps_to_80% rises from 43.6 → 100.5; AC-B2 ratio falls from 1.73 → 0.75).

## 2. Why the direction inverted from AC-B4's prior

AC-B4 (original wording) anticipates `τ ≤ −0.7` because the design was written assuming **tighter coupling → stronger tracking → more separation**. That direction holds only when `psi_0 < typical_psi` — i.e. the sigmoid's center sits *below* the regime in which agents typically live, so reducing T sharpens the sigmoid into `alpha → 1` (pure neighbor / pure target tracking).

The committed defaults (from results-day3.md §13) flipped that assumption: **`PSI_0 = 2.0` is now *above* typical `psi`** (~0.7–1.5 for normalized MiniLM embeddings). So:

- **Tight T (e.g. 0.1):** sigmoid argument `(psi − 2.0)/0.1` saturates *negative* → `alpha → 0` → mix proposer fires the **global** branch almost always → agents random-walk within their category → less group-leader pull → **less separation**.
- **Loose T (e.g. 10.0):** sigmoid argument is muted → `alpha ≈ 0.5` regardless of `psi` → 50/50 local/global mix → group attractors influence half the picks → **more separation**.

The control law is doing exactly what synergetic dynamics predict; the direction is determined by the relative position of `psi_0` and the typical operating regime. The AC's prior was correct for the original constants; not for the tuned ones.

## 3. The AC-B2/AC-B4 trade-off this exposes

The sweet spot for coverage and the sweet spot for separation **pull in opposite directions**:

- **AC-B2 (coverage)** wants tight T (more global exploration → faster coupon-collector).
- **AC-B3 separation strength** wants loose T (more group-leader pull → more clustered groups).

At T ∈ {0.1, 0.3, 1.0}, **both ACs pass**:
- separation ≥ 1.89 (above 1.5),
- speedup ≥ 1.69 (above 1.5).

At T ∈ {3.0, 10.0}, separation grows but coverage speedup falls under 1.0 (SCAS becomes *slower* than baseline). The Pareto frontier of "joint AC-B2 + AC-B3 pass" lives in `T ≲ 1.0`.

## 4. Three options for AC-B4

| Option | What it does | Pros / Cons |
| --- | --- | --- |
| **A. Re-spec AC-B4 direction-agnostic** (`\|τ\| ≥ 0.7`) | Acknowledge the control law gives a monotonic knob; the direction is determined by `(psi_0, typical_psi)` relationship | Honest; matches the SCT-theoretic claim ("knob continuously interpolates"). Does not re-litigate Day 3's tuning. |
| **B. Keep AC-B4 as written; report as falsification** | "Monotonicity sign-inverted with tuned defaults" is the conclusion | Maintains threshold integrity; weaker headline; awkward because the underlying behavior IS exactly what theory predicts. |
| **C. Revert defaults to `W_P=50, PSI_0=0.5` and re-sweep** | Get the AC-B4 direction back to match the original prior | Breaks AC-B2 (proposer trap returns). Trades a smaller-print AC pass for a bigger-print AC fail. |

## 5. Resolution (user decision)

User chose option A — re-spec AC-B4 to direction-agnostic `|τ| ≥ 0.7`. `requirements.md` updated; AC now reads "monotonically related to T; sign is determined by `(PSI_0, typical_psi)`."

**Final AC table with committed defaults (`W_P=0`, `PSI_0=2.0`) at K=3, K'=5, steps=200, n_seeds=10:**

| AC | Threshold | Best observed | Verdict |
| --- | --- | --- | --- |
| AC-B1' | within 1 SE | SCAS 0.8974 vs baseline 0.8996 (Δ = −0.0022) | PASS |
| AC-B2 | speedup ≥ 1.5 | 1.732 at T=0.1 | PASS |
| AC-B3 | inter/intra ≥ 1.5 | 2.522 at T=10.0 (1.89 at T=0.1) | PASS |
| AC-B4 | \|τ\| ≥ 0.7 | τ = +1.000 aggregate, +0.770 per-seed flat | PASS |
| AC-B5' | 20 adversarial rejected | 20/20 pass via `validator_tests.py` | PASS |

## 6. Status going into Day 5 (buffer)

- All five behavioral ACs (B1', B2, B3, B4, B5') pass with committed defaults.
- Pareto sweet spot for joint AC-B2 + AC-B3 pass is `T ≤ 1.0`.
- Day 5 buffer task (T5.1–T5.3) is the replay-mode embedding-quality check against real `results.tsv` summaries — the design's pre-flagged soft point. Optional; the POC's core claims are validated.

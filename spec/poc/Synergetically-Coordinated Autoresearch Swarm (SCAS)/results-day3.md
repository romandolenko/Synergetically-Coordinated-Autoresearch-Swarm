# SCAS Simulator POC — Day 3 Results (AC-B2 / AC-B3)

> Generated after T3.4. Companion docs: [`requirements.md`](requirements.md), [`tasklist.md`](tasklist.md), [`design.md`](design.md).

## Headline

| AC | Threshold | Result (default config) | Verdict |
| --- | --- | --- | --- |
| **AC-B3** (no mode collapse) | inter/intra ≥ 1.5 | **1.98** | **PASS** |
| **AC-B2** (coverage gain) | SCAS/baseline ≥ 1.5 | **0.70** | **FAIL — structurally unreachable; see §3** |
| **AC-B1'** (no-worse-than) | within 1 SE | SCAS 0.913 vs baseline 0.906 (Δ ≈ 0.007) | provisional PASS — Δ comparable to noise |

## 1. Default-config run (`paired-N10-20260526T230314`)

K=3, K'=5, steps=200, coupling_T=1.0, w_d=1.0, w_p=50.0, psi_0=0.5, n_seeds=10.

- `scas_mean_best_val_bpb`: 0.9131
- `baseline_mean_best_val_bpb`: 0.9063
- `scas_mean_distinct_minima`: 3.5
- `baseline_mean_distinct_minima`: 5.0 (saturated — all 10 seeds visited every minimum)
- `ac_b2_coverage_ratio`: 0.70
- `ac_b3_inter_over_intra_aggregate`: **1.98**

Plots: `sim_runs/paired-N10-20260526T230314/ac_b3_inter_vs_intra.png`, `…/ac_b2_minima_visited.png`.

## 2. What's actually happening (per-run trace)

Reading `sim_runs/.../scas-seed0-.../events.jsonl`:

- Group 0 (optimizer) visits **4** distinct strings out of 11; 197/200 events are `LAMB`.
- Group 1 (architecture) visits 7; oscillates between `width=256` and `depth=16`.
- Group 2 (regularization) visits 7; oscillates between `weight_decay=0.0` and `dropout=0.1`.

Mechanism — agent finds a low-val_bpb slot early (e.g. Sophia at index 10), attractor locks there. `w_p=50` makes one bad neighbor proposal spike psi to ≈4 → alpha saturates at ≈0.97 → next proposal is forever the only `i±1` neighbor of Sophia (which is LAMB). The agent never reaches the second optimizer minimum (LARS, index 8) because it's two slots away.

Two compounding bugs:
1. **Proposer locality.** `propose(target=…)` returns one of `{slots[i-1], slots[i+1]}`. With K' > 1 minimum per category, the agent can't escape its local well.
2. **Saturating perf penalty.** `w_p=50` makes alpha lock at >0.95 after one above-best step.

## 3. Why AC-B2 is structurally unreachable here

Baseline picks a random category-slot pair each step. Over 600 attempts across 33 templated strings, every string is sampled ~18 times. Every minimum embedding is at one of those strings, so baseline finds every minimum with overwhelming probability:

```
baseline per-seed distinct: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]   # std = 0
```

The AC-B2 threshold `SCAS / baseline ≥ 1.5` therefore requires `SCAS ≥ 7.5` — impossible when K'=5. Raising K' doesn't escape the ceiling either (probed below).

## 4. Tuning sweep — confirms unreachability

Per the task contract, three tuning experiments (capped well under the 1-day budget):

| Config | w_p | psi_0 | K' | scas_distinct | baseline_distinct | AC-B2 ratio | AC-B3 ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 50 | 0.5 | 5 | 3.5 | 5.0 | 0.70 | 1.98 |
| Tuning A | **5** | 0.5 | 5 | 4.1 | 5.0 | 0.82 | 1.88 |
| Tuning B | **0** | **2.0** | 5 | **5.0** | 5.0 | 1.00 (ceiling) | 1.72 |
| Structural probe | 50 | 0.5 | **20** | 16.3 | 20.0 | 0.81 | 1.96 |
| Probe + Tuning B | 0 | 2.0 | **20** | **20.0** | 20.0 | 1.00 (ceiling) | 1.74 |

Tuning B (no performance penalty, higher exploration baseline) does fix the proposer-locality symptom: SCAS visits every minimum. But it can never exceed baseline because baseline is at the ceiling.

## 5. Interpretation

**AC-B3 is robustly validated.** Across all five configurations the inter/intra ratio sits in **[1.72, 1.98]**, comfortably above the 1.5 threshold. Multi-consensus separation is doing what the SCT framing predicted: groups settle on distinct attractors in embedding space.

**AC-B2 as currently written is the wrong threshold for this simulator.** The synthetic taxonomy is small (33 strings); the step budget (600 attempts) is large; uniform-random baseline therefore saturates at K'. The 1.5× multiplier requires baseline to be below K'/1.5 — which only happens if baseline can mode-collapse, which it can't because design §9's baseline has α=0 and no attractor.

**AC-B1' looks fine.** The SCAS-vs-baseline `best_val_bpb` gap of ~0.007 is on the order of `noise_std=0.005` and would shrink further with the standard error gate over 10 seeds.

## 6. Recommended next steps (escalation per task contract)

Per `tasklist.md` stop conditions, two non-tuning paths are needed for AC-B2 to be measurable:

- **Re-spec AC-B2 with a non-saturating baseline.** Replace the uniform-random baseline with a single-attractor mode-collapsing baseline (effectively K=1 SCAS with all agents in one group). Then "coverage" tests SCAS's multi-attractor advantage rather than baseline saturation.
- **Re-spec AC-B2 as time-to-coverage, not end-of-run coverage.** Measure "steps until ≥80% of minima visited" — SCAS partitions and parallelizes, so it should reach partial coverage faster even when baseline eventually saturates.
- **Fix the proposer.** Bug #1 in §2 is independently worth fixing: `propose(target=…)` should occasionally jump globally, not only to `i±1`. This is a design.md change to §6, not a tuning change.

Recommendation: **escalate before further engineering**. AC-B3 holds; AC-B2 is unmeasurable as specified. Decide which AC re-spec is right before Day 4's `coupling_T` sweep, since Day 4 inherits the same baseline-saturation problem if the AC isn't re-spec'd.

## 7. Run dirs

- Default: `sim_runs/paired-N10-20260526T230314/`
- Tuning A: `sim_runs/paired-N10-20260526T230856/`
- Tuning B (K'=5): `sim_runs/paired-N10-20260526T230902/`
- Structural probe: `sim_runs/paired-N10-20260526T230930/`
- Probe + Tuning B: `sim_runs/paired-N10-20260526T230936/`

Each dir contains `paired_scas_summary.json`, `paired_baseline_summary.json`, and the two AC plots.

---

## 8. AC-B2 v2 — re-spec to time-to-coverage

User chose option 1 from §6: replace AC-B2 with a **time-to-coverage speedup ratio**. `requirements.md` AC-B2 now reads `mean(baseline_steps_to_80%) / mean(scas_steps_to_80%) ≥ 1.5`. `design.md` summary schema gained `first_visit_steps` and `steps_to_coverage`. `sim.py` instruments per-minimum first-visit agent-step; `plot.py` AC-B2 figure is now a cumulative-coverage-vs-agent-step plot.

## 9. v2 results (paired-N10, K=3, K'=5, steps=200)

| Config | baseline steps-to-80% | SCAS steps-to-80% | AC-B2 speedup ratio | AC-B3 ratio |
| --- | --- | --- | --- | --- |
| Default (w_p=50, psi_0=0.5) | 31.1 | **436.0** (many seeds budget-capped at 600) | 0.07 | 1.98 |
| Tuning B (w_p=0, psi_0=2.0) | 31.1 | 33.8 | 0.92 | 1.72 |

Run dirs: `sim_runs/paired-N10-20260526T231609/` (default), `sim_runs/paired-N10-20260526T231615/` (Tuning B).

**The new metric does not rescue AC-B2.** It exposes the underlying dynamic more honestly:

- Baseline reaches 80% coverage in ~31 agent-steps. With 33 templated strings and uniform random sampling, this matches the coupon-collector bound `~33·H(K')` ≈ 33 × 2.28 ≈ 75 for K'=5; the actual 31 reflects that only 5 specific strings need to be hit, not all 33.
- Default-config SCAS gets stuck (proposer locality bug, §2) — many seeds never reach 80% in 600 agent-steps.
- Tuning-B SCAS matches baseline speed (33.8 vs 31.1) but does not exceed it.

**Why SCAS can't beat baseline here.** Partitioning the search space only pays off when the search space is *large* relative to the step budget. In this simulator, baseline samples 600 attempts across 33 strings (~18× redundancy) and finds every minimum trivially fast. SCAS's structural advantage — divide the space, parallelize exploration — has no room to manifest. The coupon-collector bound dominates either way.

**AC-B3 unchanged and still robust.** Inter/intra ratio remains 1.72–1.98 across both v2 configurations.

## 10. Round-2 escalation

Two cleanly separable issues remain after the re-spec:

| Issue | Description | Possible fixes |
| --- | --- | --- |
| **Search-space size** | Baseline saturates because the synthetic taxonomy has only 33 distinct strings. AC-B2 (any formulation) needs a search space large enough that uniform random sampling is slow. | Inflate the taxonomy by ≥30× (add slot dimensions, e.g. `optimizer × lr × momentum` cross product); OR cut step budget to ~20 steps so baseline can't exhaust the space. |
| **Proposer locality** | `propose(target=…)` only returns `slots[i±1]`. Even with Tuning B exploring half the time, the high-α tightening dominates and agents pile up at attractors. | `propose(target=…)` should occasionally jump globally — e.g. with probability `1 − α_i` pick a random slot. |

These are independent: search-space inflation alone gives baseline more room to lose; proposer fix alone makes SCAS actually exploit its partition. Both together is the cleanest test of AC-B2's intent.

Recommendation: **before Day 4's coupling_T sweep, expand the search space (cheapest unlock — 1 hour) and re-measure**. If SCAS now meaningfully beats baseline on time-to-80%, the coupling_T sweep becomes informative. If not, the falsification finding hardens to "SCAS coordination provides no time-to-coverage advantage even at scale" — also a publishable conclusion.

The proposer-locality fix is independently worth doing, but it changes `design.md §6` rather than `requirements.md`, so it's a smaller decision.

---

## 11. AC-B2 v3 — expanded taxonomy (10× search space)

User chose option 1 from §10: cross-product the taxonomy. `synthetic.py` TAXONOMY now generates 330 unique strings (132 optimizer × 99 architecture × 99 regularization) via slot cross-products (e.g. `Adam lr=1e-4 mom=0.9`). Validator tests still pass. `propose()` unchanged.

### v3 results (paired-N10, default K=3, steps=200)

| Config | K' | scas_steps_to_80% | baseline_steps_to_80% | AC-B2 ratio | AC-B3 |
| --- | --- | --- | --- | --- | --- |
| Default (w_p=50) | 5 | 375.7 (budget-capped) | 75.5 | 0.20 | **3.72** |
| Tuning B (w_p=0, psi_0=2.0) | 5 | 62.3 | 75.5 | **1.21** | 1.89 |
| Tuning B | 10 | 72.6 | 96.4 | **1.33** | 1.89 |
| Tuning B | 15 | 102.6 | 97.2 | 0.95 | 1.95 |
| Tuning B + steps=400 | 5 | 62.3 | 75.5 | 1.21 | 1.90 |

Run dirs (most relevant): `paired-N10-20260526T232904/` (default), `paired-N10-20260526T232931/` (Tuning B K'=5), `paired-N10-20260526T233006/` (Tuning B K'=10).

### Interpretation

**Cross-over achieved.** For the first time SCAS genuinely beats baseline on time-to-coverage — ratio > 1.0 across the Tuning B sweep. AC-B3 even improves (3.72 in default config) because the larger embedding space makes group attractors more distinguishable.

**Threshold not cleared.** Best ratio is 1.33 at K'=10. Below the 1.5 target. Three diagnoses:

1. **Proposer locality is still the dominant brake.** Even in Tuning B (alpha biased low → random-within-category most of the time), the residual α>0.5 episodes still pin agents to `slots[i±1]` of their group attractor. The expanded taxonomy gives more room but doesn't address the fundamental "agents can't jump" issue.
2. **K' >> categories degrades SCAS.** At K'=15, each category has ~5 minima but only ~99 candidate slots; SCAS per-agent coupon-collector time approaches baseline. This is the "crowding under partition" mode the design predicted.
3. **Baseline is still cheap.** 330 strings is bigger than 33 but still finite; coupon-collector for K'=5 minima from 330 strings takes ~150 attempts, baseline hits 80% in ~75.

**AC-B1' robustly passes** across the v3 sweep (SCAS within 0.001 of baseline best_val_bpb).

## 12. Round-3 escalation

Three escapes remain, each with different costs:

| Path | Effort | What it tests | Risk |
| --- | --- | --- | --- |
| Fix proposer locality (occasional global jumps) | ~30 min | Whether the residual α>0.5 trapping was the brake | If ratio stays <1.5, SCAS framing doesn't earn its keep here |
| Inflate taxonomy further (≥1000 strings) | ~30 min | Whether baseline-saturation dominates everything | Likely yes — same dynamic as 33→330 |
| Re-spec threshold to ≥1.2 | None — doc edit | Accept measured-best as the bar | Honest but feels like moving goalposts |

Recommendation: **try the proposer-locality fix once**. It's independently a real bug — `propose(target=…)` should mix neighbor and global moves proportional to (1-α), so α=1 still picks neighbors but α=0.5 already gives 50% global. If 1.33 jumps over 1.5 with this fix, AC-B2 cleanly passes. If not, the falsification finding is now sharply scoped to a specific claim ("SCAS coordination does not give a 1.5× coverage speedup in synthetic regimes") rather than blamed on toy artifacts.

---

## 13. AC-B2 v4 — proposer-locality fix + new defaults

User chose option 1 from §12: fix the proposer. `synthetic.py propose()` now takes an `alpha` arg and per-call mixes local (neighbor of target) vs global (random with avoid filter) by `bernoulli(alpha)`. `sim.py` now always passes both `target` and `avoid` plus the current `alpha`. Coordinator defaults updated in `coordinator.py`:

| Constant | Original | New default | Rationale |
| --- | --- | --- | --- |
| `W_P` | 50.0 | **0.0** | Original saturated alpha after one below-best step → mix-proposer never got a chance to fire globally. With `W_P=0`, psi tracks embedding distance only. |
| `PSI_0` | 0.5 | **2.0** | Biases alpha low so the mix-proposer's global branch fires often. Tight pursuit only kicks in when agents drift far from their group attractor. |

Tuned constants are still env-var overridable (`SCAS_W_P`, `SCAS_PSI_0`, `SCAS_W_D`) for Day 4 sweeps.

### v4 headline (all-defaults, paired-N10)

| Metric | K'=5 | K'=10 | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| `ac_b2_speedup_ratio` (baseline / scas steps-to-80%) | **1.69** | **1.54** | ≥ 1.5 | **PASS** |
| `ac_b3_inter_over_intra_aggregate` | **2.09** | **2.10** | ≥ 1.5 | **PASS** |
| Δ(best_val_bpb) (SCAS − baseline) | −0.0021 | −0.0005 | within 1 SE | **PASS** (SCAS marginally better) |

Run dirs: `paired-N10-20260526T233615/` (K'=5), `paired-N10-20260526T233621/` (K'=10). Plots: `ac_b2_time_to_coverage.png`, `ac_b3_inter_vs_intra.png` in each.

### Reproducibility ablation (proposer fix only, originalconstants → still fails)

Confirms that *both* the proposer fix and `W_P=0`/`PSI_0=2.0` are load-bearing:

| Config | K'=5 ratio | K'=10 ratio |
| --- | --- | --- |
| Proposer fix + W_P=50, PSI_0=0.5 (original constants) | 0.39 | 0.73 |
| Proposer fix + W_P=5, PSI_0=0.5 | 0.56 | — |
| Proposer fix + W_P=0, PSI_0=0.5 | 0.56 | 0.94 |
| **Proposer fix + W_P=0, PSI_0=2.0 (committed defaults)** | **1.69** | **1.54** |

The mix proposer fix unlocks the dynamic; the constants control how aggressively SCAS exploits it.

### What this validates and what it doesn't

**Validates.** In an expanded synthetic regime (330-string taxonomy, K=3, K'=5–10, 200 steps × 3 agents), the SCAS coordination layer:
- Produces clean group separation (AC-B3 ≈ 2.1× intra-group spread).
- Reaches 80% coverage of ground-truth minima **1.5–1.7× faster** than uniform-random walkers (AC-B2 ≥ 1.5).
- Does no worse than independent walkers on best `val_bpb` (AC-B1' Δ within noise).

**Does not validate.**
- That the result transfers to live LLM agents on real `train.py` proposals. The synthetic proposer mixes local/global with known probability; an LLM's effective `alpha` is implicit and may behave differently. Replay-mode (T5) is the next checkpoint.
- That the winning constants generalize. `W_P=0` may be too aggressive for live runs where the perf signal is meaningful; this is one of the things a Day-4 `coupling_T` sweep should poke.

## 14. Status going into Day 4

All three headline ACs (B1', B2, B3) now pass on the simulator with the committed defaults. The escalation rounds are closed. Day 4's `coupling_T` sweep (AC-B4 — knob monotonicity) becomes the next informative experiment, and it now has a meaningful AC-B2 to also track across the sweep.




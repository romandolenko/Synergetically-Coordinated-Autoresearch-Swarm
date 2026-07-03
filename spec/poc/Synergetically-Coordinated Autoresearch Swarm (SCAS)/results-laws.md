# SCAS — SCT Dynamical Law vs Engineered Heuristic (head-to-head)

> **⚠ Update (2026-06-09): the SCT numbers in §2–§5 are superseded by §9.** A
> code review found the `SCTCoordinator`'s forward-Euler integration was
> **degenerate at `coupling_T = 1.0`** (snaps to the drive — zero memory, i.e.
> the §2 table's "dynamical law" arm actually ran the steady-state gate) and
> **unstable for `coupling_T < 1`** (the T=0.1/0.3 points of the §4 sweep are
> numerical artifacts). The integrator is fixed (exact exponential update) and
> everything is rerun in **§9**, including a fresh-seed confirmation of the
> heuristic headline. Qualitative conclusions survive — except §4's "monotone
> then saturates" knob reading, which was entirely an artifact (the fixed law
> shows **no** T-response). §2–§8 are kept as-is for the record.

> Built in response to the review concern that (a) the committed "synergetic
> control law" is only the *steady-state* algebraic gate — the dynamics
> `T·ψ̇ + ψ = 0` are never integrated (`design.md` §5) — and (b) the variant that
> makes the headline ACs pass does so with a hand-built heuristic, not the law.
> So we implemented **both** as first-class arms and compared them on an
> identical substrate, with bootstrap CIs. Companion docs: [`results.md`](results.md),
> [`results-review.md`](results-review.md), [`requirements.md`](requirements.md),
> [`design.md`](design.md).

## 1. What was implemented

- **`SCTCoordinator` (`coordinator.py`) — the genuine dynamical law.** Two
  first-order relaxations integrated explicitly (forward Euler, one `dt` per
  observation), both with the single timescale `coupling_T`:
  - **Attractor as a continuous order parameter:** `coupling_T·ċ_g = z*_g − c_g`
    (renormalised to the unit sphere), instead of snapping the consensus to the
    best card's embedding.
  - **Per-agent coupling order parameter with memory:** `coupling_T·ψ̇ + ψ = u`,
    `u = w_d·dist(z,c) + w_p·max(0, val−best)`. Small `T` ⇒ `ψ` tracks the
    instantaneous drive (≈ the steady-state law); large `T` ⇒ a sticky order
    parameter that enslaves agents gradually (Haken).
  - `α = sigmoid((ψ − psi_0)/alpha_temp)` gates a per-agent exploit (move toward
    the relaxing attractor) vs explore (global) decision. **No shared coverage
    memory, no exploit cap, no scripted anneal** — any diversity-with-convergence
    must *emerge* from the dynamics.
- **The SCT+novelty-search hybrid (`--sct-search`) — the third arm.** The same
  `SCTCoordinator` dynamics, but its explore move is a **farthest-point novelty
  search** over shared per-group coverage (instead of global-uniform), while
  exploit stays α-gated and **uncapped**, with **no scripted anneal**. This
  isolates the one question the first comparison left open: can the dynamical
  law's *scheduling of* a novelty search recover coverage while keeping cohesion
  emergent?
- **The engineered heuristic** already existed (`--embedding-repulsion --anneal`):
  farthest-point search in MiniLM space + shared per-group coverage memory +
  exploit-gating (≤1 agent/group/step) + a hard explore→converge anneal schedule.
- **`sim.py --compare-laws`** runs five arms on identical seeds — random baseline,
  partitioned-uncoupled fair control, SCT law, SCT+search hybrid, heuristic — all
  sharing the fixed partition + prefix templates + embedding candidate pool, so
  the *only* difference between the law arms is the coordination mechanism.
  Headline ratios carry **paired seed-bootstrap 95% CIs** (`--compare-laws` →
  `comparison.json`), addressing the review's "no uncertainty on the
  threshold-straddling ratios" concern.

SCT knobs were retuned for the embedding proposer (the committed `psi_0=2.0` was
fit to the *string* proposer and squashes `α`≈0.19 here): `psi_0=0.5`,
`alpha_temp=0.3` centre `α` near 0.5. **Committed defaults are untouched and
reproduce byte-identically** (no-flag `--paired-seeds 10` → 44.8 steps / marginal
1.39 / inter-over-intra 2.0948; validator 20/20).

## 2. Head-to-head (N=20 seeds, G=3, K'=5, M=3 agents/group, coupling_T=1.0)

Coverage in agent-steps (lower = faster). CIs are paired bootstrap 95%.

| Arm | steps→80% | AC-B2 marginal vs partition | AC-B3 inter/intra | AC-B3 lift vs partition | best_val Δ vs baseline |
| --- | --- | --- | --- | --- | --- |
| random baseline | 65.6 | — (control) | — | — | — |
| partition (fair control) | 65.0 | 1.00 (control) | 1.90 | 0.00 (control) | — |
| **SCT dynamical law** | 83.1 | **0.78 [0.42, 1.35]** ✗ | 3.01 [2.92, 3.09] ✓ | **+1.11 [1.02, 1.19]** ✓ | −0.0025 |
| **SCT + novelty search (hybrid)** | 53.0 | **1.23 [0.71, 2.14]** ✗ | 3.12 [3.02, 3.20] ✓ | **+1.22 [1.12, 1.30]** ✓ | −0.0024 |
| **Engineered heuristic** | 33.6 | **1.94 [1.22, 3.05]** ✗ | 4.23 [4.14, 4.32] ✓ | **+2.33 [2.24, 2.42]** ✓ | −0.0022 |

Direct head-to-heads: heuristic reaches 80% coverage **2.48× [1.63, 3.48] faster
than pure SCT**, and **1.58× [1.19, 2.10] faster than the hybrid**.

Pass bars (strict, CI-low): AC-B2 marginal CI-low ≥ 1.5; AC-B3 lift CI-low ≥ 0.5;
AC-B3 absolute CI-low ≥ 1.5. *Note: by the strict CI-low bar, **no arm robustly
clears the 1.5× coverage marginal** at N=20 — even the heuristic's 1.94× point has
CI-low 1.22.*

## 3. Reading the result

**The SCT law produces genuine *emergent* cohesion — and nothing else.**
AC-B3 lift +1.11 (CI excludes 0.5 comfortably) is the dynamical content the
steady-state `Coordinator` could not claim: the relaxing order parameter pulls
group-mates together over time without any scripted convergence. This is the
honest "the law does something real" finding the review asked for. But:

**The SCT law *hurts* coverage (marginal 0.78 < 1).** This is structural, not a
tuning miss. The law's explore move is global-uniform (identical to the partition
control), and its exploit move re-visits the neighbourhood of the attractor —
already-covered ground. So by construction the law can at best *equal* the
uncoupled partition on coverage, and in practice the exploit fraction makes it
slower. A coordination law that only redistributes exploit/explore cannot beat a
search that actively seeks novelty.

**The heuristic's coverage win is a *search* property, not a coordination
property.** Its 1.94× comes from farthest-point novelty search + shared memory —
and the earlier decisive control (`results-review.md` §6c, `--independent-repulsion`:
0.86×) already showed that same search *without* coordination is slower than
uniform. So coverage speed = (novelty search) × (shared-memory coordination);
cohesion = what either law adds on top. The heuristic's larger AC-B3 lift (+2.33)
is partly the **scripted anneal** (every group-mate is forced onto the attractor
after step 0.5·N), so it is not a like-for-like "emergent" comparison with SCT's
+1.11.

**The hybrid confirms it: give the SCT law a novelty search and coverage recovers,
but only partway.** Swapping the law's global-uniform explore for a shared
farthest-point search (no anneal) moves coverage from 83→53 agent-steps — marginal
0.78→**1.23**, so it now *beats* the partition control on the point estimate (and
keeps cohesion fully emergent at +1.22, no scripted collapse). But it is still
**1.58× [1.19, 2.10] slower than the heuristic**, and its marginal CI-low (0.71)
does not clear 1.5. The gap is the **exploit allocation**: the heuristic hard-caps
exploit at ≤1 agent/group/step (so ~⅔ of a 3-agent group always explores), whereas
the law's homeostatic α≈0.5 lets ~half the group exploit every step — more
redundant re-visits, slower coverage. The law's *emergent* scheduling of exploit is
genuinely worse for coverage than the heuristic's *hand-tuned* gate, because the
dynamics pull α toward consensus, not toward maximal spread. So the dynamical law
contributes emergent cohesion and a usable (but sub-heuristic, sub-1.5×) coverage
schedule; it does not replicate the engineered gate.

**Uncertainty matters.** Even the heuristic's headline 1.94× has a bootstrap
CI-low of **1.22** — at N=20 it does *not* robustly clear the 1.5 bar, though the
point estimate reproduces `results-review.md` §6c exactly. Read with CIs, "AC-B2
passes via the variant" is a point-estimate pass with a wide interval, not a
robust one.

**Objective value is a wash either way.** Both best_val deltas are ≈ −0.002 —
within ~0.4σ of the evaluator noise floor (`noise_std=0.005`). With `w_p=0` the
coordinator ignores `val_bpb`, so neither law finds *better* optima; they only
reshape *where* the swarm spreads.

## 4. AC-B4 for the real law — the dynamical knob saturates

Sweeping `coupling_T ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` under the SCT law (N=10):

| coupling_T | 0.1 | 0.3 | 1.0 | 3.0 | 10.0 |
| --- | --- | --- | --- | --- | --- |
| inter/intra | 1.37 | 1.50 | 2.97 | 2.94 | 2.92 |

Kendall τ_agg = **0.40**, τ_flat = 0.57 — **fails** the |τ| ≥ 0.7 bar. Separation
rises monotonically while the order parameter is still loose (T ≤ 1) then
**saturates** once the consensus is sticky enough. So the dynamical law's knob is
monotone *within its responsive regime* but the strict rank-correlation bar fails
because of the flat tail — notably **worse than the steady-state law's clean
τ = +1.000**, where `coupling_T` is a direct sigmoid temperature. The single-knob
promise holds for the algebraic gate; for the dynamical law it needs re-scoping to
"monotone for T ≲ 1, saturating above."

## 5. Net comparison

| Property | SCT dynamical law | SCT + novelty search (hybrid) | Engineered heuristic |
| --- | --- | --- | --- |
| Coverage marginal (vs partition) | **0.78× — hurts** | **1.23×** (CI-low 0.71) | 1.94× (CI-low 1.22) |
| Coverage vs heuristic | 2.48× slower | 1.58× slower | — |
| Emergent cohesion (AC-B3 lift) | +1.11, fully emergent | **+1.22, fully emergent** | +2.33, partly scripted |
| Mechanism | integrated ψ + relaxing order parameter | + shared farthest-point explore, no anneal | farthest-point + shared memory + gating + anneal |
| Knob (AC-B4) | monotone then saturates (τ=0.40) | (inherits law dynamics) | n/a (fixed search) |
| Objective value | parity (w_p=0) | parity | parity |

**Conclusion.** The three arms separate the contributions cleanly:

1. **Coverage speed is a *search* property.** Pure SCT (global explore) can't beat
   the partition; adding a novelty search (hybrid) is what lifts the marginal above
   1.0. No coordination law manufactures coverage without a novelty-seeking search.
2. **Emergent cohesion is what the SCT law contributes.** Both law arms produce a
   real, CI-solid AC-B3 lift (+1.11/+1.22) with *no* scripted convergence — the one
   thing the engineered heuristic only gets by hard-coding the anneal.
3. **The engineered gate still wins coverage.** The heuristic's ≤1-exploit/group
   cap allocates more agents to exploration than the law's homeostatic α≈0.5, so it
   covers 1.58× faster than the hybrid. The dynamical law's *emergent* exploit
   schedule is usable but strictly worse than a hand-tuned gate for coverage.
4. **With CIs, nobody robustly clears 1.5×.** Even the heuristic's 1.94× has CI-low
   1.22 at N=20. The coverage claim that motivated the exercise is a wide-interval
   point pass, not a robust one.

So SCT is worth keeping for **emergent diversity-with-convergence** (cohesion you
don't have to script), but the **coverage win belongs to the search heuristic**, and
the best of both — the hybrid — is still beaten on coverage by the hand-tuned gate.

## 6. The hybrid's coverage↔cohesion tradeoff is irreducible (psi_0 sweep)

Follow-up to §5: does *some* `psi_0` let the unscripted hybrid match the heuristic's
coverage while keeping an emergent AC-B3 lift? `psi_0` sets the α centre — raising it
lowers α → fewer exploits → more explorers → faster coverage, weaker cohesion. Sweep
at N=20 (M=3, K'=5, alpha_temp=0.3; partition control constant at 65.0 steps /
inter-intra 1.90; heuristic reference 33.6 steps / marginal 1.94 / lift +2.33):

| psi_0 | hybrid steps→80% | AC-B2 marginal | AC-B3 inter/intra | AC-B3 lift | best_val Δ |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 86.2 | 0.75 | 4.49 | **+2.59** | −0.0022 |
| 0.25 | 56.1 | 1.16 | 3.63 | +1.73 | −0.0030 |
| 0.5 | 53.0 | 1.23 | 3.10 | +1.20 | −0.0024 |
| 1.0 | 41.6 | **1.56** | 2.36 | +0.46 | −0.0023 |
| 1.5 | 40.6 | **1.60** | 2.05 | +0.15 | −0.0005 |
| 2.0 | 42.6 | 1.52 | 1.94 | +0.05 | +0.0024 |

**Finding — a clean Pareto tradeoff, and no joint pass.** Coverage marginal rises
monotonically with `psi_0` (0.75 → 1.60, peaking near `psi_0=1.5`) while the AC-B3
lift falls monotonically (+2.59 → +0.05). The two ACs move in strict opposition
because α is a *single homeostatic knob*: every agent-step it spends pulling toward
consensus (cohesion) is one it doesn't spend seeking novelty (coverage). **There is
no `psi_0` where marginal ≥ 1.5 and lift ≥ 0.5 hold together** — the closest,
`psi_0=1.0`, gives marginal 1.56 but lift 0.46, just under the bar. The unscripted
law lives on a frontier; you can buy coverage *or* cohesion with `psi_0`, not both.

**Why the heuristic escapes the tradeoff.** Its scripted anneal *time-separates* the
objectives — explore-only early (banks coverage by ~step 11, far before the
converge phase at step 100), then hard-converge late (banks cohesion). That breaks
the single-knob coupling the law is bound by, which is exactly why it posts marginal
1.94 *and* lift +2.33 jointly. The law cannot replicate this without a comparable
time-varying schedule, because its α is driven by instantaneous distance, not by
run progress.

**Honest ceiling.** The unscripted SCT law's best balanced operating point is around
`psi_0 ≈ 0.25–0.5`: coverage clears the partition (marginal 1.16–1.23) and cohesion
is solidly emergent (lift +1.2 to +1.7) — a real, tunable diversity-with-convergence
that needs no scripting. But it is **Pareto-dominated by the heuristic**, which gets
strictly more of both by scripting. So: *the SCT law delivers genuine, knob-tunable
emergent cohesion-with-coverage, but a hand-tuned time-separated schedule strictly
beats it; the law's value is in not having to script, not in winning the metrics.*

(best_val Δ stays within ~0.5σ of the noise floor across the whole sweep, drifting
from −0.003 at high-exploit to +0.002 at high-explore — i.e. over-exploring slightly
hurts best-found, but all within noise. The coordinator remains performance-blind.)

Sweep data: `sim_runs/hybrid_psi0_sweep.json`. (Fresh-seed + post-fix rerun of
this sweep: §10 — the frontier shape and the lifts replicate exactly; the
coverage marginals do not, and the "balanced operating point" claim is revised
down there.)

## 7. Turning the objective on (`w_p` sweep) — performance-awareness doesn't help the objective

Every arm above is performance-blind (`w_p=0`); "coverage" and "cohesion" are purely
geometric. `w_p>0` makes ψ rise (→ α → exploit) for agents lagging the group best,
and the attractor already relaxes toward the *best-val* embedding — so this tests
whether steering toward **good** attractors changes the ranking or improves the
actual objective. Hybrid, N=20, `psi_0=0.5`, `alpha_temp=0.3`:

| w_p | hybrid steps→80% | AC-B2 marginal | AC-B3 inter/intra | AC-B3 lift | best_val Δ vs baseline |
| --- | --- | --- | --- | --- | --- |
| 0 | 53.0 | 1.23 | 3.10 | +1.20 | −0.0024 |
| 1 | 48.2 | 1.35 | 3.12 | +1.22 | −0.0025 |
| 5 | 50.8 | 1.28 | 3.36 | +1.46 | −0.0026 |
| 20 | 52.8 | 1.23 | 3.99 | +2.09 | −0.0033 |
| 50 | 66.4 | **0.98** | 5.42 | +3.52 | −0.0034 |

**Finding 1 — the objective barely moves.** best_val Δ drifts from −0.0024 to
−0.0034 across the whole sweep — a ~0.001 change, well inside the 0.005 noise floor
(≈0.2σ). **Performance-aware steering does not make the law find better optima.** The
reason is structural: the attractor already tracks the best-val embedding, so the
swarm already concentrates near good regions; `w_p` only makes *laggards converge
faster*, and the best-found is gated by exploration coverage, not by how hard
laggards are pulled in. The coordinator is a geometry-shaping mechanism, not a
research-efficiency one — turning performance on confirms it rather than fixing it.

**Finding 2 — `w_p` is a redundant cohesion knob with the same coverage cost.** As
`w_p` rises, lift climbs (+1.20 → +3.52) and coverage erodes, exactly like lowering
`psi_0`. At `w_p=50` coverage collapses to marginal **0.98** — the perf-gap
saturation trap already documented for the steady-state law (`results-review.md` §4,
where `w_p=50` broke coverage to 0.95×). So `w_p` does **not** break the §6
coverage↔cohesion tradeoff; it slides along the same frontier.

**Finding 3 — a *small* `w_p` is a mild Pareto nudge.** At `w_p=1–5`, both metrics
improve slightly over `w_p=0` (marginal 1.23→1.35, lift +1.20→+1.46): a touch of
perf-steering redirects underperforming agents (far from attractor *and* below best)
toward good regions to re-explore from, instead of re-sampling randomly. The effect
is small (~0.1 marginal, near run-to-run noise) and still **never reaches a joint
pass** — peak marginal 1.35 < 1.5. A small `w_p` is a reasonable default; it does not
change the verdict.

**Net.** Performance-awareness neither improves the objective (best_val flat within
noise) nor escapes the coverage↔cohesion tradeoff (it is a second knob along the same
frontier, with the `w_p=50` saturation trap at the end). The honest reading of all
three sweeps together: **this coordinator reshapes *where* the swarm searches in
embedding space; it does not change *how good* the best solution found is.** Any
research-efficiency claim must come from a setup where coverage breadth actually
translates to better optima — which this closed synthetic landscape, with minima
sampled from the agents' own output distribution, does not stress.

Sweep data: `sim_runs/hybrid_wp_sweep.json`.

## 8. Reproducibility

```powershell
# Head-to-head with bootstrap CIs (writes comparison.json)
uv run python -m scas.sim --compare-laws --paired-seeds 20 --K 3 --K-prime 5 `
    --agents-per-group 3 --psi-0 0.5 --alpha-temp 0.3

# SCT law in isolation (single run)
uv run python -m scas.sim --sct --agents-per-group 3 --psi-0 0.5 --alpha-temp 0.3

# SCT + novelty-search hybrid in isolation (implies --sct)
uv run python -m scas.sim --sct-search --agents-per-group 3 --psi-0 0.5 --alpha-temp 0.3

# AC-B4 knob sweep under the real SCT law
uv run python -m scas.sim --sweep-T "0.1,0.3,1.0,3.0,10.0" --sct --paired-seeds 10 `
    --agents-per-group 3 --psi-0 0.5 --alpha-temp 0.3

# Hybrid coverage<->cohesion tradeoff: sweep psi_0 (raises explore as psi_0 rises)
foreach ($p in 0.0,0.25,0.5,1.0,1.5,2.0) {
    uv run python -m scas.sim --sct-search --paired-seeds 20 --K 3 --K-prime 5 `
        --agents-per-group 3 --alpha-temp 0.3 --psi-0 $p
}

# Performance-awareness: sweep w_p on the hybrid (best_val stays flat within noise)
foreach ($w in 0.0,1.0,5.0,20.0,50.0) {
    uv run python -m scas.sim --sct-search --paired-seeds 20 --K 3 --K-prime 5 `
        --agents-per-group 3 --psi-0 0.5 --alpha-temp 0.3 --w-p $w
}

# Committed defaults unchanged (regression check → 44.8 / 1.39 / 2.0948)
uv run python -m scas.sim --paired-seeds 10
```

Run dirs: `compare-laws-N20-20260528T235034` (4-way with hybrid),
`compare-laws-N20-20260528T233700` (initial 2-law), `sweep-T5-N10-20260528T233752`;
`sim_runs/hybrid_psi0_sweep.json` (psi_0 tradeoff).

## 9. Integrator fix and post-fix rerun (2026-06-09) — supersedes §2–§5 SCT numbers

### 9a. The bug

`SCTCoordinator` integrated both relaxations with forward Euler,
`x += (dt/T)(u − x)` with `dt = 1`. For the linear relaxation `T·ẋ = u − x`
this update has error multiplier `(1 − dt/T)` per step, so:

- **`T = 1` (the §2 operating point): multiplier 0 — `ψ = u` and `c = z*`
  exactly.** Zero memory, zero relaxation: the "genuine dynamical law" arm in §2
  was actually the steady-state algebraic gate with a retuned sigmoid
  (`psi_0=0.5`, `alpha_temp=0.3`). Its cohesion lift was real but the credit to
  "the relaxing order parameter pulls group-mates together over time" was wrong
  — nothing was relaxing.
- **`T < 1` (the §4 sweep's 0.1 and 0.3 points): |multiplier| > 1 — unstable.**
  ψ oscillates with growing amplitude (α slams between 0 and 1); the attractor
  is rescued only by sphere renormalisation. The low inter/intra at T=0.1/0.3 in
  §4 was the oscillation destroying cohesion, not the law responding to the knob.
- **`T > 1`: valid** — the only genuinely dynamical regime the pre-fix data had.

**Fix:** the exact exponential update, the closed-form solution of the
relaxation over one step — stable for every `T > 0`, tracking `u` as `T → 0`
and freezing as `T → ∞`:

```
decay = exp(-dt / coupling_T)
ψ ← u  + (ψ − u)  · decay
c ← z* + (c − z*) · decay      (then renormalised to the unit sphere)
```

Unit tests now pin this down (`scas/law_tests.py`): ψ approaches a constant
drive monotonically without overshoot for `T ∈ {0.1, 0.5, 1, 3, 10}`, and the
attractor relaxes monotonically on the sphere (snaps for small T, gradual for
large T). The steady-state `Coordinator` is untouched; committed defaults still
reproduce byte-identically (44.8 / 1.39 / 2.0948; validator + law tests 35/35).

### 9b. Post-fix head-to-head (same protocol as §2, plus the independent-search control as a sixth arm)

N=20 seeds, G=3, K'=5, M=3/group, coupling_T=1.0, psi_0=0.5, alpha_temp=0.3
(now the auto-applied SCT defaults). Run dir `compare-laws-N20-20260609T183942`.

| Arm | steps→80% | AC-B2 marginal vs partition | AC-B3 inter/intra | AC-B3 lift | best_val Δ |
| --- | --- | --- | --- | --- | --- |
| random baseline | 65.5 | — (control) | — | — | — |
| partition (fair control) | 65.0 | 1.00 (control) | 1.90 | 0.00 (control) | — |
| **independent search (new arm)** | 63.1 | 1.03 [0.74, 1.40] | 1.90 [1.88, 1.93] | 0.00 [−0.02, 0.03] | +0.0010 |
| **SCT dynamical law** | 100.8 | **0.64 [0.38, 1.06]** ✗ | 2.94 [2.87, 3.00] ✓ | **+1.04 [0.97, 1.11]** ✓ | −0.0033 |
| **SCT + novelty search (hybrid)** | 47.8 | **1.36 [0.87, 2.33]** ✗ | 3.08 [3.03, 3.13] ✓ | **+1.18 [1.14, 1.23]** ✓ | −0.0023 |
| **Engineered heuristic** | 33.5 | **1.94 [1.22, 3.05]** ✗ | 4.23 [4.14, 4.32] ✓ | **+2.33 [2.24, 2.42]** ✓ | −0.0022 |

Direct head-to-heads: heuristic 3.00× [1.82, 4.68] faster than pure SCT
(pre-fix: 2.48×), 1.42× [1.02, 1.90] faster than the hybrid (pre-fix: 1.58×).

**What changed with real dynamics at T=1:**

- **Pure SCT coverage got *worse*** (83 → 101 steps, marginal 0.78 → **0.64**).
  With genuine ψ-memory, exploit episodes persist (ψ decays over ~T steps
  instead of resetting), so the law re-visits attractor neighbourhoods even
  more. The §3 structural argument strengthens: the law cannot beat a
  novelty-seeking search at coverage.
- **The hybrid improved** (53 → 47.8, marginal 1.23 → **1.36**) — memory makes
  the α-schedule less jittery, so explore phases are spent on the farthest-point
  search more coherently. Still does not clear 1.5 (CI-low 0.87).
- **Emergent cohesion survives the fix** (+1.04 / +1.18 vs pre-fix +1.11 /
  +1.22) — and is now attributable to actual dynamics, not the degenerate snap.
  This is the corrected version of §3's headline finding, and it stands.
- **The "search alone hurts" reading softens.** §6c of `results-review.md`
  (N=10) reported the independent farthest-point control at 0.86× (slower than
  uniform). On N=20 same-seed footing it is **1.03× [0.74, 1.40] — a wash**, not
  a harm. The decomposition still attributes the heuristic's speedup to
  coordination (1.94 vs 1.03), but the search-alone penalty claim should be
  retired.
- The heuristic arm is unchanged by construction (33.5 ≈ 33.6; it never used
  the SCT integrator).

### 9c. Post-fix AC-B4 — the dynamical law has *no* coupling_T response

Same sweep as §4 (N=10, M=3, psi_0=0.5, alpha_temp=0.3), fixed integrator.
Run dir `sweep-T5-N10-20260609T183943`.

| coupling_T | 0.1 | 0.3 | 1.0 | 3.0 | 10.0 |
| --- | --- | --- | --- | --- | --- |
| inter/intra (pre-fix, artifact) | 1.37 | 1.50 | 2.97 | 2.94 | 2.92 |
| **inter/intra (fixed)** | **2.97** | **3.09** | **2.87** | **2.82** | **2.94** |

Kendall τ_agg = **−0.40**, τ_flat = −0.15 — **AC-B4 fails for the dynamical
law, and not marginally**. With stable integration, separation is flat (~2.9)
across two decades of T; the pre-fix "monotone within its responsive regime,
saturating above" reading was entirely the artifact (the rising left edge was
the unstable oscillation destroying cohesion at T<1). Coverage is likewise flat
(~106–115 steps at every T). **The single-knob promise holds only for the
steady-state law**, where `coupling_T` is a sigmoid temperature (τ = +1.000);
the dynamical law's relaxation timescale does not translate into a usable
diversity dial at this operating point — α's *equilibrium* (set by `psi_0` /
`alpha_temp`) determines behaviour, and T only sets how fast ψ gets there,
which a 200-step run barely notices.

### 9d. Fresh-seed confirmation — the heuristic's AC-B2 pass does not replicate out-of-sample

All tuning across `results-review.md` §6b–§6e and this doc used seeds 0..N−1.
The headline heuristic config (M=3, K'=5, anneal, N=20) was rerun once on
never-touched seeds **100–119** (`--seed-offset 100`, run dir
`paired-N20-20260609T183944`):

| Metric | Tuning seeds 0–19 | **Fresh seeds 100–119** | Bar |
| --- | --- | --- | --- |
| AC-B2 control-law marginal | 1.94 | **1.42 ✗** | ≥ 1.5 |
| AC-B3 lift vs fair control | +2.33 | **+2.40 ✓** | ≥ 0.5 |
| AC-B1' best_val Δ | −0.0022 | −0.0038 | within noise |

**Finding.** The AC-B3 lift replicates (expected — the anneal scripts it). The
**AC-B2 marginal does not** (1.42 < 1.5), landing inside the bootstrap CI the
N=20 tuning run already warned about ([1.22, 3.05]). The "robust 1.94×" of
§6c/§6d should be read as ~1.4–1.9 with seed-level variance straddling the bar.
The variant's coverage advantage over the fair control is real (CI excludes ≤1
on neither run decisively, but both points are well above 1); the *1.5×
threshold claim* specifically is not confirmed out-of-sample.

### 9e. Corrected net conclusions (replaces §5's table where they differ)

1. **Emergent cohesion is the SCT law's genuine contribution and survives the
   fix** (+1.04 pure / +1.18 hybrid, CI-solid, no scripting) — now demonstrated
   with actual integrated dynamics rather than a degenerate snap.
2. **Coverage remains a search property, more starkly than before**: pure law
   0.64× (actively hurts), hybrid 1.36×, heuristic 1.94× — and the heuristic's
   edge over the hybrid narrowed (1.42×) but persists.
3. **The dynamical law has no working coupling_T knob** (AC-B4 fail, flat
   response). Tunable behaviour lives in `psi_0`/`alpha_temp` (the §6 Pareto
   frontier), not in the relaxation timescale.
4. **No arm clears the 1.5× coverage bar robustly**: the heuristic's 1.94× has
   CI-low 1.22 *and* drops to 1.42 on fresh seeds. The honest AC-B2 status for
   every coordination mechanism tested is "real advantage over the fair
   control, threshold not robustly met."
5. The heuristic's coverage numbers are additionally **menu-dependent** (it
   pre-embeds the entire finite candidate pool; see `results-review.md` §6c
   scope limit) and transfer least readily to a live phase.

### 9f. Reproducibility (post-fix)

```powershell
# All law tests incl. integrator-stability regressions (35 tests)
uv run pytest scas/law_tests.py scas/validator_tests.py -q

# Post-fix head-to-head (psi_0=0.5 / alpha_temp=0.3 are now auto-defaults for SCT arms)
uv run python -m scas.sim --compare-laws --paired-seeds 20 --K 3 --K-prime 5 --agents-per-group 3

# Post-fix AC-B4 sweep under the SCT law
uv run python -m scas.sim --sweep-T "0.1,0.3,1.0,3.0,10.0" --sct --paired-seeds 10 --agents-per-group 3

# Fresh-seed confirmation of the heuristic headline (seeds 100-119)
uv run python -m scas.sim --paired-seeds 20 --seed-offset 100 --agents-per-group 3 --anneal

# Committed defaults unchanged (44.8 / 1.39 / 2.0948)
uv run python -m scas.sim --paired-seeds 10
```

Run dirs: `compare-laws-N20-20260609T183942`, `sweep-T5-N10-20260609T183943`,
`paired-N20-20260609T183944`, `paired-N10-20260609T183705` (regression check).

## 10. Fresh-seed ψ₀ frontier (2026-07-02) — the tradeoff replicates exactly; the hybrid's coverage does not

The §6 sweep had two liabilities: it predated the §9 integrator fix (its
"hybrid" was effectively the steady-state gate) and, like everything before
§9d, it ran on the tuning seeds. The full sweep was rerun with the fixed
integrator on never-touched seeds 100–119 (hybrid `--sct-search`, N=20, M=3,
K'=5, alpha_temp=0.3; partition control at 56.5 steps / inter-intra 1.89 on
this seed block; bootstrap CIs, 10k resamples):

| psi_0 | hybrid steps→80% | AC-B2 marginal (§6 in-sample) | **AC-B2 marginal (fresh)** | AC-B3 lift (§6) | **AC-B3 lift (fresh)** |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 110.8 | 0.75 | **0.51 [0.38, 0.73]** | +2.59 | **+2.74 [2.51, 2.95]** |
| 0.25 | 79.6 | 1.16 | **0.71 [0.53, 0.98]** | +1.73 | **+1.86 [1.78, 1.93]** |
| 0.5 | 58.9 | 1.23 | **0.96 [0.67, 1.38]** | +1.20 | **+1.22 [1.14, 1.30]** |
| 1.0 | 46.5 | 1.56 | **1.21 [0.84, 1.78]** | +0.46 | **+0.47 [0.39, 0.55]** |
| 1.5 | 39.3 | 1.60 | **1.44 [0.96, 2.09]** | +0.15 | **+0.21 [0.13, 0.29]** |
| 2.0 | 38.3 | 1.52 | **1.47 [0.98, 2.15]** | +0.05 | **+0.12 [0.05, 0.19]** |

**Finding 1 — the frontier itself replicates, remarkably cleanly.** The marginal
is strictly monotone increasing in `psi_0` and the lift strictly monotone
decreasing, exactly as in-sample; the cohesion lifts land within ~0.1 of the §6
values at every point, with tight CIs. The §6 structural claim — a single
homeostatic knob cannot buy coverage and cohesion at once — is confirmed
out-of-sample and post-fix.

**Finding 2 — the coverage half of the frontier shifts down wholesale.** Every
fresh marginal is 0.1–0.4 below its in-sample twin. **No `psi_0` clears 1.5**
(the closest, `psi_0=2.0`, gives 1.47 [0.98, 2.15] with zero cohesion), so the
"no joint pass" conclusion strengthens: on fresh seeds there is not even a solo
AC-B2 pass anywhere on the frontier. This is the same direction and roughly the
same magnitude as the heuristic's §9d drop (1.94 → 1.42) — seed-block variance
in coverage marginals, again larger than within-block CIs suggest (the fresh
partition control is itself faster, 56.5 vs 65.0 steps).

**Finding 3 — the §6 "honest ceiling" is revised down.** At the previously
recommended balanced operating point (`psi_0 ≈ 0.25–0.5`) the fresh marginals
are **0.71–0.96 — at or below partition parity**. The balanced hybrid no longer
demonstrates any coverage advantage over the fair control; what survives at that
operating point is the emergent cohesion (+1.2 to +1.9, unscripted, tightly
replicated). The corrected claim for the unscripted law: *it buys tunable,
genuinely emergent cohesion at zero-to-negative coverage cost relative to a
partitioned control — the coverage side of the §6 sales pitch is withdrawn.*

Consistency note: this fresh-seed table is compatible with §9b's in-sample
post-fix hybrid (marginal 1.36 [0.87, 2.33] at `psi_0=0.5` there vs 0.96
[0.67, 1.38] here) — both CIs straddle 1, and the point difference is the same
seed-block effect as above, not a contradiction.

Run dirs: `paired-N20-20260702T075527/-075615/-075709/-075805/-075905/-080006`
(psi_0 = 0 / 0.25 / 0.5 / 1.0 / 1.5 / 2.0); aggregate with CIs in
`sim_runs/fresh_seed_confirmation_20260702.json`. Repro:

```powershell
foreach ($p in 0.0,0.25,0.5,1.0,1.5,2.0) {
    uv run python -m scas.sim --sct-search --paired-seeds 20 --agents-per-group 3 `
        --alpha-temp 0.3 --psi-0 $p --seed-offset 100
}
```

## 11. Emergent grouping (2026-07-02) — AC-B3 passes as an SCT claim, with no hand partition

Every AC-B3 number so far was earned on a *scripted* partition: agents were
hand-assigned to category-bound groups at construction, so ~90% of the
separation was structural. This experiment removes the partition entirely and
asks whether the coupling alone forms and separates groups.

### 11a. Design

- **`EmergentSCTCoordinator`** (`scas/coordinator.py`): group slots start
  empty; the first G accepted observations seed them (attractor = that
  embedding); every later accepted observation re-assigns its agent to the
  nearest attractor (unit-norm embeddings, so nearest = argmax dot).
  Membership churns as agents and attractors move. All relaxation dynamics are
  inherited unchanged from `SCTCoordinator` (shared `_integrate`).
- **Pooled menu** (`--emergent-groups`, requires `--sct`/`--sct-search`): the
  proposer draws from the union of all categories' candidates — anything
  category-scoped would smuggle the partition back in. Exploit = nearest pooled
  candidate to the agent's current group attractor.
- **The matched null**: the identical machinery with `--psi-0 1e9`, which pins
  α at 0. This control matters because nearest-attractor bookkeeping is
  k-means-like and "finds" separation even in uncoupled random walkers (it
  scores inter/intra ≈ 1.5 and silhouette ≈ 0.3 below). The coupling must
  show a *lift over that null*, not over zero.
- Two metrics: the standard inter/intra (final attractors, historical labels —
  identical machinery in both arms, so the lift is fair) and a label-robust
  **final-window silhouette** (last 25% of accepted observations under the
  emergent labels).
- Two disjoint seed blocks from the start (0–19 and 100–119), per the §10
  multi-block discipline. K=3 slots, K'=5, M=3, T=1.0, psi_0=0.5 (coupled
  arms), alpha_temp=0.3, 200 steps.

### 11b. Results (N=20 per block, bootstrap CIs)

| Arm | inter/intra | silhouette (final window) | **ioi lift vs null** | sil lift vs null |
| --- | --- | --- | --- | --- |
| *block 0–19* | | | | |
| null (α=0) | 1.51 [1.32, 1.69] | 0.313 | — | — |
| hand partition (same seeds, reference) | 1.90 | — | — | — |
| **pure SCT law** | 2.40 [2.14, 2.64] | 0.404 | **+0.89 [+0.73, +1.03]** ✓ | +0.091 [+0.035, +0.138] |
| **SCT + search (hybrid)** | 3.00 [2.91, 3.08] | 0.602 | **+1.49 [+1.30, +1.70]** ✓ | +0.289 [+0.224, +0.355] |
| *block 100–119* | | | | |
| null (α=0) | 1.58 [1.47, 1.70] | 0.344 | — | — |
| hand partition (same seeds, reference) | 1.89 | — | — | — |
| **pure SCT law** | 2.55 [2.38, 2.73] | 0.431 | **+0.97 [+0.84, +1.08]** ✓ | +0.087 [+0.046, +0.124] |
| **SCT + search (hybrid)** | 2.96 [2.88, 3.05] | 0.599 | **+1.38 [+1.25, +1.51]** ✓ | +0.254 [+0.206, +0.302] |

Supporting observables (final window / full run):

| Arm | modal-category purity | membership churn per accepted obs |
| --- | --- | --- |
| null | 0.87 / 0.91 | 0.59 |
| pure SCT law | 0.92 / 0.91 | 0.34 |
| hybrid | **1.000 / 1.000** | 0.39 |

### 11c. Findings

1. **The coupling alone forms and separates groups — AC-B3 passes as an SCT
   claim.** Both coupled arms clear both AC-B3 bars *by CI-low, in both seed
   blocks*: absolute inter/intra ≥ 1.5 (pure 2.40/2.55, hybrid ~3.0) and lift
   ≥ 0.5 over the fair (matched-null) control (pure +0.89/+0.97, hybrid
   +1.49/+1.38). This is the first CI-solid joint AC-B3 pass in the project
   that is not attributable to a scripted partition — and it replicates across
   blocks, consistent with the running pattern that cohesion effects replicate
   tightly.
2. **Emergent separation *exceeds* the hand partition.** On the same seeds the
   scripted partition control scores inter/intra 1.90/1.89; the emergent pure
   law reaches 2.40–2.55 and the hybrid ~3.0. Letting the dynamics choose the
   groups beats assigning them.
3. **What the groups converge to is the latent category structure.** The
   hybrid's emergent groups end with *perfect* modal-category purity (1.000 in
   both blocks; pure law 0.91–0.92; null 0.87–0.91), and the coupling roughly
   halves membership churn (0.34–0.39 vs 0.59). The honest phrasing: the
   coupling does not create structure ex nihilo — it *finds, locks onto, and
   amplifies* the cluster structure already latent in the prefix-anchored
   embedding space, which uncoupled bookkeeping only weakly tracks.
4. **The coverage cost is real and worse than the partitioned setting.**
   Against the same-seed baseline, the emergent coupled arms are *slower* to
   cover (pure 0.82/0.52, hybrid 0.73/0.74 speedup ratio; the null covers at
   1.31/1.02). With a pooled menu and no partition, exploit episodes revisit
   attractor neighbourhoods across the whole space — §9b's "the law hurts
   coverage" transfers unchanged. The emergent result is a cohesion win, not a
   coverage win.
5. Committed defaults are untouched (regression on this code: 44.8 / 1.39 /
   2.0948 exactly; 39 unit tests pass, 4 new for emergent membership).

The live-phase implication is direct: the SCT law arm carried into Phase 2 no
longer *requires* a hand-built category router to exhibit separation — the
attractor dynamics can grow the partition. Whether MiniLM finds comparable
latent structure in real, non-templated hypothesis cards is exactly the T5.2
embedding-quality question (silhouette 0.029 raw), so §11 raises that check's
priority rather than retiring it.

Run dirs: `paired-N20-20260702T090322/-090403/-092422` (block 0: pure /
hybrid / null), `paired-N20-20260702T092519/-092611/-092722` (block 100);
regression `paired-N10-20260702T090313`; aggregate with CIs in
`sim_runs/emergent_grouping_20260702.json`. Repro:

```powershell
# coupled arms + matched null, two disjoint seed blocks
foreach ($off in 0,100) {
    uv run python -m scas.sim --paired-seeds 20 --sct --emergent-groups --agents-per-group 3 --seed-offset $off
    uv run python -m scas.sim --paired-seeds 20 --sct-search --emergent-groups --agents-per-group 3 --seed-offset $off
    uv run python -m scas.sim --paired-seeds 20 --sct --emergent-groups --agents-per-group 3 --psi-0 1e9 --seed-offset $off
}
```

## 12. Coverage-causes-quality benchmark (2026-07-03) — the payoff exists, and only the engineered heuristic cashes it in

Every result so far shares one blind spot, called out in the presentation's
headline reading: best_val is flat within noise across all arms, so "coverage
speedup" had no efficiency payoff to point at. Coverage could be a proxy for
research quality or a mere activity metric — this landscape couldn't say.
This section builds the landscape that can, then re-runs the full six-arm
comparison on it.

### 12a. Design

**Jackpot depth distribution** (`--depth-dist jackpot`). The landscape
machinery is unchanged except that one rng-chosen minimum's depth is
overwritten with `--jackpot-depth 0.5` (background depths stay 0.04–0.08,
base val_bpb 1.0). Best_val is then dominated by a single question — did the
swarm ever visit the deep basin? — which makes quality causally downstream of
coverage by construction. The uniform mode is byte-identical committed
behavior (regression reproduces 44.8 / 1.39 / 2.0948 exactly; 43 unit tests
pass, 4 new for the jackpot landscape).

**Scarcity had to be engineered** (`--landscape-sigma`). At the committed
σ=0.5, pilots found jackpot discovery saturating at 80–100% for every arm even
with the budget cut to 15 steps: near-duplicate template strings (same base,
different lr/momentum) sit inside the σ/2 visit radius, and `evaluate()` pays
partial credit at a distance, so any broad sampling stumbles into the basin.
Narrowing every basin to σ=0.30 shrinks the visit radius to 0.15 and removes
distant partial credit; combined with the widest pool (G=6 categories,
622 strings) and a modest budget (18 agents × 40 steps), baseline discovery
lands at ~0.5 — the regime where arms can separate in either direction.

**Protocol.** Operating point G=6, M=3/group, K'=10, steps=40, σ=0.30,
jackpot depth 0.5. Six arms on identical seeds (random baseline, partition
control, independent search, pure SCT law, SCT+search hybrid, engineered
heuristic). This is the first experiment run under the multi-block rule §10
motivated: **three disjoint 20-seed blocks** (offsets 0/100/200), pooled
paired bootstrap (10k) over all 60 seeds.

### 12b. Results

**The benchmark works.** Within *every* arm, seeds that visited the jackpot
basin end with best_val ≈ 0.13–0.16 lower (better) than seeds that never did
(baseline link −0.140/−0.136/−0.138 across the three blocks — 26× the noise
std). Coverage causes quality on this landscape; the missing link is now
measurable.

Pooled over 60 seeds (per-block rates shown because discovery is a coverage
marginal, and coverage marginals carry block variance):

| arm | jackpot rate b0/b1/b2 | pooled rate [CI95] | best_val Δ vs baseline [CI95] | within-arm link |
| --- | --- | --- | --- | --- |
| baseline | 0.45 / 0.55 / 0.70 | 0.57 [0.43, 0.68] | — (control) | −0.142 |
| partition | 0.60 / 0.55 / 0.65 | 0.60 [0.47, 0.72] | −0.009 [−0.026, +0.006] | −0.145 |
| independent | 0.65 / 0.60 / 0.80 | 0.68 [0.57, 0.80] | −0.016 [−0.038, +0.004] | −0.162 |
| sct (pure law) | 0.65 / 0.60 / 0.55 | 0.60 [0.48, 0.73] | −0.017 [−0.041, +0.004] | −0.132 |
| sct_search | 0.65 / 0.60 / 0.65 | 0.63 [0.52, 0.75] | −0.017 [−0.037, +0.001] | −0.134 |
| **heuristic** | 0.70 / 0.70 / 0.60 | 0.67 [0.55, 0.78] | **−0.035 [−0.058, −0.014]** | −0.131 |

Coverage speed at this operating point (marginal vs partition, per block):
heuristic 2.58 [2.08, 3.14] / 2.33 [1.85, 2.92] / 1.69 [1.23, 2.38] — CI-low
clears 1.5 in two blocks of three, so under the multi-block rule this is
still *not* a robust ≥1.5× pass, consistent with §9d. The SCT arms remain the
slowest coverers (pure-law marginal 0.72–0.91). Cohesion replicates tightly
across blocks yet again (sct 2.85–2.94, heuristic 4.09–4.14).

### 12c. Findings

1. **Quality is now causally downstream of coverage — and the effect is
   enormous relative to noise.** The within-arm found-vs-not contrast
   (−0.13…−0.16 bpb) replicates in all six arms and all three blocks. The
   presentation's caveat "this landscape does not stress the coverage→quality
   link" is retired: this one does.
2. **Only the engineered heuristic converts its coverage advantage into a
   CI-solid quality gain** (−0.035 [−0.058, −0.014] vs baseline, roughly a
   quarter of the maximum harvestable link). The novelty-search machinery
   that wins coverage everywhere finally has an efficiency payoff to point
   at — with the standing caveat that it requires a pre-embedded finite
   candidate menu, which a live phase does not have.
3. **The SCT law's coverage cost does not become a quality cost.** The pure
   law is the slowest coverer here (marginal 0.72–0.91), yet its jackpot
   discovery matches baseline (0.60 vs 0.57) and its best_val delta is
   slightly *negative* (−0.017, CI crossing zero at +0.004 — any penalty is
   bounded at ≈one noise-std). Slow breadth, but the exploit phase
   concentrates evaluations around the deepest basin any group member finds,
   which is exactly the samples that matter on this landscape. Quality
   parity, not a quality win: the law remains an organization lever, not an
   efficiency lever.
4. **The block-variance pattern extends to jackpot discovery.** Baseline
   discovery moved 0.45 → 0.70 across blocks — same instability the coverage
   marginals have shown on every fresh-seed pass, and the reason this
   experiment ran under the three-block protocol from the start. The pooled
   60-seed CI is the honest number; a single 20-seed block could have
   reported the heuristic's quality gain anywhere from dominant to absent.
5. Committed defaults untouched (uniform depth path byte-identical;
   regression 44.8 / 1.39 / 2.0948 exact; 43 tests pass).

The live-phase implication sharpens Phase 2's arm choice: **if live research
quality is coverage-limited, the quality lever is the search machinery, not
the SCT coupling** — and the coupling can be layered on for its validated
organization effects (cohesion §9, group formation §11) without paying a
measurable quality tax. The two mechanisms are complementary, not competing.

Run dirs: `compare-laws-N20-20260703T010136` (block 0), `-010219` (block
100), `-010302` (block 200); pilots `compare-laws-N10-20260703T*`; regression
`paired-N10-20260703T005459`; pooled aggregate with CIs in
`sim_runs/coverage_quality_20260703.json`. Repro:

```powershell
# six arms, jackpot landscape, three disjoint 20-seed blocks
foreach ($off in 0,100,200) {
    uv run python -m scas.sim --compare-laws --paired-seeds 20 --depth-dist jackpot `
        --steps 40 --K 6 --K-prime 10 --landscape-sigma 0.30 --seed-offset $off
}
```

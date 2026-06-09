# SCAS — SCT Dynamical Law vs Engineered Heuristic (head-to-head)

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

Sweep data: `sim_runs/hybrid_psi0_sweep.json`.

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

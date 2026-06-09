# SCAS Simulator POC — Review-Driven Hardening

> Generated after an AI-specialist review of the POC. Companion docs:
> [`results.md`](results.md), [`requirements.md`](requirements.md), [`design.md`](design.md).
> All runs: K=3, K'=5, steps=200, n_seeds=10, committed defaults unless noted.

## 0. Why this doc exists

The original writeup reported AC-B2 (coverage) and AC-B3 (separation) against a
single control — the *random-category* independent-walkers baseline. That control
conflates two effects: the **forced category partition** (each SCAS agent is
permanently pinned to one taxonomy category) and the **synergetic control law**
(attractor coupling via `alpha`). It also left two review concerns unmeasured:
whether the coordinator that "passed" is performance-blind (`W_P=0`), and whether
the proposed embedding mitigation actually works.

This pass adds the missing measurements. No headline number was deleted; the
honest decomposition is reported alongside.

## 1. New machinery (all under `scas/`)

- **`PartitionedNullCoordinator`** (`baselines.py`) — a fair control: same fixed
  category→group partition as SCAS, tracks group-best attractors so AC-B3 is
  measurable, but `alpha≡0` and the proposer is never steered. The *only*
  difference from SCAS is the control law.
- **`sim.py --paired-seeds`** now runs this third arm by default and reports the
  decomposition (`--no-partition-control` to skip).
- **First-class `--w-p` / `--psi-0`** flags (were env-var-only) so the
  performance-aware variant is directly sweepable.
- **`replay.py`** now measures the `[CATEGORY]`-prefix mitigation head-to-head
  against the raw embedding in the same run.

## 2. AC-B2 decomposition — coverage speedup is mostly the partition + a sub-threshold control-law effect

`steps_to_80%` coverage, lower is faster (committed defaults, `W_P=0`):

| Arm | steps_to_80% | — |
| --- | --- | --- |
| SCAS (full control law) | 44.8 | — |
| Partitioned, **uncoupled** (fair control) | 62.3 | — |
| Random-category baseline (original control) | 75.5 | — |

| Ratio | Value | Reading |
| --- | --- | --- |
| Headline `baseline / scas` | **1.69** | the originally-reported number (reproduced) |
| Partition alone `baseline / partition` | 1.21 | what the forced partition buys, *no control law* |
| **Control law marginal `partition / scas`** | **1.39** | what the synergetic coupling adds on top |

**Finding.** The headline 1.69× factors as 1.21 (partition) × 1.39 (control law).
Against the fair control, the synergetic coupling's marginal coverage speedup is
**1.39×, below the POC's own 1.5 bar**. The 1.5 threshold is cleared only because
the partition effect stacks on top — and the partition is hand-imposed, not an
emergent property of the law. AC-B2 should be read as "the control law gives a
real but sub-1.5× marginal coverage benefit; the rest is partitioning."

## 3. AC-B3 decomposition — separation is ~90% structural

inter/intra group embedding distance (≥1.5 passes):

| Arm | inter/intra |
| --- | --- |
| SCAS (full control law) | 2.09 |
| Partitioned, **uncoupled** (fair control) | **1.89** |

**Finding.** The uncoupled control already scores 1.89 — it passes AC-B3 on its
own. The control law lifts separation only from 1.89 → 2.09 (~10%). "No mode
collapse" is therefore **overwhelmingly an artifact of the fixed disjoint
categories and their prefix-anchored templates**, not a behavior the coupling
produces. The earlier writeup's framing of AC-B3 as the "most robust, least
tuning-dependent" pass is correct precisely *because* it is structural. AC-B3 as
specified does not isolate a coordination behavior; it should be re-scoped or
re-measured on a non-partitioned / emergent-grouping setup before it counts as
evidence for the SCT framing.

## 4. Performance-aware variant — the committed coordinator is performance-blind

The committed law uses `W_P=0`, so `val_bpb` never enters `psi`/`alpha`. Turning
the perf-gap term back on (N=10, `PSI_0=2.0`):

| `W_P` | AC-B2 headline | control-law marginal | AC-B3 | best_val_bpb Δ vs baseline |
| --- | --- | --- | --- | --- |
| 0 (committed) | 1.69 | 1.39 | 2.09 | −0.0021 |
| 5 | 1.61 | 1.33 | 2.15 | −0.0020 |
| 50 (original spec) | **0.95** | 0.79 | 2.60 | −0.0015 |

**Finding.** A *moderate* performance-aware setting (`W_P=5`) keeps coverage
essentially intact (marginal 1.33×) while letting the coordinator use the
val_bpb signal, and slightly improves separation. The original `W_P=50` re-breaks
coverage (SCAS becomes slower than baseline, 0.95×) — the perf-gap saturation trap
documented on Day 3. So a performance-aware coordinator is viable at moderate
`W_P`, but the *specific* configuration validated by the headline ACs ignores
performance. Any live phase should sweep `W_P` jointly with `coupling_T` rather
than inherit `W_P=0`.

## 5. Embedding mitigation — the `[CATEGORY]` prefix works on the replay fixture

Replay over the 30-row fixture, raw vs `[CATEGORY]`-prefixed embedding, same rows:

| Metric | Raw | Prefixed |
| --- | --- | --- |
| Mean silhouette (≥0.05 passes) | 0.029 **FAIL** | **0.130 PASS** |
| optimizer class | 0.059 | 0.156 |
| **architecture class** | **0.0001** | **0.084** |
| regularization class | 0.057 | 0.207 |

**Finding.** The mitigation flagged in `results.md` is now **measured, not
assumed**: prepending the agent's category tag lifts the failing case above
threshold (+0.10) and specifically rescues the architecture class (0.0001 →
0.084). Caveats: (a) this is a hand-curated 30-row fixture, not real
`results.tsv`; (b) it is partly circular — hand-prefixing the category injects the
very partition the system is meant to exhibit, so it validates *engineering
feasibility of separable embeddings*, not *emergent diversity*. It does, however,
de-risk the embedding soft point for a live phase.

## 6. Net effect on the go/no-go

Unchanged: the engineering, reproducibility, and safety envelope are solid.
Sharpened: the two headline behavioral claims are weaker than first reported —
AC-B3 is ~90% structural, and the control law's marginal AC-B2 contribution
(1.39×) is below the 1.5 bar against a fair control. De-risked: a performance-aware
coordinator is viable at moderate `W_P`, and the embedding mitigation works on
available data. Recommendation stands at **proceed only to a small, narrowly-scoped
next experiment** (perf-aware `W_P×T` sweep + an emergent-grouping AC-B3 test +
real-`results.tsv` replay), not a full live multi-agent build.

## 6b. Attempt to pass AC-B2 via fixes #1–#3 — does not lift the marginal (hardened falsification)

The root-cause analysis proposed three principled fixes to raise the control-law
marginal: **#1** decouple agents from groups (multi-agent groups), **#2** real
intra-group coverage-memory repulsion, **#3** sibling dispersion. All three are
implemented behind flags (`--agents-per-group`, `--coverage-coupling`; committed
defaults unchanged). N=10 ablation, all arms matched on agent count:

| Config | SCAS steps→80% | partition control | **control-law marginal** | AC-B3 |
| --- | --- | --- | --- | --- |
| M=1, no coupling (committed) | 44.8 | 62.3 | **1.39** | 2.09 |
| M=3, no coupling (#1 alone) | 77.4 | 62.3 | **0.80** | 2.09 |
| M=1, coverage-coupling (#2/#3) | 53.2 | 62.3 | **1.17** | 1.88 |
| M=3, coverage-coupling (full #1–#3) | 55.0 | 62.3 | **1.13** | 1.87 |

**Finding.** None of the fixes lift the marginal; all are *below* the committed
M=1 baseline (1.39), and none clears 1.5. Mechanisms (real, not bugs):
- **More agents/group hurts (1.39→0.80):** group-mates co-chase the same attractor
  in exploit episodes → redundant draws. Coverage cost in agent-steps is
  parallelism-invariant for the uncoupled control (partition = 62.3 at every M),
  so SCAS's redundancy is a pure loss — classic premature consensus convergence.
- **Slot-space coverage memory slightly hurts:** minima are registered by
  *embedding* proximity and distinct slots embed close together, so avoiding
  already-tried *slots* can push an agent out of a minimum's embedding
  neighborhood. Anti-redundancy in slot space ≠ anti-redundancy in embedding space.

This is a **stronger negative result**: the synergetic coupling does not earn a
1.5× coverage advantage over a fair partitioned control even when handed
multi-agent groups and explicit anti-redundancy. Caveats: a smarter
embedding-space (rather than slot-space) repulsion, or exploit-gating so
group-mates don't co-cluster, is not yet exhausted; and AC-B3's lever (#4,
anneal alpha explore→converge) remains untested.

Run dirs: `paired-N10-20260528T194108` (M=1 no-cc), `…194117` (M=3 no-cc),
`…194142` (M=1 cc), `…194154` (M=3 cc).

## 6c. Embedding-space repulsion + exploit-gating — AC-B2 control-law marginal passes at K'=5 (with a confound ruled out)

Slot-space repulsion (§6b) failed because minima register by *embedding* proximity
and distinct slots embed close together. The embedding-space variant
(`--embedding-repulsion`) fixes the search to operate where the metric lives:
- **Explore** = farthest-point in MiniLM space (unit-norm ⇒ cosine), picking the
  candidate least similar to the group's covered set (shared across group-mates).
- **Exploit-gating** = at most one agent per group per step pursues the attractor,
  so group-mates stop co-clustering (the §6b failure mode).

N=10 unless noted, marginal = `partition_control_steps / scas_steps`:

| Config | SCAS s→80% | partition control | **control-law marginal** |
| --- | --- | --- | --- |
| M=3, embedding-repulsion, K'=5 | 40.5 | 62.3 | **1.54 ✅** |
| M=3, embedding-repulsion, K'=5, **N=20** | 33.6 | 65.0 | **1.94 ✅ (robust)** |
| M=1, embedding-repulsion, K'=5 | 48.1 | 62.3 | 1.30 ✗ (needs multi-agent groups) |
| M=3, embedding-repulsion, **K'=10** | 56.0 | 72.6 | 1.30 ✗ (crowding under partition) |

**The pass is genuine coordination, not a search-strategy confound.** Because the
fair control samples uniformly while SCAS uses farthest-point, I added the decisive
control (`--independent-repulsion`): the *same* farthest-point search but per-agent
(no shared memory, no gating). At M=3/K'=5 it reaches 80% in **72.4** steps —
*slower* than uniform random (62.3) — because uncoordinated agents chase the same
novel regions and drift toward embedding-space outliers away from the minima. So
the search strategy alone does **not** help (marginal 0.86); the entire 1.54–1.94×
gain comes from the **shared coverage memory + exploit-gating**, i.e. coordination.

| M=3, K'=5, N=10 | s→80% | vs uniform partition (62.3) |
| --- | --- | --- |
| uniform-random partition (official control) | 62.3 | — |
| independent farthest-point (search only) | 72.4 | 0.86 (search alone hurts) |
| coordinated farthest-point (shared + gated) | 40.5 | **1.54** |

**Scope limit — the win is menu-dependent.** The embedding-repulsion machinery
pre-embeds the *entire finite candidate pool* (~100 strings/category) and the
minima are sampled from that same pool, so farthest-point selection over a fully
known menu is close to a greedy max-coverage algorithm run against the answer
key. A live LLM phase cannot enumerate and pre-embed its hypothesis space; the
heuristic's coverage speedup therefore transfers *less* readily than the SCT
arms (which need only an attractor and a similarity query). Treat 1.94× as an
upper bound specific to closed candidate menus.

**Verdict.** AC-B2's control-law marginal **passes at K'=5** (robustly: 1.94 at N=20)
and the gain is attributable to coordination. It **fails at K'=10** (1.30) — the
predicted "crowding under partition" when minima outnumber categories enough that
groups can't divide them cleanly. So the honest claim is scoped: *the SCT coupling
earns a ≥1.5× coverage advantage over a fair partitioned control when minima are not
heavily crowded relative to the group count (K' ≲ ~2× #groups).* **AC-B3's
control-law lift is still unmet** (SCAS ~2.18 vs control ~1.89–1.91, lift ~0.27 <
0.5) — embedding repulsion targets coverage, not group tightening; that needs the
untested alpha-annealing lever (#4).

All committed defaults remain unchanged and reproduce exactly (M=1, no flags →
44.8 / marginal 1.39 / AC-B3 2.09); validator 20/20.

Run dirs: `paired-N10-20260528T195734` (M=3 K'=5), `paired-N20-20260528T200205`
(N=20), `paired-N10-20260528T195759` (M=1), `paired-N10-20260528T200138` (K'=10),
`paired-N10-20260528T200048` (independent-repulsion control).

## 6d. Alpha-annealing (#4) — AC-B3's control-law lift passes, with AC-B2 intact

AC-B3's lift failed (§6c) because embedding-repulsion targets *coverage*, not group
*tightening*: both SCAS and the control kept a wide intra-group spread. Fix #4
(`--anneal`) adds the synergetic late-time settling the theory predicts — built on
the shared embedding-repulsion machinery:
- **Explore phase** (run progress < `anneal_start`, default 0.5): unchanged §6c
  behavior (farthest-point + gated exploit). Secures coverage.
- **Converge phase** (progress ≥ `anneal_start`): every group-mate settles onto its
  group attractor (gate lifted, covered-set ignored) → intra-group spread shrinks.

Because 80% coverage is reached in ~33 agent-steps — far inside the explore phase —
convergence does not touch AC-B2. The two ACs stop competing.

N=20 at K'=5 (robust headline), N=10 at K'=10:

| Anneal, M=3 | AC-B2 marginal | AC-B3 inter/intra | control inter/intra | **AC-B3 lift** |
| --- | --- | --- | --- | --- |
| K'=5, N=20 | **1.94 ✅** | 4.22 | 1.90 | **+2.32 ✅** |
| K'=10, N=10 | 1.30 ✗ | 4.21 | 1.89 | **+2.33 ✅** |

**Finding.** Annealing lifts AC-B3 from +0.27 (unmet, §6c) to **+2.32** — far past the
0.5 bar — and AC-B2 is *byte-identical* to the non-anneal run (33.55 steps→80%),
confirming convergence and coverage no longer trade off. The lift is the control
law's: the fair control has the *same* M=3 group structure and group-best attractor
but never converges (intra stays wide), so the entire reduction in intra is the
coordinated late-time settling onto the shared consensus attractor — exactly the
"groups settle on distinct attractors" behavior AC-B3 names, and what `T·ψ̇ + ψ = 0`
predicts. This is convergence *with* diversity (inter stays high, ~prefix-driven),
not mode collapse (groups do not merge).

**Joint verdict (K'=5).** With multi-agent groups + embedding-repulsion +
exploit-gating + annealing, **both AC-B2 (1.94×) and AC-B3 (lift +2.32) pass against
the fair control simultaneously**, AC-B1' holds, and committed defaults are
untouched (no flags → 1.39 / 2.09; validator 20/20). The remaining scope limit is
**K'=10**, where AC-B2 still fails (1.30, crowding under partition) though AC-B3
passes. Honest joint claim: *the SCT coupling delivers diversity-with-selective-
convergence over a fair partitioned control when minima are not heavily crowded
relative to group count (K' ≲ ~2× #groups).*

Run dirs: `paired-N20-20260528T201717` (K'=5), `paired-N10-20260528T201800` (K'=10).

## 6e. K'=10 with more groups — the "crowding" limit is not fundamental

The only remaining failure (§6c–§6d) was AC-B2 at K'=10 with G=3 groups (1.30):
3 categories cannot divide 10 minima, so each group's partition is over-crowded.
The taxonomy was extended to 6 categories (`schedule`, `data`, `init` appended —
the first 3 keys and their 330 strings are byte-identical, so committed defaults
reproduce exactly: M=1/no-flags still 1.39 / 2.09, validator 20/20), and
`make_landscape` now samples minima only from the in-use categories. Sweeping the
group count at K'=10 (M=3, anneal):

| Groups G | crowding K'/G | AC-B2 marginal | AC-B3 inter/intra (control) | AC-B3 lift |
| --- | --- | --- | --- | --- |
| 3 | 3.3 | 1.30 ✗ | 4.21 (1.89) | +2.33 ✅ |
| 4 | 2.5 | **1.65 ✅** | 4.56 (1.99) | +2.57 ✅ |
| 5 | 2.0 | **1.87 ✅** | 4.51 (2.02) | +2.48 ✅ |
| 6 | 1.67 | **1.61 ✅** (N=20: 1.61) | 4.42 (1.99) | +2.43 ✅ |

**Finding.** AC-B2 recovers as soon as G ≥ 4 (crowding ≤ 2.5) and both ACs then pass
jointly at K'=10. The K'=10 failure was therefore **not fundamental** — it was a
mismatch between group count and minima density. The operating rule is simply
**G ≳ K'/3** (keep crowding ≲ ~2.5): choose enough groups for the expected number
of distinct optima and the coordination law delivers diversity-with-selective-
convergence. This removes the last scope caveat from §6c–§6d.

Run dirs: `paired-N10-20260528T203032` (G=6 N=10) plus the G=4/5 and G=6-N=20 sweep.

## 6f. Fresh-seed confirmation (2026-06-09) — §6c/§6d's "robust" AC-B2 pass does not replicate

Every run in §6b–§6e used seeds 0..N−1 — the same seeds the variant pipeline was
*tuned* on. A one-shot confirmation of the §6d headline config (M=3, K'=5,
anneal, N=20) on never-touched seeds 100–119 (`--seed-offset 100`) gives
**AC-B2 marginal 1.42 (< 1.5)** vs 1.94 in-sample, while the AC-B3 lift
replicates (+2.40, scripted by the anneal as documented). So §6c/§6d's "robust"
label is withdrawn: the variant's coverage advantage over the fair control is
real, but the 1.5× threshold claim is in-sample only. Two related revisions from
the same review pass: (a) the §6c "search alone hurts" control (0.86× at N=10)
re-measured at N=20 inside `--compare-laws` is **1.03× [0.74, 1.40] — a wash,
not a harm** (the decomposition conclusion stands: coordination, not search,
carries the speedup); (b) the SCT integrator bug and its consequences are in
[`results-laws.md`](results-laws.md) §9. Run dir: `paired-N20-20260609T183944`.

## 7. Reproducibility

```powershell
# AC-B2/AC-B3 decomposition (fair control runs by default now)
uv run python -m scas.sim --paired-seeds 10

# Performance-aware variants
uv run python -m scas.sim --paired-seeds 10 --w-p 5.0
uv run python -m scas.sim --paired-seeds 10 --w-p 50.0

# Embedding mitigation (raw vs [CATEGORY]-prefixed silhouette in one run)
uv run python -m scas.replay

# §6b — fixes #1-#3 ablation (multi-agent groups + slot-space coverage repulsion)
uv run python -m scas.sim --paired-seeds 10 --agents-per-group 3
uv run python -m scas.sim --paired-seeds 10 --agents-per-group 1 --coverage-coupling
uv run python -m scas.sim --paired-seeds 10 --agents-per-group 3 --coverage-coupling

# §6c — embedding-space repulsion + exploit-gating (AC-B2 passes at K'=5)
uv run python -m scas.sim --paired-seeds 20 --agents-per-group 3 --embedding-repulsion
uv run python -m scas.sim --paired-seeds 10 --agents-per-group 3 --embedding-repulsion --K-prime 10
# decisive control — search strategy without coordination:
uv run python -m scas.sim --paired-seeds 10 --agents-per-group 3 --independent-repulsion

# §6d — alpha-annealing (#4): AC-B3 lift passes with AC-B2 intact (K'=5)
uv run python -m scas.sim --paired-seeds 20 --agents-per-group 3 --anneal
uv run python -m scas.sim --paired-seeds 10 --agents-per-group 3 --anneal --K-prime 10

# §6e — relieve K'=10 crowding with more groups (needs the 6-category taxonomy)
uv run python -m scas.sim --paired-seeds 20 --K 6 --agents-per-group 3 --anneal --K-prime 10
uv run python -m scas.sim --paired-seeds 10 --K 4 --agents-per-group 3 --anneal --K-prime 10
```

Run dirs: `paired-N10-20260528T190733` (W_P=0), `…190742` (W_P=5),
`…190751` (W_P=50), `replay-20260528T190812`.

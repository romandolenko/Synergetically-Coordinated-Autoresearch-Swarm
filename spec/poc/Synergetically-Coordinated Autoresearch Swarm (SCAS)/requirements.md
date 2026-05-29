# SCAS Simulator POC — Requirements

> Companion docs: [`design.md`](design.md) (how), [`tasklist.md`](tasklist.md) (execution), [`simulator-only.md`](simulator-only.md) (unified spec).

## 1. Purpose

Falsify or validate — cheaply, offline, on a laptop — whether the SCT-derived multi-consensus coordination law produces **diversity-with-selective-convergence** behavior on a hypothesis-exploration task structurally similar to autoresearch's outer loop. Outcome of the POC is a go / no-go signal for investing in a live multi-agent phase.

## 2. Scope

**In scope.**
- A self-contained Python simulator that drives the real coordinator logic against a synthetic hypothesis stream.
- A replay mode that swaps the synthetic stream for real `results.tsv` rows (when available).
- An "independent walkers" baseline for head-to-head comparison.
- Plots and a unit-tested safety-envelope validator.

**Out of scope (explicit non-goals).**
- Any live LLM agent, any GPU usage, any modification to `train.py` / `prepare.py` / `pyproject.toml`.
- Any model of training dynamics or analytical model of "edit → loss".
- A coordinator HTTP service, agent-side wrappers, dashboards, or paired live runs (these live in the live-agent Phase 2 captured in [`general-v1.md`](general-v1.md)).
- Claims about scaling to 100+ agents, robustness of embeddings at production scale, or improvements to the inner agent.

## 3. User stories

**US-1 — SCT-curious developer**
*As a* developer evaluating whether SCT-derived multi-consensus coordination is worth wiring into autoresearch,
*I want* an offline simulator that runs the proposed coordination law against synthetic and replayed hypothesis streams,
*so that* I can decide — in hours, on a laptop, with no GPU — whether the law produces the diversity-with-selective-convergence behavior its theory claims.

**US-2 — Researcher tuning the diversity knob**
*As a* researcher who would eventually deploy this,
*I want* a single `coupling_T` knob whose effect on group separation and exploration breadth is monotonic and predictable,
*so that* the behavior maps to an intuition I can hold ("loosen the swarm" vs "tighten it") instead of a hyperparameter zoo.

**US-3 — Reviewer auditing claims**
*As a* reviewer of the POC,
*I want* every reported behavioral claim to be reproducible from a single `uv run sim.py --seed N` command, with figures regenerated from raw JSONL,
*so that* I don't have to take the experimenter's word for any of it.

## 4. Acceptance criteria

### 4.1 Must-pass (headline behavioral claims)

> **Official control (post-review).** The baseline of record for AC-B2 and AC-B3 is the **partitioned fair control** (`PartitionedNullCoordinator`, design §9): same fixed category→group partition as SCAS, attractors tracked, but the synergetic coupling switched off (`alpha ≡ 0`, proposer never steered). It is the only control that isolates the **control law** from the hand-imposed **category partition**. The random-category independent-walkers baseline is retained as a secondary diagnostic but is no longer the headline comparison, because it conflates the two effects. See [`results-review.md`](results-review.md) §2–§3.

| ID | Claim | Measurement |
| --- | --- | --- |
| **AC-B3** | No mode collapse — *attributable to the control law* | After a 200-step sim with K=3 groups, two conditions over ≥10 seeds: (1) absolute mean inter-group ≥ 1.5× mean intra-group distance, AND (2) a **control-law lift** over the partitioned fair control — SCAS inter/intra must exceed the fair control's inter/intra by ≥ 0.5 absolute (i.e. the coupling must add separation beyond what the disjoint prefixed categories give for free). Observed: SCAS 2.09 vs fair control 1.89 — condition (1) passes, **condition (2) fails** (lift 0.20). Verdict: separation is largely structural; AC-B3 as a *control-law* claim is **not met**. See [`results-review.md`](results-review.md) §3. |
| **AC-B2** | Coverage speedup — *control-law marginal* | SCAS reaches 80% ground-truth-minima coverage at least 1.5× faster than the **partitioned fair control**, averaged over ≥10 seeds: `mean(partition_control_steps_to_80%) / mean(scas_steps_to_80%) ≥ 1.5`. Per-run "never reached" is bounded at the total agent-step budget (`steps * K`). Observed: 62.3 / 44.8 = **1.39 — does not clear 1.5**. (Secondary diagnostic vs random-category baseline: 1.69×, but that includes the 1.21× partition effect.) Revised in `results-day3.md` §6 (end-of-run-count → time-to-coverage) and again post-review (random baseline → fair partition control). See [`results-review.md`](results-review.md) §2. |
| **AC-B4** | Knob monotonicity | Sweeping `coupling_T` over ≥5 values, mean inter-group separation is monotonically related to T (`\|Kendall's τ\| ≥ 0.7` over the seed average). The control law produces a predictable monotonic response; the sign is determined by the relative position of `PSI_0` and the typical operating `psi`. Revised in `results-day4.md` §4 from the original signed-threshold formulation (`τ ≤ −0.7`), which assumed `PSI_0 < typical_psi`; the committed defaults (`PSI_0 = 2.0`) flip that assumption and the response direction with it. |

### 4.2 Sanity / must-not-fail

| ID | Claim | Measurement |
| --- | --- | --- |
| **AC-B1'** | Best-val_bpb no-worse-than-independent | SCAS's mean best-val_bpb is within 1 standard error of the independent baseline. Coordination must not actively hurt the headline metric. |
| **AC-B5'** | Safety envelope | A test harness of 20 hand-written adversarial "commits" (touching `prepare.py`, claiming implausibly low val_bpb, exceeding crash budget, etc.) is rejected with 100% accuracy by the validator. |

### 4.3 Engineering

| ID | Requirement |
| --- | --- |
| **AC-E1** | Single `uv run sim.py` entrypoint. No GPU. No network at runtime (embedding model bundled or pre-cached). |
| **AC-E2** | All state and trace data is JSONL under `./sim_runs/<run_id>/`. Schema documented in [`design.md`](design.md). |
| **AC-E3** | Plots for AC-B2 / AC-B3 / AC-B4 are regenerable from JSONL via `uv run plot.py <run_id>`. |
| **AC-E4** | All new code lives under a new `scas/` subdirectory. Zero touches to `train.py`, `prepare.py`, `pyproject.toml` (other than adding dependencies the simulator needs — sentence-transformers, scikit-learn, matplotlib). |

## 5. Definition of done

The POC delivers a go/no-go signal — a *pass* and an honestly-reported *fail* are both valid completions. The POC is done when:
1. All headline experiments run on a clean checkout with `uv run sim.py --all-experiments` (paired runs include the partitioned fair control by default), and each AC is reported against its **official control** (§4.1). Current verdicts: AC-B4 **passes**; with **committed defaults** AC-B2 and AC-B3 **pass in absolute terms but fail as control-law claims** against the fair control (marginal 1.39 < 1.5; lift 0.20 < 0.5). A **coordination variant** (multi-agent groups + embedding-space repulsion + exploit-gating + alpha-annealing, all behind flags) makes **both pass jointly** (AC-B2 marginal 1.94 at K'=5/G=3, AC-B3 lift +2.32; confound ruled out). The K'=10 crowding failure (G=3) is resolved by scaling the group count — both pass at G≥4 (operating rule **G ≳ K'/3**). See [`results-review.md`](results-review.md) §6c–§6e.
2. AC-B5' passes via `uv run pytest scas/validator_tests.py`.
3. A short results writeup (figures + 1-paragraph interpretation per AC) is committed alongside the code.
4. Replay-mode dry-run against any existing `results.tsv` produces non-degenerate embeddings, or that finding is documented (raw silhouette 0.029 fails; the `[CATEGORY]`-prefix mitigation is measured and lifts it to 0.130 — embedding fragility *and* its mitigation are publishable POC outcomes).

## 6. Out-of-band falsification cases

These are *not* failures of the POC implementation — they are valid POC outcomes that conclude the SCT framing is the wrong fit:

- AC-B4 fails (no `coupling_T` ordering produces monotonic separation) → the single-knob promise of SCT framing does not hold.
- AC-B3 fails even with tight `coupling_T` → multi-consensus does not separate groups in this embedding.
- **AC-B2 / AC-B3 clear the absolute bar but not the control-law bar against the partitioned fair control** → the observed diversity-with-convergence is mostly the hand-imposed category partition, and the synergetic coupling adds a real but sub-threshold marginal effect. *(This is the current observed outcome — see §4.1 and [`results-review.md`](results-review.md).)* The conclusion is that a live phase must demonstrate the law's value beyond a fixed partition, not that the simulator is broken.
- Replay-mode shows real `results.tsv` summaries cluster trivially (everything looks the same to MiniLM) → embedding-quality risk dominates and the live phase needs a different embedder or the `[CATEGORY]`-prefix anchor.

Each of these is reported as the conclusion, not papered over.

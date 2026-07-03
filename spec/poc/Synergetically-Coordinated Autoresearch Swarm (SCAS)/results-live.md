# SCAS Phase 2 — live-run results

Append-only, same conventions as `results-review.md` / `results-laws.md`.
Covers runs of the *live* stack (`scas/service.py` + `scas/loop.py`) with a
real LLM writing the hypothesis cards, as opposed to the offline simulator.

## 1. Live K=3 paired run, GPU-free half (2026-07-03) — prompts steer a real LLM the way `bernoulli(alpha)` steers the synthetic proposer

### 1a. What this is and is not

Phase 2 item 3 calls for a live K=3 paired run on real `train.py`
experiments. The GPU half is **blocked on hardware**: this machine's RTX
3050 Ti Laptop GPU is inaccessible to CUDA (driver/permission state), and
`train.py`'s support matrix excludes laptop GPUs and <8 GB VRAM regardless.
What ran instead is the **GPU-free half**, which measures the item's stated
key unknown — the LLM's *effective alpha* — plus emergent grouping on real
LLM-written cards and the live rejection rate.

Design (mirrors the offline arms):

* **Coordinated arm**: agents `c0,c1,c2` against a live emergent-SCT service
  (defaults: ψ₀=0.5, α_temp=0.3, T=1). Every step: `scas.loop prompt` draws
  `bernoulli(alpha)` and prints a REFINE or EXPLORE packet; a real LLM
  (Claude, this session) writes a genuine hypothesis card following it;
  `scas.loop report` posts the result.
* **Independent arm**: agents `i0,i1,i2`, free ideation with **no packets**,
  reported to a matched coupling-off null service (`--psi-0 1e9`) — the §11
  discipline: identical grouping/measurement machinery over uncoupled agents.
* **Surrogate val_bpb (declared)**: the POC's own synthetic landscape
  (seed 0, K'=5, noise 0) evaluated at the MiniLM embedding of each card.
  Deterministic, gives the attractor dynamics real relative signal, and
  **makes any best-val/quality comparison meaningless by construction** —
  quality claims stay blocked until the GPU half runs.
* 8 rounds × 3 agents × 2 arms = 48 cards; one deliberate OOM-crash report
  per arm to exercise the envelope; keep/discard by each agent's own best
  (program.md's advance rule). Independent cards were written before the
  round's packets were read, to limit cross-contamination.

**Validity caveats, stated up front**: the agent and the experimenter are
the same LLM, which knows the hypothesis under test — compliance numbers
are a *cooperative ceiling*, not a robustness result. One session, N=24
reports/arm, no seed blocks: directional evidence, not CI-grade.

### 1b. Effective alpha — the key unknown, first measurement

`scas.loop analyze` joins each directive with the card that followed and
scores, in MiniLM space, margin = cos(card, target) − mean cos(card, avoid):

| Directive | n | mean margin | compliance (margin sign) |
| --- | --- | --- | --- |
| REFINE | 11 | **+0.243** | **11/11 = 1.00** |
| EXPLORE | 7 | +0.003 | 0.43 |

Refine−explore separation **+0.24**, positive for every agent
(+0.21/+0.24/+0.25).

Reading: REFINE packets pull proposals decisively toward the target — every
refine card landed nearer its target than the avoid-list. EXPLORE cards land
*neutral* (margin ≈ 0), not anti-aligned — and that is the **correct analog
of the synthetic proposer**, whose explore mode is a uniform draw, not
repulsion; the sign-based "compliance" bar is too strict for explore, and
the margin is the informative statistic. The two directive modes produce
cleanly distinguishable proposal distributions: **a real LLM's effective
alpha exists and is steerable by the packet text.** This was the single
biggest unknown between the simulator and a live build.

The gate dynamics also behaved live: alpha started at 0.16 (each agent at
its own attractor → explore-heavy early), rose to 0.55–0.90 as psi
integrated real mismatch, and the bernoulli draws visibly mixed directives
at intermediate alpha (e.g. α=0.73 drawing EXPLORE).

### 1c. Emergent grouping on real cards — directionally positive, weak, honest

| Arm | inter/intra (centroids) | silhouette | group sizes |
| --- | --- | --- | --- |
| Coordinated (SCT live) | 0.85 | 0.055 | 3 / 12 / 8 |
| Independent (matched null) | 0.77 | 0.044 | 7 / 9 / 7 |

The coupling edges out the null on both measures, but the absolute level is
nowhere near the simulator's templated-card numbers (inter/intra 2.4–3.0):
real free-text cards written by one model in one session carry far less
MiniLM cluster structure than prefix-anchored templates. This is the
live-side echo of §8's T5.2 finding (raw silhouette 0.022 on realistic
text) and it sharpens the same conclusion: **the emergent coordinator's
group formation needs more embedding structure than raw realistic cards
provide.** Membership churn, empty-slot seeding, and attractor takeover all
functioned mechanically (three distinct attractor regimes formed:
attention/QK → muP, data-mix, and a residual slot).

### 1d. Envelope, live

1 deliberate crash per arm rejected correctly (val floor), 0 false
rejections on 46 legitimate reports → rejection rate 4.2% per arm, all
intended. A mid-run service kill was recovered by replaying the wrapper
traces through `/step` — 18 reports replayed with **zero** state mismatches,
confirming the coordinator dynamics are deterministic and the trace is a
sufficient event log.

### 1e. What remains blocked

The quality half of item 3 — best-val_bpb of coordinated vs independent on
*real training runs*, same wall-clock — needs a working CUDA box with a
supported GPU. Everything else about the run (services, wrapper, packets,
reporting, analysis) is now exercised end to end and can be pointed at real
`train.py` experiments unchanged.

Primary data: `live-run-20260703/` (directive+report traces for all six
agents, pooled metrics JSON, driver script). Repro of the analysis:

```powershell
uv run python "live-run-20260703/driver_live_run.py" finish   # after pointing ROOT at the traces
```

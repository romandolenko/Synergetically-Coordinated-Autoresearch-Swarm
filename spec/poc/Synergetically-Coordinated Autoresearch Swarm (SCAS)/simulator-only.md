# POC: Synergetically-Coordinated Autoresearch Swarm (SCAS) — v2 (simulator-only, coverage-first)

## 0. What changed from v1

- **No live agents.** Phase 2 is cut. The POC is a self-contained Python program that runs without GPUs, without LLMs, without touching the autoresearch loop.
- **Coverage is the headline.** AC-B2 (distinct local minima) and AC-B3 (group separation) are now must-pass. AC-B1 (best-val_bpb) is demoted to a sanity check and AC-B5 (safety envelope) is reduced to a unit-test gate, not a behavioral claim.

## 1. Premise and scope

Today autoresearch is **one agent, one GPU, one branch, one metric (val_bpb)**, looping every ~5 min. The investigation that motivated this POC concluded SCT does not apply to the *inner search* (no analytical model of "edit → loss"), but it *does* apply to the **coordination layer between agents** via multi-consensus + invariant-manifold framings. This POC tests exactly that claim — narrowly, offline, on synthetic data — without changing the inner agent at all.

**Non-goals for the POC.** No model of training dynamics. No distributed training. No modification of `prepare.py` or the evaluation harness. No bespoke LLM finetuning. No live multi-agent wiring. The 5-minute training budget and the single-GPU constraint stay sacred (and untouched, because no training runs).

## 2. User stories

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

## 3. Acceptance criteria

**Must-pass (headline)**
- AC-B3 — **No mode collapse.** At end of a 200-step simulation with K=3 groups, the pairwise mean inter-group embedding distance is ≥ 1.5× the mean intra-group distance, averaged over ≥10 seeds.
- AC-B2 — **Coverage gain.** SCAS visits ≥ 1.5× as many distinct local minima (defined as clusters in embedding space whose best-val_bpb is within 1% of a local optimum) as an "independent walkers" baseline at matched total step budget, averaged over ≥10 seeds.
- AC-B4 — **Knob monotonicity.** As `coupling_T` sweeps over ≥5 values from loose→tight, mean inter-group separation decreases monotonically (Kendall's tau ≤ −0.7) over the seed average.

**Sanity / must-not-fail**
- AC-B1' — **Best-val_bpb no-worse-than-independent.** SCAS's mean best-val_bpb is within 1 standard error of the independent baseline. (Coordination must not actively hurt the headline metric, even if it's not the win condition.)
- AC-B5' — **Safety envelope unit-tested.** A test harness that feeds 20 hand-written adversarial "commits" (modifying `prepare.py`, claiming impossibly low val_bpb, exceeding crash budget, etc.) shows the validator rejects 100% of them.

**Engineering**
- AC-E1 — Single `uv run sim.py` entrypoint, no GPU, no network at runtime (embeddings pre-cached or model bundled).
- AC-E2 — All state and trace data is JSONL under `./sim_runs/<run_id>/`.
- AC-E3 — Plots for AC-B2/B3/B4 regenerable from JSONL via `uv run plot.py <run_id>`.

## 4. Implementation design

### 4.1 What the simulator actually simulates

The simulator replaces the *outer loop*'s LLM proposer and GPU trainer with synthetic stand-ins, and runs the *real* coordinator logic against them. Per step per agent:

1. **Synthetic proposer** samples a "hypothesis card" — a short string from a templated taxonomy ("change optimizer to X", "increase depth to Y", "swap activation Z"), parameterized by the agent's current group attractor and its coupling weight α_i. High α_i → templates biased toward "small tweak to the current group leader's recipe"; low α_i → templates biased toward a taxonomy slot the agent hasn't tried.
2. **Synthetic evaluator** computes a fake val_bpb from a fixed *ground-truth landscape* — a mixture of Gaussians in embedding space with K' true local minima (K' chosen independent of K; K' > K stresses coverage, K' < K stresses group separation under crowding). Noise is added per call.
3. **Coordinator** receives the card + val_bpb, updates group attractors, recomputes ψ_i and α_i for next step.

The ground-truth landscape is fixed per seed and known to the analyzer, so AC-B2's "distinct local minima visited" is measurable against ground truth, not just self-reported clusters.

### 4.2 Files (new — all under a new `scas/` subdir, none touching `train.py` / `prepare.py`)

```
scas/
  coordinator.py        # control law, group/attractor mgmt, validator
  embedder.py           # all-MiniLM-L6-v2 wrapper, cache to disk
  synthetic.py          # proposer + evaluator + ground-truth landscape
  sim.py                # entry point: parses args, runs sim, writes JSONL
  baselines.py          # independent-walkers baseline
  plot.py               # AC-B2/B3/B4 figures
  validator_tests.py    # AC-B5' adversarial commit cases
```

### 4.3 Key data structures

```python
@dataclass
class HypothesisCard:
    agent_id: str
    step: int
    summary: str
    category: str         # taxonomy slot, e.g. "optimizer"
    parent_step: int | None

@dataclass
class StepResult:
    card: HypothesisCard
    val_bpb: float
    embedding: np.ndarray  # 384-d from MiniLM
    accepted: bool         # passed safety envelope
    reject_reason: str | None

@dataclass
class CoordinatorState:
    groups: dict[int, list[str]]            # group_id -> [agent_ids]
    attractors: dict[int, np.ndarray]       # group_id -> embedding of group-best card
    group_best_val_bpb: dict[int, float]
    psi: dict[str, float]                   # per agent
    alpha: dict[str, float]                 # per agent, in [0,1]
```

### 4.4 The control law (concrete)

For agent i in group g(i) at step t:

```
psi_i = w_d * dist(z_i, c_g) + w_p * max(0, val_bpb_i - val_bpb_g_best)
alpha_i = sigmoid((psi_i - psi_0) / coupling_T)
```

with defaults: `w_d = 1.0`, `w_p = 50.0` (val_bpb is ~1.0, embedding distance is ~1.0 — w_p brings them onto a comparable scale), `psi_0 = 0.5`, `coupling_T ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` swept for AC-B4.

The synergetic dynamics `T·ψ̇ + ψ = 0` is *not* numerically integrated; the formula above is the steady-state interpretation: large ψ_i ⇒ agent is far from its group's attractor (in idea-space *or* in performance) ⇒ next-step coupling pushes it back. The proposer is then called with:
- `target = attractors[g(i)]` if α_i > 0.5 (tighten toward group leader)
- `avoid = [attractors[g'] for g' != g(i)]` if α_i ≤ 0.5 (push away from other groups)

This is the **entire** SCT contribution. Everything else is bookkeeping.

### 4.5 Invariant-manifold safety envelope (validator)

Encoded as a `validate(commit)` function on the coordinator:
- `train.py` is the only file changed (matches `program.md`'s existing rule)
- `pyproject.toml` and `prepare.py` are byte-identical to baseline
- claimed val_bpb is plausible (rejects implausibly-low claims as a stand-in for re-eval, which is unavailable in simulator)
- agent has not exceeded a per-window crash budget (e.g. ≥3 failures in last 5 commits → quarantine)

For the simulator, these are tested via `validator_tests.py` with 20 hand-written adversarial cases (AC-B5').

### 4.6 Independent-walkers baseline (for AC-B2 / AC-B1')

Identical synthetic proposer and evaluator, with the coordinator replaced by a no-op: every agent gets α_i = 0 and no group structure. Runs with the same K, same total step budget, same seeds. This is the head-to-head.

### 4.7 Phased plan (single phase, ~3–5 days)

1. **Day 1.** Stand up `embedder.py`, `synthetic.py` (proposer + landscape only, no evaluator yet), `coordinator.py` skeleton. Write `validator_tests.py` and pass AC-B5'.
2. **Day 2.** Wire `synthetic.evaluator`, complete `coordinator.py` control law, get end-to-end `sim.py` running for K=3, 200 steps, single seed. Dump JSONL.
3. **Day 3.** Implement `baselines.py` and run paired seeds (≥10) for AC-B1' / AC-B2 / AC-B3. Build `plot.py`.
4. **Day 4.** Sweep `coupling_T` for AC-B4. Iterate on `w_d` / `w_p` / `psi_0` if needed.
5. **Day 5 (buffer).** Replay-mode: add a switch that feeds real `results.tsv` rows (when available) through the embedder and coordinator, to stress the embedding-quality assumption.

### 4.8 Risks

- **Toyland risk.** A synthetic landscape can be designed to make any coordination scheme look good. Mitigation: the landscape is described in `synthetic.py` upfront, with K' (true minima count) varied independently of K (groups), and AC-B3 is measured against *embedding-space* separation that isn't directly aware of the landscape.
- **Embedding quality.** Synthetic hypothesis strings may be too templated for MiniLM to embed informatively. Mitigation: the Day-5 replay-mode pass over real `results.tsv` summaries is the honesty check; if MiniLM can't separate the real categories there, we report that as a finding, not paper over it.
- **`coupling_T` non-monotonicity (AC-B4 fails).** Most likely failure mode. If it happens, the POC's conclusion is "SCT framing does not give a single principled knob here" — which is itself a useful, falsifying result.

### 4.9 What success of this POC unlocks (and what it doesn't)

- **Unlocks:** a falsified or validated *coordination layer* design, ready to wire into live agents in a follow-on phase with all the design questions already answered.
- **Does NOT unlock:** any claim about real-world swarm performance, embedding-quality robustness at scale, or the inner-agent search. Those need the live phase that's explicitly out of scope here.

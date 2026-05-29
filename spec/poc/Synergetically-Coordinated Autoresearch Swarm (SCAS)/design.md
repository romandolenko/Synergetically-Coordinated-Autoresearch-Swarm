# SCAS Simulator POC — Design

> Companion docs: [`requirements.md`](requirements.md) (what/why), [`tasklist.md`](tasklist.md) (execution).

## 1. What the simulator simulates

The simulator replaces autoresearch's *outer loop* — the LLM proposer and the GPU trainer — with synthetic stand-ins, and runs the **real** coordinator logic against them. Per step per agent:

1. **Synthetic proposer** samples a "hypothesis card" — a short string from a templated taxonomy ("change optimizer to X", "increase depth to Y", "swap activation Z"), biased by the agent's group attractor and coupling weight α_i.
2. **Synthetic evaluator** computes a fake val_bpb from a fixed *ground-truth landscape* — a mixture of Gaussians in embedding space with K' true local minima.
3. **Coordinator** receives `(card, val_bpb)`, updates group attractors, recomputes ψ_i and α_i for the next step.

The ground-truth landscape is fixed per seed and known to the analyzer, so "distinct local minima visited" (AC-B2) is measurable against ground truth, not just self-reported clusters.

## 2. Architecture

```
+--------------------------+
|  sim.py (entry point)    |
|  - seeds, K, K', steps   |
|  - sweeps                |
+------------+-------------+
             |
             v
+--------------------------+        +-----------------------+
|  scas/synthetic.py       | -----> | scas/coordinator.py   |
|  - proposer (templated)  | <----- |  - groups, attractors |
|  - evaluator (landscape) |  cards |  - psi, alpha         |
|  - ground-truth minima   |  +     |  - validator          |
+--------------------------+  val   +-----------+-----------+
             ^                                  |
             |                                  v
             |              +-------------------+---------------+
             |              |  sim_runs/<run_id>/*.jsonl        |
             |              |  (cards, results, state, sweeps)  |
             |              +-------------------+---------------+
             |                                  |
             |                                  v
             |              +-------------------+---------------+
             +------------- |  scas/plot.py  (AC-B2/B3/B4)      |
                            +-----------------------------------+

+--------------------------+
|  scas/baselines.py       |
|  - independent-walkers   |
|    (alpha_i = 0, no g)   |
+--------------------------+

+--------------------------+
|  scas/validator_tests.py |
|  - 20 adversarial cases  |
|  - AC-B5' gate           |
+--------------------------+
```

## 3. Files

```
scas/
  __init__.py
  coordinator.py        # control law, group/attractor mgmt, validator
  embedder.py           # all-MiniLM-L6-v2 wrapper, on-disk cache
  synthetic.py          # proposer + evaluator + ground-truth landscape
  baselines.py          # independent-walkers baseline
  sim.py                # entry point: parses args, runs sim, writes JSONL
  plot.py               # AC-B2/B3/B4 figures from JSONL
  validator_tests.py    # AC-B5' adversarial cases (pytest)
  replay.py             # Day-5: read results.tsv, drive coordinator
sim_runs/               # JSONL traces, one subdir per run_id (gitignored)
```

**No edits to `train.py`, `prepare.py`, `pyproject.toml`** except adding `sentence-transformers`, `scikit-learn`, `matplotlib`, and `pytest` as deps for the simulator.

## 4. Data structures

```python
@dataclass(frozen=True)
class HypothesisCard:
    agent_id: str
    step: int
    summary: str               # short NL string
    category: str              # taxonomy slot, e.g. "optimizer"
    parent_step: int | None

@dataclass(frozen=True)
class StepResult:
    card: HypothesisCard
    val_bpb: float
    embedding: np.ndarray      # 384-d from MiniLM
    accepted: bool             # passed safety envelope
    reject_reason: str | None

@dataclass
class CoordinatorState:
    groups: dict[int, list[str]]            # group_id -> [agent_ids]
    attractors: dict[int, np.ndarray]       # group_id -> embedding of group-best card
    group_best_val_bpb: dict[int, float]
    psi: dict[str, float]                   # per agent
    alpha: dict[str, float]                 # per agent, in [0,1]
```

## 5. The control law

For agent `i` in group `g(i)` at step `t`:

```
psi_i   = w_d * dist(z_i, c_g)  +  w_p * max(0, val_bpb_i - val_bpb_g_best)
alpha_i = sigmoid((psi_i - psi_0) / coupling_T)
```

Defaults (post-Day-3 tuning, see `results-day3.md` §13):
- `w_d = 1.0`
- `w_p = 0.0` — original `50.0` saturated alpha after a single below-best step and pinned agents. Disabling the perf-gap term keeps psi tied to embedding distance only.
- `psi_0 = 2.0` — original `0.5` left alpha near 0.5 most of the time, which with the original "always neighbor" proposer trapped agents at local minima. Higher `psi_0` biases alpha low so the mix-proposer's global branch fires often.
- `coupling_T ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` swept for AC-B4.

The synergetic dynamics `T·ψ̇ + ψ = 0` is *not* numerically integrated; the formula above is the **steady-state interpretation**. The coupling weight α_i drives the next proposer call:

- `alpha_i > 0.5` → proposer is given `target = attractors[g(i)]` (tighten toward group leader)
- `alpha_i ≤ 0.5` → proposer is given `avoid = [attractors[g'] for g' != g(i)]` (push away from other groups)

This is the **entire** SCT contribution. Everything else is bookkeeping.

## 6. Synthetic proposer

Templated string generation, parameterized by `(category, target?, avoid?, alpha?)`:

- Each agent's category is fixed at sim start by group assignment (e.g. group 0 = "optimizer", group 1 = "architecture", group 2 = "regularization").
- `propose()` makes a per-call choice between a *local* and a *global* branch:
  - **Local** — one-token mutation of `target`'s slot (returns `slots[i±1]` of the matched slot).
  - **Global** — uniform random slot from the category, filtered to exclude any slot named in `avoid`.
- The mix is controlled by `alpha`: `bernoulli(alpha)` selects local; otherwise global. So `alpha=1.0` is pure-local (legacy), `alpha=0.0` is pure-global, and `alpha=0.5` is 50/50. This replaces the original always-local-when-target-given semantics, which trapped agents at the first attractor they found (results-day3.md §11–13).
- Output is plain text, the input to `embedder.embed(text)`.
- Taxonomy was expanded ~10× from the original 33 strings via slot cross-products (e.g. `Adam lr=1e-4 mom=0.9`) so baseline doesn't trivially saturate within the agent-step budget.

This bakes in a known difficulty floor — if SCT can't separate three deliberately-disjoint categories under templated text, it won't separate anything.

## 7. Synthetic evaluator and ground-truth landscape

```python
def evaluate(embedding: np.ndarray, landscape: Landscape, rng) -> float:
    # mixture of K' Gaussians in 384-d embedding space
    # val_bpb = base - sum_k(depth_k * exp(-||z - mu_k||^2 / sigma_k^2)) + noise
    ...
```

- `K'` is independent of `K` (number of groups). `K' > K` stresses coverage; `K' < K` stresses group separation under crowding.
- Landscape parameters are seed-derived and dumped to the run's JSONL so analyzer code can compute "distinct local minima visited" against ground truth.
- Noise stddev ~ 1% of `base` (matches the AC-B2 cluster radius).

## 8. Safety-envelope validator

`coordinator.validate(commit: dict) -> (ok: bool, reason: str|None)` checks:

- only `train.py` is listed in `commit["files_changed"]`
- `commit["pyproject_hash"] == BASELINE_PYPROJECT_HASH`
- `commit["prepare_hash"] == BASELINE_PREPARE_HASH`
- `commit["val_bpb"] >= MIN_PLAUSIBLE_VAL_BPB` (rejects implausibly-low claims — stand-in for re-eval, which doesn't exist in simulator)
- agent's last 5 commits include ≤ 2 failures (crash-budget gate)

`validator_tests.py` instantiates 20 adversarial commits covering each rule and asserts 100% rejection.

## 9. Baseline controls (head-to-head for AC-B1' / AC-B2 / AC-B3)

**Independent-walkers baseline** (`NullCoordinator`). Identical proposer/evaluator/landscape. Coordinator replaced by a no-op: every agent gets `alpha_i = 0`, **no group assignment** (each step draws a random category), no attractor updates. Matched K, step budget, seeds.

**Partitioned fair control** (`PartitionedNullCoordinator`, added in review). Same *fixed* category→group partition as the real coordinator and tracks group-best attractors so AC-B3 is measurable — but `alpha_i ≡ 0` and the proposer is never steered (no `target`/`avoid`). The **only** difference from SCAS is the control law. This isolates what the synergetic coupling adds *on top of* the partition+prefix structure. It is the correct control for the headline claims, because the random-category baseline conflates the forced-partition effect with the control-law effect:

- AC-B2 marginal control-law speedup = `partition_steps_to_80% / scas_steps_to_80%`.
- AC-B3 control test = does inter/intra separation survive with coupling off? (It mostly does — ~1.89 vs SCAS ~2.09 — so AC-B3 is largely structural. See [`results-review.md`](results-review.md).)

`sim.py --paired-seeds` runs all three arms by default (`--no-partition-control` to skip).

## 10. JSONL trace schema

One run produces a directory `sim_runs/<run_id>/`:

- `config.json` — all args, defaults, landscape seed, embedder version
- `events.jsonl` — one row per (agent, step): `{step, agent_id, card, val_bpb, accepted, reject_reason, psi, alpha, group_id}`
- `attractors.jsonl` — one row per attractor update: `{step, group_id, new_attractor_card_id, group_best_val_bpb}`
- `landscape.json` — ground-truth minima (`mu_k`, `sigma_k`, `depth_k`) for downstream "distinct minima visited" analysis
- `summary.json` — terminal metrics: `best_val_bpb`, `inter_group_dist`, `intra_group_dist`, `distinct_minima_visited`, `n_violations`, **`first_visit_steps`** (sorted list of agent-step indices at which each new minimum was first visited; length ≤ K'), **`steps_to_coverage`** (`{"0.5": int|null, "0.8": int|null, "1.0": int|null}` — agent-step at which each coverage fraction was reached; null if never reached within the budget)

A sweep run produces N sub-runs plus a top-level `sweep.json` linking them.

## 11. Plotting (AC-E3)

`plot.py <run_id>` regenerates:

- **AC-B3 figure** — bar chart of inter-group vs intra-group distance per seed, line at 1.5× reference
- **AC-B2 figure** — distinct minima visited (SCAS vs baseline) over seeds, line at 1.5× reference
- **AC-B4 figure** — inter-group separation as a function of `coupling_T` across the sweep

All inputs are JSONL; no rerun required.

## 12. Configuration surface

A single `coupling_T` knob is the only user-facing dial. Everything else (`w_d`, `w_p`, `psi_0`, K, K', steps, seeds) is a CLI flag with documented defaults. Defaults are chosen so `uv run sim.py` with no args runs a single seed of the headline experiment in under 60 s on a laptop.

## 13. What this design deliberately does *not* include

- No HTTP service. Coordinator is an in-process Python object. Live-agent Phase 2 (out of scope) would wrap it in FastAPI; the simulator doesn't need to.
- No persistence layer beyond JSONL. No DB, no migrations.
- No re-evaluation. The simulator's `evaluate()` is the only oracle.
- No prompt augmentation logic that targets a real LLM. The proposer is the simulator-only stand-in; live-agent prompt packets are not built here.

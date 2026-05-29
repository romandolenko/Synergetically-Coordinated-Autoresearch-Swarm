# SCAS — Synergetically-Coordinated Autoresearch Swarm
## POC presentation

> A POC for applying Synergetic Control Theory (SCT) to a massively-parallel autoresearch swarm. Validated offline; simulator only; no GPUs, no LLM agents.

---

## Part 1 — What autoresearch does today

Autoresearch (Karpathy, March 2026; this repo is the Windows-RTX fork) is a small "let the AI run experiments overnight" system. Its setup is deliberately minimal:

- **One GPU, one agent, one branch, one metric.**
- The agent edits a single file (`train.py`) — the model, the optimizer, the training loop.
- Each experiment trains for **5 minutes wall clock** (the budget is fixed so all experiments compare apples-to-apples regardless of hardware).
- The metric is **val_bpb** (validation bits per byte). Lower is better.
- After each run the agent decides: keep the change (advance the branch) or revert (`git reset`).
- Run this loop overnight ⇒ ~100 experiments per night ⇒ wake up to a log.

The **goal** is autonomous, end-to-end research iteration in a constrained but real LLM-pretraining setup. The agent is whatever LLM you wire up (Claude, Codex, etc.); the human only edits `program.md`, the prompt that frames the agent's job.

### What works
- Reproducible, time-budgeted experimental loop.
- Self-contained: no external infra, no distributed training.
- Guardrails — the agent can't touch `prepare.py` (data, evaluation harness) or `pyproject.toml` (dependencies). The evaluation function is sacred.

### What doesn't
- **One agent at a time.** The loop is strictly serial. A second GPU buys nothing unless you spin up an independent agent on an independent branch — and those agents don't talk to each other.
- **Mode collapse.** A single agent that hits a local minimum keeps refining around it. There's no mechanism for "try a totally different family of changes" unless the LLM happens to volunteer the leap.
- **No collective memory.** Two parallel agents that independently discover the same dead-end won't know it; two that find complementary improvements can't merge them automatically.

These limitations are exactly the gap "massively collaborative autoresearch" would close — and the question the POC asks is *how* to close them principled-ly rather than hacking a hub-and-spoke service together.

---

## Part 2 — Why Synergetic Control Theory (SCT)

The naïve fix is "spin up K agents, give each a different prompt, gather results." It works but produces an undifferentiated swarm: no separation guarantees, no recovery from one bad agent, no principled coordination knob.

SCT — specifically its **multi-consensus** and **invariant-manifold** subfields — gives a principled framing for the coordination layer between agents (not for any single agent's internal search, which stays model-free because there is no analytical model of "edit → loss").

### SCT principles that map onto autoresearch

1. **Distributed, model-free coordination.** SCT works with local interactions over a communication graph between agents. No central orchestrator needed.

2. **Multi-consensus, not single-consensus.** Plain consensus drives all agents to one value — useless here, that's mode collapse. Multi-consensus drives *sub-groups* to *distinct* attractors. Each group covers a different region of hypothesis space.

3. **Macro-variable + smooth dynamics.** SCT defines a per-agent scalar `ψ` (psi) on the agent's coordination error, then imposes `T·ψ̇ + ψ = 0` — exponential decay toward the attractor manifold. The time constant `T` is a single coordinated-vs-loose knob.

4. **Invariant manifold = safety envelope.** Define the "honest search" manifold by the existing guardrails (eval is locked, `prepare.py` immutable, `pyproject.toml` frozen, 5-min budget honored). The control objective is to keep the entire swarm on that manifold even as it explores. One reward-hacking agent can't pull the collective off.

5. **Leader-follower for promotion.** A validated, low-val_bpb config becomes a "leader" attractor that other agents in its group track. Confirms Karpathy's existing "promote-on-improvement" step but generalized to groups.

### Where SCT cannot help

The inner agent — "given this code, what edit will lower val_bpb?" — has no analytical model. SCT shapes coordination, not cognition. The LLM does the proposing; the controller does the coordinating. Clean division of labor.

### The honest soft point

SCT assumes the agent state lives in a **continuous metric space**. Autoresearch hypotheses are discrete semantic objects (code diffs, hyperparameter configs). To make the math work, we embed each agent's current hypothesis into a vector space where proximity tracks semantic similarity. This embedding is lossy. Every downstream guarantee inherits its imperfection — you get *soft* guarantees about a *model* of the swarm, not hard guarantees about the swarm.

---

## Part 3 — The SCAS simulator, term by term

The SCAS simulator is a self-contained Python program that runs the SCAS coordination law against a synthetic stand-in for autoresearch's outer loop. No GPUs, no LLM agents, runs on a laptop in seconds.

Below is every term that appears in the simulator, in the order you'd meet them following a single agent through one step.

### 3.1 Agents, groups, categories

- **K agents** (default `K=3`). Each agent is a process in real autoresearch; in the simulator it's a row in a loop.
- **K groups** (one per agent in the default). Each group is assigned one **category** of changes: `optimizer`, `architecture`, `regularization`. The category determines which slice of the search space the agent's proposer draws from.
- **`agent_to_group[agent_id]`**: which group each agent belongs to.
- **`group_to_category[group_id]`**: which category each group covers.

The category partition is the "distributed control" anchor: it's what gives multi-consensus its starting separation. Without it, all groups would drift toward the same attractor.

### 3.2 The hypothesis card

Each step, an agent emits a **HypothesisCard** — a short natural-language string describing the proposed change. Example: `change optimizer to LAMB lr=1e-4 mom=0.9`. The card has an `agent_id`, `step`, `category`, and the `summary` text. In real autoresearch the card would be the agent's plain-English description of its intended `train.py` edit.

### 3.3 The embedder

`scas/embedder.py` wraps **MiniLM (`all-MiniLM-L6-v2`)**, a 384-dimensional sentence encoder. It takes a hypothesis card's text and produces a unit-norm vector `z ∈ ℝ^384`. Embeddings are SHA-cached on disk so repeated calls are free.

This vector is the *continuous metric space embedding* that everything downstream needs. The "soft point" warning applies here: if MiniLM doesn't distinguish two genuinely different hypotheses, the controller can't either. (T5.2 measures this directly.)

### 3.4 Group attractor

For each group `g`, the simulator maintains a single point `c_g` in embedding space — the **attractor**. By construction:

- `c_g` is the embedding of the **best-val_bpb hypothesis card** the group has produced so far.
- It updates only on an *accepted* improvement.

This is the "multi-consensus" target. Each group converges around its own `c_g`. With K=3 and three disjoint categories, you get three distinct convergence points.

### 3.5 The macro-variable ψ

For agent `i` in group `g(i)` with current embedding `z_i` and current val_bpb `v_i`:

```
ψ_i = w_d · ‖z_i − c_{g(i)}‖ + w_p · max(0, v_i − v*_{g(i)})
```

where:
- `‖z_i − c_g‖` — Euclidean distance from agent's embedding to its group's attractor.
- `v*_g` — the group's current-best val_bpb.
- `w_d`, `w_p` — weights mixing the **idea-space distance** and the **performance gap**.

Read it as: "how far off is this agent — both in idea space and in performance — from where its group is converging?" Large ψ ⇒ agent is "lost." Small ψ ⇒ agent is at the group's current sweet spot.

### 3.6 The coupling α (alpha)

ψ feeds a sigmoid:

```
α_i = σ((ψ_i − ψ_0) / T)        with α_i ∈ [0, 1]
```

where:
- `ψ_0` — the **bias** that determines where the sigmoid is centered. Sets the agent's "neutral" psi.
- `T = coupling_T` — the **time constant**, the single user-facing knob.

This is the synergetic dynamics `T·ψ̇ + ψ = 0` collapsed to its steady-state interpretation. α is the agent's **coupling weight** for the next step: how strongly to track the group leader vs. explore globally.

### 3.7 The mix proposer

The synthetic proposer takes `category`, `target` (the group attractor's card text), `avoid` (other groups' attractor texts), and `alpha`. Per call:

- With probability `alpha`: **local branch** — return one of `slots[i±1]` of the target slot. (Pure attractor-tracking.)
- With probability `1 − alpha`: **global branch** — return a random slot from the category, filtered to exclude any slot named in `avoid`. (Active exploration that stays out of other groups' regions.)

`alpha = 1` ⇒ pure neighbor (legacy). `alpha = 0` ⇒ pure random-with-avoid. `alpha = 0.5` ⇒ 50/50.

The mix replaced an earlier "binary fork" semantic (`if alpha > 0.5 then target else avoid`) which trapped agents at the first attractor they found — see results-day3.md §11–13 for the bug and the fix.

### 3.8 The synthetic landscape

The simulator needs a stand-in for the val_bpb function. `synthetic.Landscape` defines a **mixture of K' Gaussians** in 384-d embedding space:

```
val_bpb(z) = base − Σ_k depth_k · exp(−‖z − μ_k‖² / σ_k²) + 𝒩(0, noise_std²)
```

- `μ_k` — the **K' true minima centers**. Placed at actual proposer-output embeddings so the proposer *can* reach them.
- `depth_k` — how deep each minimum is (typically `0.04–0.08`; baseline `1.0`).
- `σ_k` — the width of each Gaussian.
- `noise_std = 0.005` — per-call gaussian noise (matches the AC cluster radius).

The landscape is **seed-deterministic**: same seed ⇒ same `μ_k`, `depth_k`. Stored in `landscape.json` so analyzers can score "distinct minima visited" against ground truth.

`K'` (number of true minima) is independent of `K` (number of groups). `K' > K` stresses coverage; `K' < K` stresses group separation under crowding.

### 3.9 The safety-envelope validator

Before any commit is accepted into the coordinator, `validate(commit, recent_failures)` checks:

- `files_changed == ['train.py']` — matches `program.md`'s existing rule.
- `pyproject_hash == BASELINE_PYPROJECT_HASH` — no new deps.
- `prepare_hash == BASELINE_PREPARE_HASH` — eval harness untouched.
- `val_bpb ≥ MIN_PLAUSIBLE_VAL_BPB = 0.5` — rejects implausibly-low claims (stand-in for re-eval, which doesn't exist in the simulator).
- `recent_failures ≤ MAX_FAILURES_IN_LAST_5 = 2` — crash-budget gate.

Any failure ⇒ reject with a non-empty reason; agent has to revert. This is the **invariant-manifold safety envelope** in code.

### 3.10 The independent-walkers baseline

`scas/baselines.py` provides a **NullCoordinator** with the same interface as the real coordinator but `alpha = 0` for everyone, no group structure, no attractor updates. Each baseline agent picks a random category and a random slot every step. This is the "what would unguided parallel agents do?" comparison for AC-B1' / AC-B2.

### 3.11 Coverage metric (AC-B2)

For each ground-truth minimum `μ_k`, we record the **first agent-step at which any agent's embedding came within `0.5 · σ_k`** of `μ_k`. This gives a per-minimum first-visit step. From those, we compute:

- `steps_to_50pct_coverage` — agent-step at which ⌈0.5·K'⌉ minima have been visited.
- `steps_to_80pct_coverage` — agent-step at which ⌈0.8·K'⌉ minima have been visited.
- `steps_to_100pct_coverage` — agent-step at which all K' minima have been visited.

AC-B2 compares `mean(baseline_steps_to_80%) / mean(scas_steps_to_80%)` — the **coverage speedup**. ≥ 1.5 means SCAS reaches 80% coverage at least 1.5× faster than the baseline.

### 3.12 Group separation metric (AC-B3)

- **inter-group distance** — mean pairwise distance between group attractors.
- **intra-group distance** — mean distance from each agent embedding to its group attractor.

`inter / intra ≥ 1.5` means groups are clustered tightly around their attractors AND spaced far apart from each other in embedding space — i.e. **no mode collapse**.

### 3.13 Knob monotonicity (AC-B4) — Kendall's τ

We sweep `coupling_T ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` and at each T measure `inter/intra`. **Kendall's τ** is a rank-correlation statistic on the (T, separation) pairs:

- `τ = +1` — strictly monotonic, separation grows with T.
- `τ = −1` — strictly monotonic, separation shrinks with T.
- `τ = 0` — no monotonic relationship.

AC-B4 requires `|τ| ≥ 0.7` — the knob produces a predictable monotonic response. (The sign depends on `(ψ_0, typical_ψ)`; see §4.)

### 3.14 Silhouette score (T5.2 — embedding quality)

For the replay-mode check: take real-world hypothesis descriptions, label them with their "true" category via keyword heuristic, embed them with MiniLM, then compute **silhouette score** — a `[-1, 1]` measure of "how well-separated are the labeled clusters in this embedding space."

> 0.05 ⇒ embeddings have non-trivial structure aligned with the labels. ≤ 0 ⇒ structure is random or anti-aligned.

---

## Part 4 — What was measured, and what came out

The simulator was run end-to-end across five behavioral acceptance criteria (AC-B1' through AC-B5') and one embedding-quality check (T5.2). After three rounds of iteration on the AC specs and the simulator constants (see results-day3.md and results-day4.md for the full diary), the final results with committed defaults (`W_P=0`, `PSI_0=2.0`, mix-proposer, 330-string taxonomy):

| AC | Threshold | Best observed | Verdict |
| --- | --- | --- | --- |
| **AC-B1'** — best-val_bpb parity | within 1 SE | SCAS 0.8974 vs baseline 0.8996 (Δ = −0.0022) | **PASS** |
| **AC-B2** — coverage speedup | ratio ≥ 1.5 | **1.73 at T=0.1**, 1.69 at T=1.0, 1.54 at K'=10 | **PASS** at T ≤ 1.0 |
| **AC-B3** — no mode collapse | inter/intra ≥ 1.5 | **2.52 at T=10**, 1.89 at T=0.1 | **PASS** across the entire T-sweep |
| **AC-B4** — knob monotonicity | \|τ\| ≥ 0.7 | **τ = +1.00 aggregate**, +0.77 per-seed flat | **PASS** |
| **AC-B5'** — safety envelope | 20/20 adversarial rejected | 20 passed in 0.20 s | **PASS** |
| **T5.2** — embedding quality | silhouette > 0.05 | **0.029 over 30 real-style rows** | **FAIL (informative)** |

### Headline reading

- **The SCAS coordination layer works in the synthetic regime.** Multi-consensus produces clean group separation; the mix-proposer + tuned constants give a real 1.5–1.7× coverage speedup over independent walkers; the coupling_T knob is monotonic.
- **The Pareto frontier of "joint AC-B2 + AC-B3 pass" lives at `T ≤ 1.0`.** Tight T → fast coverage. Loose T → strong separation. They pull in opposite directions; T = 0.1 to 1.0 is the joint sweet spot.
- **The safety envelope is bookkeeping but does its job.** No surprises.
- **The embedding-quality finding is real and live-phase-relevant.** Synthetic descriptions carry explicit category prefixes (`change optimizer to …`, `tune architecture: …`) that MiniLM uses as anchors. Real `results.tsv`-style descriptions don't — and the architecture category specifically collapses to silhouette ≈ 0. This is the design's pre-flagged soft point and it came true.

### What this validates and what it doesn't

**Validates** (in a small synthetic regime): the SCAS coordination layer produces diversity-with-selective-convergence behavior at the small-N scale. The single `coupling_T` knob is meaningful and monotonic. The safety envelope rejects all adversarial commits in the test set.

**Does not validate**: that the result transfers to live LLM agents. That the committed constants generalize to a richer search space. That T5.2's recommended mitigation (a `[CATEGORY]` prompt prefix) actually works on real autoresearch descriptions.

---

## Part 5 — Next steps to prove the idea

The POC's job was to falsify or validate the *coordination layer* design cheaply, offline. With validation in hand, the next steps move toward a live-agent Phase 2:

### Immediate (~hours, no GPUs)

1. **Validate the embedding-quality mitigation.** Modify `program.md` so the agent prefixes its hypothesis description with a category tag (`[OPTIMIZER]`, `[ARCH]`, `[REG]`). Re-run T5.2 against a fresh 30-row sample. Pass silhouette > 0.05 ⇒ proceed; fail ⇒ try a richer embedder (sentence-T5 large, or one fine-tuned on research descriptions) before committing to the prefix approach.

2. **Stress the small-N assumption.** Sweep K ∈ {3, 5, 8} in the simulator to confirm AC-B3 separation holds with more groups than categories (multi-consensus claim under crowding).

### Short-term Phase 2 (~days, one GPU box)

3. **Build the coordinator as an HTTP service.** Extract `scas.coordinator.Coordinator` behind a small FastAPI surface: `POST /step` (agent reports a commit) returns `{alpha, target_card_text, avoid_list, accepted, reject_reason}`.

4. **Wrap the autoresearch agent loop.** Modify the existing loop in `program.md` so each iteration:
   - Calls the coordinator pre-step to get its prompt-augmentation packet.
   - Runs the existing train.py loop.
   - Calls the coordinator post-step to report (commit, val_bpb).
   - Acts on the validator's accept/reject decision (revert if rejected).

5. **Live K=3 paired run vs. K=3 independent baseline.** Same total wall-clock; measure best-val_bpb, coverage of "distinct hypothesis categories visited" (using the human-readable cards, not embeddings, as ground truth), and rate of validator rejections. The live-phase analog of AC-B1' / AC-B2 / AC-B3.

### Medium-term (~weeks, multi-GPU or multi-host)

6. **Scale to K=8–15 across multiple boxes.** This is where multi-consensus crowding (K' < K) becomes a real test. Validate AC-B3 holds under switching topologies (agents joining and leaving asynchronously).

7. **Adversarial-agent harness.** Deliberately deploy one agent prompted to reward-hack. Verify the safety envelope quarantines it without contaminating the rest. This is the live-phase analog of AC-B5'.

8. **Replace the keyword-heuristic category router** with a learned classifier (or LLM router) once enough live runs exist to label categories reliably.

### Out-of-scope but worth flagging

- **Inner-agent improvement** — the LLM's proposer quality. SCAS shapes coordination, not cognition. Improving the proposer is a separate axis (better prompts, model upgrades, RLHF on autoresearch transcripts) and is independent of everything above.
- **Massively-collaborative deployment** — 100+ agents across a real cluster. SCT's distributed-control results scale, but the simulator validates small-N only. A 100-agent claim needs its own validation pass once the K=15 multi-host test holds.

---

## Appendix — the recipe that landed (commit defaults)

```
# Coordination law (scas/coordinator.py)
W_D                  = 1.0     # weight on idea-space distance in psi
W_P                  = 0.0     # weight on performance gap (originally 50.0)
PSI_0                = 2.0     # sigmoid bias (originally 0.5)
DEFAULT_COUPLING_T   = 1.0     # the user-facing knob

# Proposer (scas/synthetic.py)
TAXONOMY: 132 optimizer + 99 architecture + 99 regularization = 330 strings
propose(category, target, avoid, alpha) — local/global mix by bernoulli(alpha)

# Validator (scas/coordinator.py)
MIN_PLAUSIBLE_VAL_BPB = 0.5
MAX_FAILURES_IN_LAST_5 = 2
```

Reproduce in 36 seconds on a laptop:

```powershell
uv sync
uv run pytest scas/                                     # AC-B5' — 20 passed
uv run python -m scas.sim --all-experiments             # AC-B1'/B2/B3/B4
uv run python -m scas.replay                            # T5.2 silhouette
uv run python -m scas.plot sim_runs/<latest_run_id>     # auto-detect plots
```

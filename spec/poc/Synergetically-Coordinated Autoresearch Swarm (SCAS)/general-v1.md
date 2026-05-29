# POC: Synergetically-Coordinated Autoresearch Swarm (SCAS) — v1 (general plan)

> **Status:** Superseded by [`simulator-only.md`](simulator-only.md) for actual execution. Kept here as the broader design from which the simulator-only POC was carved out — it documents the live multi-agent vision and the scoping choices that led to v2.

## 0. Premise and scope

Today autoresearch is **one agent, one GPU, one branch, one metric (val_bpb)**, looping every ~5 min. The earlier investigation concluded SCT does not apply to the *inner search* (no analytical model of "edit → loss"), but it *does* apply to the **coordination layer between agents** via multi-consensus + invariant-manifold framings. This POC tests exactly that claim — narrowly — without changing the inner agent at all.

**Non-goals for the POC.** No model of training dynamics. No distributed training. No modification of `prepare.py` or the evaluation harness. No bespoke LLM finetuning. The 5-minute budget and the single-GPU constraint stay sacred.

**The honest engineering reality.** "Massively collaborative" is the *aspiration* the design must not foreclose. The POC itself is small: **3–5 agents**, run on either one machine sequentially (time-sliced GPU) or on 2–3 machines if available. The point is to prove the *coordination law* earns its keep at small N; scaling-out is a follow-on.

## 1. User stories

**US-1 — Researcher running an overnight swarm**
*As a* user who would normally launch one autoresearch agent before bed,
*I want* to launch a coordinated swarm of K agents,
*so that* the morning's best val_bpb is at least as good as K independent agents at the same total compute, AND the swarm has explored a measurably broader region of hypothesis space (no mode collapse).

**US-2 — Researcher tuning the coordination strength**
*As a* user,
*I want* to tune a single knob (`coupling_T`) that interpolates between "K independent random walkers" and "K agents tightly tracking the current leader",
*so that* I can trade exploration for exploitation without re-coding.

**US-3 — Researcher trusting the swarm**
*As a* user,
*I want* the coordinator to **reject** any agent commit that violates the existing guardrails (modifies `prepare.py`, edits `pyproject.toml`, exceeds the time budget, hits OOM repeatedly, or claims an improvement the re-eval cannot reproduce),
*so that* one reward-hacking or buggy agent cannot contaminate the collective record.

**US-4 — Researcher inspecting what the swarm did**
*As a* user,
*I want* a single dashboard view of the swarm — per-agent hypothesis (in plain English), per-group attractor, current macro-variable ψ_i, coupling weight α_i, and group-best val_bpb —
*so that* I can audit *why* an agent moved where it did and rewind individual agents if needed.

**US-5 — Researcher validating the control law cheaply**
*As a* developer of SCAS,
*I want* an **offline simulator** that replays past `results.tsv` rows (or synthetic ones) into the coordinator,
*so that* I can validate the control law's stability and group-separation behavior without burning GPU hours.

## 2. Acceptance criteria

**Functional**
- AC-F1 — K agents (K=3 for headline POC, parameterizable to 5) each run on dedicated branches `autoresearch/<tag>-a{0..K-1}`, with the standard inner loop from `program.md` unchanged.
- AC-F2 — A **coordinator service** holds: (a) per-agent state, (b) group assignments, (c) group attractors in embedding space, (d) the rolling best val_bpb per group and globally, (e) the safety-envelope rules.
- AC-F3 — Before each agent's next iteration, the coordinator returns a **prompt augmentation packet**: assigned group, coupling weight α_i ∈ [0,1], one "group leader" recipe to track if α_i is high, K-1 "things other groups already tried" to *avoid* if α_i is low.
- AC-F4 — Coordinator rejects any commit that touches files outside `train.py` or fails re-eval-on-coordinator-host (when feasible), and forces the agent to revert.
- AC-F5 — Offline simulator runs the full control law against a recorded or synthetic `results.tsv` stream without invoking the LLM agents.

**Behavioral (the falsifiable claims)**
- AC-B1 — **No-worse-than-independent on best.** With matched total compute, SCAS's best val_bpb is ≤ the median best-val_bpb of K=3 independent baseline agents over ≥5 paired runs.
- AC-B2 — **Better coverage.** SCAS visits ≥1.5× as many *distinct local minima* (clusters in hypothesis-embedding space with val_bpb within 1% of a local best) as the independent baseline, measured at end of the run.
- AC-B3 — **No mode collapse.** At end of run, the pairwise mean embedding distance between groups is ≥ 1.5× the mean within-group distance.
- AC-B4 — **Knob monotonicity.** As `coupling_T` decreases from "loose" to "tight", coverage decreases monotonically and best-val_bpb improvement rate increases monotonically (on the simulator, smoothed over ≥10 seeds).
- AC-B5 — **Safety envelope holds.** In an adversarial test where one agent is prompted to "improve val_bpb by any means including hacking the eval", the coordinator rejects 100% of out-of-manifold commits.

**Non-functional**
- AC-N1 — Coordinator overhead < 5 s per agent iteration (negligible vs the 5-minute training budget).
- AC-N2 — POC runs on the existing single-GPU dev box; live multi-agent uses GPU time-slicing if only one GPU is available.
- AC-N3 — All swarm state is human-readable JSONL files in `~/.cache/autoresearch/swarm/` — no DB.

## 3. Implementation design

### 3.1 Architecture

```
+----------------------+        +------------------------+
|  Agent process a_i   | <----> |   Coordinator (HTTP)   |
|  (Claude/Codex CLI)  |        |   - graph, attractors  |
|  - reads program.md  |        |   - macro-variable ψ_i |
|  - edits train.py    |        |   - coupling α_i       |
|  - runs uv run train |        |   - safety envelope    |
|  - reports result    |        |   - JSONL state store  |
+----------------------+        +------------------------+
         ^   ^
         |   |   (one process per agent; GPU time-sliced
         |   |    on single-GPU host, or 1-per-host on
         |   |    multi-host setup)
```

### 3.2 The five concrete pieces

**(1) Hypothesis card and embedding.** Each agent, before committing, writes a structured "hypothesis card" to the coordinator:
```json
{ "agent": "a2", "parent_commit": "...", "summary": "<=2 sentences",
  "diff_stats": {"files": ["train.py"], "+lines": 14, "-lines": 3},
  "category_hints": ["optimizer", "lr_schedule"] }
```
Coordinator embeds the `summary + category_hints` with a small local sentence-transformer (e.g. `all-MiniLM-L6-v2` — already-pip-installable, no GPU required). This is the **continuous embedding the prior investigation flagged as the soft point** — we accept the imperfection explicitly and log raw cards alongside embeddings so any downstream finding can be re-checked against the human-readable text.

**(2) Groups and attractors (multi-consensus).** At swarm start, the coordinator generates K seed hypotheses (one per group) by prompting the LLM with K disjoint *families* (e.g. "optimizer changes", "architectural width/depth", "regularization/data") from a short hand-written taxonomy. Each agent is assigned to one group g(i). Each group's attractor c_g is **the embedding of that group's current best-val_bpb hypothesis card** — updated whenever a within-group improvement is accepted.

**(3) Macro-variable and control law.** For each agent i:

```
psi_i = w_d * dist(z_i, c_{g(i)}) + w_p * max(0, val_bpb_i - val_bpb_{g(i)}_best)
```

The synergetic dynamics `T·ψ̇ + ψ = 0` is *not* numerically integrated; instead we use its **steady-state interpretation**: large ψ_i ⇒ the agent is far from its group's attractor (in idea-space *or* in performance), so the next step's **coupling weight** is

```
alpha_i = sigmoid((psi_i - psi_0) / T)   in [0,1]
```

with `T = coupling_T` the single user-facing knob. **High α_i → prompt the agent to track the group leader's recipe. Low α_i → prompt the agent to push further from the group attractor (and away from other groups' cards).** This is the entire control surface.

**(4) Invariant manifold = safety envelope.** Encoded as a `validate(commit)` function on the coordinator:
- `train.py` is the only file changed (matches `program.md`'s existing rule)
- `pyproject.toml` and `prepare.py` are byte-identical to baseline
- claimed val_bpb is **re-evaluated** on the coordinator's smoke-test (or accepted with a `provisional` flag if re-eval host unavailable)
- agent has not exceeded a per-window crash budget (e.g. ≥3 OOMs in last 5 commits → quarantine)
- VRAM stayed within the agent's tier floor

Any failure → coordinator instructs the agent to `git reset` and logs a `manifold_violation` row.

**(5) Offline simulator (Track A — build first).** A Python script that takes either real `results.tsv` rows or a synthetic generator (val_bpb sampled from per-group Gaussians with a shared global trend) and drives the coordinator without any LLM/GPU calls. This is how AC-B4 (knob monotonicity) and the stability sanity check are validated *before* a single agent is launched live. **This is the cheapest way to falsify the entire control law early.**

### 3.3 Two-phase delivery

**Phase 1 — Simulator + coordinator (no live agents).** ~2–3 days of work.
Deliverables: coordinator HTTP service, embedding pipeline, control law, safety-envelope validator, offline simulator with synthetic + replay modes, plots for AC-B3/B4. **Exit criterion: AC-B4 holds on the simulator.**

**Phase 2 — Live K=3 swarm.** ~2–3 days of work on top of Phase 1.
Deliverables: agent-side wrapper that calls the coordinator before/after each loop iteration and injects the prompt-augmentation packet into the next-iteration prompt; minimal "swarm dashboard" (one HTML page that polls coordinator state); paired runs vs independent baseline. **Exit criterion: AC-B1, AC-B2, AC-B5 hold across ≥5 paired runs.**

### 3.4 What the POC will *not* prove (stated up front)

- It will not prove the control law scales to 100+ agents — only that the structure permits it and the small-N behavior matches theory. AC-B4's monotonicity on the simulator is the closest we get to a scaling signal.
- It will not give "hard guarantees" — only directional ones, because the embedding is lossy (as flagged in the earlier investigation).
- It will not improve the *inner agent*. Any val_bpb gain comes from coordination, not from a smarter proposer.

### 3.5 Risks and open questions

- **Embedding quality.** If MiniLM embeddings don't distinguish "swap Adam→Lion" from "swap Adam→AdamW with cosine schedule", group attractors collapse and AC-B3 fails. Mitigation: simulator validates this on real `results.tsv` summaries before live phase.
- **Re-eval cost.** Re-evaluating each commit on the coordinator host is another 5 min/commit. For Phase 2 the cheap path is `provisional` accept + periodic audit re-eval, not full re-eval.
- **Single-GPU contention.** If one box runs all K agents, GPU time-slicing serializes them and we're not really parallel. Honest call to make: either accept serialization for the POC (still validates coordination logic) or require ≥2 GPUs.

## 4. Scoping decision that produced v2

Two choices were taken to carve `simulator-only.md` out of this general plan:

1. **Compute path:** simulator-only (skip the live multi-agent phase entirely). Cheapest, fastest, lowest-risk; falsifies the control law without touching real agents.
2. **Priority claim if scope is cut:** coverage / no mode collapse (AC-B2/B3). The most distinctive SCT-derived claim and the one the prior investigation flagged as load-bearing.

The live-agent Phase 2 design above (US-1, US-3, US-4, AC-F1–F4, AC-B1, AC-B5, AC-N1–N3) is preserved here as the follow-on roadmap once the simulator validates the coordination law.

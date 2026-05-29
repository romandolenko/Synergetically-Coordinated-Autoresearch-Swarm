# SCAS Simulator POC — Task List

> Companion docs: [`requirements.md`](requirements.md) (what/why), [`design.md`](design.md) (how).
> Each task lists the AC(s) and design section(s) it satisfies, and exit conditions you can mechanically check.

## Day 0 — Setup (≤ half a day)

- [ ] **T0.1** Create `scas/` subdir and `__init__.py`.
  - Refs: design §3.
  - Exit: `python -c "import scas"` succeeds.
- [ ] **T0.2** Add deps to `pyproject.toml`: `sentence-transformers`, `scikit-learn`, `matplotlib`, `pytest`, `numpy`. Run `uv sync`.
  - Refs: AC-E4.
  - Exit: `uv sync` clean; `uv run python -c "import sentence_transformers, sklearn, matplotlib, pytest, numpy"` succeeds.
- [ ] **T0.3** Add `sim_runs/` to `.gitignore`.
  - Exit: `git status` doesn't list `sim_runs/` after a run.

## Day 1 — Embedder, synthetic skeleton, validator (AC-B5')

- [ ] **T1.1** `scas/embedder.py`: wrap `all-MiniLM-L6-v2`. Cache embeddings to `~/.cache/scas/embeddings.sqlite` keyed by sha256(text).
  - Refs: design §4, AC-E1 (no network at runtime → pre-fetch the model in T0.2).
  - Exit: `embedder.embed("hello") == embedder.embed("hello")`, second call is cache hit.
- [ ] **T1.2** `scas/synthetic.py` (partial): templated taxonomy with 3 categories ("optimizer", "architecture", "regularization"), each with ≥10 slot values. `propose(category, target=None, avoid=None) -> str`. No evaluator yet.
  - Refs: design §6.
  - Exit: `propose("optimizer", target=None, avoid=None)` returns distinct strings on repeated calls; `propose("optimizer", target=<some_str>, avoid=None)` returns a string with high textual overlap with target.
- [ ] **T1.3** `scas/coordinator.py` skeleton: dataclasses from design §4, `CoordinatorState`, `validate(commit)` function from design §8. Constants `BASELINE_PYPROJECT_HASH`, `BASELINE_PREPARE_HASH` computed once from disk at import.
  - Refs: design §4, §8.
  - Exit: importing the module computes the two hashes without error.
- [ ] **T1.4** `scas/validator_tests.py`: 20 adversarial commit cases (≥4 per rule in design §8). Each asserts the validator rejects with a non-empty `reason`.
  - Refs: **AC-B5'**.
  - Exit: `uv run pytest scas/validator_tests.py -q` → 20 passed, 0 failed.

## Day 2 — Evaluator, control law, end-to-end single seed

- [ ] **T2.1** `scas/synthetic.py` (complete): `Landscape` dataclass with `K'` Gaussians in 384-d space; `evaluate(embedding, landscape, rng) -> float`. Seeded deterministically.
  - Refs: design §7.
  - Exit: `evaluate(z, L, rng)` is deterministic given a fixed rng seed; the minimum over 10k random `z` is within 5% of `base - max_k(depth_k)`.
- [ ] **T2.2** `scas/coordinator.py` (complete): implement `psi_i` / `alpha_i` per design §5. Group assignment at sim start (one category per group for K=3). Attractor updates on accepted improvement.
  - Refs: design §5, **US-2**.
  - Exit: unit test asserting that with high `psi_i` (large dist) and small `coupling_T`, `alpha_i` saturates near 1; with small `psi_i` and large T, near 0.5.
- [ ] **T2.3** `scas/sim.py`: CLI entry point. Args: `--seed`, `--K` (default 3), `--K-prime` (default 5), `--steps` (default 200), `--coupling-T` (default 1.0). Runs one configuration end-to-end, writes JSONL per design §10.
  - Refs: design §10, **AC-E1**, **AC-E2**.
  - Exit: `uv run scas/sim.py --seed 0` completes in < 60s on laptop and produces a `sim_runs/<run_id>/` dir with all four JSONL files described in design §10.

## Day 3 — Baseline, paired runs, headline AC measurement

- [ ] **T3.1** `scas/baselines.py`: independent-walkers (coordinator no-op variant). Same CLI surface as `sim.py --baseline`.
  - Refs: design §9.
  - Exit: `uv run scas/sim.py --baseline --seed 0` writes a JSONL run; all rows show `alpha=0, group_id=None`.
- [ ] **T3.2** `sim.py --paired-seeds N`: runs SCAS and baseline on seeds `0..N-1`, writes a `summary.json` per run with the AC metrics (per design §10).
  - Refs: **AC-B1'**, **AC-B2**, **AC-B3**.
  - Exit: `uv run scas/sim.py --paired-seeds 10` finishes and produces 20 sub-runs + 2 aggregate summaries.
- [ ] **T3.3** `scas/plot.py`: AC-B3 figure (inter vs intra distance per seed, 1.5× reference), AC-B2 figure (distinct minima visited: SCAS vs baseline). Both read JSONL.
  - Refs: **AC-E3**.
  - Exit: `uv run scas/plot.py <run_id>` produces two PNGs in the run dir.
- [ ] **T3.4** Measure: do AC-B2 and AC-B3 pass over 10 seeds with defaults?
  - Refs: **AC-B2**, **AC-B3** (headline).
  - Exit: if pass, document numbers in a `results.md` next to plots; if fail, iterate on `w_d` / `w_p` / `psi_0` (capped at ≤ 1 day of tuning before escalating as a falsification finding).

## Day 4 — Knob sweep (AC-B4)

- [ ] **T4.1** `sim.py --sweep-T "0.1,0.3,1.0,3.0,10.0" --paired-seeds 10`: runs the full T-sweep across seeds. Writes `sweep.json` linking the per-T runs.
  - Refs: **AC-B4**.
  - Exit: `sweep.json` exists with `n_T * n_seeds` entries.
- [ ] **T4.2** `plot.py` extension: AC-B4 figure — inter-group separation vs `coupling_T`, seed-averaged with error bars.
  - Refs: **AC-E3**, **AC-B4**.
  - Exit: PNG produced; computed Kendall's tau printed to stdout.
- [ ] **T4.3** Measure: Kendall's tau ≤ −0.7?
  - Refs: **AC-B4** (headline).
  - Exit: pass → document; fail → this is a valid POC outcome (requirements §6), document it as such instead of paving over.

## Day 5 (buffer) — Replay mode + writeup

- [ ] **T5.1** `scas/replay.py`: read a real `results.tsv` (or a hand-authored one), embed each row's `description` column, feed `(card, val_bpb)` pairs into the coordinator in commit order. Skip the proposer entirely.
  - Refs: requirements §6 (embedding-quality falsification case).
  - Exit: completes without error on at least one synthetic-but-realistic 30-row TSV; logs per-row `psi`, `alpha`, `group_id`.
- [ ] **T5.2** Quick check: do the embeddings cluster non-trivially? (silhouette score over assigned categories > 0.05).
  - Refs: requirements §6.
  - Exit: pass → embedding quality is acceptable; fail → document as a finding (embeddings are too coarse for real autoresearch summaries, blocking live phase until addressed).
- [ ] **T5.3** Write `results.md`: one paragraph per headline AC interpreting the figures, plus a short "what we learned about embedding quality" section from T5.2.
  - Refs: Definition of done #3.
  - Exit: file exists; figures linked; conclusions are stated, not hedged.

## Cross-cutting gates (verify before declaring done)

- [ ] **G1** `uv run pytest scas/` is green.
- [ ] **G2** `uv run scas/sim.py --all-experiments` reruns AC-B1' / AC-B2 / AC-B3 / AC-B4 from scratch on a clean checkout in under 30 min on a laptop.
- [ ] **G3** No edits to `train.py` / `prepare.py`. `pyproject.toml` only gained the simulator deps listed in T0.2.
  - Refs: **AC-E4**.
- [ ] **G4** A new contributor can run `uv sync && uv run scas/sim.py && uv run scas/plot.py <run_id>` and see the headline figures with no extra setup.
  - Refs: **US-3**.

## Stop conditions / when to escalate instead of grind

- If T3.4 fails after one day of tuning `w_d` / `w_p` / `psi_0`, **stop tuning**. The POC's conclusion is that the synthetic-landscape setup as designed does not exhibit AC-B2/B3 — report and discuss before any further engineering.
- If T1.1 burns more than half a day on `sentence-transformers` install/runtime issues on Windows, fall back to a smaller embedder (e.g. spaCy `en_core_web_md` 300-d vectors) and document the swap.
- If any task discovers a latent flaw in the design (e.g. attractor updates need to be smoothed; `psi_0` needs to be per-agent), **update `design.md` first** and link the change in the task before continuing.

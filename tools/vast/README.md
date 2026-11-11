# vast.ai harness for SCAS real-data runs

Rents a Linux GPU box, runs the real `program.md` experiment loop on real `train.py`,
brings `results.tsv` home, and feeds it through the SCAS analysis stack.

This closes the review's **S1** ("produce real `results.tsv` data before anything else")
and gives S2/S3 a working substrate.

---

## Prerequisites

```powershell
pip install vastai            # already present: v1.5.5
```

1. **Credits.** `vastctl keys` prints the account credit; top up at
   <https://cloud.vast.ai/billing/>. An overnight run is a few dollars (see costs below).
2. **API key** lives in `tools/vast/.env.vast` (gitignored, already written).
   Rotate it at <https://cloud.vast.ai/manage-keys/> — it was pasted into a chat
   transcript, so treat the current one as compromised once testing is done.
3. **SSH key** — `vastctl keys` generates `~/.ssh/id_ed25519_vast` and registers it.

---

## The loop

```powershell
python tools/vast/vastctl.py keys                  # one-time
python tools/vast/vastctl.py search                # what's on the market
python tools/vast/vastctl.py up                    # rent cheapest RTX 4090 <= $0.60/hr
python tools/vast/vastctl.py provision             # clone, uv sync, prepare.py, smoke test
python tools/vast/vastctl.py sweep --hours 6       # detached; 3 baseline repeats + 25 experiments
python tools/vast/vastctl.py logs -f               # watch
python tools/vast/vastctl.py status                # uptime + spend so far
python tools/vast/vastctl.py finish                # wait, fetch + verify, then DESTROY the box

# or do the last step by hand:
python tools/vast/vastctl.py fetch                 # -> vast_runs/<label>-<ts>/
python tools/vast/vastctl.py down                  # DESTROY (stop paying)

uv run python tools/vast/analyze.py vast_runs/<label>-<ts>/results.tsv
```

`down` is the one that stops the meter. `stop-sweep` only kills the experiment loop.

`finish` (or `sweep --auto-down`, which runs it straight after launching) polls until the
sweep process exits, downloads the results, and destroys the instance **only if** the
sweep log ends with its `=== done` line and the local copy checks out: `results.tsv` has
the same line count as on the box, `sweep_state.json` agrees with it, and `runs/` came
down. If any check fails, or the box stops answering, it leaves the instance running and
exits non-zero, so a crashed sweep can be resumed. `--keep` verifies without destroying.
It runs on your machine, so it must stay awake and online until the sweep ends; nothing
on the box can shut the instance down by itself.

### Cost

At the current market (`search` output, 2026-08-22): RTX 4090 from **$0.269/hr**.

| | |
| --- | --- |
| 26-experiment sweep (~8 min each) | ~3.5 h → **~$0.95** |
| provisioning (uv sync + 673 MB dataset + tokenizer + smoke) | ~0.3 h → ~$0.08 |
| 8 h overnight | ~$2.15 |

Storage bills separately (~$0.20–0.40 / GB-month, prorated); 60 GB for a few hours is cents.

---

## What the sweep actually does

`remote/sweep.py` runs program.md steps 2–9 verbatim, with a deterministic proposer
in place of the LLM:

```
git checkout -- train.py
apply edits from hypotheses.json      (module-level constants only)
uv run train.py > runs/<id>.log
grep val_bpb / peak_vram_mb
append a row to results.tsv           (canonical 5-column schema)
improved? git commit : git checkout -- train.py
```

Output:

- `results.tsv` — real val_bpb on real hardware, 5 columns, drops straight into `scas.replay`
- `runs/<id>.log` — full train.py output per experiment, tracebacks included
- `sweep_state.json` — per-experiment timing, MFU, exact edit, **ground-truth category**

It is resumable (`--resume`), honours a hard `--hours` budget (it will not start a run
it cannot finish), and refuses no-op edits so a mistyped constant can't silently record
a fake experiment. `--dry-run` validates the plan against the real `train.py` without
touching a GPU:

```powershell
python tools/vast/remote/sweep.py --plan tools/vast/remote/hypotheses.json --repo . --dry-run
```

### What it is not

The descriptions in `hypotheses.json` are **authored, not LLM-generated**. The
`val_bpb` numbers are fully real; the card *text* is not. So the silhouette in
analysis section 3 is an **upper bound** on what real agent cards would score — it
does not retire review concern C1, it calibrates it. Swapping the proposer for a real
agent is the natural next step and touches only the plan-loading half of `sweep.py`.

---

## Analysis

`analyze.py` writes `report.md` next to the tsv with four sections:

1. **Run health** — baseline, best, improvement, crash rate, min/experiment, MFU.
2. **Safety-envelope floor** — replays the live rule (a report may not claim less than
   50% of the best value already accepted) over the rows and lists any honest result it
   would reject. The old constant floor of 0.5 rejected 8 of 26 rows of the first real
   sweep (2026-11-11). **This gates S2.**
3. **T5.2 replay** — runs `scas.replay` on the real rows: raw MiniLM silhouette and
   the `[CATEGORY]`-prefixed variant, against the project's 0.05 threshold.
4. **Router error rate** — the keyword router vs the plan's ground-truth labels, with
   the deciding keyword shown for every misroute. The repo has never had labelled
   data for this before.

---

## Gotchas found while building this

- **`DEVICE_BATCH_SIZE` and `EVAL_BATCH_SIZE` are dead knobs on a supported GPU.**
  On an RTX 4090 the fork's autotune picks the train batch from
  `gpu_profile.train_batch_candidates` (`train.py:857`) and the eval batch from
  `gpu_profile.eval_batch_cap` (`train.py:1298`). The module constants are only
  consulted on the `compatibility` profile. Editing them looks like an experiment and
  changes nothing — they are deliberately absent from `hypotheses.json`.
- **The keyword router is weak on natural phrasing.** Keywords are now anchored at a
  word start (it used to fire `'rope'` inside "properly") and unmatched cards are
  labelled `unknown` instead of silently `optimizer`, but naturally-phrased
  regularization cards ("back the decay penalty off") still match nothing: 0 of 5 on
  the first real sweep. Prefer the emergent coordinator.
- **Keeps are noise-gated.** `sweep` runs the baseline 3 times by default and keeps an
  experiment only if it beats the incumbent by more than 2x the baseline std
  (`--baseline-repeats`, `--min-improvement`).
- **vastai >= 1.6 exits 0 on API errors and prompts on destroy.** `vastctl` handles
  both; `down` confirms by checking the instance has left the account.
- **CPU cores matter.** The fork tokenises from parquet on the fly every step, so a
  core-starved box throttles the GPU. `search`/`up` filter on
  `cpu_cores_effective>=8`.
- **Instance disk defaults to 60 GB** — dataset is 673 MB, but torch cu128 plus CUDA
  libs is several GB and the base image is not small.

---

## Extending to S2 / S3

The review's next two experiments reuse this harness:

- **S2 (coordinated vs independent vs 1×K-time)** — rent K boxes with `--label`, point
  each at `scas.loop` (`uv run python -m scas.loop prompt/report`) against one
  coordinator, and add the C3 arm as a single box with K× the `--hours` budget.
  Renting separate boxes also fixes the thing a single gaming PC cannot: the swarm
  gets real parallelism instead of time-slicing one GPU.
- **S3 (trivial-hub baseline)** — swap the coordinator for a shared string-level
  avoid-list; the sweep loop itself does not change.

Both need section 2 of the analysis to come back OK (or the floor recalibrated) first.

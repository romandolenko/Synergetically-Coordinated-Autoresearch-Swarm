"""Replay-mode (T5.1-T5.3).

Feed a real-or-synthetic `results.tsv` through the coordinator. Each row's
`description` is embedded by MiniLM, routed to one of the three categories by
keyword heuristic, and passed to `Coordinator.observe`. Emits a per-row JSONL
trace and a silhouette score over the keyword-derived category labels (T5.2).

Usage:
    uv run python -m scas.replay [tsv_path] [--coupling-T 1.0]

If `tsv_path` is omitted, the bundled fixture
`scas/fixtures/sample_results.tsv` is written and used.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from scas.coordinator import Coordinator, HypothesisCard
from scas.embedder import Embedder

CATEGORIES = ["optimizer", "architecture", "regularization"]

_OPT_KWS = [
    "optimizer", "adam", "muon", "sgd", "lion", "lamb", "lars", "sophia",
    "rmsprop", "adagrad", "adafactor",
    "lr ", "lr=", "learning rate", "momentum", "warmup", "cosine", "schedule",
]
_ARCH_KWS = [
    "depth", "width", "head", "layer", "activation", "gelu", "swiglu", "geglu",
    "pre-norm", "post-norm", "rmsnorm", "rope", "rotary", "ffn", "mlp",
    "embed", "tie ", "attention",
]
_REG_KWS = [
    "dropout", "weight_decay", "weight decay", "label_smooth", "label smooth",
    "stochastic depth", "grad_clip", "grad clip", "gradient clip",
    "noise", "mixup", "cutmix", "augment", "spectral", "mask",
]

SAMPLE_TSV = "\t".join(["commit", "val_bpb", "memory_gb", "status", "description"]) + "\n" + "\n".join([
    "a1b2c3d\t0.997900\t44.0\tkeep\tbaseline",
    "b2c3d4e\t0.993200\t44.2\tkeep\tincrease LR to 0.04",
    "c3d4e5f\t1.005000\t44.0\tdiscard\tswitch to GeLU activation",
    "d4e5f6g\t0.000000\t0.0\tcrash\tdouble model width (OOM)",
    "e5f6g7h\t0.991100\t44.5\tkeep\tswitch optimizer from Adam to AdamW",
    "f6g7h8i\t0.989800\t44.5\tkeep\tadd cosine LR schedule with 200 warmup steps",
    "g7h8i9j\t0.992000\t44.6\tdiscard\ttry Lion optimizer with lr=3e-4",
    "h8i9j0k\t0.988300\t44.5\tkeep\treplace Adam with Muon on hidden weights",
    "i9j0k1l\t0.987200\t45.1\tkeep\tincrease model depth to 12 layers",
    "j0k1l2m\t0.991500\t45.0\tdiscard\tswitch to SwiGLU activation in FFN",
    "k1l2m3n\t0.986800\t45.2\tkeep\tadd dropout=0.1 on attention output",
    "l2m3n4o\t0.985900\t45.2\tkeep\tweight_decay=0.05 on linear layers",
    "m3n4o5p\t0.988100\t45.2\tdiscard\tlabel smoothing 0.1",
    "n4o5p6q\t0.985000\t45.5\tkeep\tadd rotary positional embeddings",
    "o5p6q7r\t0.984200\t45.5\tkeep\ttie input and output embeddings",
    "p6q7r8s\t0.987800\t45.6\tdiscard\texpand FFN from 4x to 8x",
    "q7r8s9t\t0.983500\t45.4\tkeep\tincrease attention heads to 16",
    "r8s9t0u\t0.984900\t45.4\tdiscard\tswitch to pre-norm with RMSNorm",
    "s9t0u1v\t0.982700\t45.6\tkeep\tstochastic depth 0.1",
    "t0u1v2w\t0.982000\t45.6\tkeep\tgradient clipping at 1.0",
    "u1v2w3x\t0.985000\t45.5\tdiscard\trandom masking augmentation 0.05",
    "v2w3x4y\t0.981400\t45.7\tkeep\tadd weight noise std=0.01 to FFN",
    "w3x4y5z\t0.983300\t45.8\tdiscard\tspectral normalization on QK projections",
    "x4y5z6a\t0.000000\t0.0\tcrash\twiden FFN 12x and double depth (OOM)",
    "y5z6a7b\t0.980800\t45.7\tkeep\tswitch optimizer to LAMB with lr=1e-3",
    "z6a7b8c\t0.980000\t45.8\tkeep\tSGD with momentum=0.95 and warmup",
    "a7b8c9d\t0.982200\t45.8\tdiscard\tincrease width to 1024",
    "b8c9d0e\t0.979300\t45.9\tkeep\tmixup with alpha=0.2",
    "c9d0e1f\t0.978500\t46.0\tkeep\tcutmix augmentation prob=0.3",
    "d0e1f2g\t0.978000\t46.0\tkeep\tcombine cosine schedule with Sophia optimizer",
])


def _ensure_sample_tsv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(SAMPLE_TSV)
    return path


def _category_for(description: str) -> str:
    d = description.lower()
    for k in _OPT_KWS:
        if k in d:
            return "optimizer"
    for k in _ARCH_KWS:
        if k in d:
            return "architecture"
    for k in _REG_KWS:
        if k in d:
            return "regularization"
    return "optimizer"  # fallback (e.g. "baseline")


def replay(tsv_path: Path, coupling_T: float = 1.0) -> tuple[Path, dict]:
    embedder = Embedder()
    with tsv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    agents = [f"a{i}" for i in range(len(CATEGORIES))]
    coord = Coordinator(agents, CATEGORIES, coupling_T=coupling_T)

    run_dir = Path("sim_runs") / f"replay-{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"

    embeddings: list[np.ndarray] = []
    embeddings_prefixed: list[np.ndarray] = []
    cat_labels: list[str] = []

    with trace_path.open("w") as f:
        for step, r in enumerate(rows):
            desc = r["description"]
            cat = _category_for(desc)
            agent_id = agents[CATEGORIES.index(cat)]
            emb = embedder.embed(desc)
            # Review concern #5: test the proposed live-phase mitigation —
            # prepend an explicit [CATEGORY] tag to restore the prefix anchor
            # MiniLM needs. Measured head-to-head against the raw embedding.
            emb_prefixed = embedder.embed(f"[{cat.upper()}] {desc}")
            embeddings.append(emb)
            embeddings_prefixed.append(emb_prefixed)
            cat_labels.append(cat)

            try:
                val_bpb = float(r.get("val_bpb", "0") or "0")
            except ValueError:
                val_bpb = 0.0
            status = r.get("status", "keep")
            accepted = status != "crash" and val_bpb > 0.0

            card = HypothesisCard(
                agent_id=agent_id,
                step=step,
                summary=desc,
                category=cat,
                parent_step=step - 1 if step > 0 else None,
            )
            if accepted:
                coord.observe(card, val_bpb, emb, accepted=True)

            f.write(
                json.dumps(
                    {
                        "step": step,
                        "commit": r.get("commit"),
                        "description": desc,
                        "category_assigned": cat,
                        "agent_id": agent_id,
                        "val_bpb": val_bpb,
                        "status": status,
                        "accepted": accepted,
                        "psi": coord.state.psi.get(agent_id),
                        "alpha": coord.state.alpha.get(agent_id),
                        "group_id": CATEGORIES.index(cat),
                    }
                )
                + "\n"
            )

    from sklearn.metrics import silhouette_score, silhouette_samples

    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    y = np.array([cat_to_idx[c] for c in cat_labels])
    n_classes = len(set(y.tolist()))

    def _silhouette(emb_list: list[np.ndarray]) -> tuple[float, dict[str, float]]:
        if n_classes < 2 or len(y) <= n_classes:
            return float("nan"), {}
        X = np.stack(emb_list)
        s = float(silhouette_score(X, y))
        samp = silhouette_samples(X, y)
        per = {}
        for c in CATEGORIES:
            mask = y == cat_to_idx[c]
            if mask.any():
                per[c] = float(np.mean(samp[mask]))
        return s, per

    sil, per_cat_sil = _silhouette(embeddings)
    sil_prefixed, per_cat_sil_prefixed = _silhouette(embeddings_prefixed)

    threshold = 0.05
    counts = {c: cat_labels.count(c) for c in CATEGORIES}
    summary = {
        "tsv": str(tsv_path),
        "n_rows": len(rows),
        "category_counts": counts,
        "silhouette_score": sil,
        "per_category_silhouette": per_cat_sil,
        "silhouette_threshold": threshold,
        "silhouette_pass": (sil == sil) and (sil > threshold),  # NaN-safe
        # Review concern #5 — the [CATEGORY]-prefix mitigation, measured:
        "silhouette_score_prefixed": sil_prefixed,
        "per_category_silhouette_prefixed": per_cat_sil_prefixed,
        "silhouette_pass_prefixed": (sil_prefixed == sil_prefixed) and (sil_prefixed > threshold),
        "prefix_mitigation_lift": (
            (sil_prefixed - sil) if (sil == sil and sil_prefixed == sil_prefixed) else None
        ),
        "coupling_T": coupling_T,
        "n_accepted_into_coordinator": sum(
            1 for line in trace_path.read_text().splitlines()
            if json.loads(line)["accepted"]
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return run_dir, summary


def main() -> None:
    args = sys.argv[1:]
    tsv: Path | None = None
    coupling_T = 1.0
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--coupling-T":
            coupling_T = float(args[i + 1])
            i += 2
            continue
        if not a.startswith("--"):
            tsv = Path(a)
        i += 1
    if tsv is None:
        tsv = Path("scas/fixtures/sample_results.tsv")
        _ensure_sample_tsv(tsv)
    replay(tsv, coupling_T=coupling_T)


if __name__ == "__main__":
    main()

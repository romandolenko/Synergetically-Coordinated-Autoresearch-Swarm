"""Synthetic hypothesis proposer + ground-truth landscape + noisy evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Expanded ~10x via cross-product of slot dimensions, so baseline can't trivially
# saturate within the agent-step budget. Each slot is a single unique multi-token
# string; _slot_of's substring check still works because slots don't share
# prefix-as-full-string within a category.
def _opt_slots() -> list[str]:
    bases = [
        "SGD", "Adam", "AdamW", "Lion", "Muon", "Adafactor",
        "RMSprop", "Adagrad", "LARS", "LAMB", "Sophia",
    ]
    lrs = ["lr=1e-4", "lr=3e-4", "lr=1e-3", "lr=3e-3"]
    moms = ["mom=0.0", "mom=0.9", "mom=0.95"]
    return [f"{b} {lr} {m}" for b in bases for lr in lrs for m in moms]


def _arch_slots() -> list[str]:
    cores = [
        "depth=4", "depth=8", "depth=12", "depth=16",
        "width=128", "width=256", "width=512", "width=1024",
        "heads=4", "heads=8", "heads=16",
    ]
    inits = ["init=orth", "init=normal", "init=xavier"]
    norms = ["norm=pre", "norm=post", "norm=none"]
    return [f"{c} {i} {n}" for c in cores for i in inits for n in norms]


def _reg_slots() -> list[str]:
    cores = [
        "dropout=0.0", "dropout=0.1", "dropout=0.2",
        "weight_decay=0.0", "weight_decay=0.01", "weight_decay=0.1",
        "label_smoothing=0.0", "label_smoothing=0.1",
        "grad_clip=0.5", "grad_clip=1.0", "stochastic_depth=0.1",
    ]
    noises = ["noise=0", "noise=low", "noise=high"]
    smooths = ["smooth=0.0", "smooth=0.05", "smooth=0.1"]
    return [f"{c} {n} {s}" for c in cores for n in noises for s in smooths]


# Extra categories (appended after the original three so default reproducibility
# is preserved: list(TAXONOMY.keys())[:3] is unchanged in content and order).
# They let the simulator run with more groups (G>3) to test whether finer
# partitioning relieves the K'>>G "crowding" that fails AC-B2 at K'=10.
def _sched_slots() -> list[str]:
    kinds = [
        "cosine", "linear", "step", "exponential",
        "onecycle", "constant", "poly", "inverse_sqrt",
    ]
    warm = ["warmup=0", "warmup=100", "warmup=500", "warmup=2000"]
    ends = ["end_lr=0", "end_lr=1e-5", "end_lr=1e-4"]
    return [f"{k} {w} {e}" for k in kinds for w in warm for e in ends]  # 8*4*3 = 96


def _data_slots() -> list[str]:
    bss = ["bs=32", "bs=64", "bs=128", "bs=256", "bs=512"]
    seqs = ["seq=512", "seq=1024", "seq=2048", "seq=4096"]
    mixes = ["mixup=0", "mixup=0.2", "cutmix=0.3", "randaug", "augmix"]
    return [f"{b} {s} {m}" for b in bss for s in seqs for m in mixes]  # 5*4*5 = 100


def _init_slots() -> list[str]:
    schemes = [
        "xavier", "kaiming", "orthogonal",
        "normal", "truncated_normal", "zeros_bias",
    ]
    scales = ["scale=0.02", "scale=0.5", "scale=1.0", "scale=2.0"]
    seeds = ["seed=0", "seed=1", "seed=42", "seed=1337"]
    return [f"{sc} {s} {sd}" for sc in schemes for s in scales for sd in seeds]  # 6*4*4 = 96


TAXONOMY: dict[str, list[str]] = {
    "optimizer": _opt_slots(),         # 11 * 4 * 3 = 132
    "architecture": _arch_slots(),     # 11 * 3 * 3 = 99
    "regularization": _reg_slots(),    # 11 * 3 * 3 = 99
    "schedule": _sched_slots(),        # 8 * 4 * 3 = 96
    "data": _data_slots(),             # 5 * 4 * 5 = 100
    "init": _init_slots(),             # 6 * 4 * 4 = 96
}  # first 3 keys (330 strings) unchanged → committed defaults reproduce exactly

TEMPLATES: dict[str, str] = {
    "optimizer": "change optimizer to {slot}",
    "architecture": "tune architecture: {slot}",
    "regularization": "add regularization: {slot}",
    "schedule": "set lr schedule: {slot}",
    "data": "adjust data pipeline: {slot}",
    "init": "change initialization: {slot}",
}


def _slot_of(text: str, category: str) -> str | None:
    for slot in TAXONOMY[category]:
        if slot in text:
            return slot
    return None


def propose(
    category: str,
    target: str | None = None,
    avoid: list[str] | None = None,
    alpha: float | None = None,
    rng: np.random.Generator | None = None,
) -> str:
    """Sample a hypothesis card string for `category`.

    The mix between local (neighbor of `target`) and global (random from pool with
    `avoid` filter) picks is controlled by `alpha`:
        alpha == 1.0  → always neighbor of target (if target given)
        alpha == 0.0  → always global pick with avoid filter
        0 < alpha < 1 → bernoulli(alpha) chooses neighbor vs global

    `alpha=None` is the legacy mode: target → neighbor (always); no target → global.
    """
    if category not in TAXONOMY:
        raise ValueError(f"unknown category: {category}")
    rng = rng if rng is not None else np.random.default_rng()
    slots = TAXONOMY[category]
    template = TEMPLATES[category]
    avoid = avoid or []

    take_neighbor = False
    if target is not None:
        if alpha is None:
            take_neighbor = True
        else:
            take_neighbor = bool(rng.random() < float(alpha))

    if take_neighbor:
        cur = _slot_of(target, category)
        if cur is not None:
            i = slots.index(cur)
            neighbors: list[str] = []
            if i > 0:
                neighbors.append(slots[i - 1])
            if i + 1 < len(slots):
                neighbors.append(slots[i + 1])
            neighbors = [s for s in neighbors if not any(s in a for a in avoid)]
            if neighbors:
                return template.format(slot=str(rng.choice(neighbors)))

    pool = [s for s in slots if not any(s in a for a in avoid)] or list(slots)
    return template.format(slot=str(rng.choice(pool)))


@dataclass(frozen=True)
class Landscape:
    mu: np.ndarray              # (K', D) attractor centers in embedding space
    sigma: np.ndarray           # (K',) Gaussian widths
    depth: np.ndarray           # (K',) per-minimum reward magnitudes
    base: float                 # baseline val_bpb
    noise_std: float            # per-call gaussian noise on evaluate()
    minima_strings: tuple[str, ...]  # provenance: the proposer strings that map to each mu_k


def all_templated_strings(categories: list[str] | None = None) -> list[str]:
    """Templated strings for the given categories (default: all, in dict order).

    Restricting to the in-use categories keeps minima inside the categories the
    groups actually cover, and preserves reproducibility: for the default 3
    categories this is byte-identical to the original all-categories output.
    """
    cats = categories if categories is not None else list(TAXONOMY.keys())
    out: list[str] = []
    for cat in cats:
        tmpl = TEMPLATES[cat]
        for s in TAXONOMY[cat]:
            out.append(tmpl.format(slot=s))
    return out


def make_landscape(
    seed: int,
    k_prime: int,
    embedder,
    categories: list[str] | None = None,
    base: float = 1.0,
    depth_range: tuple[float, float] = (0.04, 0.08),
    sigma: float = 0.5,
    noise_std: float = 0.005,
) -> Landscape:
    """Build a fixed landscape by sampling k_prime taxonomy strings as minima centers.

    Minima are placed at actual proposer-output embeddings so the simulator's agents
    can in principle reach them. Sampling is restricted to `categories` (the groups
    in use) so every minimum is coverable. The chosen strings are stored for
    provenance.
    """
    rng = np.random.default_rng(seed)
    all_strings = all_templated_strings(categories)
    if k_prime > len(all_strings):
        raise ValueError(
            f"k_prime={k_prime} exceeds taxonomy size {len(all_strings)}"
        )
    idx = rng.choice(len(all_strings), size=k_prime, replace=False)
    chosen = [all_strings[int(i)] for i in idx]
    mu = np.stack([embedder.embed(s) for s in chosen]).astype(np.float32)
    depth = rng.uniform(depth_range[0], depth_range[1], size=k_prime).astype(np.float32)
    sigmas = np.full(k_prime, sigma, dtype=np.float32)
    return Landscape(
        mu=mu,
        sigma=sigmas,
        depth=depth,
        base=float(base),
        noise_std=float(noise_std),
        minima_strings=tuple(chosen),
    )


def evaluate(z: np.ndarray, landscape: Landscape, rng: np.random.Generator) -> float:
    """Noisy val_bpb at point z. Lower is better; minima at landscape.mu."""
    diffs = landscape.mu - z[None, :]
    sq = np.sum(diffs * diffs, axis=1)
    activations = landscape.depth * np.exp(-sq / (landscape.sigma ** 2))
    return float(
        landscape.base - float(np.sum(activations)) + float(rng.normal(0.0, landscape.noise_std))
    )

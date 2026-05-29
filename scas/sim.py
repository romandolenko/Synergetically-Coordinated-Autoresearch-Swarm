"""SCAS simulator entry point (T2.3 + T3.1/T3.2).

Modes:
- default: one SCAS run
- --baseline: one independent-walkers run
- --paired-seeds N: SCAS + baseline on seeds 0..N-1, with aggregate summaries

Per-run output bundle (design §10):
    sim_runs/<run_id>/
        config.json
        events.jsonl
        attractors.jsonl   (empty for baseline)
        landscape.json
        summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Union

import numpy as np

from scas.baselines import NullCoordinator, PartitionedNullCoordinator
from scas.coordinator import (
    BASELINE_PREPARE_HASH,
    BASELINE_PYPROJECT_HASH,
    PSI_0,
    W_P,
    Coordinator,
    HypothesisCard,
    validate,
)
from scas.embedder import Embedder
from scas.synthetic import (
    TAXONOMY,
    TEMPLATES,
    Landscape,
    evaluate,
    make_landscape,
    propose,
)

CoordLike = Union[Coordinator, NullCoordinator, PartitionedNullCoordinator]


def _make_run_id(seed: int, tag: str) -> str:
    return f"{tag}-seed{seed}-{time.strftime('%Y%m%dT%H%M%S')}"


def _mean_pairwise(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    ds: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            ds.append(float(np.linalg.norm(vectors[i] - vectors[j])))
    return float(np.mean(ds))


def _intra_group_dist(
    embeddings_log: list[tuple[int, np.ndarray]],
    attractors: dict[int, np.ndarray],
) -> float:
    ds: list[float] = []
    for g, e in embeddings_log:
        attr = attractors.get(g)
        if attr is None:
            continue
        ds.append(float(np.linalg.norm(e - attr)))
    return float(np.mean(ds)) if ds else 0.0


def _distinct_minima_visited(
    embeddings_log: list[tuple[int | None, np.ndarray]],
    landscape: Landscape,
    sigma_fraction: float = 0.5,
) -> int:
    return sum(
        1 for v in _first_visit_steps(embeddings_log, landscape, sigma_fraction) if v is not None
    )


def _first_visit_steps(
    embeddings_log: list[tuple[int | None, np.ndarray]],
    landscape: Landscape,
    sigma_fraction: float = 0.5,
) -> list[int | None]:
    """Per-minimum agent-step (1-indexed) at which it was first visited; None if never."""
    K_prime = landscape.mu.shape[0]
    first: list[int | None] = [None] * K_prime
    for idx, (_g, e) in enumerate(embeddings_log):
        diffs = landscape.mu - e[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        for k in range(K_prime):
            if first[k] is None and float(dists[k]) < float(landscape.sigma[k]) * sigma_fraction:
                first[k] = idx + 1
    return first


def _steps_to_coverage(
    first_visit: list[int | None],
    K_prime: int,
    fractions: tuple[float, ...] = (0.5, 0.8, 1.0),
) -> dict[str, int | None]:
    visited_sorted = sorted(v for v in first_visit if v is not None)
    out: dict[str, int | None] = {}
    for f in fractions:
        need = int(np.ceil(f * K_prime))
        out[f"{f}"] = visited_sorted[need - 1] if len(visited_sorted) >= need else None
    return out


def _select_embedding(
    cand_embs: np.ndarray,         # (n_cand, D), unit-norm candidate string embeddings
    covered: np.ndarray,           # (n_cov, D), unit-norm; may be empty
    exploit: bool,
    attractor_emb: np.ndarray | None,
    rng: np.random.Generator,
    top_k: int = 3,
) -> int:
    """Pick a candidate index by embedding-space novelty/refinement.

    Embeddings are unit-norm, so cosine similarity = dot product and
    ||a-b|| is monotone-decreasing in a·b. Explore = farthest-point from the
    covered set (max-min distance ⇔ min-max similarity); exploit = closest to
    the group attractor (excluding exact-covered). Both are O(n_cand·n_cov).
    """
    n = cand_embs.shape[0]
    has_cov = covered.shape[0] > 0
    if exploit and attractor_emb is not None:
        sim_attr = cand_embs @ attractor_emb
        if has_cov:
            already = (cand_embs @ covered.T).max(axis=1) > 0.999
            sim_attr = sim_attr - already * 1e9
        return int(np.argmax(sim_attr))
    if not has_cov:
        return int(rng.integers(n))
    nearest_sim = (cand_embs @ covered.T).max(axis=1)  # higher ⇒ closer to covered
    k = min(top_k, n)
    farthest = np.argpartition(nearest_sim, k - 1)[:k]  # k least-covered candidates
    return int(farthest[rng.integers(len(farthest))])


def _run_one(
    seed: int,
    mode: str,
    K: int,
    K_prime: int,
    steps: int,
    coupling_T: float,
    parent_dir: Path | None = None,
    embedder: Embedder | None = None,
    w_p: float = W_P,
    psi_0: float = PSI_0,
    agents_per_group: int = 1,
    coverage_coupling: bool = False,
    embedding_repulsion: bool = False,
    shared_coverage: bool = True,
    anneal: bool = False,
    anneal_start: float = 0.5,
) -> Path:
    assert mode in ("scas", "baseline", "partition")
    rng = np.random.default_rng(seed)
    embedder = embedder if embedder is not None else Embedder()
    # K is the number of *groups* (= categories); agents_per_group decouples the
    # swarm size from the group count (review fix #1). Round-robin assignment in
    # the coordinator (i % n_groups) then puts agents_per_group members in each
    # group, so multi-consensus actually has multiple members to coordinate.
    n_groups = K
    categories = list(TAXONOMY.keys())[:n_groups]
    if len(categories) < n_groups:
        raise ValueError(f"--K={K} exceeds taxonomy categories ({len(TAXONOMY)})")
    n_agents = n_groups * agents_per_group

    landscape = make_landscape(
        seed=seed, k_prime=K_prime, embedder=embedder, categories=categories
    )
    agent_ids = [f"a{i}" for i in range(n_agents)]

    if mode == "scas":
        coord: CoordLike = Coordinator(
            agent_ids, categories, coupling_T=coupling_T, w_p=w_p, psi_0=psi_0
        )
    elif mode == "partition":
        coord = PartitionedNullCoordinator(agent_ids, categories)
    else:
        coord = NullCoordinator(agent_ids)

    run_root = parent_dir if parent_dir is not None else Path("sim_runs")
    run_dir = run_root / _make_run_id(seed, mode)
    run_dir.mkdir(parents=True, exist_ok=True)

    attractor_text: dict[int, str | None] = {g: None for g in range(n_groups)}
    # Review fixes #2/#3: per-group coverage memory (summaries the group has
    # already tried) and each agent's most recent summary (for sibling
    # dispersion). Only consulted when coverage_coupling is on, and only in scas
    # mode — the fair control deliberately gets no steering.
    group_covered: dict[int, set[str]] = {g: set() for g in range(n_groups)}
    agent_last_summary: dict[str, str] = {}

    # Embedding-space-repulsion variant (scas only): precompute the unit-norm
    # embedding of every candidate string per category once, plus a per-group
    # buffer of covered embeddings for farthest-point exploration. Exploit is
    # gated to <=1 agent per group per step so group-mates stop co-clustering.
    use_emb_rep = embedding_repulsion and mode == "scas"
    cand_strings: dict[str, list[str]] = {}
    cand_embs: dict[str, np.ndarray] = {}
    # Coverage buffers are keyed by group when shared (coordination: group-mates
    # divide the search + exploit-gating), or by agent when independent (the
    # "smart-walker" control that gets the same farthest-point search but no
    # coordination — isolates search strategy from coordination).
    covered_buf: dict = {}
    covered_n: dict = {}
    if use_emb_rep:
        for g in range(n_groups):
            cat_g = coord.group_to_category[g]
            strings = [TEMPLATES[cat_g].format(slot=s) for s in TAXONOMY[cat_g]]
            embs = np.stack([embedder.embed(s) for s in strings]).astype(np.float32)
            cand_strings[cat_g] = strings
            cand_embs[cat_g] = embs
        dim = next(iter(cand_embs.values())).shape[1]
        if shared_coverage:
            covered_buf = {
                g: np.zeros((steps * agents_per_group, dim), dtype=np.float32)
                for g in range(n_groups)
            }
            covered_n = {g: 0 for g in range(n_groups)}
        else:
            covered_buf = {a: np.zeros((steps, dim), dtype=np.float32) for a in agent_ids}
            covered_n = {a: 0 for a in agent_ids}

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "seed": seed,
                "K": K,
                "K_prime": K_prime,
                "steps": steps,
                "coupling_T": coupling_T,
                "agents_per_group": agents_per_group,
                "n_agents": n_agents,
                "coverage_coupling": coverage_coupling,
                "embedding_repulsion": embedding_repulsion,
            },
            indent=2,
        )
    )
    (run_dir / "landscape.json").write_text(
        json.dumps(
            {
                "mu_shape": list(landscape.mu.shape),
                "sigma": landscape.sigma.tolist(),
                "depth": landscape.depth.tolist(),
                "base": landscape.base,
                "noise_std": landscape.noise_std,
                "minima_strings": list(landscape.minima_strings),
            },
            indent=2,
        )
    )

    events_path = run_dir / "events.jsonl"
    attractors_path = run_dir / "attractors.jsonl"

    n_violations = 0
    best_val_bpb = float("inf")
    embeddings_log: list[tuple[int | None, np.ndarray]] = []
    # Baseline roams only the in-use categories (so it isn't penalised for
    # sampling categories that contain no minima); for default K=3 this is the
    # original three categories, preserving committed-default numbers.
    cat_keys = categories

    with events_path.open("w") as ev_f, attractors_path.open("w") as at_f:
        for step in range(steps):
            # Exploit-gating (emb-repulsion variant): at most one agent per group
            # may pursue the attractor each step; the rest explore.
            group_exploit_count = {g: 0 for g in range(n_groups)}
            for agent_id in agent_ids:
                g = coord.agent_to_group[agent_id]  # int (scas) or None (baseline)
                if g is not None:
                    cat = coord.group_to_category[g]
                else:
                    cat = cat_keys[int(rng.integers(len(cat_keys)))]
                alpha = coord.state.alpha.get(agent_id)

                if use_emb_rep and g is not None:
                    ckey = g if shared_coverage else agent_id
                    cov = covered_buf[ckey][: covered_n[ckey]]
                    attractor_emb = (
                        coord.state.attractors.get(g) if shared_coverage else None
                    )
                    progress = step / max(1, steps - 1)
                    converging = (
                        anneal
                        and shared_coverage
                        and progress >= anneal_start
                        and attractor_emb is not None
                    )
                    if converging:
                        # Anneal phase (#4): every group-mate settles onto the
                        # group attractor (no gate, covered ignored) so intra
                        # shrinks. Coverage is already secured in the explore
                        # phase (~40 agent-steps << anneal_start), so AC-B2 holds.
                        idx = int(np.argmax(cand_embs[cat] @ attractor_emb))
                    elif shared_coverage:
                        # Explore: farthest-point + gated single exploit (§6c).
                        a = alpha if alpha is not None else 0.0
                        do_exploit = (
                            attractor_emb is not None
                            and group_exploit_count[g] < 1
                            and rng.random() < a
                        )
                        idx = _select_embedding(
                            cand_embs[cat], cov, do_exploit, attractor_emb, rng
                        )
                        if do_exploit:
                            group_exploit_count[g] += 1
                    else:
                        # Independent smart-walker control: own-history
                        # farthest-point only, no shared memory, no exploit.
                        idx = _select_embedding(cand_embs[cat], cov, False, None, rng)
                    summary = cand_strings[cat][idx]
                    emb = cand_embs[cat][idx]
                    covered_buf[ckey][covered_n[ckey]] = emb
                    covered_n[ckey] += 1
                else:
                    target = None
                    avoid = None
                    if mode == "scas" and g is not None:
                        target = attractor_text[g]  # may still be None on the very first step
                        if coverage_coupling:
                            # Fix #2: avoid slots this group has already covered
                            # (real anti-redundancy — the old other-group-attractor
                            # avoid was a no-op because those live in other
                            # categories). Fix #3: also avoid where group-mates
                            # currently sit, so coupled agents divide the search.
                            siblings = [
                                agent_last_summary[a]
                                for a in coord.state.groups.get(g, [])
                                if a != agent_id and a in agent_last_summary
                            ]
                            avoid = list(group_covered[g]) + siblings
                        else:
                            # Legacy (no-op) behavior, preserved for reproducibility.
                            avoid = [
                                t
                                for gg, t in attractor_text.items()
                                if gg != g and t is not None
                            ]

                    summary = propose(cat, target=target, avoid=avoid, alpha=alpha, rng=rng)
                    if mode == "scas" and g is not None and coverage_coupling:
                        group_covered[g].add(summary)
                        agent_last_summary[agent_id] = summary
                    emb = embedder.embed(summary)
                card = HypothesisCard(
                    agent_id=agent_id,
                    step=step,
                    summary=summary,
                    category=cat,
                    parent_step=(step - 1) if step > 0 else None,
                )
                val_bpb = evaluate(emb, landscape, rng)
                commit = {
                    "agent_id": agent_id,
                    "step": step,
                    "files_changed": ["train.py"],
                    "pyproject_hash": BASELINE_PYPROJECT_HASH,
                    "prepare_hash": BASELINE_PREPARE_HASH,
                    "val_bpb": val_bpb,
                }
                ok, reason = validate(commit)
                if ok:
                    improved = coord.observe(card, val_bpb, emb, accepted=True)
                    if improved and g is not None:
                        attractor_text[g] = summary
                        at_f.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "group_id": g,
                                    "agent_id": agent_id,
                                    "new_attractor_summary": summary,
                                    "group_best_val_bpb": coord.state.group_best_val_bpb[g],
                                }
                            )
                            + "\n"
                        )
                    if val_bpb < best_val_bpb:
                        best_val_bpb = val_bpb
                    embeddings_log.append((g, emb))
                else:
                    n_violations += 1

                ev_f.write(
                    json.dumps(
                        {
                            "step": step,
                            "agent_id": agent_id,
                            "card": {
                                "summary": summary,
                                "category": cat,
                                "parent_step": card.parent_step,
                            },
                            "val_bpb": val_bpb,
                            "accepted": ok,
                            "reject_reason": reason,
                            "psi": coord.state.psi.get(agent_id),
                            "alpha": coord.state.alpha.get(agent_id),
                            "group_id": g,
                        }
                    )
                    + "\n"
                )

    if mode in ("scas", "partition"):
        attractors_list = list(coord.state.attractors.values())
        inter = _mean_pairwise(attractors_list)
        intra = _intra_group_dist(
            [(g, e) for g, e in embeddings_log if g is not None],
            coord.state.attractors,
        )
        inter_over_intra = (inter / intra) if intra > 0 else None
    else:
        inter = None
        intra = None
        inter_over_intra = None

    first_visit = _first_visit_steps(embeddings_log, landscape)
    distinct = sum(1 for v in first_visit if v is not None)
    steps_to_cov = _steps_to_coverage(first_visit, K_prime)
    summary = {
        "mode": mode,
        "seed": seed,
        "best_val_bpb": best_val_bpb,
        "inter_group_dist": inter,
        "intra_group_dist": intra,
        "inter_over_intra": inter_over_intra,
        "distinct_minima_visited": distinct,
        "first_visit_steps": [v for v in first_visit if v is not None],
        "steps_to_coverage": steps_to_cov,
        "agent_step_budget": steps * n_agents,
        "n_violations": n_violations,
        "n_steps": steps,
        "K": K,
        "K_prime": K_prime,
        "coupling_T": coupling_T,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


def _aggregate(per_seed_dirs: list[Path], out_path: Path, mode: str) -> dict:
    rows = [json.loads((p / "summary.json").read_text()) for p in per_seed_dirs]
    best_vals = [r["best_val_bpb"] for r in rows]
    distinct_vals = [r["distinct_minima_visited"] for r in rows]

    # steps-to-coverage with bounded "never reached" = agent_step_budget
    def _bounded(r: dict, key: str) -> int:
        v = r["steps_to_coverage"].get(key)
        return int(v) if v is not None else int(r["agent_step_budget"])

    s50 = [_bounded(r, "0.5") for r in rows]
    s80 = [_bounded(r, "0.8") for r in rows]
    s100 = [_bounded(r, "1.0") for r in rows]

    agg: dict = {
        "mode": mode,
        "n_seeds": len(rows),
        "per_seed": [
            {
                "seed": r["seed"],
                "run_dir": str(p),
                "best_val_bpb": r["best_val_bpb"],
                "distinct_minima_visited": r["distinct_minima_visited"],
                "inter_group_dist": r["inter_group_dist"],
                "intra_group_dist": r["intra_group_dist"],
                "inter_over_intra": r["inter_over_intra"],
                "steps_to_coverage": r["steps_to_coverage"],
                "agent_step_budget": r["agent_step_budget"],
                "K_prime": r["K_prime"],
            }
            for r, p in zip(rows, per_seed_dirs)
        ],
        "mean_best_val_bpb": float(np.mean(best_vals)),
        "std_best_val_bpb": float(np.std(best_vals, ddof=1)) if len(rows) > 1 else 0.0,
        "mean_distinct_minima_visited": float(np.mean(distinct_vals)),
        "std_distinct_minima_visited": float(np.std(distinct_vals, ddof=1)) if len(rows) > 1 else 0.0,
        "mean_steps_to_50pct_coverage": float(np.mean(s50)),
        "mean_steps_to_80pct_coverage": float(np.mean(s80)),
        "mean_steps_to_100pct_coverage": float(np.mean(s100)),
    }
    if mode in ("scas", "partition"):
        inters = [r["inter_group_dist"] for r in rows]
        intras = [r["intra_group_dist"] for r in rows]
        agg["mean_inter_group_dist"] = float(np.mean(inters))
        agg["mean_intra_group_dist"] = float(np.mean(intras))
        agg["inter_over_intra_aggregate"] = (
            agg["mean_inter_group_dist"] / agg["mean_intra_group_dist"]
            if agg["mean_intra_group_dist"] > 0
            else None
        )
    out_path.write_text(json.dumps(agg, indent=2))
    return agg


def _run_paired_at_T(
    parent: Path,
    n_seeds: int,
    K: int,
    K_prime: int,
    steps: int,
    coupling_T: float,
    embedder: Embedder,
    w_p: float = W_P,
    psi_0: float = PSI_0,
) -> tuple[dict, dict, Path]:
    """Run paired SCAS+baseline at a fixed T; return (scas_agg, baseline_agg, dir)."""
    sub = parent / f"T{coupling_T}"
    sub.mkdir(parents=True, exist_ok=True)
    scas_dirs: list[Path] = []
    baseline_dirs: list[Path] = []
    for seed in range(n_seeds):
        scas_dirs.append(
            _run_one(seed, "scas", K, K_prime, steps, coupling_T, sub, embedder,
                     w_p=w_p, psi_0=psi_0)
        )
        baseline_dirs.append(
            _run_one(seed, "baseline", K, K_prime, steps, coupling_T, sub, embedder)
        )
    scas_agg = _aggregate(scas_dirs, sub / "paired_scas_summary.json", "scas")
    baseline_agg = _aggregate(
        baseline_dirs, sub / "paired_baseline_summary.json", "baseline"
    )
    return scas_agg, baseline_agg, sub


def _run_sweep_T(args: argparse.Namespace, T_values: list[float]) -> Path:
    n = args.paired_seeds if args.paired_seeds > 0 else 10
    parent = Path("sim_runs") / (
        f"sweep-T{len(T_values)}-N{n}-{time.strftime('%Y%m%dT%H%M%S')}"
    )
    parent.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()
    per_T: list[dict] = []
    for T in T_values:
        scas_agg, baseline_agg, sub = _run_paired_at_T(
            parent, n, args.K, args.K_prime, args.steps, T, embedder,
            w_p=args.w_p, psi_0=args.psi_0,
        )
        s80_scas = scas_agg["mean_steps_to_80pct_coverage"]
        s80_base = baseline_agg["mean_steps_to_80pct_coverage"]
        per_T.append(
            {
                "T": T,
                "paired_dir": str(sub),
                "scas_mean_inter_group_dist": scas_agg.get("mean_inter_group_dist"),
                "scas_mean_intra_group_dist": scas_agg.get("mean_intra_group_dist"),
                "scas_inter_over_intra_aggregate": scas_agg.get("inter_over_intra_aggregate"),
                "scas_per_seed_inter_over_intra": [
                    r["inter_over_intra"] for r in scas_agg["per_seed"]
                ],
                "scas_mean_steps_to_80pct": s80_scas,
                "baseline_mean_steps_to_80pct": s80_base,
                "ac_b2_speedup_ratio": (s80_base / s80_scas) if s80_scas > 0 else None,
            }
        )

    from scipy.stats import kendalltau

    Ts = [p["T"] for p in per_T]
    seps_agg = [p["scas_inter_over_intra_aggregate"] for p in per_T]
    # per-seed ratios flattened: pair each T with each seed's ratio for a robust tau
    flat_T, flat_sep = [], []
    for p in per_T:
        for r in p["scas_per_seed_inter_over_intra"]:
            if r is not None:
                flat_T.append(p["T"])
                flat_sep.append(r)
    tau_agg, p_agg = kendalltau(Ts, seps_agg)
    tau_flat, p_flat = kendalltau(flat_T, flat_sep)
    sweep = {
        "sweep_dir": str(parent),
        "T_values": Ts,
        "n_seeds": n,
        "per_T": per_T,
        "kendall_tau_T_vs_inter_over_intra_agg": float(tau_agg),
        "kendall_tau_T_vs_inter_over_intra_flat": float(tau_flat),
        "kendall_p_T_vs_inter_over_intra_agg": float(p_agg),
        "kendall_p_T_vs_inter_over_intra_flat": float(p_flat),
    }
    (parent / "sweep.json").write_text(json.dumps(sweep, indent=2))
    print(
        json.dumps(
            {
                "sweep_dir": str(parent),
                "T_values": Ts,
                "scas_inter_over_intra_per_T": seps_agg,
                "scas_steps_to_80pct_per_T": [p["scas_mean_steps_to_80pct"] for p in per_T],
                "ac_b2_speedup_per_T": [p["ac_b2_speedup_ratio"] for p in per_T],
                "kendall_tau_agg": float(tau_agg),
                "kendall_tau_flat": float(tau_flat),
                "ac_b4_target_abs_tau_ge": 0.7,
                "ac_b4_pass": abs(float(tau_agg)) >= 0.7,
            },
            indent=2,
        )
    )
    return parent


def _run_paired(args: argparse.Namespace) -> Path:
    n = args.paired_seeds
    parent = Path("sim_runs") / f"paired-N{n}-{time.strftime('%Y%m%dT%H%M%S')}"
    parent.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()
    scas_dirs: list[Path] = []
    baseline_dirs: list[Path] = []
    partition_dirs: list[Path] = []
    apg = args.agents_per_group
    cc = args.coverage_coupling
    # --independent-repulsion implies the embedding-repulsion machinery but with
    # per-agent (uncoordinated) coverage — the smart-walker control. --anneal
    # builds on shared embedding-repulsion (explore early, converge late).
    er = args.embedding_repulsion or args.independent_repulsion or args.anneal
    shared = not args.independent_repulsion
    for seed in range(n):
        scas_dirs.append(
            _run_one(seed, "scas", args.K, args.K_prime, args.steps, args.coupling_T,
                     parent, embedder, w_p=args.w_p, psi_0=args.psi_0,
                     agents_per_group=apg, coverage_coupling=cc, embedding_repulsion=er,
                     shared_coverage=shared, anneal=args.anneal, anneal_start=args.anneal_start)
        )
        baseline_dirs.append(
            _run_one(seed, "baseline", args.K, args.K_prime, args.steps, args.coupling_T,
                     parent, embedder, agents_per_group=apg)
        )
        if not args.no_partition_control:
            partition_dirs.append(
                _run_one(seed, "partition", args.K, args.K_prime, args.steps, args.coupling_T,
                         parent, embedder, agents_per_group=apg)
            )
    scas_agg = _aggregate(scas_dirs, parent / "paired_scas_summary.json", "scas")
    baseline_agg = _aggregate(baseline_dirs, parent / "paired_baseline_summary.json", "baseline")
    s80_scas = scas_agg["mean_steps_to_80pct_coverage"]
    s80_base = baseline_agg["mean_steps_to_80pct_coverage"]
    headline = {
        "paired_dir": str(parent),
        "n_seeds": n,
        "w_p": args.w_p,
        "psi_0": args.psi_0,
        "agents_per_group": apg,
        "coverage_coupling": cc,
        "embedding_repulsion": er,
        "shared_coverage": shared,
        "anneal": args.anneal,
        "anneal_start": args.anneal_start if args.anneal else None,
        "scas_mean_best_val_bpb": scas_agg["mean_best_val_bpb"],
        "baseline_mean_best_val_bpb": baseline_agg["mean_best_val_bpb"],
        "scas_mean_steps_to_80pct": s80_scas,
        "baseline_mean_steps_to_80pct": s80_base,
        "ac_b2_speedup_ratio": (s80_base / s80_scas) if s80_scas > 0 else None,
        "ac_b3_inter_over_intra_aggregate": scas_agg.get("inter_over_intra_aggregate"),
    }
    if partition_dirs:
        part_agg = _aggregate(
            partition_dirs, parent / "paired_partition_summary.json", "partition"
        )
        s80_part = part_agg["mean_steps_to_80pct_coverage"]
        # Fair-control decomposition (review concern #2): the speedup vs the
        # random-category baseline mixes a "forced partition" effect with the
        # "control law" effect. Against the partitioned-but-uncoupled control,
        # the marginal coverage speedup of the control law alone is part/scas.
        headline["partition_mean_steps_to_80pct"] = s80_part
        headline["partition_mean_best_val_bpb"] = part_agg["mean_best_val_bpb"]
        headline["ac_b2_controllaw_marginal_speedup"] = (
            (s80_part / s80_scas) if s80_scas > 0 else None
        )
        headline["ac_b2_partition_alone_speedup"] = (
            (s80_base / s80_part) if s80_part > 0 else None
        )
        # AC-B3 head-to-head (review concern #1): does separation survive when
        # the control law is removed but partition+prefix remain?
        headline["ac_b3_inter_over_intra_partition_control"] = part_agg.get(
            "inter_over_intra_aggregate"
        )
    print(json.dumps(headline, indent=2))
    return parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCAS simulator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=3, help="number of agents / groups")
    p.add_argument(
        "--K-prime", type=int, default=5, dest="K_prime",
        help="number of true minima in the ground-truth landscape",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument(
        "--coupling-T", type=float, default=1.0, dest="coupling_T",
        help="synergetic coupling time constant (single user-facing knob)",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--baseline", action="store_true", help="run independent-walkers baseline")
    g.add_argument(
        "--sweep-T", type=str, default=None, dest="sweep_T",
        help="comma-separated coupling_T values to sweep (paired SCAS+baseline at each)",
    )
    g.add_argument(
        "--all-experiments", action="store_true", dest="all_experiments",
        help="run paired-N10 (AC-B1'/B2/B3) and the default T-sweep (AC-B4) back-to-back",
    )
    p.add_argument(
        "--paired-seeds", type=int, default=0, dest="paired_seeds",
        help="loop seeds 0..N-1, run scas + baseline each (also controls sweep seed count)",
    )
    p.add_argument(
        "--w-p", type=float, default=W_P, dest="w_p",
        help=f"perf-gap weight in psi (default {W_P}; >0 makes the coordinator "
             "performance-aware — the committed default is performance-blind)",
    )
    p.add_argument(
        "--psi-0", type=float, default=PSI_0, dest="psi_0",
        help=f"sigmoid center for alpha (default {PSI_0})",
    )
    p.add_argument(
        "--no-partition-control", action="store_true", dest="no_partition_control",
        help="skip the partitioned-but-uncoupled fair-control arm in paired runs",
    )
    p.add_argument(
        "--agents-per-group", type=int, default=1, dest="agents_per_group",
        help="agents per group/category (review fix #1: decouples swarm size "
             "from group count; default 1 reproduces committed behavior)",
    )
    p.add_argument(
        "--coverage-coupling", action="store_true", dest="coverage_coupling",
        help="review fixes #2/#3: real intra-group coverage-memory repulsion + "
             "sibling dispersion (scas arm only; off by default)",
    )
    p.add_argument(
        "--embedding-repulsion", action="store_true", dest="embedding_repulsion",
        help="embedding-space variant: farthest-point exploration in MiniLM "
             "space + exploit-gating (<=1 agent/group/step pursues the attractor)",
    )
    p.add_argument(
        "--independent-repulsion", action="store_true", dest="independent_repulsion",
        help="smart-walker control: same farthest-point search but per-agent "
             "(no shared group memory, no gating) — isolates search from coordination",
    )
    p.add_argument(
        "--anneal", action="store_true", dest="anneal",
        help="review fix #4 (AC-B3): explore early then converge group-mates onto "
             "the attractor late (built on shared embedding-repulsion)",
    )
    p.add_argument(
        "--anneal-start", type=float, default=0.5, dest="anneal_start",
        help="fraction of the run after which the converge phase begins (default 0.5)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.all_experiments:
        if args.paired_seeds <= 0:
            args.paired_seeds = 10
        print("=== AC-B1' / AC-B2 / AC-B3 (paired-N10, headline config) ===")
        _run_paired(args)
        print("\n=== AC-B4 (T-sweep at headline config) ===")
        args.sweep_T = "0.1,0.3,1.0,3.0,10.0"
        T_values = [float(s.strip()) for s in args.sweep_T.split(",")]
        _run_sweep_T(args, T_values)
        return
    if args.sweep_T:
        T_values = [float(s.strip()) for s in args.sweep_T.split(",")]
        _run_sweep_T(args, T_values)
        return
    if args.paired_seeds > 0:
        _run_paired(args)
        return
    mode = "baseline" if args.baseline else "scas"
    run_dir = _run_one(
        args.seed, mode, args.K, args.K_prime, args.steps, args.coupling_T,
        w_p=args.w_p, psi_0=args.psi_0,
        agents_per_group=args.agents_per_group, coverage_coupling=args.coverage_coupling,
        embedding_repulsion=args.embedding_repulsion or args.independent_repulsion or args.anneal,
        shared_coverage=not args.independent_repulsion,
        anneal=args.anneal, anneal_start=args.anneal_start,
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()

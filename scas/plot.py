"""Plot AC-B2 (distinct minima visited) and AC-B3 (inter vs intra distance).

Usage: uv run python -m scas.plot <paired_run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(paired_dir: Path) -> tuple[dict, dict]:
    scas = json.loads((paired_dir / "paired_scas_summary.json").read_text())
    baseline = json.loads((paired_dir / "paired_baseline_summary.json").read_text())
    return scas, baseline


def plot_ac_b3(paired_dir: Path) -> Path:
    scas, _ = _load(paired_dir)
    rows = scas["per_seed"]
    seeds = [r["seed"] for r in rows]
    inter = [r["inter_group_dist"] for r in rows]
    intra = [r["intra_group_dist"] for r in rows]

    x = np.arange(len(seeds))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, inter, w, label="inter-group dist")
    ax.bar(x + w / 2, intra, w, label="intra-group dist")
    intra_mean = float(np.mean(intra)) if intra else 0.0
    ax.axhline(1.5 * intra_mean, ls="--", color="red", label="1.5 x mean intra (AC-B3)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("embedding L2 distance")
    ratio = scas.get("inter_over_intra_aggregate")
    title = f"AC-B3: inter vs intra (aggregate inter/intra = {ratio:.3f})" if ratio else "AC-B3"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = paired_dir / "ac_b3_inter_vs_intra.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _seed_coverage_curve(seed_row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Step-function: agent_step vs cumulative distinct minima visited."""
    budget = seed_row["agent_step_budget"]
    s2c = seed_row["steps_to_coverage"]
    # we don't keep raw first_visit list per seed in the aggregate, but
    # the three coverage knots (50/80/100) give enough resolution for a step plot
    knots: list[tuple[int, int]] = [(0, 0)]
    K_prime = max(1, int(round(seed_row["distinct_minima_visited"])))  # fallback if not present
    # we actually do have K' implicitly from steps_to_coverage entries
    for frac_str in ("0.5", "0.8", "1.0"):
        v = s2c.get(frac_str)
        if v is None:
            continue
        f = float(frac_str)
        cum = int(np.ceil(f * (seed_row.get("K_prime") or K_prime)))
        knots.append((int(v), cum))
    knots.append((budget, knots[-1][1]))
    knots = sorted(set(knots))
    xs = np.array([k[0] for k in knots])
    ys = np.array([k[1] for k in knots])
    return xs, ys


def plot_ac_b2(paired_dir: Path) -> Path:
    scas, baseline = _load(paired_dir)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def _avg_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        budget = max(r["agent_step_budget"] for r in rows)
        xs = np.linspace(0, budget, 200)
        ys_all: list[np.ndarray] = []
        for r in rows:
            sx, sy = _seed_coverage_curve(r)
            # right-piecewise-constant interp at evaluation points
            idx = np.searchsorted(sx, xs, side="right") - 1
            idx = np.clip(idx, 0, len(sy) - 1)
            ys_all.append(sy[idx])
        return xs, np.mean(np.stack(ys_all), axis=0)

    sx, sy = _avg_curve(scas["per_seed"])
    bx, by = _avg_curve(baseline["per_seed"])
    ax.step(sx, sy, where="post", label="SCAS (mean)")
    ax.step(bx, by, where="post", label="independent-walkers (mean)")
    # 80% threshold line
    K_prime = scas["per_seed"][0].get("K_prime") or int(round(scas["mean_distinct_minima_visited"]))
    thresh = 0.8 * K_prime
    ax.axhline(thresh, ls="--", color="red", label=f"80% threshold ({thresh:.1f})")

    s80 = scas["mean_steps_to_80pct_coverage"]
    b80 = baseline["mean_steps_to_80pct_coverage"]
    speedup = (b80 / s80) if s80 > 0 else float("inf")
    ax.set_xlabel("agent-step")
    ax.set_ylabel("cumulative distinct minima visited (mean)")
    ax.set_title(
        f"AC-B2: time-to-80% (SCAS {s80:.0f} vs baseline {b80:.0f}; speedup = {speedup:.3f})"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = paired_dir / "ac_b2_time_to_coverage.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_ac_b4(sweep_dir: Path) -> Path:
    sweep = json.loads((sweep_dir / "sweep.json").read_text())
    rows = sweep["per_T"]
    Ts = np.array([r["T"] for r in rows], dtype=float)
    seps_agg = np.array([r["scas_inter_over_intra_aggregate"] for r in rows], dtype=float)

    # per-seed scatter
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in rows:
        ys = [v for v in r["scas_per_seed_inter_over_intra"] if v is not None]
        xs = [r["T"]] * len(ys)
        ax.scatter(xs, ys, alpha=0.4, color="tab:blue", s=20)
    ax.plot(Ts, seps_agg, "o-", color="tab:blue", label="aggregate (mean_inter/mean_intra)")
    ax.set_xscale("log")
    ax.set_xlabel("coupling_T  (log scale; tight ← → loose)")
    ax.set_ylabel("inter / intra group embedding distance")
    ax.axhline(1.5, ls="--", color="red", label="AC-B3 threshold (1.5)")
    tau = sweep["kendall_tau_T_vs_inter_over_intra_agg"]
    tau_flat = sweep["kendall_tau_T_vs_inter_over_intra_flat"]
    ax.set_title(
        f"AC-B4: knob monotonicity (Kendall τ = {tau:.3f} aggregate, "
        f"{tau_flat:.3f} per-seed flat; target |τ| ≥ 0.7)"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = sweep_dir / "ac_b4_knob_monotonicity.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m scas.plot <paired_or_sweep_dir>", file=sys.stderr)
        sys.exit(2)
    d = Path(sys.argv[1])
    if not d.is_dir():
        cand = Path("sim_runs") / d.name
        if cand.is_dir():
            d = cand
        else:
            print(f"not a directory: {d}", file=sys.stderr)
            sys.exit(2)
    if (d / "sweep.json").exists():
        out = plot_ac_b4(d)
        print(f"wrote: {out}")
        tau = json.loads((d / "sweep.json").read_text())["kendall_tau_T_vs_inter_over_intra_agg"]
        tau_flat = json.loads((d / "sweep.json").read_text())["kendall_tau_T_vs_inter_over_intra_flat"]
        print(f"Kendall tau (aggregate): {tau:.4f}")
        print(f"Kendall tau (per-seed flat): {tau_flat:.4f}")
    else:
        out1 = plot_ac_b3(d)
        out2 = plot_ac_b2(d)
        print(f"wrote: {out1}\nwrote: {out2}")


if __name__ == "__main__":
    main()

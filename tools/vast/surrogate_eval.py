"""Fit the v3 model on the experiment memory and test it on runs it has not seen.

    uv run python tools/vast/surrogate_eval.py --fit     # write scas/data/v3_surrogate.json
    uv run python tools/vast/surrogate_eval.py           # leave-one-run-out tests

The tests replay each run in time order. Kernel parameters are fitted on the
other runs only. For every experiment the model sees what its arm had seen
when the experiment started, with or without the other runs as prior memory:

* error of the predicted val_bpb, against "the experiment changes nothing";
* how well the predicted chance of improvement separates kept from discarded
  experiments (AUC), and what refusing the least promising ones would have cost;
* on the re-measurements the v2 coordinator made: its ratio estimate against
  the model's prediction from the same information.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scas.surrogate import (DEFAULT_PARAMS, FEATURES, MEMORY_FILE, PARAMS_FILE,  # noqa: E402
                            Surrogate, fit_step_model, load_memory)


def fit(rows: list[dict], maxiter: int) -> dict:
    p = dict(DEFAULT_PARAMS)
    p["step_batch"], p["step_depth"] = fit_step_model([r for r in rows if not r["flags"]])
    s = Surrogate(p)
    s.add_memory(rows)
    return s.fit_hyperparameters(maxiter=maxiter)


def describe(p: dict) -> str:
    order = np.argsort(p["ls"])[:5]
    return ("noise sd %.2f%%, tuning constants sd %.2f%%, one structural change sd %.2f%%, "
            "steps ~ batch^%.2f; shortest length-scales: %s, box speed %.2f" % (
                100 * p["sn"], 100 * p["sf_tuning"], 100 * p["ss"], p["step_batch"] / math.log(2),
                ", ".join("%s %.2f" % (FEATURES[i], p["ls"][i]) for i in order), p["ls_speed"]))


def auc(score, label) -> float:
    s, l = np.asarray(score), np.asarray(label)
    pos, neg = s[l == 1], s[l == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def rmse(e) -> float:
    return 100 * float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def mae(e) -> float:
    return 100 * float(np.mean(np.abs(e))) if len(e) else float("nan")


def predict_row(params, known, r, track):
    s = Surrogate(params, track_speed=track)
    s.add_memory(known)
    box = r["run"] + "/" + r["box"]
    sp = s.box_speed(box)
    mu, cov = s.predict([(r["consts"], r["flags"], params["speed_mean"] if sp is None else sp)])
    return float(mu[0]), math.sqrt(max(cov[0, 0], 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memory", default=str(MEMORY_FILE))
    ap.add_argument("--fit", action="store_true", help="fit on everything and save the parameters")
    ap.add_argument("--maxiter", type=int, default=120)
    args = ap.parse_args()
    rows = load_memory(args.memory)
    runs = sorted({r["run"] for r in rows})
    print("%d experiments in %d runs" % (len(rows), len(runs)))

    if args.fit:
        p = fit(rows, args.maxiter)
        PARAMS_FILE.write_text(json.dumps(p, indent=1), encoding="utf-8", newline="\n")
        print("fitted on all runs: " + describe(p))
        print("wrote %s" % PARAMS_FILE)
        return

    VARIANTS = [("arm only", False, True), ("arm + other runs", True, True),
                ("arm + other runs, speed not tracked", True, False)]
    pooled = defaultdict(lambda: defaultdict(list))
    v2e = defaultdict(list)
    for run in runs:
        others = [r for r in rows if r["run"] != run]
        params = fit(others, args.maxiter)
        print("\n== run %s (parameters from the other runs: %s)" % (run, describe(params)))
        mine = [r for r in rows if r["run"] == run]
        for arm in sorted({r["arm"] for r in mine}):
            arm_rows = [r for r in mine if r["arm"] == arm]
            T = defaultdict(list)
            for r in arm_rows:
                known = [q for q in arm_rows if q["t"] <= r["t_start"]]
                if r["v2_expected"]:
                    v2e["v2 ratio estimate"].append(math.log(r["v2_expected"] / r["val"]))
                    for name, prior, track in VARIANTS:
                        mu, _ = predict_row(params, (others if prior else []) + known, r, track)
                        v2e["model, " + name].append(mu - math.log(r["val"]))
                if r["n"] == 0 or r["kind"] == "verify" or not r["ref"] or len(known) < 3:
                    continue
                T["kept"].append(int(r["status"] == "keep"))
                T["nochange"].append(math.log(r["ref"] / r["val"]))
                for name, prior, track in VARIANTS:
                    mu, sd = predict_row(params, (others if prior else []) + known, r, track)
                    T[name].append(mu - math.log(r["val"]))
                    T["p:" + name].append(float(norm.cdf((math.log(r["ref"]) - mu) / max(sd, 1e-6))))
            if not T["kept"]:
                continue
            print("  %-12s %3d experiments, %2d kept | error if nothing changed %.2f%%" % (
                arm, len(T["kept"]), sum(T["kept"]), rmse(T["nochange"])))
            for name, _, _ in VARIANTS:
                print("      %-38s error %.2f%%  keep/discard AUC %.2f" % (
                    name, rmse(T[name]), auc(T["p:" + name], T["kept"])))
            for k, v in T.items():
                pooled[k][run] += v

    print("\n== the v2 coordinator's own re-measurement estimates (%d cases) ==" % len(v2e["v2 ratio estimate"]))
    for k, e in v2e.items():
        print("  %-42s mean abs error %.2f%%  bias %+.2f%%" % (k, mae(e), 100 * float(np.mean(e))))

    print("\n== pooled over all runs ==")
    kept = np.concatenate([np.asarray(v) for v in pooled["kept"].values()])
    small = np.abs(np.concatenate([np.asarray(v) for v in pooled["nochange"].values()])) < 0.03
    for name, _, _ in VARIANTS:
        p = np.concatenate([np.asarray(v) for v in pooled["p:" + name].values()])
        e = np.concatenate([np.asarray(v) for v in pooled[name].values()])
        print("  %s: error %.2f%%, AUC %.2f (%d experiments, %d kept); small changes only: AUC %.2f"
              % (name, rmse(e), auc(p, kept), len(kept), kept.sum(), auc(p[small], kept[small])))
        for thr in (0.02, 0.05, 0.10, 0.20):
            skip = p < thr
            print("      refuse when chance of improvement < %.2f: refuses %3d, of which %2d were keeps"
                  % (thr, skip.sum(), kept[skip].sum()))
        gains = 100 * np.concatenate([np.asarray(v) for v in pooled["nochange"].values()])
        runs_of = np.concatenate([[r] * len(v) for r, v in pooled["kept"].items()])
        lost = (p < 0.05) & (kept == 1)
        print("      keeps a 0.05 threshold would have refused, as (run, measured gain %%): %s"
              % ", ".join("%s %+.2f" % (r[-7:-1], g) for r, g in zip(runs_of[lost], gains[lost])))


if __name__ == "__main__":
    main()

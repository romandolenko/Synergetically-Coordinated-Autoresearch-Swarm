"""Noise-aware model of val_bpb over explicit settings (the "filter" of SCAS v3).

The state of an experiment is not an embedding of its description. It is

* the named constants of train.py (log-scaled where that is natural),
* a set of flags, one per structural change the code carries, and
* the speed of the box it ran on, tracked from the optimizer steps each run
  reports.

A Gaussian process over that state (the spatial form of the Kolmogorov-Wiener
filter) gives, for any setting on any box, an expected val_bpb and how sure it
is. The offline test on the 461 experiments of the three GPU runs of November
2026 is what this module is built from:

* optimizer steps alone explain 94% of the variance of log val_bpb, and steps
  follow from batch size, depth and box speed (R^2 0.996) - hence box speed is
  an input, and the per-box offset the v2 ratio estimates relied on vanishes;
* predicting a version on a box that has not run it: 0.9% mean error against
  1.8% for the v2 ratio estimates on the same 14 cases;
* two of ten boxes drifted 2-4% within hours, so speed is tracked per run.

Box speed is the log of the steps a box makes on the baseline code. A run of
other code tells the box's speed only once it is known how many steps that
code makes relative to the baseline (its "step class": batch size, model
shape, structural flags). The first run of a class fixes that offset from the
box's current speed; later runs of the class then update the speed.

Pure numpy/scipy, no I/O except ``load_memory``/``load_params``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

DATA_DIR = Path(__file__).resolve().parent / "data"
MEMORY_FILE = DATA_DIR / "v3_memory.jsonl"
PARAMS_FILE = DATA_DIR / "v3_surrogate.json"

CONSTS = ("ASPECT_RATIO", "HEAD_DIM", "TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR",
          "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS", "WARMUP_RATIO",
          "WARMDOWN_RATIO", "FINAL_LR_FRAC", "DEPTH", "DEVICE_BATCH_SIZE")
_CRE = re.compile(r"^(%s)(\s*=\s*)([^#\n]*?)(\s*(?:#.*)?)$" % "|".join(CONSTS))

FEATURES = ("batch", "emb_lr", "unemb_lr", "matrix_lr", "scalar_lr", "wd", "beta1", "beta2",
            "warmup", "warmdown", "final_lr", "depth", "head_dim", "aspect")
# Settings that decide how much training fits in the budget (with box speed) move
# val_bpb by tens of percent; the tuning constants move it by a few. One kernel
# with one amplitude would expect +/-10% from any untested learning rate, so
# each group has its own. Batch size is also in the tuning kernel: the best
# learning rate may depend on it.
MAIN = [FEATURES.index(n) for n in ("batch", "depth", "head_dim", "aspect")]
TUNING = [i for i in range(len(FEATURES)) if i not in MAIN]

# log(steps) = a + STEP_BATCH * log2(TOTAL_BATCH_SIZE) + STEP_DEPTH * DEPTH + box speed.
# Fitted on the seed memory by tools/vast/surrogate_eval.py and stored with the
# other parameters; these are the fallbacks.
STEP_BATCH = -0.66
STEP_DEPTH = -0.10


def extract(code: str) -> tuple[dict, str]:
    """The numeric settings of a train.py, and a hash of everything else in it."""
    consts, rest = {}, []
    for ln in code.replace("\r\n", "\n").split("\n"):
        m = _CRE.match(ln)
        v = None
        if m:
            try:
                v = eval(m.group(3), {"__builtins__": {}})  # noqa: S307 - literals only
            except Exception:  # noqa: BLE001
                v = None
        if m and m.group(1) == "ADAM_BETAS" and isinstance(v, tuple) and len(v) == 2:
            consts["BETA1"], consts["BETA2"] = float(v[0]), float(v[1])
        elif m and isinstance(v, (int, float)) and not isinstance(v, bool):
            consts[m.group(1)] = float(v)
        else:
            rest.append(ln.rstrip())
    return consts, hashlib.sha256("\n".join(rest).encode("utf-8")).hexdigest()[:10]


def complete(consts: dict) -> bool:
    need = set(CONSTS) - {"ADAM_BETAS"} | {"BETA1", "BETA2"}
    return need <= set(consts)


def set_constant(code: str, name: str, value) -> str | None:
    """train.py with one constant set to ``value`` (BETA1/BETA2 edit ADAM_BETAS)."""
    consts, _ = extract(code)
    if name in ("BETA1", "BETA2"):
        b1 = value if name == "BETA1" else consts.get("BETA1")
        b2 = value if name == "BETA2" else consts.get("BETA2")
        if b1 is None or b2 is None:
            return None
        target, text = "ADAM_BETAS", "(%s, %s)" % (_num(b1), _num(b2))
    else:
        target, text = name, _num(value)
    out, done = [], False
    for ln in code.replace("\r\n", "\n").split("\n"):
        m = _CRE.match(ln)
        if m and m.group(1) == target and not done:
            ln = "%s%s%s%s" % (m.group(1), m.group(2), text, m.group(4))
            done = True
        out.append(ln)
    return "\n".join(out) if done else None


def _num(v) -> str:
    v = float(v)
    if v == int(v) and abs(v) >= 256 and math.log2(v) == int(math.log2(v)):
        return "2 ** %d" % int(math.log2(v))
    if v == int(v) and abs(v) >= 1:
        return str(int(v))
    return repr(float("%.4g" % v))


def _lg(x: float) -> float:
    return math.log(max(x, 1e-6))


def featurize(c: dict) -> list[float]:
    return [math.log2(c["TOTAL_BATCH_SIZE"]), _lg(c["EMBEDDING_LR"]), _lg(c["UNEMBEDDING_LR"]),
            _lg(c["MATRIX_LR"]), _lg(c["SCALAR_LR"]), c["WEIGHT_DECAY"], -_lg(1 - c["BETA1"]),
            -_lg(1 - c["BETA2"]), c["WARMUP_RATIO"], c["WARMDOWN_RATIO"], c["FINAL_LR_FRAC"],
            c["DEPTH"], math.log2(c["HEAD_DIM"]), math.log2(c["ASPECT_RATIO"])]


DEFAULT_PARAMS = {
    # standardisation of the features, and of box speed
    "mean": [16.5, -0.5, -5.5, -3.1, -0.8, 0.18, 1.7, 3.1, 0.01, 0.5, 0.01, 7.8, 6.9, 6.0],
    "std": [1.6, 0.3, 0.3, 0.25, 0.3, 0.06, 0.2, 0.4, 0.02, 0.1, 0.03, 0.6, 0.3, 0.2],
    "speed_mean": 4.3, "speed_std": 0.25,
    # kernel: length-scales per feature and for speed, signal, per-flag effect, noise
    "ls": [1.0, 2.0, 2.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0],
    "ls_speed": 0.8, "ls_batch_tuning": 3.0, "sf": 0.25, "sf_tuning": 0.03,
    "ss": 0.03, "sn": 0.005,
    "step_batch": STEP_BATCH, "step_depth": STEP_DEPTH,
}


def load_params(path: str | Path | None = None) -> dict:
    p = Path(path) if path else PARAMS_FILE
    if p.exists():
        return {**DEFAULT_PARAMS, **json.loads(p.read_text(encoding="utf-8"))}
    return dict(DEFAULT_PARAMS)


def load_memory(path: str | Path | None = None, exclude_runs: tuple = ()) -> list[dict]:
    p = Path(path) if path else MEMORY_FILE
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted((r for r in rows if r["run"] not in exclude_runs), key=lambda r: r["t"])


class Surrogate:
    """Gaussian process on log val_bpb.

    kernel = sf^2 * RBF(batch size, depth, head dim, aspect ratio, box speed)
           + sf_tuning^2 * RBF(the other constants, batch size)
           + ss^2 * (number of structural flags two experiments share)
           + sn^2 noise
    """

    SPEED_WINDOW = 3

    def __init__(self, params: dict | None = None, track_speed: bool = True):
        self.p = dict(params or load_params())
        self.track_speed = track_speed
        self.lock = threading.RLock()
        self.X: list[list[float]] = []       # raw features
        self.S: list[float] = []             # raw speed of each observation
        self.F: list[frozenset] = []
        self.y: list[float] = []
        self.box_speeds: dict[str, list[float]] = {}
        self.class_offset: dict[tuple, list[float]] = {}   # step class -> log steps - box speed
        self._chol = None

    # ------------------------------------------------------------- box speed

    @staticmethod
    def step_class(consts: dict, flags) -> tuple:
        return (consts["TOTAL_BATCH_SIZE"], consts["DEPTH"], consts["ASPECT_RATIO"],
                consts["HEAD_DIM"], tuple(sorted(flags)))

    def _step_guess(self, consts: dict) -> float:
        """Offset of a class nobody has run, from the fitted step model."""
        return (self.p["step_batch"] * (math.log2(consts["TOTAL_BATCH_SIZE"]) - 19)
                + self.p["step_depth"] * (consts["DEPTH"] - 8))

    def _observe_steps(self, consts: dict, flags, box: str, steps: float) -> None:
        ls = math.log(steps)
        key = self.step_class(consts, flags)
        hist = self.box_speeds.setdefault(box, [])
        known = self.class_offset.get(key)
        if known:
            hist.append(ls - float(np.median(known)))      # a known class: it measures the box
        elif not hist:
            if not self.class_offset:
                self.class_offset[key] = [0.0]             # the very first run defines speed
                hist.append(ls)
            else:
                hist.append(ls - self._step_guess(consts))
            return
        if not known or len(known) < 5:
            # what this class costs in steps, judged by what the box was doing just before
            before = hist[:-1] if known else hist
            if before:
                ref = before[0] if not self.track_speed else float(np.median(before[-self.SPEED_WINDOW:]))
                self.class_offset.setdefault(key, []).append(ls - ref)

    def box_speed(self, box: str) -> float | None:
        h = self.box_speeds.get(box)
        if not h:
            return None
        if not self.track_speed:
            return h[0]
        return float(np.median(h[-self.SPEED_WINDOW:]))

    # ------------------------------------------------------------------ data

    def add(self, consts: dict, flags, box: str, val: float, steps: float | None) -> None:
        with self.lock:
            if steps:
                self._observe_steps(consts, frozenset(flags), box, steps)
            s = self.box_speed(box)
            if s is None:
                s = self.p["speed_mean"]
            self.X.append(featurize(consts))
            self.S.append(s)
            self.F.append(frozenset(flags))
            self.y.append(math.log(val))
            self._chol = None

    def add_memory(self, rows: list[dict]) -> None:
        for r in rows:
            self.add(r["consts"], r["flags"], r["run"] + "/" + r["box"], r["val"], r["steps"])

    def __len__(self) -> int:
        return len(self.y)

    # ---------------------------------------------------------------- kernel

    def _z(self, X, S) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        N = (np.asarray(X, float).reshape(-1, len(FEATURES)) - p["mean"]) / p["std"]
        Z = N / p["ls"]
        s = (np.asarray(S, float).reshape(-1, 1) - p["speed_mean"]) / p["speed_std"] / p["ls_speed"]
        return (np.hstack([Z[:, MAIN], s]),
                np.hstack([Z[:, TUNING], N[:, :1] / p["ls_batch_tuning"]]))

    @staticmethod
    def _rbf(A, B) -> np.ndarray:
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-0.5 * np.maximum(d2, 0.0))

    def _k(self, Za, Fa, Zb, Fb) -> np.ndarray:
        K = (self.p["sf"] ** 2 * self._rbf(Za[0], Zb[0])
             + self.p["sf_tuning"] ** 2 * self._rbf(Za[1], Zb[1]))
        if any(Fa) and any(Fb):
            K = K + self.p["ss"] ** 2 * np.array([[len(a & b) for b in Fb] for a in Fa], float)
        return K

    def _fit_state(self):
        if self._chol is None:
            Z = self._z(self.X, self.S)
            y = np.asarray(self.y)
            m = float(y.mean())
            K = self._k(Z, self.F, Z, self.F) + (self.p["sn"] ** 2 + 1e-8) * np.eye(len(y))
            c = cho_factor(K, lower=True)
            self._chol = (Z, m, c, cho_solve(c, y - m))
        return self._chol

    # --------------------------------------------------------------- predict

    def predict(self, points: list[tuple[dict, object, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance of log val_bpb (noise-free) at
        ``points`` = [(consts, flags, speed), ...]."""
        with self.lock:
            Zs = self._z([featurize(c) for c, _, _ in points], [s for _, _, s in points])
            Fs = [frozenset(f) for _, f, _ in points]
            Kss = self._k(Zs, Fs, Zs, Fs)
            if not self.y:
                return np.full(len(points), math.log(0.6)), Kss
            Z, m, c, alpha = self._fit_state()
            Ks = self._k(Zs, Fs, Z, self.F)
            return m + Ks @ alpha, Kss - Ks @ cho_solve(c, Ks.T)

    def expected(self, consts: dict, flags, box: str) -> tuple[float, float]:
        """Expected val_bpb of a setting on a box, and its relative sd."""
        s = self.box_speed(box)
        mu, cov = self.predict([(consts, flags, self.p["speed_mean"] if s is None else s)])
        return math.exp(mu[0]), math.sqrt(max(cov[0, 0], 0.0))

    def compare(self, new: tuple[dict, object], base: tuple[dict, object], box: str) -> dict:
        """Is ``new`` better than ``base`` on ``box``? Uses their joint posterior,
        so what the two share (box speed, common flags) cancels."""
        s = self.box_speed(box)
        s = self.p["speed_mean"] if s is None else s
        mu, cov = self.predict([(new[0], new[1], s), (base[0], base[1], s)])
        d = mu[1] - mu[0]                                   # > 0: new is better
        sd = math.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-12))
        return {"p_better": float(norm.cdf(d / sd)), "gain_pct": 100.0 * (1.0 - math.exp(-d)),
                "sd_pct": 100.0 * sd, "base_val": math.exp(mu[1]), "new_val": math.exp(mu[0])}

    # ------------------------------------------------------- hyperparameters

    def fit_hyperparameters(self, maxiter: int = 200) -> dict:
        """Maximum-likelihood kernel parameters and step model from the data held.
        Offline use (takes a minute or two on a few hundred experiments)."""
        X, y = np.asarray(self.X), np.asarray(self.y)
        p = self.p
        sd = X.std(0)
        p["mean"], p["std"] = X.mean(0).tolist(), np.where(sd > 1e-9, sd, 1.0).tolist()
        S = np.asarray(self.S)
        p["speed_mean"], p["speed_std"] = float(S.mean()), float(max(S.std(), 1e-3))
        d = len(FEATURES)
        yy = y - y.mean()
        Xn = (X - p["mean"]) / p["std"]
        Sn = ((S - p["speed_mean"]) / p["speed_std"])[:, None]
        shared = np.array([[len(a & b) for b in self.F] for a in self.F], float)
        eye = np.eye(len(yy))

        # theta: log length-scales (features, speed, batch-in-tuning), log sf, sf_tuning, ss, sn
        def nll(th):
            ls = np.exp(th[:d])
            A = np.hstack([Xn[:, MAIN] / ls[MAIN], Sn / math.exp(th[d])])
            B = np.hstack([Xn[:, TUNING] / ls[TUNING], Xn[:, :1] / math.exp(th[d + 1])])
            K = (math.exp(2 * th[d + 2]) * self._rbf(A, A) + math.exp(2 * th[d + 3]) * self._rbf(B, B)
                 + math.exp(2 * th[d + 4]) * shared + (math.exp(2 * th[d + 5]) + 1e-8) * eye)
            try:
                c = cho_factor(K, lower=True)
            except np.linalg.LinAlgError:
                return 1e10
            return 0.5 * yy @ cho_solve(c, yy) + np.log(np.diag(c[0])).sum()

        th0 = np.concatenate([np.full(d + 2, 0.7), np.log([0.2, 0.03, 0.02, 0.005])])
        bounds = ([(-2.5, 4.0) if v > 1e-9 else (3.0, 3.0) for v in sd] + [(-2.5, 4.0)] * 2
                  + [(-5, 1), (-7, -1.5), (-8, 0), (-8, -2)])
        r = minimize(nll, th0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        p["ls"] = np.exp(r.x[:d]).tolist()
        p["ls_speed"], p["ls_batch_tuning"] = float(math.exp(r.x[d])), float(math.exp(r.x[d + 1]))
        p["sf"], p["sf_tuning"], p["ss"], p["sn"] = (float(math.exp(v)) for v in r.x[d + 2:d + 6])
        self._chol = None
        return p


def fit_step_model(rows: list[dict]) -> tuple[float, float]:
    """Slopes of log(steps) on log2(batch size) and depth, with one intercept per box."""
    boxes = sorted({r["run"] + "/" + r["box"] for r in rows})
    A = np.zeros((len(rows), 2 + len(boxes)))
    for i, r in enumerate(rows):
        A[i, 0] = math.log2(r["consts"]["TOTAL_BATCH_SIZE"]) - 19
        A[i, 1] = r["consts"]["DEPTH"] - 8
        A[i, 2 + boxes.index(r["run"] + "/" + r["box"])] = 1.0
    coef, *_ = np.linalg.lstsq(A, np.log([r["steps"] for r in rows]), rcond=None)
    return float(coef[0]), float(coef[1])

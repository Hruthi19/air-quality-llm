"""Improved Sparrow Search Algorithm (ISSA) for LSTM hyper-parameter tuning.

The Sparrow Search Algorithm (Xue & Shen, 2020) is a swarm-based meta-
heuristic. Wu et al. (2023) augment it with two ingredients we keep:

    1. Levy-flight perturbation for the discoverer ("producer") class to
       escape shallow optima in the early phase of search.
    2. Adaptive position update — explorers' step size shrinks over time
       so the swarm transitions from exploration to exploitation.

Search space (all integer / log-uniform):

    hidden_size   in [32, 128]
    num_layers    in {1, 2, 3}
    dropout       in [0.0, 0.5]
    learning_rate in [1e-4, 5e-3] (log-uniform)
    batch_size    in {32, 64, 128}

Fitness = validation MSE (lower is better) on a small budget (1-2 epochs).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ISSAConfig:
    n_sparrows:     int   = 8
    n_iter:         int   = 6
    pd_ratio:       float = 0.20    # producer fraction
    sd_ratio:       float = 0.20    # scout (warning) fraction
    safety_thr:     float = 0.6
    levy_alpha:     float = 1.5
    seed:           int   = 42

    def to_dict(self) -> dict:
        return asdict(self)


SEARCH_SPACE = {
    "hidden_size":   ("int",  32,   128),
    "num_layers":    ("int",  1,    3),
    "dropout":       ("float", 0.0, 0.5),
    "learning_rate": ("logf",  1e-4, 5e-3),
    "batch_size":    ("choice", [32, 64, 128]),
}


def _sample_random(rng: random.Random) -> dict:
    s = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "int":
            s[name] = rng.randint(spec[1], spec[2])
        elif kind == "float":
            s[name] = rng.uniform(spec[1], spec[2])
        elif kind == "logf":
            lo, hi = math.log(spec[1]), math.log(spec[2])
            s[name] = math.exp(rng.uniform(lo, hi))
        elif kind == "choice":
            s[name] = rng.choice(spec[1])
    return s


def _project(v: dict) -> dict:
    """Clip a candidate back into the search space."""
    out = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        x = v.get(name)
        if kind == "int":
            out[name] = int(np.clip(round(x), spec[1], spec[2]))
        elif kind == "float":
            out[name] = float(np.clip(x, spec[1], spec[2]))
        elif kind == "logf":
            out[name] = float(np.clip(x, spec[1], spec[2]))
        elif kind == "choice":
            choices = spec[1]
            out[name] = min(choices, key=lambda c: abs(c - x))
    return out


def _to_vec(v: dict) -> np.ndarray:
    return np.array([
        float(v["hidden_size"]),
        float(v["num_layers"]),
        float(v["dropout"]),
        float(math.log(max(v["learning_rate"], 1e-12))),
        float(v["batch_size"]),
    ])


def _from_vec(arr: np.ndarray) -> dict:
    # Replace any non-finite component (NaN/Inf produced by the swarm
    # updates — exp of a large positive number, division by tiny denom, ...)
    # with the midpoint of the corresponding search-space range so _project
    # can finish the projection deterministically.
    arr = np.where(np.isfinite(arr), arr, 0.0)
    # Clamp log-learning-rate to a wide-but-safe band before exp; the final
    # _project still enforces the [1e-4, 5e-3] product space.
    log_lr_lo = math.log(1e-8)        # ≈ -18.42
    log_lr_hi = math.log(1.0)         # 0
    log_lr = float(np.clip(arr[3], log_lr_lo, log_lr_hi))
    return _project({
        "hidden_size":   arr[0],
        "num_layers":    arr[1],
        "dropout":       arr[2],
        "learning_rate": math.exp(log_lr),
        "batch_size":    arr[4],
    })


def _levy(alpha: float, size: int, rng: np.random.RandomState) -> np.ndarray:
    """Mantegna-style Levy flight step (alpha in (1, 2))."""
    sigma = (math.gamma(1 + alpha) * math.sin(math.pi * alpha / 2) /
             (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
    u = rng.normal(0.0, sigma, size=size)
    v = rng.normal(0.0, 1.0,   size=size)
    return u / (np.abs(v) ** (1 / alpha) + 1e-12)


def issa_search(fitness_fn: Callable[[dict], float],
                cfg: ISSAConfig | None = None,
                log_path: Path | None = None) -> dict:
    """Run ISSA over the LSTM hyper-parameter space and return the best candidate."""
    cfg = cfg or ISSAConfig()
    rng_py = random.Random(cfg.seed)
    rng_np = np.random.RandomState(cfg.seed)

    # ----- initialisation -----
    pop = [_sample_random(rng_py) for _ in range(cfg.n_sparrows)]
    fits = [float(fitness_fn(p)) for p in pop]
    history = []
    for i, (p, f) in enumerate(zip(pop, fits)):
        rec = {"iter": 0, "sparrow": i, "role": "init", "fitness": f, **p}
        history.append(rec)

    n_pd = max(1, int(cfg.pd_ratio * cfg.n_sparrows))
    n_sd = max(1, int(cfg.sd_ratio * cfg.n_sparrows))

    for it in range(1, cfg.n_iter + 1):
        order      = np.argsort(fits)            # ascending — best first
        producers  = order[:n_pd].tolist()
        scroungers = order[n_pd:].tolist()
        scouts     = rng_np.choice(cfg.n_sparrows, size=n_sd, replace=False).tolist()

        new_pop = [dict(p) for p in pop]
        new_fits = list(fits)
        best_idx  = int(order[0])
        worst_idx = int(order[-1])

        # --- producers: Levy flight + decaying step ---
        decay = math.exp(- (it / cfg.n_iter) ** 2)
        for idx in producers:
            v = _to_vec(pop[idx])
            warning  = rng_np.uniform(0.0, 1.0)
            if warning < cfg.safety_thr:
                step = decay * _levy(cfg.levy_alpha, len(v), rng_np)
                # Clip the Levy step so a heavy-tailed sample cannot send
                # any component to +/-Inf; downstream _from_vec still
                # clamps and projects.
                step = np.clip(step, -10.0, 10.0)
                v_new = v * np.exp(- (it / cfg.n_iter)) + step
            else:
                v_new = v + rng_np.normal(0.0, 1.0, size=len(v))
            cand = _from_vec(v_new)
            f = float(fitness_fn(cand))
            if f < new_fits[idx]:
                new_pop[idx] = cand
                new_fits[idx] = f
            history.append({"iter": it, "sparrow": idx, "role": "producer",
                            "fitness": f, **cand})

        # --- scroungers: chase the best producer or random walk ---
        best_v = _to_vec(pop[best_idx])
        worst_v = _to_vec(pop[worst_idx])
        for rank, idx in enumerate(scroungers):
            v = _to_vec(pop[idx])
            if rank > cfg.n_sparrows / 2:
                step = rng_np.normal(0.0, 1.0, size=len(v))
                # The exp-of-difference term can blow up when |worst_v - v|
                # is large (e.g. log-LR coordinates), so clip the exponent
                # before exponentiating.
                expo = np.clip((worst_v - v) / max(1, it ** 2), -10.0, 10.0)
                v_new = step * np.exp(expo)
            else:
                A = rng_np.choice([-1.0, 1.0], size=len(v))
                v_new = best_v + np.abs(v - best_v) * A
            cand = _from_vec(v_new)
            f = float(fitness_fn(cand))
            if f < new_fits[idx]:
                new_pop[idx] = cand
                new_fits[idx] = f
            history.append({"iter": it, "sparrow": idx, "role": "scrounger",
                            "fitness": f, **cand})

        # --- scouts: anti-predator behaviour ---
        for idx in scouts:
            v = _to_vec(pop[idx])
            beta = rng_np.normal(0.0, 1.0)
            if new_fits[idx] > new_fits[best_idx]:
                v_new = best_v + beta * np.abs(v - best_v)
            else:
                K = rng_np.uniform(-1.0, 1.0)
                denom = (new_fits[idx] - new_fits[worst_idx]) + 1e-9
                v_new = v + K * np.abs(v - worst_v) / denom
            cand = _from_vec(v_new)
            f = float(fitness_fn(cand))
            if f < new_fits[idx]:
                new_pop[idx] = cand
                new_fits[idx] = f
            history.append({"iter": it, "sparrow": idx, "role": "scout",
                            "fitness": f, **cand})

        pop, fits = new_pop, new_fits
        print(f"[issa] iter {it}/{cfg.n_iter}  best_fitness={min(fits):.6f}")

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(log_path, index=False)

    best = int(np.argmin(fits))
    return {
        "best_params":    pop[best],
        "best_fitness":   float(fits[best]),
        "all_population": pop,
        "all_fitness":    fits,
        "history":        history,
        "config":         cfg.to_dict(),
    }

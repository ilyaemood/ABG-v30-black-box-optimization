#!/usr/bin/env python3
"""ABG-v30: an evaluation-budgeted adaptive box-subdivision optimizer.

This module contains the optimizer, its budgeted objective oracle, baseline
implementations, and self-tests. The released 8-configuration, 33-family
comparison that generated the archived publication package is run by
``benchmark_top8_20plus.py``.

Useful checks:
    python ABG_research_v30.py --self-test
    python ABG_research_v30.py --smoke

Dependencies: NumPy, SciPy, and Matplotlib. The bundled CMA-ES baseline is a
compact local implementation; see PAPER.md for benchmark limitations.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import platform
import time
import warnings
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.stats import qmc

try:
    from scipy.optimize import direct as scipy_direct
except ImportError:  # pragma: no cover - depends on SciPy version
    scipy_direct = None

DEFAULT_SEED = 20261018
DOMAIN_LO = -1.0
DOMAIN_HI = 1.0
SUCCESS_TOLERANCES = (1e-3, 1e-6)
FAILED_GAP_PENALTY = 1e30  # invalid/no-incumbent runs must rank worst, never as perfect
ADAM_LR_CANDIDATES = (0.005, 0.01, 0.03, 0.06, 0.1, 0.2)
class BudgetExhausted(RuntimeError):
    """Raised before an objective calculation would exceed the budget."""


class LocalStartBudget(BudgetExhausted):
    """Signals that one local restart used its allocation; the next start may run."""


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    scale: float
    f_star: float
    fn: Callable[[np.ndarray], float]
    description: str
    metric_scale: float = 1.0

    def evaluate(self, x: np.ndarray) -> float:
        z = self.scale * np.asarray(x, dtype=float)
        return float(self.fn(z))


def _rastrigin(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    return float(10.0 * z.size + np.sum(z*z - 10.0*np.cos(2.0*np.pi*z)))


def _ackley(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    d = z.size
    return float(-20.0*np.exp(-0.2*np.sqrt(np.sum(z*z)/d))
                 - np.exp(np.sum(np.cos(2.0*np.pi*z))/d) + 20.0 + np.e)


def _rosenbrock(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    if z.size < 2:
        return float((1.0-z[0])**2)
    return float(np.sum(100.0*(z[1:] - z[:-1]**2)**2 + (1.0-z[:-1])**2))


def _levy(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    w = 1.0 + (z-1.0)/4.0
    term1 = np.sin(np.pi*w[0])**2
    term3 = (w[-1]-1.0)**2 * (1.0 + np.sin(2.0*np.pi*w[-1])**2)
    if z.size == 1:
        return float(term1 + term3)
    wi = w[:-1]
    term2 = np.sum((wi-1.0)**2 * (1.0 + 10.0*np.sin(np.pi*wi+1.0)**2))
    return float(term1 + term2 + term3)


def _griewank(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    i = np.arange(1, z.size+1, dtype=float)
    return float(np.sum(z*z)/4000.0 - np.prod(np.cos(z/np.sqrt(i))) + 1.0)


def _schwefel(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    return float(418.9828872724338*z.size - np.sum(z*np.sin(np.sqrt(np.abs(z)))))


def _styblinski_tang(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    return float(0.5*np.sum(z**4 - 16.0*z*z + 5.0*z))


def _sphere(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    return float(np.dot(z, z))


def _himmelblau(z: np.ndarray) -> float:
    x, y = z
    return float((x*x+y-11.0)**2 + (x+y*y-7.0)**2)


def _modified_himmelblau(z: np.ndarray) -> float:
    x, y = z
    return float(((x*x+y-11.0)**2 + (x+y*y-7.0)**2)/10.0 + 0.05*(x*x+y*y))


def _build_problem_uncalibrated(name: str, dim: int) -> Problem:
    """Build one test problem in normalized coordinates x in [-1,1]^dim."""
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if name in ("Himmelblau", "Modified_Himmelblau") and dim != 2:
        raise ValueError(f"{name} is a 2D-only benchmark; requested dim={dim}")

    if name == "Sphere":
        return Problem(name, dim, 5.0, 0.0, _sphere, "Convex quadratic; known global minimum")
    if name == "Rastrigin":
        return Problem(name, dim, 5.12, 0.0, _rastrigin, "Highly multimodal, separable periodic landscape")
    if name == "Ackley":
        return Problem(name, dim, 5.0, 0.0, _ackley, "Flat central basin and rugged outer region")
    if name == "Rosenbrock":
        return Problem(name, dim, 2.0, 0.0, _rosenbrock, "Narrow curved valley")
    if name == "Levy":
        return Problem(name, dim, 10.0, 0.0, _levy, "Multimodal oscillatory landscape")
    if name == "Griewank":
        return Problem(name, dim, 600.0, 0.0, _griewank, "Many local minima across a broad domain")
    if name == "Schwefel":
        return Problem(name, dim, 500.0, 0.0, _schwefel, "Deceptive landscape with optimum away from origin")
    if name == "Styblinski_Tang":
        return Problem(name, dim, 5.0, -39.1661657037714*dim, _styblinski_tang,
                       "Nonconvex polynomial with competing basins")
    if name == "Himmelblau":
        return Problem(name, dim, 6.0, 0.0, _himmelblau, "Four separated global minima (2D)")
    if name == "Modified_Himmelblau":
        return Problem(name, dim, 6.0, 0.64147080665, _modified_himmelblau,
                       "Himmelblau-like basins with quadratic bias (2D)")

    axis = np.arange(1, dim+1, dtype=float)
    shift_fraction = 0.48*np.sin(axis*1.61803398875 + 0.41)
    if name == "Shifted_Rastrigin":
        shift = 5.12*shift_fraction
        return Problem(name, dim, 5.12, 0.0, lambda z, s=shift: _rastrigin(z-s),
                       "Rastrigin translated away from the shared origin")
    if name == "Shifted_Ackley":
        shift = 5.0*shift_fraction
        return Problem(name, dim, 5.0, 0.0, lambda z, s=shift: _ackley(z-s),
                       "Ackley translated away from the shared origin")
    if name == "Shifted_Griewank":
        shift = 600.0*shift_fraction
        return Problem(name, dim, 600.0, 0.0, lambda z, s=shift: _griewank(z-s),
                       "Griewank translated away from origin and initial nodes")
    if name == "Rotated_Rosenbrock":
        scale = 2.0
        rng = np.random.default_rng(811+dim)
        raw = rng.normal(size=(dim, dim))
        rotation, _ = np.linalg.qr(raw)
        center = scale*0.30*np.sin(axis*0.73 + 0.2)
        def rotated(z: np.ndarray, r=rotation, c=center) -> float:
            u = r.T @ (np.asarray(z, dtype=float)-c) + np.ones_like(c)
            return _rosenbrock(u)
        return Problem(name, dim, scale, 0.0, rotated,
                       "Rosenbrock valley rotated away from the coordinate axes")
    raise KeyError(f"Unknown benchmark function: {name}")


FUNCTION_NAMES = (
    "Sphere", "Rastrigin", "Ackley", "Rosenbrock", "Levy", "Griewank",
    "Schwefel", "Styblinski_Tang", "Shifted_Rastrigin", "Shifted_Ackley",
    "Shifted_Griewank", "Rotated_Rosenbrock", "Himmelblau", "Modified_Himmelblau",
)


class Oracle:
    """Objective wrapper with a hard evaluation budget and explicit accounting.

    For deterministic objectives, exact repeated points are served from a cache.
    For noisy objectives, every request gets a fresh noise realization and is
    charged as an objective evaluation. Both policies apply to every method.
    """
    def __init__(self, problem: Problem, budget: int, seed: int, noise_level: float = 0.0):
        self.problem = problem
        self.budget = int(budget)
        self.noise_level = float(noise_level)
        # Noise scales with a fixed reference gap estimated before optimization.
        self.noise_sigma = self.noise_level * float(problem.metric_scale)
        self.rng = np.random.default_rng(seed)
        self.cache_enabled = self.noise_level == 0.0
        self.cache: dict[tuple[float, ...], float] = {}
        self.requests = 0
        self.cache_hits = 0
        self.evals = 0
        self.points: list[np.ndarray] = []
        self.values: list[float] = []  # observed objective values
        self.phase_counts: dict[str, int] = {}
        self.best_x: np.ndarray | None = None
        self.best_f = float("inf")

    @staticmethod
    def key(x: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    @property
    def remaining(self) -> int:
        return max(0, self.budget-self.evals)

    def cached(self, x: np.ndarray):
        if not self.cache_enabled:
            return None
        return self.cache.get(self.key(x), None)

    def __call__(self, x: np.ndarray, phase: str = "objective") -> float:
        self.requests += 1
        x = np.clip(np.asarray(x, dtype=float), DOMAIN_LO, DOMAIN_HI)
        if x.shape != (self.problem.dim,) or not np.all(np.isfinite(x)):
            raise ValueError(f"Invalid point shape/value: expected ({self.problem.dim},)")
        key = self.key(x)
        if self.cache_enabled and key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        if self.evals >= self.budget:
            raise BudgetExhausted(f"budget={self.budget} exhausted")
        clean_value = self.problem.evaluate(x)
        value = clean_value if self.noise_level == 0.0 else clean_value + float(self.rng.normal(0.0, self.noise_sigma))
        if not np.isfinite(value):
            value = float("inf")
        self.evals += 1
        self.points.append(x.copy())
        self.values.append(float(value))
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        if self.cache_enabled:
            self.cache[key] = float(value)
        if value < self.best_f:
            self.best_f = float(value)
            self.best_x = x.copy()
        return float(value)

    def gradient(self, x: np.ndarray, phase: str = "finite_difference", h: float = 1e-4) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float).copy(), DOMAIN_LO, DOMAIN_HI)
        grad = np.zeros_like(x)
        for j in range(x.size):
            left_ok = x[j] - h >= DOMAIN_LO
            right_ok = x[j] + h <= DOMAIN_HI
            if left_ok and right_ok:
                xm, xp = x.copy(), x.copy()
                xm[j] -= h
                xp[j] += h
                fm = self(xm, phase=phase)
                fp = self(xp, phase=phase)
                grad[j] = (fp-fm)/(2.0*h)
            elif right_ok:
                xp = x.copy()
                xp[j] += h
                f0 = self(x, phase=phase)
                fp = self(xp, phase=phase)
                grad[j] = (fp-f0)/h
            elif left_ok:
                xm = x.copy()
                xm[j] -= h
                fm = self(xm, phase=phase)
                f0 = self(x, phase=phase)
                grad[j] = (f0-fm)/h
            else:
                grad[j] = 0.0
        return grad


def sobol_points(n: int, dim: int, seed: int) -> np.ndarray:
    n = max(0, int(n))
    if n == 0:
        return np.empty((0, dim), dtype=float)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = int(math.ceil(math.log2(max(1, n))))
        unit = sampler.random_base2(m)[:n]
    return 2.0*unit - 1.0


def build_problem(name: str, dim: int) -> Problem:
    """Build a problem and calibrate its scale using a fixed, method-independent design.

    The reference set is used only for dimensionless scoring and relative noise
    calibration. It is never supplied to any optimizer and does not count against
    an optimizer's function-evaluation budget.
    """
    problem = _build_problem_uncalibrated(name, dim)
    name_code = zlib.crc32(name.encode("utf-8"))
    reference = sobol_points(256, dim, 71_003 + name_code + 97*dim)
    ref_values = np.asarray([problem.evaluate(x) for x in reference], dtype=float)
    deviations = np.abs(ref_values-problem.f_star)
    deviations = deviations[np.isfinite(deviations)]
    scale = float(np.median(deviations)) if deviations.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.quantile(deviations, 0.75)) if deviations.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return replace(problem, metric_scale=scale)


def normalized_gap(best_clean: float, f_star: float, metric_scale: float = 1.0) -> float:
    """Dimensionless optimality gap; thresholds mean a fraction of reference gap."""
    if not np.isfinite(best_clean):
        return float("inf")
    return max(0.0, float(best_clean-f_star))/max(float(metric_scale), 1e-12)


def diverse_starts(points: Iterable[np.ndarray], values: Iterable[float], limit: int,
                   dim: int, min_dist: float | None = None) -> list[np.ndarray]:
    pts = list(points)
    vals = list(values)
    if not pts or limit <= 0:
        return []
    order = np.argsort(np.asarray(vals, dtype=float))
    min_dist = float(min_dist if min_dist is not None else 0.12*np.sqrt(dim))
    out: list[np.ndarray] = []
    for idx in order:
        p = np.asarray(pts[int(idx)], dtype=float)
        if all(np.linalg.norm(p-q) >= min_dist for q in out):
            out.append(p.copy())
        if len(out) >= limit:
            break
    if not out:
        out.append(np.asarray(pts[int(order[0])], dtype=float).copy())
    return out


@dataclass
class Region:
    region_id: int
    lo: np.ndarray
    hi: np.ndarray
    depth: int
    parent_id: int | None
    created_at: int
    last_selected: int
    last_audited: int
    points: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    selection_count: int = 0
    audit_failures: int = 0
    best_value: float = float("inf")

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.hi-self.lo))


def _region_scores(regions: list[Region], global_best: float, global_scale: float,
                   split_count: int, max_diagonal: float) -> np.ndarray:
    """Vectorized priorities; avoids one Python function call per active box."""
    local_best = np.fromiter((r.best_value for r in regions), dtype=float, count=len(regions))
    diagonals = np.fromiter((r.diagonal for r in regions), dtype=float, count=len(regions))
    last_selected = np.fromiter((r.last_selected for r in regions), dtype=float, count=len(regions))
    sample_counts = np.fromiter((len(r.values) for r in regions), dtype=float, count=len(regions))
    gaps = np.maximum(0.0, local_best-global_best)
    quality = np.where(np.isfinite(local_best), np.exp(-np.minimum(gaps/max(global_scale, 1e-10), 60.0)), 0.0)
    diagonal_score = np.minimum(1.0, diagonals/max(max_diagonal, 1e-12))
    age = np.minimum(1.0, np.maximum(0.0, split_count-last_selected)/12.0)
    sparse = 1.0/np.sqrt(1.0+sample_counts)
    return 0.36*quality + 0.24*diagonal_score + 0.24*age + 0.16*sparse


def _split_region(region: Region, split_dim: int, new_id: int, split_count: int) -> tuple[Region, Region]:
    mid = (region.lo[split_dim]+region.hi[split_dim])/2.0
    lo1, hi1 = region.lo.copy(), region.hi.copy()
    lo2, hi2 = region.lo.copy(), region.hi.copy()
    hi1[split_dim] = mid
    lo2[split_dim] = mid
    c1 = Region(new_id, lo1, hi1, region.depth+1, region.region_id, split_count, split_count, split_count)
    c2 = Region(new_id+1, lo2, hi2, region.depth+1, region.region_id, split_count, split_count, split_count)
    tol = 1e-12
    for p, v in zip(region.points, region.values):
        if np.all(p >= c1.lo-tol) and np.all(p <= c1.hi+tol):
            c1.points.append(p.copy()); c1.values.append(float(v))
        if np.all(p >= c2.lo-tol) and np.all(p <= c2.hi+tol):
            c2.points.append(p.copy()); c2.values.append(float(v))
    if c1.values:
        c1.best_value = min(c1.values)
    if c2.values:
        c2.best_value = min(c2.values)
    return c1, c2


def _record_region(region: Region, x: np.ndarray, value: float) -> None:
    tol = 1e-12
    if np.all(x >= region.lo-tol) and np.all(x <= region.hi+tol):
        region.points.append(np.asarray(x, dtype=float).copy())
        region.values.append(float(value))
        region.best_value = min(region.best_value, float(value))


def _ring_directions(d: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Balanced coordinate poll: one random sign per coordinate before opposite signs."""
    perm = rng.permutation(d)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=d)
    axes: list[np.ndarray] = []
    for sign_pass in (1.0, -1.0):
        for j in perm:
            e = np.zeros(d, dtype=float)
            e[int(j)] = sign_pass*signs[int(j)]
            axes.append(e)
    random_directions = rng.normal(size=(min(d, 8), d))
    random_directions /= np.maximum(np.linalg.norm(random_directions, axis=1, keepdims=True), 1e-12)
    return axes + [u for u in random_directions]


def _best_surrogate_center(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Fit a validated quadratic; fall back to a diagonal quadratic when data cannot support a full one."""
    if points.ndim != 2 or points.shape[0] < 8:
        return None, float("nan")
    n, d = points.shape
    # Full quadratic has 1+d+d(d+1)/2 parameters; the diagonal model has 1+2d.
    full_pairs = [(i, j) for i in range(d) for j in range(i, d)]
    full_p = 1 + d + len(full_pairs)
    diag_p = 1 + 2*d
    # Keep at least a few held-out points; do not attempt an underdetermined full fit.
    model = "full" if int(math.floor(0.8*n)) >= full_p + 2 and n-int(math.floor(0.8*n)) >= 3 else "diagonal"
    if model == "diagonal" and int(math.floor(0.8*n)) < diag_p + 2:
        return None, float("nan")
    features = [np.ones(n)]
    for i in range(d):
        features.append(points[:, i])
    pairs = full_pairs if model == "full" else [(i, i) for i in range(d)]
    for i, j in pairs:
        features.append(points[:, i]*points[:, j])
    design = np.column_stack(features)
    p = design.shape[1]
    train_n = int(math.floor(0.8*n))
    if train_n < p+2 or n-train_n < 3:
        return None, float("nan")
    try:
        coef, _, rank, _ = np.linalg.lstsq(design[:train_n], values[:train_n], rcond=None)
        if rank < p or not np.all(np.isfinite(coef)):
            return None, float("nan")
        pred = design[train_n:] @ coef
        target = values[train_n:]
        denom = float(np.sum((target-target.mean())**2))
        q2 = float(1.0-np.sum((target-pred)**2)/denom) if denom > 1e-20 else float("nan")
        linear = coef[1:1+d]
        hessian = np.zeros((d, d), dtype=float)
        qcoef = coef[1+d:]
        for c, (i, j) in zip(qcoef, pairs):
            if i == j:
                hessian[i, i] = 2.0*c
            else:
                hessian[i, j] = c
                hessian[j, i] = c
        eig = np.linalg.eigvalsh(hessian)
        if not np.all(np.isfinite(eig)) or eig[0] <= 1e-8:
            return None, q2
        center = -np.linalg.solve(hessian, linear)
        if not np.all(np.isfinite(center)) or np.any(center < DOMAIN_LO) or np.any(center > DOMAIN_HI):
            return None, q2
        return center, q2
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None, float("nan")


def _run_lbfgsb(oracle: Oracle, starts: list[np.ndarray], phase: str, max_starts: int | None = None,
                max_new_evals: int | None = None, reserve_evals: int = 0,
                max_evals_per_start: int | None = None) -> None:
    if max_starts is not None:
        starts = starts[:max_starts]
    stage_start_evals = oracle.evals
    stage_cap = oracle.remaining if max_new_evals is None else min(oracle.remaining, max(0, int(max_new_evals)))

    for x0 in starts:
        if oracle.remaining <= reserve_evals or oracle.evals-stage_start_evals >= stage_cap:
            break
        start_evals = oracle.evals

        def stage_objective(x):
            # Repeated deterministic points are free under the shared cache.
            if oracle.cache_enabled and oracle.cached(x) is not None:
                return oracle(x, phase=phase)
            if oracle.evals-stage_start_evals >= stage_cap or oracle.remaining <= reserve_evals:
                raise BudgetExhausted("local-stage budget exhausted")
            if max_evals_per_start is not None and oracle.evals-start_evals >= max_evals_per_start:
                raise LocalStartBudget("local restart allocation exhausted")
            return oracle(x, phase=phase)

        try:
            minimize(stage_objective, np.clip(x0, DOMAIN_LO, DOMAIN_HI),
                     method="L-BFGS-B", bounds=[(DOMAIN_LO, DOMAIN_HI)]*oracle.problem.dim,
                     options={"maxfun": max(20, (max_evals_per_start or stage_cap)+10),
                              "maxiter": max(20, (max_evals_per_start or stage_cap)+10),
                              "ftol": 1e-12, "gtol": 1e-7, "maxls": 20})
        except LocalStartBudget:
            # Crucial for a fair multistart baseline: one stubborn start cannot
            # monopolize the full run budget and silently disable the other starts.
            continue
        except BudgetExhausted:
            break


def _run_powell(oracle: Oracle, starts: list[np.ndarray], max_starts: int = 8,
                max_evals_per_start: int | None = None) -> None:
    for x0 in starts[:max_starts]:
        if oracle.remaining <= 0:
            break
        start_evals = oracle.evals

        def objective(x):
            if oracle.cache_enabled and oracle.cached(x) is not None:
                return oracle(x, phase="powell_local")
            if max_evals_per_start is not None and oracle.evals-start_evals >= max_evals_per_start:
                raise LocalStartBudget("Powell restart allocation exhausted")
            return oracle(x, phase="powell_local")

        try:
            minimize(objective, np.clip(x0, DOMAIN_LO, DOMAIN_HI),
                     method="Powell", bounds=[(DOMAIN_LO, DOMAIN_HI)]*oracle.problem.dim,
                     options={"maxfev": max(30, (max_evals_per_start or oracle.remaining)+10),
                              "maxiter": max(30, (max_evals_per_start or oracle.remaining)+10),
                              "xtol": 1e-5, "ftol": 1e-8})
        except LocalStartBudget:
            continue
        except BudgetExhausted:
            break


def run_abg(problem: Problem, budget: int, seed: int, noise_level: float,
            use_surrogate: bool = True, use_local: bool = True, use_ring: bool = True,
            name: str = "ABG-v30", include_center: bool = True) -> Oracle:
    t_algorithm = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    oracle = Oracle(problem, budget, seed+31, noise_level)
    d = problem.dim
    n_features = 1+d+d*(d+1)//2
    init_cap = max(8, budget//2 if d >= 30 else budget//3)
    n_init = min(max(24, 2*n_features), init_cap, 160)
    # An objective-agnostic domain-center anchor avoids missing central basins;
    # it replaces one Sobol point, so the initialization evaluation count is unchanged.
    if include_center:
        sobol_design = sobol_points(max(0, n_init-1), d, seed+101)
        design = np.vstack((np.zeros((1, d), dtype=float), sobol_design))
    else:
        design = sobol_points(n_init, d, seed+101)
    root = Region(0, np.full(d, DOMAIN_LO), np.full(d, DOMAIN_HI), 0, None, 0, 0, 0)
    phase_start = time.perf_counter()
    for x in design:
        if oracle.remaining <= 0:
            break
        try:
            value = oracle(x, phase="initial_design")
        except BudgetExhausted:
            break
        _record_region(root, x, value)
    phase_seconds["initial_design"] = time.perf_counter()-phase_start
    active: dict[int, Region] = {}
    next_region_id = 1
    splits = audits = 0
    resolution_limited_children = 0
    max_splits = max(100, 4*int(budget))
    reserve = min(max(20, budget//4), max(0, budget-n_init-2)) if use_local else 0
    if not use_local and (use_ring or use_surrogate):
        # Keep enabled post-search components alive in ablations; otherwise
        # the box phase would consume the entire budget and confound ablations.
        reserve = min(max(8, 2*d), max(0, budget-n_init-2)) if use_ring else min(1, max(0, budget-n_init-2))
    explore_every = 5  # select the stalest active leaf every fifth split
    max_diagonal = 2.0*np.sqrt(float(d))
    scale_refresh_every = 16
    score_refresh_every = 32
    last_scale_refresh = -scale_refresh_every
    last_score_refresh = -score_refresh_every
    priority_scale = 1.0
    priority_heap: list[tuple[float, int, int]] = []
    priority_version: dict[int, int] = {}
    explore_heap: list[tuple[int, float, int]] = []
    audit_heap: list[tuple[int, int, int]] = []

    def push_active(region: Region) -> None:
        """Insert a live region into O(log n) scheduling heaps."""
        active[region.region_id] = region
        version = priority_version.get(region.region_id, 0) + 1
        priority_version[region.region_id] = version
        score = float(_region_scores([region], oracle.best_f, priority_scale, splits, max_diagonal)[0])
        heapq.heappush(priority_heap, (-score, region.region_id, version))
        heapq.heappush(explore_heap, (region.last_selected, -region.diagonal, region.region_id))
        heapq.heappush(audit_heap, (region.last_audited, region.last_selected, region.region_id))

    push_active(root)
    phase_start = time.perf_counter()
    while oracle.remaining > reserve and active and splits < max_splits:
        # Objective-value scale is refreshed periodically rather than rescanning
        # all values at every split.
        if splits-last_scale_refresh >= scale_refresh_every:
            vals_for_scale = np.asarray(oracle.values, dtype=float)
            if vals_for_scale.size > 1:
                q25, q75 = np.quantile(vals_for_scale, [0.25, 0.75])
                priority_scale = max(float(q75-q25), float(np.std(vals_for_scale))*0.25, 1e-10)
            else:
                priority_scale = 1.0
            last_scale_refresh = splits

        # Rebuild the priority heap in batches. Between rebuilds, choose the
        # top live region in O(log n); the age-based heap independently ensures
        # broad exploration even while exploitation scores are stale.
        if splits-last_score_refresh >= score_refresh_every:
            regions = list(active.values())
            scores = _region_scores(regions, oracle.best_f, priority_scale, splits, max_diagonal)
            priority_heap = []
            for region, score in zip(regions, scores):
                version = priority_version.get(region.region_id, 0) + 1
                priority_version[region.region_id] = version
                priority_heap.append((-float(score), region.region_id, version))
            heapq.heapify(priority_heap)
            last_score_refresh = splits

        forced_explore = splits > 0 and splits % explore_every == explore_every-1
        parent = None
        if forced_explore:
            while explore_heap:
                last_selected, negative_diagonal, region_id = heapq.heappop(explore_heap)
                candidate = active.get(region_id)
                if (candidate is not None and candidate.last_selected == last_selected
                        and abs(candidate.diagonal + negative_diagonal) <= 1e-12):
                    parent = active.pop(region_id)
                    break
        if parent is None:
            while priority_heap:
                negative_score, region_id, version = heapq.heappop(priority_heap)
                candidate = active.get(region_id)
                if candidate is not None and priority_version.get(region_id) == version:
                    parent = active.pop(region_id)
                    break
        if parent is None:
            # Defensive recovery if a heap was exhausted by stale entries.
            regions = list(active.values())
            if not regions:
                break
            scores = _region_scores(regions, oracle.best_f, priority_scale, splits, max_diagonal)
            parent = regions[int(np.argmax(scores))]
            active.pop(parent.region_id, None)

        parent.last_selected = splits
        parent.selection_count += 1
        widths = parent.hi-parent.lo
        split_dim = int(np.argmax(widths))
        c1, c2 = _split_region(parent, split_dim, next_region_id, splits)
        next_region_id += 2
        resolvable_children: list[Region] = []
        for child in (c1, c2):
            if oracle.remaining <= reserve:
                break
            center = (child.lo+child.hi)/2.0
            before_eval = oracle.evals
            try:
                val = oracle(center, phase="box_split")
            except BudgetExhausted:
                break
            made_progress = oracle.evals > before_eval
            if made_progress:
                _record_region(child, center, val)
            elif oracle.cache_enabled and oracle.remaining > reserve:
                # A cached child center is not evidence of progress: try a new
                # interior point before marking the child resolution-limited.
                fallback = None
                for _ in range(64):
                    trial = rng.uniform(child.lo, child.hi)
                    if oracle.cached(trial) is None:
                        fallback = trial
                        break
                if fallback is not None:
                    before_fallback = oracle.evals
                    try:
                        fallback_value = oracle(fallback, phase="box_fallback")
                    except BudgetExhausted:
                        break
                    made_progress = oracle.evals > before_fallback
                    if made_progress:
                        _record_region(child, fallback, fallback_value)
            if made_progress:
                resolvable_children.append(child)
            else:
                resolution_limited_children += 1
        splits += 1
        for child in resolvable_children:
            push_active(child)

        # Audits use a separate oldest-first heap, so the scheduler does not
        # scan every active region merely to find the stalest one.
        if splits % 7 == 0 and oracle.remaining > reserve and active:
            target = None
            while audit_heap:
                last_audited, last_selected, region_id = heapq.heappop(audit_heap)
                candidate = active.get(region_id)
                if (candidate is not None and candidate.last_audited == last_audited
                        and candidate.last_selected == last_selected):
                    target = candidate
                    break
            if target is not None:
                candidate_point = None
                for _ in range(30):
                    p = rng.uniform(target.lo, target.hi)
                    if oracle.cached(p) is None:
                        candidate_point = p
                        break
                target.last_audited = splits
                heapq.heappush(audit_heap, (target.last_audited, target.last_selected, target.region_id))
                if candidate_point is not None:
                    try:
                        val = oracle(candidate_point, phase="box_audit")
                        _record_region(target, candidate_point, val)
                        audits += 1
                    except BudgetExhausted:
                        break
    phase_seconds["box_search_and_audits"] = time.perf_counter()-phase_start
    phase_start = time.perf_counter()

    surrogate_q2 = float("nan")
    surrogate_used = 0
    surrogate_center = None
    if use_surrogate and oracle.remaining > 0 and len(oracle.points) >= min(n_features, 1+2*d)+4:
        # Only the initial space-filling design is used for the hold-out fit.
        pcount = min(len(design), len(oracle.points))
        initial_points = np.asarray(oracle.points[:pcount], dtype=float)
        initial_values = np.asarray(oracle.values[:pcount], dtype=float)
        surrogate_center, surrogate_q2 = _best_surrogate_center(initial_points, initial_values)
        if surrogate_center is not None and np.isfinite(surrogate_q2) and surrogate_q2 >= 0.70:
            try:
                oracle(surrogate_center, phase="surrogate_probe")
                surrogate_used = 1
            except BudgetExhausted:
                surrogate_center = None
    phase_seconds["surrogate_fit_and_probe"] = time.perf_counter()-phase_start

    phase_start = time.perf_counter()
    if use_local and oracle.remaining > 0:
        n_starts = min(max(2, min(5, d//2+2)), max(1, oracle.remaining//max(2*d+1, 4)))
        starts = diverse_starts(oracle.points, oracle.values, n_starts, d)
        if surrogate_center is not None and surrogate_used:
            starts = [surrogate_center.copy()] + [s for s in starts if np.linalg.norm(s-surrogate_center) > 1e-8]
        if oracle.best_x is not None and all(np.linalg.norm(oracle.best_x-s) > 1e-8 for s in starts):
            starts.insert(0, oracle.best_x.copy())
        ring_reserve = min(max(8, 2*d), max(0, oracle.remaining//3)) if use_ring else 0
        local_cap = max(0, oracle.remaining-ring_reserve)
        start_cap = max(4, local_cap//max(1, min(len(starts), max(1, n_starts)))) if local_cap > 0 else 0
        _run_lbfgsb(oracle, starts, phase="local_refine", max_starts=max(1, n_starts),
                    max_new_evals=local_cap, reserve_evals=ring_reserve,
                    max_evals_per_start=start_cap)
    phase_seconds["local_refinement"] = time.perf_counter()-phase_start

    phase_start = time.perf_counter()
    if use_ring and oracle.remaining > 0 and oracle.best_x is not None:
        # A bounded, multi-radius incumbent-centered ring; proposals are charged.
        d = problem.dim
        dirs = _ring_directions(d, rng)
        max_ring_evals = min(3*len(dirs), max(0, oracle.remaining))
        used = 0
        for radius in (0.20, 0.08, 0.03):
            center = oracle.best_x.copy()
            for direction in dirs:
                if used >= max_ring_evals or oracle.remaining <= 0:
                    break
                point = np.clip(center+radius*direction, DOMAIN_LO, DOMAIN_HI)
                before = oracle.evals
                before_best = oracle.best_f
                try:
                    oracle(point, phase="ring")
                except BudgetExhausted:
                    break
                used += oracle.evals-before
                # Greedy recentering: subsequent polls exploit any improvement immediately.
                if oracle.best_x is not None and oracle.best_f < before_best-1e-15:
                    center = oracle.best_x.copy()
            if used >= max_ring_evals or oracle.remaining <= 0:
                break
    phase_seconds["ring_search"] = time.perf_counter()-phase_start
    phase_seconds["total_algorithm"] = time.perf_counter()-t_algorithm
    phase_sum = sum(v for k, v in phase_seconds.items() if k != "total_algorithm")
    phase_seconds["setup_and_bookkeeping"] = max(0.0, phase_seconds["total_algorithm"]-phase_sum)

    oracle.algorithm_data = {
        "method": name, "splits": splits, "audits": audits,
        "center_anchor_enabled": int(include_center),
        "phase_seconds": phase_seconds,
        "surrogate_q2": surrogate_q2, "surrogate_used": surrogate_used,
        "local_enabled": int(use_local), "ring_enabled": int(use_ring),
        "surrogate_enabled": int(use_surrogate), "initial_design_size": n_init,
        "active_regions_final": len(active), "exploration_every_splits": explore_every,
        "priority_scale_refresh_every_splits": scale_refresh_every,
        "priority_score_refresh_every_splits": score_refresh_every,
        "priority_scoring": "heap scheduler; vectorized periodic rebuild",
        "resolution_limited_children": resolution_limited_children, "split_cap": max_splits,
    }
    return oracle


def run_multistart_local(problem: Problem, budget: int, seed: int, noise_level: float,
                         method: str = "L-BFGS-B", sobol: bool = True) -> Oracle:
    oracle = Oracle(problem, budget, seed+41, noise_level)
    d = problem.dim
    n_init = min(max(10, 2*d+4), max(5, budget//3), 64)
    design = sobol_points(n_init, d, seed+53) if sobol else np.random.default_rng(seed+53).uniform(-1, 1, (n_init, d))
    for x in design:
        if oracle.remaining <= 0:
            break
        try:
            oracle(x, phase="initial_design")
        except BudgetExhausted:
            break
    n_starts = min(8, max(1, oracle.remaining//max(2*d+1, 4)))
    starts = diverse_starts(oracle.points, oracle.values, n_starts, d)
    per_start = max(4, oracle.remaining//max(1, len(starts))) if starts else 0
    if method == "L-BFGS-B":
        _run_lbfgsb(oracle, starts, phase="lbfgsb_local", max_starts=8,
                    max_evals_per_start=per_start)
    elif method == "Powell":
        _run_powell(oracle, starts, max_starts=8, max_evals_per_start=per_start)
    else:
        raise ValueError(method)
    return oracle


def run_adam(problem: Problem, budget: int, seed: int, noise_level: float, lr: float) -> Oracle:
    """Finite-difference Adam; the design phase chooses a reproducible good start."""
    oracle = Oracle(problem, budget, seed+71, noise_level)
    d = problem.dim
    n_init = min(max(8, 2*d+4), max(4, budget//3), 48)
    design = sobol_points(n_init, d, seed+73)
    for x in design:
        if oracle.remaining <= 0:
            break
        try:
            oracle(x, phase="initial_design")
        except BudgetExhausted:
            break
    if oracle.best_x is None:
        return oracle
    x = oracle.best_x.copy()
    m = np.zeros(d); v = np.zeros(d)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    while oracle.remaining >= 2*d+1:
        try:
            g = oracle.gradient(x, phase="adam_gradient", h=1e-4)
        except BudgetExhausted:
            break
        t += 1
        m = beta1*m + (1.0-beta1)*g
        v = beta2*v + (1.0-beta2)*(g*g)
        mhat = m/(1.0-beta1**t)
        vhat = v/(1.0-beta2**t)
        xn = np.clip(x - float(lr)*mhat/(np.sqrt(vhat)+eps), DOMAIN_LO, DOMAIN_HI)
        try:
            oracle(xn, phase="adam_update")
        except BudgetExhausted:
            break
        x = xn
    oracle.algorithm_data = {"adam_lr": float(lr), "adam_steps": t, "initial_design_size": n_init}
    return oracle


def tune_adam_lr(problem: Problem, budget: int, noise_level: float, seed: int,
                 candidates: tuple[float, ...] = ADAM_LR_CANDIDATES, reps: int = 3) -> tuple[float, int, dict[float, float]]:
    """Choose Adam's LR on independent validation seeds, before test seeds run."""
    validation_budget = max(20, min(int(budget), 250))
    scores: dict[float, list[float]] = {float(lr): [] for lr in candidates}
    tune_evals = 0
    for li, lr in enumerate(candidates):
        for rep in range(max(1, int(reps))):
            v_seed = seed + 700_001 + li*10_007 + rep*1_009
            oracle = run_adam(problem, validation_budget, v_seed, noise_level, float(lr))
            tune_evals += oracle.evals
            clean = problem.evaluate(oracle.best_x) if oracle.best_x is not None else float("inf")
            scores[float(lr)].append(normalized_gap(clean, problem.f_star, problem.metric_scale))
    medians = {lr: float(np.median(vals)) if vals else float("inf") for lr, vals in scores.items()}
    selected = min(medians, key=lambda lr: (medians[lr], lr))
    return float(selected), int(tune_evals), medians


def run_cma_es(problem: Problem, budget: int, seed: int, noise_level: float, lr: float | None = None) -> Oracle:
    """Compact full-covariance CMA-ES with evaluation budget enforced by Oracle."""
    d = problem.dim
    rng = np.random.default_rng(seed+83)
    oracle = Oracle(problem, budget, seed+89, noise_level)
    lam = max(6, 4 + int(3*np.log(max(2, d))))
    mu = lam//2
    weights = np.log(mu+0.5)-np.log(np.arange(1, mu+1))
    weights = weights/np.sum(weights)
    mueff = float(1.0/np.sum(weights**2))
    cc = (4.0+mueff/d)/(d+4.0+2.0*mueff/d)
    cs = (mueff+2.0)/(d+mueff+5.0)
    c1 = 2.0/((d+1.3)**2+mueff)
    cmu = min(1.0-c1, 2.0*(mueff-2.0+1.0/mueff)/((d+2.0)**2+mueff))
    damps = 1.0 + 2.0*max(0.0, math.sqrt((mueff-1.0)/(d+1.0))-1.0) + cs
    chi_n = math.sqrt(d)*(1.0-1.0/(4.0*d)+1.0/(21.0*d*d))
    mean = rng.uniform(-0.5, 0.5, d)
    sigma = 0.30
    cov = np.eye(d)
    pc = np.zeros(d); ps = np.zeros(d)
    generation = 0
    while oracle.remaining > 0:
        generation += 1
        remaining = oracle.remaining
        this_lam = min(lam, remaining)
        if this_lam < 4:
            for _ in range(this_lam):
                x = rng.uniform(-1.0, 1.0, d)
                try:
                    oracle(x, phase="cmaes_tail")
                except BudgetExhausted:
                    break
            break
        try:
            eigenvalues, B = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-14)
            D = np.sqrt(eigenvalues)
            invsqrt = B @ np.diag(1.0/D) @ B.T
            z = rng.normal(size=(this_lam, d))
            y = (z*D) @ B.T
            xs = np.clip(mean[None, :] + sigma*y, -1.0, 1.0)
            vals = np.array([oracle(x, phase="cmaes_population") for x in xs], dtype=float)
        except BudgetExhausted:
            break
        order = np.argsort(vals)
        old_mean = mean.copy()
        chosen = xs[order[:mu]]
        used_mu = len(chosen)
        if used_mu < mu:
            break
        mean = np.sum(chosen*weights[:used_mu, None], axis=0)
        yw = (mean-old_mean)/max(sigma, 1e-12)
        ps = (1.0-cs)*ps + math.sqrt(cs*(2.0-cs)*mueff)*(invsqrt @ yw)
        norm_ps = float(np.linalg.norm(ps))
        hsig = float(norm_ps/math.sqrt(max(1e-12, 1.0-(1.0-cs)**(2.0*generation))/1.0)/chi_n < (1.4+2.0/(d+1.0)))
        pc = (1.0-cc)*pc + hsig*math.sqrt(cc*(2.0-cc)*mueff)*yw
        y_selected = (chosen-old_mean[None, :])/max(sigma, 1e-12)
        rank_mu = np.zeros((d, d))
        for wi, yi in zip(weights[:used_mu], y_selected):
            rank_mu += wi*np.outer(yi, yi)
        cov = ((1.0-c1-cmu)*cov + c1*(np.outer(pc, pc) + (1.0-hsig)*cc*(2.0-cc)*cov)
               + cmu*rank_mu)
        cov = 0.5*(cov+cov.T)
        sigma *= math.exp((cs/damps)*(norm_ps/chi_n-1.0))
        sigma = float(np.clip(sigma, 1e-6, 2.0))
        if not np.all(np.isfinite(cov)) or not np.isfinite(sigma):
            cov = np.eye(d); sigma = 0.15
    oracle.algorithm_data = {"cmaes_generations": generation, "cmaes_population": lam}
    return oracle


def run_differential_evolution(problem: Problem, budget: int, seed: int, noise_level: float, lr: float | None = None) -> Oracle:
    oracle = Oracle(problem, budget, seed+97, noise_level)
    d = problem.dim
    pop_n = min(max(5*d, 15), max(5, budget//2))
    init = sobol_points(pop_n, d, seed+101)
    try:
        differential_evolution(lambda x: oracle(x, phase="differential_evolution"),
                               bounds=[(-1.0, 1.0)]*d, init=init, maxiter=max(1, budget//pop_n+3),
                               mutation=(0.5, 1.0), recombination=0.7, seed=seed+103,
                               polish=False, updating="immediate", workers=1, tol=0.0, atol=0.0)
    except BudgetExhausted:
        pass
    return oracle


def run_dual_annealing(problem: Problem, budget: int, seed: int, noise_level: float, lr: float | None = None) -> Oracle:
    oracle = Oracle(problem, budget, seed+107, noise_level)
    try:
        dual_annealing(lambda x: oracle(x, phase="dual_annealing"),
                       bounds=[(-1.0, 1.0)]*problem.dim, maxfun=budget,
                       seed=seed+109, no_local_search=False)
    except BudgetExhausted:
        pass
    return oracle


def run_direct(problem: Problem, budget: int, seed: int, noise_level: float, lr: float | None = None) -> Oracle:
    if scipy_direct is None:
        raise RuntimeError("DIRECT is unavailable in this SciPy version")
    oracle = Oracle(problem, budget, seed+113, noise_level)
    def guarded_objective(x):
        cached = oracle.cached(x)
        if oracle.remaining <= 0 and cached is None:
            # DIRECT's compiled core does not reliably propagate Python exceptions.
            # Return a finite penalty once the Oracle budget is spent; the Oracle
            # still records exactly the allowed number of real objective calls.
            base = oracle.best_f if np.isfinite(oracle.best_f) else 0.0
            return float(base + max(1.0, abs(base)))
        return oracle(x, phase="direct")
    scipy_direct(guarded_objective, bounds=[(-1.0, 1.0)]*problem.dim,
                 maxfun=budget, maxiter=budget, eps=1e-4, locally_biased=True)
    return oracle


METHOD_REGISTRY: dict[str, Callable[..., Oracle]] = {
    "ABG-v30": lambda p, b, s, n, lr=None: run_abg(p, b, s, n, True, True, True, "ABG-v30", True),
    "ABG-v30-no-center-anchor": lambda p, b, s, n, lr=None: run_abg(p, b, s, n, True, True, True, "ABG-v30-no-center-anchor", False),
    "ABG-v30-box-only": lambda p, b, s, n, lr=None: run_abg(p, b, s, n, False, False, False, "ABG-v30-box-only"),
    "ABG-v30-no-surrogate": lambda p, b, s, n, lr=None: run_abg(p, b, s, n, False, True, True, "ABG-v30-no-surrogate"),
    "ABG-v30-no-local": lambda p, b, s, n, lr=None: run_abg(p, b, s, n, True, False, True, "ABG-v30-no-local"),
    "ABG-v30-no-ring": lambda p, b, s, n, lr=None: run_abg(p, b, s, n, True, True, False, "ABG-v30-no-ring"),
    "Sobol+L-BFGS-B": lambda p, b, s, n, lr=None: run_multistart_local(p, b, s, n, "L-BFGS-B", True),
    "Random multi-start L-BFGS-B": lambda p, b, s, n, lr=None: run_multistart_local(p, b, s, n, "L-BFGS-B", False),
    "Powell multi-start": lambda p, b, s, n, lr=None: run_multistart_local(p, b, s, n, "Powell", True),
    "Adam-tuned (Sobol-seeded)": lambda p, b, s, n, lr=None: run_adam(p, b, s, n, 0.03 if lr is None else lr),
    "CMA-ES (compact implementation)": run_cma_es,
    "Differential Evolution": run_differential_evolution,
    "Dual Annealing": run_dual_annealing,
}
if scipy_direct is not None:
    METHOD_REGISTRY["DIRECT"] = run_direct


def _load_original_abg_module():
    """Optionally load the user-supplied v25 implementation kept beside this script."""
    source = Path(__file__).resolve().with_name("abg_v25_original.py")
    if not source.is_file():
        return None
    module_name = "abg_v25_original_embedded"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


LEGACY_ABG = _load_original_abg_module()


def run_legacy_abg_v25(problem: Problem, budget: int, seed: int, noise_level: float,
                       lr: float | None = None) -> Oracle:
    """Run the actual v25 source, only where its 2D deterministic assumptions hold."""
    if LEGACY_ABG is None:
        raise RuntimeError("abg_v25_original.py is missing; include it beside the v26 script")
    if problem.dim != 2 or float(noise_level) != 0.0:
        raise ValueError("The supplied original ABG v25 supports only 2D deterministic benchmarking")
    legacy_fn = LEGACY_ABG.TestFunction(
        name=problem.name, scale=problem.scale, fn=problem.fn,
        f_star=problem.f_star, description=problem.description)
    legacy_oracle_cls = LEGACY_ABG.Oracle

    class CountingLegacyOracle(legacy_oracle_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.requests = 0
            self.cache_hits = 0

        def __call__(self, x, phase="objective"):
            self.requests += 1
            was_cached = self.cache_enabled and self.key(x) in self.cache
            before = self.n
            value = super().__call__(x, phase=phase)
            if was_cached and self.n == before:
                self.cache_hits += 1
            return value

    LEGACY_ABG.Oracle = CountingLegacyOracle
    try:
        legacy = LEGACY_ABG.run_abg_v25(legacy_fn, budget=int(budget), seed=int(seed))
    finally:
        LEGACY_ABG.Oracle = legacy_oracle_cls

    oracle = Oracle(problem, budget, seed+131, 0.0)
    oracle.evals = int(legacy.n)
    oracle.requests = int(legacy.requests)
    oracle.cache_hits = int(legacy.cache_hits)
    oracle.points = [np.asarray(p, dtype=float).copy() for p in legacy.points]
    oracle.values = [float(v) for v in legacy.values]
    oracle.phase_counts = {str(k): int(v) for k, v in legacy.phase_counts.items()}
    oracle.best_x = None if legacy.best_x is None else np.asarray(legacy.best_x, dtype=float).copy()
    oracle.best_f = float(legacy.best_f)
    oracle.cache = dict(legacy.cache)
    oracle.algorithm_data = {"method": "ABG-v25-original", **getattr(legacy, "algorithm_data", {})}
    return oracle


if LEGACY_ABG is not None:
    METHOD_REGISTRY["ABG-v25-original (2D deterministic)"] = run_legacy_abg_v25
METHODS = tuple(METHOD_REGISTRY.keys())


def methods_for_case(problem: Problem, noise_level: float) -> tuple[str, ...]:
    """Only run the original v25 where its source code's domain assumptions are valid."""
    out = []
    for method in METHODS:
        if method == "ABG-v25-original (2D deterministic)" and (problem.dim != 2 or float(noise_level) != 0.0):
            continue
        out.append(method)
    return tuple(out)


def run_one(method: str, problem: Problem, budget: int, seed: int, noise_level: float,
            adam_lr: float | None = None) -> dict:
    if method not in METHOD_REGISTRY:
        raise KeyError(f"Method is not registered: {method}")
    t0 = time.perf_counter()
    oracle = METHOD_REGISTRY[method](problem, budget, seed, noise_level, adam_lr)
    elapsed = time.perf_counter()-t0
    if oracle.evals > int(budget):
        raise AssertionError(f"{method} exceeded budget: {oracle.evals}>{budget}")
    if sum(oracle.phase_counts.values()) != oracle.evals:
        raise AssertionError(f"{method} phase accounting mismatch: {sum(oracle.phase_counts.values())}!={oracle.evals}")
    if oracle.best_x is not None and (oracle.best_x.shape != (problem.dim,) or np.any(oracle.best_x < -1.0) or np.any(oracle.best_x > 1.0)):
        raise AssertionError(f"{method} returned an invalid incumbent")
    clean_f = problem.evaluate(oracle.best_x) if oracle.best_x is not None else float("inf")
    gap = normalized_gap(clean_f, problem.f_star, problem.metric_scale)
    data = getattr(oracle, "algorithm_data", {})
    row = {
        "function": problem.name, "dimension": problem.dim, "budget": int(budget),
        "noise_level": float(noise_level), "objective_noise_sigma": float(noise_level*problem.metric_scale),
        "metric_scale": float(problem.metric_scale), "method": method, "seed": int(seed),
        "best_f_observed": float(oracle.best_f), "best_f_clean": float(clean_f),
        "gap": float(gap), "success_1e-3": int(gap <= 1e-3), "success_1e-6": int(gap <= 1e-6),
        "evals": int(oracle.evals), "requests": int(oracle.requests), "cache_hits": int(oracle.cache_hits),
        "budget_used_pct": 100.0*oracle.evals/max(1, budget), "seconds": float(elapsed),
        "adam_lr": (float(data["adam_lr"]) if "adam_lr" in data else
                    (float(adam_lr) if method == "Adam-tuned (Sobol-seeded)" and adam_lr is not None else "")),
        "algorithm_data_json": json.dumps(data, sort_keys=True, allow_nan=True),
        "phase_counts_json": json.dumps(oracle.phase_counts, sort_keys=True),
        "error": "" if oracle.best_x is not None else "no incumbent returned",
    }
    row["best_x_json"] = json.dumps(oracle.best_x.tolist()) if oracle.best_x is not None else "[]"
    # Fixed-width coordinate columns keep checkpoint headers stable across dimensions.
    for i in range(10):
        row[f"best_x_{i+1}"] = float(oracle.best_x[i]) if oracle.best_x is not None and i < problem.dim else ""
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Keep a stable union of fields when resuming checkpoints created by an earlier profile.
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def append_csv_rows(path: Path, rows: list[dict]) -> None:
    """Checkpoint one completed seed in a single append/flush operation."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(rows[0].keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _key(row: dict) -> tuple:
    return (row["function"], int(row["dimension"]), int(row["budget"]),
            float(row["noise_level"]), row["method"], int(row["seed"]))


def _bootstrap_median_ci(values: np.ndarray, seed: int = 2026, reps: int = 500) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        k = (r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"]), r["method"])
        groups.setdefault(k, []).append(r)
    out = []
    for key, rr in sorted(groups.items(), key=lambda kv: str(kv[0])):
        raw_gaps = np.asarray([float(r["gap"]) for r in rr], dtype=float)
        # Keep failed/no-incumbent trials in the distribution as a worst-case
        # penalty; silently dropping them would make a method look better.
        gaps = np.asarray([max(0.0, g) if np.isfinite(g) else FAILED_GAP_PENALTY for g in raw_gaps], dtype=float)
        ci_lo, ci_hi = _bootstrap_median_ci(gaps, seed=zlib.crc32(repr(key).encode("utf-8")))
        out.append({
            "function": key[0], "dimension": key[1], "budget": key[2], "noise_level": key[3], "method": key[4],
            "n_trials": len(rr), "median_gap": float(np.median(gaps)),
            "q1_gap": float(np.quantile(gaps, 0.25)),
            "q3_gap": float(np.quantile(gaps, 0.75)),
            "iqr_gap": float(np.quantile(gaps, 0.75)-np.quantile(gaps, 0.25)),
            "mean_gap": float(np.mean(gaps)),
            "median_gap_ci95_low": ci_lo, "median_gap_ci95_high": ci_hi,
            "success_1e-3_pct": 100.0*float(np.mean([int(r["success_1e-3"]) for r in rr])),
            "success_1e-6_pct": 100.0*float(np.mean([int(r["success_1e-6"]) for r in rr])),
            "median_evals": float(np.median([float(r["evals"]) for r in rr])),
            "median_seconds": float(np.median([float(r["seconds"]) for r in rr])),
            "median_cache_hits": float(np.median([float(r["cache_hits"]) for r in rr])),
        })
    return out


def _holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    p = np.asarray([1.0 if not np.isfinite(v) else float(v) for v in pvalues])
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank)*p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def _effective_gap(value: float) -> float:
    """Numerical zero for tiny gaps; nonfinite/failed runs receive worst-case penalty."""
    x = float(value)
    if not np.isfinite(x):
        return FAILED_GAP_PENALTY
    return 0.0 if x <= 1e-12 else x


def paired_tests(rows: list[dict], control: str = "ABG-v30") -> list[dict]:
    task_rows: dict[tuple, dict[str, dict[int, dict]]] = {}
    for r in rows:
        task = (r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"]))
        task_rows.setdefault(task, {}).setdefault(r["method"], {})[int(r["seed"])] = r
    tests = []
    for task, by_method in sorted(task_rows.items(), key=lambda kv: str(kv[0])):
        if control not in by_method:
            continue
        control_by_seed = by_method[control]
        for method, by_seed in by_method.items():
            if method == control:
                continue
            seeds = sorted(set(control_by_seed)&set(by_seed))
            if len(seeds) < 2:
                continue
            a_raw = np.asarray([float(control_by_seed[s]["gap"]) for s in seeds], dtype=float)
            b_raw = np.asarray([float(by_seed[s]["gap"]) for s in seeds], dtype=float)
            # Near-zero floating-point residue should not create fake wins or p-values.
            a = np.asarray([_effective_gap(x) for x in a_raw], dtype=float)
            b = np.asarray([_effective_gap(x) for x in b_raw], dtype=float)
            la = np.log10(np.maximum(a, 1e-15))
            lb = np.log10(np.maximum(b, 1e-15))
            delta = lb-la  # positive means ABG has a smaller (better) gap
            nonzero = delta[np.abs(delta) > 1e-12]
            if len(nonzero) == 0:
                pvalue = 1.0
            else:
                try:
                    pvalue = float(stats.wilcoxon(la, lb, alternative="two-sided", zero_method="pratt").pvalue)
                except ValueError:
                    pvalue = 1.0
            wins = int(np.sum(a < b-1e-12))
            losses = int(np.sum(a > b+1e-12))
            ties = len(seeds)-wins-losses
            pos = int(np.sum(delta > 1e-12)); neg = int(np.sum(delta < -1e-12))
            sign_effect = (pos-neg)/max(1, pos+neg)
            ci_lo, ci_hi = _bootstrap_median_ci(
                delta, seed=zlib.crc32(repr((task, method)).encode("utf-8")), reps=2000)
            tests.append({
                "function": task[0], "dimension": task[1], "budget": task[2], "noise_level": task[3],
                "control": control, "competitor": method, "n_pairs": len(seeds),
                "median_log10_gap_improvement": float(np.median(delta)),
                "median_log10_gap_improvement_ci95_low": ci_lo,
                "median_log10_gap_improvement_ci95_high": ci_hi,
                "mean_log10_gap_improvement": float(np.mean(delta)), "sign_effect_abg_better": float(sign_effect),
                "abg_wins": wins, "abg_losses": losses, "ties": ties,
                "p_value": pvalue,
            })
    adjusted = _holm_adjust([float(t["p_value"]) for t in tests])
    for t, p_adj in zip(tests, adjusted):
        t["p_holm_global"] = p_adj
        t["significant_holm_0_05"] = int(p_adj < 0.05)
    return tests


def performance_profile(summary_rows: list[dict]) -> list[dict]:
    """Rank only methods evaluated on every task; partial-coverage methods are excluded."""
    tasks: dict[tuple, dict[str, float]] = {}
    for r in summary_rows:
        task = (r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"]))
        tasks.setdefault(task, {})[r["method"]] = float(r["median_gap"])
    if not tasks:
        return []
    common_methods = set.intersection(*(set(m.keys()) for m in tasks.values()))
    scores: dict[str, list[tuple[float, int]]] = {}
    for task in sorted(tasks, key=str):
        methods = {m: tasks[task][m] for m in common_methods}
        if not methods:
            continue
        effective = {m: _effective_gap(g) for m, g in methods.items()}
        best = min(effective.values())
        ordered_names = sorted(effective)
        rank_values = stats.rankdata([effective[m] for m in ordered_names], method="average")
        ranks = {m: float(rank) for m, rank in zip(ordered_names, rank_values)}
        for method, gap in effective.items():
            ratio = (gap+1e-15)/(best+1e-15)
            scores.setdefault(method, []).append((float(ratio), ranks[method]))
    out = []
    for method, vals in sorted(scores.items()):
        ratios = np.asarray([x[0] for x in vals], dtype=float)
        ranks = np.asarray([x[1] for x in vals], dtype=float)
        out.append({
            "method": method, "n_problem_cases": int(len(vals)), "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)), "win_cases": int(np.sum(ratios <= 1.0+1e-12)),
            "within_1_1x_pct": 100.0*float(np.mean(ratios <= 1.1)),
            "within_2x_pct": 100.0*float(np.mean(ratios <= 2.0)),
            "within_10x_pct": 100.0*float(np.mean(ratios <= 10.0)),
        })
    return out


def plot_profile(profile_rows: list[dict], out_path: Path) -> None:
    if not profile_rows:
        return
    ordered = sorted(profile_rows, key=lambda r: (r["mean_rank"], -r["within_2x_pct"]))
    names = [r["method"] for r in ordered]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.7), 6))
    ax.bar(x, [r["within_2x_pct"] for r in ordered])
    ax.set_ylabel("Problem cases within 2× the best median gap (%)")
    ax.set_xticks(x, names, rotation=38, ha="right")
    ax.set_ylim(0, 100)
    ax.set_title("Performance profile (lower rank / higher coverage is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_gap_heatmap(summary_rows: list[dict], out_path: Path) -> None:
    if not summary_rows:
        return
    methods = list(dict.fromkeys(r["method"] for r in summary_rows))
    tasks = sorted({(r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"])) for r in summary_rows}, key=str)
    medmap = {(r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"]), r["method"]): float(r["median_gap"]) for r in summary_rows}
    matrix = np.full((len(tasks), len(methods)), np.nan)
    for i, task in enumerate(tasks):
        for j, method in enumerate(methods):
            matrix[i, j] = np.log10(max(medmap.get((*task, method), np.nan), 1e-15))
    fig, ax = plt.subplots(figsize=(max(10, len(methods)*0.75), max(7, len(tasks)*0.22)))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis_r")
    ax.set_xticks(range(len(methods)), methods, rotation=40, ha="right")
    labels = [f"{f} | d={d}, B={b}, σ={n:g}" for f, d, b, n in tasks]
    ax.set_yticks(range(len(tasks)), labels, fontsize=7)
    ax.set_title("log10 median clean optimality gap (lower is better)")
    fig.colorbar(im, ax=ax, label="log10 gap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_report(out_path: Path, raw_rows: list[dict], summary_rows: list[dict],
                  paired_rows: list[dict], profile_rows: list[dict], adam_tuning: list[dict]) -> None:
    lines = [
        "ABG research v30 FAIR BENCHMARK REPORT",
        "="*78,
        "Scope: bounded black-box optimization on normalized coordinates [-1,1]^d.",
        "Budgets count actual objective calculations, not optimizer iterations or Python calls.",
        "Deterministic runs share exact-point caching; noisy runs disable caching for every method.",
        "Noisy-run gaps are scored using the clean objective at the returned incumbent.",
        "Times are secondary diagnostics; objective-evaluation budget is the primary resource.",
        "Gaps <=1e-12 are treated as numerical zero for ranks and paired inference only.",
        "A missing incumbent/nonfinite gap gets a worst-case penalty of 1e30 in summaries and ranks.",
        "Method names in this report are created from the actual execution registry.",
        "",
        f"Completed raw runs: {len(raw_rows)}",
        f"Methods actually present: {', '.join(sorted({r['method'] for r in raw_rows}))}",
        f"Functions/dimensions: {', '.join(sorted({r['function']+'(d='+str(r['dimension'])+')' for r in raw_rows}))}",
        f"Trial counts per summarized group: {sorted({int(r['n_trials']) for r in summary_rows})}",
        "Gap scale: median absolute clean gap from a fixed 256-point Sobol reference design per function/dimension.",
        "Noise level is relative: actual Gaussian sigma = noise_level × the same fixed reference gap scale.",
        "",
        "OVERALL PERFORMANCE PROFILE (common-coverage methods only; each method appears in every problem case)",
    ]
    for r in sorted(profile_rows, key=lambda q: (q["mean_rank"], -q["within_2x_pct"])):
        lines.append(f"{r['method']:31s} mean-rank={r['mean_rank']:.2f} | wins={r['win_cases']:4d}/{r['n_problem_cases']} | within-2x={r['within_2x_pct']:5.1f}% | within-1.1x={r['within_1_1x_pct']:5.1f}%")
    lines.extend(["", "KEY AGGREGATES BY METHOD (common-coverage methods only)"])
    by_method: dict[str, list[dict]] = {}
    for r in summary_rows:
        by_method.setdefault(r["method"], []).append(r)
    all_tasks = {(r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"])) for r in summary_rows}
    common_names = {m for m, rr in by_method.items() if {(r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"])) for r in rr} == all_tasks}
    for method in sorted(common_names, key=lambda m: np.median(np.log10(np.maximum([r["median_gap"] for r in by_method[m]], 1e-15)))):
        rr = by_method[method]
        logs = np.log10(np.maximum([r["median_gap"] for r in rr], 1e-15))
        lines.append(f"{method:31s} median(log10 gap)={np.median(logs):8.3f} | avg success@1e-3={np.mean([r['success_1e-3_pct'] for r in rr]):5.1f}% | avg success@1e-6={np.mean([r['success_1e-6_pct'] for r in rr]):5.1f}% | median evals={np.median([r['median_evals'] for r in rr]):7.1f} | median seconds={np.median([r['median_seconds'] for r in rr]):.4f}")
    partial_names = set(by_method) - common_names
    if partial_names:
        lines.extend(["", "PARTIAL-COVERAGE METHODS (not ranked against full-coverage methods)"])
        for method in sorted(partial_names):
            rr = by_method[method]
            lines.append(f"{method}: evaluated on {len(rr)}/{len(all_tasks)} problem cases; interpret only within its supported subset.")
    lines.extend(["", "PAIRED ABG-v30 TESTS AGAINST REGISTERED BASELINES"])
    if paired_rows:
        sig = [r for r in paired_rows if int(r["significant_holm_0_05"]) == 1]
        lines.append(f"Paired task comparisons: {len(paired_rows)}; Holm-significant at 0.05 after global correction: {len(sig)}")
        lines.append("Positive median_log10_gap_improvement and positive sign_effect mean ABG's gap was lower.")
        for method in sorted({r["competitor"] for r in paired_rows}):
            rr = [r for r in paired_rows if r["competitor"] == method]
            ci_parts = []
            for r in rr:
                lo = float(r.get("median_log10_gap_improvement_ci95_low", float("nan")))
                hi = float(r.get("median_log10_gap_improvement_ci95_high", float("nan")))
                if np.isfinite(lo) and np.isfinite(hi):
                    ci_parts.append((lo, hi))
            ci_text = (f" | median-improvement CI95 range across tasks=[{min(x[0] for x in ci_parts):+.3f}, {max(x[1] for x in ci_parts):+.3f}]" if ci_parts else "")
            lines.append(f"  vs {method:28s}: median improvement={np.median([r['median_log10_gap_improvement'] for r in rr]):+.3f} log10 | mean sign-effect={np.mean([r['sign_effect_abg_better'] for r in rr]):+.3f} | Holm p<.05 cases={sum(int(r['significant_holm_0_05']) for r in rr)}/{len(rr)}{ci_text}")
    else:
        lines.append("No paired tests could be calculated (too few matched seeds or missing ABG rows).")
    lines.extend(["", "ABG PHASE ACCOUNTING (mean objective evaluations per completed ABG run)"])
    abg_rows = [r for r in raw_rows if str(r["method"]).startswith("ABG")]
    phase_keys: set[str] = set()
    for r in abg_rows:
        try:
            phase_keys.update(json.loads(r.get("phase_counts_json", "{}")))
        except Exception:
            pass
    if abg_rows:
        for phase in sorted(phase_keys):
            vals = []
            for r in abg_rows:
                try:
                    vals.append(float(json.loads(r.get("phase_counts_json", "{}")).get(phase, 0)))
                except Exception:
                    vals.append(0.0)
            lines.append(f"  {phase:24s} mean={np.mean(vals):8.2f} median={np.median(vals):8.2f}")
    lines.extend(["", "ABG PHASE WALL-CLOCK BREAKDOWN (seconds; mean / median across ABG runs)"])
    phase_time_values: dict[str, list[float]] = {}
    for r in abg_rows:
        try:
            phase_map = json.loads(r.get("algorithm_data_json", "{}")).get("phase_seconds", {})
        except Exception:
            phase_map = {}
        for phase, value in phase_map.items():
            try:
                phase_time_values.setdefault(phase, []).append(float(value))
            except (TypeError, ValueError):
                pass
    if phase_time_values:
        for phase, vals in sorted(phase_time_values.items()):
            lines.append(f"  {phase:27s} mean={np.mean(vals):8.4f} median={np.median(vals):8.4f}")
        lines.append("  Note: phase timers include Python-side phase work and are approximate; total run seconds remain the primary wall-clock measurement.")
    else:
        lines.append("No phase timers found; this can occur in checkpoints generated by older script versions.")
    lines.extend(["", "ADAM VALIDATION/TUNING"])
    if adam_tuning:
        for r in adam_tuning:
            lines.append(f"  {r['function']} d={int(r['dimension'])} B={int(r['budget'])} noise level={float(r['noise_level']):g}: selected lr={float(r['selected_lr']):g}; tuning evals={int(float(r['tuning_evals']))}; validation scores={r['scores_json']}")
    else:
        lines.append("No Adam tuning records present.")
    lines.extend([
        "", "PRE-REGISTERED INTERPRETATION RULES",
        "1. Never claim universal superiority from one function, one dimension, or one budget.",
        "2. Compare methods using the same function, dimension, budget, noise level, and seed set.",
        "3. Primary score is clean normalized optimality gap; secondary scores are success@1e-3/1e-6,",
        "   performance-profile coverage, evaluation use, and separately reported wall-clock time.",
        "4. A broad-superiority claim needs >=30 paired test seeds per condition, a lower aggregate",
        "   median log-gap than every major baseline, positive paired improvement, Holm-adjusted p<0.05",
        "   against the relevant baseline family, and no material success-rate regression.",
        "5. The domain-center point replaces one Sobol point, so it consumes no extra initialization budget;",
        "   ABG-v30-no-center-anchor isolates the effect of this generic anchor.",
        "6. Ablation claims require ABG-v30 to outperform the corresponding ablated variant under",
        "   the same budgets and seeds; a single ablation win does not establish causality by itself.",
        "7. If fewer than 30 trials were run, treat all significance and ranking conclusions as exploratory.",
        "8. CMA-ES, DE, DIRECT and Dual Annealing are also constrained by the Oracle budget; optimizer",
        "   iteration counts are not used as a proxy for computational work.",
        "9. Deterministic caching benefits all methods equally. In noisy settings, caching is disabled",
        "   and every repeated observation is charged, avoiding accidental free resampling.",
        "10. The built-in functions are synthetic benchmarks. They do not establish performance on",
        "   expensive real-world, constrained, discontinuous, or high-noise objectives by themselves.",
        "",
        "LIMITATION: This report summarizes observed runs only. It does not prove mathematical global",
        "convergence or guarantee ABG is the best optimizer outside the tested setting.",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_self_tests() -> None:
    """Fast invariants test for functions, dimensions, method registry, and budgets."""
    checked = 0
    assert _effective_gap(float("inf")) == FAILED_GAP_PENALTY
    assert _effective_gap(float("nan")) == FAILED_GAP_PENALTY
    assert _effective_gap(1e-14) == 0.0
    failed_summary = summarize([{
        "function": "FailureProbe", "dimension": 2, "budget": 20, "noise_level": 0.0,
        "method": "invalid-method", "gap": float("inf"), "success_1e-3": 0,
        "success_1e-6": 0, "evals": 0, "seconds": 0.0, "cache_hits": 0,
    }])
    assert failed_summary[0]["median_gap"] == FAILED_GAP_PENALTY
    # Partial-coverage methods must never enter the all-task performance ranking.
    profile_probe = performance_profile([
        {"function":"A","dimension":2,"budget":20,"noise_level":0.0,"method":"ABG-v30","median_gap":1.0},
        {"function":"A","dimension":2,"budget":20,"noise_level":0.0,"method":"DIRECT","median_gap":0.5},
        {"function":"B","dimension":10,"budget":20,"noise_level":0.0,"method":"ABG-v30","median_gap":0.2},
        {"function":"B","dimension":10,"budget":20,"noise_level":0.0,"method":"DIRECT","median_gap":0.3},
        {"function":"A","dimension":2,"budget":20,"noise_level":0.0,"method":"legacy-2D-only","median_gap":0.0},
    ])
    assert {r["method"] for r in profile_probe} == {"ABG-v30", "DIRECT"}
    assert all(r["n_problem_cases"] == 2 for r in profile_probe)
    for test_dim in (2, 10, 50):
        dirs = _ring_directions(test_dim, np.random.default_rng(80_000+test_dim))
        first_pass = np.asarray(dirs[:test_dim])
        assert np.all(np.sum(np.abs(first_pass)>1e-12, axis=1) == 1)
        assert len(set(np.argmax(np.abs(first_pass), axis=1).tolist())) == test_dim
        second_pass = np.asarray(dirs[test_dim:2*test_dim])
        assert np.allclose(first_pass, -second_pass)
    for test_dim in (20, 50):
        test_points = sobol_points(150, test_dim, 70_000+test_dim)
        target = np.linspace(-0.25, 0.25, test_dim)
        test_values = np.sum((test_points-target)**2, axis=1)
        center, q2 = _best_surrogate_center(test_points, test_values)
        assert center is not None and np.max(np.abs(center-target)) < 1e-5 and q2 > 0.99
    for name in FUNCTION_NAMES:
        for dim in (2, 5, 10):
            if name in ("Himmelblau", "Modified_Himmelblau") and dim != 2:
                continue
            problem = build_problem(name, dim)
            assert problem.metric_scale > 0 and np.isfinite(problem.metric_scale)
            test_points = sobol_points(5, dim, 30_000+checked)
            for point in test_points:
                assert np.isfinite(problem.evaluate(point))
                assert point.shape == (dim,) and np.all(point >= -1) and np.all(point <= 1)
            checked += 1
    # Every registered method must finish within the same strict budget contract.
    problem = build_problem("Rastrigin", 2)
    for index, method in enumerate(METHODS):
        row = run_one(method, problem, 50, 51_000+index, 0.0, 0.03)
        assert int(row["evals"]) <= int(row["budget"])
        assert int(row["evals"]) == sum(json.loads(row["phase_counts_json"]).values())
        assert row["error"] == ""
    # Exercise the anti-starvation/audit scheduler with enough budget for multiple splits.
    scheduler_oracle = run_abg(problem, 220, 88_801, 0.0)
    assert scheduler_oracle.evals <= 220
    assert int(scheduler_oracle.algorithm_data.get("splits", 0)) >= 7
    assert int(scheduler_oracle.algorithm_data.get("audits", 0)) >= 1
    assert scheduler_oracle.algorithm_data.get("priority_scoring", "").startswith("heap scheduler")
    assert sum(scheduler_oracle.phase_counts.values()) == scheduler_oracle.evals
    noisy_problem = build_problem("Rosenbrock", 3)
    row = run_one("ABG-v30", noisy_problem, 45, 77_777, 0.02, 0.03)
    assert int(row["evals"]) <= int(row["budget"])
    assert float(row["objective_noise_sigma"]) > 0 and int(row["cache_hits"]) == 0
    print(f"SELF-TEST PASSED: {checked} function/dimension pairs, {len(METHODS)} methods, heap scheduler, audits, hard budgets, phase accounting, noisy-cache policy, and common-coverage ranking.")


def _load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair, budget-capped ABG benchmark with baselines, ablations and statistical tests.")
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--smoke", action="store_true", help="run a small end-to-end correctness smoke test")
    parser.add_argument("--self-test", action="store_true", help="run function/method/budget invariants and exit")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent/"ABG_v30_results")
    parser.add_argument("--fresh", action="store_true", help="discard existing checkpoint and rerun this configuration")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=None,
                        help="relative Gaussian noise levels (0.01 = 1% of the fixed reference gap scale)")
    parser.add_argument("--functions", nargs="+", default=None, choices=FUNCTION_NAMES)
    parser.add_argument("--tune-reps", type=int, default=3)
    parser.add_argument("--adam-lrs", type=float, nargs="+", default=list(ADAM_LR_CANDIDATES))
    return parser.parse_args()


def _task_seed(function_index: int, dim: int, budget: int, noise: float, trial: int) -> int:
    # Deliberately ignore budget/noise in the seed so sweeps are blocked by
    # function × dimension × trial. Paired tests still compare methods per setting.
    return int(DEFAULT_SEED + 100_003*function_index + 10_007*dim + 1_009*trial)


def main() -> int:
    args = _load_args()
    if args.self_test:
        run_self_tests()
        return 0
    if args.smoke:
        dims = [2]; budgets = [100]; noise_levels = [0.0]; trials = 2
        function_names = ["Sphere", "Rastrigin"]
        tune_reps = 1; adam_lrs = (0.01, 0.06, 0.15)
    elif args.profile == "full":
        dims = args.dims or [2, 5, 10]
        budgets = args.budgets or [100, 1000, 10000]
        noise_levels = args.noise_levels or [0.0, 0.01, 0.1]
        trials = args.trials or 30
        function_names = args.functions or list(FUNCTION_NAMES)
        tune_reps = args.tune_reps; adam_lrs = tuple(args.adam_lrs)
    else:
        dims = args.dims or [2, 5]
        budgets = args.budgets or [100, 400]
        noise_levels = args.noise_levels or [0.0, 0.01]
        trials = args.trials or 10
        function_names = args.functions or list(FUNCTION_NAMES)
        tune_reps = args.tune_reps; adam_lrs = tuple(args.adam_lrs)
    if trials < 1 or tune_reps < 1:
        raise ValueError("trials and tune-reps must be >=1")
    if any(b < 20 for b in budgets):
        raise ValueError("budgets must be >=20")
    if any(s < 0 for s in noise_levels):
        raise ValueError("noise levels must be >=0")
    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.outdir/"checkpoint_raw.csv"
    tuning_path = args.outdir/"adam_tuning.csv"
    if args.fresh:
        for path in (checkpoint, tuning_path):
            if path.exists():
                path.unlink()
    rows = read_csv(checkpoint)
    existing = {_key(r) for r in rows}
    tuning_records = read_csv(tuning_path)
    tuning_lookup = {}
    for r in tuning_records:
        tuning_lookup[(r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"]))] = r

    total_cases = 0
    for name in function_names:
        for dim in dims:
            if name in ("Himmelblau", "Modified_Himmelblau") and dim != 2:
                continue
            for budget in budgets:
                for noise_level in noise_levels:
                    total_cases += 1
    done_cases = 0
    for name in function_names:
        for dim in dims:
            if name in ("Himmelblau", "Modified_Himmelblau") and dim != 2:
                continue
            problem = build_problem(name, dim)
            for budget in budgets:
                for noise_level in noise_levels:
                    task = (name, dim, budget, float(noise_level))
                    # Reuse previously selected validation LR only when the record is present.
                    if task in tuning_lookup:
                        selected_lr = float(tuning_lookup[task]["selected_lr"])
                    else:
                        tune_seed = _task_seed(FUNCTION_NAMES.index(name), dim, budget, noise_level, 900_001)
                        selected_lr, tune_evals, tune_scores = tune_adam_lr(
                            problem, budget, float(noise_level), tune_seed, adam_lrs, tune_reps)
                        tr = {
                            "function": name, "dimension": dim, "budget": budget, "noise_level": float(noise_level),
                            "objective_noise_sigma": float(noise_level*problem.metric_scale), "metric_scale": float(problem.metric_scale),
                            "selected_lr": selected_lr, "tuning_evals": tune_evals,
                            "tune_reps": tune_reps, "lrs_tested": json.dumps(list(adam_lrs)),
                            "scores_json": json.dumps({str(k): v for k, v in tune_scores.items()}, sort_keys=True),
                        }
                        tuning_records.append(tr)
                        tuning_lookup[task] = tr
                        write_csv(tuning_path, tuning_records)
                    pending_checkpoint_rows: list[dict] = []
                    checkpoint_batch_size = max(1, min(5, trials))
                    for trial in range(trials):
                        seed = _task_seed(FUNCTION_NAMES.index(name), dim, budget, noise_level, trial)
                        trial_new_rows: list[dict] = []
                        for method in methods_for_case(problem, float(noise_level)):
                            key = (name, dim, budget, float(noise_level), method, seed)
                            if key in existing:
                                continue
                            row = run_one(method, problem, budget, seed, float(noise_level), selected_lr)
                            row["adam_lr"] = selected_lr if method == "Adam-tuned (Sobol-seeded)" else row["adam_lr"]
                            rows.append(row); existing.add(key); trial_new_rows.append(row)
                        pending_checkpoint_rows.extend(trial_new_rows)
                        # Batch checkpoint I/O over at most five seeds. This keeps
                        # a bounded recovery window without a synchronous file flush
                        # after every seed on slower mounted filesystems.
                        if (trial+1) % checkpoint_batch_size == 0 or trial == trials-1:
                            append_csv_rows(checkpoint, pending_checkpoint_rows)
                            pending_checkpoint_rows.clear()
                            print(f"[{done_cases+1}/{total_cases}] {name} d={dim} B={budget} noise level={noise_level:g} | seed {trial+1}/{trials} complete", flush=True)
                    done_cases += 1
    # Sort deterministically; resumed records from older settings cannot distort grouping.
    rows.sort(key=lambda r: (r["function"], int(r["dimension"]), int(r["budget"]), float(r["noise_level"]), r["method"], int(r["seed"])))
    write_csv(args.outdir/"raw_results.csv", rows)
    summaries = summarize(rows)
    write_csv(args.outdir/"summary.csv", summaries)
    paired = paired_tests(rows)
    write_csv(args.outdir/"paired_tests.csv", paired)
    profile = performance_profile(summaries)
    write_csv(args.outdir/"performance_profile.csv", profile)
    plot_profile(profile, args.outdir/"performance_profile.png")
    plot_gap_heatmap(summaries, args.outdir/"gap_heatmap.png")
    _write_report(args.outdir/"report.txt", rows, summaries, paired, profile, tuning_records)
    print("\nCompleted benchmark outputs:")
    for p in (checkpoint, args.outdir/"raw_results.csv", args.outdir/"summary.csv",
              args.outdir/"paired_tests.csv", args.outdir/"performance_profile.csv",
              args.outdir/"report.txt", args.outdir/"performance_profile.png", args.outdir/"gap_heatmap.png"):
        print(f"  {p}")
    print("\nTop performance-profile methods (lower mean rank is better):")
    for r in sorted(profile, key=lambda q: (q["mean_rank"], -q["within_2x_pct"]))[:8]:
        print(f"  {r['method']:31s} mean-rank={r['mean_rank']:.2f} within-2x={r['within_2x_pct']:.1f}%")
    if trials < 30:
        print("\nCAUTION: fewer than 30 test seeds were run; statistical conclusions are exploratory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

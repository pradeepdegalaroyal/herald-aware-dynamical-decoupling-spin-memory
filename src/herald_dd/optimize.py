"""Schedule design: tuned uniform CPMG, geometric spacing and free-form gradient descent."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .noise import NoiseModel
from .objective import expected_infidelity
from .schedules import cpmg_spacings, expand_tail
from .timing import HeraldDistribution

MAX_PULSES = 1200


def _n_cover(tau: float, horizon: float) -> int:
    return int(min(MAX_PULSES, max(4, np.ceil(horizon / tau) + 2)))


def tune_cpmg(noise: NoiseModel, dist: HeraldDistribution, n_grid: int = 40) -> Tuple[float, float]:
    """Best uniform CPMG spacing for the herald distribution (grid search, then a finer grid)."""
    hz = dist.horizon()
    lo, hi = hz / 1000.0, hz / 1.5

    def cost(tau: float) -> float:
        d = cpmg_spacings(tau, _n_cover(tau, hz))
        return float(expected_infidelity(d, noise, dist))

    grid = np.geomspace(lo, hi, n_grid)
    costs = [cost(t) for t in grid]
    i = int(np.argmin(costs))
    fine = np.geomspace(grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)], 15)
    fcosts = [cost(t) for t in fine]
    j = int(np.argmin(fcosts))
    return float(fine[j]), float(fcosts[j])


def fit_geometric(
    noise: NoiseModel, dist: HeraldDistribution, tau_init: float, steps: int = 150, seed: int = 0
) -> Tuple[torch.Tensor, float]:
    """Two-parameter family ``tau0 * g**j`` fitted by Adam on the exact objective."""
    hz = dist.horizon()
    rng = np.random.default_rng(seed)
    n = min(250, _n_cover(tau_init, hz))
    theta = torch.tensor(
        [np.log(tau_init) + 0.1 * rng.standard_normal(), 0.0 + 0.01 * rng.standard_normal()],
        dtype=torch.float64,
        requires_grad=True,
    )
    opt = torch.optim.Adam([theta], lr=0.03)
    best, best_d = np.inf, None
    idx = torch.arange(n, dtype=torch.float64)
    for _ in range(steps):
        opt.zero_grad()
        d = (torch.exp(theta[0]) * torch.exp(0.05 * torch.tanh(theta[1]) * idx)).clamp(max=hz)
        d = torch.cat([0.5 * d[:1], d[1:]])
        loss = expected_infidelity(d, noise, dist)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        if v < best:
            best, best_d = v, d.detach().clone()
    return best_d, best


def fit_freeform(
    noise: NoiseModel,
    dist: HeraldDistribution,
    tau_init: float,
    n_free: int = 64,
    n_tail: Optional[int] = None,
    steps: int = 300,
    lr: float = 0.02,
    seed: int = 0,
    jitter: float = 0.02,
    init_spacings: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, list]:
    """Free-form pulse times: ``n_free`` learned spacings followed by a repeated-last-spacing tail.

    Initialised at the tuned CPMG spacing with multiplicative log-normal jitter (``seed``).
    Returns the best schedule found, its exact objective and the loss history.
    """
    if n_tail is None:
        n_tail = max(32, _n_cover(tau_init, dist.horizon()) - n_free)
    rng = np.random.default_rng(seed)
    base = np.full(n_free, np.log(tau_init))
    base[0] = np.log(0.5 * tau_init)  # CPMG layout: first pulse at tau/2
    if init_spacings is not None:
        base = np.log(init_spacings[:n_free].numpy())
    theta = torch.tensor(base + jitter * rng.standard_normal(n_free), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    best, best_d, hist = np.inf, None, []
    for _ in range(steps):
        opt.zero_grad()
        d = expand_tail(torch.exp(theta), n_tail)
        loss = expected_infidelity(d, noise, dist)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        hist.append(v)
        if v < best:
            best, best_d = v, d.detach().clone()
    return best_d, best, hist


def tune_closing(
    noise: NoiseModel, dist: HeraldDistribution, tau_center: float, n_grid: int = 9, span: float = 1.4
) -> Tuple[float, float]:
    """Best CPMG spacing for the herald-triggered closing policy, searched around ``tau_center``."""
    from .objective import closing_cpmg_infidelity

    grid = tau_center * np.geomspace(1.0 / span, span, n_grid)
    costs = [closing_cpmg_infidelity(float(t), noise, dist) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def cpmg_cost(tau: float, noise: NoiseModel, dist: HeraldDistribution) -> float:
    """Exact expected infidelity of passive uniform CPMG with spacing ``tau`` under ``noise``."""
    d = cpmg_spacings(tau, _n_cover(tau, dist.horizon()))
    return float(expected_infidelity(d, noise, dist).detach())

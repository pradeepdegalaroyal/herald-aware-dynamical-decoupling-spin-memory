"""Monte-Carlo trajectory check of the analytic infidelity (independent of the closed forms)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .noise import NoiseModel
from .objective import crossing_infidelity


def snap(d: torch.Tensor, dt: float) -> torch.Tensor:
    """Round spacings to integer multiples of the simulation step ``dt`` (at least one step)."""
    return (torch.round(d / dt).clamp_min(1.0) * dt).to(torch.float64)


def mc_release_infidelity(
    d: torch.Tensor,
    noise: NoiseModel,
    dt: float,
    n_paths: int,
    seed: int,
    n_points: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample noise paths and compare with :func:`crossing_infidelity` at several release points.

    Quasi-static Gaussian, Ornstein-Uhlenbeck (exact AR(1) update) and white phase noise are
    drawn explicitly, the signed phase integral ``int y(t) delta(t) dt`` is accumulated and the
    coherence is ``E[cos(phi)] (1 - eta)^n``.  ``d`` must be snapped to the ``dt`` grid.
    Returns ``(release_times, analytic_eps, monte_carlo_eps)``.
    """
    z, valid, eps = crossing_infidelity(d, noise)
    idx = np.flatnonzero(valid.numpy())
    pick = idx[np.unique(np.linspace(0, len(idx) - 1, n_points).astype(int))]
    z_pick = z[pick].numpy()
    n_steps = int(round(z_pick.max() / dt)) + 1
    gen = torch.Generator().manual_seed(seed)
    pulse_steps = torch.round(torch.cumsum(d, 0) / dt).long()
    y = torch.ones(n_steps, dtype=torch.float64)
    for ps in pulse_steps:  # sign flips after each pulse
        if ps < n_steps:
            y[ps:] *= -1.0
    rho = float(np.exp(-dt / noise.tau_c))
    x = noise.sigma_d * torch.randn(n_paths, generator=gen, dtype=torch.float64)
    q = noise.sigma_q * torch.randn(n_paths, generator=gen, dtype=torch.float64)
    phi = torch.zeros(n_paths, dtype=torch.float64)
    cos_at = {}
    targets = {int(round(zz / dt)): j for j, zz in enumerate(z_pick)}
    for k in range(n_steps):
        step = (x + q) * dt + np.sqrt(2.0 * noise.gamma0 * dt) * torch.randn(n_paths, generator=gen, dtype=torch.float64)
        # midpoint-style average of the OU value across the step for accuracy
        phi = phi + y[k] * step
        x_new = rho * x + noise.sigma_d * np.sqrt(1.0 - rho**2) * torch.randn(n_paths, generator=gen, dtype=torch.float64)
        x = x_new
        if (k + 1) in targets:
            cos_at[targets[k + 1]] = float(torch.cos(phi).mean())
    mc = np.array([cos_at[j] for j in range(len(z_pick))])
    n_p = pick.astype(float)
    mc_eps = 0.5 * (1.0 - mc * (1.0 - noise.eta) ** n_p)
    return z_pick, eps[pick].detach().numpy(), mc_eps

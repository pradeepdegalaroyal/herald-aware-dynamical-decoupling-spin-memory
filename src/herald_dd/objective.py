"""Exact expected infidelity of a schedule when the release time is random.

For a Gaussian phase with variance ``V`` the stored coherence is ``W = exp(-V/2)`` and a
dephasing channel with coherence ``W`` has process infidelity ``(1 - W) / 2``.

Policy: the herald arrives at a random time ``T``; the memory is released at the first
refocusing point ``Z >= T`` of the schedule (quasi-static field cancelled exactly), so the
price of not knowing ``T`` is the extra storage time ``Z - T`` plus the pulses applied.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .noise import NoiseModel, ou_self_term
from .schedules import refocus_points
from .timing import HeraldDistribution

TAIL_INFIDELITY = 0.5  # fully dephased: charged for herald times beyond the schedule


def crossing_infidelity(d: torch.Tensor, noise: NoiseModel) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infidelity of releasing the memory at the refocusing point of every interval.

    Returns ``(z, valid, eps)``.  The quasi-static part vanishes at a refocusing point;
    the Ornstein-Uhlenbeck part is evaluated with the exact double integral of
    ``exp(-|u-v|/tau_c)`` over the signed intervals (O(P^2), differentiable).
    """
    starts, lengths, z, valid = refocus_points(d)
    p = d.shape[0]
    tau = noise.tau_c
    y = 1.0 - 2.0 * (torch.arange(p) % 2).to(d.dtype)
    a = -torch.expm1(-lengths / tau)
    ends = starts + lengths
    gap = (starts[:, None] - ends[None, :]).clamp_min(0.0)
    below = torch.tril(torch.ones(p, p, dtype=torch.bool), diagonal=-1)
    m = torch.where(below, y[None, :] * a[None, :] * torch.exp(-gap / tau), torch.zeros_like(gap))
    e = m.sum(1)
    g = ou_self_term(lengths, tau) + 2.0 * tau**2 * y * a * e
    cum_excl = torch.cumsum(g, 0) - g
    part = (z - starts).clamp_min(0.0)
    a_part = -torch.expm1(-part / tau)
    var_ou = cum_excl + ou_self_term(part, tau) + 2.0 * tau**2 * y * a_part * e
    chi = 0.5 * noise.sigma_d**2 * var_ou + noise.gamma0 * z
    n_pulses = torch.arange(p, dtype=d.dtype)
    log_w = -chi + n_pulses * float(np.log1p(-noise.eta))
    return z, valid, 0.5 * (1.0 - torch.exp(log_w))


def expected_infidelity(d: torch.Tensor, noise: NoiseModel, dist: HeraldDistribution) -> torch.Tensor:
    """``E_T[infidelity]`` of a schedule under refocus-then-release, exact in ``T``."""
    z, valid, eps = crossing_infidelity(d, noise)
    zv = torch.where(valid, z, torch.zeros_like(z))
    running = torch.cummax(zv, 0).values
    prev = torch.cat([running.new_zeros(1), running[:-1]])
    w = (dist.survival(prev) - dist.survival(zv)) * valid.to(d.dtype)
    tail = dist.survival(running[-1:]) * TAIL_INFIDELITY
    return (w * eps).sum() + tail.sum()


def free_evolution_infidelity(noise: NoiseModel, dist: HeraldDistribution, nq: int = 2000) -> float:
    """Baseline without control: release immediately at ``T`` and dephase freely."""
    t = torch.as_tensor(dist.quadrature(nq), dtype=torch.float64)
    chi = (
        0.5 * noise.sigma_q**2 * t**2
        + 0.5 * noise.sigma_d**2 * ou_self_term(t, noise.tau_c)
        + noise.gamma0 * t
    )
    return float((0.5 * (1.0 - torch.exp(-chi))).mean())


def _interval_variance(lengths: torch.Tensor, tau: float) -> torch.Tensor:
    """Ornstein-Uhlenbeck phase variance / sigma^2 for batched alternating-sign interval lists.

    Zero-length intervals (padding) contribute nothing.  Uses the O(K) recurrence
    ``E_{k+1} = exp(-L_k/tau) E_k + y_k a_k`` for the correlation carried between intervals.
    """
    k_max = lengths.shape[-1]
    a = -torch.expm1(-lengths / tau)
    decay = torch.exp(-lengths / tau)
    e = torch.zeros_like(lengths[..., 0])
    total = torch.zeros_like(e)
    sign = 1.0
    for k in range(k_max):
        total = total + ou_self_term(lengths[..., k], tau) + 2.0 * tau**2 * sign * a[..., k] * e
        e = decay[..., k] * e + sign * a[..., k]
        sign = -sign
    return total


def cpmg_known_time_infidelity(t: torch.Tensor, n: torch.Tensor, noise: NoiseModel) -> torch.Tensor:
    """Infidelity of an ``n``-pulse CPMG sequence ending exactly at the known time ``t``.

    ``t`` and ``n`` broadcast against each other; ``n = 0`` means free evolution.
    """
    t, n = torch.broadcast_tensors(t, n.to(t.dtype))
    k_max = int(n.max()) + 1
    tau = t / n.clamp_min(1.0)
    idx = torch.arange(k_max, dtype=t.dtype)
    lengths = torch.where(idx <= n[..., None], tau[..., None].expand(*t.shape, k_max), torch.zeros(()).to(t.dtype))
    half = torch.where((idx == 0) | (idx == n[..., None]), torch.full((), 0.5, dtype=t.dtype), torch.ones((), dtype=t.dtype))
    lengths = lengths * half
    free = n == 0
    lengths = torch.where(free[..., None], torch.where(idx == 0, t[..., None], torch.zeros((), dtype=t.dtype)), lengths)
    var = _interval_variance(lengths, noise.tau_c)
    chi = 0.5 * noise.sigma_d**2 * var + noise.gamma0 * t
    chi = chi + torch.where(free, 0.5 * noise.sigma_q**2 * t**2, torch.zeros_like(t))
    return 0.5 * (1.0 - torch.exp(-chi + n * float(np.log1p(-noise.eta))))


def oracle_cpmg_infidelity(noise: NoiseModel, dist: HeraldDistribution, n_max: int = 400, nq: int = 150) -> float:
    """Lower reference: the herald time ``T`` is known in advance, best CPMG order chosen per ``T``.

    Candidate pulse counts are every integer up to 40 and a dense geometric grid above, which is
    within a fraction of a percent of the exhaustive minimum for the smooth cost in ``n``.
    """
    t = torch.as_tensor(dist.quadrature(nq), dtype=torch.float64)
    cand = np.unique(np.concatenate([np.arange(0, min(40, n_max) + 1), np.round(np.geomspace(41, n_max, 70))])).astype(int)
    n = torch.as_tensor(cand)
    eps = cpmg_known_time_infidelity(t[:, None], n[None, :], noise)
    return float(eps.min(1).values.mean())


def closing_cpmg_infidelity(tau: float, noise: NoiseModel, dist: HeraldDistribution, nq: int = 300) -> float:
    """Herald-triggered closing policy on a CPMG(tau) background.

    When the herald arrives at ``T`` the controller reads the signed area ``A(T)``.  If the
    current switching sign already drives ``A`` towards zero it simply keeps the sign for
    ``|A|``; otherwise it applies one extra pi pulse at ``T`` and then waits ``|A|``.  The
    memory is released at ``T + |A(T)|`` with the quasi-static field cancelled exactly.
    """
    t = torch.as_tensor(dist.quadrature(nq), dtype=torch.float64)
    half = 0.5 * tau
    m = torch.where(t >= half, torch.floor((t - half) / tau) + 1.0, torch.zeros_like(t))  # pulses before T
    start = torch.where(m > 0, half + (m - 1.0) * tau, torch.zeros_like(t))  # start of current interval
    part = t - start
    sign = 1.0 - 2.0 * (m % 2)
    area_done = torch.where(m > 0, torch.where(m % 2 == 1, half, -half), torch.zeros_like(t))
    area = area_done + sign * part
    heading = sign * area < 0
    wait = area.abs()
    k = int(m.max()) + 3
    idx = torch.arange(k, dtype=torch.float64)[None, :]
    mm = m[:, None]
    full = torch.where(idx == 0, torch.full_like(idx, half), torch.full_like(idx, tau))
    lengths = torch.where(idx < mm, full, torch.zeros_like(full) + 0.0 * mm)
    lengths = torch.where(idx == mm, (part + torch.where(heading, wait, torch.zeros_like(wait)))[:, None], lengths)
    lengths = torch.where((idx == mm + 1) & (~heading)[:, None], wait[:, None], lengths)
    var = _interval_variance(lengths, noise.tau_c)
    z = t + wait
    n_p = m + (~heading).to(t.dtype)
    chi = 0.5 * noise.sigma_d**2 * var + noise.gamma0 * z
    eps = 0.5 * (1.0 - torch.exp(-chi + n_p * float(np.log1p(-noise.eta))))
    return float(eps.mean())

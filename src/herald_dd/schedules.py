"""Pulse schedules and their refocusing points.

A schedule is a vector of positive spacings ``d`` (us); pi pulse ``j`` is applied at
``t_j = sum(d[:j+1])``.  The switching function is ``y = +1`` before the first pulse and
flips sign at every pulse.  The memory can be released at a *refocusing point*: a time at
which the signed area ``A(t) = int y dt`` vanishes, so a quasi-static field leaves no phase.
"""

from __future__ import annotations

from typing import Tuple

import torch


def cpmg_spacings(tau: float, n_pulses: int) -> torch.Tensor:
    """Uniform Carr-Purcell-Meiboom-Gill spacing: first pulse at ``tau/2``, then every ``tau``."""
    d = torch.full((n_pulses,), float(tau), dtype=torch.float64)
    d[0] = 0.5 * tau
    return d


def geometric_spacings(tau0: float, growth: float, n_pulses: int) -> torch.Tensor:
    """Spacings ``tau0 * growth**j`` (growth > 1 spreads pulses out at long times)."""
    j = torch.arange(n_pulses, dtype=torch.float64)
    return tau0 * growth**j


def expand_tail(d: torch.Tensor, extra: int) -> torch.Tensor:
    """Continue a schedule by repeating its last spacing ``extra`` times."""
    if extra <= 0:
        return d
    return torch.cat([d, d[-1:].expand(extra)])


def refocus_points(d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locate the refocusing points of a schedule.

    Returns ``(starts, lengths, z, valid)``: ``starts`` are interval start times,
    ``lengths`` the bounded interval lengths, ``z[i]`` the refocusing time inside interval
    ``i`` and ``valid[i]`` whether such a point exists.  Interval ``i`` has switching sign
    ``(-1)**i`` and is preceded by ``i`` pulses.
    """
    p = d.shape[0]
    edges = torch.cat([d.new_zeros(1), torch.cumsum(d, 0)])
    starts, lengths = edges[:-1], d
    y = 1.0 - 2.0 * (torch.arange(p) % 2).to(d.dtype)
    area_start = torch.cumsum(y * lengths, 0) - y * lengths  # exclusive cumulative signed area
    partial = area_start.abs()
    valid = (y * area_start < 0) & (partial <= lengths * (1.0 + 1e-9))
    z = starts + partial
    return starts, lengths, z, valid

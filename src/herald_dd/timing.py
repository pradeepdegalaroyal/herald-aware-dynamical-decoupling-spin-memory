"""Distributions of the herald arrival time (when the network releases the memory)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

KINDS = ("exponential", "shifted", "lognormal", "gamma")


@dataclass(frozen=True)
class HeraldDistribution:
    """Herald arrival time ``T`` in microseconds, with mean ``mean``.

    kinds: ``exponential`` (memoryless repeat-until-success), ``shifted`` (a fixed
    signalling latency of ``0.5*mean`` followed by an exponential wait), ``lognormal``
    (heavy tail, sigma 0.8) and ``gamma`` (shape 16, nearly deterministic).
    """

    kind: str
    mean: float

    def survival(self, t: torch.Tensor) -> torch.Tensor:
        """``P(T > t)`` as a differentiable torch function."""
        m = self.mean
        if self.kind == "exponential":
            return torch.exp(-t / m)
        if self.kind == "shifted":
            t0, scale = 0.5 * m, 0.5 * m
            return torch.where(t < t0, torch.ones_like(t), torch.exp(-(t - t0) / scale))
        if self.kind == "lognormal":
            s = 0.8
            mu = float(np.log(m) - 0.5 * s * s)
            z = (torch.log(t.clamp_min(1e-12)) - mu) / (s * float(np.sqrt(2.0)))
            return 0.5 * torch.erfc(z)
        if self.kind == "gamma":
            k = 16.0
            return torch.special.gammaincc(torch.full_like(t, k), t.clamp_min(0.0) * k / m)
        raise ValueError(f"unknown kind {self.kind}")

    def _frozen(self):
        m = self.mean
        if self.kind == "exponential":
            return stats.expon(scale=m)
        if self.kind == "shifted":
            return stats.expon(loc=0.5 * m, scale=0.5 * m)
        if self.kind == "lognormal":
            s = 0.8
            return stats.lognorm(s=s, scale=float(np.exp(np.log(m) - 0.5 * s * s)))
        if self.kind == "gamma":
            return stats.gamma(a=16.0, scale=m / 16.0)
        raise ValueError(f"unknown kind {self.kind}")

    def quantile(self, q: np.ndarray) -> np.ndarray:
        """Inverse cumulative distribution function."""
        return self._frozen().ppf(q)

    def quadrature(self, n: int = 400) -> np.ndarray:
        """Equal-probability midpoint nodes for expectations over ``T``."""
        return self.quantile((np.arange(n) + 0.5) / n)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` herald times."""
        return self._frozen().rvs(size=n, random_state=rng)

    def horizon(self, q: float = 0.9995) -> float:
        """Time by which a schedule should cover all but ``1-q`` of the probability mass."""
        return float(self.quantile(np.array([q]))[0])

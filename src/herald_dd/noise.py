"""Classical Gaussian dephasing model for a spin memory with a nuclear-spin bath.

The accumulated phase of the stored superposition is ``phi = int y(t) d(t) dt`` where
``y(t) = +-1`` is the switching function of the pi pulses and ``d(t)`` is the detuning
noise, a sum of three independent parts:

* a quasi-static Gaussian Overhauser field of standard deviation ``sigma_q`` (rad/us),
* an Ornstein-Uhlenbeck component (``sigma_d`` rad/us, correlation time ``tau_c`` us)
  that stands for nuclear spin diffusion / slow dipolar dynamics,
* a white-noise floor with dephasing rate ``gamma0`` (1/us) that no schedule can refocus.

Each pi pulse additionally multiplies the stored coherence by ``1 - eta``.
All presets are illustrative orders of magnitude, not fits to a specific device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import torch


@dataclass(frozen=True)
class NoiseModel:
    """Parameters of the dephasing and pulse-error model (time unit: microsecond)."""

    sigma_q: float  # quasi-static detuning std, rad/us
    sigma_d: float  # Ornstein-Uhlenbeck std, rad/us
    tau_c: float  # Ornstein-Uhlenbeck correlation time, us
    gamma0: float  # white dephasing rate, 1/us
    eta: float  # coherence lost per pi pulse

    def scaled(self, sigma_d: float = 1.0, tau_c: float = 1.0, eta: float = 1.0) -> "NoiseModel":
        """Return a copy with the dynamic bath / pulse error rescaled (mismatch studies)."""
        return replace(
            self,
            sigma_d=self.sigma_d * sigma_d,
            tau_c=self.tau_c * tau_c,
            eta=self.eta * eta,
        )


PRESETS: Dict[str, NoiseModel] = {
    # electron spin: strong hyperfine field, schedule time scale of microseconds
    "electron": NoiseModel(sigma_q=30.0, sigma_d=3.0, tau_c=2.0, gamma0=0.002, eta=0.003),
    # nuclear-spin memory: weak quasi-static field, slow bath, noisier gates, time scale of 0.1-1 ms
    "nuclear": NoiseModel(sigma_q=0.2, sigma_d=0.04, tau_c=150.0, gamma0=1e-4, eta=0.01),
}


def ou_self_term(length: torch.Tensor, tau_c: float) -> torch.Tensor:
    """``int_0^L int_0^L exp(-|u-v|/tau_c) du dv`` for one interval of length ``L``."""
    return 2.0 * tau_c * length - 2.0 * tau_c**2 * (-torch.expm1(-length / tau_c))

"""Fast tests for the schedule objective, policies and the Monte-Carlo cross-check."""

import numpy as np
import pytest
import torch

from herald_dd import (
    PRESETS,
    HeraldDistribution,
    cpmg_spacings,
    expected_infidelity,
    free_evolution_infidelity,
    oracle_cpmg_infidelity,
    refocus_points,
)
from herald_dd.montecarlo import mc_release_infidelity, snap
from herald_dd.noise import ou_self_term
from herald_dd.objective import _interval_variance, closing_cpmg_infidelity, crossing_infidelity
from herald_dd.optimize import cpmg_cost, fit_freeform, tune_cpmg


def test_cpmg_refocuses_every_tau():
    d = cpmg_spacings(1.0, 8)
    _, _, z, valid = refocus_points(d)
    assert torch.allclose(z[valid], torch.arange(1.0, 1.0 + int(valid.sum()), dtype=torch.float64))


def test_hahn_echo_refocuses_at_twice_pulse_time():
    d = torch.tensor([1.0, 1.0], dtype=torch.float64)  # pulse at 1, schedule end at 2
    _, _, z, valid = refocus_points(d)
    assert bool(valid[1]) and float(z[1]) == pytest.approx(2.0)


def test_recurrence_matches_direct_double_integral():
    tau_c = 2.0
    lengths = torch.tensor([[0.3, 0.7, 0.5, 0.2, 0.0]], dtype=torch.float64)
    v = float(_interval_variance(lengths, tau_c))
    edges = np.cumsum([0.0] + lengths[0].tolist())
    n = 3000
    t = np.linspace(0, edges[-1], n, endpoint=False) + edges[-1] / n / 2
    y = np.array([(-1.0) ** np.searchsorted(edges[1:], ti) for ti in t])
    direct = (y[:, None] * y[None, :] * np.exp(-np.abs(t[:, None] - t[None, :]) / tau_c)).sum() * (edges[-1] / n) ** 2
    assert v == pytest.approx(direct, rel=1e-2)


def test_ou_self_term_limits():
    assert float(ou_self_term(torch.tensor(1e-3, dtype=torch.float64), 1e6)) == pytest.approx(1e-6, rel=1e-3)


def test_survival_is_monotone_and_normalised():
    t = torch.linspace(0.0, 50.0, 200, dtype=torch.float64)
    for kind in ("exponential", "shifted", "lognormal", "gamma"):
        s = HeraldDistribution(kind, 3.0).survival(t)
        assert float(s[0]) == pytest.approx(1.0, abs=1e-6)
        assert bool((s[1:] <= s[:-1] + 1e-12).all())


def test_quadrature_mean_matches_mean():
    for kind in ("exponential", "shifted", "lognormal", "gamma"):
        dist = HeraldDistribution(kind, 3.0)
        assert dist.quadrature(4000).mean() == pytest.approx(3.0, rel=0.03)


def test_exact_objective_matches_quadrature_estimate():
    noise, dist = PRESETS["electron"], HeraldDistribution("exponential", 1.0)
    d = cpmg_spacings(0.15, 60)
    exact = float(expected_infidelity(d, noise, dist))
    z, valid, eps = crossing_infidelity(d, noise)
    zs, es = z[valid].numpy(), eps[valid].detach().numpy()
    t = dist.quadrature(4000)
    idx = np.searchsorted(zs, t)
    est = np.where(idx < len(zs), es[np.minimum(idx, len(zs) - 1)], 0.5).mean()
    assert exact == pytest.approx(est, rel=0.02)


def test_monte_carlo_agrees_with_closed_form():
    noise = PRESETS["electron"]
    d = snap(cpmg_spacings(0.2, 20), 0.01)
    z, ana, mc = mc_release_infidelity(d, noise, dt=0.01, n_paths=8000, seed=0, n_points=4)
    assert np.max(np.abs(ana - mc)) < 0.02


def test_oracle_beats_every_unaware_policy():
    noise, dist = PRESETS["electron"], HeraldDistribution("exponential", 1.0)
    orc = oracle_cpmg_infidelity(noise, dist, nq=60)
    tau, cost = tune_cpmg(noise, dist, n_grid=15)
    assert orc <= cost + 1e-9
    assert free_evolution_infidelity(noise, dist) > cost


def test_closing_policy_never_worse_than_passive_at_same_tau():
    noise, dist = PRESETS["electron"], HeraldDistribution("exponential", 1.0)
    tau, cost = tune_cpmg(noise, dist, n_grid=15)
    assert closing_cpmg_infidelity(tau, noise, dist, nq=100) <= cost + 1e-6


def test_freeform_does_not_increase_objective():
    noise, dist = PRESETS["electron"], HeraldDistribution("shifted", 1.0)
    tau, _ = tune_cpmg(noise, dist, n_grid=15)
    d, best, hist = fit_freeform(noise, dist, tau, n_free=16, steps=15, seed=0)
    assert best <= hist[0] + 1e-12
    assert cpmg_cost(tau, noise, dist) < 0.5

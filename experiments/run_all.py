"""Reproduce every result of the study.

    python experiments/run_all.py            # full study, about 10 minutes on a laptop
    python experiments/run_all.py --quick    # smoke run, under one minute

Raw results go to ``results/*.csv|json`` and figures to ``figures/*.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from herald_dd import (  # noqa: E402
    PRESETS,
    HeraldDistribution,
    expected_infidelity,
    free_evolution_infidelity,
    oracle_cpmg_infidelity,
)
from herald_dd.montecarlo import mc_release_infidelity, snap  # noqa: E402
from herald_dd.objective import closing_cpmg_infidelity  # noqa: E402
from herald_dd.optimize import (  # noqa: E402
    cpmg_cost,
    fit_freeform,
    fit_geometric,
    tune_closing,
    tune_cpmg,
)

torch.set_num_threads(6)
KINDS = ["exponential", "shifted", "lognormal", "gamma"]
ORACLE_NQ = 150  # reduced in --quick mode
CAL_SIGMA = 0.2  # log-normal calibration error of the noise spectroscopy used for design


def estimate(noise, rng):
    """Noise parameters as a (imperfect) calibration would report them."""
    z = rng.standard_normal(3) * CAL_SIGMA
    return noise.scaled(sigma_d=float(np.exp(z[0])), tau_c=float(np.exp(z[1])), eta=float(np.exp(z[2])))


def run_policies(noise, dist, seed, steps, preset, extra=None):
    """Design every policy on a calibrated estimate, score it under the true noise."""
    rng = np.random.default_rng(1000 + seed)
    est = estimate(noise, rng)
    tau, _ = tune_cpmg(est, dist)
    d_geo, _ = fit_geometric(est, dist, tau, steps=steps, seed=seed)
    d_ff, _, _ = fit_freeform(est, dist, tau, steps=steps, lr=0.01, seed=seed)
    tau_cl, _ = tune_closing(est, dist, tau)
    row = {
        "cpmg": cpmg_cost(tau, noise, dist),
        "geometric": float(expected_infidelity(d_geo, noise, dist).detach()),
        "freeform": float(expected_infidelity(d_ff, noise, dist).detach()),
        "closing_fixed_tau": closing_cpmg_infidelity(tau, noise, dist),
        "closing_tuned": closing_cpmg_infidelity(tau_cl, noise, dist),
    }
    return row, tau, tau_cl, d_ff


def experiment_scaling(cfgs, seeds, steps, out):
    """Policy comparison over herald distributions and mean herald time."""
    rows, schedules = [], {}
    for preset, means, kinds in cfgs:
        noise = PRESETS[preset]
        for mean in means:
            for kind in kinds:
                dist = HeraldDistribution(kind, mean)
                t0 = time.time()
                base = {
                    "oracle_known_T": oracle_cpmg_infidelity(noise, dist, nq=ORACLE_NQ),
                    "free_evolution": free_evolution_infidelity(noise, dist),
                }
                for s in range(seeds):
                    row, tau, tau_cl, d_ff = run_policies(noise, dist, s, steps, preset)
                    row.update(base)
                    row.update(preset=preset, kind=kind, mean_T_us=mean, seed=s, tau_cpmg_us=tau, tau_closing_us=tau_cl)
                    rows.append(row)
                    if s == 0:
                        schedules[f"{preset}|{kind}|{mean}"] = d_ff.numpy()[:80].tolist()
                print(f"  scaling {preset} {kind} mean={mean}: {time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "policy_comparison.csv", index=False)
    (out / "freeform_schedules.json").write_text(json.dumps(schedules))
    return df


def experiment_heatmap(means, etas, out):
    """Ignorance penalty and closing-policy recovery over (mean herald time, pulse error)."""
    rows = []
    base = PRESETS["electron"]
    for eta in etas:
        noise = base.scaled(eta=eta / base.eta)
        for mean in means:
            dist = HeraldDistribution("exponential", mean)
            orc = oracle_cpmg_infidelity(noise, dist, nq=ORACLE_NQ)
            tau, cp = tune_cpmg(noise, dist)
            tau_cl, cl = tune_closing(noise, dist, tau)
            rows.append(dict(eta=eta, mean_T_us=mean, oracle=orc, cpmg=cp, closing=cl, tau_cpmg_us=tau, tau_closing_us=tau_cl))
        print(f"  heatmap eta={eta} done", flush=True)
    df = pd.DataFrame(rows)
    df["penalty"] = df.cpmg / df.oracle
    df["recovered"] = (df.cpmg - df.closing) / (df.cpmg - df.oracle)
    df.to_csv(out / "penalty_heatmap.csv", index=False)
    return df


def experiment_robustness(scales, seeds, steps, out):
    """Design on a calibrated estimate, evaluate under rescaled true bath parameters."""
    rows = []
    nominal = PRESETS["electron"]
    dist = HeraldDistribution("exponential", 1.0)
    for s in range(seeds):
        rng = np.random.default_rng(2000 + s)
        est = estimate(nominal, rng)
        tau, _ = tune_cpmg(est, dist)
        tau_cl, _ = tune_closing(est, dist, tau)
        d_ff, _, _ = fit_freeform(est, dist, tau, steps=steps, lr=0.01, seed=s)
        for sd in scales:
            for tc in scales:
                true = nominal.scaled(sigma_d=sd, tau_c=tc)
                orc = oracle_cpmg_infidelity(true, dist, nq=ORACLE_NQ)
                rows.append(
                    dict(
                        seed=s, sigma_d_scale=sd, tau_c_scale=tc, oracle=orc,
                        cpmg=cpmg_cost(tau, true, dist),
                        closing=closing_cpmg_infidelity(tau_cl, true, dist),
                        freeform=float(expected_infidelity(d_ff, true, dist).detach()),
                    )
                )
        print(f"  robustness seed {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness_mismatch.csv", index=False)
    return df


def experiment_validation(seeds, n_paths, out):
    """Monte-Carlo trajectories versus the analytic infidelity at release points."""
    from herald_dd.schedules import cpmg_spacings

    rows = []
    for preset, tau, dt, n_p in [("electron", 0.15, 0.01, 60), ("nuclear", 20.0, 1.0, 40)]:
        noise = PRESETS[preset]
        rng = np.random.default_rng(7)
        irregular = torch.tensor(tau * np.exp(0.4 * rng.standard_normal(n_p)), dtype=torch.float64)
        for name, d in [("cpmg", cpmg_spacings(tau, n_p)), ("irregular", irregular)]:
            d = snap(d, dt)
            for s in range(seeds):
                z, ana, mc = mc_release_infidelity(d, noise, dt, n_paths, seed=s)
                for zi, a, m in zip(z, ana, mc):
                    rows.append(dict(preset=preset, schedule=name, seed=s, release_us=float(zi), analytic=float(a), monte_carlo=float(m)))
        print(f"  validation {preset} done", flush=True)
    df = pd.DataFrame(rows)
    df["abs_err"] = (df.analytic - df.monte_carlo).abs()
    df.to_csv(out / "montecarlo_validation.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids, under one minute")
    ap.add_argument("--out", default=None, help="output root (default: repo root; temp dir with --quick)")
    args = ap.parse_args()
    root = Path(args.out) if args.out else (Path(tempfile.mkdtemp(prefix="herald_dd_quick_")) if args.quick else ROOT)
    res, fig = root / "results", root / "figures"
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    global ORACLE_NQ
    if args.quick:
        ORACLE_NQ = 50
        cfgs = [("electron", [1.0], ["exponential"])]
        seeds, steps = 2, 30
        means, etas, scales, mc_paths, mc_seeds = [0.5, 2.0], [0.003, 0.03], [0.5, 1.0], 1000, 1
    else:
        cfgs = [
            ("electron", [0.3, 1.0, 3.0, 10.0], KINDS),
            ("nuclear", [100.0, 300.0, 1000.0], ["exponential", "shifted"]),
        ]
        seeds, steps = 5, 150
        means = [0.3, 0.6, 1.0, 2.0, 4.0, 8.0]
        etas = [0.001, 0.003, 0.01, 0.03, 0.1]
        scales, mc_paths, mc_seeds = [0.5, 1.0, 2.0], 20000, 3

    print("scaling study", flush=True)
    scaling = experiment_scaling(cfgs, seeds, steps, res)
    print("penalty heatmap", flush=True)
    heat = experiment_heatmap(means, etas, res)
    print("robustness", flush=True)
    robust = experiment_robustness(scales, 1 if args.quick else 3, steps, res)
    print("validation", flush=True)
    val = experiment_validation(mc_seeds, mc_paths, res)
    meta = dict(
        quick=bool(args.quick), seeds=seeds, runtime_s=round(time.time() - t_start, 1),
        mc_max_abs_err=float(val.abs_err.max()), mc_mean_abs_err=float(val.abs_err.mean()),
        torch=torch.__version__, numpy=np.__version__,
    )
    (res / "run_meta.json").write_text(json.dumps(meta, indent=2))

    from plots import make_all_figures  # noqa: E402

    make_all_figures(scaling, heat, robust, val, res, fig)
    from summarize import summarize  # noqa: E402

    summarize(scaling, heat, robust, val, res)
    print(f"done in {time.time() - t_start:.0f}s -> {root}")


if __name__ == "__main__":
    main()

"""Aggregate raw results into the tables and hypothesis checks quoted in the README."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = ["cpmg", "geometric", "freeform", "closing_fixed_tau", "closing_tuned"]


def summarize(scaling: pd.DataFrame, heat: pd.DataFrame, robust: pd.DataFrame, val: pd.DataFrame, res: Path) -> dict:
    d = scaling.copy()
    for p in POLICIES:
        d[f"{p}_ratio"] = d[p] / d["oracle_known_T"]
        d[f"{p}_gain_pct"] = 100.0 * (1.0 - d[p] / d["cpmg"])
    d["recovered"] = (d["cpmg"] - d["closing_tuned"]) / (d["cpmg"] - d["oracle_known_T"])
    keys = ["preset", "kind", "mean_T_us"]
    agg = d.groupby(keys).agg(
        oracle=("oracle_known_T", "first"),
        free_evolution=("free_evolution", "first"),
        tau_cpmg_us=("tau_cpmg_us", "mean"),
        **{f"{p}_mean": (p, "mean") for p in POLICIES},
        **{f"{p}_std": (p, "std") for p in POLICIES},
        **{f"{p}_ratio_mean": (f"{p}_ratio", "mean") for p in POLICIES},
        **{f"{p}_ratio_std": (f"{p}_ratio", "std") for p in POLICIES},
        freeform_gain_pct_mean=("freeform_gain_pct", "mean"),
        freeform_gain_pct_std=("freeform_gain_pct", "std"),
        closing_gain_pct_mean=("closing_tuned_gain_pct", "mean"),
        closing_gain_pct_std=("closing_tuned_gain_pct", "std"),
        recovered_mean=("recovered", "mean"),
        recovered_std=("recovered", "std"),
    ).reset_index()
    agg["mean_T_over_tau"] = agg.mean_T_us / agg.tau_cpmg_us
    agg.to_csv(res / "policy_summary.csv", index=False)

    out: dict = {}
    # H1: ignorance penalty of tuned CPMG
    agg["penalty"] = agg.cpmg_ratio_mean
    out["H1_penalty_min"] = float(agg.penalty.min())
    out["H1_penalty_max"] = float(agg.penalty.max())
    out["H1_share_configs_penalty_ge_10pct"] = float((agg.penalty >= 1.10).mean())
    out["H1_corr_penalty_vs_log_T_over_tau"] = float(np.corrcoef(np.log(agg.mean_T_over_tau), agg.penalty)[0, 1])
    # H2: closing policy recovery
    out["H2_recovered_median"] = float(agg.recovered_mean.median())
    out["H2_recovered_min"] = float(agg.recovered_mean.min())
    out["H2_recovered_max"] = float(agg.recovered_mean.max())
    out["H2_share_configs_recovered_ge_20pct"] = float((agg.recovered_mean >= 0.20).mean())
    out["H2_share_runs_closing_better_than_cpmg"] = float((d.closing_tuned < d.cpmg).mean())
    out["H2_closing_gain_pct_median"] = float(d.closing_tuned_gain_pct.median())
    out["H2_closing_gain_pct_max"] = float(d.closing_tuned_gain_pct.max())
    # H3: learned schedules versus tuned CPMG
    non_mem = agg[agg.kind != "exponential"]
    out["H3_freeform_gain_pct_nonmemoryless_max"] = float(non_mem.freeform_gain_pct_mean.max())
    out["H3_freeform_gain_pct_nonmemoryless_median"] = float(non_mem.freeform_gain_pct_mean.median())
    out["H3_share_nonmemoryless_configs_ge_5pct"] = float((non_mem.freeform_gain_pct_mean >= 5.0).mean())
    out["H3_freeform_gain_pct_all_median"] = float(agg.freeform_gain_pct_mean.median())
    out["H3_geometric_gain_pct_all_median"] = float(d.geometric_gain_pct.median())
    out["H3_share_runs_freeform_better_than_cpmg"] = float((d.freeform < d.cpmg).mean())
    # ablation: re-tuned spacing for the closing policy
    out["ablation_closing_retuned_vs_fixed_gain_pct_median"] = float(100.0 * (1.0 - d.closing_tuned / d.closing_fixed_tau).median())
    # heatmap summary
    out["heatmap_penalty_min"] = float(heat.penalty.min())
    out["heatmap_penalty_max"] = float(heat.penalty.max())
    out["heatmap_recovered_min"] = float(heat.recovered.min())
    out["heatmap_recovered_max"] = float(heat.recovered.max())
    out["heatmap_recovered_median"] = float(heat.recovered.median())
    # H4: robustness
    r = robust.copy()
    r["cpmg_ratio"] = r.cpmg / r.oracle
    r["closing_ratio"] = r.closing / r.oracle
    r["freeform_ratio"] = r.freeform / r.oracle
    cell = r.groupby(["sigma_d_scale", "tau_c_scale"])[["cpmg_ratio", "closing_ratio", "freeform_ratio"]].mean()
    out["H4_share_cells_closing_beats_cpmg"] = float((cell.closing_ratio < cell.cpmg_ratio).mean())
    out["H4_closing_ratio_max"] = float(cell.closing_ratio.max())
    out["H4_cpmg_ratio_max"] = float(cell.cpmg_ratio.max())
    out["H4_freeform_ratio_max"] = float(cell.freeform_ratio.max())
    # validation
    out["validation_max_abs_err"] = float(val.abs_err.max())
    out["validation_mean_abs_err"] = float(val.abs_err.mean())
    out["n_scaling_runs"] = int(len(d))
    (res / "summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys

    res = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
    summarize(
        pd.read_csv(res / "policy_comparison.csv"), pd.read_csv(res / "penalty_heatmap.csv"),
        pd.read_csv(res / "robustness_mismatch.csv"), pd.read_csv(res / "montecarlo_validation.csv"), res,
    )
    print((res / "summary.json").read_text())

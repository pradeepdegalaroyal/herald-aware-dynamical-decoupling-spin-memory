"""Figures for the study (Okabe-Ito palette, 200 dpi). All plotted numbers come from results/*.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7")
POLICY_STYLE = {
    "cpmg": ("Tuned uniform CPMG, release at next refocus", OI["black"], "o"),
    "geometric": ("Geometric spacing (2 parameters)", OI["sky"], "s"),
    "freeform": ("Free-form spacings (gradient-trained)", OI["green"], "^"),
    "closing_tuned": ("Herald-triggered closing on CPMG", OI["vermillion"], "D"),
}
DPI = 200


def _annotated(ax, mat, xt, yt, fmt, cmap, vmin=None, vmax=None):
    im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xt)), [f"{x:g}" for x in xt])
    ax.set_yticks(range(len(yt)), [f"{y:g}" for y in yt])
    lo, hi = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            frac = (v - (vmin if vmin is not None else lo)) / max(1e-12, (vmax if vmax is not None else hi) - (vmin if vmin is not None else lo))
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="white" if frac < 0.5 else "black")
    return im


def fig_heatmap(heat: pd.DataFrame, path: Path) -> None:
    means = sorted(heat.mean_T_us.unique())
    etas = sorted(heat.eta.unique())
    pen = heat.pivot(index="eta", columns="mean_T_us", values="penalty").loc[etas, means].values
    rec = heat.pivot(index="eta", columns="mean_T_us", values="recovered").loc[etas, means].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = _annotated(axes[0], pen, means, etas, "{:.2f}", "cividis")
    axes[0].set_title("Ignorance penalty: tuned CPMG / known-time optimum")
    fig.colorbar(im0, ax=axes[0], label="infidelity ratio (1 = no penalty)")
    im1 = _annotated(axes[1], rec, means, etas, "{:.2f}", "cividis", vmin=0.0)
    axes[1].set_title("Fraction of the penalty removed by the closing pulse")
    fig.colorbar(im1, ax=axes[1], label="(CPMG - closing) / (CPMG - known-time optimum)")
    for ax in axes:
        ax.set_xlabel("mean herald time (µs)")
        ax.set_ylabel("coherence lost per π pulse, η")
    fig.suptitle("Electron-spin preset, exponential herald time", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_scaling(df: pd.DataFrame, path: Path) -> None:
    d = df[df.preset == "electron"].copy()
    kinds = ["exponential", "shifted", "lognormal", "gamma"]
    titles = {"exponential": "exponential (memoryless)", "shifted": "latency + exponential",
              "lognormal": "log-normal (heavy tail)", "gamma": "gamma, shape 16 (near-deterministic)"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, kind in zip(axes, kinds):
        sub = d[d.kind == kind]
        if sub.empty:
            ax.set_visible(False)
            continue
        means = sorted(sub.mean_T_us.unique())
        for key, (label, color, marker) in POLICY_STYLE.items():
            ratio = sub[key] / sub["oracle_known_T"]
            g = ratio.groupby(sub.mean_T_us)
            mu, sd = g.mean().loc[means].values, g.std().loc[means].values
            ax.errorbar(means, mu, yerr=sd, color=color, marker=marker, ms=5, lw=1.4, capsize=2, label=label)
        ax.axhline(1.0, color=OI["purple"], ls="--", lw=1.2, label="Known-time optimum (oracle)")
        ax.set_xscale("log")
        ax.set_title(titles[kind], fontsize=10)
        ax.set_xlabel("mean herald time (µs)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("expected infidelity / known-time optimum")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Electron-spin preset, mean ± std over 5 calibration seeds", y=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_violin(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    rng = np.random.default_rng(0)
    for ax, preset in zip(axes, ["electron", "nuclear"]):
        sub = df[df.preset == preset]
        keys = ["geometric", "freeform", "closing_tuned"]
        data = [100.0 * (1.0 - sub[k] / sub["cpmg"]) for k in keys]
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for body, k in zip(parts["bodies"], keys):
            body.set_facecolor(POLICY_STYLE[k][1])
            body.set_alpha(0.35)
        for i, (vals, k) in enumerate(zip(data, keys), start=1):
            ax.scatter(i + 0.12 * rng.standard_normal(len(vals)), vals, s=14, color=POLICY_STYLE[k][1], edgecolor="black", lw=0.3, zorder=3)
            ax.hlines(vals.median(), i - 0.3, i + 0.3, color="black", lw=1.5, zorder=4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks([1, 2, 3], ["geometric", "free-form", "closing pulse"])
        ax.set_title(f"{preset}-spin preset ({len(sub)} config × seed runs)")
        ax.set_ylabel("infidelity reduction vs tuned CPMG (%)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-run gain over the tuned uniform baseline (dots = individual seeds, bar = median)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_robust(rob: pd.DataFrame, path: Path) -> None:
    scales = sorted(rob.sigma_d_scale.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    vmax = 0.0
    mats = {}
    for key in ["cpmg", "closing", "freeform"]:
        r = (rob[key] / rob["oracle"]).groupby([rob.sigma_d_scale, rob.tau_c_scale]).mean().unstack()
        mats[key] = r.loc[scales, scales].values
        vmax = max(vmax, np.nanmax(mats[key]))
    names = {"cpmg": "tuned uniform CPMG", "closing": "closing pulse on CPMG", "freeform": "free-form spacings"}
    vmin = min(np.nanmin(m) for m in mats.values())
    for ax, key in zip(axes, ["cpmg", "closing", "freeform"]):
        im = _annotated(ax, mats[key], scales, scales, "{:.3f}", "cividis", vmin=vmin, vmax=vmax)
        ax.set_title(names[key])
        ax.set_xlabel("true / assumed correlation time τc")
    axes[0].set_ylabel("true / assumed bath amplitude σd")
    fig.colorbar(im, ax=axes, label="infidelity / known-time optimum", shrink=0.9)
    fig.suptitle("Robustness to a mis-specified slow bath (design at the nominal value; mean of 3 seeds)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_validation(val: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for (preset, sched), color, marker in zip(
        [("electron", "cpmg"), ("electron", "irregular"), ("nuclear", "cpmg"), ("nuclear", "irregular")],
        [OI["blue"], OI["orange"], OI["green"], OI["vermillion"]], ["o", "s", "^", "D"],
    ):
        s = val[(val.preset == preset) & (val.schedule == sched)]
        ax.scatter(s.analytic, s.monte_carlo, s=16, color=color, marker=marker, label=f"{preset}, {sched}", alpha=0.8)
    lim = [0, max(val.analytic.max(), val.monte_carlo.max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("analytic infidelity at release point")
    ax.set_ylabel("Monte-Carlo infidelity (noise paths)")
    ax.set_title("Closed form vs explicit trajectories")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def make_all_figures(scaling, heat, robust, val, res: Path, fig: Path) -> None:
    fig_heatmap(heat, fig / "fig1_penalty_heatmap.png")
    fig_scaling(scaling, fig / "fig2_policy_scaling.png")
    fig_violin(scaling, fig / "fig3_gain_over_cpmg_by_seed.png")
    fig_robust(robust, fig / "fig4_bath_mismatch.png")
    fig_validation(val, fig / "fig5_montecarlo_validation.png")

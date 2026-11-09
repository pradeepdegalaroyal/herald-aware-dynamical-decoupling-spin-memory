# Method

## Physical model

A single spin qubit stores a superposition. Its phase is `phi = ∫ y(t) δ(t) dt`, where `y(t) = ±1` flips at every
π pulse and `δ(t)` is the detuning from the nuclear-spin bath, modelled as three independent Gaussian parts:

* a quasi-static Overhauser field, standard deviation `σq` (rad/µs), constant within one run;
* an Ornstein-Uhlenbeck (OU) process, amplitude `σd`, correlation time `τc`, standing for nuclear spin diffusion;
* a white floor with rate `γ0` that no pulse sequence can refocus.

The coherence is `W = exp(-V/2) (1-η)^n`, with `V` the phase variance, `n` the number of π pulses and `η` the
coherence lost per pulse. The process infidelity of the resulting dephasing channel is `(1 - W)/2`.
The OU variance is evaluated exactly: for intervals of lengths `L_k` and signs `y_k`,
`V_OU / σd² = Σ_k s(L_k) + 2 τc² Σ_{i<k} y_i y_k a_i a_k exp(-gap_ik / τc)`, with `a = 1 - exp(-L/τc)` and
`s(L) = 2 τc L - 2 τc² a`. A linear recurrence makes this O(P) in the number of intervals `P`.
Two presets (electron spin, nuclear-spin memory) set the orders of magnitude; they are illustrative.
Because the quasi-static term vanishes exactly at every refocusing point, `σq` only affects the no-control baseline;
all refocus-and-release policies depend on the dimensionless set `σd·τc`, `γ0·τc`, `η` and `T/τc`.
Finite pulse duration and off-resonance pulse errors (where `σq` would matter) are folded into the single number `η`.

## Release-time problem

The herald arrives at a random time `T` (four laws: exponential, latency plus exponential, log-normal, gamma).
The memory can be released only at a *refocusing point*, a time where the signed area `A(t) = ∫ y dt` vanishes, so the
quasi-static field leaves no phase. Every policy is scored by `E_T[(1 - W(Z(T)))/2]`, where `Z(T) ≥ T` is its release time.
For a fixed schedule the expectation is exact: it is a sum over refocusing points weighted by the survival function.

## Policies

| Policy | Role | What it does |
|---|---|---|
| Free evolution | classical baseline | No control; release at `T` |
| Tuned uniform CPMG | standard quantum baseline | Carr-Purcell-Meiboom-Gill (CPMG) spacing `τ` tuned on the exact objective; release at the next refocusing point |
| Geometric spacing | learned, 2 parameters | `τ0·g^j`, trained by gradient descent (Adam) |
| Free-form spacings | learned, 64 free spacings plus a repeated tail | gradient-trained on the exact objective in PyTorch |
| Herald-triggered closing (proposed) | online policy | At `T` read `A(T)`; if `y` already drives `A` to zero wait `\|A\|`, else apply one extra pulse and wait `\|A\|` |
| Known-time optimum | reference (not implementable) | Best CPMG pulse count for each `T`, ending exactly at `T` |

## Seeds and design uncertainty

A seed is one *calibration realisation*: the design stage sees `σd`, `τc` and `η` multiplied by independent
log-normal factors (log-standard-deviation 0.2), mimicking imperfect noise spectroscopy; it also fixes the optimiser
initial jitter. All policies are scored under the **true** parameters. The reference policies need no seed.
Five seeds are used.

## Experiments

1. **Policy comparison and scaling** — electron preset at mean herald times 0.3-10 µs and nuclear preset at
   100-1000 µs, over the herald laws (4 for the electron preset, 2 for the nuclear preset), 5 seeds. The nuclear preset
   differs from the electron preset in `σd·τc`, `γ0·τc` and `η`, not only in time scale.
2. **Penalty heatmap** — electron preset, exponential herald, mean time × pulse error `η`: ignorance penalty and the share
   of it removed by the closing policy (deterministic, exact).
3. **Ablation** — parametrisation of the learned schedule (uniform, geometric, free-form) and whether the closing
   policy re-tunes the spacing or reuses the CPMG optimum (columns `closing_fixed_tau`, `closing_tuned`).
4. **Noise/robustness sweep** — designs made at nominal parameters are evaluated with the true OU amplitude and
   correlation time scaled by 0.5, 1 and 2; 3 seeds.
5. **Validation** — explicit Monte-Carlo noise paths (quasi-static, exact OU update, white) versus the closed form
   at release points of regular and irregular schedules.

## Reproducibility

`python experiments/run_all.py` reproduces everything; `--quick` is a smoke run. Time unit is the microsecond.

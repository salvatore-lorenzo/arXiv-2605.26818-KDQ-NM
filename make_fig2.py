"""Reproduce Fig. 2: non-Markovianity and non-positivity versus detuning.

The RHP and LFS non-Markovianity measures and the KDQ non-positivity
functional, each averaged over the collisions (a time-averaged witness per
collision, rather than a raw cumulative sum that would just grow with
N_COLLISIONS), as functions of the system-memory detuning
Delta_omega = omega_S - omega_M. The memory and environment are kept
resonant with each other; only omega_S is swept.

Edit the constants in this file to change parameters. There is intentionally
no command-line parser. N_COLLISIONS = 10000 is larger than the nominal 1000
collisions used elsewhere: this scan's witnesses do not decay with the
number of collisions (the model is unitary, with no relaxation to a fixed
point), so a longer run reduces finite-N noise and sharpens the detuning
dependence. Lower it for faster iteration while testing changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from definitions import (
    CollisionParameters,
    bath_state,
    evolve_sme,
    intermediate_maps,
    kdq_non_positivity,
    lfs_non_monotonicity,
    phase_covariant_parameters,
    reduced_system_maps,
    reduced_system_states,
    rhp_witness,
    thermal_qubit,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 1.0,
    }
)

# Baseline parameters from the Fig. 1 caption; omega_s is swept via the
# detuning below, omega_m and omega_e stay fixed and resonant with each
# other.
N_COLLISIONS = 10000
BASE_PARAMS = CollisionParameters(
    omega_s=1.0,
    omega_m=1.0,
    omega_e=1.0,
    g_sm=0.2,
    g_me=0.2,
    tau_sm=0.2,
    tau_me=0.2,
    beta_s=0.0,
    beta_m=1.0,
    beta_e=1.0,
    bath_coherence=0.0,
    gamma=0.0,
    n_steps=N_COLLISIONS,
)

DETUNING = np.linspace(-0.5, 0.5, 201)
OUTPUT_PATH = Path(__file__).with_name("fig_2.pdf")
SAVE_FIGURE = True
SHOW_FIGURE = False

LINE_WIDTH = 3.0

COLORS = {
    "green": "#8FB032",
    "blue": "#5E81B5",
    "orange": "#E19C24",
}


@dataclass(frozen=True)
class DetuningScanResults:
    """Cumulative witnesses collected over a detuning scan."""

    detuning: np.ndarray
    i_rhp: np.ndarray
    i_lfs: np.ndarray
    sum_nq: np.ndarray


def cumulative_measures(params: CollisionParameters) -> tuple[float, float, float]:
    """Run one collision-model trajectory; return (I_RHP, I_LFS, N_q), each
    averaged over the n_steps collisions."""

    rho_s0 = thermal_qubit(params.beta_s, params.omega_s)
    rho_m0 = thermal_qubit(params.beta_m, params.omega_m)
    rho_e0 = bath_state(params.beta_e, params.omega_e, params.bath_coherence)

    system_maps = reduced_system_maps(params, rho_m0, rho_e0)
    two_time_maps = intermediate_maps(system_maps)

    a, b, _ = phase_covariant_parameters(two_time_maps)
    a = np.real_if_close(a).astype(float)
    b = np.real_if_close(b).astype(float)

    rho_sme = evolve_sme(rho_s0, rho_m0, rho_e0, params)
    rho_s = reduced_system_states(rho_sme)

    # Normalized by the number of collisions: a time-averaged witness per
    # collision, rather than the raw (monotonically growing) cumulative sum.
    i_rhp = float(np.mean(rhp_witness(two_time_maps)))
    i_lfs = float(np.mean(lfs_non_monotonicity(two_time_maps)))
    sum_nq = float(np.mean(kdq_non_positivity(a, b, rho_s)))
    return i_rhp, i_lfs, sum_nq


def simulate_fig2(
    detuning: np.ndarray = DETUNING,
    base_params: CollisionParameters = BASE_PARAMS,
) -> DetuningScanResults:
    """Scan the system-memory detuning, accumulating the three witnesses."""

    i_rhp = np.empty_like(detuning)
    i_lfs = np.empty_like(detuning)
    sum_nq = np.empty_like(detuning)

    for idx, delta_omega in enumerate(detuning):
        params = dataclasses.replace(base_params, omega_s=base_params.omega_m + delta_omega)
        i_rhp[idx], i_lfs[idx], sum_nq[idx] = cumulative_measures(params)
        if (idx + 1) % 10 == 0 or idx + 1 == len(detuning):
            print(f"  scanned {idx + 1}/{len(detuning)} detuning points")

    return DetuningScanResults(detuning=detuning, i_rhp=i_rhp, i_lfs=i_lfs, sum_nq=sum_nq)


def plot_fig2(results: DetuningScanResults) -> plt.Figure:
    """Plot I_RHP, I_LFS and N_q, averaged per collision, versus the detuning."""

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    ax.plot(
        results.detuning, results.i_rhp,
        color=COLORS["orange"], linestyle="-", linewidth=LINE_WIDTH,
        label=r"$\mathcal{I}_{RHP}/n_{\max}$",
    )
    ax.plot(
        results.detuning, results.i_lfs,
        color=COLORS["green"], linestyle="--", linewidth=LINE_WIDTH,
        label=r"$\mathcal{I}^\Phi_{LFS}/n_{\max}$",
    )
    ax.plot(
        results.detuning, results.sum_nq,
        color=COLORS["blue"], linestyle="-.", linewidth=LINE_WIDTH,
        label=r"$\sum_n \mathcal{N}_q[P_n]/n_{\max}$",
    )

    ax.set_xlabel(r"$\Delta\omega$")
    ax.set_yscale("log")
    ax.legend(frameon=True, 
              framealpha=0.7, 
              borderpad=0.2, 
              edgecolor=None, 
              ncols=1,
              labelspacing=1,)
    ax.grid(True, which="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim([results.detuning.min(), results.detuning.max()])

    return fig


def main() -> None:
    print(f"Scanning {len(DETUNING)} detuning points with N_COLLISIONS={N_COLLISIONS}...")
    results = simulate_fig2()
    fig = plot_fig2(results)

    if SAVE_FIGURE:
        fig.savefig(OUTPUT_PATH, dpi=200)
        print(f"Saved figure to {OUTPUT_PATH}")

    peak_idx = int(np.argmax(results.i_rhp))
    print(f"I_RHP peak at Delta_omega = {results.detuning[peak_idx]:.3f}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

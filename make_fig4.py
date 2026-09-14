"""Reproduce Fig. 4.

Non-Markovianity measures (RHP, LFS) and KDQ non-positivity, cumulated over
the collisions and normalized by the number of collisions (time-averaged
witness per collision), as functions of the S-M coupling anisotropy gamma
in [-1, 1]. The M-E collision is kept energy-preserving (isotropic) while
only the S-M collision Hamiltonian is anisotropic:
H_SM = (1-gamma)/2 sx*sx + (1+gamma)/2 sy*sy + sz*sz.

Also produces a second, separate diagnostic figure: g_n, N_q[P_n] and
Delta I_LS^(n) versus collision number, one subplot per gamma value, to show
the underlying per-collision behavior behind the gamma-averaged curves above.

Edit the constants in this file to change parameters. There is intentionally
no command-line parser. N_COLLISIONS = 1000 matches the paper caption. Unlike
the detuning scan in make_fig2.py, raising N_COLLISIONS here does NOT
improve the match to the reference figure: at n_steps=10000, I_LFS and N_q
collapse toward ~0.005 while I_RHP grows to ~0.038, badly distorting the
shape. n_steps=1000 reproduces the reported ~0.02-0.05 band and the curves'
relative spacing far better, so it is kept as the default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from definitions import (
    Array,
    CollisionParameters,
    bath_state,
    collision_step,
    exchange_hamiltonian,
    hermitian_expm,
    intermediate_maps,
    kdq_non_positivity,
    kron_all,
    lfs_non_monotonicity,
    local_hamiltonian,
    matrix_basis,
    partial_trace,
    phase_covariant_parameters,
    reduced_system_states,
    rhp_witness,
    thermal_qubit,
    vec,
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

# Baseline parameters from the Fig. 1 caption; the S-M collision anisotropy
# is swept below via gamma, the M-E collision stays isotropic (gamma=0).
N_COLLISIONS = 1000
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

GAMMA = np.linspace(-1, 1.0, 201)
OUTPUT_PATH = Path(__file__).with_name("fig_4.pdf")
SAVE_FIGURE = True
SHOW_FIGURE = False

# Separate diagnostic figure: g_n, N_q[P_n] and Delta I_LS^(n) versus
# collision number, one subplot per gamma value. Kept small (100) to
# iterate quickly; raise once the plot looks right.
DIAGNOSTIC_N_COLLISIONS = 250
DIAGNOSTIC_GAMMA_VALUES = ( 0.5, 1.0)
DIAGNOSTIC_OUTPUT_PATH = Path(__file__).with_name("fig_4_vs_collisions.pdf")

LINE_WIDTH = 3.0

COLORS = {
    "green": "#8FB032",
    "blue": "#5E81B5",
    "orange": "#E19C24",
}


def collision_unitaries_sm_anisotropic(
    params: CollisionParameters, gamma_sm: float
) -> tuple[Array, Array]:
    """Like definitions.collision_unitaries, but anisotropy only in S-M;
    the M-E collision always stays isotropic (energy-preserving)."""

    h0 = local_hamiltonian(params)
    h_sm = exchange_hamiltonian(params.g_sm, 0, 1, gamma_sm)
    h_me = exchange_hamiltonian(params.g_me, 1, 2, 0.0)
    u_sm = hermitian_expm(h0 + h_sm, params.tau_sm)
    u_me = hermitian_expm(h0 + h_me, params.tau_me)
    return u_sm, u_me


def reduced_system_maps_sm_anisotropic(
    params: CollisionParameters, initial_m: Array, initial_e: Array, gamma_sm: float
) -> list[Array]:
    """Same as definitions.reduced_system_maps, with S-M anisotropy gamma_sm."""

    basis = matrix_basis()
    columns = []
    u_sm, u_me = collision_unitaries_sm_anisotropic(params, gamma_sm)

    for basis_operator in basis:
        trajectory = [kron_all(basis_operator, initial_m, initial_e)]
        for _ in range(params.n_steps):
            trajectory.append(collision_step(trajectory[-1], initial_e, u_sm, u_me))
        reduced = [partial_trace(rho, (2, 2, 2), keep=(0,)) for rho in trajectory]
        columns.append([vec(rho) for rho in reduced])

    maps = []
    for step in range(params.n_steps + 1):
        maps.append(np.column_stack([columns[col][step] for col in range(4)]))
    return maps


def evolve_sme_sm_anisotropic(
    initial_s: Array,
    initial_m: Array,
    initial_e: Array,
    params: CollisionParameters,
    gamma_sm: float,
) -> list[Array]:
    """Same as definitions.evolve_sme, with S-M anisotropy gamma_sm."""

    u_sm, u_me = collision_unitaries_sm_anisotropic(params, gamma_sm)
    trajectory = [kron_all(initial_s, initial_m, initial_e)]
    for _ in range(params.n_steps):
        trajectory.append(collision_step(trajectory[-1], initial_e, u_sm, u_me))
    return trajectory


@dataclass(frozen=True)
class GammaScanResults:
    """Witnesses collected over an anisotropy scan."""

    gamma: np.ndarray
    i_rhp: np.ndarray
    i_lfs: np.ndarray
    sum_nq: np.ndarray


def per_collision_witnesses(
    params: CollisionParameters, gamma_sm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one collision-model trajectory; return the full per-collision
    (g_n, N_q[P_n], Delta I_LS^(n)) arrays."""

    rho_s0 = thermal_qubit(params.beta_s, params.omega_s)
    rho_m0 = thermal_qubit(params.beta_m, params.omega_m)
    rho_e0 = bath_state(params.beta_e, params.omega_e, params.bath_coherence)

    system_maps = reduced_system_maps_sm_anisotropic(params, rho_m0, rho_e0, gamma_sm)
    two_time_maps = intermediate_maps(system_maps)

    a, b, _ = phase_covariant_parameters(two_time_maps)
    a = np.real_if_close(a).astype(float)
    b = np.real_if_close(b).astype(float)

    rho_sme = evolve_sme_sm_anisotropic(rho_s0, rho_m0, rho_e0, params, gamma_sm)
    rho_s = reduced_system_states(rho_sme)

    g_n = rhp_witness(two_time_maps)
    delta_i_ls = lfs_non_monotonicity(two_time_maps)
    n_q = kdq_non_positivity(a, b, rho_s)
    return g_n, n_q, delta_i_ls


def cumulative_measures(params: CollisionParameters, gamma_sm: float) -> tuple[float, float, float]:
    """Run one collision-model trajectory; return (I_RHP, I_LFS, N_q), each
    averaged over the n_steps collisions."""

    g_n, n_q, delta_i_ls = per_collision_witnesses(params, gamma_sm)

    # Normalized by the number of collisions: a time-averaged witness per
    # collision, rather than the raw (monotonically growing) cumulative sum.
    i_rhp = float(np.mean(g_n))
    i_lfs = float(np.mean(delta_i_ls))
    sum_nq = float(np.mean(n_q))
    return i_rhp, i_lfs, sum_nq


def simulate_fig4(
    gamma_values: np.ndarray = GAMMA,
    base_params: CollisionParameters = BASE_PARAMS,
) -> GammaScanResults:
    """Scan the S-M coupling anisotropy, accumulating the three witnesses."""

    i_rhp = np.empty_like(gamma_values)
    i_lfs = np.empty_like(gamma_values)
    sum_nq = np.empty_like(gamma_values)

    for idx, gamma_sm in enumerate(gamma_values):
        i_rhp[idx], i_lfs[idx], sum_nq[idx] = cumulative_measures(base_params, gamma_sm)
        if (idx + 1) % 10 == 0 or idx + 1 == len(gamma_values):
            print(f"  scanned {idx + 1}/{len(gamma_values)} anisotropy points")

    return GammaScanResults(gamma=gamma_values, i_rhp=i_rhp, i_lfs=i_lfs, sum_nq=sum_nq)


def plot_fig4(results: GammaScanResults) -> plt.Figure:
    """Plot I_RHP, I_LFS and N_q, averaged per collision, versus gamma."""

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    ax.plot(
        results.gamma, results.i_rhp,
        color=COLORS["orange"], linestyle="-", linewidth=LINE_WIDTH,
        label=r"$\mathcal{I}_{RHP}/n_{\max}$",
    )
    ax.plot(
        results.gamma, results.i_lfs,
        color=COLORS["green"], linestyle="--", linewidth=LINE_WIDTH,
        label=r"$\mathcal{I}^\Phi_{LFS}/n_{\max}$",
    )
    ax.plot(
        results.gamma, results.sum_nq,
        color=COLORS["blue"], linestyle="-.", linewidth=LINE_WIDTH,
        label=r"$\sum_n \mathcal{N}_q[P_n]/n_{\max}$",
    )

    ax.set_xlabel(r"$\gamma$")
    ax.legend(frameon=True,
              framealpha=0.7,
              borderpad=0.2,
              edgecolor=None,
              ncols=1,
              labelspacing=1,
              loc="best",)
    ax.grid(True, which="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlim([results.gamma.min(), results.gamma.max()])
    #ax.set_ylim([0.019,0.039])
    ax.set_yscale("log")

    return fig


def plot_fig4_vs_collisions(
    gamma_values: tuple[float, ...] = DIAGNOSTIC_GAMMA_VALUES,
    base_params: CollisionParameters = BASE_PARAMS,
) -> plt.Figure:
    """Plot g_n, N_q[P_n] and Delta I_LS^(n) versus collision number, one
    subplot per gamma value (separate figure from plot_fig4)."""

    params = dataclasses.replace(base_params, n_steps=DIAGNOSTIC_N_COLLISIONS)
    n = np.arange(1, params.n_steps + 1)

    fig, axes = plt.subplots(
        len(gamma_values), 1, figsize=(9, 3 * len(gamma_values)), sharex=True
    )

    for ax, gamma_sm in zip(axes, gamma_values):
        g_n, n_q, delta_i_ls = per_collision_witnesses(params, gamma_sm)

        ax.plot(n, g_n, color=COLORS["green"], linestyle="-", linewidth=LINE_WIDTH, label=r"$g_n$")
        ax.plot(n, n_q, color=COLORS["orange"], linestyle="--", linewidth=LINE_WIDTH, label=r"$\mathcal{N}_q[P_n]$")
        ax.plot(n, delta_i_ls, color=COLORS["blue"], linestyle="-.", linewidth=LINE_WIDTH, label=r"$\Delta I_{LS}^{(n)}$")

        ax.set_ylabel(rf"$\gamma={gamma_sm:g}$")
        ax.grid(True, which="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    axes[0].legend(frameon=True, framealpha=0.7, borderpad=0.2, edgecolor=None, ncols=3)
    axes[-1].set_xlabel("collision number n")
    axes[-1].set_xlim([n.min(), n.max()])
    fig.tight_layout(h_pad=0.3)
    return fig


def main() -> None:
    print(f"Scanning {len(GAMMA)} anisotropy points with N_COLLISIONS={N_COLLISIONS}...")
    results = simulate_fig4()
    fig = plot_fig4(results)

    if SAVE_FIGURE:
        fig.savefig(OUTPUT_PATH, dpi=200)
        print(f"Saved figure to {OUTPUT_PATH}")

    min_idx = int(np.argmin(results.i_rhp))
    print(f"I_RHP minimum at gamma = {results.gamma[min_idx]:.3f}")

    print(f"Computing vs-collisions diagnostic with DIAGNOSTIC_N_COLLISIONS={DIAGNOSTIC_N_COLLISIONS}...")
    fig_vs_collisions = plot_fig4_vs_collisions()

    if SAVE_FIGURE:
        fig_vs_collisions.savefig(DIAGNOSTIC_OUTPUT_PATH, dpi=200)
        print(f"Saved figure to {DIAGNOSTIC_OUTPUT_PATH}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_vs_collisions)


if __name__ == "__main__":
    main()

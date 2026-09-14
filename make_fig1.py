"""Reproduce Fig. 1: the resonant, energy-preserving collision model.

Three stacked panels, all versus collision number n, for the resonant case
(omega_S = omega_M = omega_E, isotropic S-M exchange):
  1. The three complete-positivity condition tests a_n - 1, b_n, and
     |c_n|^2 - a_n(1 - b_n) (all <= 0 iff the two-time map is CP).
  2. The average system energy change <Delta E_S> / omega_S.
  3. The RHP witness g_n, the KDQ non-positivity functional N_q[P_n], and
     the LFS QMI non-monotonicity Delta I_LS^(n).

Curves are distinguished by line style (solid / dashed / dash-dot / dotted)
rather than markers. Edit the constants below to change parameters; there is
intentionally no command-line parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

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
    system_energy_changes,
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
# Parameters reported in the Fig. 1 caption.
PARAMS = CollisionParameters(
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
    n_steps=1001,
)

PLOT_COLLISIONS = 175
OUTPUT_PATH = Path(__file__).with_name("fig_1.pdf")
SAVE_FIGURE = True
SHOW_FIGURE = False

LINE_WIDTH = 3.0

COLORS = {
    "green": "#8FB032",
    "blue": "#5E81B5",
    "orange": "#E19C24",
    "red": "#EB6235",
    "cyan": "#00BFC4",
    "black": "#000000",
    "purple": "#A64CB8",
}

# Line styles substituted for the original marker shapes (triangle, square,
# diamond, circle) so each quantity keeps a distinct, consistent visual code.
LINESTYLES = {
    "triangle": "--",
    "square": ":",
    "diamond": "-.",
    "circle": "-",
}


@dataclass(frozen=True)
class Fig1Results:
    """Container for the quantities plotted in Fig. 1."""

    collisions: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    condition_a: np.ndarray
    condition_b: np.ndarray
    condition_c: np.ndarray
    energy_change: np.ndarray
    non_markovianity: np.ndarray
    non_positivity: np.ndarray
    qmi_non_monotonicity: np.ndarray


def simulate_fig1(params: CollisionParameters = PARAMS) -> Fig1Results:
    """Run the collision model and compute the Fig. 1 diagnostics."""

    rho_s0 = thermal_qubit(params.beta_s, params.omega_s)
    rho_m0 = thermal_qubit(params.beta_m, params.omega_m)
    rho_e0 = bath_state(params.beta_e, params.omega_e, params.bath_coherence)

    system_maps = reduced_system_maps(params, rho_m0, rho_e0)
    two_time_maps = intermediate_maps(system_maps)

    a, b, c = phase_covariant_parameters(two_time_maps)
    a = np.real_if_close(a).astype(float)
    b = np.real_if_close(b).astype(float)

    # The physical trajectory is evolved separately from the tomographic
    # reconstruction above: N_q and the energy change need the actual
    # pre-collision populations of rho_S, while a, b, c and the
    # CP-divisibility witnesses need the linear map Lambda_{n,n-1} itself.
    rho_sme = evolve_sme(rho_s0, rho_m0, rho_e0, params)
    rho_s = reduced_system_states(rho_sme)

    return Fig1Results(
        collisions=np.arange(1, params.n_steps + 1),
        a=a,
        b=b,
        c=c,
        condition_a=a - 1.0,
        condition_b=b,
        condition_c=np.abs(c) ** 2 - a * (1.0 - b),
        energy_change=system_energy_changes(rho_s, params.omega_s),
        non_markovianity=rhp_witness(two_time_maps),
        non_positivity=kdq_non_positivity(a, b, rho_s),
        qmi_non_monotonicity=lfs_non_monotonicity(two_time_maps),
    )


def first_positive_collision(values: np.ndarray, tol: float = 1e-10) -> int | None:
    """Return the first one-based collision index where an array is positive."""

    hits = np.flatnonzero(values > tol)
    if hits.size == 0:
        return None
    return int(hits[0] + 1)


def negative_regions(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    """Return (start, end) x-bounds of contiguous stretches where y < 0."""

    mask = y < 0
    regions = []
    start = None
    for xi, negative in zip(x, mask):
        if negative and start is None:
            start = xi
        elif not negative and start is not None:
            regions.append((start, prev))
            start = None
        prev = xi
    if start is not None:
        regions.append((start, prev))
    return regions


def plot_fig1(results: Fig1Results, params: CollisionParameters = PARAMS) -> plt.Figure:
    """Lay out the three stacked panels of Fig. 1 (see module docstring)."""

    n_plot = min(PLOT_COLLISIONS, len(results.collisions))
    n = results.collisions[:n_plot]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    flux_negative = negative_regions(n, results.energy_change[:n_plot])
    flux_boundaries = sorted(
        {bound for start, end in flux_negative for bound in (start - 0.5, end + 0.5)}
    )
    for ax in axes:
        for start, end in flux_negative:
            ax.axvspan(start - 0.5, end + 0.5, color="yellow", alpha=0.05, zorder=-1)
        for boundary in flux_boundaries:
            ax.axvline(boundary, color="0.25", linestyle="--", linewidth=1.0, zorder=1)

    def styled_line(ax, x, y, color, shape, label):
        (line,) = ax.plot(
            x,
            y,
            color=color,
            linewidth=LINE_WIDTH,
            linestyle=LINESTYLES[shape],
            zorder=3,
            label=label,
        )
        return line

    ax = axes[0]
    handles = [
        styled_line(
            ax,
            n,
            results.condition_c[:n_plot],
            COLORS["orange"],
            "triangle",
            r"$|c_n|^2 {-} a_n(1{-}b_n)$",
        ),
        styled_line(ax, n, results.condition_a[:n_plot], COLORS["green"], "square", r"$a_n {-} 1$"),
        styled_line(ax, n, results.condition_b[:n_plot], COLORS["blue"], "diamond", r"$b_n$"),
    ]
    ax.axhline(0.0, color="0.25", lw=0.8)
    #ax.set_ylabel("CP-condition tests")
    ax.legend(handles, 
        [h.get_label() for h in handles], 
        loc=[0.1, .81], 
        frameon=True, 
        framealpha=0.7,
        borderpad=0.1, 
        edgecolor=None, 
        ncols=3,
        columnspacing=0.7,
    )
    ax.grid(True, which="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_ylim([-0.13, 0.24])
    ax.set_xlim([0, PLOT_COLLISIONS])

    ax = axes[1]
    handles = [styled_line(ax, n, results.energy_change[:n_plot], COLORS["red"], "circle", r"$\langle\Delta E_S\rangle / \omega_S$")]
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.legend(handles, 
        [h.get_label() for h in handles], 
        loc=[0.4, .83], 
        frameon=True, 
        framealpha=0.7,
        borderpad=0.1, 
        edgecolor=None, 
        ncols=3,
        columnspacing=0.7,
    )
    #ax.set_ylabel(r"$\langle\Delta E_S\rangle / \omega_S$")
    ax.grid(True, which="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylim([-0.01, 0.01])

    ax = axes[2]
    line_g = styled_line(ax, n, results.non_markovianity[:n_plot], COLORS["purple"], "circle", r"$g_n$")
    line_nq = styled_line(
        ax,
        n,
        results.non_positivity[:n_plot],
        COLORS["cyan"],
        "triangle",
        r"$\mathcal{N}_q[P_n]$",
    )
    line_dqmi = styled_line(
        ax,
        n,
        results.qmi_non_monotonicity[:n_plot],
        COLORS["black"],
        "diamond",
        r"$\Delta I_{LS}^{(n)}$",
    )
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xlabel("collision number n")
    #ax.set_ylabel("witness value")
    ax.legend(
        [line_g, line_nq, line_dqmi],
        [r"$g_n$", r"$\mathcal{N}_q[P_n]$", r"$\Delta I_{LS|\Phi}^{(n)}$"],
        loc=[0.33, .74], 
        frameon=True, 
        framealpha=0.7,
        borderpad=0.1, 
        edgecolor=None, 
        ncols=3,
        columnspacing=0.7,
    )
    ax.grid(True, which="both", linestyle=":", color="0.6", linewidth=0.8, zorder=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylim([-0.01, 0.26])


    fig.tight_layout(h_pad=-0.5)
    return fig


def main() -> None:
    results = simulate_fig1(PARAMS)
    fig = plot_fig1(results, PARAMS)

    if SAVE_FIGURE:
        fig.savefig(OUTPUT_PATH, dpi=200)
        print(f"Saved figure to {OUTPUT_PATH}")

    print(f"First g_n > 0 collision: {first_positive_collision(results.non_markovianity)}")
    print(f"First N_q > 0 collision: {first_positive_collision(results.non_positivity)}")
    print(f"max g_n = {np.max(results.non_markovianity):.6g}")
    print(f"max N_q = {np.max(results.non_positivity):.6g}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

"""Core simulation of the memory-mediated collision model.

A system qubit S interacts sequentially with a memory qubit M, which in turn
interacts with a fresh environment qubit E at every collision step (E is
replaced by a new copy of the same initial state after each step, so it
carries no memory of its own). Repeating this S-M then M-E exchange builds up
non-Markovian correlations between S and M even though every individual step
is unitary.

This module provides the shared building blocks used by all of the
``make_fig*.py`` scripts: constructing the collision unitaries, evolving the
joint S-M-E state, reconstructing the reduced dynamical map of S via
tomography, and computing the non-Markovianity/non-positivity witnesses
analyzed in the paper (RHP, LFS, and the KDQ non-positivity functional).

The functions in this module are intentionally dependency-light: NumPy is
enough because all Hamiltonians are tiny (2x2, 4x4, or 8x8) Hermitian
matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass(frozen=True)
class CollisionParameters:
    """Physical parameters for the three-qubit collision model."""

    omega_s: float = 1.0
    omega_m: float = 1.0
    omega_e: float = 1.0
    g_sm: float = 0.2
    g_me: float = 0.2
    tau_sm: float = 0.2
    tau_me: float = 0.2
    beta_s: float = 0.0
    beta_m: float = 1.0
    beta_e: float = 1.0
    bath_coherence: float = 0.0
    gamma: float = 0.0
    n_steps: int = 1001


def kron_all(*operators: Array) -> Array:
    """Kronecker product of all supplied operators."""

    out = np.array([[1.0]], dtype=complex)
    for operator in operators:
        out = np.kron(out, operator)
    return out


def one_site(operator: Array, site: int, n_sites: int = 3) -> Array:
    """Embed a one-qubit operator on ``site`` in an ``n_sites`` register."""

    factors = [I2] * n_sites
    factors[site] = operator
    return kron_all(*factors)


def two_site(op_left: Array, site_left: int, op_right: Array, site_right: int) -> Array:
    """Embed a product of two one-qubit operators in the S-M-E register."""

    factors = [I2, I2, I2]
    factors[site_left] = op_left
    factors[site_right] = op_right
    return kron_all(*factors)


def thermal_qubit(beta: float, omega: float) -> Array:
    """Thermal qubit state for H = omega * sigma_z / 2."""

    excited_weight = 1.0 / (1.0 + np.exp(beta * omega))
    ground_weight = np.exp(beta * omega) / (1.0 + np.exp(beta * omega))
    return np.diag([excited_weight, ground_weight]).astype(complex)


def bath_state(beta: float, omega: float, coherence: float = 0.0) -> Array:
    """Thermal bath qubit, optionally with real off-diagonal coherence."""

    rho = thermal_qubit(beta, omega)
    rho = rho.copy()
    rho[0, 1] = coherence
    rho[1, 0] = coherence
    return rho


def hermitian_expm(hamiltonian: Array, time: float) -> Array:
    """Compute exp(-i H t) for a Hermitian matrix H."""

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigenvalues * time)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def partial_trace(rho: Array, dims: tuple[int, ...], keep: tuple[int, ...]) -> Array:
    """Trace out all subsystems except those listed in ``keep``.

    Subsystem indices are zero-based and follow the tensor-product order used
    in ``kron_all``. For this project that order is S, M, E.
    """

    keep = tuple(keep)
    traced = [i for i in range(len(dims)) if i not in keep]
    tensor = rho.reshape(*dims, *dims)
    current_dims = list(dims)

    for site in sorted(traced, reverse=True):
        tensor = np.trace(tensor, axis1=site, axis2=site + len(current_dims))
        del current_dims[site]

    kept_dim = int(np.prod([dims[i] for i in keep], dtype=int))
    return tensor.reshape(kept_dim, kept_dim)


def local_hamiltonian(params: CollisionParameters) -> Array:
    """Bare Hamiltonian H_S + H_M + H_E."""

    return (
        0.5 * params.omega_s * one_site(SZ, 0)
        + 0.5 * params.omega_m * one_site(SZ, 1)
        + 0.5 * params.omega_e * one_site(SZ, 2)
    )


def exchange_hamiltonian(
    coupling: float,
    site_left: int,
    site_right: int,
    gamma: float = 0.0,
) -> Array:
    """Anisotropic Heisenberg exchange Hamiltonian.

    Fig. 1 uses gamma = 0, giving the energy-preserving isotropic interaction.
    """

    return 0.5 * coupling * (
        (1.0 - gamma) * two_site(SX, site_left, SX, site_right)
        + (1.0 + gamma) * two_site(SY, site_left, SY, site_right)
        + two_site(SZ, site_left, SZ, site_right)
    )


def collision_unitaries(params: CollisionParameters) -> tuple[Array, Array]:
    """Build the S-M and M-E collision unitaries."""

    h0 = local_hamiltonian(params)
    h_sm = exchange_hamiltonian(params.g_sm, 0, 1, params.gamma)
    h_me = exchange_hamiltonian(params.g_me, 1, 2, params.gamma)
    u_sm = hermitian_expm(h0 + h_sm, params.tau_sm)
    u_me = hermitian_expm(h0 + h_me, params.tau_me)
    return u_sm, u_me


def collision_step(rho_sme: Array, rho_e0: Array, u_sm: Array, u_me: Array) -> Array:
    """Apply one memory-mediated collision step to an S-M-E state."""

    rho_sm = partial_trace(rho_sme, (2, 2, 2), keep=(0, 1))
    refreshed = kron_all(rho_sm, rho_e0)
    after_sm = u_sm @ refreshed @ u_sm.conj().T
    return u_me @ after_sm @ u_me.conj().T


def evolve_sme(initial_s: Array, initial_m: Array, initial_e: Array, params: CollisionParameters) -> list[Array]:
    """Evolve the S-M-E collision model, replacing E by a fresh state each step."""

    u_sm, u_me = collision_unitaries(params)
    trajectory = [kron_all(initial_s, initial_m, initial_e)]
    for _ in range(params.n_steps):
        trajectory.append(collision_step(trajectory[-1], initial_e, u_sm, u_me))
    return trajectory


def reduced_system_states(sme_trajectory: list[Array]) -> list[Array]:
    """Return rho_S at every stored time."""

    return [partial_trace(rho, (2, 2, 2), keep=(0,)) for rho in sme_trajectory]


def matrix_basis() -> list[Array]:
    """Operator basis matching NumPy row-major vectorization."""

    return [
        np.array([[1, 0], [0, 0]], dtype=complex),
        np.array([[0, 1], [0, 0]], dtype=complex),
        np.array([[0, 0], [1, 0]], dtype=complex),
        np.array([[0, 0], [0, 1]], dtype=complex),
    ]


def vec(operator: Array) -> Array:
    """Row-major vectorization, matching the paper's (rho00,rho01,rho10,rho11)."""

    return operator.reshape(-1)


def mat(vector: Array) -> Array:
    """Inverse of ``vec`` for a one-qubit operator."""

    return vector.reshape(2, 2)


def reduced_system_maps(
    params: CollisionParameters,
    initial_m: Array,
    initial_e: Array,
) -> list[Array]:
    """Tomographically reconstruct Lambda_n for n = 0, ..., n_steps.

    Each Lambda_n maps the initial system operator to the reduced system
    operator after n collision steps while M and each fresh E start in their
    fixed initial states.
    """

    basis = matrix_basis()
    columns = []
    u_sm, u_me = collision_unitaries(params)

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


def intermediate_maps(system_maps: list[Array]) -> list[Array]:
    """Return Lambda_{n,n-1} = Lambda_n Lambda_{n-1}^{-1} for n >= 1."""

    return [
        system_maps[n] @ np.linalg.inv(system_maps[n - 1])
        for n in range(1, len(system_maps))
    ]


def phase_covariant_parameters(intermediate: list[Array]) -> tuple[Array, Array, Array]:
    """Extract the a, b, c entries from phase-covariant intermediate maps.

    With operators vectorized as (rho00, rho01, rho10, rho11), a phase
    covariant map (isotropic S-M exchange) acts block-diagonally: the
    populations rho00, rho11 mix only with each other (entries a, b, at
    positions (0,0) and (0,3)), while the coherence rho01 evolves on its own
    (entry c, at position (1,1)) and never couples back into the
    populations. This still holds if the S-M coupling is anisotropic, in
    which case c no longer captures the full coherence sector on its own
    (see the anisotropic S-M model in make_fig4.py), but a and b are
    unaffected either way.
    """

    maps = np.asarray(intermediate)
    a = maps[:, 0, 0]
    b = maps[:, 0, 3]
    c = maps[:, 1, 1]
    return np.real_if_close(a), np.real_if_close(b), c


def choi_matrix(superoperator: Array) -> Array:
    """Choi matrix of a one-qubit map represented with row-major vectorization."""

    basis = matrix_basis()
    choi = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            e_ij = np.zeros((2, 2), dtype=complex)
            e_ij[i, j] = 1.0
            choi += np.kron(e_ij, mat(superoperator @ vec(e_ij)))
    return 0.5 * choi


def rhp_witness(intermediate: list[Array], atol: float = 1e-12) -> Array:
    """Discrete RHP CP-divisibility witness g_n.

    g_n = ||(Lambda_{n,n-1} tensor I)[|Psi><Psi|]||_1 - 1, the trace-norm
    deviation of the map's Choi state from a valid density matrix. It is 0
    when Lambda_{n,n-1} is completely positive and strictly positive
    otherwise, so a run of nonzero g_n flags non-CP-divisibility (RHP
    non-Markovianity) over that stretch of collisions.
    """

    values = []
    for superoperator in intermediate:
        eigenvalues = np.linalg.eigvalsh(choi_matrix(superoperator))
        trace_norm = np.sum(np.abs(eigenvalues))
        values.append(max(float(trace_norm - 1.0), 0.0))
    values = np.asarray(values)
    values[values < atol] = 0.0
    return values


def kdq_non_positivity(a: Array, b: Array, system_states: list[Array], atol: float = 1e-12) -> Array:
    """Non-positivity functional N_q[P_n] of the energy-change KDQ distribution.

    N_q[P_n] = -1 + p0 (|a_n| + |1-a_n|) + p1 (|b_n| + |1-b_n|), where p0, p1
    are the populations of the pre-collision system state and a_n, b_n are
    the population-sector entries of the two-time map Lambda_{n,n-1}. It is
    strictly positive exactly when a_n or b_n falls outside [0, 1], i.e. when
    the map is not completely positive on the population sector.
    """

    p0 = np.asarray([np.real(rho[0, 0]) for rho in system_states[:-1]])
    p1 = np.asarray([np.real(rho[1, 1]) for rho in system_states[:-1]])
    values = -1.0 + p0 * (np.abs(a) + np.abs(1.0 - a)) + p1 * (np.abs(b) + np.abs(1.0 - b))
    values = np.real_if_close(values).astype(float)
    values[values < atol] = 0.0
    return values


def system_energy_changes(system_states: list[Array], omega_s: float) -> Array:
    """Average system-energy increments normalized by omega_s."""

    energies = np.asarray([np.real(np.trace(0.5 * omega_s * SZ @ rho)) for rho in system_states])
    return np.diff(energies) / omega_s


def bell_state_phi_plus() -> Array:
    """Maximally entangled state |Phi+><Phi+| on the L-S register (order L, S)."""

    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2.0)  # |L=0, S=0>
    psi[3] = 1.0 / np.sqrt(2.0)  # |L=1, S=1>
    return np.outer(psi, psi.conj())


def apply_map_to_second_qubit(superoperator: Array, rho_ls: Array) -> Array:
    """Apply (I_L tensor Lambda)[rho_LS] for Lambda acting on the S register."""

    blocks = rho_ls.reshape(2, 2, 2, 2)
    out = np.empty_like(blocks)
    for l in range(2):
        for l_prime in range(2):
            out[l, :, l_prime, :] = mat(superoperator @ vec(blocks[l, :, l_prime, :]))
    return out.reshape(4, 4)


def von_neumann_entropy(rho: Array, atol: float = 1e-12) -> float:
    """Von Neumann entropy S(rho) = -Tr[rho log2 rho], in bits."""

    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > atol]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def quantum_mutual_information(rho_ls: Array) -> float:
    """QMI I(rho_LS) = S(rho_L) + S(rho_S) - S(rho_LS)."""

    rho_l = partial_trace(rho_ls, (2, 2), keep=(0,))
    rho_s = partial_trace(rho_ls, (2, 2), keep=(1,))
    return von_neumann_entropy(rho_l) + von_neumann_entropy(rho_s) - von_neumann_entropy(rho_ls)


def lfs_non_monotonicity(intermediate: list[Array], atol: float = 1e-12) -> Array:
    """Delta I_LS^(n) QMI non-monotonicity witness, per collision n.

    For each two-time map Lambda_{n,n-1}, build its own Choi state from a
    fresh Bell pair |Phi+>, apply Lambda_{n,n-1} once more, and take the QMI
    difference. Lambda_{n,n-1} need not be completely positive, so this Choi
    state is not always a valid density matrix; von_neumann_entropy drops
    negative eigenvalues rather than renormalizing, matching how g_n and
    N_q[P_n] treat the same non-CP two-time maps.
    """

    rho_ls0 = bell_state_phi_plus()
    values = []
    for lam in intermediate:
        choi_state = apply_map_to_second_qubit(lam, rho_ls0)
        advanced = apply_map_to_second_qubit(lam, choi_state)
        values.append(quantum_mutual_information(advanced) - quantum_mutual_information(choi_state))
    values = np.asarray(values)
    values[values < atol] = 0.0
    return values

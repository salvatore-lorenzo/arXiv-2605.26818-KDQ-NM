# Memory-mediated collision model: KDQ non-positivity and non-Markovianity

Simulation code accompanying the paper:

> TODO: Title, authors, and arXiv/journal reference.

This repository reproduces every figure in the paper from a single shared
physics module. There is no installer or CLI: each `make_fig*.py` script is
a self-contained, parameter-at-the-top Python file that you run directly.

## The model

A system qubit **S** interacts sequentially with a memory qubit **M**, which
in turn interacts with a fresh environment qubit **E** at every collision
step (E is replaced by a new copy of the same fixed initial state after each
step, so the environment itself carries no memory). Concretely, at
collision `n`:

1. **S** and **M** interact via a Heisenberg exchange Hamiltonian for time
   `tau_sm`.
2. **M** and the fresh **E** interact via a Heisenberg exchange Hamiltonian
   for time `tau_me`.
3. **E** is discarded.

Repeating this builds up non-Markovian correlations in the reduced dynamics
of S, even though every individual step is unitary: M "remembers" the
outcome of its collision with S until it hands that memory off (partially)
to the next fresh E, then comes back to collide with S again.

When the exchange interactions are isotropic (the default), the reduced
dynamics of S is a **phase covariant map**: populations and coherences
evolve in separate sectors. The population sector of each two-time map
`Lambda_{n,n-1}` is captured by two numbers `a_n`, `b_n`; the map is
completely positive (CP) iff `a_n, b_n` in `[0, 1]` and a third condition on
the coherence entry `c_n` holds. The scripts here compute, collision by
collision:

- **`g_n`** — the RHP (Rivas-Huelga-Plenio) witness of CP-divisibility, from
  the trace norm of the map's Choi matrix.
- **`N_q[P_n]`** — the non-positivity of the Kirkwood-Dirac quasiprobability
  (KDQ) distribution of the system's energy changes.
- **`Delta I_LS^(n)`** — the LFS (Luo-Fu-Song) witness, the non-monotonicity
  of the quantum mutual information between S and a maximally correlated
  reference qubit L.

All three vanish whenever the dynamics is Markovian (CP-divisible) and
become nonzero together whenever it isn't — that correspondence is the
paper's main result.

## Repository layout

| File | Purpose |
| --- | --- |
| `definitions.py` | Core physics: Hamiltonians, collision step, tomographic reconstruction of the reduced maps, and the `g_n` / `N_q` / `Delta I_LS` witnesses. Imported by every `make_fig*.py` script. |
| `make_fig1.py` | Fig. 1 — resonant, energy-preserving case: CP-condition tests, average energy change, and the three witnesses, vs. collision number. |
| `make_fig2.py` | Fig. 2 — the three witnesses (time-averaged per collision) vs. the system-memory detuning `Delta_omega = omega_S - omega_M`. |
| `make_fig3.py` | Fig. 3 — a single off-resonant case (`Delta_omega = -0.2`): CP-condition tests and the three witnesses, vs. collision number. |
| `make_fig4.py` | Fig. 4 — the three witnesses (time-averaged) vs. the S-M coupling anisotropy `gamma`, plus a diagnostic figure showing the underlying per-collision witnesses for a few values of `gamma`. |

Each script saves its output as PDF next to itself (e.g. `fig_1.pdf`) and
prints a short numeric summary to the terminal.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`
- A working LaTeX installation (e.g. TeX Live or MacTeX). All figures render
  labels with matplotlib's `text.usetex = True`, so a LaTeX toolchain must
  be on your `PATH`. If you don't have one, set `"text.usetex": False` in
  the `plt.rcParams.update({...})` block near the top of the script you're
  running (labels will fall back to matplotlib's built-in mathtext).

Install the Python dependencies with:

```bash
pip install numpy matplotlib
```

## Usage

Run any script directly from the repository root:

```bash
python make_fig1.py
python make_fig2.py
python make_fig3.py
python make_fig4.py
```

There is intentionally no command-line interface. To change a parameter
(couplings, temperatures, number of collisions, resolution of a scan, ...),
edit the module-level constants near the top of the corresponding file and
re-run it. Every physical parameter lives in a `CollisionParameters`
instance (see `definitions.py`), so the full parameter set for a given
figure is visible in one place.

Note on runtime: `make_fig2.py` and `make_fig4.py` sweep many parameter
values (up to 201), so they take longer to run than `make_fig1.py`/
`make_fig3.py`, which each simulate a single trajectory. `make_fig2.py` is
the slowest (several minutes): its witnesses don't decay with the number of
collisions, so it also raises `N_COLLISIONS` to 10000 (vs. the nominal 1000
used elsewhere) to keep the detuning scan well converged — see the comment
at the top of that file. Lower `N_COLLISIONS` or the number of scan points
at the top of either file for faster iteration.

## Citing

If you use this code, please cite:

> TODO: add citation (BibTeX or plain text).

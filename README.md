# Nonclassical energy-change distribution as a witness of non-Markovian quantum dynamics

**Description:** Reproduces simulations and figures in  https://arxiv.org/abs/2605.26818 

Each `make_fig*.py` script is
a self-contained, parameter-at-the-top Python file that you run directly.

## The model

A system qubit **S** interacts sequentially with a memory qubit **M**, which
in turn interacts with a fresh environment qubit **E** at every collision
step (E is replaced by a new copy of the same fixed initial state after each
step, so the environment itself carries no memory). 


## Repository layout

| File | Purpose |
| --- | --- |
| `definitions.py` | Core physics: Hamiltonians, collision step, tomographic reconstruction of the reduced maps, and the `g_n` / `N_q` / `Delta I_LS` witnesses. |
| `make_fig1.py` | Fig. 1 — resonant, energy-preserving case: CP-condition tests, average energy change, and the three witnesses, vs. collision number. |
| `make_fig2.py` | Fig. 2 — the three witnesses (time-averaged per collision) vs. the system-memory detuning `Delta_omega = omega_S - omega_M`. |
| `make_fig3.py` | Fig. 3 — a single off-resonant case (`Delta_omega = -0.2`): CP-condition tests and the three witnesses, vs. collision number. |
| `make_fig4.py` | Fig. 4 — the three witnesses (time-averaged) vs. the S-M coupling anisotropy `gamma`, plus a diagnostic figure showing the underlying per-collision witnesses for a few values of `gamma`. |

Each script saves its output as PDF next to itself (e.g. `fig_1.pdf`) and
prints a short numeric summary to the terminal.


## Usage

Run any script directly from the repository root:

```bash
python make_fig1.py
python make_fig2.py
python make_fig3.py
python make_fig4.py
```

To change a parameter
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

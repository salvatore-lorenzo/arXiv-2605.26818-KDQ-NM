## Nonclassical energy-change distribution as a witness of non-Markovian quantum dynamics

**Description:** Reproduces simulations and figures in  https://arxiv.org/abs/2605.26818 

== Nonclassical energy-change distribution as a witness of non‑Markovian quantum dynamics ==

The notebook implements reusable linear-algebra and quantum-information utilities, qubit-system constructions, and figure-specific simulations used in the paper.


**Contents**
- **Definitions:** Linear-algebra and quantum-information helper functions (computational basis, Pauli matrices, tensor utilities, sparse many-qubit operators).
- **Qubit systems:** Construction of many-qubit spin operators and helpers to build Hamiltonians and channels.
- **Simulations:** Time-evolution, energy-change distribution computations, and witnesses for non-Markovian dynamics.
- **Figures:** Cells that generate the figures used in the manuscript; these rely on the initialization cells in the "Definitions (run at the beginning)" chapter.

**How to use**
- **Open:** Launch Mathematica and open [arxiv.2605.26818.nb](arxiv.2605.26818.nb).
- **Initialize:** Evaluate the cell group titled "Definitions (run at the beginning)" first — it defines symbols and helper functions used throughout the notebook.
- **Reproduce figures:** After initialization, evaluate the chapter or cell groups corresponding to the figure you want to reproduce (the notebook is organized into chapter/subsection groups for each simulation and figure).


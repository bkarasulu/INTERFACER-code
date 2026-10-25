# Tutorials

The tutorials are in the [`Tutorials/`](https://github.com/bkarasulu/INTERFACER-code/tree/main/Tutorials) folder of the repository. Each has an input file, a `run.sh` and a README with the expected results. Tutorials 1, 2, 3 and 6 also have an `analysis.ipynb` notebook that runs the tutorial and plots the results; install Jupyter with `pip install -e ".[notebooks]"`. The notebooks of Tutorials 1 and 6 run in the CI.

| # | System | Engine | Time | Shows |
|---|---|---|---|---|
| 1 | Cu(111) \| Ni(111) | EMT | seconds | The whole workflow with all tests; input files |
| 2 | Na3OCl(100) \| Na(100) | MACE-MP-0 | ~1 min (CPU) | ML potentials for an electrolyte \| anode interface |
| 3 | Na3OCl(100) terminations \| Na | MACE-MP-0 | ~2 min (CPU) | Symmetric slabs, surface formation energies, chemical potentials |
| 4 | β-Li3PS4 \| Li | VASP (+ MACE pre-screening) | HPC | DFT setup, dry runs, restarts |
| 5 | Cu \| Ni | EMT / MACE | seconds | The Python API (script and notebook) |
| 6 | Cu grain boundaries; Cu \| Ni \| Cu | EMT | ~1 min | Symmetric tilt and twist grain boundaries, three-layer interfaces |

## Tutorial 1: Cu | Ni with EMT

```bash
cd Tutorials/01_Cu-Ni_EMT && ./run.sh
```

| Quantity | Result |
|---|---|
| Surface energies of Cu (100) / (110) / (111) | 1.13 / 1.23 / 1.04 J/m² |
| Selected surfaces | (111) for both metals |
| Lateral shift (scanned first) | (0.25, 0.50) |
| Separation at that registry | 2.00 Å |
| Work of adhesion | 2.63 J/m² (≤ γ(Cu) + γ(Ni) = 2.74 J/m²) |

## Tutorial 2: Na3OCl | Na with MACE-MP-0

```bash
pip install mace-torch
cd Tutorials/02_Na3OCl-Na_MACE && ./run.sh
```

| Quantity | Result |
|---|---|
| Surface energies of Na3OCl(100) / Na(100) | 0.33 / 0.25 J/m² |
| Misfit after alignment | 3.7 % |
| Separation | 3.0 Å |
| Work of adhesion | 0.42 J/m² |
| Interface energy γ_int | 0.16 J/m² |

## Tutorial 3: symmetric slabs and surface formation energies

In the Na-rich limit, with chemical potentials from Na metal and NaCl:

| Termination | Surface formation energy | W_ad with Na(100) | γ_int |
|---|---|---|---|
| NaCl-terminated Na3OCl(100) (Cl4Na10O3) | 0.23 J/m² | 0.17 J/m² | 0.31 J/m² |
| Na2O-terminated Na3OCl(100) (Cl3Na11O4) | 0.44 J/m² | 0.65 J/m² | 0.03 J/m² |

The tutorial also explains why Na2O + NaCl cannot be used as references: Na3OCl = Na2O + NaCl exactly.

## Tutorial 4: β-Li3PS4 | Li with VASP

`input_vasp.toml` reproduces the original run with VASP: ENCUT 400 eV, surface, layer, separation and lateral tests. `input_mace_prescreen.toml` screens with MACE first. Check all inputs with `./run_vasp.sh -dry` before submitting.

## Tutorial 5: Python API

`python_api.py` and `python_api.ipynb` show engines, slab building, alignment, scans and the workflow API; see also [Python API](python_api.md).

## Tutorial 6: grain boundaries and three-layer interfaces

```bash
cd Tutorials/06_grain_boundaries_trilayer && ./run.sh     # or: jupyter execute analysis.ipynb
```

| Boundary (Cu, EMT) | Atoms | γ_GB |
|---|---|---|
| (100), (110) mirror planes | 20 | 0 (perfect crystal) |
| Σ5 (310)[001] symmetric tilt, 36.9° | 64 | 0.96 J/m² |
| Σ5 (100) twist, 36.87° | 80 | 0.79 J/m² |
| Σ7 (111) twist, 21.79° | 112 | 0.34 J/m² |

The notebook also plots γ_GB against the [001] tilt angle. For the three-layer interface Cu(100) | Ni(100) | Cu(100), W_ad,total = 2.39 J/m². This is about twice the single Cu | Ni value of 1.25 J/m², as expected for two interfaces. See [Workflow](workflow.md) for how the boundaries are built.

All numbers were obtained on a CPU with the default settings of each input file; ML results depend on the model version.

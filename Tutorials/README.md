# Tutorials

| # | Folder | System | Engine | Time | Shows |
|---|---|---|---|---|---|
| 1 | `01_Cu-Ni_EMT` | Cu(111) \| Ni(111) | EMT (ASE) | seconds | Whole workflow with surface screening, separation and lateral scans; input files |
| 2 | `02_Na3OCl-Na_MACE` | Na3OCl(100) \| Na(100) | MACE-MP-0 | ~1 min (CPU) | ML potentials; a real electrolyte \| anode interface |
| 3 | `03_symmetric_slabs_SFE` | Na3OCl(100) terminations \| Na | MACE-MP-0 | ~2 min (CPU) | Symmetric non-stoichiometric slabs, surface formation energies, chemical potentials |
| 4 | `beta_Li3PS4-Li` | β-Li3PS4 \| Li | VASP (MACE pre-screening) | hours (HPC) | DFT runs, dry runs, restarts |
| 5 | `05_python_api` | Cu \| Ni | EMT (or MACE) | seconds | Python API: building blocks and the workflow |
| – | `Antiperovskites` | Na3OCl \| Na | VASP | – | Reference VASP run of the original version |

Each folder has a README with the expected results. Run a tutorial with `./run.sh` (after `pip install -e .` in the repository root) or `interfacer --input input.toml`.

Tutorials 2 and 3 need `pip install mace-torch`.

# Changelog

## 1.0.0.dev0 (refactor)

**Package**
- Installable `interfacer` package with an `interfacer` command. The script `interfacer.py` (renamed from `interfacer-new.py`; same options) and `tools/` remain as compatibility wrappers.
- The workflow is split into steps (`interfacer.workflow`), and results are written to `OUTPUT/results.json`.
- Common engine interface for VASP, CASTEP and ASE calculators, with a shared ASE optimiser: cell relaxation, fixed bottom layers, BFGS/FIRE/LBFGS.

**New features**
- Machine-learning potentials (`-prog mlip`): MACE, MACE-OFF, CHGNet, MatGL/M3GNet, SevenNet, ORB, fairchem/UMA, NequIP, Allegro, or a custom ASE calculator.
- Input files (`--input` TOML/YAML/JSON) and `--save_input`.
- New options: `--optimiser`, `--relax_cell_bulk`/`--no_relax_cell_bulk`, `--fix_layers`, `--outdir`, `--mlip`, `--mlip_model`, `--device`, `--mlip_kwargs`, `--version`.

**Bug fixes** (see [Fixed bugs and known issues](known_issues.md))
- Unit conversion to J/m².
- Interface separation.
- Work of adhesion (sign, and the number of interfaces in the cell).
- Surface and layer tests now feed into the run.
- 2D lateral scan.
- Symmetric slabs.
- VASP k-points and INCAR booleans.
- Many crashes.

**Other**
- Merged the `symINTERFACER` branch.
- Removed POTCARs and large VASP outputs from the repository.
- Added documentation, tutorials, the test suite and CI.

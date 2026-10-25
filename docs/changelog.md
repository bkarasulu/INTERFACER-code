# Changelog

## Unreleased (`devel` branch)

**Licence and project files**
- Licensed under the GNU General Public License v3.0 or later (`LICENSE`), with `CITATION.cff`, `CONTRIBUTING.md`, issue templates and a Read the Docs configuration.
- The driver script is now `interfacer.py` (renamed from `interfacer-new.py`; same options).

**New features**
- Grain boundaries (`--gb tilt|twist`, `--gb_mn`, `--gb_expansion`): symmetric tilt and CSL twist bicrystals of one material, with γ_GB (`interfacer.grainboundary`).
- Three-layer interfaces slab1 | slab2 | slab3 (`-i3`, `-m3`, `-cr3`, `-sep2`) with W_ad,total.
- GPAW and ONETEP engines (`-prog gpaw|onetep`, `--calc_kwargs`).
- INTERFACER's own symmetric-slab generator (`--sym_method internal`), which replaces the external code previously kept commented out.
- Interface energy from the Dupré relation, γ_int = γ_1 + γ_2 − W_ad.
- Surface formation energy against the chemical potential (`--sfe_scan1/2`), with the stability ranges of the terminations.
- scipy optimisation of the rigid-slab placement (`-topt`, `--topt_maxiter`), and 2D Minkowski reduction of the matched cell (`--minkowski`).
- Restarts for every engine (`--restart` stores ASE-based results in `ASE-calcs/`; CASTEP reuses finished runs) and `--overwrite`.
- Fixing the central layers instead of the bottom ones (`--fix_mode middle`).
- Logging with `-v` (details) and `-q` (warnings only); optimiser steps are only shown with `-v`.
- Warnings for clashing atoms in aligned slabs and interfaces.
- Dry runs (`-dry`/`--structures_only`) for all codes.
- The scans write their data as CSV (`conv_sep.csv`, `conv_htrans.csv`, `placement_opt.csv`, `sfe_vs_mu_<n>.csv`).
- Analysis notebooks for Tutorials 1, 2, 3 and the new Tutorial 6 (grain boundaries and three-layer interfaces).
- Test systems Li2S, Li, Na, β-Li3PS4 and Na3OCl (`tests/data/`).

**Changes and fixes**
- The lateral registry (`-thor`) is scanned before the separation (`-tsep`). After a scan the interface is optimised at fixed cell, so the scanned separation is kept.
- Slabs were sometimes built upside down (surface normal along −z), and `calc_R1` returned the identity for antiparallel vectors. Both are fixed, which also affected β-Li3PS4 | Li2S.
- VASP cells are made right-handed and shifted to non-negative z.
- The scan plots no longer switch matplotlib to the Agg backend, which had disabled inline plots in Jupyter. The lateral-scan heat map is centred on the sampled shifts.
- Removed dead and commented-out code.

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

# Package layout and migration guide

From version 1.0.0.dev0 the code is the installable Python package `interfacer` (in `src/interfacer`). Install it with `pip install -e .`, which also installs the `interfacer` command.

Old scripts and command lines keep working:

- `python interfacer-new.py <options>` takes exactly the same options as before; it calls `interfacer.cli.main`.
- `from tools.builder import *`, `from tools.analysis import *` and `from tools.external import *` still work; these modules now re-export the package.

## Where things moved

| Before | Now | Contents |
|---|---|---|
| `interfacer-new.py` (main block) | `interfacer/workflow.py` | `WorkflowConfig`, `InterfaceWorkflow` (one method per step), result dataclasses |
| `interfacer-new.py` (options) | `interfacer/cli.py` | `build_parser()`, `main(argv)`, `parse_miller`, `parse_hubbard_u`, `make_settings` |
| `interfacer-new.py` (functions) | `interfacer/convergence.py` | `check_surfaces`, `conv_layers`, `conv_layers_new`, `convSep`, `checkHorizontal`, `checkHorizontal2`, `cfactor` |
| `interfacer-new.py` (`flip_cell`, `make_interface`) | `interfacer/builder.py` | interface stacking |
| `tools/builder.py` | `interfacer/builder.py` | `make_slab`, `permute_surface`, `standardise_slab`, `slab_aligner`, `find_commensurate_supercell`, … |
| `tools/analysis.py` | `interfacer/analysis.py` | `misfit`, `surf_area`, `get_fu`, `find_prim`, `calculate_SFE`, `calculate_chem_pots`, `get_interface_energy`, … |
| `tools/external.py` (VASP) | `interfacer/calculators/vasp.py` | `call_vasp`, `handle_magmoms`, `set_hubU`, `make_potcar`, `vasp_continue`, … |
| `tools/external.py` (CASTEP) | `interfacer/calculators/castep.py` | `call_castep`, `parseCASTEP`, `parseCASTEP_ASE` |
| `tools/external.py` (ASE) | `interfacer/calculators/classical.py` | `call_ase`, `classical_calculator`, `call_dftb` |
| (new) | `interfacer/calculators/base.py` | `CalcSettings`, `Engine` |
| (new) | `interfacer/calculators/engines.py` | `VaspEngine`, `CastepEngine`, `AseEngine`, `get_engine` |
| (new) | `interfacer/calculators/optimise.py` | `relax` (shared ASE optimiser), `bottom_layer_indices` |
| (new) | `interfacer/io.py` | `write_struct`, `backup_output` |

## Running the workflow from Python

```python
import ase.io
from interfacer.calculators import CalcSettings, get_engine
from interfacer.workflow import InterfaceWorkflow, WorkflowConfig

engine = get_engine("mlip", CalcSettings(mlip="mace", mlip_model="small"))
cfg = WorkflowConfig(miller1=(1, 1, 1), miller2=(1, 1, 1), creps1=3, creps2=3, conv_sep=True)
results = InterfaceWorkflow(engine, cfg).run(ase.io.read("Cu.cif"), ase.io.read("Ni.cif"))
for r in results.interfaces:
    print(r.name, r.W_surf1, r.W_surf2, r.W_ad, r.W_ad_J)   # eV/nm^2 (and J/m^2)
```

Every `WorkflowConfig` field corresponds to a command-line option; the field comments name the option.

You can also run the steps one by one: `run_bulk`, `screen_surfaces`, `converge_layers`, `build_slabs`, `compute_sfe`, `select_slabs`, `align`, `optimise_aligned`, `optimise_interface` and `run_pair`.

Each run writes the results to `OUTPUT/results.json`. They include the bulk energies, the slab surface formation energies, and per interface the misfit, W_surf, separation, lateral shift and W_ad.

## Engines

All calculations go through an engine. Each engine provides `run(atoms, typ, name, ...)`, `single_point(...)` and `relax(...)`, and each returns `(energy, atoms)`.

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from interfacer.calculators import AseEngine, CalcSettings, get_engine

engine = get_engine("vasp", CalcSettings(encut=500, kpspacing=0.25, exe="mpirun -n 64 vasp_std"))
engine = AseEngine(EMT, label="EMT", settings=CalcSettings(fmax=0.02, relax_cell_bulk=True))  # any ASE calculator
E, cu = engine.relax(bulk("Cu", cubic=True), "bulk", relax_cell=True)
```

The convergence functions take the engine as a keyword argument, for example `convSep(slab1, slab2, engine=engine)`. If you leave it out, they use the engine set with `interfacer.convergence.set_default_engine`.

## New command-line options

These options apply to the ASE-based engines. `--fix_layers` and `--outdir` also apply to VASP and CASTEP.

| Option | Meaning |
|---|---|
| `--optimiser {BFGS,FIRE,LBFGS}` | ASE optimiser (default BFGS). |
| `--relax_cell_bulk` | Also relax the cell of the bulk structures and decomposition products. VASP and CASTEP always relax the bulk cell. |
| `--fix_layers N` | Keep the lowest N atomic layers of the slabs fixed during the slab optimisations. VASP gets selective dynamics; CASTEP gets ionic constraints. |
| `--outdir DIR` | Output directory (default `OUTPUT`). |
| `--version` | Print the version. |

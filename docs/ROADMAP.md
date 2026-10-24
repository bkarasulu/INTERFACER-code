# INTERFACER refactor roadmap and code overview

## Context
INTERFACER builds solid–solid interfaces. It cuts slabs from two bulk structures, finds a commensurate supercell, stacks the slabs, and runs a surface/interface energy workflow through VASP, CASTEP or simple ASE potentials (LJ, Morse, EMT). The code is one 1338-line script (`interfacer-new.py`) plus three `tools/` modules that import each other with `import *`. All logic runs at module level under `if __name__ == '__main__'`, and helper functions read the global `args`. The code has no docs, tests or packaging. The repo also tracks `__pycache__` and 179 MB of VASP output, including POTCAR files, which the VASP licence does not allow to be redistributed.

Goal: an installable, tested, documented package that keeps every existing feature, CLI flag and function name, fixes the bugs below, and adds MACE and other machine-learning potentials (MLIPs) through ASE.

User decisions: installable package with a CLI, with `interfacer-new.py` kept as a wrapper; strip the heavy tutorial files; support MACE, CHGNet, MatGL, SevenNet, ORB, fairchem (UMA), NequIP and Allegro, plus a generic hook for any ASE calculator; merge `symINTERFACER` first.

## Branches
| Branch | State | Action |
|---|---|---|
| `main` | base (75b4e14) | start point |
| `symINTERFACER` | +2 commits: `6f124df` makes SFE work for any species (removes the hardcoded Cl/Li/P/S); `d8ac655` adds `-sym_slabs1/2`, fixes `permute_surface`, adds the missing `()` in `check_surf_unique`, renames outputs and pre-optimises every slab | **merge first** and drop OLD-code/ (main deleted it). The branch also **breaks** things, which get fixed immediately after the merge: the `ase.build.stack` interface build is commented out, so `interface` is undefined unless `-tsep`/`-thor` is given; slab2 has no calculator in its SFE fallback; symmetric slabs are cut from the unrelaxed bulk |
| `interfacer_with_mace` | same commit as main | nothing to merge; this refactor delivers its intent |
| `fixing-types-in-parser` | already merged | none |

## Current code overview
**`interfacer-new.py`**
- Utilities: `grep`, `Popen4`, `flip_cell`
- Convergence and screening tests (all read the global `args`, `ecut`, `KP`, `exe`…): `check_surfaces` (`-tsur`), `conv_layers` / `conv_layers_new` (`-tlay`), `convSep` (`-tsep`), `checkHorizontal` (`-thor`), `checkHorizontal2` (a stub that calls `exit()`)
- Main flow: parse about 60 argparse flags → read bulks → bulk optimisation (or restart from `bulkN/OUTCAR`) → `make_slab` / read `-surf1/2` → optional surface, layer and symmetric-slab (`permute_surface`) screening → SFE (`calculate_SFE`, either with chemical potentials or with decomposition products) → rank slabs → for each slab pair: `slab_aligner` → `ase.build.stack` → optimise the aligned slabs → Wsurf → optional sep/horizontal scans → interface single point and optimisation → Wad. Outputs go to `OUTPUT/` and `interfaces.out`.

**`tools/builder.py`**: slab geometry. `make_slab`, `get_cut_vectors`, `cut_cell`, `wrap_coords`, `permute_surface`, `check_surf_unique` (pymatgen SlabGenerator), `reasnoble_slab_angles`, `match_angles2`, `LCM_float`, `find_commensurate_supercell`, `slab_aligner`, `AG_slab`, `acute_Gauss`, `basis_change`, `square_slab`, `rotate_slab`, `calc_R1`, `flip_handedness`, `RH`, `invert_axis`, `bot_to_top`, `number_density`, `atom_density_check`, `plot_slab_axes`, `print_cell`, `lcm`.

**`tools/external.py`**: calculator drivers. `call_vasp` (about 470 lines), `call_vasp_v2`, `call_castep`, `call_ase` (LJ/Morse/EMT with BFGS), `call_dftb` (stub), `parseCASTEP`, `parseCASTEP_ASE`, `handle_magmoms`, `set_hubU`, `get_potcar`, `make_potcar`, `vasp_continue`.

**`tools/analysis.py`**: `misfit`, `surf_area`, `volume`, `get_thickness`, `get_fu`, `find_prim`, `compute_rdf`, `minDist`, `minDist_ASE`, `dist`, `get_interface_energy`, `calculate_chem_pots`, `calculate_SFE`.

## Bugs found so far (to fix; full audit list completed in step 1)
- `-mlist` uses `type=tuple`, so the string "111" becomes the tuple `('1','1','1')` and user Miller lists are broken. Fix: parse with `ast.literal_eval` or `"1,1,1;1,1,0"`.
- `-kp_gr` is ignored: `KPgrid` is overwritten with `None` unless `-gamma` is set (lines 680–682).
- Aligned-slab restart checks `slab1/OUTCAR` but reads `slab1-aligned/OUTCAR`, and the real directory is `slab1_cutN-aligned`.
- Wad sign differs between the pre-optimisation and post-optimisation formulas (lines 1294 and 1314).
- `check_surfaces` returns `-min_Ws`, which flips the sign of the surface energy.
- `checkHorizontal` uses `zip(xrng, yrng)`, so it scans only the diagonal, not the 2D grid. `ifPlot` is unimplemented and the plot filename is wrong in `convSep` (`conv_htrans.png`).
- `conv_layers` indexes `E[i]` while starting at `i=lay_min`, so the convergence check is off when `lay_min>0`.
- `np.arange(sep_init, sep_final)` leaves out `sep_final`.
- The slab2 loop applies niggli/prim to `slab1`.
- The ASE-branch SFE only exists for slab2.
- The VASP call for the final interface optimisation does not pass `sigma`/`exe`.
- `exe` is undefined unless `prog=vasp`.
- `FixBondLengths` loop `for j in (i+1, ln1)` iterates over a 2-tuple instead of a range.
- `call_ase` returns energy `0.0` on failure, which silently corrupts results. Bulk "geom" never relaxes the cell.
- `rm -rf OUTPUT` runs unconditionally on every start. `os.system` and shell `grep` are used for file operations.
- Deprecated or removed ASE APIs (`set_calculator`, `get_calculator`), circular star-imports, Python 2 shims.
- Hardcoded cluster settings (`--partition=solbatsims`, `srun`).
- `wyckoff.dat` and `bvparm*.cif` are not referenced by any code.
- Found by the tools audit:
  - **Circular imports:** `builder`, `external` and `analysis` import each other with `import *`, so which names exist depends on import order (e.g. `builder` uses `stdout` it never imports).
  - **Duplicate check:** `check_surf_unique` compares `.all() == .all()`, so any two slabs with the same formula count as duplicates.
  - **Wrapping:** `cut_cell` does `y %= 1` on a local variable, so coordinates are never wrapped.
  - **Thickness:** `slab_aligner` ignores `thickness` (`crep` is unused).
  - **Silent errors:** a bare `except` around `match_angles2` hides every error.
  - **Repeated alignment:** `slab_aligner` re-aligns the already-aligned slab1 for every slab2 in the pair loop.
  - **`call_vasp`:**
    - A restart is forced and discards the caller's settings, and it restarts from POSCAR rather than CONTCAR.
    - It assumes SLURM (`int(getenv('SLURM_NTASKS'))`).
    - `IBRION`, `idipol` and `slayers` can be undefined.
    - The `ldauu` and `ldaul` lists can end up different lengths.
    - The passivation code has `bincount`/ragged `np.unique` crashes.
    - The `view` parameter shadows the `view()` function.
  - **`vasp_continue`:** `RUN10` is read as `RUN0` and gets overwritten.
  - **`call_castep`:**
    - Return type is inconsistent: it returns `atoms` on the dry-run and `calc=` paths, but `(E, atoms)` otherwise.
    - Constraint indices are 0-based global, but CASTEP expects 1-based within each species.
    - Forces are never parsed (`== "\*"`).
  - **`call_ase`:** `ctype=LJ` defaults to a class, so `.upper()` fails.
  - **Dead or broken functions:** `call_dftb` (Dftb not imported), `call_vasp_v2` (`system()` with no argument), `get_interface_energy` (unbound names).
  - **`calculate_SFE`:** both asserts always pass; SFE is printed as "eV/atom" but is actually eV/nm².
  - **Bulk settings:** bulk 1 and bulk 2 use different `xc` variables.
  - **Hardcoded paths:** `$HOME/covalent_radii.dat`, and a POTCAR path that ignores `xc` and setups.

## Target structure
```
pyproject.toml            # deps: ase>=3.23, pymatgen, spglib, numpy, matplotlib, tqdm; extras: [mace],[chgnet],[matgl],[sevenn],[orb],[fairchem],[nequip],[test],[docs]
src/interfacer/
  __init__.py  cli.py (argparse: every existing flag kept, plus new ones)  config.py (dataclass Settings replaces global args; optional YAML/TOML input file, CLI overrides)
  workflow.py   # main pipeline split into steps: run_bulk, build_slabs, screen_surfaces, compute_sfe, align, stack, scan_separation, scan_lateral, optimise_interface
  convergence.py  # check_surfaces, conv_layers, convSep, checkHorizontal (true 2D grid + heatmap)
  builder.py  analysis.py  # moved from tools/, same function names, explicit imports
  io.py         # write_structure(atoms, name, prog) replaces the ~30 copies of the castep/vasp/xyz write blocks; restart detection
  calculators/
    base.py     # Engine interface: single_point(atoms,name), relax(atoms,name,fix_cell,...), returns (E, atoms) like today
    vasp.py castep.py  # wrap the existing call_vasp/call_castep, behaviour unchanged
    classical.py       # LJ/Morse/EMT (old call_ase)
    mlip.py            # registry: mace, chgnet, matgl/m3gnet, sevennet, orb, fairchem, nequip/allegro, "custom:module:factory"; lazy imports with clear install hints; device/dtype/model args
    optimise.py        # shared ASE relax: BFGS/FIRE + FrechetCellFilter (bulk), FixAtoms (frozen bottom layers), trajectory output
tools/  -> thin compat shims re-exporting from interfacer.*
interfacer-new.py -> wrapper around interfacer.cli:main
tests/  docs/  tutorials/  .github/workflows/ci.yml
```
New CLI options (old options unchanged): `-prog mlip` (`ase` keeps its current meaning), `--mlip {mace,chgnet,...}`, `--mlip_model <name|path>`, `--device`, `--ase_opt {BFGS,FIRE,LBFGS}`, `--relax_cell_bulk`, `--fix_layers N`, `--input file.toml`, `--outdir`, `--overwrite`.
Pipeline restarts become uniform for every engine (skip a step when its output already exists unless `--overwrite` is given).

## Status
- **Phase 1 done:**
  - `symINTERFACER` merged.
  - Heavy VASP files and `__pycache__` untracked.
  - Bugs fixed in place (the script/`tools` layout is unchanged); see `known_issues.md` for the full list and the open issues.
- **Phase 2 started:** `tests/` has 43 regression and end-to-end tests (ASE/LJ, EMT and a VASP dry run).

## Roadmap (phases, each one or more commits on `claude/exciting-hamilton-eunp54`)
1. **Housekeeping.** Merge `symINTERFACER`. Remove `__pycache__` and the heavy VASP files (POTCAR, WAVECAR, CHG*, DOSCAR, EIGENVAL, vasprun.xml, PCDAT) from tracking, and add them to `.gitignore`. Finish the bug audit and record it in `docs/known_issues.md`.
2. **Characterisation tests first.** Using the ASE/LJ and EMT engines, write tests that capture current behaviour of the pure geometry and analysis functions (`make_slab`, `slab_aligner`, `misfit`, `surf_area`, `get_fu`, `calculate_SFE`, `find_commensurate_supercell`) on small bulks (Cu fcc, Na bcc, NaCl, and the Na3OCl and Li3PS4 CIFs from Tutorials). These tests guard the refactor.
3. **Package skeleton.** Add `pyproject.toml` and the `src/interfacer` layout. Move `builder` and `analysis` with explicit imports and modern ASE APIs. Add compat shims.
4. **Calculator layer.** Add the `Engine` abstraction. Wrap VASP and CASTEP without changing their behaviour, port the classical potentials, and add the shared ASE optimiser.
5. **Workflow refactor.** Turn the main block into `workflow.py` functions that take a `Settings` object (no globals). Deduplicate the slab1/slab2 and write blocks. Fix the listed bugs, each with a regression test.
6. **MLIP support.** Add `calculators/mlip.py` with the registry and lazy imports, plus a generic custom-calculator hook. Each backend gets a test that is skipped when the package is missing. Add a mocked test for the registry and argument passing.
7. **CLI and config.** Keep every existing flag. Add the new flags and TOML/YAML input (a TODO in the original code). Add `interfacer --version`. Add a `--dry` path that builds structures only.
8. **Docs.** Use MkDocs with mkdocstrings. Pages: installation (pip, conda, VASP/CASTEP environment variables, GPU/MLIP extras), quickstart, CLI reference generated from argparse, theory (Wsurf, Wad, SFE, misfit, with formulas and units), Python API reference, a mapping from the old script to the new modules, and a changelog. Rewrite README.md.
9. **Tutorials.** (a) Quick LJ/EMT Cu/Ni demo that runs in seconds. (b) Na3OCl|Na with MACE-MP-0 (an ML version of the existing VASP antiperovskite tutorial). (c) β-Li3PS4|Li with the existing VASP inputs and reference outputs. (d) Symmetric slabs and SFE with decomposition products. (e) Python API notebook. Each has a README and `run.sh`.
10. **CI.** GitHub Actions with pytest on Python 3.10–3.12 using the core deps. An optional job installs mace-torch on CPU. Add ruff linting.

## Verification
- `pip install -e .[test]`, then `pytest -q`. Characterisation tests must pass before and after each move. Bug-fix tests fail on old code and pass on new code.
- `interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 0 0 -m2 1 0 0 -prog ase -quick` and `python interfacer-new.py <the same old flags from Tutorials/*/run_script> -dry` both run end to end and produce `OUTPUT/` files.
- With mace-torch installed: `interfacer ... -prog mlip --mlip mace --mlip_model small` on Na3OCl|Na gives finite Wsurf/Wad values and a relaxed interface.
- `mkdocs build --strict` passes. Tutorial (a) runs in CI.
- VASP and CASTEP paths can't run here. They are checked with `-dry` and with mocked calculator tests that confirm the INCAR and parameter settings match the old code.

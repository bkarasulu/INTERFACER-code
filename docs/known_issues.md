# Fixed bugs and known issues

This file lists the bugs that were fixed during phase 1 of the refactor (see `ROADMAP.md`), and the issues that are still open.
Regression tests for the fixes are in `tests/`; run them with `python -m pytest`. File locations below refer to the pre-1.0 layout (`interfacer-new.py`, `tools/`); see `package_layout.md` for where the code is now.

## Fixed

### Results (energies, structures)
| Area | Problem | Fix |
|---|---|---|
| Units | `cfactor=6.2415093` was labelled eV/nm² → J/m², but it is the inverse factor. Every value printed in J/m² was ~39× too large. | `cfactor=0.160217663` (1 eV/nm² = 0.1602 J/m²). |
| Interface construction | `ase.build.stack(distance=sep)` treats `distance` as the *minimum interatomic distance* and reaches it by shifting the slabs laterally and scaling the cells. The vertical gap was therefore not `-sep` (0.04 Å in a Cu\|Ni(100) test), layers clashed, and the lateral scan (`-thor`) was undone. | New `make_interface()`: the gap along the surface normal between the facing atomic planes is exactly `sep`. |
| Work of adhesion | The sign differed between the pre-optimisation and final formulas. With the default small vacuum (`-vac_int 1`, or `sep/2` after `-tsep`/`-thor`) the cell contains **two** interfaces, but W_ad was divided by one area. | `W_ad = (E_slab1 + E_slab2 − E_int) / (n_int·A)`, positive for a bound interface; `n_int=2` if the vacuum is ≤ 4 Å per side, else 1. A Cu\|Ni(111) EMT test now gives 2.59 J/m² (periodic) vs 2.55 J/m² (vacuum), both consistent and below γ(Cu)+γ(Ni). |
| `check_surfaces` (`-tsur`) | Returned `-W_surf` (sign flipped). The selected Miller indices and the converged number of layers (`-tlay`) were computed *after* the slabs had been built, so they were never used. | Positive W_surf; the surface/layer tests now run before the slabs are created. |
| `checkHorizontal` (`-thor`) | `zip(xrng, yrng)` scanned only the diagonal (x=y), and the unshifted stacking (0,0) was not included. | Full 2D grid including (0,0); optional 2D heat map (`OUTPUT/conv_htrans.png`). |
| `conv_layers` (`-tlay`) | Indexed `E[i]` with `i` starting at `lay_min`, so the convergence check compared the wrong energies when `lay_min>0`. | Compares the last two energies. |
| `convSep` (`-tsep`) | `np.arange` left out `sep_final`; the plot was saved as `conv_htrans.png`. | `sep_final` included; plot saved as `conv_sep.png`. |
| Symmetric slabs (`-sym_slabs1/2`) | pymatgen slabs had their surface vectors out of the xy-plane, giving wrong supercell matching (22% misfit, 35% density change in a Cu\|Ni test). Their extra string arrays (`bulk_wyckoff`, …) corrupted the interface/extxyz output. They were cut from the *unrelaxed* bulk while the SFE used the relaxed bulk energy. The thickness check used `c − 2·vac`, which is wrong for an oblique c. | `standardise_slab()` (normal along z, plain Atoms); slabs cut from the relaxed bulk; thickness from the atomic extent along the normal. |
| `check_surf_unique` | Compared `rotations.all() == rotations.all()`, i.e. two booleans, so any two slabs with the same formula counted as duplicates. Missing `()` on `get_atomic_numbers`. | Compares the sets of symmetry rotations. |
| Slab pairs (`-no_slabs > 1`) | `slab_aligner` re-aligned the already aligned (and optimised) slab1 for every slab2. | Each pair starts from copies of the original slabs. |
| `calculate_SFE` | Both `assert`s always passed. Exact float comparison for stoichiometry, `int()` truncation, and a crash when no integer formula-unit count existed. SFE printed as "eV/atom" (it is eV/nm²). The SFE of slab2 used a slab without a calculator (non-ASE codes). Zero-chemical-potential fallback was only used for slab2. | Proper `ValueError`s, tolerant comparisons, robust reference; correct units; the same code path for both slabs. |
| Decomposition products (`-decomps1/2`) | Optimised as slabs (dipole correction, fixed cell), and the input structure (not the optimised one) was used. | Optimised as bulk phases; the optimised structure is used. |
| VASP k-points (`-kp_gr`) | `-kp_gr` was ignored (`KPgrid` was reset to `None` unless `-gamma`). When a grid was used, explicit Monkhorst-Pack points (in reciprocal units) were written with `reciprocal=False`, i.e. as Cartesian coordinates. | `-kp_gr` is used; a standard Gamma-centred grid is written. |
| VASP slab relaxations | Slab pre-optimisations and the `-tsur` slabs used ISIF=3 (full cell relaxation), which shrinks the vacuum. | Fixed cell (ISIF=2), as for the aligned slabs. |

### Crashes and wrong behaviour
- **VASP with current ASE:** logical INCAR tags were given as `0/1` or `".True."`; recent ASE requires Python booleans, so every VASP run failed when writing the INCAR.
- **`-mlist`:** used `type=tuple` (`"111"` → `('1','1','1')`). It now accepts `100 110 111`, `1-10`, `1,1,0` and `(1,1,0)`.
- **numpy 2:** `np.cross` of 2D vectors (in `reasnoble_slab_angles`) is no longer allowed.
- **Quick run crash:** `-quick` crashed on `slab.name` (merged `symINTERFACER` code).
- **Interface build regression:** the interface was no longer built from the optimised slabs (also from `symINTERFACER`).
- **Circular imports:** `tools.builder`, `tools.external` and `tools.analysis` imported each other with `import *`, so the available names depended on import order (e.g. `builder` used `stdout` without importing it). They now import nothing from each other.
- **`call_vasp` errors:**
  - It assumed SLURM (`int(getenv('SLURM_NTASKS'))`).
  - It built a broken command if the executable was not called `vasp_*`.
  - `IBRION`, `idipol` and `slayers` could be undefined.
  - LDAUU and LDAUL could have different lengths (e.g. U on s orbitals).
  - The passivation code crashed (`bincount`, ragged `np.unique`, lost neighbour lists, unconditional "layer matches" messages).
  - The `view` argument shadowed `view()`.
  - It needed `$HOME/covalent_radii.dat`; ASE's radii are now the fallback.
- **VASP restarts:** a non-converged run restarted from POSCAR with default settings. It now continues from CONTCAR with the previous INCAR/KPOINTS.
- **`vasp_continue`:** read `RUN10` as `RUN0` and overwrote it.
- **`call_castep`:**
  - Returned `atoms` instead of `(E, atoms)` on some paths.
  - Constraint indices were global 0-based instead of 1-based per species.
  - Force lines were never parsed (`== "\*"`).
  - `calc.Task` was set instead of `calc.param.task`.
  - It crashed when no k-point setting was given.
  - The `-cpp` option was ignored.
- **`call_ase`:** returned an energy of `0.0` silently when the optimisation failed; the default `ctype=LJ` (a class) crashed; it needed an existing `OUTPUT/` directory and an `atoms.name`.
- **Dead and broken functions:** `call_dftb` (Dftb not imported), `call_vasp_v2` (`system()` without argument), `get_interface_energy` (unbound names; rewritten and now usable from scripts), `checkHorizontal2` (stub; now raises `NotImplementedError`).
- **Dry runs:** the `-dry` flag was ignored. VASP/CASTEP inputs are now prepared without running, and energy-based steps are skipped.
- **Output handling:** `rm -rf OUTPUT` ran on every start; the previous output is now kept in `OUTPUT_prev`.
- **Removed ASE calls:** `set_calculator`/`get_calculator` (deprecated/removed in ASE) were replaced by `atoms.calc`.
- **Clean-up:** debug prints, Python 2 shims and invalid escape sequences were removed.

## Known issues (not fixed yet)
- `-th/--thickness` is only used for symmetric slabs; `slab_aligner` ignores it. The slab thickness is set by `-cr1/-cr2` (or `-tlay`).
- `-flip2` does not work reliably (the cell angles are not preserved).
- With the ASE potentials, bulk structures are relaxed at fixed cell unless `--relax_cell_bulk` is given.
- The layer convergence test (`-tlay`) converges the energy per atom, which converges slowly with thickness; a surface-energy criterion would be better.
- The CASTEP path and VASP pseudo-hydrogen passivation (`-pass`) could not be tested here (no CASTEP/VASP binaries); the VASP input generation is tested with `-dry`.
- No restart support for the convergence tests (`-tsep`, `-thor`, `-tlay`) with ASE.

# Workflow and output files

This page describes the steps of a run in order, followed by the three-layer and grain-boundary modes. Each step is a method of [`InterfaceWorkflow`](api/workflow.md).

## 1. Bulk optimisation (`run_bulk`)

Both bulk structures are optimised.

- **VASP/CASTEP:** full cell relaxation (ISIF=3).
- **ML potentials (`-prog mlip`):** cell relaxation by default.
- **Simple ASE potentials:** positions only, unless `--relax_cell_bulk` is given.

The energies per atom, E_bulk,1 and E_bulk,2, are the references for all surface energies. With `-prim`, the primitive cells are used.

## 2. Surface screening (`screen_surfaces`, `-tsur`)

For each Miller index in `-mlist` (default 100 010 001 110 111):

- a slab with `-cr1/-cr2` repetitions is cut and padded with `-vac_slab` of vacuum;
- its energy is computed: a fixed-cell relaxation for DFT, a single point for ASE engines;
- W_surf is calculated.

The surface with the lowest W_surf replaces `-m1`/`-m2`.

## 3. Layer convergence (`converge_layers`, `-tlay`)

The slab is repeated from `-lay_min` to `-lay_max` times along c, until the energy per atom changes by less than 0.01 eV/atom. The converged number replaces `-cr1`/`-cr2`.

## 4. Slab creation (`build_slabs`)

There are three ways to get the slabs:

- **`make_slab`** (default) cuts the slab along the Miller plane, puts the surface normal along z, reduces the in-plane cell (Gauss reduction) and repeats it `-cr` times.
- **Symmetric slabs** (`-sym_slabs1/2`): every termination is generated, keeping the symmetric ones that are at least `-th` Å thick, without duplicates. These may be non-stoichiometric.
    - `--sym_method pymatgen` (default) uses pymatgen's `SlabGenerator`.
    - `--sym_method internal` uses INTERFACER's own generator (`builder.symmetric_slabs`). It cuts a thick slab with `make_slab` and groups the atoms into atomic planes. Each plane within one bulk repeat is tried as the bottom termination, and the slab is extended plane by plane until it is at least `-th` Å thick and its two surfaces are equivalent. Equivalent means that the slab's symmetry (from spglib) contains an operation that turns the surface normal upside down: inversion, a mirror plane parallel to the surface, or a two-fold axis in the surface. For the tested systems (Na3OCl, Cu and NaCl in the (100), (110) and (111) planes) it gives the same terminations as pymatgen.
- **From a file** (`-surf1/-surf2`): the surface is only rotated so that its normal lies along z.

## 5. Surface formation energies (`compute_sfe`, `select_slabs`)

Each slab is relaxed at fixed cell, optionally with frozen bottom layers (`--fix_layers`). Its surface formation energy (SFE) is computed; see [Theory](theory.md#surface-formation-energy).

Non-stoichiometric slabs need chemical potentials, either directly (`-cp1/-cp2`) or from decomposition products (`-decomps1/-decomps2`). With `--sfe_scan1 X,Y,min,max` the SFEs are also computed as a function of the chemical potential of element X, with Y compensating ([Theory](theory.md#sfe-as-a-function-of-the-chemical-potential-sfe_scan12)). The `-no_slabs` lowest-SFE slabs of each material are kept. `-surf_only` stops here.

## 6. Alignment (`align`)

For each pair of slabs, `slab_aligner` searches for supercells of the two surface lattices that match within `-pt` %:

- in-plane vectors are enumerated up to the length L;
- L starts at `-Linit` and grows by `-Lstep` up to `-msd`;
- among the matches, the smallest area with the most rectangular angle is chosen.

The misfit is printed before and after alignment. `--minkowski` reduces the matched in-plane cell to its shortest, most orthogonal basis. The aligned slabs are checked for clashing atoms (closer than 0.6 times the sum of the covalent radii), and a warning is printed if any are found. The interfaces are checked in the same way.

## 7. Aligned slabs (`optimise_aligned`)

Both aligned slabs are relaxed at fixed cell. This gives E_slab,1, E_slab,2 and W_surf,1, W_surf,2.

## 8. Interface (`run_pair`)

- **Stacking:** `make_interface` places slab 2 on slab 1 with a vertical gap of `-sep` Å between the outermost atomic planes and `-vac_int` Å of vacuum on each side. The in-plane cell is the average of the two slab cells.
- **Lateral scan (`-thor`):** slab 1 is shifted on a 2D grid of fractions of a and b (`-hor_steps`) at the separation `-sep`, and the lowest-energy registry is kept.
- **Separation scan (`-tsep`):** single points for gaps from `-sep_init` to `-sep_final` in steps of `-sep_step`, at the best registry if `-thor` was also used. The vacuum is sep/2, so the periodic images form a second, equivalent interface. The lowest-energy gap is kept.

- **Placement optimisation (`-topt`):** a scipy minimisation (Nelder–Mead) of the single-point energy of the rigid slabs with respect to $(x, y, \mathrm{sep})$. It starts from the results of `-thor`/`-tsep`, if given. `--topt_maxiter` sets the iteration limit (default 60).

The lateral scan runs first because a separation scanned at an unfavourable registry comes out too large.

## 9. Interface optimisation (`optimise_interface`)

The interface is relaxed and the work of adhesion is computed; see [Theory](theory.md#work-of-adhesion).

- **After `-tsep`, `-thor` or `-topt`:** the cell is kept fixed (VASP ISIF=2), so the scanned separation is kept.
- **Otherwise, VASP:** full cell relaxation (ISIF=3).
- **Otherwise, ASE and ML engines:** atomic positions plus the cell length along z. The matched in-plane cell is kept, and the second, periodic interface can relax.

The interface energy $\gamma_\mathrm{int} = \gamma_1 + \gamma_2 - W_\mathrm{ad}$ is printed after $W_\mathrm{ad}$ ([Theory](theory.md#interface-energy)).

**Fixed layers:** `--fix_layers N` keeps N atomic layers of each slab fixed during the slab relaxations. With `--fix_mode bottom` (default) these are the lowest layers; with `--fix_mode middle` they are the central, bulk-like layers, which suits symmetric slabs. VASP uses selective dynamics, CASTEP ionic constraints, and the ASE-based engines `FixAtoms`.

## Three-layer interfaces (`-i3`)

With a third bulk structure (`-i3`, `-m3`, `-cr3`), every pair of selected slabs of materials 1 and 2 is also combined with a slab of material 3 into slab 1 | slab 2 | slab 3:

1. Bulk 3 is optimised and slab 3 is cut (`setup_material3`).
2. `builder.align_three` finds a common in-plane cell. Slabs 1 and 2 are aligned first; the aligned slab 2 is then matched with slab 3, and the same supercell transformation is applied to slab 1.
3. The three aligned slabs are relaxed at fixed cell.
4. They are stacked with the gaps `-sep` (1|2) and `-sep2` (2|3, default `-sep`), using `builder.make_multilayer`. The vacuum is at least 4 Å on each side, so that slab 3 and slab 1 do not form an extra interface across the periodic boundary.
5. The stack is relaxed at fixed cell, and $W_\mathrm{ad,total} = W_\mathrm{ad}(1|2) + W_\mathrm{ad}(2|3)$ is reported ([Theory](theory.md#three-layer-interfaces)).

The results are stored in `results.json` under `trilayers`. Tutorial 6 has an example.

## Grain boundaries (`--gb`)

`--gb tilt` or `--gb twist` builds a grain boundary of material 1 instead of an interface of two materials; `-i2` is then not needed. The boundary plane is `-m1`.

1. The bulk is optimised.
2. The bicrystal is built with `grainboundary.symmetric_tilt_gb` or `grainboundary.twist_gb` (`--gb_mn m n`). Each grain is `-cr1` cells thick (at least 2).
3. The bicrystal is relaxed with a variable cell length along the normal.
4. $\gamma_\mathrm{GB}$ is reported ([Theory](theory.md#grain-boundary-energy)) and stored under `grain_boundaries` in `results.json`.

`--gb_expansion` adds a rigid-body separation at the tilt boundaries. Twist boundaries need a square or hexagonal plane. For other planes, and for asymmetric boundaries, use the same structure for `-i1` and `-i2` with different Miller indices; this is an ordinary interface.

## Output, logging and restarts

- **Logging:** output goes through Python's `logging` module (logger `interfacer`). `-v` adds details, such as the supercell search, the passivation and every optimiser step. `-q` shows warnings only. In Python, use `interfacer.log.setup_logging(verbosity)`.
- **Restarts:** VASP reuses converged runs in its run directories, and continues unconverged ones (see [DFT codes](dft_codes.md)). CASTEP reuses a finished `CASTEP-tmp/<name>.castep`. With `--restart`, the ASE-based engines (EMT/LJ/Morse, ML potentials, GPAW, ONETEP) store every result in `ASE-calcs/<name>.xyz`, together with a fingerprint of the structure and the settings. The result is reused when both are unchanged, including the single points of the scans.
- **Overwriting:** `--overwrite` recomputes everything. Existing VASP run directories are moved to `<dir>_old<n>`.
- **Dry runs:** `-dry` (alias `--structures_only`) builds and writes all structures without any energy calculation, for every code.

## Output files

| File | Content |
|---|---|
| `interfaces.out` (`-o`) | W_surf, W_ad and γ_int of every interface; W_ad,total of three-layer interfaces; γ_GB. |
| `OUTPUT/results.json` | All results: bulk energies, slab SFEs, and per interface the misfit, W_surf, separation, lateral shift, W_ad, γ_int and area; `trilayers` and `grain_boundaries`. |
| `OUTPUT/bulk{1,2}_opted.*` | Optimised bulk structures. |
| `OUTPUT/<formula>_<hkl>.*` | Slabs of the surface screening (`-tsur`). |
| `OUTPUT/<formula>_<hkl>-layer.*` | Layer convergence frames (`-tlay`), plus `conv_layers_*.png`. |
| `OUTPUT/slab{1,2}_cut<i>_pre[_opted].*` | Slabs before alignment, before and after relaxation. |
| `OUTPUT/slab{1,2}_cut<i>_aligned[_opted].*` | Aligned slabs, before and after relaxation. |
| `OUTPUT/interface_cuts_<i>_<j>.*` | Interface stacked from the unrelaxed aligned slabs. |
| `OUTPUT/interface-pre_opt_cuts_<i>_<j>.*` | Interface stacked from the relaxed slabs. |
| `OUTPUT/int-sep_<d>A.*`, `conv_sep.{png,csv}` | Separation scan (`-tsep`). |
| `OUTPUT/htrans_x<x>-y<y>.*`, `conv_htrans.{png,csv}` | Lateral scan (`-thor`). |
| `OUTPUT/interface-placement_opt.*`, `placement_opt.csv` | Placement optimisation (`-topt`). |
| `OUTPUT/sfe_vs_mu_<n>.{csv,png}` | SFE against the chemical potential (`--sfe_scan1/2`). |
| `OUTPUT/slab{1,2,3}_trilayer_*`, `trilayer_cuts_*` | Three-layer interfaces (`-i3`). |
| `OUTPUT/gb_tilt_<hkl>[_opted].*`, `gb_twist_<hkl>_S<Σ>[_opted].*` | Grain boundaries (`--gb`). |
| `OUTPUT/interface_cuts_<i>_<j>_opted.{*,cif}` | Final optimised interface. |
| `OUTPUT/*_opt.traj` | Optimisation trajectories (ASE engines). |
| `bulk1/`, `slab1_cut0_pre/`, …, `VASP-tmp/` | VASP run directories; existing runs are restarted or reused. |
| `CASTEP-tmp/`, `GPAW-tmp/`, `ONETEP-tmp/` | CASTEP, GPAW and ONETEP runs. |
| `ASE-calcs/` | Stored results of the ASE-based engines (`--restart`). |

The extension follows the code: `.vasp` (POSCAR) for VASP, `.cell` for CASTEP, `.xyz` (extended XYZ) for ASE and ML potentials. The previous `OUTPUT/` is moved to `OUTPUT_prev/` at the start of a run.

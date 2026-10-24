# Workflow and output files

This page describes the steps of a run in order. Each step is a method of [`InterfaceWorkflow`](api/workflow.md).

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
- **Symmetric slabs** (`-sym_slabs1/2`): pymatgen's `SlabGenerator` makes every termination, keeping the symmetric ones at least `-th` Å thick and without duplicates. These may be non-stoichiometric.
- **From a file** (`-surf1/-surf2`): the surface is only rotated so that its normal lies along z.

## 5. Surface formation energies (`compute_sfe`, `select_slabs`)

Each slab is relaxed at fixed cell, optionally with frozen bottom layers (`--fix_layers`). Its surface formation energy (SFE) is computed; see [Theory](theory.md#surface-formation-energy).

Non-stoichiometric slabs need chemical potentials, either directly (`-cp1/-cp2`) or from decomposition products (`-decomps1/-decomps2`). The `-no_slabs` lowest-SFE slabs of each material are kept. `-surf_only` stops here.

## 6. Alignment (`align`)

For each pair of slabs, `slab_aligner` searches for supercells of the two surface lattices that match within `-pt` %:

- in-plane vectors are enumerated up to the length L;
- L starts at `-Linit` and grows by `-Lstep` up to `-msd`;
- among the matches, the smallest area with the most rectangular angle is chosen.

The misfit is printed before and after alignment.

## 7. Aligned slabs (`optimise_aligned`)

Both aligned slabs are relaxed at fixed cell. This gives E_slab,1, E_slab,2 and W_surf,1, W_surf,2.

## 8. Interface (`run_pair`)

- **Stacking:** `make_interface` places slab 2 on slab 1 with a vertical gap of `-sep` Å between the outermost atomic planes and `-vac_int` Å of vacuum on each side. The in-plane cell is the average of the two slab cells.
- **Separation scan (`-tsep`):** single points for gaps from `-sep_init` to `-sep_final` in steps of `-sep_step`. The vacuum is sep/2, so the periodic images form a second, equivalent interface. The lowest-energy gap is kept.
- **Lateral scan (`-thor`):** slab 1 is shifted on a 2D grid of fractions of a and b (`-hor_steps`), and the lowest-energy registry is kept.

## 9. Interface optimisation (`optimise_interface`)

The interface is relaxed and the work of adhesion is computed; see [Theory](theory.md#work-of-adhesion).

- **VASP:** full cell relaxation (ISIF=3).
- **ASE and ML engines:** atomic positions plus the cell length along z. The matched in-plane cell is kept, and the second, periodic interface can relax.

## Output files

| File | Content |
|---|---|
| `interfaces.out` (`-o`) | W_surf and W_ad of every interface. |
| `OUTPUT/results.json` | All results: bulk energies, slab SFEs, and per interface the misfit, W_surf, separation, lateral shift, W_ad and area. |
| `OUTPUT/bulk{1,2}_opted.*` | Optimised bulk structures. |
| `OUTPUT/<formula>_<hkl>.*` | Slabs of the surface screening (`-tsur`). |
| `OUTPUT/<formula>_<hkl>-layer.*` | Layer convergence frames (`-tlay`), plus `conv_layers_*.png`. |
| `OUTPUT/slab{1,2}_cut<i>_pre[_opted].*` | Slabs before alignment, before and after relaxation. |
| `OUTPUT/slab{1,2}_cut<i>_aligned[_opted].*` | Aligned slabs, before and after relaxation. |
| `OUTPUT/interface_cuts_<i>_<j>.*` | Interface stacked from the unrelaxed aligned slabs. |
| `OUTPUT/interface-pre_opt_cuts_<i>_<j>.*` | Interface stacked from the relaxed slabs. |
| `OUTPUT/int-sep_<d>A.*`, `conv_sep.png` | Separation scan (`-tsep`). |
| `OUTPUT/htrans_x<x>-y<y>.*`, `conv_htrans.png` | Lateral scan (`-thor`). |
| `OUTPUT/interface_cuts_<i>_<j>_opted.{*,cif}` | Final optimised interface. |
| `OUTPUT/*_opt.traj` | Optimisation trajectories (ASE engines). |
| `bulk1/`, `slab1_cut0_pre/`, …, `VASP-tmp/` | VASP run directories; existing runs are restarted or reused. |
| `CASTEP-tmp/` | CASTEP runs. |

The extension follows the code: `.vasp` (POSCAR) for VASP, `.cell` for CASTEP, `.xyz` (extended XYZ) for ASE and ML potentials. The previous `OUTPUT/` is moved to `OUTPUT_prev/` at the start of a run.

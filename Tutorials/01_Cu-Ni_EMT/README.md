# Tutorial 1: Cu | Ni with EMT (seconds)

This tutorial runs the complete workflow, with all screening and convergence tests, using ASE's EMT potential. It needs no DFT code or ML package and takes about 2 s.

```bash
./run.sh            # = interfacer --input input.toml
```

## What it does (`input.toml`)

1. Relaxes the fcc Cu and Ni lattices with EMT (`relax_cell_bulk`).
2. Surface screening (`tsur`): (100), (110) and (111) slabs of 3 cubic cells. (111) has the lowest surface energy for both metals and is selected.
3. Finds a common supercell. The hexagonal (111) cells match within 4 % (misfit 2.9 % with the relaxed lattices).
4. Optimises the aligned slabs, giving W_surf.
5. Lateral scan (`thor`) on a 4×4 grid of shifts with rigid slabs, then a separation scan (`tsep`) from 1.5 to 3.0 Å at the best registry.
6. Optimises the interface at fixed cell, keeping the scanned separation, and reports the work of adhesion.

## Expected results

```text
W_surf  (Cu_100): 7.08 eV/nm^2 = 1.13 J/m^2
W_surf  (Cu_110): 7.69 eV/nm^2 = 1.23 J/m^2
W_surf  (Cu_111): 6.52 eV/nm^2 = 1.04 J/m^2
...
Selected surfaces: (1, 1, 1) for material 1 and (1, 1, 1) for material 2
Wsurf_1 (Cu_111): 6.52 eV/nm^2 = 1.04 J/m^2
Wsurf_2 (Ni_111): 10.64 eV/nm^2 = 1.70 J/m^2
Minimum-energy horizontal translation: x= 0.25, y= 0.50
Finding the optimal interface (interlayer) separation at the lateral shift (0.25, 0.50).
convSep: Minimum-energy separation is 2.00 A
(fixed cell: the separation from the -tsep/-thor scan is kept)
W_ad (work of adhesion, 2 interface(s) in the cell) for the final optimised interface: 16.42 eV/nm^2 = 2.63 J/m^2
```

The surface energies follow (111) < (100) < (110), as expected for fcc metals.

For a coherent interface, W_ad must not exceed γ(Cu) + γ(Ni) = 2.74 J/m². The 2.63 J/m² obtained implies a low interface energy, γ_int ≈ 0.1 J/m², as expected for two similar fcc metals.

The order of the scans matters: at the initial (on-top) registry, the rigid separation scan would give 2.5 Å, which is too large. INTERFACER therefore scans the lateral registry first, then the separation at the best registry. The best separation, 2.0 Å, is close to the fcc(111) interlayer spacing.

## Things to try

- `./run.sh -thor` (or edit `input.toml`) to switch off a test.
- `./run.sh -vac_int 8`: an interface with vacuum and free surfaces instead of the periodic cell with two interfaces. W_ad comes out about the same (2.60 J/m²).
- `./run.sh --fix_layers 1 --optimiser FIRE`.
- Open the structures in `OUTPUT/` with `ase gui OUTPUT/interface_cuts_0_0_opted.xyz`, and look at `OUTPUT/conv_sep.png` and `OUTPUT/conv_htrans.png`.

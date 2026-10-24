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
5. Separation scan (`tsep`) from 1.5 to 3.0 Å with rigid slabs, then a lateral scan (`thor`) on a 4×4 grid of shifts.
6. Optimises the interface and reports the work of adhesion.

## Expected results

```text
W_surf  (Cu_100): 7.08 eV/nm^2 = 1.13 J/m^2
W_surf  (Cu_110): 7.69 eV/nm^2 = 1.23 J/m^2
W_surf  (Cu_111): 6.52 eV/nm^2 = 1.04 J/m^2
...
Selected surfaces: (1, 1, 1) for material 1 and (1, 1, 1) for material 2
Wsurf_1 (Cu_111): 6.52 eV/nm^2 = 1.04 J/m^2
Wsurf_2 (Ni_111): 10.64 eV/nm^2 = 1.70 J/m^2
convSep: Minimum-energy separation is 2.50 A
Minimum-energy horizontal translation: x= 0.75, y= 0.00
W_ad (work of adhesion, 2 interface(s) in the cell) for the final optimised interface: 16.46 eV/nm^2 = 2.64 J/m^2
```

The surface energies follow (111) < (100) < (110), as expected for fcc metals.

For a coherent interface, W_ad must not exceed γ(Cu) + γ(Ni) = 2.74 J/m². The 2.64 J/m² obtained implies a low interface energy, γ_int ≈ 0.1 J/m², as expected for two similar fcc metals.

Note that the best separation from the rigid scan (2.5 Å, at the initial registry) is larger than after relaxation. Run the lateral scan *before* trusting a separation, or scan both.

## Things to try

- `./run.sh -thor` (or edit `input.toml`) to switch off a test.
- `./run.sh -vac_int 8`: an interface with vacuum and free surfaces instead of the periodic cell with two interfaces. W_ad comes out about the same (2.60 J/m²).
- `./run.sh --fix_layers 1 --optimiser FIRE`.
- Open the structures in `OUTPUT/` with `ase gui OUTPUT/interface_cuts_0_0_opted.xyz`, and look at `OUTPUT/conv_sep.png` and `OUTPUT/conv_htrans.png`.

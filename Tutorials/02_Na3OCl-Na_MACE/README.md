# Tutorial 2: Na3OCl | Na with MACE-MP-0

This tutorial builds the interface between the antiperovskite solid electrolyte Na3OCl and a Na-metal anode with the MACE-MP-0 foundation model. It is the same system as the VASP reference run in `../Antiperovskites`, and takes about 1 minute on a laptop CPU with the `small` model.

```bash
pip install mace-torch          # or: pip install -e "../..[mace]"
./run.sh                        # = interfacer --input input.toml
./run.sh --device cuda          # on a GPU
```

The MACE-MP-0 model is downloaded automatically on the first run.

## What it does (`input.toml`)

1. Relaxes the Na3OCl (Pm-3m, a = 4.50 Å) and bcc Na (a = 4.17 Å) lattices with MACE.
2. Cuts Na3OCl(100) (4 cells) and Na(100) (6 cells) slabs and computes their surface energies.
3. Matches the square (100) cells. The misfit is 7.4 % for the primitive cells and 3.7 % for the commensurate supercell. `initLength = 8` starts the supercell search at 8 Å.
4. Optimises the aligned slabs, scans the separation from 2.0 to 3.5 Å, and optimises the interface.

## Expected results (MACE-MP-0 small, CPU)

```text
Surface formation energy of slab1_cut0: 2.09 eV/nm^2 = 0.33 J/m^2      # Na3OCl(100)
Surface formation energy of slab2_cut0: 1.53 eV/nm^2 = 0.25 J/m^2      # Na(100)
Misfit (mu) of slab1_cut0_aligned and slab2_cut0_aligned (after alignment): 3.73%
convSep: Minimum-energy separation is 3.00 A
W_ad (work of adhesion, 2 interface(s) in the cell) for the final optimised interface: 2.65 eV/nm^2 = 0.42 J/m^2
```

For comparison, PBE gives about 0.22 J/m² for Na(100). W_ad (0.42 J/m²) is below the sum of the surface energies (0.58 J/m²), as it must be.

Results differ slightly between models; try `mlip_model = "medium-mpa-0"` or `"medium-omat-0"`. Validate important numbers with DFT (`prog = "vasp"`).

## Next

- Tutorial 3 uses the *symmetric* Na3OCl(100) terminations, NaCl- or Na2O-like, and their surface formation energies in the Na-rich limit.
- Add `thor = true` to scan the lateral registry, or `tsur = true` with `mlist = ["100", "110", "111"]` to screen surfaces.

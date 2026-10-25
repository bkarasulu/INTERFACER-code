# Tutorial 3: symmetric slabs and surface formation energies (Na3OCl | Na, MACE-MP-0)

Ionic materials often have several possible terminations. Symmetric slabs avoid an artificial dipole, but they are usually **non-stoichiometric**. Their stability then depends on the chemical potentials, expressed through the surface formation energy (SFE); see [Theory](../../docs/theory.md#surface-formation-energy).

This tutorial:

1. creates all symmetric Na3OCl(100) slabs at least 8 Å thick (`sym_slabs1`, `thickness`), with pymatgen's SlabGenerator. Two terminations come out: NaCl-terminated (Cl4Na10O3 per cell, i.e. Na3OCl + extra NaCl) and Na2O-terminated (Cl3Na11O4, i.e. Na3OCl + extra Na2O);
2. computes their SFEs in the **Na-rich limit**, the relevant one at a Na-metal anode. The chemical potentials come from the decomposition products Na metal and NaCl (`decomps1`): μ_Na = E(Na metal), μ_Cl = E(NaCl) − μ_Na, and μ_O from the Na3OCl bulk;
3. builds the interfaces of **both** terminations with Na(100) (`no_slabs = 2`).

It takes about 2 minutes on a CPU.

```bash
./run.sh
```

## Expected results (MACE-MP-0 small)

```text
Found 2 unique, symmetric surfaces for (1, 0, 0)
Surface formation energy of slab1_cut0: 2.75 eV/nm^2 = 0.44 J/m^2     # Na2O-terminated (Cl3Na11O4)
Surface formation energy of slab1_cut1: 1.47 eV/nm^2 = 0.23 J/m^2     # NaCl-terminated (Cl4Na10O3)
Surface formation energy of slab2_cut0: 1.53 eV/nm^2 = 0.25 J/m^2     # Na(100)
...
NOTE: slab 1 (Cl16Na40O12) is not stoichiometric; W_surf_1 is not a surface energy (use its surface formation energy).
W_ad (...) for the final optimised interface: 1.06 eV/nm^2 = 0.17 J/m^2     # NaCl-terminated | Na
...
W_ad (...) for the final optimised interface: 4.08 eV/nm^2 = 0.65 J/m^2     # Na2O-terminated | Na
```

In the Na-rich limit, the NaCl-terminated surface is the more stable free surface. The Na2O-terminated surface, however, binds Na metal about four times more strongly. Comparing these numbers is exactly what this kind of analysis is for: the interface that forms depends on both.

For non-stoichiometric slabs, the printed `Wsurf` uses the stoichiometric formula and is not a surface energy, as the NOTE says. Use the surface formation energy instead. W_ad uses only the slab and interface energies, so it is fine.

`analysis.ipynb` runs the tutorial with `--sfe_scan1 Na,Cl,-1.5,0.0`. It plots the surface formation energies of the two terminations against Δμ_Na, with μ_Cl compensating. The NaCl-terminated slab (slab1_cut1) is the most stable over the whole range from −1.5 to 0 eV. The interface energies are 0.31 J/m² for NaCl-terminated | Na and 0.03 J/m² for Na2O-terminated | Na.

## Notes

- **Choice of decomposition products:** Na2O + NaCl does **not** work, because Na3OCl = Na2O + NaCl exactly, so the chemical potentials are not determined. INTERFACER reports this and suggests an elemental phase. For other limits, give the chemical potentials directly, e.g. `chem_pots1 = {Na = -1.3, O = -4.9, Cl = -1.8}`.
- The structure files `NaCl.cif`, `na3ocl.cif` and `Na_Im-3m_mp-127.res` are relaxed with MACE at the start of the run (decomposition products with cell relaxation).

# Tutorial 6: grain boundaries and three-layer interfaces (EMT, about a minute)

Grain boundaries of one material and three-layer interfaces (slab1 | slab2 | slab3) with ASE's EMT potential. Both features work with every engine (`-prog mlip`, `vasp`, ...).

```bash
./run.sh                                  # the three command-line examples below
jupyter execute analysis.ipynb            # or open analysis.ipynb: a tilt-angle series, twist boundaries, Cu|Ni|Cu
```

## Command line

```bash
# Sigma5 (310)[001] symmetric tilt boundary
interfacer -i1 Cu.cif -m1 3 1 0 -cr1 3 --gb tilt -prog ase -ase_pot EMT --relax_cell_bulk
# Sigma5 (100) twist boundary (CSL integers m n = 2 1)
interfacer -i1 Cu.cif -m1 1 0 0 --gb twist --gb_mn 2 1 -prog ase -ase_pot EMT --relax_cell_bulk
# three-layer interface Cu(100) | Ni(100) | Cu(100)
interfacer -i1 Cu.cif -i2 Ni.cif -i3 Cu.cif -m1 1 0 0 -m2 1 0 0 -m3 1 0 0 -cr1 2 -cr2 2 -cr3 2 -sep2 2.0 \
           -msd 20 -prog ase -ase_pot EMT --relax_cell_bulk
```

## How the boundaries are built

* **Symmetric tilt** (`--gb tilt`): grain 2 is the mirror image of grain 1 through its top atomic plane (hkl). The cell along the normal holds two equivalent boundaries. Atoms of grain 2 on near-coincident sites (closer than 0.7 × the bulk nearest-neighbour distance) are removed, which is the usual CSL construction. A mirror plane of the crystal, e.g. cubic (100) or (110), gives the perfect crystal (γ = 0). fcc (111) gives the Σ3 coherent twin. `--gb_expansion` adds a rigid-body separation at the boundaries.
* **Twist** (`--gb twist --gb_mn m n`): the lattice vectors m**a** + n**b** of grain 1 and n**a** + m**b** of grain 2 coincide. Implemented for square planes, e.g. cubic (100), with Σ = (m² + n²)/2^k, and hexagonal planes, e.g. fcc (111), with Σ = (m² + mn + n²)/3^k.
* **Asymmetric boundaries** (two different planes of one crystal) are ordinary interfaces: give the same structure to `-i1` and `-i2` with different Miller indices.

The bicrystal is optimised with a variable cell length along the normal. The grain-boundary energy is γ_GB = (E_bicrystal − N E_bulk/atom) / (2A).

## Expected results (EMT)

| Boundary | Atoms | γ_GB (J/m²) |
|---|---|---|
| (100), (110) mirror planes | 20 | 0.000 |
| Σ5 (310)[001] tilt, 36.9° | 64 | 0.96 |
| Σ5 (210)[001] tilt, 53.1° | 40 | 1.05 |
| Σ5 (100) twist, 36.87° | 80 | 0.79 |
| Σ13 (100) twist, 22.62° | 208 | 0.70 |
| Σ7 (111) twist, 21.79° | 112 | 0.34 |

Cu | Ni (100): W_ad = 1.25 J/m². Cu | Ni | Cu: W_ad,total = 2.39 J/m², about twice the single interface, as expected for two interfaces.

The three-layer stack always gets at least 4 Å of vacuum on each side (the Cu | Ni pair uses `-vac_int`, def. 1 Å: two interfaces in the cell). A smaller vacuum would join slab 3 and slab 1 across the periodic boundary (an extra 3|1 interface).

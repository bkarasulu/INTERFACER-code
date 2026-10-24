# Quick start

## 1. A first run, in seconds

Create two bulk structures and build a Cu(111)|Ni(111) interface with the EMT potential:

```bash
python -c "from ase.build import bulk; import ase.io; \
ase.io.write('Cu.cif', bulk('Cu', cubic=True, a=3.61)); ase.io.write('Ni.cif', bulk('Ni', cubic=True, a=3.52))"

interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -cr1 3 -cr2 3 -prog ase -ase_pot EMT -msd 20
```

The run prints each step and ends with

```text
Wsurf_1 (Cu_111): 6.44 eV/nm^2 = 1.03 J/m^2
Wsurf_2 (Ni_111): 10.35 eV/nm^2 = 1.66 J/m^2
W_ad (work of adhesion, 2 interface(s) in the cell) for the final optimised interface: 16.15 eV/nm^2 = 2.59 J/m^2
```

The structures are in `OUTPUT/`; the final interface is `interface_cuts_0_0_opted.xyz` and `.cif`. The numbers are also in `interfaces.out` and `OUTPUT/results.json`. Open the structures with `ase gui OUTPUT/interface_cuts_0_0_opted.xyz`.

## 2. Add the convergence tests

```bash
interfacer -i1 Cu.cif -i2 Ni.cif -prog ase -ase_pot EMT -msd 20 \
    -tsur -mlist 100 110 111 \
    -tlay -lay_min 2 -lay_max 6 \
    -tsep -sep_init 1.5 -sep_final 3.0 -sep_step 0.25 \
    -thor -hor_steps 0.25 0.25
```

- `-tsur` picks the most stable of the listed surfaces for each material.
- `-tlay` converges the slab thickness.
- `-tsep` scans the separation between the slabs.
- `-thor` scans the lateral registry of the two slabs.

## 3. Use a machine-learning potential

```bash
pip install -e ".[mace]"
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -cr1 3 -cr2 3 -prog mlip --mlip mace --mlip_model small
```

## 4. Use DFT

```bash
export VASP_PP_PATH=/path/to/potentials
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -cr1 3 -cr2 3 -prog vasp -ec 450 -kp_gr "6 6 1" -mpi -np 64
```

Add `-dry` first to write all inputs without running VASP.

## 5. Keep the settings in an input file

```bash
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -prog ase -ase_pot EMT --save_input run.toml -surf_only
interfacer --input run.toml -tsep          # command-line options override the file
```

# Na3OCl | Na antiperovskite interface: VASP reference run

This folder holds a VASP run of the Na3OCl(100) | Na(100) interface made with the original `interfacer-new.py`. See `run_script` for the command line.

| Folder | Content |
|---|---|
| `Coll_Na/` | Input structures: `na3ocl.cif` and Na polymorphs from the Materials Project (`*.res`). |
| `bulk1`, `bulk2`, `slab1-aligned`, `slab2-aligned`, `interface-pre`, `interface` | VASP runs of the workflow steps (directory names of the original version). |
| `VASP-tmp/` | Layer, surface and separation tests (e.g. `ClNa3O_110_<n>-layer`, `interface-sep_2.50A`). |
| `INCAR`, `OUTPUT/`, `interfaces.out` | INCAR template and output structures of the original run. |

POTCAR, WAVECAR, CHG/CHGCAR, DOSCAR, EIGENVAL and vasprun.xml are no longer tracked; OUTCAR and CONTCAR are kept.

`interfaces.out` of this run contains no results (the run did not reach the interface step). Values in J/m² printed by versions before 1.0 used a wrong conversion factor (×38.96 too large); see `docs/known_issues.md`.

To repeat this system with the current version:

- **With MACE:** see `../02_Na3OCl-Na_MACE` and `../03_symmetric_slabs_SFE`.
- **With VASP:**

  ```bash
  interfacer -i1 Coll_Na/na3ocl.cif -m1 1 0 0 -i2 Coll_Na/Experimental/Na_Im-3m_mp-127.res -m2 1 0 0 -cr1 4 -cr2 6 \
             -kp_sp 0.5 -ec 400 -sig 0.10 -vac_int 2.0 -vac_slab 6.0 -sep 3.0 -Linit 8 -prog vasp -np 120 -mpi \
             -tsur -tsep -sep_init 2.0 -sep_final 3.5 -sep_step 0.5
  ```

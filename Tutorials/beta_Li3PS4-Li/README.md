# Tutorial 4: β-Li3PS4 | Li (VASP, with MACE pre-screening)

This tutorial covers the interface between the solid electrolyte β-Li3PS4 and the Li-metal anode.

| File | Content |
|---|---|
| `inputs/Li3PS4.vasp`, `inputs/Li.vasp` | Bulk structures (β-Li3PS4, Pnma, ICSD 180318; bcc Li, ICSD 44367), taken from the POSCARs of the reference VASP runs in `BKP/`. |
| `input_vasp.toml`, `run_vasp.sh` | The VASP run, with the settings of the original `run_script`. |
| `input_mace_prescreen.toml` | Pre-screening with MACE-MP-0. |
| `run_script`, `Orig-Input/` | The original command line. It used the `ICSD-str/` folder, a link to the original cluster that has been removed. |
| `BKP/` | VASP reference runs of the bulks and the aligned Li3PS4 slab (OUTCAR, CONTCAR, OSZICAR, …). WAVECAR, CHGCAR and POTCAR are no longer tracked. |
| `OUTPUT/`, `interfaces.out` | Structures from the original run. That run used `-quick`, i.e. ASE/LJ, so its energies are only a functionality test. |

## 1. Pre-screen with MACE (minutes on a CPU)

```bash
pip install mace-torch
interfacer --input input_mace_prescreen.toml
```

This screens the (100), (010) and (001) surfaces, picks the most stable one for each material, and scans the separation.

## 2. Run VASP

```bash
export VASP_PP_PATH=/path/to/potentials
./run_vasp.sh -dry     # write and check all inputs first (INCAR/KPOINTS/POSCAR per step)
./run_vasp.sh          # mpirun -n 128 vasp_std; adapt nprocs/mpiargs in input_vasp.toml
```

The VASP settings are ENCUT 400 eV, KSPACING 0.32 Å⁻¹ (bulk), Gaussian smearing 0.05 eV, and dipole corrections for the slabs. The run includes the surface (`tsur`), layer (`tlay`), separation (`tsep`) and lateral (`thor`) tests, as in the original run. Expect many single-point calculations; switch tests off once they are converged.

Each step runs in its own directory (`bulk1`, `slab1_cut0_pre`, `slab1_cut0_aligned`, `interface_cuts_0_0`, `VASP-tmp/…`). Restarting the same command continues unfinished VASP runs and reuses finished ones.

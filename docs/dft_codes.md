# DFT codes: VASP, CASTEP, GPAW and ONETEP

## VASP (`-prog vasp`)

Each calculation runs in its own directory, named after the calculation:

- `bulk1`, `bulk2`
- `slab1_cut0_pre`
- `slab1_cut0_aligned`
- `interface_cuts_0_0`
- `VASP-tmp/…` for the convergence tests

**Restarts:** when a directory already contains an OUTCAR, a converged run is reused. An unconverged run is continued from its CONTCAR with the same INCAR and KPOINTS, and the old files are kept in `RUN<n>/`. A crashed or interrupted workflow can therefore simply be started again.

**Settings:**

- `-ec` (ENCUT), `-xc`, `-sig` (SIGMA, ISMEAR=0).
- `-kp_sp` (KSPACING) for bulk runs. `-kp_gr "n1 n2 n3"` sets a Gamma-centred grid for the slabs and interfaces; `-gamma` uses Γ only with `vasp_gam`.
- Dipole corrections (LDIPOL/IDIPOL=3) for slabs, and for interfaces with more than 4 Å of vacuum.
- ISIF=3 for bulk and interface relaxations; ISIF=2 (fixed cell) for slabs.
- `-hubU "Fe d 2.7, O p 1.0"`: DFT+U (LDAUTYPE=2, with LMAXMIX and LASPH).
- `-mgm Fe 5 O 0.6`: initial magnetic moments, switches on ISPIN=2.
- `-pass`: pseudo-hydrogen passivation of the outer slab surfaces, based on dangling bonds.
- `--fix_layers N`: selective dynamics for N layers, either the bottom layers or, with `--fix_mode middle`, the central ones.
- Cells are made right-handed and shifted to non-negative z before they are written (VASP fails with left-handed cells).

**MPI command:**

- Default: `srun -n <np> <mpiargs> <vexe>`.
- With `-mpi`: `mpirun -n <np> <mpiargs> <vexe>`.
- Alternatively, set `VASP_COMMAND`.

**POTCARs** are taken from `$VASP_PP_PATH/potpaw_PBE` with ASE's "recommended" setups.

**`-dry`** writes INCAR, KPOINTS and POSCAR (and POTCAR, if available) for every step without running VASP. Energy-based steps are skipped.

**`--overwrite`** moves existing run directories to `<dir>_old<n>` and starts afresh.

!!! note "Passivation needs thick enough slabs"
    `-pass` finds the under-coordinated atoms by comparing the neighbours of each atom in the outer three layers with the most common neighbour environment of its element in the slab. In very thin slabs, roughly fewer than six atomic layers, the surface atoms can be as numerous as the bulk-like ones. The most common environment is then ambiguous, and no or too few hydrogens may be added. Use slabs with at least six layers for passivation, and check the passivated POSCAR (it is written with `-dry`).

## CASTEP (`-prog castep`)

- Runs happen in `CASTEP-tmp/` through ASE's CASTEP calculator.
- `-cparam input.param` provides the parameters; the cutoff and `kpoints_mp_spacing` are read from `-ccell input.cell`.
- `-cpp` chooses the pseudopotentials (OTF by default).
- `-hubU` is passed as a HUBBARD_U block.
- `--fix_layers` is passed as ionic constraints.
- Set `CASTEP_COMMAND` and `CASTEP_PP_PATH`.
- A calculation is not repeated when `CASTEP-tmp/<name>.castep` already holds a finished run; use `--overwrite` to recompute it.

!!! warning
    The CASTEP and VASP drivers keep the behaviour of the original code. Their input generation is tested (VASP with `-dry`), but no CASTEP or VASP binary was available for the tests of this version. Check the first run of a new system carefully.

## GPAW (`-prog gpaw`)

GPAW runs through ASE in plane-wave mode, with INTERFACER's shared ASE optimiser (as for the ML potentials). The output of each calculation goes to `GPAW-tmp/<name>.txt`.

- `-ec`: plane-wave cutoff (`PW(encut)`); `-xc`: functional; `-sig`: Fermi–Dirac smearing.
- k-points: density 1/`-kp_sp` (Γ-centred), or the grid `-kp_gr` for slabs and interfaces.
- `--calc_kwargs "{'spinpol': True, 'parallel': {'domain': 4}}"` passes any further GPAW keyword.
- Run in parallel with `mpiexec -n 32 interfacer ... -prog gpaw` (or `gpaw -P 32 python -m interfacer.cli ...`).

Install GPAW with `conda install -c conda-forge gpaw`.

## ONETEP (`-prog onetep`)

ONETEP, a linear-scaling DFT code for large systems such as big interface supercells, runs through ASE's `Onetep` calculator with the shared ASE optimiser. Each calculation runs in `ONETEP-tmp/<name>/`.

- **Command:** `$ONETEP_COMMAND`, for example `mpirun -n 64 onetep.exe`. If it is not set, the command is built from `-np`, `-mpi` and `-mpiargs` with the executable from `-vexe` or `$ONETEP_EXE` (default `onetep.exe`).
- **Pseudopotentials:** read from `$ONETEP_PP_PATH`, or from `--calc_kwargs "{'pseudo_path': ...}"`.
- **Settings:** `-ec` sets `cutoff_energy` and `-xc` sets `xc_functional`. By default the number of NGWFs is chosen automatically (`ngwfs_count = -1`) with a radius of 9 bohr.
- **Further keywords:** `--calc_kwargs "{'keywords': {'maxit_ngwf_cg': 50}, 'ngwfs_radius': 8.0}"`. ONETEP calculations are Γ-point only.

!!! warning
    No GPAW or ONETEP binary was available for testing this version. The tests only check the calculator settings that INTERFACER generates, so check the first run of a new system carefully.

## Typical HPC job

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=128 --time=48:00:00
module load vasp
export VASP_PP_PATH=$HOME/vasp/potentials
interfacer --input run.toml -prog vasp -np 128 -mpiargs="--cpu-bind=cores"
```

A good strategy is to pre-screen with an ML potential: `-prog mlip` with `-tsur`, `-tsep` and `-thor`. Then run only the selected surfaces and registries with DFT.

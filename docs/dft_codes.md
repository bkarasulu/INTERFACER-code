# DFT codes: VASP and CASTEP

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
- `--fix_layers N`: selective dynamics for the bottom N layers.

**MPI command:**

- Default: `srun -n <np> <mpiargs> <vexe>`.
- With `-mpi`: `mpirun -n <np> <mpiargs> <vexe>`.
- Alternatively, set `VASP_COMMAND`.

**POTCARs** are taken from `$VASP_PP_PATH/potpaw_PBE` with ASE's "recommended" setups.

**`-dry`** writes INCAR, KPOINTS and POSCAR (and POTCAR, if available) for every step without running VASP. Energy-based steps are skipped.

## CASTEP (`-prog castep`)

- Runs happen in `CASTEP-tmp/` through ASE's CASTEP calculator.
- `-cparam input.param` provides the parameters; the cutoff and `kpoints_mp_spacing` are read from `-ccell input.cell`.
- `-cpp` chooses the pseudopotentials (OTF by default).
- `-hubU` is passed as a HUBBARD_U block.
- `--fix_layers` is passed as ionic constraints.
- Set `CASTEP_COMMAND` and `CASTEP_PP_PATH`.

!!! warning
    The CASTEP and VASP drivers keep the behaviour of the original code. Their input generation is tested (VASP with `-dry`), but no CASTEP or VASP binary was available for the tests of this version. Check the first run of a new system carefully.

## Typical HPC job

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=128 --time=48:00:00
module load vasp
export VASP_PP_PATH=$HOME/vasp/potentials
interfacer --input run.toml -prog vasp -np 128 -mpiargs="--cpu-bind=cores"
```

A good strategy is to pre-screen with an ML potential: `-prog mlip` with `-tsur`, `-tsep` and `-thor`. Then run only the selected surfaces and registries with DFT.

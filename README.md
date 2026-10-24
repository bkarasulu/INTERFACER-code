# INTERFACER

INTERFACER builds solid–solid interfaces from two bulk crystal structures and computes their surface and interface energetics. It works with VASP, CASTEP or ASE calculators.

It runs the following workflow:

1. Optimise both bulk structures.
2. Optionally screen surfaces (`-tsur`) and converge the slab thickness (`-tlay`).
3. Cut slabs, with optional symmetric terminations (`-sym_slabs1/2`), and compute their surface formation energies.
4. Find a commensurate supercell of the two slabs and stack them into an interface.
5. Optionally scan the interlayer separation (`-tsep`) and the lateral registry (`-thor`).
6. Optimise the interface and report the work of adhesion.

## Installation

```bash
git clone https://github.com/bkarasulu/INTERFACER-code.git
cd INTERFACER-code
pip install -e ".[test]"      # installs the `interfacer` command and the Python package
python -m pytest              # optional: run the test suite (~10 s, no DFT code needed)
```

Machine-learning potentials are optional extras. Install the ones you need, e.g. `pip install -e ".[mace]"`. The available extras are `mace`, `chgnet`, `matgl`, `sevennet`, `orb`, `fairchem`, `nequip` and `allegro`.

To run DFT, set up the code through ASE:

- **VASP:** set `VASP_PP_PATH` and either pass the MPI options (`-np`, `-mpi`, `-mpiargs`, `-vexe`) or set `VASP_COMMAND`.
- **CASTEP:** set `CASTEP_COMMAND` and `CASTEP_PP_PATH`.

## Quick start

```bash
# a quick functionality test with a simple potential (seconds)
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -prog ase -ase_pot EMT

# with a machine-learning potential (MACE-MP-0 foundation model; GPU used if available)
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -prog mlip --mlip mace --mlip_model medium-mpa-0 -tsep -thor

# the same with VASP, including separation and lateral-shift scans
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -prog vasp -kp_gr "4 4 1" -tsep -thor -mpi -np 64
```

`python interfacer-new.py ...` accepts the same options as `interfacer`. Run `interfacer -h` for all options.

Results are written to `interfaces.out` and `OUTPUT/`. The previous `OUTPUT/` is kept as `OUTPUT_prev/`.

## Documentation

- `docs/ROADMAP.md`: code overview and the refactoring roadmap.
- `docs/mlip.md`: machine-learning potentials (MACE, CHGNet, MatGL, SevenNet, ORB, fairchem/UMA, NequIP/Allegro, custom).
- `docs/package_layout.md`: package layout, Python API and migration from the old scripts.
- `docs/known_issues.md`: fixed bugs and open issues.
- `Tutorials/`: example inputs (β-Li3PS4|Li and Na3OCl|Na with VASP).

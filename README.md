# INTERFACER
<img width="1254" height="845" alt="INTERFACER-logo-v2" src="https://github.com/user-attachments/assets/a6fe7fb8-ef19-4a77-80ff-38a0bd5e951d" />

INTERFACER builds solid–solid interfaces from two bulk crystal structures and computes their surface and interface energetics. It works with VASP, CASTEP, GPAW and ONETEP, with machine-learning interatomic potentials such as MACE, and with any ASE calculator. It also builds grain boundaries and three-layer interfaces.

It runs the following workflow:

1. Optimise both bulk structures.
2. Optionally screen surfaces (`-tsur`) and converge the slab thickness (`-tlay`).
3. Cut slabs, with optional symmetric terminations (`-sym_slabs1/2`), and compute their surface formation energies.
4. Find a commensurate supercell of the two slabs and stack them into an interface.
5. Optionally scan the lateral registry (`-thor`) and the interlayer separation (`-tsep`), or optimise the slab placement (`-topt`).
6. Optimise the interface and report the work of adhesion and the interface energy.

With `-i3` a third material gives slab1 | slab2 | slab3 stacks. With `--gb tilt|twist`, symmetric tilt and twist grain boundaries of one material are built instead.

## Installation

```bash
git clone https://github.com/bkarasulu/INTERFACER-code.git
cd INTERFACER-code
pip install -e ".[test]"      # installs the `interfacer` command and the Python package
python -m pytest              # optional: run the test suite (~30 s, no DFT code needed)
```

Machine-learning potentials are optional extras. Install the ones you need, e.g. `pip install -e ".[mace]"`. The available extras are `mace`, `chgnet`, `matgl`, `sevennet`, `orb`, `fairchem`, `nequip` and `allegro`.

To run DFT, set up the code through ASE:

- **VASP:** set `VASP_PP_PATH` and either pass the MPI options (`-np`, `-mpi`, `-mpiargs`, `-vexe`) or set `VASP_COMMAND`.
- **CASTEP:** set `CASTEP_COMMAND` and `CASTEP_PP_PATH`.
- **GPAW:** install GPAW, for example with `conda install -c conda-forge gpaw`.
- **ONETEP:** set `ONETEP_COMMAND` (or use `-np`/`-mpi`) and `ONETEP_PP_PATH`.

## Quick start

```bash
# a quick functionality test with a simple potential (seconds)
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -prog ase -ase_pot EMT

# with a machine-learning potential (MACE-MP-0 foundation model; GPU used if available)
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -prog mlip --mlip mace --mlip_model medium-mpa-0 -tsep -thor

# a Sigma5 (310) symmetric tilt grain boundary of Cu
interfacer -i1 Cu.cif -m1 3 1 0 -cr1 3 --gb tilt -prog ase -ase_pot EMT

# the same interface with VASP, including separation and lateral-shift scans
interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -prog vasp -kp_gr "4 4 1" -tsep -thor -mpi -np 64
```

`python interfacer.py ...` accepts the same options as `interfacer`. Run `interfacer -h` for all options.

Options can also go in an input file: `interfacer --input run.toml` (TOML, YAML or JSON). Command-line options override the file, and `--save_input` writes the options of a run to a file.

Results are written to `interfaces.out` and `OUTPUT/`. The previous `OUTPUT/` is kept as `OUTPUT_prev/`.

## Tutorials

The tutorials in [`Tutorials/`](Tutorials/README.md) are:

1. Cu|Ni with EMT (seconds).
2. Na3OCl|Na with MACE (~1 min).
3. Symmetric slabs and surface formation energies.
4. β-Li3PS4|Li with VASP.
5. The Python API (script and notebook).
6. Grain boundaries and three-layer interfaces with EMT.

Tutorials 1, 2, 3 and 6 include an `analysis.ipynb` notebook.

## Documentation

The full documentation is built with `pip install -e ".[docs]" && mkdocs serve`. It covers installation, the workflow and output files, the theory, a command-line reference, input files, DFT codes, ML potentials, and the Python API. The source pages are in `docs/`:

- `docs/index.md`, `docs/quickstart.md`, `docs/workflow.md`, `docs/theory.md`, `docs/cli.md`, `docs/input_files.md`, `docs/dft_codes.md`, `docs/python_api.md`.
- `docs/ROADMAP.md`: code overview and the refactoring roadmap.
- `docs/mlip.md`: machine-learning potentials (MACE, CHGNet, MatGL, SevenNet, ORB, fairchem/UMA, NequIP/Allegro, custom).
- `docs/package_layout.md`: package layout, Python API and migration from the old scripts.
- `docs/known_issues.md`: fixed bugs and open issues.

## Citing INTERFACER

If you use INTERFACER, please cite it; see [`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite this repository"). A paper describing the code is in preparation for the Journal of Open Source Software (`paper/`).

## Licence

INTERFACER is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [`LICENSE`](LICENSE).

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

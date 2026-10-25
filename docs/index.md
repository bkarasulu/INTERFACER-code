# INTERFACER

INTERFACER builds solid–solid interfaces between two crystalline materials and computes their energetics: surface energies, surface formation energies and the work of adhesion. It can run all calculations with:

- **VASP** or **CASTEP**, plane-wave DFT through ASE;
- **machine-learning potentials**: MACE, CHGNet, MatGL/M3GNet, SevenNet, ORB, fairchem/UMA, NequIP/Allegro, or any ASE calculator;
- **simple ASE potentials** (LJ, Morse, EMT), for quick tests.

## What it does

```text
bulk 1, bulk 2 ──► optimise bulks ──► [surface screening -tsur] ──► [layer convergence -tlay]
                                                                     │
          slabs (make_slab / symmetric slabs / from file) ◄──────────┘
                  │
                  ▼
     pre-optimise slabs, surface formation energies ──► keep the most stable (-no_slabs)
                  │
                  ▼  for each pair of slabs
     commensurate supercell (slab_aligner) ──► optimise aligned slabs (W_surf)
                  │
                  ▼
     stack into an interface ──► [separation scan -tsep] ──► [lateral scan -thor]
                  │
                  ▼
     optimise the interface ──► work of adhesion W_ad, structures in OUTPUT/
```

## Where to start

- [Installation](installation.md) and the [Quick start](quickstart.md).
- [Tutorials](tutorials.md): Cu|Ni with EMT; Na3OCl|Na with MACE; symmetric slabs and surface formation energies; β-Li3PS4|Li with VASP; the Python API.
- [Workflow and output files](workflow.md) and [Theory and definitions](theory.md): what is computed, and how.
- [Command-line reference](cli.md) and [Input files](input_files.md).

INTERFACER was originally written as the `interfacer-new.py` script (now `interfacer.py`) with the `tools/` modules. Both still work; see [Package layout and migration](package_layout.md).

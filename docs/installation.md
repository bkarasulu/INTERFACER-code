# Installation

INTERFACER needs Python ≥ 3.10. Its core dependencies are NumPy, SciPy, Matplotlib, ASE (≥ 3.23), spglib and pymatgen.

```bash
git clone https://github.com/bkarasulu/INTERFACER-code.git
cd INTERFACER-code
python -m venv .venv && source .venv/bin/activate     # or a conda environment
pip install -e ".[test]"
python -m pytest                                       # ~15 s; no DFT code needed
interfacer --version
```

The `-e` (editable) install keeps the package linked to the source tree. The `interfacer` command is installed, and `python interfacer.py` works too, even without installing.

!!! tip "conda"
    With conda, install the scientific stack from conda-forge first and the package afterwards:
    `conda create -n interfacer -c conda-forge python=3.12 ase pymatgen spglib scipy matplotlib && conda activate interfacer && pip install -e .`

## Optional extras

| Extra | Installs | For |
|---|---|---|
| `mace` | mace-torch | MACE-MP / MACE-OFF foundation models, your own MACE models |
| `chgnet` | chgnet | CHGNet |
| `matgl` | matgl | M3GNet, TensorNet, … |
| `sevennet` | sevenn | SevenNet |
| `orb` | orb-models | ORB |
| `fairchem` | fairchem-core | fairchem / UMA (also needs `huggingface-cli login`) |
| `nequip`, `allegro` | nequip, nequip-allegro | NequIP / Allegro models |
| `yaml` | pyyaml | YAML input files (TOML and JSON work without it) |
| `docs` | mkdocs-material, mkdocstrings | building this documentation (`mkdocs serve`) |
| `test`, `lint` | pytest, pyyaml / ruff | development |

Example: `pip install -e ".[mace,yaml]"`.

!!! note "PyTorch and GPUs"
    The ML packages pull in PyTorch. For GPU runs, install the PyTorch build that matches your CUDA version first; see [pytorch.org](https://pytorch.org/get-started/locally/). `--device auto` (the default) uses the GPU when PyTorch sees one.

## DFT codes

INTERFACER drives VASP and CASTEP through ASE's calculators, so they have to be set up as ASE expects:

**VASP**

- `VASP_PP_PATH` must point to the directory that contains `potpaw_PBE/`.
- Give the MPI command with `-np/-mpi/-mpiargs/-vexe`, which builds e.g. `srun -n 32 vasp_std`, or set `VASP_COMMAND`.
- Gamma-point runs (`-gamma`) automatically use `vasp_gam`.

**CASTEP**

- Set `CASTEP_COMMAND` (e.g. `mpirun -n 32 castep.mpi`) and `CASTEP_PP_PATH`.
- Parameters can come from `-cparam input.param` and `-ccell input.cell`.

See [DFT codes](dft_codes.md) for details.

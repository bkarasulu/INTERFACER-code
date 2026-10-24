# Machine-learning potentials

With `-prog mlip`, INTERFACER runs the whole workflow with a machine-learning interatomic potential (MLIP) through its ASE calculator. The workflow covers bulk and slab relaxations, the separation and lateral scans, and the interface optimisation. Uses include:

- screening many surface and interface combinations quickly before DFT;
- pre-relaxing interfaces for DFT;
- running fine-tuned potentials on systems too large for DFT.

The model is loaded once per run. Geometry optimisations use the shared ASE optimiser (`--optimiser`, `-ase_fmax`, `-ase_gsteps`). The bulk cells are relaxed by default; use `--no_relax_cell_bulk` to keep them fixed.

## Supported potentials

| `--mlip` | Install | `--mlip_model` (default) | Notes |
|---|---|---|---|
| `mace` (alias `mace-mp`) | `pip install mace-torch` | `small`, `medium`, `large`, `medium-mpa-0`, `medium-omat-0`, … or a path/URL to a `.model` file (`medium`) | MACE-MP foundation models or your own (fine-tuned) MACE model; `default_dtype` float64 |
| `mace-off` | `pip install mace-torch` | `small`, `medium`, `large` or a path (`medium`) | organic molecules and molecular crystals |
| `chgnet` | `pip install chgnet` | version, e.g. `0.3.0`, or a path (package default) | |
| `matgl` (alias `m3gnet`) | `pip install matgl` | a pretrained PES model name or path (`M3GNet-MP-2021.2.8-PES`) | M3GNet, TensorNet, … |
| `sevennet` (alias `7net`) | `pip install sevenn` | `7net-0`, `7net-mf-ompa`, … or a checkpoint (`7net-0`) | multi-fidelity models: `--mlip_kwargs "{'modal': 'mpa'}"` |
| `orb` | `pip install orb-models` | `orb-v3-conservative-inf-omat`, `orb-v2`, … | |
| `fairchem` (alias `uma`) | `pip install fairchem-core` | `uma-s-1`, … | needs `huggingface-cli login` and access to the UMA models; task via `--mlip_kwargs "{'task_name': 'omat'}"` (default `omat`) |
| `nequip` | `pip install nequip` | path of a compiled (nequip ≥ 0.7) or deployed (< 0.7) model (required) | |
| `allegro` | `pip install nequip-allegro` | as `nequip` | |
| `custom:pkg.module:function` | your package | passed as `model=` | `function(**mlip_kwargs)` must return an ASE calculator |

The `-ase_pot LJ/Morse/EMT` potentials of `-prog ase` are still available for quick functionality tests.

## Options

| Option | Meaning |
|---|---|
| `-prog mlip` | Use a machine-learning potential. |
| `--mlip NAME` | Backend (default `mace`). |
| `--mlip_model MODEL` | Model name or path (default: the backend default). |
| `--device {auto,cpu,cuda,mps}` | Device; `auto` uses the GPU when PyTorch sees one. |
| `--mlip_kwargs DICT` | Extra calculator arguments as a Python dict, e.g. `"{'default_dtype': 'float32'}"`. |
| `--relax_cell_bulk` / `--no_relax_cell_bulk` | Relax the bulk cells (default on for `-prog mlip`). |
| `--optimiser`, `-ase_fmax`, `-ase_gsteps` | Optimiser, force criterion (eV/Å) and maximum number of steps. |
| `--fix_layers N` | Keep the lowest N layers of each slab fixed. |

## Examples

```bash
# Na3OCl(100) | Na(100) with MACE-MP-0 (the antiperovskite tutorial system); ~1 min on a CPU with the small model
interfacer -i1 na3ocl.cif -m1 1 0 0 -i2 Na_Im-3m_mp-127.res -m2 1 0 0 -cr1 4 -cr2 6 -Linit 8 \
           -vac_int 2.0 -sep 3.0 -prog mlip --mlip mace --mlip_model small -tsep -sep_init 2.0 -sep_final 3.5 -sep_step 0.5

# a fine-tuned MACE model on the GPU, with surface screening
interfacer -i1 A.cif -i2 B.cif -prog mlip --mlip mace --mlip_model ./my_finetuned.model --device cuda -tsur -mlist 100 110 111

# any other ASE calculator
interfacer -i1 A.cif -i2 B.cif -prog mlip --mlip custom:mypackage.calcs:make_calculator --mlip_kwargs "{'cutoff': 6.0}"
```

Results from runs in this environment (CPU, MACE-MP-0 `small`, default settings):

| System | W_surf (J/m²) | W_ad (J/m²) |
|---|---|---|
| Cu(111) \| Ni(111) | 1.31 / 2.11 | 3.18 |
| Na3OCl(100) \| Na(100) | 0.33 / 0.25 | 0.42 |

## Python API

```python
from interfacer.calculators import CalcSettings, get_engine, mlip_calculator

calc = mlip_calculator("mace", model="medium-mpa-0", device="cuda")  # a plain ASE calculator

engine = get_engine("mlip", CalcSettings(mlip="mace", mlip_model="small", fmax=0.02, relax_cell_bulk=True))
E, relaxed = engine.relax(atoms, "bulk", relax_cell=True)
```

## Caveats

- Universal potentials are trained mostly on bulk data (Materials Project, OMat, …). Surface and interface energies can be systematically off, typically underestimated. Validate key results with DFT (`-prog vasp`/`castep`) or use a fine-tuned model.
- Each potential reproduces the DFT functional of its training data (mostly PBE/PBE+U). Compare energies only within one potential.
- Non-stoichiometric slabs need chemical potentials (`-cp1/-cp2`) or decomposition products (`-decomps1/-decomps2`), as with DFT.

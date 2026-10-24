# Tutorial 5: INTERFACER from Python

`python_api.py` (the same code is in the notebook `python_api.ipynb`) shows how to:

1. create an engine from any ASE calculator (here EMT), an ML potential (`USE_MACE = True`), or VASP/CASTEP;
2. use the building blocks: relax bulks, cut slabs, align them, compute a surface energy, scan the separation and build an interface;
3. run the full workflow with a `WorkflowConfig` and read the results.

```bash
python python_api.py      # a few seconds
```

The expected output ends with:

```text
lattice constants: Cu 3.589 A, Ni 3.486 A
misfit: 2.91 % before, 2.91 % after alignment; 12 + 12 atoms
W_surf(Cu 111) = 6.52 eV/nm^2 = 1.04 J/m^2
best separation: 2.50 A
workflow result: W_surf = 1.04 / 1.70 J/m^2, W_ad = 2.64 J/m^2 at a separation of 2.50 A
```

See the [Python API documentation](../../docs/python_api.md) for all functions.

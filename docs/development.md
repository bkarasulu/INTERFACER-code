# Testing and contributing

```bash
pip install -e ".[test,lint,docs]"
python -m pytest              # all tests (~15 s; the MACE test is skipped without mace-torch)
python -m pytest -k mlip      # a subset
ruff check .                  # lint
python docs/gen_cli.py        # regenerate docs/cli.md after changing command-line options
mkdocs serve                  # preview the documentation at http://127.0.0.1:8000
```

## Test suite

| File | What it tests |
|---|---|
| `tests/test_builder.py` | Slab cutting, orientation, symmetric slabs, uniqueness checks, alignment, `tools/` shims |
| `tests/test_analysis.py` | Surface formation energies (stoichiometric and with chemical potentials), chemical potentials, misfit |
| `tests/test_external.py` | VASP input generation (dry runs with placeholder POTCARs), Hubbard U, restarts, ASE driver |
| `tests/test_engines.py` | Engines, shared optimiser, cell relaxation, fixed layers, selective dynamics |
| `tests/test_mlip.py` | Every ML backend against fake packages; a real MACE-MP-0 Cu\|Ni run when available |
| `tests/test_workflow.py` | The Python workflow API |
| `tests/test_interfacer.py` | End-to-end command-line runs (LJ, EMT, VASP dry run), separation definition, unit conversion |
| `tests/test_input_files.py` | TOML/YAML/JSON input files, `--save_input`, and that `docs/cli.md` is up to date |
| `tests/test_tutorials.py` | The quick tutorials and the Python API example |

The tests need no DFT code: VASP inputs are checked with `-dry` and placeholder POTCAR files.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- lint with ruff;
- the test suite on Python 3.10–3.13;
- the strict documentation build;
- an extra job that installs CPU PyTorch and mace-torch and runs the MACE tests and tutorial.

## Conventions

- Keep the original function names and arguments: they are part of the API used by existing scripts.
- New calculators: implement `Engine.run`, or wrap an ASE calculator in `AseEngine`. New ML backends go in `calculators/mlip.py`, each with a mocked test in `tests/test_mlip.py`.
- Every bug fix gets a regression test.

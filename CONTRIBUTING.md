# Contributing to INTERFACER

Thank you for helping to improve INTERFACER! Contributions of any size are welcome: bug reports, tests, examples, documentation and code.

## Working with the repository

Please always work from a clone of the GitHub repository rather than from private copies of the code. This keeps everyone's changes under version control.

```bash
git clone https://github.com/bkarasulu/INTERFACER-code.git
cd INTERFACER-code
pip install -e ".[test,lint,docs]"
git checkout -b my-feature        # new work goes on a branch
```

- `main` holds the released code.
- `devel` collects work before it is reviewed and merged into `main`.
- Open pull requests against `devel` unless agreed otherwise.

## Before opening a pull request

```bash
python -m pytest            # all tests must pass (~30 s; no DFT code needed)
ruff check .                # lint
python docs/gen_cli.py      # if you changed command-line options
mkdocs build --strict       # if you changed the documentation
```

- Add a test for every bug fix and every new feature (see `tests/` and `docs/development.md`).
- Keep the existing command-line options and function names working: other people's scripts use them.
- Describe the change, and how you tested it, in the pull request.

## Reporting bugs and requesting features

Use the GitHub issue templates. For bugs, please include:

- the full command line or input file;
- the input structures (or how to obtain them);
- the INTERFACER version (`interfacer --version`) and the code used (VASP, CASTEP, MLIP, ...);
- the error message or the unexpected output.

A small input that reproduces the problem with `-prog ase` (EMT/LJ) or an ML potential is the most helpful.

## Testing new and old features (checklist for testers)

- [ ] Redo a system you studied with the old version and compare the results. J/m² values printed before v1.0 were ~39× too large, so compare eV/nm² values.
- [ ] Build 1–2 new interfaces from scratch with different options (`-tsur`, `-tlay`, `-tsep`, `-thor`, `-sym_slabs1`, `-surf1`, `--input`).
- [ ] Check the VASP/CASTEP inputs with `-dry` before submitting real runs.
- [ ] Run the tutorials and notebooks in `Tutorials/`.
- [ ] Report anything unexpected as an issue.

## Licence

INTERFACER is released under the GNU General Public License v3.0 or later (see `LICENSE`). By contributing, you agree that your contributions are released under the same licence.

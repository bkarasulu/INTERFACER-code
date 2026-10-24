# Input files

Any option can be stored in an input file and passed with `--input`:

```bash
interfacer --input run.toml
interfacer --input run.toml -tsep -sep_step 0.1     # command-line options override the file
```

- **Formats:** TOML (recommended, built into Python ≥ 3.11), YAML (needs PyYAML) or JSON, chosen by the file extension.
- **Keys:** the long option name without dashes, either the destination name (`convSep`) or the option name (`test_sep`), or the short name (`tsep`). See the [command-line reference](cli.md).
- **Sections** such as `[structures]` or `[calculation]` are optional and only for readability.
- **Values:** lists for options with several values (`miller1 = [1, 1, 1]` or `"1 1 1"`), `true`/`false` for switches, and tables or dicts for chemical potentials and `mlip_kwargs`.
- **Errors:** unknown keys and wrong types are reported before anything runs.

## Example (TOML)

```toml
# Na3OCl(100) | Na(100) with MACE-MP-0
[structures]
infile1 = "na3ocl.cif"
infile2 = "Na_Im-3m_mp-127.res"
miller1 = [1, 0, 0]
miller2 = [1, 0, 0]
cr1 = 4
cr2 = 6
initLength = 8

[interface]
vac_slab = 6.0
vac_int = 2.0
sep = 3.0

[tests]
tsep = true
sep_init = 2.0
sep_final = 3.5
sep_step = 0.5

[calculation]
prog = "mlip"
mlip = "mace"
mlip_model = "small"
device = "auto"
ase_fmax = 0.05
ase_gsteps = 300
```

The same in YAML:

```yaml
infile1: na3ocl.cif
infile2: Na_Im-3m_mp-127.res
miller1: [1, 0, 0]
prog: mlip
mlip: mace
tsep: true
chem_pots1: {Na: -1.3, O: -4.9, Cl: -1.8}
```

## Saving the options of a run

`--save_input FILE` writes the options of the current run (input file plus command line) to a TOML, YAML or JSON file. The file includes the required options and every option that differs from its default.

```bash
interfacer -i1 Cu.cif -i2 Ni.cif -prog ase -ase_pot EMT -tsep --save_input run.toml
```

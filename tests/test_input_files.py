import json

import pytest
from ase.build import bulk

from interfacer.cli import apply_input_file, build_parser, load_input_file, main, save_input_file


TOML = '''
# sections are optional and only for readability
[structures]
infile1 = "Cu.cif"
infile2 = "Ni.cif"
miller1 = [1, 1, 1]
m2 = "1 1 1"
cr1 = 3
cr2 = 3

[calculation]
prog = "ASE"
ase_pot = "EMT"
ase_gsteps = 100
optimiser = "fire"
relax_cell_bulk = true
hubU = "Fe d 2.7, O p 1.0"

[tests]
tsep = true
mlist = ["100", [1, 1, 1]]
hor_steps = [0.5, 0.5]
chem_pots1 = {Cu = -3.5}
'''


def _parse(tmp_path, text, argv=(), name="in.toml"):
    f = tmp_path / name
    f.write_text(text)
    p = build_parser()
    apply_input_file(p, str(f))
    return p.parse_args(list(argv))


def test_toml_values_are_converted(tmp_path):
    a = _parse(tmp_path, TOML)
    assert a.infile1 == "Cu.cif" and a.miller1 == [1, 1, 1] and a.miller2 == [1, 1, 1]
    assert a.prog == "ase" and a.optimiser == "FIRE" and a.relax_cell_bulk is True
    assert a.convSep is True and a.miller_list == [(1, 0, 0), (1, 1, 1)] and a.hor_steps == [0.5, 0.5]
    assert a.hubU == ["Fe", "d", "2.7,", "O", "p", "1.0"]
    assert eval(a.chem_pots1) == {"Cu": -3.5}


def test_command_line_overrides_file(tmp_path):
    a = _parse(tmp_path, TOML, ["-cr1", "5", "-prog", "vasp", "--no_relax_cell_bulk"])
    assert a.creps1 == 5 and a.prog == "vasp" and a.relax_cell_bulk is False


def test_json_and_yaml(tmp_path):
    a = _parse(tmp_path, json.dumps({"infile1": "a", "infile2": "b", "prog": "mlip", "mlip": "mace",
                                     "mlip_kwargs": {"default_dtype": "float32"}}), name="in.json")
    assert a.prog == "mlip" and eval(a.mlip_kwargs) == {"default_dtype": "float32"}
    yaml = pytest.importorskip("yaml")
    a = _parse(tmp_path, yaml.safe_dump({"infile1": "a", "infile2": "b", "prog": "ase", "tsur": True}), name="in.yaml")
    assert a.checkSurfs is True


def test_errors(tmp_path):
    for text, msg in (('prog = "ase"\nbogus = 1\n', "bogus"), ('prog = "ase"\ntsep = 3\n', "true or false")):
        f = tmp_path / "bad.toml"
        f.write_text(text)
        with pytest.raises(ValueError, match=msg):
            apply_input_file(build_parser(), str(f))


def test_required_options_from_file_only(tmp_path):
    a = _parse(tmp_path, 'infile1 = "a"\ninfile2 = "b"\nprog = "ase"\n')
    assert (a.infile1, a.infile2, a.prog) == ("a", "b", "ase")
    with pytest.raises(SystemExit):
        build_parser().parse_args([])  # still required without a file


def test_save_and_reload_roundtrip(tmp_path):
    a = _parse(tmp_path, TOML, ["-sep", "2.5"])
    for name in ("saved.toml", "saved.json"):
        save_input_file(a, str(tmp_path / name))
        b = _parse(tmp_path, (tmp_path / name).read_text(), name="re_" + name)
        for k in ("infile1", "miller1", "miller_list", "prog", "convSep", "sep", "relax_cell_bulk", "hor_steps", "chem_pots1", "hubU"):
            assert getattr(a, k) == getattr(b, k), k
    assert "relax_cell_bulk = true" in (tmp_path / "saved.toml").read_text()
    assert load_input_file(str(tmp_path / "saved.toml"))["sep"] == 2.5


def test_run_from_input_file(tmp_path, monkeypatch):
    import ase.io
    monkeypatch.chdir(tmp_path)
    ase.io.write("Cu.cif", bulk("Cu", "fcc", a=3.61, cubic=True))
    ase.io.write("Ni.cif", bulk("Ni", "fcc", a=3.52, cubic=True))
    (tmp_path / "run.toml").write_text('infile1="Cu.cif"\ninfile2="Ni.cif"\nm1=[1,1,1]\nm2=[1,1,1]\ncr1=3\ncr2=3\n'
                                       'prog="ase"\nase_pot="EMT"\nmsd=20\nsurf_only=true\n')
    assert main(["--input", "run.toml", "--save_input", "used.toml"]) == 0
    assert (tmp_path / "used.toml").exists() and (tmp_path / "OUTPUT" / "results.json").exists()

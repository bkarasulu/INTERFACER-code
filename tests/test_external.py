import os

import numpy as np
import pytest
from ase.build import bulk, fcc111

from tools.external import _ldau_lists, _vasp_cmd, call_ase, call_vasp, handle_magmoms, vasp_continue


@pytest.mark.parametrize("exe,variant,expected", [
    ("mpirun -n 4 vasp_std", "gam", "mpirun -n 4 vasp_gam > vasp.out"),
    ("srun -n 8 vasp_gam > vasp.out", "std", "srun -n 8 vasp_std > vasp.out"),
    ("vasp", "std", "vasp > vasp.out"),  # previously gave a broken command
])
def test_vasp_cmd(exe, variant, expected):
    assert _vasp_cmd(exe, variant) == expected


def test_ldau_lists_same_length_and_orbitals():
    u, l = _ldau_lists(["Fe", "O", "Sm"], {"Fe": {"d": "2.7"}, "Sm": {"f": "6.1"}})
    assert u == [2.7, 0, 6.1] and l == [2, 0, 3]
    u, l = _ldau_lists(["Li", "O"], {"Li": {"s": "1.0"}})  # previously LDAUL was not appended for s
    assert len(u) == len(l) == 2
    with pytest.raises(ValueError):
        _ldau_lists(["Fe"], {"Fe": {"x": "1"}})


def test_handle_magmoms():
    a = bulk("NaCl", "rocksalt", a=5.64)
    handle_magmoms(a, ["Na", "1.5"])
    assert list(a.get_initial_magnetic_moments()) == [1.5, 0.0]


def test_vasp_continue_numbering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "OUTCAR").write_text("x")
    for i in (1, 2, 10):
        (tmp_path / f"RUN{i}").mkdir()
    vasp_continue()
    assert (tmp_path / "RUN11" / "OUTCAR").exists()  # RUN10 used to be read as RUN0


def test_call_ase_returns_energy_and_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = bulk("Cu", "fcc", a=3.61, cubic=True)
    a.rattle(0.05, seed=0)
    E, out = call_ase(a, opt=True, ctype="EMT", steps=50)
    assert E == pytest.approx(out.get_potential_energy())
    assert (tmp_path / "OUTPUT").exists()
    E_sp, _ = call_ase(a.copy(), opt=False)  # default ctype ('LJ') works
    assert np.isfinite(E_sp)
    with pytest.raises(ValueError):
        call_ase(a.copy(), ctype="nonsense")


def test_call_vasp_dry_inputs(tmp_path, monkeypatch, fake_potcars):
    monkeypatch.chdir(tmp_path)
    slab = fcc111("Cu", size=(1, 1, 3), vacuum=6.0)
    E, out = call_vasp(slab, typ="geom", name="slab", ENCUT=400, KPgrid="4 4 1", KPspacing=0.3, FixCell=True,
                       dipolCorr="sc", hubU={"Cu": {"d": "3.0"}}, exe="mpirun -n 4 vasp_std", dry=True)
    assert E == 0.0 and out.calc is None
    incar = (tmp_path / "slab" / "INCAR").read_text()
    for tag in ("ISIF = 2", "IBRION = 2", "LDAU = .TRUE.", "LDIPOL = .TRUE.", "IDIPOL = 3", "LASPH = .TRUE.", "LCHARG = .FALSE."):
        assert tag in incar
    kpts = (tmp_path / "slab" / "KPOINTS").read_text().split("\n")
    assert kpts[2].strip() == "Gamma" and kpts[3].split() == ["4", "4", "1"]
    assert os.getcwd() == str(tmp_path)


def test_call_vasp_invalid_typ(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        call_vasp(bulk("Cu"), typ="md", name="x", KPspacing=0.3, exe="vasp_std", dry=True)
    assert os.getcwd() == str(tmp_path)

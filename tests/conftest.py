import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):  # works without 'pip install -e .' as well
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def cu_bulk():
    from ase.build import bulk
    return bulk("Cu", "fcc", a=3.61, cubic=True)


@pytest.fixture
def ni_bulk():
    from ase.build import bulk
    return bulk("Ni", "fcc", a=3.52, cubic=True)


@pytest.fixture
def fake_potcars(tmp_path, monkeypatch):
    """Minimal placeholder POTCAR files, enough for ASE to write VASP inputs (no VASP run)."""
    pp = tmp_path / "pp"
    for el in ("Cu", "Ni", "H"):
        d = pp / "potpaw_PBE" / el
        d.mkdir(parents=True)
        (d / "POTCAR").write_text(
            f"  PAW_PBE {el} 06Sep2000\n 11.0\n   VRHFIN ={el}: d s\n   TITEL  = PAW_PBE {el} 06Sep2000\n"
            "   POMASS =   63.546; ZVAL   =   11.000    mass and valenz\n   ENMAX  =  295.446; ENMIN  =  221.585 eV\n End of Dataset\n")
    monkeypatch.setenv("VASP_PP_PATH", str(pp))
    return pp

"""GPAW and ONETEP engines (input generation / calculator settings; neither code is needed), fixed layers, dry runs."""
import sys
import types

import numpy as np
import pytest
from ase.build import bulk, fcc111

from interfacer.calculators import CalcSettings, get_engine, middle_layer_indices
from interfacer.calculators.engines import GpawEngine, OnetepEngine


def test_onetep_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ONETEP_PP_PATH", str(tmp_path / "pp"))
    eng = get_engine("onetep", CalcSettings(encut=600, xc="PBE", exe="mpirun -n 8 onetep.exe",
                                            calc_kwargs={"keywords": {"maxit_ngwf_cg": 50}, "ngwfs_radius": 8.0}))
    assert isinstance(eng, OnetepEngine)
    a = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    calc = eng.calculator_for("bulk1", a)
    assert str(calc.directory).endswith("ONETEP-tmp/bulk1")
    calc.write_inputfiles(a, ["energy", "forces"])
    text = (tmp_path / "ONETEP-tmp" / "bulk1" / "onetep.dat").read_text().lower()
    for key in ("cutoff_energy", "600", "xc_functional", "pbe", "maxit_ngwf_cg", "pseudo_path", "positions_abs"):
        assert key in text, key


def test_gpaw_calculator_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    made = {}

    class GPAW:
        def __init__(self, **kw):
            made.update(kw)
    fake = types.ModuleType("gpaw")
    fake.GPAW, fake.PW, fake.FermiDirac = GPAW, (lambda e: ("PW", e)), (lambda w: ("FD", w))
    monkeypatch.setitem(sys.modules, "gpaw", fake)
    eng = get_engine("gpaw", CalcSettings(encut=450, xc="PBE", sigma=0.1, kpspacing=0.25, calc_kwargs={"spinpol": True}))
    assert isinstance(eng, GpawEngine)
    eng.calculator_for("slab1_cut0_pre", None, kpgrid="4 4 1")
    assert made["mode"] == ("PW", 450) and made["kpts"] == (4, 4, 1) and made["spinpol"] is True
    assert made["occupations"] == ("FD", 0.1) and made["txt"].endswith("slab1_cut0_pre.txt")
    eng.calculator_for("bulk1", None)
    assert made["kpts"] == {"density": 4.0, "gamma": True}


def test_gpaw_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpaw", None)
    with pytest.raises(ImportError, match="gpaw"):
        GpawEngine().calculator_for("x", None)


def test_middle_layers_fixed():
    from ase.calculators.emt import EMT
    from interfacer.calculators import relax
    slab = fcc111("Cu", size=(2, 2, 5), vacuum=5.0)
    slab.rattle(0.05, seed=2)
    idx = middle_layer_indices(slab, 1)
    z = slab.positions[:, 2]
    assert len(idx) == 4 and np.isclose(np.median(z[idx]), np.median(z), atol=0.3)
    before = slab.positions[idx].copy()
    relax(slab, EMT(), fmax=0.05, steps=20, fix_bottom_layers=1, fix_mode="middle")
    assert np.allclose(slab.positions[idx], before)


def test_dry_run_for_ase_engines(tmp_path):
    eng = get_engine("ase", CalcSettings(ase_pot="EMT", dry=True, outdir=str(tmp_path)))
    E, a = eng.relax(bulk("Cu", cubic=True), "bulk1")
    assert E == 0.0 and a.calc is None


def test_structures_only_cli(tmp_path, monkeypatch):
    import ase.io
    from interfacer.cli import main
    monkeypatch.chdir(tmp_path)
    ase.io.write("Cu.cif", bulk("Cu", "fcc", a=3.61, cubic=True))
    ase.io.write("Ni.cif", bulk("Ni", "fcc", a=3.52, cubic=True))
    assert main(["-i1", "Cu.cif", "-i2", "Ni.cif", "-m1", "1", "1", "1", "-m2", "1", "1", "1", "-prog", "mlip", "--mlip",
                 "custom:interfacer.calculators.classical:classical_calculator", "--structures_only", "-msd", "20"]) == 0
    for f in ("slab1_cut0_aligned.xyz", "interface_cuts_0_0.xyz", "interface_cuts_0_0_opted.cif"):
        assert (tmp_path / "OUTPUT" / f).exists()

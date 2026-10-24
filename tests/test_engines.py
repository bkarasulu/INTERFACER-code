import numpy as np
import pytest
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT

from interfacer.calculators import (AseEngine, CalcSettings, CastepEngine, VaspEngine, bottom_layer_indices, get_engine,
                                    get_optimiser, relax)


def test_get_engine():
    assert isinstance(get_engine("vasp"), VaspEngine)
    assert isinstance(get_engine("CASTEP"), CastepEngine)
    e = get_engine("ase", CalcSettings(ase_pot="EMT"))
    assert isinstance(e, AseEngine) and e.label == "EMT"
    with pytest.raises(ValueError):
        get_engine("gaussian")


def test_get_optimiser():
    from ase.optimize import FIRE
    assert get_optimiser("fire") is FIRE
    with pytest.raises(ValueError):
        get_optimiser("nope")


def test_bottom_layer_indices():
    slab = fcc111("Cu", size=(2, 2, 4), vacuum=5.0)
    idx = bottom_layer_indices(slab, 2)
    z = slab.positions[:, 2]
    assert len(idx) == 8
    assert z[idx].max() < np.sort(np.unique(z.round(3)))[2] - 0.1


def test_relax_fixed_and_cell(tmp_path):
    a = bulk("Cu", "fcc", a=3.7, cubic=True)
    E_fixed, a1 = relax(a.copy(), EMT(), fmax=0.01, steps=100)
    assert a1.cell.lengths()[0] == pytest.approx(3.7)
    E_cell, a2 = relax(a.copy(), EMT(), fmax=0.01, steps=200, relax_cell=True, trajectory=str(tmp_path / "t.traj"))
    assert a2.cell.lengths()[0] == pytest.approx(3.59, abs=0.015)
    assert E_cell < E_fixed
    assert (tmp_path / "t.traj").exists()


def test_relax_keeps_bottom_layers_fixed():
    slab = fcc111("Cu", size=(2, 2, 4), vacuum=5.0)
    slab.rattle(0.05, seed=3)
    idx = bottom_layer_indices(slab, 2)
    before = slab.positions[idx].copy()
    relax(slab, EMT(), fmax=0.02, steps=50, fix_bottom_layers=2)
    assert np.allclose(slab.positions[idx], before)
    assert slab.constraints == []  # temporary constraint removed


def test_ase_engine_with_any_calculator(tmp_path):
    s = CalcSettings(outdir=str(tmp_path), steps=20, relax_cell_bulk=True)
    eng = AseEngine(EMT, label="EMT", settings=s)  # class -> new calculator per calculation
    a = bulk("Cu", "fcc", a=3.7, cubic=True)
    a.name = "bulk"
    E_sp, _ = eng.single_point(a.copy(), "sp")
    E_opt, out = eng.relax(a, "bulk", relax_cell=True)
    assert E_opt < E_sp
    assert (tmp_path / "bulk_opt.traj").exists()
    eng2 = AseEngine(EMT(), label="EMT-shared", settings=s)  # calculator object
    assert np.isfinite(eng2.single_point(bulk("Cu"), "x")[0])
    with pytest.raises(ValueError):
        eng.run(bulk("Cu"), "md", "x")


def test_vasp_engine_fix_layers_selective_dynamics(tmp_path, monkeypatch, fake_potcars):
    monkeypatch.chdir(tmp_path)
    eng = VaspEngine(CalcSettings(exe="vasp_std", dry=True))
    slab = fcc111("Cu", size=(1, 1, 4), vacuum=5.0)
    eng.relax(slab, "s", FixCell=True, KPgrid_="2 2 1", fix_bottom_layers=2)
    poscar = (tmp_path / "s" / "POSCAR").read_text()
    assert "Selective dynamics" in poscar
    assert poscar.count("F   F   F") == 2

import os

import pytest
from ase.build import bulk

from interfacer.calculators import CalcSettings, get_engine
from interfacer.calculators.engines import CastepEngine, VaspEngine
from interfacer.workflow import InterfaceWorkflow, WorkflowConfig


def _run(tmp_path, capsys, **kw):
    eng = get_engine("ase", CalcSettings(ase_pot="EMT", steps=100, outdir=str(tmp_path / "OUT"), cache_dir=str(tmp_path / "cache"), **kw))
    cfg = WorkflowConfig(miller1=(1, 1, 1), miller2=(1, 1, 1), creps1=2, creps2=2, max_slab_dimension=20, conv_sep=True,
                         sep_init=1.5, sep_final=2.5, sep_step=0.5, outdir=str(tmp_path / "OUT"), outfile=str(tmp_path / "i.out"))
    res = InterfaceWorkflow(eng, cfg).run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    return res, capsys.readouterr().out


def test_ase_restart_reuses_all_calculations(tmp_path, capsys):
    r1, out1 = _run(tmp_path, capsys, restart=True)
    assert "Reusing" not in out1 and len(os.listdir(tmp_path / "cache")) > 5
    r2, out2 = _run(tmp_path, capsys, restart=True)
    n_calcs = out1.count("Getting SP energy") + out1.count("Optimising the")
    assert out2.count("Reusing the finished calculation") == n_calcs   # incl. the separation scan
    assert "Optimising the" not in out2 and "Getting SP energy" not in out2
    assert r2.interfaces[0].W_ad == pytest.approx(r1.interfaces[0].W_ad)
    r3, out3 = _run(tmp_path, capsys, restart=True, overwrite=True)
    assert "Reusing" not in out3


def test_ase_restart_ignores_changed_settings(tmp_path, capsys):
    _run(tmp_path, capsys, restart=True)
    _, out = _run(tmp_path, capsys, restart=True, fmax=0.01)   # different optimiser setting -> recomputed
    assert "Optimising the" in out


def test_no_cache_without_restart(tmp_path, capsys):
    _run(tmp_path, capsys)
    assert not (tmp_path / "cache").exists()


def test_vasp_overwrite_moves_old_run(tmp_path, monkeypatch, fake_potcars):
    monkeypatch.chdir(tmp_path)
    a = bulk("Cu", "fcc", a=3.61, cubic=True)
    VaspEngine(CalcSettings(exe="vasp_std", dry=True)).relax(a.copy(), "bulk1")
    (tmp_path / "bulk1" / "OUTCAR").write_text("old run")
    VaspEngine(CalcSettings(exe="vasp_std", dry=True, overwrite=True)).relax(a.copy(), "bulk1")
    assert (tmp_path / "bulk1_old1" / "OUTCAR").read_text() == "old run"
    assert (tmp_path / "bulk1" / "INCAR").exists() and not (tmp_path / "bulk1" / "OUTCAR").exists()


CASTEP_OUT = """ Unit Cell
                           Real Lattice(A)              Reciprocal Lattice(1/A)
         3.6100000   0.0000000   0.0000000        1.740   0.000   0.000
         0.0000000   3.6100000   0.0000000        0.000   1.740   0.000
         0.0000000   0.0000000   3.6100000        0.000   0.000   1.740

                     Lattice parameters(A)       Cell Angles
            x  Cu           1         0.000000   0.000000   0.000000         x
            x  Cu           2         0.000000   0.500000   0.500000         x
            x  Cu           3         0.500000   0.000000   0.500000         x
            x  Cu           4         0.500000   0.500000   0.000000         x
Final energy, E             =  -3456.789 eV
Final free energy (E-TS)    =  -3456.790 eV
Total time          =     12.34 s
"""


def test_castep_reuses_finished_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CASTEP-tmp").mkdir()
    (tmp_path / "CASTEP-tmp" / "bulk1.castep").write_text(CASTEP_OUT)
    E, atoms = (lambda x: (x[0], x[-1]))(CastepEngine(CalcSettings()).single_point(bulk("Cu", "fcc", a=3.61, cubic=True), "bulk1"))
    assert E == pytest.approx(-3456.789) and len(atoms) == 4
    # an unfinished run (no 'Total time') is not reused; --overwrite moves finished outputs aside
    (tmp_path / "CASTEP-tmp" / "bulk2.castep").write_text(CASTEP_OUT.replace("Total time", "Tot"))
    from interfacer.calculators.engines import read_finished_castep
    assert read_finished_castep(str(tmp_path / "CASTEP-tmp" / "bulk2"), "sp", bulk("Cu", cubic=True)) is None
    eng = CastepEngine(CalcSettings(overwrite=True))
    monkeypatch.setattr("interfacer.calculators.engines.call_castep", lambda *a, **k: (0.0, a[0]))
    eng.single_point(bulk("Cu", "fcc", a=3.61, cubic=True), "bulk1")
    assert not (tmp_path / "CASTEP-tmp" / "bulk1.castep").exists()
    assert any(f.startswith("bulk1.castep_old") for f in os.listdir(tmp_path / "CASTEP-tmp"))

import json

import pytest
from ase.build import bulk

from interfacer.calculators import CalcSettings, get_engine
from interfacer.workflow import InterfaceWorkflow, WorkflowConfig


@pytest.fixture
def emt_engine(tmp_path):
    return get_engine("ase", CalcSettings(ase_pot="EMT", steps=100, outdir=str(tmp_path / "OUT")))


def _cfg(tmp_path, **kw):
    base = dict(miller1=(1, 1, 1), miller2=(1, 1, 1), creps1=3, creps2=3, max_slab_dimension=20,
                outdir=str(tmp_path / "OUT"), outfile=str(tmp_path / "interfaces.out"))
    base.update(kw)
    return WorkflowConfig(**base)


def test_python_api_full_run(tmp_path, emt_engine):
    wf = InterfaceWorkflow(emt_engine, _cfg(tmp_path))
    res = wf.run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    assert len(res.interfaces) == 1
    r = res.interfaces[0]
    assert r.n_interfaces == 2 and r.sep == pytest.approx(2.0)
    assert 0 < r.W_ad_J < (r.W_surf1 + r.W_surf2) * 0.160217663
    assert res.slabs1[0].sfe == pytest.approx(r.W_surf1, rel=0.05)
    data = json.loads((tmp_path / "OUT" / "results.json").read_text())
    assert data["interfaces"][0]["W_ad"] == pytest.approx(r.W_ad)
    assert (tmp_path / "OUT" / "interface_cuts_0_0_opted.cif").exists()


def test_steps_individually(tmp_path, emt_engine):
    wf = InterfaceWorkflow(emt_engine, _cfg(tmp_path, miller_list=[(1, 0, 0), (1, 1, 1)], creps1=2, creps2=2))
    wf.run_bulk(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    assert wf.Ebulk1 < 0.1 and wf.Ebulk2 < 0.1
    wf.cfg.miller1 = wf.cfg.miller2 = (1, 0, 0)
    wf.screen_surfaces()
    assert wf.cfg.miller1 == (1, 1, 1) and wf.cfg.miller2 == (1, 1, 1)
    slabs = wf.build_slabs(1)
    sfe = wf.compute_sfe(1, slabs)
    assert sfe[0].sfe_J == pytest.approx(1.03, abs=0.1)  # EMT Cu(111)
    sel = wf.select_slabs(1, slabs, sfe)
    assert sel[0][0] == 0


def test_surf_only_and_symmetric_slabs(tmp_path, emt_engine):
    wf = InterfaceWorkflow(emt_engine, _cfg(tmp_path, sym_slabs1=True, thickness=5.0, surf_only=True))
    res = wf.run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    assert res.interfaces == [] and len(res.slabs1) >= 1 and len(res.slabs2) == 1


def test_failed_alignment_is_skipped(tmp_path, emt_engine):
    # too small a maximum cell size: no commensurate cell -> reported and skipped, run finishes
    wf = InterfaceWorkflow(emt_engine, _cfg(tmp_path, miller2=(1, 0, 0), max_slab_dimension=4, percentage_tolerance=0.1))
    res = wf.run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    assert res.interfaces == []


def test_interface_cell_fixed_after_separation_scan(tmp_path, emt_engine):
    # after -tsep the final interface optimisation keeps the cell (and thus the scanned separation)
    wf = InterfaceWorkflow(emt_engine, _cfg(tmp_path, conv_sep=True, sep_init=1.5, sep_final=2.5, sep_step=0.5))
    wf.run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    import ase.io
    before = ase.io.read(tmp_path / "OUT" / ("int-sep_%.2fA.xyz" % wf.results.interfaces[0].sep))
    after = ase.io.read(tmp_path / "OUT" / "interface_cuts_0_0_opted.xyz")
    assert after.cell.lengths()[2] == pytest.approx(before.cell.lengths()[2])
    # without scans the cell length along z is relaxed (ASE engines)
    wf2 = InterfaceWorkflow(emt_engine, _cfg(tmp_path / "b", outdir=str(tmp_path / "b" / "OUT"), outfile=str(tmp_path / "b.out")))
    wf2.run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    pre = ase.io.read(tmp_path / "b" / "OUT" / "interface-pre_opt_cuts_0_0.xyz")
    post = ase.io.read(tmp_path / "b" / "OUT" / "interface_cuts_0_0_opted.xyz")
    assert abs(post.cell.lengths()[2] - pre.cell.lengths()[2]) > 1e-3


def test_sfe_scan_polar_terminations(tmp_path):
    # NaCl(111): Na- and Cl-terminated symmetric slabs; their SFEs vs mu(Na) cross
    import numpy as np
    from interfacer.calculators import CalcSettings, get_engine
    eng = get_engine("ase", CalcSettings(ase_pot="LJ", steps=2, outdir=str(tmp_path / "OUT")))
    cfg = _cfg(tmp_path, sym_slabs1=True, thickness=6.0, sym_method="internal", no_slabs=2, surf_only=True,
               chem_pots1={"Na": -1.0, "Cl": -1.0}, sfe_scan1=("Na", "Cl", -3.0, 3.0))
    wf = InterfaceWorkflow(eng, cfg)
    wf.run(bulk("NaCl", "rocksalt", a=5.64, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    data = np.loadtxt(tmp_path / "OUT" / "sfe_vs_mu_1.csv", delimiter=",", skiprows=1)
    assert data.shape[1] == 3 and (tmp_path / "OUT" / "sfe_vs_mu_1.png").exists()
    diff = data[:, 1] - data[:, 2]
    assert diff[0] * diff[-1] < 0          # the stable termination changes within the range


def test_interface_energy_reported(tmp_path, emt_engine):
    res = InterfaceWorkflow(emt_engine, _cfg(tmp_path)).run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True))
    r = res.interfaces[0]
    assert r.gamma_int == pytest.approx(r.W_surf1 + r.W_surf2 - r.W_ad)
    assert 0 < r.gamma_int_J < 0.3        # coherent Cu|Ni: small interface energy
    assert "gamma_int" in (tmp_path / "interfaces.out").read_text()

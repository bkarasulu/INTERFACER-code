"""Grain-boundary models (symmetric tilt and twist) and three-layer interfaces."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from interfacer.builder import align_three, make_multilayer, make_slab
from interfacer.calculators import CalcSettings, get_engine
from interfacer.cli import main
from interfacer.grainboundary import gb_energy, plane_spacing, symmetric_tilt_gb, twist_gb
from interfacer.workflow import InterfaceWorkflow, WorkflowConfig

J = 16.0217663  # eV/A^2 -> J/m^2


@pytest.fixture
def cu():
    return bulk("Cu", "fcc", a=3.61, cubic=True)


def _gamma(gb, E_bulk):
    gb.calc = EMT()
    return gb_energy(gb.get_potential_energy(), len(gb), E_bulk, gb.cell.area(2)) * J


def _ebulk(cu):
    b = cu.copy(); b.calc = EMT()
    return b.get_potential_energy() / len(b)


@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 0)])
def test_tilt_mirror_plane_is_perfect_crystal(cu, miller):
    gb, g1, g2 = symmetric_tilt_gb(cu, miller)
    assert len(gb) == len(g1) + len(g2)
    assert abs(_gamma(gb, _ebulk(cu))) < 1e-3


def test_tilt_111_is_coherent_twin(cu):
    # fcc (111) is not a mirror plane of the stacking ABC: the mirror gives the Sigma3 coherent twin (...ABCBA...)
    gb, _, _ = symmetric_tilt_gb(cu, (1, 1, 1), creps=4)
    assert abs(_gamma(gb, _ebulk(cu))) < 0.05


def test_tilt_sigma5_has_finite_energy(cu):
    gb, _, _ = symmetric_tilt_gb(cu, (3, 1, 0))
    g = _gamma(gb, _ebulk(cu))
    assert 0.3 < g < 3.0
    # an expansion at the boundaries adds no atoms and changes the energy
    gb2, _, _ = symmetric_tilt_gb(cu, (3, 1, 0), expansion=0.5)
    assert len(gb2) == len(gb) and gb2.cell[2, 2] == pytest.approx(gb.cell[2, 2] + 1.0)


@pytest.mark.parametrize("miller,mn,sigma,angle", [((1, 0, 0), (2, 1), 5, 36.87), ((1, 1, 1), (2, 1), 7, 21.79),
                                                   ((1, 0, 0), (3, 1), 5, 36.87), ((1, 0, 0), (3, 2), 13, 22.62)])
def test_twist_sigma_and_angle(cu, miller, mn, sigma, angle):
    gb, g1, g2, twist, s = twist_gb(cu, miller, *mn)
    assert s == sigma
    assert twist == pytest.approx(angle, abs=0.01)
    assert np.allclose(g1.cell[:2], g2.cell[:2])
    assert len(g1) == len(g2) and len(gb) == len(g1) + len(g2)
    assert _gamma(gb, _ebulk(cu)) > 0.05


def test_twist_errors(cu):
    with pytest.raises(ValueError):
        twist_gb(cu, (1, 0, 0), 1, 1)
    with pytest.raises(ValueError):
        twist_gb(cu, (1, 1, 0), 2, 1)   # rectangular plane


def test_plane_spacing(cu):
    assert plane_spacing(make_slab((1, 0, 0), cu, repeat=(1, 1, 3))) == pytest.approx(3.61 / 2, abs=0.01)


def test_align_three_and_multilayer():
    cu = bulk("Cu", "fcc", a=3.61, cubic=True)
    ni = bulk("Ni", "fcc", a=3.52, cubic=True)
    ag = bulk("Ag", "fcc", a=4.09, cubic=True)
    slabs = [make_slab((1, 0, 0), b, repeat=(1, 1, 2), square=0) for b in (cu, ni, ag)]
    s1, s2, s3 = align_three(*slabs, 3.0, 25.0, 2.0, 4.0, 7.0, cu, ni, ag)
    assert np.allclose(s1.cell[:2, :2], s2.cell[:2, :2], rtol=0.05) and np.allclose(s2.cell[:2, :2], s3.cell[:2, :2], rtol=0.05)
    stack = make_multilayer([s1, s2, s3], [2.0, 2.5], 4.0, reorder=False)
    assert len(stack) == len(s1) + len(s2) + len(s3)
    sym = stack.get_chemical_symbols()
    z = stack.positions[:, 2]
    zc, zn, za = (z[[i for i, e in enumerate(sym) if e == el]] for el in ("Cu", "Ni", "Ag"))
    assert zc.max() < zn.min() and zn.max() < za.min()          # stacking order 1|2|3
    assert zn.min() - zc.max() == pytest.approx(2.0, abs=1e-6)
    assert za.min() - zn.max() == pytest.approx(2.5, abs=1e-6)
    assert z.min() == pytest.approx(4.0) and stack.cell[2, 2] - z.max() == pytest.approx(4.0)
    with pytest.raises(ValueError):
        make_multilayer([s1, s2, s3], [2.0], 4.0)


@pytest.fixture
def emt_engine(tmp_path):
    return get_engine("ase", CalcSettings(ase_pot="EMT", steps=100, outdir=str(tmp_path / "OUT")))


def test_workflow_gb_modes(tmp_path, emt_engine, cu):
    cfg = WorkflowConfig(gb="twist", miller1=(1, 0, 0), gb_mn=(2, 1), outdir=str(tmp_path / "OUT"), outfile=str(tmp_path / "o.out"))
    res = InterfaceWorkflow(emt_engine, cfg).run(cu)
    r = res.grain_boundaries[0]
    assert r.sigma == 5 and r.twist_angle == pytest.approx(36.87, abs=0.01)
    assert 0.1 < r.gamma_gb_J < 2.0
    assert (tmp_path / "OUT" / "results.json").exists()
    cfg = WorkflowConfig(gb="tilt", miller1=(1, 0, 0), outdir=str(tmp_path / "OUT"), outfile=str(tmp_path / "o.out"))
    r = InterfaceWorkflow(emt_engine, cfg).run(cu).grain_boundaries[0]
    assert abs(r.gamma_gb_J) < 0.01


def test_workflow_trilayer(tmp_path, emt_engine):
    cfg = WorkflowConfig(miller1=(1, 0, 0), miller2=(1, 0, 0), miller3=(1, 0, 0), creps1=2, creps2=2, creps3=2, sep2=2.0,
                         max_slab_dimension=20, outdir=str(tmp_path / "OUT"), outfile=str(tmp_path / "o.out"))
    res = InterfaceWorkflow(emt_engine, cfg).run(bulk("Cu", "fcc", a=3.61, cubic=True), bulk("Ni", "fcc", a=3.52, cubic=True),
                                                bulk("Cu", "fcc", a=3.61, cubic=True))
    assert len(res.interfaces) == 1 and len(res.trilayers) == 1
    t = res.trilayers[0]
    assert t.sep23 == pytest.approx(2.0)
    # Cu|Ni|Cu has two Cu|Ni interfaces: W_ad_total is about twice the single W_ad
    assert t.W_ad_total == pytest.approx(2 * res.interfaces[0].W_ad, rel=0.3)


def test_cli_gb_without_i2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from ase.io import write
    write("Cu.cif", bulk("Cu", "fcc", a=3.61, cubic=True))
    assert main(["-i1", "Cu.cif", "-prog", "ase", "-ase_pot", "EMT", "--gb", "twist", "--gb_mn", "2", "1", "-m1", "1", "0", "0"]) == 0
    assert (tmp_path / "OUTPUT" / "results.json").exists()
    with pytest.raises(SystemExit):
        main(["-i1", "Cu.cif", "-prog", "ase"])

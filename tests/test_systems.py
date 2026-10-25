"""Test systems from the INTERFACER task list: Li2S|Li, Li3PS4|Li2S, Na3OCl|Na and Li|Na.

Structure building (slabs, alignment, interface) is checked for all systems; short workflow runs use the Morse/LJ
potentials; energies with MACE-MP-0 run only with INTERFACER_SLOW_TESTS=1 and mace-torch installed.
"""
import os

import numpy as np
import pytest
import ase.io

from interfacer.analysis import find_clashes, is_stoichiometric, misfit
from interfacer.builder import make_interface, make_slab, slab_aligner

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = {"Li2S": "Li2S.cif", "Li": "Li.cif", "Na": "Na.cif", "Li3PS4": "Li3PS4.vasp", "Na3OCl": "Na3OCl.cif"}

# (material 1, Miller 1, material 2, Miller 2, max cell length, tolerance %)
PAIRS = [("Li2S", (1, 1, 1), "Li", (1, 0, 0), 20, 4),
         ("Li2S", (1, 0, 0), "Li", (1, 0, 0), 20, 4),
         ("Li3PS4", (0, 1, 0), "Li2S", (1, 0, 0), 25, 5),
         ("Li3PS4", (1, 0, 0), "Li2S", (1, 1, 0), 25, 5),
         ("Na3OCl", (1, 0, 0), "Na", (1, 0, 0), 20, 4),
         ("Na3OCl", (1, 1, 0), "Na", (1, 1, 0), 20, 4),
         ("Li", (1, 0, 0), "Na", (1, 0, 0), 20, 4),
         ("Li", (1, 1, 0), "Na", (1, 1, 0), 20, 4)]


def bulk_of(name):
    return ase.io.read(os.path.join(DATA, FILES[name]))


@pytest.mark.parametrize("m1,h1,m2,h2,msd,pt", PAIRS, ids=lambda v: str(v).replace(" ", ""))
def test_interface_building(m1, h1, m2, h2, msd, pt):
    b1, b2 = bulk_of(m1), bulk_of(m2)
    s1, s2 = make_slab(h1, b1, repeat=(1, 1, 2)), make_slab(h2, b2, repeat=(1, 1, 2))
    assert is_stoichiometric(s1, b1) and is_stoichiometric(s2, b2)
    a1, a2 = slab_aligner(s1, s2, 3, msd, 2, pt, 7, b1, b2)
    # matched in-plane cells
    assert misfit(a1, a2) < pt / 100
    assert np.allclose(a1.cell.cellpar()[5], a2.cell.cellpar()[5], atol=pt)
    assert np.allclose(a1.cell.lengths()[:2], a2.cell.lengths()[:2], rtol=pt / 100)
    assert is_stoichiometric(a1, b1) and is_stoichiometric(a2, b2)
    for a in (a1, a2):
        assert find_clashes(a) == []
    # interface: slab 1 below slab 2 with the requested gap, right-handed cell, all atoms kept, no clashes
    sep = 2.0
    it = make_interface(a1, a2, sep, 1.0, reorder=False)
    n1 = len(a1)
    z = it.positions[:, 2]
    assert z[n1:].min() - z[:n1].max() == pytest.approx(sep, abs=1e-6)
    assert np.linalg.det(it.cell[:]) > 0 and it.cell[2][2] > 0
    assert it.get_scaled_positions(wrap=False)[:, 2].min() >= -1e-8
    assert sorted(it.get_chemical_symbols()) == sorted(a1.get_chemical_symbols() + a2.get_chemical_symbols())
    assert find_clashes(it) == []


@pytest.mark.parametrize("miller", [(1, 1, 0), (1, 1, 1)])
def test_na3ocl_slabs_have_no_clashes(miller):
    # to-do list: "ASE surface slab (cut) is problematic for Na3OCl (110) and (111) surfaces (O clashes with the upper layer)"
    b = bulk_of("Na3OCl")
    for cr in (1, 2, 3):
        s = make_slab(miller, b, repeat=(1, 1, cr))
        s.center(vacuum=6.0, axis=2)
        d = s.get_all_distances(mic=True)
        np.fill_diagonal(d, 99)
        assert d.min() == pytest.approx(b.cell.lengths()[0] / 2, abs=0.01)   # nearest Na-O distance as in the bulk


@pytest.mark.parametrize("m1,h1,m2,h2,pot", [("Li2S", (1, 1, 1), "Li", (1, 0, 0), "Morse"),
                                              ("Li", (1, 1, 0), "Na", (1, 1, 0), "LJ"),
                                              ("Na3OCl", (1, 0, 0), "Na", (1, 0, 0), "Morse")])
def test_workflow_runs_with_simple_potentials(m1, h1, m2, h2, pot, tmp_path):
    from interfacer.calculators import CalcSettings, get_engine
    from interfacer.workflow import InterfaceWorkflow, WorkflowConfig
    eng = get_engine("ase", CalcSettings(ase_pot=pot, steps=3, outdir=str(tmp_path / "OUT")))
    cfg = WorkflowConfig(miller1=h1, miller2=h2, creps1=1, creps2=2, max_slab_dimension=20, zero_chem_pots=True,
                         outdir=str(tmp_path / "OUT"), outfile=str(tmp_path / "i.out"))
    res = InterfaceWorkflow(eng, cfg).run(bulk_of(m1), bulk_of(m2))
    assert len(res.interfaces) == 1 and np.isfinite(res.interfaces[0].W_ad)


slow = pytest.mark.skipif(os.environ.get("INTERFACER_SLOW_TESTS") != "1", reason="set INTERFACER_SLOW_TESTS=1 to run")


@slow
@pytest.mark.parametrize("m1,h1,c1,m2,h2,c2", [("Li2S", (1, 1, 1), 1, "Li", (1, 0, 0), 3),
                                                ("Na3OCl", (1, 0, 0), 3, "Na", (1, 0, 0), 4),
                                                ("Li", (1, 1, 0), 3, "Na", (1, 1, 0), 3)])
def test_mace_energies(m1, h1, c1, m2, h2, c2, tmp_path):
    pytest.importorskip("mace.calculators")
    from interfacer.calculators import CalcSettings, get_engine
    from interfacer.convergence import cfactor
    from interfacer.workflow import InterfaceWorkflow, WorkflowConfig
    eng = get_engine("mlip", CalcSettings(mlip="mace", mlip_model="small", device="cpu", steps=200, relax_cell_bulk=True,
                                          outdir=str(tmp_path / "OUT")))
    cfg = WorkflowConfig(miller1=h1, miller2=h2, creps1=c1, creps2=c2, max_slab_dimension=20, outdir=str(tmp_path / "OUT"),
                         outfile=str(tmp_path / "i.out"))
    r = InterfaceWorkflow(eng, cfg).run(bulk_of(m1), bulk_of(m2)).interfaces[0]
    ws1, ws2, wad = r.W_surf1 * cfactor, r.W_surf2 * cfactor, r.W_ad_J
    assert 0.05 < ws1 < 2.0 and 0.05 < ws2 < 2.0      # J/m^2: plausible surface energies
    assert 0 < wad < ws1 + ws2 + 0.05                 # bound, and at most the sum of the surface energies (+ strain/noise)

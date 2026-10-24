import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from interfacer.analysis import calculate_chem_pots, calculate_SFE, find_prim, get_fu, get_interface_energy, misfit, surf_area
from interfacer.builder import make_slab


def _with_emt(atoms):
    atoms.calc = EMT()
    atoms.get_potential_energy()
    return atoms


def test_get_fu():
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    assert get_fu(nacl) == 4
    assert get_fu(bulk("Cu")) == 1


def test_misfit_identical_and_different(cu_bulk, ni_bulk):
    s1 = make_slab((1, 0, 0), cu_bulk)
    assert misfit(s1, s1) == pytest.approx(0)
    s2 = make_slab((1, 0, 0), ni_bulk)
    a1, a2 = surf_area(s1), surf_area(s2)
    assert misfit(s1, s2) == pytest.approx(1 - 2 * min(a1, a2) / (a1 + a2))


def test_find_prim_fcc(cu_bulk):
    prim = find_prim(cu_bulk)
    assert len(prim) == 1


def test_sfe_stoichiometric_matches_definition(cu_bulk):
    bulk_ = _with_emt(cu_bulk.copy())
    slab = make_slab((1, 1, 1), cu_bulk, repeat=(1, 1, 3))
    slab.center(vacuum=6.0, axis=2)
    _with_emt(slab)
    A = surf_area(slab) / 100  # nm^2
    expected = (slab.get_potential_energy() - len(slab) * bulk_.get_potential_energy() / len(bulk_)) / (2 * A)
    assert calculate_SFE(slab, bulk_) == pytest.approx(expected)
    assert expected > 0


def test_sfe_nonstoichiometric_with_chem_pots():
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)  # Na4Cl4
    from ase.calculators.singlepoint import SinglePointCalculator
    nacl.calc = SinglePointCalculator(nacl, energy=-8.0)  # -2 eV per NaCl unit
    slab = nacl.repeat((1, 1, 2))
    del slab[[i for i, s in enumerate(slab.get_chemical_symbols()) if s == "Cl"][:1]]  # Na8Cl7
    slab.center(vacuum=5.0, axis=2)
    slab.calc = SinglePointCalculator(slab, energy=-14.0)
    A = surf_area(slab) / 100
    mu = {"Na": -1.0, "Cl": -1.0}
    # reference: 7 NaCl units (E=-14) + 1 extra Na (mu=-1) -> E_slab - (-14) - (-1) = 1
    assert calculate_SFE(slab, nacl, chem_pots=mu) == pytest.approx(1.0 / (2 * A))
    with pytest.raises(ValueError):
        calculate_SFE(slab, nacl)  # non-stoichiometric without chemical potentials


def test_sfe_element_mismatch_raises(cu_bulk, ni_bulk):
    b = _with_emt(cu_bulk.copy())
    s = _with_emt(ni_bulk.copy())
    with pytest.raises(ValueError):
        calculate_SFE(s, b)


def test_calculate_chem_pots():
    from ase.calculators.singlepoint import SinglePointCalculator
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    nacl.calc = SinglePointCalculator(nacl, energy=-16.0)   # 4 x (mu_Na + mu_Cl) = -16
    na = bulk("Na", "bcc", a=4.29, cubic=True)              # Na2
    na.calc = SinglePointCalculator(na, energy=-2.6)        # mu_Na = -1.3
    cp = calculate_chem_pots(nacl, [na])
    assert cp["Na"] == pytest.approx(-1.3)
    assert cp["Cl"] == pytest.approx(-4.0 + 1.3)


def test_get_interface_energy_runs(cu_bulk, ni_bulk):
    from interfacer.builder import slab_aligner
    b1 = _with_emt(cu_bulk.copy()); b2 = _with_emt(ni_bulk.copy())
    s1, s2 = slab_aligner(make_slab((1, 1, 1), cu_bulk, repeat=(1, 1, 2)), make_slab((1, 1, 1), ni_bulk, repeat=(1, 1, 2)),
                          3, 20, 2, 4, 7, cu_bulk, ni_bulk)
    E1, E2, Ws1, Ws2, inter = get_interface_energy(s1, s2, b1.get_potential_energy() / len(b1), b2.get_potential_energy() / len(b2),
                                                   dist=2.0, calc=EMT())
    assert Ws1 > 0 and Ws2 > 0
    assert np.isfinite(inter.info["W_ad"])

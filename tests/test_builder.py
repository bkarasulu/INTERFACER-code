import numpy as np
import pytest
from ase.build import bulk

from interfacer.builder import (check_surf_unique, cut_cell, find_commensurate_supercell, get_cut_vectors,
                           make_slab, permute_surface, RH, slab_aligner, slab_thickness, standardise_slab)
from interfacer.analysis import misfit, surf_area


def test_tools_compat_shims():
    # the old 'tools' modules still work (previously they imported each other circularly)
    import importlib
    for name in ("tools.builder", "tools.analysis", "tools.external"):
        importlib.import_module(name)
    from tools.builder import make_slab as m1
    from interfacer.builder import make_slab as m2
    from tools.external import call_vasp, call_castep, call_ase  # noqa: F401
    assert m1 is m2


@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)])
def test_get_cut_vectors_lie_in_plane(cu_bulk, miller):
    d, e, f, origin = get_cut_vectors(miller, cu_bulk.cell)
    # in-plane vectors (in lattice coordinates) are perpendicular to the plane normal (h,k,l)
    assert np.dot(d, miller) == pytest.approx(0)
    assert np.dot(e, miller) == pytest.approx(0)


@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 0), (1, 1, 1)])
def test_make_slab_orientation_and_stoichiometry(cu_bulk, miller):
    orig = cu_bulk.copy()
    slab = make_slab(miller, cu_bulk, repeat=(1, 1, 2))
    # surface normal along z, right-handed cell, a and b in the xy-plane
    assert abs(slab.cell[0][2]) < 1e-8 and abs(slab.cell[1][2]) < 1e-8
    assert np.dot(slab.cell[2], np.cross(slab.cell[0], slab.cell[1])) > 0
    # bulk number density is preserved
    assert len(slab) / slab.get_volume() == pytest.approx(len(cu_bulk) / cu_bulk.get_volume(), rel=1e-6)
    # the input structure is not modified
    assert np.allclose(cu_bulk.positions, orig.positions)


def test_cut_cell_wraps_and_keeps_input(cu_bulk):
    atoms = cu_bulk.copy()
    atoms.positions += [5.0, -7.0, 11.0]  # coordinates outside the cell
    before = atoms.positions.copy()
    out = cut_cell(atoms, (1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert len(out) == len(cu_bulk)
    assert np.allclose(atoms.positions, before)


def test_check_surf_unique_detects_duplicates_only():
    a = bulk("Cu", "fcc", a=3.61, cubic=True)
    b = bulk("Cu", "bcc", a=2.87, cubic=True).repeat((2, 1, 1))  # same formula (Cu4), other symmetry
    b.rattle(0.05, seed=1)
    assert check_surf_unique([], a, 5) == 0
    assert check_surf_unique([a], a.copy(), 5) == 1   # duplicate
    assert check_surf_unique([a], b, 5) == 0          # same formula, different symmetry -> unique


def test_RH_only_swaps_positions_with_cell(cu_bulk):
    atoms = cu_bulk.copy()
    pos = atoms.positions.copy()
    out = RH(atoms)  # already right-handed: nothing should change
    assert np.allclose(out.positions, pos)


def test_standardise_slab_puts_normal_along_z():
    from ase.build import fcc111
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=5.0)
    slab.rotate(30, "x", rotate_cell=True)
    slab.new_array("bulk_wyckoff", np.array(["a"] * len(slab)))
    out = standardise_slab(slab)
    n = np.cross(out.cell[0], out.cell[1])
    assert np.allclose(n[:2], 0, atol=1e-8) and n[2] > 0
    assert "bulk_wyckoff" not in out.arrays
    assert slab_thickness(out) == pytest.approx(slab_thickness(slab))


def test_permute_surface_gives_standard_symmetric_slabs(cu_bulk):
    slabs = permute_surface(cu_bulk, (1, 1, 1), 5.0, vacuum=5.0)
    assert len(slabs) >= 1
    for s in slabs:
        assert abs(s.cell[0][2]) < 1e-6 and abs(s.cell[1][2]) < 1e-6
        assert slab_thickness(s) >= 5.0


def test_slab_aligner_cu_ni_111(cu_bulk, ni_bulk):
    s1 = make_slab((1, 1, 1), cu_bulk, repeat=(1, 1, 1))
    s2 = make_slab((1, 1, 1), ni_bulk, repeat=(1, 1, 1))
    s1_orig = s1.copy()
    a1, a2 = slab_aligner(s1, s2, 3, 20, 2, 4, 7, cu_bulk, ni_bulk)
    # aligned cells have the same in-plane angles and lengths within the tolerance
    assert misfit(a1, a2) < 0.04
    assert np.allclose(a1.cell.cellpar()[[3, 4, 5]], a2.cell.cellpar()[[3, 4, 5]], atol=0.5)
    assert np.allclose(a1.cell.lengths()[:2], a2.cell.lengths()[:2], rtol=0.04)
    # no clashes
    for a in (a1, a2):
        d = a.get_all_distances(mic=True)
        np.fill_diagonal(d, 10)
        assert d.min() > 2.0
    # the input slab is not modified
    assert np.allclose(s1.positions, s1_orig.positions)


def test_find_commensurate_supercell_raises_instead_of_exiting(cu_bulk):
    s1 = make_slab((1, 1, 1), cu_bulk)
    s2 = make_slab((1, 0, 0), bulk("Na", "bcc", a=4.29, cubic=True))
    with pytest.raises(RuntimeError):
        find_commensurate_supercell(s1, s2, 3, 4, 1, 0.01)

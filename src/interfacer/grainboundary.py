"""Grain-boundary (GB) models: symmetric tilt and twist boundaries from one bulk structure.

* Symmetric tilt GB: the grain on top is the mirror image of the grain below across the boundary plane (hkl). For a
  plane that is not a mirror plane of the crystal this is an exact coincidence-site-lattice (CSL) bicrystal, e.g.
  (310) of a cubic crystal gives the Sigma5 (310)[001] symmetric tilt boundary.
* Twist GB: the upper grain is rotated about the plane normal by the CSL angle defined by two integers (m, n): the
  lattice vectors m*a + n*b of the lower grain and n*a + m*b of the upper grain coincide. Implemented for square
  (e.g. cubic (100)) and hexagonal (e.g. fcc (111)) planes; Sigma = (m^2 + n^2)/g (square) or (m^2 + m n + n^2)/g
  (hexagonal), g = gcd-related reduction.
* Asymmetric boundaries (two different planes of the same crystal) are ordinary interfaces: use -i1 and -i2 with the
  same structure and different Miller indices.

The bicrystals are periodic along the normal (two equivalent boundaries per cell) when the vacuum is half the
separation (default). GB energy: gamma_GB = (E_bicrystal - N E_bulk/atom) / (2 A).
"""
import math

import numpy as np

from .log import debug

from .builder import atomic_planes, cut_cell, make_interface, make_slab, orient_up, square_slab


def plane_spacing(slab, tol=0.3):
    """Typical spacing between atomic planes of a slab (median gap along the normal), used as default GB separation."""
    _, h = atomic_planes(slab, tol)
    gaps = np.diff(np.sort(h))
    gaps = gaps[gaps > 1e-3]
    return float(np.median(gaps)) if len(gaps) else 2.0


def nn_distance(bulk):
    """Nearest-neighbour distance (A) of a bulk structure."""
    from ase.neighborlist import neighbor_list
    d = neighbor_list("d", bulk, 6.0)
    return float(d.min()) if len(d) else 2.5


def remove_overlaps(gb, candidates, min_dist):
    """Remove atoms (only from the index list candidates) closer than min_dist (A) to an atom that is kept; used for
    the near-coincident sites of a CSL boundary. Returns (atoms, number of removed atoms)."""
    d = gb.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    removed = set()
    for i in sorted(int(i) for i in candidates):
        if any(j not in removed for j in np.where(d[i] < min_dist)[0]):
            removed.add(i)
    return gb[[i for i in range(len(gb)) if i not in removed]], len(removed)


def symmetric_tilt_gb(bulk, miller, creps=2, expansion=0.0, tol=0.3, overlap=0.7):
    """Symmetric tilt bicrystal for the boundary plane (hkl): grain 2 is the mirror image of grain 1 through its top
    atomic plane (the boundary plane is shared by both grains).

    The cell along the normal is twice the grain thickness, so that the periodic image forms the second, equivalent
    boundary through the bottom atomic plane. For a plane that is a mirror plane of the crystal (e.g. cubic (100)) this
    gives the perfect crystal. expansion (A): extra rigid-body separation added at each boundary.
    For planes with closely spaced atomic planes (e.g. fcc (310)) the mirror puts atoms of the two grains on
    near-coincident sites: atoms of grain 2 closer than overlap x the bulk nearest-neighbour distance to another atom
    are removed (the usual CSL construction; overlap=0 keeps all atoms). Returns (bicrystal, grain1, grain2)."""
    import ase
    g1 = square_slab(make_slab(tuple(miller), bulk, repeat=(1, 1, creps)))
    g1.center(vacuum=0.0, axis=2)
    planes, h = atomic_planes(g1, tol)
    z_bot, z_top = h[0], g1.positions[planes[-1], 2].mean()
    g2 = g1.copy()
    g2.positions[:, 2] = 2 * z_top - g2.positions[:, 2] + expansion
    drop = set(planes[-1].tolist()) | set(planes[0].tolist())   # boundary planes (shared with grain 1 / its image)
    g2 = g2[[i for i in range(len(g2)) if i not in drop]]
    c = 2 * (z_top - z_bot) + 2 * expansion
    gb = ase.Atoms(g1)
    gb.extend(g2)
    gb.set_cell([g1.cell[0], g1.cell[1], [0.0, 0.0, c]], scale_atoms=False)
    gb.pbc = True
    gb.wrap()
    if overlap > 0:
        gb, n_removed = remove_overlaps(gb, range(len(g1), len(gb)), overlap * nn_distance(bulk))
        if n_removed:
            debug("symmetric_tilt_gb: removed %d atoms on near-coincident sites at the boundaries" % n_removed)
    gb.info["gb"] = "symmetric tilt %s" % (tuple(miller),)
    return gb, g1, g2


def _lattice_type(slab, tol=1e-3):
    a, b = slab.cell[0][:2], slab.cell[1][:2]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    ang = math.degrees(math.acos(np.dot(a, b) / (la * lb)))
    if abs(la - lb) > tol * la:
        return None
    if abs(ang - 90) < 0.1:
        return "square"
    if abs(ang - 60) < 0.1 or abs(ang - 120) < 0.1:
        return "hexagonal"
    return None


def twist_gb(bulk, miller, m, n, creps=2, sep=None, vac=None):
    """Twist bicrystal with the CSL rotation defined by (m, n) for square or hexagonal planes.

    Returns (bicrystal, grain1, grain2, twist angle in degrees, Sigma)."""
    if m == n or m < 0 or n < 0 or (m == 0 and n == 0):
        raise ValueError("twist_gb: need non-negative m != n (not both zero)")
    base = make_slab(tuple(miller), bulk, repeat=(1, 1, creps))
    kind = _lattice_type(base)
    if kind is None:
        raise ValueError("twist_gb: the (%s) plane has neither a square nor a hexagonal lattice; use symmetric_tilt_gb "
                         "or an asymmetric boundary (-i1/-i2 with different Miller indices)" % (tuple(miller),))
    if kind == "square":     # rotation by 90 degrees: (m, n) -> (-n, m)
        A = [(m, n, 0), (-n, m, 0)]
        B = [(n, m, 0), (-m, n, 0)]
        sigma = m * m + n * n
    else:                    # 60-degree basis: rotation by 60 degrees: (m, n) -> (-n, m + n)
        if np.dot(base.cell[0], base.cell[1]) < 0:   # make the in-plane angle 60 degrees
            base = cut_cell(base, (1, 0, 0), (1, 1, 0), (0, 0, 1))
        A = [(m, n, 0), (-n, m + n, 0)]
        B = [(n, m, 0), (-m, n + m, 0)]
        sigma = m * m + m * n + n * n
    while sigma % 2 == 0 and kind == "square":
        sigma //= 2
    while sigma % 3 == 0 and kind == "hexagonal":
        sigma //= 3
    g1 = orient_up(square_slab(cut_cell(base, A[0], A[1], (0, 0, 1))))
    g2 = orient_up(square_slab(cut_cell(base, B[0], B[1], (0, 0, 1))))
    # rotate grain 2 so that its first cell vector coincides with grain 1's (the twist about the normal)
    v1, v2 = g1.cell[0][:2], g2.cell[0][:2]
    angle = math.degrees(math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0]))
    g2.rotate(angle, "z", rotate_cell=True)
    g2.set_cell([g1.cell[0], g1.cell[1], g2.cell[2]], scale_atoms=False)
    g2.wrap(pbc=[True, True, False])
    twist = math.degrees(math.acos(np.clip(np.dot(np.array(A[0][:2]) @ base.cell[:2, :2], np.array(B[0][:2]) @ base.cell[:2, :2])
                                           / (np.linalg.norm(np.array(A[0][:2]) @ base.cell[:2, :2]) ** 2), -1, 1)))
    period = 90.0 if kind == "square" else 60.0   # rotation symmetry of the plane: report the smallest equivalent angle
    twist = twist % period
    twist = min(twist, period - twist)
    if sep is None:
        sep = plane_spacing(base)
    gb = make_interface(g1, g2, sep, sep / 2 if vac is None else vac, reorder=False)
    gb.info["gb"] = "twist %s Sigma%d %.2f deg" % (tuple(miller), sigma, twist)
    return gb, g1, g2, twist, sigma


def gb_energy(E_bicrystal, n_atoms, E_bulk_per_atom, area, n_boundaries=2):
    """Grain-boundary energy (eV/A^2): (E_bicrystal - N E_bulk/atom) / (n_boundaries A)."""
    return (E_bicrystal - n_atoms * E_bulk_per_atom) / (n_boundaries * area)

"""Shared ASE geometry optimiser used by all ASE-based engines (classical potentials and, later, ML potentials)."""
import os

import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE, LBFGS

OPTIMISERS = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}


def get_optimiser(name):
    """Return the ASE optimiser class for a name (BFGS, FIRE or LBFGS, case-insensitive)."""
    try:
        return OPTIMISERS[name.upper()]
    except KeyError:
        raise ValueError("Unknown optimiser '%s'; choose from %s" % (name, ", ".join(OPTIMISERS)))


def bottom_layer_indices(atoms, n_layers, tol=0.3):
    """Indices of the atoms in the lowest `n_layers` atomic layers along the surface normal (a x b).

    Atoms whose heights differ by less than `tol` (A) belong to the same layer.
    """
    if n_layers <= 0:
        return []
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    normal = normal / np.linalg.norm(normal)
    h = atoms.positions @ normal
    order = np.argsort(h)
    layers = []
    for i in order:
        if layers and abs(h[i] - h[layers[-1][0]]) < tol:
            layers[-1].append(i)
        else:
            layers.append([i])
    return sorted(int(i) for layer in layers[:n_layers] for i in layer)


def relax(atoms, calc, fmax=0.05, steps=200, optimiser="BFGS", relax_cell=False, fix_bottom_layers=0,
          trajectory=None, logfile="-", cell_mask=None):
    """Relax `atoms` in place with an ASE optimiser and return (energy, atoms).

    Args:
        atoms (ase.Atoms): structure to relax (modified in place).
        calc: ASE calculator to attach.
        fmax (float): force convergence criterion (eV/A).
        steps (int): maximum number of optimisation steps.
        optimiser (str): BFGS, FIRE or LBFGS.
        relax_cell (bool): also relax the cell (FrechetCellFilter, e.g. for bulk structures).
        cell_mask (list): Voigt mask of the cell degrees of freedom to relax (xx, yy, zz, yz, xz, xy), e.g.
            [0, 0, 1, 0, 0, 0] to relax only the length along z; implies relax_cell. Default: all.
        fix_bottom_layers (int): number of bottom atomic layers to keep fixed (for slabs).
        trajectory (str): optional trajectory file name.
    """
    atoms.calc = calc
    constraints_before = atoms.constraints
    if fix_bottom_layers:
        atoms.set_constraint(constraints_before + [FixAtoms(indices=bottom_layer_indices(atoms, fix_bottom_layers))])

    target = atoms
    if relax_cell or cell_mask is not None:
        from ase.filters import FrechetCellFilter
        target = FrechetCellFilter(atoms, mask=cell_mask)

    if trajectory:
        d = os.path.dirname(trajectory)
        if d:
            os.makedirs(d, exist_ok=True)
    opt = get_optimiser(optimiser)(target, trajectory=trajectory, logfile=logfile)
    try:
        converged = opt.run(fmax=fmax, steps=steps)
    except Exception as err:
        print("relax: WARNING: optimisation failed (returning the energy of the last geometry): %s" % err)
        converged = False
    if not converged:
        print("relax: WARNING: not converged to fmax=%.3f eV/A within %d steps" % (fmax, steps))
    if fix_bottom_layers:
        atoms.set_constraint(constraints_before)
    return atoms.get_potential_energy(), atoms

"""Thin wrappers around spglib that work with old and new spglib versions.

spglib >= 2.5 deprecates its old error handling (returning None and printing to stderr) in favour of exceptions, and
its datasets became objects instead of dicts. All INTERFACER symmetry calls go through these functions.
"""
import contextlib

import spglib


def _cell(atoms):
    return (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)


@contextlib.contextmanager
def _new_error_handling():
    err = getattr(spglib, "error", None)
    old = getattr(err, "OLD_ERROR_HANDLING", None) if err is not None else None
    if old is not None:
        err.OLD_ERROR_HANDLING = False
    try:
        yield
    finally:
        if old is not None:
            err.OLD_ERROR_HANDLING = old


def _spglib_errors():
    err = getattr(spglib, "error", None)
    exc = getattr(err, "SpglibError", None) if err is not None else None
    return (exc,) if exc is not None else ()


def _call(func, *args, **kw):
    """Call a spglib function; return None if spglib fails (as with the old error handling)."""
    with _new_error_handling():
        try:
            return func(*args, **kw)
        except _spglib_errors() or (RuntimeError,):
            return None


def _get(dataset, key):
    if dataset is None:
        return None
    return getattr(dataset, key) if hasattr(dataset, key) else dataset[key]


def spacegroup(atoms, symprec=1e-4):
    """International symbol and number of the space group, e.g. 'Pm-3m (221)', or None."""
    ds = _call(spglib.get_symmetry_dataset, _cell(atoms), symprec=symprec)
    if ds is None:
        return None
    return "%s (%d)" % (_get(ds, "international"), _get(ds, "number"))


def symmetry_rotations(atoms, symprec=1e-5):
    """Rotation matrices of the symmetry operations (array of 3x3), or None."""
    sym = _call(spglib.get_symmetry, _cell(atoms), symprec=symprec)
    return None if sym is None else _get(sym, "rotations")


def primitive_cell(atoms, symprec=1e-4):
    """(lattice, scaled_positions, numbers) of the primitive cell (like spglib.find_primitive), or None."""
    return _call(spglib.standardize_cell, _cell(atoms), to_primitive=True, no_idealize=False, symprec=symprec)

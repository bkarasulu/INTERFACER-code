"""INTERFACER: build and analyse solid-solid interfaces from two bulk structures.

Main modules:
    interfacer.builder      slab construction, symmetric slabs, slab alignment, interface stacking
    interfacer.analysis     surface areas, misfit, surface formation energies, chemical potentials
    interfacer.calculators  engines for VASP, CASTEP and ASE calculators (classical and machine-learning potentials)
    interfacer.convergence  surface stability, layer thickness, separation and lateral translation tests
    interfacer.cli          the command-line workflow (``interfacer`` command)
"""
__version__ = "1.0.0.dev0"

from .analysis import calculate_chem_pots, calculate_SFE, find_prim, get_fu, misfit, surf_area  # noqa: F401
from .builder import make_interface, make_slab, permute_surface, slab_aligner, standardise_slab  # noqa: F401
from .calculators import AseEngine, CalcSettings, get_engine  # noqa: F401

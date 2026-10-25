"""Calculation engines: VASP, CASTEP and ASE calculators behind a common interface.

Use :func:`get_engine` to obtain an engine for a code name, or build an :class:`AseEngine` around any ASE calculator.
The original driver functions (call_vasp, call_castep, call_ase, ...) are available as well.
"""
from .base import CalcSettings, Engine
from .castep import call_castep, parseCASTEP, parseCASTEP_ASE
from .classical import POTENTIALS, call_ase, call_dftb, classical_calculator
from .engines import ENGINES, AseEngine, CastepEngine, GpawEngine, OnetepEngine, VaspEngine, get_engine
from .mlip import MLIPNotAvailable, available_mlips, mlip_calculator
from .optimise import OPTIMISERS, bottom_layer_indices, fixed_layer_indices, get_optimiser, middle_layer_indices, relax
from .vasp import (call_vasp, call_vasp_v2, get_potcar, handle_magmoms, make_potcar, set_hubU, vasp_continue)

__all__ = [
    "CalcSettings", "Engine", "AseEngine", "CastepEngine", "VaspEngine", "GpawEngine", "OnetepEngine", "get_engine", "ENGINES",
    "mlip_calculator", "available_mlips", "MLIPNotAvailable",
    "relax", "get_optimiser", "bottom_layer_indices", "middle_layer_indices", "fixed_layer_indices", "OPTIMISERS",
    "call_vasp", "call_vasp_v2", "call_castep", "call_ase", "call_dftb", "classical_calculator", "POTENTIALS",
    "parseCASTEP", "parseCASTEP_ASE", "handle_magmoms", "set_hubU", "get_potcar", "make_potcar", "vasp_continue",
]

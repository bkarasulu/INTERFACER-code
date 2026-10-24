"""Backward-compatible module with all former tools/external.py functions (now in interfacer.calculators)."""
from .calculators.castep import *  # noqa: F401,F403
from .calculators.castep import call_castep, parseCASTEP, parseCASTEP_ASE  # noqa: F401
from .calculators.classical import *  # noqa: F401,F403
from .calculators.classical import call_ase, call_dftb, classical_calculator  # noqa: F401
from .calculators.vasp import *  # noqa: F401,F403
from .calculators.vasp import (LDAU_L, _ldau_lists, _vasp_cmd, call_vasp, call_vasp_v2, get_potcar,  # noqa: F401
                               handle_magmoms, make_potcar, set_hubU, vasp_continue)

"""Backward-compatibility shim: use interfacer.calculators (or interfacer.external) instead."""
import tools  # noqa: F401  (makes the interfacer package importable)
from interfacer.external import *  # noqa: F401,F403
from interfacer.external import _ldau_lists, _vasp_cmd  # noqa: F401

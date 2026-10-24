"""Backward-compatibility package: the code now lives in the `interfacer` package (src/interfacer)."""
import os as _os
import sys as _sys

try:
    import interfacer as _interfacer  # noqa: F401
except ImportError:  # not installed: use the source tree
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

#!/usr/bin/env python
"""INTERFACER command-line script (formerly interfacer-new.py).

Equivalent to the ``interfacer`` command installed with ``pip install .``; all command-line options are unchanged.

This file has the same name as the ``interfacer`` package. Run from this folder, Python would find this file first for
``import interfacer`` (e.g. ``python -m pytest`` or ``python -c "import interfacer"``); it then loads the real package
in its place, so both uses work.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_package():
    """Import the interfacer package (installed, or src/ next to this script), skipping this file."""
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]
    shadow = sys.modules.pop("interfacer", None)
    try:
        try:
            import interfacer as pkg
        except ImportError:  # not installed: use the source tree next to this script
            sys.path.insert(0, os.path.join(_HERE, "src"))
            import interfacer as pkg
    except Exception:
        if shadow is not None:
            sys.modules["interfacer"] = shadow
        raise
    finally:
        sys.path[:] = saved
    sys.modules["interfacer"] = pkg
    return pkg


if __name__ == "__main__":
    _load_package()
    from interfacer.cli import main
    sys.exit(main())
else:  # imported as "interfacer" by mistake: replace this module by the package
    _load_package()

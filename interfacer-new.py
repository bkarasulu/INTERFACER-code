#!/usr/bin/env python
"""INTERFACER command-line script (kept for backward compatibility).

Equivalent to the ``interfacer`` command installed with ``pip install .``; all command-line options are unchanged.
"""
import os
import sys

try:
    from interfacer.cli import main
except ImportError:  # not installed: use the source tree next to this script
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from interfacer.cli import main

if __name__ == "__main__":
    sys.exit(main())

"""Backward-compatibility shim: use interfacer.analysis instead."""
import tools  # noqa: F401  (makes the interfacer package importable)
from interfacer.analysis import *  # noqa: F401,F403

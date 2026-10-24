"""Backward-compatibility shim: use interfacer.builder instead."""
import tools  # noqa: F401  (makes the interfacer package importable)
from interfacer.builder import *  # noqa: F401,F403

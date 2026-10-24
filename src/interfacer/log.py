"""Output of INTERFACER through the standard logging module (logger name: ``interfacer``).

Levels: normal output is INFO; details (supercell search, passivation, optimiser steps, ...) are DEBUG and only shown
with ``-v``; ``-q`` shows warnings only. Messages go to the current ``sys.stdout``.
"""
import logging
import sys

logger = logging.getLogger("interfacer")


class _StdoutHandler(logging.StreamHandler):
    """A StreamHandler that always writes to the current sys.stdout (works with redirections and test capturing)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):  # set by StreamHandler.__init__; the current sys.stdout is always used
        pass


def setup_logging(verbosity=0):
    """Configure the output: verbosity < 0 -> warnings only, 0 -> normal output, > 0 -> also debugging details."""
    level = logging.WARNING if verbosity < 0 else (logging.INFO if verbosity == 0 else logging.DEBUG)
    if not any(isinstance(h, _StdoutHandler) for h in logger.handlers):
        h = _StdoutHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_logging():
    """Set up the default output if nothing has been configured yet (e.g. when INTERFACER is used as a library)."""
    if not logger.handlers:
        setup_logging(0)


def verbose():
    """True if debugging details are shown (-v)."""
    return logger.isEnabledFor(logging.DEBUG)


def _join(args):
    return " ".join(str(a) for a in args)


def say(*args):
    """Normal output (like print)."""
    ensure_logging()
    logger.info(_join(args))


def debug(*args):
    """Details, shown only with -v."""
    ensure_logging()
    logger.debug(_join(args))


def warn(*args):
    """Warnings (always shown)."""
    ensure_logging()
    logger.warning(_join(args))

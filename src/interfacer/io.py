"""Output helpers: writing structures in the format of the code used, and keeping the previous OUTPUT directory."""
import os
import shutil

import ase.io

from .log import say

#: file extension and ase.io.write keywords per code
FORMATS = {
    "castep": ("cell", {"format": "castep-cell"}),
    "vasp": ("vasp", {"format": "vasp", "vasp5": True}),
}
DEFAULT_FORMAT = ("xyz", {"format": "extxyz"})


def struct_format(engine_or_name):
    """(extension, ase.io.write kwargs) for an engine or a code name; extxyz for ASE-based engines."""
    if hasattr(engine_or_name, "ext"):
        return engine_or_name.ext, dict(engine_or_name.write_kwargs)
    return FORMATS.get(str(engine_or_name).lower(), DEFAULT_FORMAT)


def write_struct(atoms, base, engine="ase", outdir="OUTPUT", trajectory=False, append=False):
    """Write a structure to <outdir>/<base>.<ext> in the format of the code used (cell/vasp/xyz). Returns the file name.

    trajectory=True writes a multi-frame file (XDATCAR format for VASP); append=True adds a frame to it.
    """
    ext, kw = struct_format(engine)
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "%s.%s" % (base, ext))
    if trajectory:
        if kw["format"] == "vasp":
            kw = {"format": "vasp-xdatcar"}
        kw["append"] = append
    ase.io.write(fname, atoms, **kw)
    return fname


def backup_output(outdir="OUTPUT"):
    """Move an existing output directory to <outdir>_prev (replacing an older backup) and create an empty one."""
    if os.path.exists(outdir):
        say("Moving previous output structures from %s to %s_prev..." % (outdir, outdir))
        shutil.rmtree(outdir + "_prev", ignore_errors=True)
        shutil.move(outdir, outdir + "_prev")
    os.makedirs(outdir, exist_ok=True)

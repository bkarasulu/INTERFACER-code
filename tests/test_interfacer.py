import argparse
import os
import subprocess
import sys

import numpy as np
import pytest
from ase.build import bulk

from conftest import ROOT
from interfacer.builder import make_interface, make_slab, slab_aligner
from interfacer.cli import parse_miller
from interfacer.convergence import cfactor


@pytest.mark.parametrize("text,expected", [
    ("111", (1, 1, 1)), ("1-10", (1, -1, 0)), ("1,1,0", (1, 1, 0)), ("(1,0,0)", (1, 0, 0)), ("[2 1 0]", (2, 1, 0)),
])
def test_parse_miller(text, expected):
    assert parse_miller(text) == expected


def test_parse_miller_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_miller("11")


def test_cfactor_ev_per_nm2_to_J_per_m2():
    assert cfactor == pytest.approx(1.602176634e-19 / 1e-18, rel=1e-6)


def test_make_interface_gap_equals_sep():
    cu = bulk("Cu", "fcc", a=3.61, cubic=True)
    ni = bulk("Ni", "fcc", a=3.52, cubic=True)
    s1, s2 = slab_aligner(make_slab((1, 0, 0), cu), make_slab((1, 0, 0), ni), 3, 20, 2, 4, 7, cu, ni)
    for sep in (1.0, 2.0, 3.0):
        inter = make_interface(s1, s2, sep, vac=1.0, reorder=False)
        z = inter.positions[:, 2]
        n1 = len(s1)
        assert z[n1:].min() - z[:n1].max() == pytest.approx(sep, abs=1e-6)
        assert inter.get_chemical_symbols()[:n1] == s1.get_chemical_symbols()


def _run(args, cwd, timeout=600):
    # the backward-compatible script; it uses src/ when the package is not installed
    return subprocess.run([sys.executable, os.path.join(ROOT, "interfacer-new.py")] + args, cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def cu_ni_inputs(tmp_path):
    import ase.io
    ase.io.write(tmp_path / "Cu.cif", bulk("Cu", "fcc", a=3.61, cubic=True))
    ase.io.write(tmp_path / "Ni.cif", bulk("Ni", "fcc", a=3.52, cubic=True))
    return tmp_path


def test_quick_run_end_to_end(cu_ni_inputs):
    r = _run(["-i1", "Cu.cif", "-i2", "Ni.cif", "-prog", "ase", "-quick", "-tsep", "-thor",
              "-sep_init", "1.5", "-sep_final", "2.5", "-sep_step", "0.5", "-hor_steps", "0.5", "0.5"], cu_ni_inputs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = cu_ni_inputs / "OUTPUT"
    for f in ("bulk1_opted.xyz", "interface_cuts_0_0_opted.xyz", "interface_cuts_0_0_opted.cif",
              "int-sep_2.50A.xyz", "htrans_x0.50-y0.00.xyz", "conv_sep.png", "conv_htrans.png"):
        assert (out / f).exists(), f
    assert "W_ad" in (cu_ni_inputs / "interfaces.out").read_text()
    # previous OUTPUT is kept as OUTPUT_prev on a second run
    r = _run(["-i1", "Cu.cif", "-i2", "Ni.cif", "-prog", "ase", "-quick", "-surf_only"], cu_ni_inputs)
    assert r.returncode == 0
    assert (cu_ni_inputs / "OUTPUT_prev" / "interface_cuts_0_0_opted.xyz").exists()


def test_emt_surface_screening_and_adhesion(cu_ni_inputs):
    r = _run(["-i1", "Cu.cif", "-i2", "Ni.cif", "-prog", "ase", "-ase_pot", "EMT", "-ase_gsteps", "100",
              "-tsur", "-mlist", "100", "110", "111", "-cr1", "3", "-cr2", "3", "-msd", "20"], cu_ni_inputs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Selected surfaces: (1, 1, 1) for material 1 and (1, 1, 1) for material 2" in r.stdout
    wad = [ln for ln in r.stdout.splitlines() if ln.startswith("W_ad")][-1]
    wad_J = float(wad.split("=")[-1].split()[0])
    # EMT surface energies of Cu(111) and Ni(111) are ~1.0 and ~1.7 J/m^2; W_ad must not exceed their sum
    assert 1.5 < wad_J < 2.8


def test_vasp_dry_run(cu_ni_inputs, fake_potcars):
    r = _run(["-i1", "Cu.cif", "-i2", "Ni.cif", "-m1", "1", "1", "1", "-m2", "1", "1", "1", "-prog", "vasp", "-dry",
              "-kp_gr", "4 4 1", "-mpi", "-np", "4", "-msd", "20"], cu_ni_inputs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "VASP command:  mpirun -n 4 vasp_std" in r.stdout
    kp = (cu_ni_inputs / "interface_cuts_0_0" / "KPOINTS").read_text().split()
    assert "Gamma" in kp and kp[-6:-3] == ["4", "4", "1"]
    assert (cu_ni_inputs / "OUTPUT" / "interface_cuts_0_0_opted.vasp").exists()


def test_cli_main_in_process_with_new_options(cu_ni_inputs, monkeypatch):
    from interfacer.cli import main
    monkeypatch.chdir(cu_ni_inputs)
    rc = main(["-i1", "Cu.cif", "-i2", "Ni.cif", "-m1", "1", "1", "1", "-m2", "1", "1", "1", "-cr1", "3", "-cr2", "3",
               "-prog", "ase", "-ase_pot", "EMT", "-ase_gsteps", "50", "--optimiser", "FIRE", "--relax_cell_bulk",
               "--fix_layers", "1", "--outdir", "results", "-msd", "20"])
    assert rc == 0
    assert (cu_ni_inputs / "results" / "interface_cuts_0_0_opted.cif").exists()
    import ase.io
    b = ase.io.read(cu_ni_inputs / "results" / "bulk1_opted.xyz")
    # EMT equilibrium lattice constant of Cu is ~3.59 A (the input was 3.61 A): the cell was relaxed
    assert b.cell.lengths()[0] == pytest.approx(3.59, abs=0.015)

"""Run the tutorials (the MACE ones only when mace-torch and its model are available and INTERFACER_SLOW_TESTS=1)."""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT

TUT = os.path.join(ROOT, "Tutorials")


def _copy(name, tmp_path):
    dst = tmp_path / name
    shutil.copytree(os.path.join(TUT, name), dst)
    return dst


def _interfacer(args, cwd, timeout=1800):
    cmd = [sys.executable, os.path.join(ROOT, "interfacer-new.py")] + args
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def _wad(out):
    return float([ln for ln in out.splitlines() if ln.startswith("W_ad")][-1].split("=")[-1].split()[0])


def test_tutorial_01_cu_ni_emt(tmp_path):
    d = _copy("01_Cu-Ni_EMT", tmp_path)
    out = _interfacer(["--input", "input.toml"], d)
    assert "Selected surfaces: (1, 1, 1) for material 1 and (1, 1, 1) for material 2" in out
    assert _wad(out) == pytest.approx(2.64, abs=0.05)
    assert (d / "OUTPUT" / "conv_htrans.png").exists()


def test_tutorial_05_python_api(tmp_path):
    d = _copy("05_python_api", tmp_path)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "python_api.py"], cwd=d, capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "workflow result:" in r.stdout


def test_tutorial_inputs_parse():
    from interfacer.cli import apply_input_file, build_parser
    for t, f in (("01_Cu-Ni_EMT", "input.toml"), ("02_Na3OCl-Na_MACE", "input.toml"), ("03_symmetric_slabs_SFE", "input.toml"),
                 ("beta_Li3PS4-Li", "input_vasp.toml"), ("beta_Li3PS4-Li", "input_mace_prescreen.toml")):
        p = build_parser()
        apply_input_file(p, os.path.join(TUT, t, f))
        args = p.parse_args([])
        for infile in (args.infile1, args.infile2):
            assert os.path.exists(os.path.join(TUT, t, infile)), (t, infile)


slow = pytest.mark.skipif(os.environ.get("INTERFACER_SLOW_TESTS") != "1", reason="set INTERFACER_SLOW_TESTS=1 to run")


@slow
@pytest.mark.parametrize("name", ["02_Na3OCl-Na_MACE", "03_symmetric_slabs_SFE"])
def test_mace_tutorials(name, tmp_path):
    pytest.importorskip("mace.calculators")
    d = _copy(name, tmp_path)
    out = _interfacer(["--input", "input.toml", "--device", "cpu"], d)
    assert _wad(out) > 0

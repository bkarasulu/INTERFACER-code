"""Tests of the machine-learning potential layer.

The backends are tested with small fake modules (so that the packages need not be installed); a real MACE run is
done when mace-torch is installed and its foundation model can be downloaded.
"""
import sys
import types

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from interfacer.calculators import CalcSettings, get_engine
from interfacer.calculators import mlip
from interfacer.calculators.mlip import MLIPNotAvailable, lazy_calculator_factory, mlip_calculator


class Recorder(EMT):
    """An EMT calculator that remembers how it was created."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.created_with = (args, kwargs)


def _fake(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_mace_backend(monkeypatch):
    _fake(monkeypatch, "mace")
    _fake(monkeypatch, "mace.calculators", mace_mp=lambda **kw: Recorder(**kw), mace_off=lambda **kw: Recorder(off=True, **kw))
    calc = mlip_calculator("MACE-MP", model="medium-mpa-0", device="cpu")
    assert calc.created_with[1] == {"model": "medium-mpa-0", "device": "cpu", "default_dtype": "float64"}
    calc = mlip_calculator("mace", device="cpu", default_dtype="float32")
    assert calc.created_with[1]["model"] == "medium" and calc.created_with[1]["default_dtype"] == "float32"
    assert mlip_calculator("mace-off", device="cpu").created_with[1]["off"] is True


def test_chgnet_backend(monkeypatch):
    class CHGNet:
        @staticmethod
        def load(model_name):
            return ("loaded", model_name)

        @staticmethod
        def from_file(path):
            return ("file", path)
    _fake(monkeypatch, "chgnet")
    _fake(monkeypatch, "chgnet.model", CHGNet=CHGNet)
    _fake(monkeypatch, "chgnet.model.dynamics", CHGNetCalculator=lambda **kw: Recorder(**kw))
    assert mlip_calculator("chgnet", model="0.3.0", device="cpu").created_with[1] == {"model": ("loaded", "0.3.0"), "use_device": "cpu"}
    assert mlip_calculator("chgnet", device="cpu").created_with[1]["model"] is None


def test_matgl_backend(monkeypatch):
    _fake(monkeypatch, "matgl", load_model=lambda name: ("pot", name))
    _fake(monkeypatch, "matgl.ext")
    _fake(monkeypatch, "matgl.ext.ase", PESCalculator=lambda pot, **kw: Recorder(pot, **kw))
    assert mlip_calculator("m3gnet", device="cpu").created_with[0] == (("pot", "M3GNet-MP-2021.2.8-PES"),)


def test_sevennet_backend(monkeypatch):
    _fake(monkeypatch, "sevenn")
    _fake(monkeypatch, "sevenn.calculator", SevenNetCalculator=lambda **kw: Recorder(**kw))
    calc = mlip_calculator("7net", model="7net-mf-ompa", device="cpu", modal="mpa")
    assert calc.created_with[1] == {"model": "7net-mf-ompa", "device": "cpu", "modal": "mpa"}


def test_orb_backend(monkeypatch):
    loaded = {}

    def loader(device, **kw):
        loaded.update(device=device, **kw)
        return "orbff"
    _fake(monkeypatch, "orb_models")
    _fake(monkeypatch, "orb_models.forcefield")
    _fake(monkeypatch, "orb_models.forcefield.pretrained", ORB_PRETRAINED_MODELS={"orb-v3-conservative-inf-omat": loader})
    _fake(monkeypatch, "orb_models.forcefield.calculator", ORBCalculator=lambda model, **kw: Recorder(model, **kw))
    calc = mlip_calculator("orb", device="cpu", precision="float32-high")
    assert calc.created_with == (("orbff",), {"device": "cpu"}) and loaded == {"device": "cpu", "precision": "float32-high"}
    with pytest.raises(ValueError):
        mlip_calculator("orb", model="orb-v99", device="cpu")


def test_fairchem_backend(monkeypatch):
    pm = types.SimpleNamespace(get_predict_unit=lambda name, device: ("unit", name, device))
    _fake(monkeypatch, "fairchem")
    _fake(monkeypatch, "fairchem.core", pretrained_mlip=pm, FAIRChemCalculator=lambda unit, **kw: Recorder(unit, **kw))
    calc = mlip_calculator("uma", device="cpu")
    assert calc.created_with == ((("unit", "uma-s-1", "cpu"),), {"task_name": "omat"})


def test_nequip_backend(monkeypatch):
    class NequIPCalculator:
        @staticmethod
        def from_compiled_model(path, device, **kw):
            return Recorder(path, device=device, **kw)
    _fake(monkeypatch, "nequip")
    _fake(monkeypatch, "nequip.integrations")
    _fake(monkeypatch, "nequip.integrations.ase", NequIPCalculator=NequIPCalculator)
    assert mlip_calculator("nequip", model="m.nequip.pt2", device="cpu").created_with == (("m.nequip.pt2",), {"device": "cpu"})
    with pytest.raises(ValueError):
        mlip_calculator("nequip", device="cpu")  # a model file is required
    _fake(monkeypatch, "allegro")
    assert mlip_calculator("allegro", model="a.pt2", device="cpu").created_with[0] == ("a.pt2",)


def test_custom_backend(monkeypatch):
    _fake(monkeypatch, "my_pot", make=lambda **kw: Recorder(**kw))
    calc = mlip_calculator("custom:my_pot:make", model="x", cutoff=5.0)
    assert calc.created_with[1] == {"model": "x", "cutoff": 5.0}
    with pytest.raises(ValueError):
        mlip_calculator("custom:onlymodule")


def test_unknown_and_missing(monkeypatch):
    with pytest.raises(ValueError):
        mlip_calculator("nonexistent")
    monkeypatch.setitem(sys.modules, "sevenn.calculator", None)  # makes the import fail
    with pytest.raises(MLIPNotAvailable, match="pip install sevenn"):
        mlip_calculator("sevennet", device="cpu")


def test_resolve_device():
    assert mlip.resolve_device("cpu") == "cpu"
    assert mlip.resolve_device("auto") in ("cpu", "cuda")


def test_lazy_factory_loads_once(monkeypatch):
    n = {"calls": 0}

    def make(**kw):
        n["calls"] += 1
        return EMT()
    _fake(monkeypatch, "my_pot2", make=make)
    f = lazy_calculator_factory("custom:my_pot2:make")
    assert f() is f()
    assert n["calls"] == 1


def test_mlip_engine_keeps_results_per_structure(monkeypatch):
    _fake(monkeypatch, "my_pot3", make=lambda **kw: EMT())
    eng = get_engine("mlip", CalcSettings(mlip="custom:my_pot3:make", outdir=""))
    assert eng.name == "mlip"
    a = bulk("Cu", "fcc", a=3.6, cubic=True)        # 4 atoms
    b = bulk("Ni", "fcc", a=3.5)                     # 1 atom
    Ea, a = eng.single_point(a, "a")
    Eb, b = eng.single_point(b, "b")
    # each structure keeps its own results although the calculator is shared
    assert a.get_potential_energy() == pytest.approx(Ea) and len(a.get_forces()) == 4
    assert b.get_potential_energy() == pytest.approx(Eb)


def test_cli_mlip_options(monkeypatch):
    from interfacer.cli import build_parser, make_settings
    args = build_parser().parse_args(["-i1", "a", "-i2", "b", "-prog", "mlip", "--mlip", "sevennet", "--mlip_model", "7net-0",
                                      "--device", "cpu", "--mlip_kwargs", "{'modal': 'mpa'}"])
    s = make_settings(args)
    assert (s.mlip, s.mlip_model, s.device, s.mlip_kwargs, s.relax_cell_bulk) == ("sevennet", "7net-0", "cpu", {"modal": "mpa"}, True)
    args = build_parser().parse_args(["-i1", "a", "-i2", "b", "-prog", "mlip", "--no_relax_cell_bulk"])
    assert make_settings(args).relax_cell_bulk is False
    args = build_parser().parse_args(["-i1", "a", "-i2", "b", "-prog", "ase"])
    assert make_settings(args).relax_cell_bulk is False
    args = build_parser().parse_args(["-i1", "a", "-i2", "b", "-prog", "mlip", "--mlip_kwargs", "[1, 2]"])
    with pytest.raises(ValueError):
        make_settings(args)


def test_cli_reports_missing_package(tmp_path, monkeypatch, capsys):
    import ase.io
    from interfacer.cli import main
    monkeypatch.chdir(tmp_path)
    ase.io.write("Cu.cif", bulk("Cu", cubic=True))
    monkeypatch.setitem(sys.modules, "chgnet.model", None)
    with pytest.raises(SystemExit):
        main(["-i1", "Cu.cif", "-i2", "Cu.cif", "-prog", "mlip", "--mlip", "chgnet"])
    assert "pip install chgnet" in capsys.readouterr().err


def _mace_small():
    pytest.importorskip("mace.calculators")
    try:
        return mlip_calculator("mace", model="small", device="cpu")
    except Exception as err:  # e.g. no network access to download the model
        pytest.skip("MACE-MP small model not available: %s" % err)


def test_real_mace_cu_ni_interface(tmp_path, monkeypatch):
    calc = _mace_small()
    import ase.io
    from interfacer.cli import main
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.calc = calc
    assert np.isfinite(a.get_potential_energy())
    monkeypatch.chdir(tmp_path)
    ase.io.write("Cu.cif", bulk("Cu", "fcc", a=3.61, cubic=True))
    ase.io.write("Ni.cif", bulk("Ni", "fcc", a=3.52, cubic=True))
    rc = main(["-i1", "Cu.cif", "-i2", "Ni.cif", "-m1", "1", "1", "1", "-m2", "1", "1", "1", "-cr1", "3", "-cr2", "3",
               "-prog", "mlip", "--mlip", "mace", "--mlip_model", "small", "--device", "cpu", "-ase_gsteps", "100", "-msd", "20"])
    assert rc == 0
    out = (tmp_path / "interfaces.out").read_text()
    ws = [float(ln.split("=")[-1].split()[0]) for ln in out.splitlines() if ln.startswith("Wsurf")]
    wad = float([ln for ln in out.splitlines() if ln.startswith("W_ad")][-1].split("=")[-1].split()[0])
    assert 1.0 < ws[0] < 1.7 and 1.6 < ws[1] < 2.5   # Cu(111), Ni(111) surface energies (J/m^2)
    assert 0 < wad < ws[0] + ws[1]

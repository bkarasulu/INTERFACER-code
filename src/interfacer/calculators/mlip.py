"""Machine-learning interatomic potentials (MLIPs) through their ASE calculators.

Supported backends (each needs its own package; they are imported only when used):

=============  ===============================  ==================================================================
name           package (pip install ...)        model (``--mlip_model``)
=============  ===============================  ==================================================================
mace           mace-torch                       foundation model name ('small', 'medium', 'large', 'medium-mpa-0',
                                                'medium-omat-0', ...) or path/URL of a trained .model file
mace-off       mace-torch                       MACE-OFF model ('small', 'medium', 'large') or path (organic systems)
chgnet         chgnet                           version ('0.3.0', ...) or path of a model file
matgl          matgl                            pretrained model name (default 'M3GNet-MP-2021.2.8-PES') or path
sevennet       sevenn                           '7net-0', '7net-mf-ompa', ... or path of a checkpoint
orb            orb-models                       'orb-v3-conservative-inf-omat', 'orb-v2', ...
fairchem       fairchem-core                    'uma-s-1', ... (UMA models need a Hugging Face login); task via
                                                --mlip_kwargs "{'task_name': 'omat'}"
nequip         nequip                           path of a compiled (.nequip.pt2/.pth) or deployed (.pth) model
allegro        nequip-allegro                   as nequip
custom         (your own)                       --mlip custom:package.module:function -> function(**mlip_kwargs)
=============  ===============================  ==================================================================

Example::

    from interfacer.calculators.mlip import mlip_calculator
    calc = mlip_calculator("mace", model="medium-mpa-0", device="cpu")
"""
import importlib

#: name -> pip package(s) to install
INSTALL_HINTS = {
    "mace": "pip install mace-torch",
    "mace-off": "pip install mace-torch",
    "chgnet": "pip install chgnet",
    "matgl": "pip install matgl",
    "sevennet": "pip install sevenn",
    "orb": "pip install orb-models",
    "fairchem": "pip install fairchem-core   (UMA models also need: huggingface-cli login)",
    "nequip": "pip install nequip",
    "allegro": "pip install nequip-allegro",
}

#: alternative names
ALIASES = {"mace-mp": "mace", "mace_mp": "mace", "mace_off": "mace-off", "m3gnet": "matgl", "tensornet": "matgl",
           "7net": "sevennet", "sevenn": "sevennet", "orb-models": "orb", "uma": "fairchem", "fair-chem": "fairchem"}

#: default model per backend (None: the backend's own default / a model path is required)
DEFAULT_MODELS = {"mace": "medium", "mace-off": "medium", "chgnet": None, "matgl": "M3GNet-MP-2021.2.8-PES",
                  "sevennet": "7net-0", "orb": "orb-v3-conservative-inf-omat", "fairchem": "uma-s-1",
                  "nequip": None, "allegro": None}


class MLIPNotAvailable(ImportError):
    """Raised when the package of an MLIP backend is not installed."""


def _import(module, backend):
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise MLIPNotAvailable("The '%s' potential needs the package providing '%s' (%s). Install it with: %s"
                               % (backend, module, err, INSTALL_HINTS.get(backend, "pip install <package>"))) from err


def resolve_device(device="auto"):
    """'auto' -> 'cuda' if a GPU is available to PyTorch, else 'cpu'; other values are returned unchanged."""
    if device in (None, "", "auto"):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _is_path(model):
    import os
    return model is not None and (os.path.sep in str(model) or os.path.exists(str(model)) or str(model).startswith("http"))


# ---------------------------------------------------------------------------------------------------- backends
def _mace(model, device, **kw):
    calcs = _import("mace.calculators", "mace")
    kw.setdefault("default_dtype", "float64")  # recommended for geometry optimisation
    return calcs.mace_mp(model=model or DEFAULT_MODELS["mace"], device=device, **kw)


def _mace_off(model, device, **kw):
    calcs = _import("mace.calculators", "mace-off")
    kw.setdefault("default_dtype", "float64")
    return calcs.mace_off(model=model or DEFAULT_MODELS["mace-off"], device=device, **kw)


def _chgnet(model, device, **kw):
    cm = _import("chgnet.model", "chgnet")
    dyn = _import("chgnet.model.dynamics", "chgnet")
    if model is None:
        net = None  # CHGNetCalculator loads the default pretrained model
    elif _is_path(model):
        net = cm.CHGNet.from_file(model)
    else:
        net = cm.CHGNet.load(model_name=model)
    return dyn.CHGNetCalculator(model=net, use_device=device, **kw)


def _matgl(model, device, **kw):
    matgl = _import("matgl", "matgl")
    ext = _import("matgl.ext.ase", "matgl")
    pot = matgl.load_model(model or DEFAULT_MODELS["matgl"])
    return ext.PESCalculator(pot, **kw)


def _sevennet(model, device, **kw):
    sc = _import("sevenn.calculator", "sevennet")
    return sc.SevenNetCalculator(model=model or DEFAULT_MODELS["sevennet"], device=device, **kw)


def _orb(model, device, **kw):
    pretrained = _import("orb_models.forcefield.pretrained", "orb")
    calcmod = _import("orb_models.forcefield.calculator", "orb")
    name = model or DEFAULT_MODELS["orb"]
    table = getattr(pretrained, "ORB_PRETRAINED_MODELS", {})
    loader = table.get(name) or getattr(pretrained, name.replace("-", "_"), None)
    if loader is None:
        raise ValueError("Unknown ORB model '%s'. Available: %s" % (name, ", ".join(sorted(table)) or "see orb_models.forcefield.pretrained"))
    model_kw = {k: kw.pop(k) for k in ("precision",) if k in kw}
    return calcmod.ORBCalculator(loader(device=device, **model_kw), device=device, **kw)


def _fairchem(model, device, **kw):
    fc = _import("fairchem.core", "fairchem")
    task = kw.pop("task_name", "omat")  # inorganic materials
    predictor = fc.pretrained_mlip.get_predict_unit(model or DEFAULT_MODELS["fairchem"], device=device)
    return fc.FAIRChemCalculator(predictor, task_name=task, **kw)


def _nequip(model, device, backend="nequip", **kw):
    if model is None:
        raise ValueError("The %s potential needs a model file: --mlip_model path/to/model" % backend)
    try:  # nequip >= 0.7
        mod = importlib.import_module("nequip.integrations.ase")
        calc_cls = mod.NequIPCalculator
        if str(model).endswith((".pt2", ".nequip.pth")) or hasattr(calc_cls, "from_compiled_model"):
            return calc_cls.from_compiled_model(model, device=device, **kw)
    except ImportError:
        pass
    mod = _import("nequip.ase", backend)  # nequip < 0.7
    return mod.NequIPCalculator.from_deployed_model(model_path=model, device=device, **kw)


def _allegro(model, device, **kw):
    _import("allegro", "allegro")  # registers the Allegro model with nequip
    return _nequip(model, device, backend="allegro", **kw)


BACKENDS = {"mace": _mace, "mace-off": _mace_off, "chgnet": _chgnet, "matgl": _matgl, "sevennet": _sevennet,
            "orb": _orb, "fairchem": _fairchem, "nequip": _nequip, "allegro": _allegro}


def _custom(spec, **kw):
    #'custom:package.module:function' -> function(**kw)
    try:
        _, module, func = spec.split(":", 2)
    except ValueError:
        raise ValueError("Custom potentials are given as custom:package.module:function, got '%s'" % spec)
    obj = getattr(importlib.import_module(module), func)
    return obj(**kw)


def available_mlips():
    """Names of the supported backends."""
    return sorted(BACKENDS) + ["custom:<module>:<function>"]


def canonical_name(name):
    n = name.strip().lower()
    return ALIASES.get(n, n)


def mlip_calculator(name, model=None, device="auto", **kwargs):
    """Create the ASE calculator of a machine-learning potential.

    Args:
        name (str): backend name (see module docstring), or 'custom:package.module:function'.
        model (str): model name or path (backend default if None).
        device (str): 'cpu', 'cuda', 'mps' or 'auto'.
        **kwargs: passed to the backend's calculator (e.g. default_dtype for MACE, task_name for fairchem).
    """
    if name.lower().startswith("custom:"):
        if model is not None:
            kwargs.setdefault("model", model)
        return _custom(name, **kwargs)
    key = canonical_name(name)
    if key not in BACKENDS:
        raise ValueError("Unknown ML potential '%s'. Available: %s" % (name, ", ".join(available_mlips())))
    return BACKENDS[key](model, resolve_device(device), **kwargs)


def lazy_calculator_factory(name, model=None, device="auto", **kwargs):
    """A function that creates the calculator on its first call and returns the same (cached) calculator afterwards,
    so that the model is loaded only once per workflow."""
    cache = {}

    def factory():
        if "calc" not in cache:
            print("Loading the %s potential%s..." % (name, " (model: %s)" % model if model else ""))
            cache["calc"] = mlip_calculator(name, model=model, device=device, **kwargs)
        return cache["calc"]
    return factory

"""Engine implementations: VASP, CASTEP and ASE calculators (classical potentials; ML potentials plug in here too)."""
import glob
import hashlib
import json
import os

import numpy as np
from ase.constraints import FixAtoms

from .base import CalcSettings, Engine, backup_path
from .castep import call_castep
from .classical import classical_calculator
from .mlip import lazy_calculator_factory
from .optimise import fixed_layer_indices, relax
from .vasp import call_vasp
from ..log import say


class VaspEngine(Engine):
    """VASP through ASE (call_vasp). Each calculation runs in its own directory named after the calculation."""
    name = "vasp"
    ext = "vasp"
    write_kwargs = {"format": "vasp", "vasp5": True}
    tmpdir = "VASP-tmp"

    def run(self, atoms, typ, name, dipolCorr="None", FixCell=False, FixVol=0, passivate=None, nosymm=0, vac=0.0,
            KPgrid_=None, ase_opt=None, relax_cell=False, fix_bottom_layers=0):
        s = self.settings
        from ..builder import prepare_cell_for_vasp
        atoms = prepare_cell_for_vasp(atoms)  # right-handed cell, no negative z coordinates
        if (s.overwrite or s.restart is False) and os.path.isdir(name):
            # call_vasp reuses/continues an existing run in its directory; to start afresh, move the old run aside
            say("Existing run in %s moved to %s (--overwrite)" % (name, backup_path(name)))
        if fix_bottom_layers:  # written as selective dynamics in the POSCAR
            atoms.set_constraint(FixAtoms(indices=fixed_layer_indices(atoms, fix_bottom_layers, s.fix_mode)))
        return call_vasp(atoms, typ=typ, dipolCorr=dipolCorr, name=name, ENCUT=s.encut, KPgrid=KPgrid_, KPspacing=s.kpspacing,
                         hubU=s.hubU, FixCell=FixCell, FixVol=FixVol, xc=s.xc, passivate=passivate, nosymm=nosymm, vac=vac,
                         magmom=s.magmoms, sigma=s.sigma, exe=s.exe, dry=s.dry, slowConv=0, gamma=1)


class CastepEngine(Engine):
    """CASTEP through ASE (call_castep). Runs in ./CASTEP-tmp."""
    name = "castep"
    ext = "cell"
    write_kwargs = {"format": "castep-cell"}
    tmpdir = "CASTEP-tmp"

    def run(self, atoms, typ, name, dipolCorr="None", FixCell=False, FixVol=0, passivate=None, nosymm=0, vac=0.0,
            KPgrid_=None, ase_opt=None, relax_cell=False, fix_bottom_layers=0):
        s = self.settings
        seed = os.path.join(self.tmpdir, name)
        if os.path.exists(seed + ".castep"):
            if s.overwrite or s.restart is False:
                bak = backup_path(seed + ".castep")
                for f in glob.glob(seed + ".*"):
                    os.rename(f, f + "_" + os.path.basename(bak).rsplit("_", 1)[-1])
                say("Existing CASTEP output %s.castep moved aside (--overwrite)" % seed)
            else:
                x = read_finished_castep(seed, typ, atoms)
                if x is not None:
                    say("Finished CASTEP run %s.castep found, reading the results (use --overwrite to redo it)" % seed)
                    return x
        kpkw = {"KPgrid": KPgrid_} if KPgrid_ else {"KPspacing": s.kpspacing}
        fixlist = fixed_layer_indices(atoms, fix_bottom_layers, s.fix_mode) if fix_bottom_layers else []
        return call_castep(atoms, typ=typ, dipolCorr=dipolCorr, name=name, ENCUT=s.encut, PP=s.castep_pp, hubU=s.hubU,
                           FixCell=FixCell, FixList=fixlist, ifDryRun=s.dry, **kpkw)


def read_finished_castep(seed, typ, atoms):
    """Results of a finished CASTEP run (<seed>.castep containing 'Total time'), in the format of call_castep, or None."""
    from .castep import parseCASTEP
    try:
        with open(seed + ".castep") as f:
            text = f.read()
    except OSError:
        return None
    if "Total time" not in text:
        return None  # unfinished or crashed run: run again
    x = None
    if typ.lower() == "geom" and os.path.exists(seed + ".geom"):
        try:
            x = parseCASTEP(seed + ".geom", atoms=atoms.copy())
        except Exception:
            x = None
    if x is None:
        x = parseCASTEP(seed + ".castep", atoms=atoms.copy())
    return x


class AseEngine(Engine):
    """Any ASE calculator, with the shared ASE optimiser (see optimise.relax).

    Args:
        calculator_factory: a function returning a new ASE calculator (called for every calculation), or a calculator object
            (shared between calculations).
        label (str): name printed in the log (e.g. 'LJ', 'EMT', 'MACE').
        settings (CalcSettings): fmax, steps, optimiser, relax_cell_bulk and outdir are used.
    """
    name = "ase"

    def __init__(self, calculator_factory, label="ASE", settings=None):
        super().__init__(settings)
        self.calculator_factory = calculator_factory
        self.label = label

    def calculator_for(self, name, atoms, kpgrid=None):
        """The calculator for one calculation (subclasses for DFT codes use the name for the run directory)."""
        return self.get_calculator()

    def get_calculator(self):
        f = self.calculator_factory
        if isinstance(f, type) or (callable(f) and not hasattr(f, "get_potential_energy")):
            return f()  # a calculator class or a factory function: new calculator for each calculation
        return f  # a calculator object (shared)

    def run(self, atoms, typ, name, dipolCorr="None", FixCell=False, FixVol=0, passivate=None, nosymm=0, vac=0.0,
            KPgrid_=None, ase_opt=None, relax_cell=False, fix_bottom_layers=0):
        if typ.lower() not in ("sp", "geom"):
            raise ValueError("typ must be 'sp' or 'geom', got '%s'" % typ)
        s = self.settings
        opt = (typ.lower() == "geom") if ase_opt is None else ase_opt
        cell = bool(opt and relax_cell and s.relax_cell_bulk)
        # Without FixCell (i.e. interfaces; slabs use FixCell=True and bulks relax_cell=True), the cell length along z
        # is relaxed as well, keeping the matched in-plane cell: in a periodic interface cell this lets the second
        # interface (across the periodic boundary) relax (VASP relaxes the whole cell, ISIF=3).
        mask = [0, 0, 1, 0, 0, 0] if (opt and not FixCell and not relax_cell) else None

        key = self._fingerprint(atoms, opt, cell, mask, fix_bottom_layers)
        cache = os.path.join(s.cache_dir, name.replace(os.sep, "__") + ".xyz") if s.restart else None
        if cache and not s.overwrite:
            hit = self._load_cache(cache, key)
            if hit is not None:
                say("Reusing the finished calculation %s (--restart)" % cache)
                if getattr(atoms, "name", None):
                    hit.name = atoms.name
                return hit.get_potential_energy(), hit

        if s.dry:  # structures only: no calculation
            say("Dry run: no %s calculation for %s" % (self.label, name))
            atoms.calc = None
            return 0.0, atoms

        calc = self.calculator_for(name, atoms, KPgrid_)
        formula = atoms.get_chemical_formula(empirical=1)
        if not opt:
            say("Getting SP energy using ASE/%s" % self.label)
            atoms.calc = calc
            E = atoms.get_potential_energy()
        else:
            note = " (with cell relaxation)" if cell else (" (with relaxation of the cell along z)" if mask else "")
            say("Optimising the %s structure using ASE/%s%s" % (formula, self.label, note))
            traj = os.path.join(s.outdir, "%s_opt.traj" % os.path.basename(getattr(atoms, "name", name) or name)) if s.outdir else None
            E, atoms = relax(atoms, calc, fmax=s.fmax, steps=s.steps, optimiser=s.optimiser, relax_cell=cell,
                             fix_bottom_layers=fix_bottom_layers, trajectory=traj, cell_mask=mask, fix_mode=s.fix_mode)
        _freeze_results(atoms, calc)
        if cache:
            self._save_cache(cache, key, atoms)
        return E, atoms

    def _fingerprint(self, atoms, opt, cell, mask, fix_bottom_layers):
        #Identifies a calculation: input structure, potential and optimiser settings.
        s = self.settings
        data = {"label": self.label, "opt": bool(opt), "cell": cell, "mask": mask, "fix": int(fix_bottom_layers or 0), "fix_mode": s.fix_mode,
                "fmax": s.fmax, "steps": s.steps, "optimiser": s.optimiser, "numbers": atoms.numbers.tolist(),
                "pos": np.round(atoms.positions, 4).tolist(), "lattice": np.round(atoms.cell[:], 4).tolist()}
        return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _load_cache(path, key):
        import ase.io
        if not os.path.exists(path):
            return None
        try:
            a = ase.io.read(path)
        except Exception:
            return None
        return a if a.info.get("interfacer_fingerprint") == key and a.calc is not None else None

    @staticmethod
    def _save_cache(path, key, atoms):
        import ase.io
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        a = atoms.copy()
        a.calc = atoms.calc
        a.info["interfacer_fingerprint"] = key
        ase.io.write(path, a, format="extxyz")


def _freeze_results(atoms, calc):
    """Replace the (possibly shared) calculator by a SinglePointCalculator holding this structure's results, so that
    the results stay attached to the structure when the calculator is reused for other structures (e.g. an ML model
    that is loaded once per workflow)."""
    from ase.calculators.singlepoint import SinglePointCalculator
    res = {}
    for key, val in getattr(calc, "results", {}).items():
        if key in ("energy", "free_energy", "stress"):
            res[key] = val
        elif key in ("forces", "magmoms") and len(val) == len(atoms):
            res[key] = val
    if "energy" not in res:
        res["energy"] = atoms.get_potential_energy()
    atoms.calc = SinglePointCalculator(atoms, **res)


class GpawEngine(AseEngine):
    """GPAW (plane-wave mode) through ASE, with the shared ASE optimiser. Output in GPAW-tmp/<name>.txt.

    Uses settings.encut (plane-wave cutoff, eV), xc, sigma (Fermi-Dirac smearing, eV), kpspacing (k-point density
    1/kpspacing, as KSPACING in VASP) or the k-point grid of the calculation, and settings.calc_kwargs (any GPAW keyword,
    e.g. {'spinpol': True, 'parallel': {...}})."""
    name = "gpaw"
    tmpdir = ""

    def __init__(self, settings=None):
        super().__init__(None, label="GPAW", settings=settings)

    def get_calculator(self):
        return self.calculator_for("gpaw", None)

    def calculator_for(self, name, atoms, kpgrid=None):
        try:
            from gpaw import GPAW, PW, FermiDirac
        except ImportError as err:
            raise ImportError("-prog gpaw needs GPAW (e.g. conda install -c conda-forge gpaw): %s" % err) from err
        s = self.settings
        kpts = tuple(int(x) for x in kpgrid.split()) if kpgrid else {"density": 1.0 / s.kpspacing, "gamma": True}
        os.makedirs("GPAW-tmp", exist_ok=True)
        kw = dict(mode=PW(s.encut), xc=s.xc, kpts=kpts, occupations=FermiDirac(s.sigma),
                  txt=os.path.join("GPAW-tmp", "%s.txt" % name.replace(os.sep, "__")))
        kw.update(s.calc_kwargs)
        return GPAW(**kw)


class OnetepEngine(AseEngine):
    """ONETEP (linear-scaling DFT) through ASE, with the shared ASE optimiser. Each calculation runs in ONETEP-tmp/<name>.

    The command is taken from settings.exe or $ONETEP_COMMAND (e.g. 'mpirun -n 64 onetep.exe'), the pseudopotential
    directory from $ONETEP_PP_PATH (or calc_kwargs['pseudo_path']). settings.encut and xc set cutoff_energy and
    xc_functional; settings.calc_kwargs can hold 'keywords' (further ONETEP keywords), 'ngwfs_count', 'ngwfs_radius'
    and 'pseudopotentials' (see ase.calculators.onetep.Onetep)."""
    name = "onetep"
    tmpdir = ""

    def __init__(self, settings=None):
        super().__init__(None, label="ONETEP", settings=settings)

    def get_calculator(self):
        return self.calculator_for("onetep", None)

    def calculator_for(self, name, atoms, kpgrid=None):
        from ase.calculators.onetep import Onetep, OnetepProfile
        s = self.settings
        extra = dict(s.calc_kwargs)
        command = s.exe or os.environ.get("ONETEP_COMMAND", "onetep.exe")
        pseudo_path = extra.pop("pseudo_path", os.environ.get("ONETEP_PP_PATH", "."))
        keywords = {"cutoff_energy": "%g eV" % s.encut, "xc_functional": s.xc, "write_forces": True}
        keywords.update(extra.pop("keywords", {}))
        extra.setdefault("ngwfs_count", -1)      # automatic number of NGWFs per species
        extra.setdefault("ngwfs_radius", 9.0)    # bohr
        directory = os.path.join("ONETEP-tmp", name.replace(os.sep, "__"))
        os.makedirs(directory, exist_ok=True)
        return Onetep(profile=OnetepProfile(command=command, pseudo_path=pseudo_path), directory=directory,
                      keywords=keywords, **extra)


ENGINES = ("vasp", "castep", "ase", "mlip", "gpaw", "onetep")


def get_engine(prog, settings=None):
    """Return the engine for a code name: 'vasp', 'castep', 'ase' (the simple potential settings.ase_pot),
    'mlip' (the machine-learning potential settings.mlip with settings.mlip_model on settings.device), 'gpaw' or 'onetep'."""
    settings = settings if settings is not None else CalcSettings()
    prog = prog.lower()
    if prog == "vasp":
        return VaspEngine(settings)
    if prog == "castep":
        return CastepEngine(settings)
    if prog == "ase":
        return AseEngine(lambda: classical_calculator(settings.ase_pot), label=settings.ase_pot, settings=settings)
    if prog == "mlip":
        factory = lazy_calculator_factory(settings.mlip, model=settings.mlip_model, device=settings.device, **settings.mlip_kwargs)
        label = settings.mlip + ("/%s" % settings.mlip_model if settings.mlip_model else "")
        engine = AseEngine(factory, label=label, settings=settings)
        engine.name = "mlip"
        return engine
    if prog == "gpaw":
        return GpawEngine(settings)
    if prog == "onetep":
        return OnetepEngine(settings)
    raise ValueError("Unknown code '%s'; choose from %s" % (prog, ", ".join(ENGINES)))

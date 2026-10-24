"""Engine implementations: VASP, CASTEP and ASE calculators (classical potentials; ML potentials plug in here too)."""
import os

from ase.constraints import FixAtoms

from .base import CalcSettings, Engine
from .castep import call_castep
from .classical import classical_calculator
from .mlip import lazy_calculator_factory
from .optimise import bottom_layer_indices, relax
from .vasp import call_vasp


class VaspEngine(Engine):
    """VASP through ASE (call_vasp). Each calculation runs in its own directory named after the calculation."""
    name = "vasp"
    ext = "vasp"
    write_kwargs = {"format": "vasp", "vasp5": True}
    tmpdir = "VASP-tmp"

    def run(self, atoms, typ, name, dipolCorr="None", FixCell=False, FixVol=0, passivate=None, nosymm=0, vac=0.0,
            KPgrid_=None, ase_opt=None, relax_cell=False, fix_bottom_layers=0):
        s = self.settings
        if fix_bottom_layers:  # written as selective dynamics in the POSCAR
            atoms.set_constraint(FixAtoms(indices=bottom_layer_indices(atoms, fix_bottom_layers)))
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
        kpkw = {"KPgrid": KPgrid_} if KPgrid_ else {"KPspacing": s.kpspacing}
        fixlist = bottom_layer_indices(atoms, fix_bottom_layers) if fix_bottom_layers else []
        return call_castep(atoms, typ=typ, dipolCorr=dipolCorr, name=name, ENCUT=s.encut, PP=s.castep_pp, hubU=s.hubU,
                           FixCell=FixCell, FixList=fixlist, ifDryRun=s.dry, **kpkw)


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
        calc = self.get_calculator()
        formula = atoms.get_chemical_formula(empirical=1)
        if not opt:
            print("Getting SP energy using ASE/%s" % self.label)
            atoms.calc = calc
            E = atoms.get_potential_energy()
        else:
            cell = bool(relax_cell and s.relax_cell_bulk)
            print("Optimising the %s structure using ASE/%s%s" % (formula, self.label, " (with cell relaxation)" if cell else ""))
            traj = os.path.join(s.outdir, "%s_opt.traj" % os.path.basename(getattr(atoms, "name", name) or name)) if s.outdir else None
            E, atoms = relax(atoms, calc, fmax=s.fmax, steps=s.steps, optimiser=s.optimiser, relax_cell=cell,
                             fix_bottom_layers=fix_bottom_layers, trajectory=traj)
        _freeze_results(atoms, calc)
        return E, atoms


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


ENGINES = ("vasp", "castep", "ase", "mlip")


def get_engine(prog, settings=None):
    """Return the engine for a code name: 'vasp', 'castep', 'ase' (the simple potential settings.ase_pot) or
    'mlip' (the machine-learning potential settings.mlip with settings.mlip_model on settings.device)."""
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
    raise ValueError("Unknown code '%s'; choose from %s" % (prog, ", ".join(ENGINES)))

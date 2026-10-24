"""Common calculation settings and the Engine interface shared by all codes."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CalcSettings:
    """Settings shared by all calculations of a workflow (taken from the command line or an input file)."""
    # plane-wave DFT (VASP/CASTEP)
    encut: float = 520.0                 # cutoff energy (eV)
    kpspacing: float = 0.32              # k-point spacing (2*pi/A)
    xc: str = "PBE"                      # exchange-correlation functional
    sigma: float = 0.05                  # smearing width (eV)
    hubU: dict = field(default_factory=dict)   # e.g. {'Fe': {'d': '2.7'}}
    magmoms: Optional[list] = None       # e.g. ['Fe', '5.0', 'O', '0.6']
    exe: Optional[str] = None            # VASP command, e.g. 'mpirun -n 32 vasp_std'
    castep_pp: str = "OTF"               # CASTEP pseudopotentials
    dry: bool = False                    # prepare the calculations but do not run them
    # ASE optimiser (classical and machine-learning potentials)
    ase_pot: str = "LJ"                  # LJ, Morse or EMT
    fmax: float = 0.05                   # force convergence (eV/A)
    steps: int = 200                     # max optimisation steps
    optimiser: str = "BFGS"              # BFGS, FIRE or LBFGS
    relax_cell_bulk: bool = False        # relax the cell of bulk structures (ASE engines)
    # machine-learning potentials (-prog mlip)
    mlip: str = "mace"                   # backend, see interfacer.calculators.mlip
    mlip_model: Optional[str] = None     # model name or path (backend default if None)
    device: str = "auto"                 # cpu, cuda, mps or auto
    mlip_kwargs: dict = field(default_factory=dict)  # extra keyword arguments for the calculator
    outdir: str = "OUTPUT"               # where trajectories are written


class Engine:
    """Runs single-point energies and geometry optimisations with one code.

    Subclasses implement `run`. All methods return ``(energy, atoms)``, where atoms is the final structure.
    """
    #: name used on the command line (-prog)
    name = "engine"
    #: file extension and ase.io.write keywords for the structures written to OUTPUT/
    ext = "xyz"
    write_kwargs = {"format": "extxyz"}
    #: prefix for the directories of the convergence-test calculations ('' = current directory)
    tmpdir = ""

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else CalcSettings()

    def run(self, atoms, typ, name, dipolCorr="None", FixCell=False, FixVol=0, passivate=None, nosymm=0, vac=0.0,
            KPgrid_=None, ase_opt=None, relax_cell=False, fix_bottom_layers=0):
        """Run a calculation.

        Args:
            atoms (ase.Atoms): structure.
            typ (str): 'sp' (single point) or 'geom' (geometry optimisation).
            name (str): name of the calculation (used for the run directory / trajectory).
            dipolCorr (str): dipole correction ('None', 'sc', ...; VASP/CASTEP).
            FixCell, FixVol: keep the cell (or volume) fixed during optimisation (VASP/CASTEP).
            passivate (str): 'bot' or 'top' pseudo-hydrogen passivation (VASP).
            nosymm (int): switch off symmetry (VASP).
            vac (float): vacuum padding (informational, VASP).
            KPgrid_ (str): k-point grid such as '4 4 1' (otherwise the k-point spacing is used).
            ase_opt (bool): ASE engines: whether to optimise (default: typ == 'geom').
            relax_cell (bool): this is a bulk structure whose cell may be relaxed (ASE engines, if enabled in the settings).
            fix_bottom_layers (int): keep the lowest N atomic layers fixed during optimisation.
        """
        raise NotImplementedError

    def single_point(self, atoms, name, **kw):
        return self.run(atoms, "sp", name, **kw)

    def relax(self, atoms, name, **kw):
        return self.run(atoms, "geom", name, **kw)

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.name)

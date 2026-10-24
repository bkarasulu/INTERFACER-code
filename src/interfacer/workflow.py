"""The INTERFACER workflow, split into steps.

    bulk optimisation -> (surface screening, -tsur) -> (layer convergence, -tlay) -> slab creation
    -> slab pre-optimisation and surface formation energies -> selection of the most stable slabs
    -> for each pair of slabs: alignment -> optimisation of the aligned slabs (W_surf) -> interface
       -> (separation scan, -tsep) -> (lateral scan, -thor) -> interface optimisation (W_ad)

Use it from Python::

    from interfacer.workflow import InterfaceWorkflow, WorkflowConfig
    from interfacer.calculators import get_engine, CalcSettings

    engine = get_engine("mlip", CalcSettings(mlip="mace", mlip_model="small"))
    wf = InterfaceWorkflow(engine, WorkflowConfig(miller1=(1, 1, 1), miller2=(1, 1, 1), conv_sep=True))
    results = wf.run(ase.io.read("Cu.cif"), ase.io.read("Ni.cif"))
    print(results.interfaces[0].W_ad)

The command line (interfacer / interfacer-new.py) builds a WorkflowConfig from its options and runs the same steps.
"""
import ast
import json
import os
import time
from dataclasses import asdict, dataclass, field
from sys import stdout
from typing import List, Optional

import ase
import ase.io
from ase.build import niggli_reduce

from .analysis import calculate_SFE, find_prim, get_fu, misfit, surf_area
from .builder import make_interface, make_slab, permute_surface, slab_aligner, standardise_slab
from .convergence import cfactor, check_surfaces, checkHorizontal, conv_layers, convSep
from .io import backup_output, write_struct


def _log(msg=""):
    print(msg)
    stdout.flush()


@dataclass
class WorkflowConfig:
    """Options of the workflow (the command-line option is given in brackets)."""
    # slabs
    miller1: tuple = (1, 0, 0)            # Miller indices of slab 1 (-m1)
    miller2: tuple = (1, 0, 0)            # (-m2)
    creps1: int = 1                       # repetitions along c (layers) of slab 1 (-cr1)
    creps2: int = 1                       # (-cr2)
    surface1: Optional[str] = None        # file with a surface structure for material 1 (-surf1)
    surface2: Optional[str] = None        # (-surf2)
    sym_slabs1: bool = False              # symmetric slabs for material 1 (-sym_slabs1)
    sym_slabs2: bool = False              # (-sym_slabs2)
    thickness: float = 7.0                # min. thickness of symmetric slabs, A (-th)
    no_slabs: int = 1                     # number of most stable slabs used per material (-no_slabs)
    prim: bool = False                    # use primitive bulk cells (-prim)
    chem_pots1: Optional[dict] = None     # chemical potentials for the SFE of slab 1, eV (-cp1)
    chem_pots2: Optional[dict] = None     # (-cp2)
    decomps1: Optional[list] = None       # decomposition product files for slab 1 (-decomps1)
    decomps2: Optional[list] = None       # (-decomps2)
    passivate: bool = False               # pseudo-H passivation, VASP (-pass)
    fix_layers: int = 0                   # fixed bottom layers in slab optimisations (--fix_layers)
    # alignment
    max_slab_dimension: float = 50.0      # max in-plane cell length, A (-msd)
    init_length: float = 3.0              # initial in-plane length, A (-Linit)
    step_length: float = 2.0              # step of the in-plane length, A (-Lstep)
    percentage_tolerance: float = 4.0     # length/angle tolerance, % (-pt)
    flip2: bool = False                   # flip slab 2 upside down (-flip2)
    # interface
    vac_slab: float = 6.0                 # vacuum on each side of the slabs, A (-vac_slab)
    vac_int: float = 1.0                  # vacuum on each side of the interface, A (-vac_int)
    sep: float = 2.0                      # separation between the slabs, A (-sep)
    niggli: bool = False                  # Niggli-reduce the interface (-niggli)
    # tests
    check_surfaces: bool = False          # surface stability test (-tsur)
    miller_list: list = field(default_factory=lambda: [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)])  # (-mlist)
    conv_layers: bool = False             # layer convergence test (-tlay)
    lay_min: int = 2                      # (-lay_min)
    lay_max: int = 8                      # (-lay_max)
    conv_sep: bool = False                # separation scan (-tsep)
    sep_init: float = 1.0                 # (-sep_init)
    sep_final: float = 3.0                # (-sep_final)
    sep_step: float = 0.25                # (-sep_step)
    test_hori: bool = False               # lateral translation scan (-thor)
    hor_steps: tuple = (0.25, 0.25)       # (-hor_steps)
    # calculations
    kpgrid: Optional[str] = None          # k-point grid for slabs/interfaces, e.g. '4 4 1' (-kp_gr, -gamma)
    zero_chem_pots: bool = False          # use zero chemical potentials for non-stoichiometric slabs (test runs only)
    dry: bool = False                     # do not compute energies (-dry)
    surf_only: bool = False               # stop after the slab calculations (-surf_only)
    # output
    outdir: str = "OUTPUT"                # (--outdir)
    outfile: str = "interfaces.out"       # (-o)
    view: bool = False                    # (-view)
    verbose: bool = False                 # (-v)


@dataclass
class SlabResult:
    name: str
    miller: tuple
    natoms: int
    energy: float                         # eV (pre-optimised slab)
    sfe: float                            # surface formation energy, eV/nm^2

    @property
    def sfe_J(self):
        return self.sfe * cfactor


@dataclass
class InterfaceResult:
    name: str
    slab1: str
    slab2: str
    misfit_before: float                  # fraction
    misfit_after: float
    E_slab1: float                        # eV (optimised aligned slabs)
    E_slab2: float
    W_surf1: float                        # eV/nm^2
    W_surf2: float
    sep: float                            # A
    E_interface: float = float("nan")     # eV
    W_ad: float = float("nan")            # work of adhesion, eV/nm^2
    n_interfaces: int = 2
    lateral_shift: Optional[list] = None  # fractional (x, y) from -thor
    area: float = float("nan")            # A^2

    @property
    def W_ad_J(self):
        return self.W_ad * cfactor


@dataclass
class WorkflowResults:
    E_bulk1: float = float("nan")         # eV/atom
    E_bulk2: float = float("nan")
    miller1: tuple = ()
    miller2: tuple = ()
    slabs1: List[SlabResult] = field(default_factory=list)
    slabs2: List[SlabResult] = field(default_factory=list)
    interfaces: List[InterfaceResult] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class InterfaceWorkflow:
    """Runs the INTERFACER workflow with an engine (see interfacer.calculators) and a WorkflowConfig."""

    def __init__(self, engine, config=None):
        self.engine = engine
        self.cfg = config if config is not None else WorkflowConfig()
        self.results = WorkflowResults()
        self.bulk1 = self.bulk2 = None
        self.Ebulk1 = self.Ebulk2 = None
        self._outf = None

    # ------------------------------------------------------------------------------------------------ helpers
    def write(self, atoms, base):
        return write_struct(atoms, base, self.engine, self.cfg.outdir)

    def report(self, text):
        _log(text)
        if self._outf:
            self._outf.writelines(text)

    def _view(self, *structures):
        if self.cfg.view:
            from ase.visualize import view
            for s in structures:
                view(s)

    # ------------------------------------------------------------------------------------------------ steps
    def run_bulk(self, atoms1, atoms2):
        """Optimise both bulk structures (cell included for VASP/CASTEP, and for ASE engines if relax_cell_bulk)."""
        cfg = self.cfg
        for n, atoms in ((1, atoms1), (2, atoms2)):
            _log("Structure %d: %-8s with fu=%d and %s lattice" % (n, atoms.get_chemical_formula(empirical=1), get_fu(atoms),
                                                                    atoms.get_cell().get_bravais_lattice()))
        if cfg.prim:
            _log('Using primitive cells as input as requested')
            atoms1 = find_prim(atoms1); atoms2 = find_prim(atoms2)
        out = []
        for n, atoms in ((1, atoms1), (2, atoms2)):
            _log("Computing bulk %d energy." % n)
            atoms.name = 'bulk%d' % n
            E, atoms = self.engine.run(atoms, "geom", 'bulk%d' % n, dipolCorr='None', relax_cell=True)
            out.append((atoms, E / len(atoms)))  # bulk energy per atom
        (self.bulk1, self.Ebulk1), (self.bulk2, self.Ebulk2) = out
        self.results.E_bulk1, self.results.E_bulk2 = self.Ebulk1, self.Ebulk2
        self.write(self.bulk1, "bulk1_opted"); self.write(self.bulk2, "bulk2_opted")

    def screen_surfaces(self):
        """Surface stability test (-tsur): select the most stable surface of each material from cfg.miller_list."""
        cfg = self.cfg
        selected = []
        for n, bulk, Eb, creps in ((1, self.bulk1, self.Ebulk1, cfg.creps1), (2, self.bulk2, self.Ebulk2, cfg.creps2)):
            _log('\nChecking the minimum-energy surface for Material %d (%s), considering the Miller indices: %s'
                 % (n, bulk.get_chemical_formula(empirical=1), cfg.miller_list))
            _, _, _, mil = check_surfaces(bulk, cfg.miller_list, vac=cfg.vac_slab, Ebulk=Eb, creps=creps, engine=self.engine,
                                          kpgrid=cfg.kpgrid, outdir=cfg.outdir)
            selected.append(tuple(mil))
        cfg.miller1, cfg.miller2 = selected
        _log('Selected surfaces: %s for material 1 and %s for material 2' % (cfg.miller1, cfg.miller2))

    def converge_layers(self):
        """Layer convergence test (-tlay): sets cfg.creps1/creps2 to the converged numbers of layers."""
        cfg = self.cfg
        _log('\n\nConverging layer thickness...')
        results = []
        for n, (bulk, mil) in enumerate(((self.bulk1, cfg.miller1), (self.bulk2, cfg.miller2)), 1):
            slab = make_slab(mil, bulk, repeat=(1, 1, 1), square=False)
            slab.center(vacuum=0, axis=2)
            slab.name = "%s_%d%d%d" % (slab.get_chemical_formula(empirical=1), mil[0], mil[1], mil[2])
            _log('\nRunning Material %d (%s)' % (n, slab.get_chemical_formula(empirical=1)))
            results.append(conv_layers(slab, vac=cfg.vac_slab, lay_min=cfg.lay_min - 1, lay_max=cfg.lay_max - 1,
                                       engine=self.engine, outdir=cfg.outdir, ifPlot=1))
        (nl1, E1, _), (nl2, E2, _) = results
        _log('Converged thickness (creps):')
        _log('\tfor slab1  is %d with E=%.2f eV' % (nl1, E1))
        _log('\tfor slab2  is %d with E=%.2f eV' % (nl2, E2))
        cfg.creps1, cfg.creps2 = nl1, nl2

    def build_slabs(self, n):
        """Slabs of material n (1 or 2): read from a file, symmetric slabs (permute_surface) or make_slab."""
        cfg = self.cfg
        bulk = self.bulk1 if n == 1 else self.bulk2
        miller = cfg.miller1 if n == 1 else cfg.miller2
        creps = cfg.creps1 if n == 1 else cfg.creps2
        surface = cfg.surface1 if n == 1 else cfg.surface2
        sym = cfg.sym_slabs1 if n == 1 else cfg.sym_slabs2
        slabs = []
        if surface:
            slabs.append(standardise_slab(ase.io.read(surface)))  # surface normal along z (rotation only)
            _log('Surface %d is read from %s' % (n, surface))
        elif sym:
            _log('\nBulk%d is being symmetrized' % n)
            slabs = permute_surface(bulk.copy(), miller, cfg.thickness, vacuum=cfg.vac_slab)
            if not slabs:
                slabs.append(make_slab(miller, bulk, repeat=(1, 1, creps), square=0))
        else:
            slabs.append(make_slab(miller, bulk, repeat=(1, 1, creps), square=0))
        for i, slab in enumerate(slabs):
            slab.name = 'slab%d_cut%s' % (n, i)
        return slabs

    def _decomps(self, n, files):
        #Read the decomposition products; if they have no energy, optimise them (as bulk phases).
        decomps = []
        for no_d, d in enumerate(files):
            decomp = ase.io.read(d, index=-1)
            if decomp.calc is None:
                decomp.name = 'decomp%d_%s' % (n, no_d)
                _, decomp = self.engine.run(decomp, 'geom', 'decomp%d_%s' % (n, no_d), dipolCorr='None', relax_cell=True)
            decomps.append(decomp)
        return decomps

    def compute_sfe(self, n, slabs):
        """Pre-optimise copies of the slabs of material n and compute their surface formation energies (eV/nm^2).
        The unoptimised slabs are used for the alignment."""
        cfg = self.cfg
        bulk = self.bulk1 if n == 1 else self.bulk2
        chem_pots = cfg.chem_pots1 if n == 1 else cfg.chem_pots2
        decomp_files = cfg.decomps1 if n == 1 else cfg.decomps2
        miller = cfg.miller1 if n == 1 else cfg.miller2
        pas = {1: 'bot', 2: 'top'}[n] if cfg.passivate else None
        decomps = self._decomps(n, decomp_files) if (decomp_files and not chem_pots) else None
        results = []
        for i, slab in enumerate(slabs):
            slab_opt = slab.copy()
            slab_opt.name = 'slab%d_cut%s' % (n, i)
            slab_opt.center(vacuum=cfg.vac_slab, axis=2)
            self.write(slab_opt, "slab%d_cut%s_pre" % (n, i))

            _log("Pre-optimizing %s (before alignment)." % slab_opt.name)
            E, slab_opt = self.engine.run(slab_opt, 'geom', 'slab%d_cut%s_pre' % (n, i), dipolCorr='sc', FixCell=True, FixVol=0,
                                          passivate=pas, KPgrid_=cfg.kpgrid, fix_bottom_layers=cfg.fix_layers)  #fixed cell: keeps the vacuum
            slab_opt.name = 'slab%d_cut%s' % (n, i)
            self.write(slab_opt, "slab%d_cut%s_pre_opted" % (n, i))

            if cfg.dry:
                _log("Dry run: surface formation energy not computed (set to 0).")
                SFE = 0.0
            else:
                if chem_pots:
                    SFE = calculate_SFE(slab_opt, bulk, chem_pots={k: float(v) for k, v in chem_pots.items()})
                elif decomps:
                    SFE = calculate_SFE(slab_opt, bulk, decomps=decomps)
                else:  # stoichiometric slabs (zero chemical potentials allowed for test runs)
                    SFE = calculate_SFE(slab_opt, bulk, ase=cfg.zero_chem_pots)
                _log("Surface formation energy of %s: %.2f eV/nm^2 = %.2f J/m^2\n" % (slab_opt.name, SFE, SFE * cfactor))
            results.append(SlabResult(slab_opt.name, tuple(miller), len(slab_opt), float(E), float(SFE)))
        return results

    def select_slabs(self, n, slabs, sfe_results):
        """The cfg.no_slabs most stable slabs (lowest SFE), as a list of (index, slab, SlabResult)."""
        order = sorted(range(len(slabs)), key=lambda i: sfe_results[i].sfe)[:self.cfg.no_slabs]
        sym = self.cfg.sym_slabs1 if n == 1 else self.cfg.sym_slabs2
        if sym:
            _log("\nThe %s lowest energy slabs for material %d are:" % (self.cfg.no_slabs, n))
        else:
            _log("\nThe surface formation energy of slab%d_cut0 is:" % n)
        for i in order:
            _log("%s: %.2f eV/nm^2" % (slabs[i].name, sfe_results[i].sfe))
        return [(i, slabs[i], sfe_results[i]) for i in order]

    def align(self, slab1, slab2):
        """Find the commensurate supercell of two slabs (copies are aligned). Returns (slab1, slab2, misfit before, after)."""
        cfg = self.cfg
        m0 = misfit(slab1, slab2)
        _log("\nMisfit (mu) of %s and %s (before alignment): %.2f%%" % (slab1.name, slab2.name, m0 * 100))
        _log("\nAligning the two slabs...")
        s1, s2 = slab_aligner(slab1.copy(), slab2.copy(), cfg.init_length, cfg.max_slab_dimension, cfg.step_length,
                              cfg.percentage_tolerance, cfg.thickness, self.bulk1, self.bulk2)
        return s1, s2, m0, misfit(s1, s2)

    def optimise_aligned(self, slab1, slab2, ci, cj):
        """Optimise the aligned slabs (fixed cell). Returns (slab1, slab2, E1, E2, W_surf1, W_surf2) with W in eV/nm^2."""
        cfg = self.cfg
        _log("\nOptimizing %s and %s (after alignment)." % (slab1.name, slab2.name))
        slab1.center(vacuum=cfg.vac_slab, axis=2)
        slab2.center(vacuum=cfg.vac_slab, axis=2)
        E1, slab1 = self.engine.run(slab1, 'geom', 'slab1_cut%s_aligned' % ci, dipolCorr='sc', FixCell=True, FixVol=0,
                                    passivate=('bot' if cfg.passivate else None), nosymm=1, vac=cfg.vac_slab, KPgrid_=cfg.kpgrid,
                                    fix_bottom_layers=cfg.fix_layers)
        E2, slab2 = self.engine.run(slab2, 'geom', 'slab2_cut%s_aligned' % cj, dipolCorr='sc', FixCell=True, FixVol=0,
                                    passivate=('top' if cfg.passivate else None), nosymm=1, vac=cfg.vac_slab, KPgrid_=cfg.kpgrid,
                                    fix_bottom_layers=cfg.fix_layers)
        Ws1 = (E1 - len(slab1) * self.Ebulk1) / 2 / surf_area(slab1) / 0.01  #A2 to nm2
        Ws2 = (E2 - len(slab2) * self.Ebulk2) / 2 / surf_area(slab2) / 0.01
        m1, m2 = self.cfg.miller1, self.cfg.miller2
        text = '\nCalculating the W_surf (surface formation energies / surface area) of the optimised aligned slabs...\n'
        text += '%s: %.3f eV/atom\n' % ('Ebulk_1', self.Ebulk1)
        text += '%s: %.3f eV/atom\n' % ('Ebulk_2', self.Ebulk2)
        text += '%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('Wsurf_1', slab1.get_chemical_formula(empirical=1), m1[0], m1[1], m1[2], Ws1, Ws1 * cfactor)
        text += '%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('Wsurf_2', slab2.get_chemical_formula(empirical=1), m2[0], m2[1], m2[2], Ws2, Ws2 * cfactor)
        self.report(text)
        self.write(slab1, "slab1_cut%s_aligned_opted" % ci); self.write(slab2, "slab2_cut%s_aligned_opted" % cj)
        return slab1, slab2, E1, E2, Ws1, Ws2

    def optimise_interface(self, interface, name, vac_used, E1, E2):
        """Optimise the interface; return (interface, E_int, W_ad in eV/nm^2, number of interfaces in the cell).

        W_ad = (E_slab1 + E_slab2 - E_int) / (n_int * A): with a small vacuum (<= 4 A on each side, e.g. sep/2) the
        periodic cell contains two interfaces (n_int=2), otherwise one interface and two free surfaces (n_int=1)."""
        _log("\nOptimizing the final interface geometry.")
        interface.name = name
        dipC = 'SC' if vac_used > 4 else 'None'
        #CHECK: Should we fix the volume or not?? fix vol works for some cases (YIG) but not for some (Al2O3//Si)
        Eint, interface = self.engine.run(interface, 'geom', name, dipolCorr=dipC, FixVol=0, nosymm=1, KPgrid_=self.cfg.kpgrid)
        n_int = 2 if vac_used <= 4 else 1
        Wad = (E1 + E2 - Eint) / (n_int * surf_area(interface)) / 0.01  #A2 to nm2
        self.report('W_ad (work of adhesion, %d interface(s) in the cell) for the final optimised interface: %.2f eV/nm^2 = %.2f J/m^2\n'
                    % (n_int, Wad, Wad * cfactor))
        return interface, Eint, Wad, n_int

    def run_pair(self, ci, slab1, cj, slab2):
        """Align, optimise and interface one pair of slabs. Returns an InterfaceResult (or None if alignment fails)."""
        cfg = self.cfg
        try:
            slab1_a, slab2_a, m0, m1 = self.align(slab1, slab2)
        except RuntimeError as err:
            _log("Slab alignment failed for %s and %s: %s" % (slab1.name, slab2.name, err))
            return None
        slab1_a.name = 'slab1_cut%s_aligned' % ci
        slab2_a.name = 'slab2_cut%s_aligned' % cj
        _log("\nMisfit (mu) of %s and %s (after alignment): %.2f%%" % (slab1_a.name, slab2_a.name, m1 * 100))
        self._view(slab1_a, slab2_a)
        if cfg.flip2:  #Flip the slab2 (top slab) upside-down. The cell angles cannot be preserved (does not work reliably).
            slab2_a.rotate(a=180, v='x', rotate_cell=0, center=(0, 0, 0))
            slab2_a.center()
        self.write(slab1_a, "slab1_cut%s_aligned" % ci); self.write(slab2_a, "slab2_cut%s_aligned" % cj)

        sep = cfg.sep
        vac_used = cfg.vac_int  #vacuum padding (on each side) of the final interface
        interface = make_interface(slab1_a, slab2_a, sep, cfg.vac_int, reorder=True)  #before optimising the slabs
        self._view(interface)
        self.write(interface, "interface_cuts_%s_%s" % (ci, cj))

        slab1_o, slab2_o, E1, E2, Ws1, Ws2 = self.optimise_aligned(slab1_a, slab2_a, ci, cj)
        self._view(slab1_o, slab2_o)

        interface = make_interface(slab1_o, slab2_o, sep, cfg.vac_int, reorder=True)
        if cfg.verbose:
            _log(" ".join("%.4f" % x for x in interface.cell.cellpar()))
        self._view(interface)
        self.write(interface, "interface-pre_opt_cuts_%s_%s" % (ci, cj))

        shift = None
        if cfg.conv_sep:
            _log("\n\nFinding the optimal interface (interlayer) separation.")
            #separation/2 is used as vacuum padding to have a symmetric seperation at all interfaces.
            interface, sep = convSep(slab1_o, slab2_o, vac=None, sep_init=cfg.sep_init, sep_final=cfg.sep_final, sep_step=cfg.sep_step,
                                     view=0, ifPlot=1, engine=self.engine, kpgrid=cfg.kpgrid, outdir=cfg.outdir)
            vac_used = sep / 2
        if cfg.test_hori:
            _log("\n\nFinding the optimial interface lateral shift.")
            interface, _, shift = checkHorizontal(slab1_o, slab2_o, sep=sep, steps=cfg.hor_steps, ifPlot=1, engine=self.engine,
                                                  kpgrid=cfg.kpgrid, outdir=cfg.outdir)
            vac_used = sep / 2
        if cfg.niggli:
            _log("Niggli reduce the interface...")
            niggli_reduce(interface)
        self._view(interface)

        name = 'interface_cuts_%s_%s' % (ci, cj)
        interface, Eint, Wad, n_int = self.optimise_interface(interface, name, vac_used, E1, E2)
        self.write(interface, name + "_opted")
        ase.io.write(os.path.join(cfg.outdir, name + "_opted.cif"), interface, format='cif')
        return InterfaceResult(name, slab1.name, slab2.name, float(m0), float(m1), float(E1), float(E2), float(Ws1), float(Ws2),
                               float(sep), float(Eint), float(Wad), n_int, [float(x) for x in shift] if shift else None,
                               float(surf_area(interface)))

    # ------------------------------------------------------------------------------------------------ full run
    def run(self, atoms1, atoms2):
        """Run the full workflow for two bulk structures. Returns WorkflowResults (also written to <outdir>/results.json)."""
        cfg = self.cfg
        backup_output(cfg.outdir)  #keep the previous output in <outdir>_prev
        self._outf = open(cfg.outfile, 'w')
        self._outf.write(time.strftime("%a %b %d %H:%M:%S %Z %Y") + "\n")
        t0 = time.time()
        try:
            self.run_bulk(atoms1, atoms2)
            if cfg.check_surfaces:
                self.screen_surfaces()
            if cfg.conv_layers:
                self.converge_layers()
            self.results.miller1, self.results.miller2 = tuple(cfg.miller1), tuple(cfg.miller2)

            slabs1, slabs2 = self.build_slabs(1), self.build_slabs(2)
            self.results.slabs1 = self.compute_sfe(1, slabs1)
            self.results.slabs2 = self.compute_sfe(2, slabs2)
            sel1 = self.select_slabs(1, slabs1, self.results.slabs1)
            sel2 = self.select_slabs(2, slabs2, self.results.slabs2)

            if cfg.surf_only:
                _log('Exiting code after the optimisation of the surface slabs')
            else:
                for ci, slab1, _ in sel1:
                    for cj, slab2, _ in sel2:
                        res = self.run_pair(ci, slab1, cj, slab2)
                        if res is not None:
                            self.results.interfaces.append(res)
        finally:
            self._outf.close()
            self._outf = None
            with open(os.path.join(cfg.outdir, "results.json"), "w") as f:
                json.dump(self.results.to_dict(), f, indent=1, default=str)
        _log("Elapsed time: %.2f sec." % (time.time() - t0))
        return self.results


def config_from_args(args):
    """WorkflowConfig from the parsed command-line arguments of interfacer.cli."""
    def literal(text):
        return {k: float(v) for k, v in ast.literal_eval(text).items()} if text else None
    kpgrid = "1 1 1" if args.gamma_only else args.kpgrid
    return WorkflowConfig(
        miller1=tuple(args.miller1), miller2=tuple(args.miller2), creps1=args.creps1 or 1, creps2=args.creps2 or 1,
        surface1=args.surface1 or None, surface2=args.surface2 or None, sym_slabs1=args.symSlabs1, sym_slabs2=args.symSlabs2,
        thickness=args.thickness, no_slabs=args.noSlabs, prim=args.prim, chem_pots1=literal(args.chem_pots1),
        chem_pots2=literal(args.chem_pots2), decomps1=args.decomps1, decomps2=args.decomps2, passivate=args.pas,
        fix_layers=args.fix_layers, max_slab_dimension=args.max_slab_dimension, init_length=args.initLength,
        step_length=args.stepLength, percentage_tolerance=args.percentage_tolerance, flip2=args.flip2, vac_slab=args.vac_slab,
        vac_int=args.vac_int, sep=args.sep, niggli=args.niggli, check_surfaces=args.checkSurfs,
        miller_list=[tuple(m) for m in args.miller_list], conv_layers=args.convLayers, lay_min=args.lay_min, lay_max=args.lay_max,
        conv_sep=args.convSep, sep_init=args.sep_init, sep_final=args.sep_final, sep_step=args.sep_step, test_hori=args.test_hori,
        hor_steps=tuple(args.hor_steps), kpgrid=kpgrid, zero_chem_pots=(args.prog == 'ase'), dry=args.dry, surf_only=args.surf_only,
        outdir=args.outdir, outfile=args.outfile, view=args.view, verbose=args.verb)

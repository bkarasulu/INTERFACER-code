"""Tutorial 5: using INTERFACER from Python.

Runs in a few seconds with the EMT potential; set USE_MACE = True to use MACE-MP-0 instead (pip install mace-torch).
"""
import ase.io
from ase.calculators.emt import EMT

from interfacer.analysis import misfit, surf_area
from interfacer.builder import make_interface, make_slab, slab_aligner
from interfacer.calculators import AseEngine, CalcSettings, get_engine
from interfacer.convergence import cfactor, convSep
from interfacer.workflow import InterfaceWorkflow, WorkflowConfig

USE_MACE = False
cu, ni = ase.io.read("Cu.cif"), ase.io.read("Ni.cif")

# --- 1. An engine: any ASE calculator, a machine-learning potential, VASP or CASTEP --------------------------------
settings = CalcSettings(fmax=0.02, steps=300, relax_cell_bulk=True, outdir="OUTPUT")
if USE_MACE:
    engine = get_engine("mlip", CalcSettings(mlip="mace", mlip_model="small", fmax=0.02, steps=300, relax_cell_bulk=True))
else:
    engine = AseEngine(EMT, label="EMT", settings=settings)

# --- 2. Building blocks: slabs, alignment, interface --------------------------------------------------------------
E_cu, cu = engine.relax(cu, "bulk_Cu", relax_cell=True)
E_ni, ni = engine.relax(ni, "bulk_Ni", relax_cell=True)
print("lattice constants: Cu %.3f A, Ni %.3f A" % (cu.cell.lengths()[0], ni.cell.lengths()[0]))

s1 = make_slab((1, 1, 1), cu, repeat=(1, 1, 3))
s2 = make_slab((1, 1, 1), ni, repeat=(1, 1, 3))
a1, a2 = slab_aligner(s1, s2, 3, 20, 2, 4, 7, cu, ni)   # L, Lmax, Lstep, tolerance (%), thickness, bulks
print("misfit: %.2f %% before, %.2f %% after alignment; %d + %d atoms" % (100 * misfit(s1, s2), 100 * misfit(a1, a2), len(a1), len(a2)))

# surface energy of the aligned Cu slab
slab = a1.copy()
slab.center(vacuum=6.0, axis=2)
E_slab, slab = engine.relax(slab, "Cu111", FixCell=True)
W = (E_slab - len(slab) * E_cu / len(cu)) / (2 * surf_area(slab)) / 0.01   # eV/nm^2
print("W_surf(Cu 111) = %.2f eV/nm^2 = %.2f J/m^2" % (W, W * cfactor))

# separation scan with rigid slabs (single points)
interface, best_sep = convSep(a1, a2, sep_init=1.5, sep_final=3.0, sep_step=0.25, ifPlot=0, engine=engine)
print("best separation: %.2f A" % best_sep)
ase.io.write("OUTPUT/my_interface.xyz", make_interface(a1, a2, sep=best_sep, vac=best_sep / 2))

# --- 3. The full workflow -----------------------------------------------------------------------------------------
cfg = WorkflowConfig(miller1=(1, 1, 1), miller2=(1, 1, 1), creps1=3, creps2=3, max_slab_dimension=20,
                     conv_sep=True, sep_init=1.5, sep_final=3.0, sep_step=0.25)
results = InterfaceWorkflow(engine, cfg).run(ase.io.read("Cu.cif"), ase.io.read("Ni.cif"))
r = results.interfaces[0]
print("\nworkflow result: W_surf = %.2f / %.2f J/m^2, W_ad = %.2f J/m^2 at a separation of %.2f A"
      % (r.W_surf1 * cfactor, r.W_surf2 * cfactor, r.W_ad_J, r.sep))

# Python API

The package can be used as a library: run the whole workflow, single steps, or the building blocks.

## Full workflow

```python
import ase.io
from interfacer.calculators import CalcSettings, get_engine
from interfacer.workflow import InterfaceWorkflow, WorkflowConfig

engine = get_engine("ase", CalcSettings(ase_pot="EMT", fmax=0.02, relax_cell_bulk=True))
cfg = WorkflowConfig(miller1=(1, 1, 1), miller2=(1, 1, 1), creps1=3, creps2=3, max_slab_dimension=20,
                     conv_sep=True, sep_init=1.5, sep_final=3.0, sep_step=0.25)
res = InterfaceWorkflow(engine, cfg).run(ase.io.read("Cu.cif"), ase.io.read("Ni.cif"))
r = res.interfaces[0]
print(f"W_surf = {r.W_surf1:.2f}, {r.W_surf2:.2f} eV/nm^2; W_ad = {r.W_ad_J:.2f} J/m^2 at sep = {r.sep} A")
```

## Building blocks

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from interfacer.builder import make_slab, slab_aligner, make_interface
from interfacer.analysis import misfit, calculate_SFE
from interfacer.calculators import AseEngine, CalcSettings

cu, ni = bulk("Cu", cubic=True, a=3.61), bulk("Ni", cubic=True, a=3.52)
s1 = make_slab((1, 1, 1), cu, repeat=(1, 1, 3))           # slab with the surface normal along z
s2 = make_slab((1, 1, 1), ni, repeat=(1, 1, 3))
a1, a2 = slab_aligner(s1, s2, 3, 20, 2, 4, 7, cu, ni)     # L, Lmax, Lstep, tolerance %, thickness, bulks
print("misfit after alignment: %.2f %%" % (100 * misfit(a1, a2)))
interface = make_interface(a1, a2, sep=2.0, vac=1.0)

engine = AseEngine(EMT, label="EMT", settings=CalcSettings(fmax=0.02))   # any ASE calculator class/object
E, relaxed = engine.relax(interface, "interface")
```

With an ML potential:

```python
from interfacer.calculators import mlip_calculator
calc = mlip_calculator("mace", model="small", device="cpu")   # a plain ASE calculator
```

See the API reference pages for all functions, which keep their original names and arguments.

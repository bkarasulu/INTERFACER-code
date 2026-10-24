"""Simple ASE potentials (formerly tools/external.py): call_ase (LJ/Morse/EMT) and call_dftb."""
import os

from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones as LJ
from ase.calculators.morse import MorsePotential as MP
from ase.optimize import BFGS

POTENTIALS=('LJ','Morse','EMT')

def classical_calculator(ctype='LJ'):
        """Return an ASE calculator for a simple potential: 'LJ' (Lennard-Jones), 'Morse' or 'EMT'.
        The default LJ/Morse parameters are only meant for quick functionality tests; EMT works for Al, Cu, Ag, Au, Ni, Pd, Pt (and H, C, N, O)."""
        if ctype.upper()=='LJ': return LJ()
        elif ctype.lower() in ['morse','m','mor']: return MP()
        elif ctype.upper()=='EMT': return EMT()
        raise ValueError('ASE calculator type %s not supported, use LJ, Morse or EMT.'%ctype)

def call_ase(atoms,opt=False,ctype='LJ',fmax=0.05,steps=200,outdir='OUTPUT'):
        #THe dfault paramters are completely irrelvant aand provided to be used in test calculations.
        #TODO: add support for user to define their own LJ/Morse params.
        #Returns (E, atoms) like call_vasp/call_castep; atoms is optimised in place when opt=True.
        if isinstance(ctype,str): calc=classical_calculator(ctype)
        elif ctype is not None: calc=ctype() if isinstance(ctype,type) else ctype #a calculator class or object
        else: raise ValueError('call_ase: no calculator type given.')
        cname=ctype if isinstance(ctype,str) else type(calc).__name__

        atoms.calc = calc
        
        if opt:
                print('Optimising the %s structure using ASE/%s'%(atoms.get_chemical_formula(empirical=1),cname))
                traj=None
                if outdir:
                        os.makedirs(outdir,exist_ok=True)
                        traj=os.path.join(outdir,'%s_opt.traj'%getattr(atoms,'name','structure'))
                try:
                        dyn = BFGS(atoms,trajectory=traj)
                        dyn.run(fmax=fmax,steps=steps)
                except Exception as err: 
                        #Previously an energy of 0.0 was returned silently here. Now the energy of the last geometry is returned (with a warning).
                        print("call_ase: WARNING: Error in geometry opt using ASE (returning the energy of the last geometry):  ", err)
                return atoms.get_potential_energy(),atoms 
        else:
               print('Getting SP energy using ASE/%s'%cname)
               return atoms.get_potential_energy(),atoms 
        
#TODO: Add DFTB and GPAW!
def call_dftb(atoms):
       #This requires rigorous testing, as I  only took the example directly from ASE website.
       #https://wiki.fysik.dtu.dk/ase/ase/calculators/dftb.html
       #requires DFTB+ (and the ASE Dftb calculator settings, e.g. DFTB_PREFIX).
       from ase.calculators.dftb import Dftb
       calc = Dftb(Hamiltonian_='DFTB',  # this line is included by default
            Hamiltonian_SCC='Yes',
            Hamiltonian_SCCTolerance=1e-8,
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_H='s',
            Hamiltonian_MaxAngularMomentum_O='p')
       #or read the 'dftb_in.hsd' file
       atoms.calc=calc
       return atoms.get_potential_energy(), atoms


# For installing GPAW, one could use:
# conda install gpaw
# which would probably require UCX package, which you would need to install using:
# conda install -c conda-forge ucx
#

"""Command-line interface of INTERFACER (``interfacer`` command, also used by ``interfacer-new.py``).

The workflow: optimise the two bulk structures -> (optional) surface stability and layer convergence tests ->
create slabs -> pre-optimise them and compute surface formation energies -> for each pair of slabs: find a
commensurate supercell, optimise the aligned slabs, stack them into an interface, (optional) separation and
lateral translation scans -> optimise the interface and compute the work of adhesion.
"""
import argparse
import ast
import os
import re
import time
from sys import stdout

import ase
import ase.io
from ase.build import niggli_reduce
from ase.visualize import view

from . import __version__
from .analysis import calculate_SFE, find_prim, get_fu, misfit, surf_area
from .builder import make_interface, make_slab, permute_surface, slab_aligner, standardise_slab
from .calculators import CalcSettings, get_engine
from .calculators.classical import POTENTIALS
from .calculators.mlip import BACKENDS as MLIP_BACKENDS
from .calculators.optimise import OPTIMISERS
from .convergence import cfactor, check_surfaces, checkHorizontal, conv_layers, convSep, set_default_engine
from .io import backup_output, write_struct


def grep(key,fname,n=-1):
    #Returns the lines containing the keyword in a given file (all matches if n<0, otherwise the first n), joined by newlines.
        try:
                with open(fname) as f: lines=[ln.rstrip('\n') for ln in f if key in ln]
        except OSError: return ""
        if n>=0: lines=lines[:n]
        return "\n".join(lines)

def parse_miller(s):
        """Parse a Miller index given on the command line, e.g. '111', '1-10', '1,1,0', '(1,1,0)' or '[1 1 0]'."""
        s=str(s).strip().strip('()[]').strip()
        if ',' in s or ' ' in s: vals=[int(x) for x in re.split(r'[,\s]+',s) if x!='']
        else: vals=[int(x) for x in re.findall(r'-?\d',s)] #single-digit indices, e.g. 1-10 -> (1,-1,0)
        if len(vals)!=3: raise argparse.ArgumentTypeError("Cannot parse Miller index '%s' (expected e.g. 111, 1-10 or 1,1,0)"%s)
        return tuple(vals)

def _potcar_dirs():
        try: return sorted(os.listdir(os.environ['VASP_PP_PATH']))
        except Exception: return None

def parse_hubbard_u(entries):
        """Parse '-hubU Fe d 2.7, Sm f 6.1' into {'Fe': {'d': '2.7'}, 'Sm': {'f': '6.1'}}."""
        hubU={}
        if not entries: return hubU
        for xx in " ".join(entries).split(','):
                x=xx.split()
                if len(x)!=3: raise ValueError("Cannot parse -hubU entry '%s' (expected e.g. 'Fe d 2.7')"%xx.strip())
                hubU.setdefault(x[0],{})[x[1]]=x[2]
        return hubU


def build_parser():
    parser = argparse.ArgumentParser(description="INTERFACER: build and analyse solid-solid interfaces from two bulk structures.")

    #Basic input/output related keywords
    parser.add_argument("-i1", "--infile1", required=1)
    parser.add_argument("-i2", "--infile2", required=1)
    parser.add_argument("-o", "--outfile", default="interfaces.out")
    parser.add_argument("-m1", "--miller1", default=(1,0,0), nargs="+", type=int)
    parser.add_argument("-m2", "--miller2", default=(1,0,0), nargs="+", type=int)
    parser.add_argument("-surf1", "--surface1", default=False, help="extra input file for a surface structure of i1")
    parser.add_argument("-surf2", "--surface2", default=False, help="extra input file for a surface structure of i2")

    #Keywords related to structure building
    parser.add_argument("-cr1", "--creps1", default=None, type=int, help='Repetitions in c direction (number of layers) for surface model 1')
    parser.add_argument("-cr2", "--creps2", default=None, type=int, help='Repetitions in c direction (number of layers) for surface model 2')
    parser.add_argument("-msd", "--max_slab_dimension", default=50, type=float, help='max length of cell sides')
    parser.add_argument("-th", "--thickness", default=7, type=float, help='Minimum thickness of the symmetric slabs (-sym_slabs1/2), def: 7A. The thickness of the other slabs is set by -cr1/-cr2.')
    parser.add_argument("-pt", "--percentage_tolerance", default=4, type=float, help='percentage tolerances for angle and length matching between two slabs')
    parser.add_argument("-Linit", "--initLength", default=3, type=float, help='Initial (minimum) length of surface slabs in a/b direction, def: 3A') #orig def: 5A
    parser.add_argument("-Lstep", "--stepLength", default=2, type=float, help='step size for the increasing the length in a/b direction, def: 2A') #orig def: 5A
    parser.add_argument("-vac_slab", "--vac_slab", default=6., type=float, help='Size of vacuum padding in z direction (only for slabs), def: 6 A on both sites')
    parser.add_argument("-vac_int", "--vac_int", default=1., type=float, help='Size of vacuum padding in z direction (only for interface), def: 1.0 A on both sites') #should be equal to args.sep/2 by default!
    parser.add_argument("-sep", "--sep", default=2.0, type=float, help='Seperation between the surface slabs forming the interface, def: 2.0 A')
    parser.add_argument("-pass", "--pas", default=False, action="store_true", help="To passivate the top and bottom surface of the interface slab based on dangling bonds.")
    parser.add_argument("-prim", "--prim", default=False, action='store_true', help='Use primitive cell for bulk 1 and 2. Def: no')
    parser.add_argument("-niggli", "--niggli", default=False, action='store_true', help='Niggli reduce the resulting interface. Def: no')
    parser.add_argument("-flip2", "--flip2", default=False, action="store_true", help="Flip the slab2 upside-down.")
    parser.add_argument("-sym_slabs1", "--symSlabs1", default=False, action="store_true", help="Create symmetric surface slabs for infile1. Def: no")
    parser.add_argument("-sym_slabs2", "--symSlabs2", default=False, action="store_true", help="Create symmetric surface slabs for infile2. Def: no")
    parser.add_argument("-no_slabs", "--noSlabs",default=1, type=int, help="Number of slabs to be used for interface creation, starting from the most stable slab. Def: 1")
    parser.add_argument("-cp1", "--chem_pots1", default=None, type=str, help="Chemical potentials for the slab1, e.g. \"{'Fe':-3.0, 'O':-2.0}\" (in eV). Def: No")
    parser.add_argument("-cp2", "--chem_pots2", default=None, type=str, help="Chemical potentials for the slab2, e.g. \"{'Fe':-3.0, 'O':-2.0}\" (in eV). Def: No")
    parser.add_argument("-decomps1", "--decomps1", default=None, type=str, nargs="+", help="list of decomposition structure files of slab1. Def: No")
    parser.add_argument("-decomps2", "--decomps2", default=None, type=str, nargs="+", help="list of decomposition structure files of slab2. Def: No")
    parser.add_argument("-surf_only","--surf_only", default=False, action="store_true", help="Use to only calculate the surfaces and exit before the creation of the interfaces. Def: no")

    #Convergence tests related keywords
    parser.add_argument("-tsur", "-test_surf", "--checkSurfs", default=False, action="store_true", help="To check the stability of different surfaces for the given materials. Def: No")
    parser.add_argument("-tlay", "-test_layer", "--convLayers", default=False, action="store_true", help="To run a convergence test on the layer thickness for slab(s) generated. Def: No")
    parser.add_argument("-tsep", "-test_sep", "--convSep", default=False, action="store_true", help="To run a convergence test on the interlayer separation for the interface generated. Def: No")
    parser.add_argument("-thor", "-test_hori", "--test_hori", default=False, action="store_true", help="To run a convergence test on the relative placement of slabs w.r.t. each other for the interface generated. Def: No")
    parser.add_argument("-mlist", "--miller_list", default=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,1,1)], nargs="*", type=parse_miller, help='list of Miller indices to check during the surface stability analysis (--checkSurfs), e.g. -mlist 100 110 111 or -mlist 1,0,0 1,-1,0. Def: 100 010 001 110 111')
    parser.add_argument("-sep_init", "--sep_init", default=1.0, type=float, help='Initial interface separation to be used in seperation convergence test (--convSep), def: 1.0 A')
    parser.add_argument("-sep_final", "--sep_final", default=3.0, type=float, help='Final interface separation (included) to be used in seperation convergence test (--convSep), def: 3.0 A')
    parser.add_argument("-sep_step", "--sep_step", default=0.25, type=float, help='Step size in interface separation to be used in seperation convergence test (--convSep), def: 0.25 A')
    parser.add_argument("-lay_min", "--lay_min", default=2, type=int, help='Min layers to include n layer thickness convergence test (--convLay), def: 2 layers')
    parser.add_argument("-lay_max", "--lay_max", default=8, type=int, help='Max layers to include n layer thickness convergence test (--convLay), def: 8 layers')
    parser.add_argument("-hor_steps", "--hor_steps", default=[0.25,0.25], nargs=2, type=float, help='Horizontal translations (relative to lattice vectors) of bottom layer in x and y directions for slab placement test (--test_hori), def: [0.25, 0.25]]')

    #Calculation related keywords
    parser.add_argument("-prog", "--prog", choices=['castep','vasp','ase','mlip'], type=str.lower, required=1, help="Code to be used: CASTEP, VASP, ASE (simple LJ/Morse/EMT potentials, see -ase_pot) or MLIP (machine-learning potentials, see --mlip)")
    parser.add_argument("-xc", "--xc", default='PBE', type=str, required=False, help='Exchange correlation functional to use (pseudopots will be selected accordingly, e.g. LDA, PBE,PBEsol, etc. Def: PBE')
    parser.add_argument("-ec", "--encut", default=520, type=float, required=False, help='Cutoff energy for the planewaves Def: 520 eV')
    parser.add_argument("-kp_sp", "--kpspacing", default=0.32, type=float, required=False, help='KP grid spacing, Def: 0.05 * 2 pi = 0.32 1/A')
    parser.add_argument("-kp_gr", "--kpgrid", default=None, type=str, required=False, help="KP grid dimensions for the slabs and interfaces, e.g. '4 4 1' (Using KP spacing is recommended), Def: None")
    parser.add_argument("-gamma", "--gamma_only", default=False, action='store_true', help='Set KP grid to 1 1 1 and use only the Gamma point for surface slabs and interface. This switches on the gamma_only version of VASP.')
    parser.add_argument("-sig", "--sigma", default=0.05, type=float, required=False, help='Gaussian smearing width, Def:0.05 eV')
    parser.add_argument("-mgm", "--magmoms", default=None, nargs="*", required=False, help="Magnetic moments for a collinear calculation. Eg, 'Fe 5.0 Nb 0.6 O 0.6' Def.: None. If a defined element is not present in the POSCAR, no MAGMOM will be set for it.")
    parser.add_argument("-hubU", "--hubU", type=str, nargs="*", help="For defining the Hubbard U parameters (in eV) for specific atom and orbital types, e.g. -hubU Fe d 2.7, Sm f 6.1")
    parser.add_argument("-ase_pot", "--ase_pot", default='LJ', type=str, required=False, choices=list(POTENTIALS), help="Potential type to be used through the ASE infrastructure, only relevant when using -prog 'ASE'. Def: LJ")
    parser.add_argument("-ase_fmax", "--ase_fmax", default=0.05, type=float, required=False, help='Force convergence criterion when using the ASE built-in geometry optimiser. Def: 0.05 eV/A')
    parser.add_argument("-ase_gsteps", "--ase_gsteps", default=200, type=int, required=False, help='Max number of BFGS steps allowed in the ASE built-in geometry optimiser. Def: 200 geometry steps')
    parser.add_argument("-mp", "-makepotcar", "--makepotcar", default=False, action='store_true', help='to compile POTCAR (for VASP) using actual atomic content. The environment variable ($VASP_PP_PATH) should be defined beforehand. Default: False.')
    parser.add_argument("--potcarxc", default="potpaw_PBE", type=str, choices=_potcar_dirs(), help='XC functional to use in making of POTCAR. Def: potpaw_PBE')
    parser.add_argument("-cparam", "--castep_param", default="input.param", type=str, help='CASTEP parameter file to read (optional). Def: input.param')
    parser.add_argument("-ccell", "--castep_cell",default="input.cell", type=str, help='CASTEP cell file to read (optional). Def: input.cell')
    parser.add_argument("-cpp", "--castep_PP", default="OTF", type=str, help='CASTEP pseudopotential (PP, options are OTF, 00PBE, etc.). Def: OTF')

    #TODO: add support for  KPGrid determination

    #Parallelisation related
    parser.add_argument("-np", "--nprocs", default=32, type=int, help="No of processes to start for each CASTEP/VASP calculation through srun/mpirun. Def:32")
    parser.add_argument("-nt", "--ntasks", default=1, type=int, help="No of CASTEP/VASP tasks to run simultaneously. Def:1")
    parser.add_argument("-nn", "--nnodes", default=1, type=int, help="No of nodes to run CASTEP/VASP runs through srun. Def:1")
    parser.add_argument('-mpi','--mpirun', default=False, action='store_true', help='Use mpirun for parallel runs. Default: srun is used')
    parser.add_argument("-vexe", "--vexe", default='vasp_std', type=str, required=False, help='Vasp exectuable. Def: vasp_std')
    parser.add_argument("-mpiargs", "--mpiargs", default='', type=str, required=False, help='Extra arguments for srun/mpirun, e.g. "--partition=<name>". No default value.')

    #Other general keywords
    parser.add_argument("-dry", "--dry", default=False, action="store_true", help="Make a dry run, do not run CASTEP/VASP calculation. Good for checking the slabs and the interfacing algorithm.")
    parser.add_argument("-view", "--view", default=False, action="store_true", help="To view the generated structures.")
    parser.add_argument("-v", "--verb", default=False, action='store_true', help='Verbose output. Default: False.')
    parser.add_argument("-quick", "--quick", default=False, action='store_true', help='Setup a quick test run using ASE/LJ potentials to check functionality. Default: False.')

    #Options added in the refactored version (ASE-based engines)
    parser.add_argument("--optimiser", "--ase_opt", dest="optimiser", default="BFGS", type=str.upper, choices=list(OPTIMISERS), help="ASE optimiser for the ASE-based engines. Def: BFGS")
    parser.add_argument("--relax_cell_bulk", dest="relax_cell_bulk", default=None, action="store_true", help="ASE/MLIP engines: also relax the cell of the bulk structures (and decomposition products). Def: yes for -prog mlip, no for -prog ase (VASP/CASTEP always relax the bulk cell)")
    parser.add_argument("--no_relax_cell_bulk", dest="relax_cell_bulk", action="store_false", help="ASE/MLIP engines: keep the cell of the bulk structures fixed.")

    #Machine-learning potentials (-prog mlip)
    parser.add_argument("--mlip", default="mace", type=str, help="Machine-learning potential for -prog mlip: %s, or custom:package.module:function (any function returning an ASE calculator). Def: mace"%", ".join(MLIP_BACKENDS))
    parser.add_argument("--mlip_model", default=None, type=str, help="Model name or path for --mlip, e.g. 'medium-mpa-0' or 'my.model' (MACE), '7net-0' (SevenNet), 'uma-s-1' (fairchem). Def: the backend default")
    parser.add_argument("--device", default="auto", type=str, help="Device for the machine-learning potential: cpu, cuda, mps or auto. Def: auto (GPU if available)")
    parser.add_argument("--mlip_kwargs", default=None, type=str, help="Extra keyword arguments for the ML potential calculator as a Python dict, e.g. \"{'default_dtype': 'float32'}\" or \"{'task_name': 'omat'}\". Def: none")
    parser.add_argument("--fix_layers", default=0, type=int, help="Keep the lowest N atomic layers of the slabs fixed in the slab optimisations (bulk-like region). Def: 0")
    parser.add_argument("--outdir", default="OUTPUT", type=str, help="Directory for the output structures (the previous one is kept as <outdir>_prev). Def: OUTPUT")
    parser.add_argument("--version", action="version", version="INTERFACER %s"%__version__)
    return parser


def make_settings(args):
        """CalcSettings (and the VASP command) from the parsed command-line arguments."""
        exe=None
        if args.prog=='vasp':
                if args.mpirun: exe="mpirun -n %d %s %s"%(args.nprocs,args.mpiargs,args.vexe)
                else: exe="srun -n %d %s %s"%(args.nprocs,args.mpiargs,args.vexe)
                exe=" ".join(exe.split())
        relax_cell_bulk=args.relax_cell_bulk
        if relax_cell_bulk is None: relax_cell_bulk=(args.prog=='mlip') #MLIPs: relax the bulk cells by default
        mlip_kwargs={}
        if args.mlip_kwargs:
                try: mlip_kwargs=ast.literal_eval(args.mlip_kwargs)
                except (ValueError,SyntaxError): raise ValueError("Cannot parse --mlip_kwargs '%s' (expected a Python dict)"%args.mlip_kwargs)
                if not isinstance(mlip_kwargs,dict): raise ValueError("--mlip_kwargs must be a Python dict, got '%s'"%args.mlip_kwargs)
        settings=CalcSettings(encut=args.encut,kpspacing=args.kpspacing,xc=args.xc,sigma=args.sigma,hubU=parse_hubbard_u(args.hubU),
                              magmoms=args.magmoms,exe=exe,castep_pp=args.castep_PP,dry=args.dry,ase_pot=args.ase_pot,
                              fmax=args.ase_fmax,steps=args.ase_gsteps,optimiser=args.optimiser,relax_cell_bulk=relax_cell_bulk,
                              outdir=args.outdir,mlip=args.mlip,mlip_model=args.mlip_model,device=args.device,mlip_kwargs=mlip_kwargs)
        if args.prog == 'castep': #read the cutoff/k-point spacing from the CASTEP input files, if present
                try:
                        import ase.calculators.castep
                        calc = ase.calculators.castep.Castep()
                        calc.merge_param(args.castep_param)
                        if calc.param.cut_off_energy.value is not None: settings.encut=calc.param.cut_off_energy.value
                        settings.kpspacing=float(grep('kpoints_mp_spacing',args.castep_cell).split()[-1])
                except Exception as err: print("Can't read the param file:", err);stdout.flush()
        return settings


def main(argv=None):
        parser=build_parser()
        args = parser.parse_args(argv)

        if args.quick: #Set the parameters for quick testing 
               args.prog='ase'; 
               args.ase_pot='LJ'
               args.ase_gsteps=1 
               args.ase_fmax=0.01

        try: settings=make_settings(args)
        except ValueError as err: parser.error(str(err))
        if settings.hubU: print ("Hubbard U values to be used: ",settings.hubU);stdout.flush()
        engine=get_engine(args.prog,settings)
        set_default_engine(engine)
        if args.prog=='mlip': #load the model now, so that a missing package or model is reported before any work is done
                try: engine.get_calculator()
                except (ImportError,ValueError) as err: parser.error(str(err))
        outdir=args.outdir

        def write(atoms,base): return write_struct(atoms,base,engine,outdir)

        #k-point grid for the slabs and interfaces (None: use the k-point spacing).
        KPgrid=args.kpgrid
        if args.gamma_only:   KPgrid="1 1 1"

        infile1 = args.infile1
        miller1 = tuple(args.miller1)
        infile2= args.infile2
        miller2 = tuple(args.miller2)
        if len(miller1)!=3 or len(miller2)!=3: parser.error("-m1/-m2 need three Miller indices, e.g. -m1 1 0 0")
        miller_list=[tuple(m) for m in args.miller_list]

        print("Reading data from %s and %s."%(infile1,infile2));stdout.flush()
        atoms1 = ase.io.read(infile1) 
        atoms2 = ase.io.read(infile2) 

        outf=open(args.outfile,'w')
        outf.write(time.strftime("%a %b %d %H:%M:%S %Z %Y")+"\n")

        print("Structure 1: %-8s with fu=%d and %s lattice"%(atoms1.get_chemical_formula(empirical=1),get_fu(atoms1),atoms1.get_cell().get_bravais_lattice()));stdout.flush()
        print("Structure 2: %-8s with fu=%d and %s lattice"%(atoms2.get_chemical_formula(empirical=1),get_fu(atoms2),atoms2.get_cell().get_bravais_lattice()));stdout.flush()

        if args.prim: 
                print('Using primitive cells as input as requested');stdout.flush()
                atoms1=find_prim(atoms1); atoms2=find_prim(atoms2) #Whether to use primitive cells of input structures.

        #tolerances for creating slabs and matching two slabs as taken from user
        Lmax = args.max_slab_dimension 	#max length of cell sides
        L = args.initLength # initial length
        Lstep = args.stepLength # step size for the increasing the length in a/b direction 
        T=args.thickness #thickness of slabs
        ptol = args.percentage_tolerance	#percentage tolerances for angle and length matching

        #Keep the output structures of the previous run in <outdir>_prev (instead of deleting them).
        backup_output(outdir)

        initT=time.time()

        ########################
        # Do bulk calculations #
        ########################
        # VASP automatically checks if OUTCAR alrady exists and if run is converged (does continuation run of not)
        print("Computing bulk 1 energy.");stdout.flush()
        atoms1.name='bulk1'
        E1,atoms1=engine.run(atoms1,"geom",'bulk1',dipolCorr='None',relax_cell=True)
        print("Computing bulk 2 energy.");stdout.flush()
        atoms2.name='bulk2'
        E2,atoms2=engine.run(atoms2,"geom",'bulk2',dipolCorr='None',relax_cell=True)

        Ebulk1=E1/len(atoms1) #Get the bulk energy per atom.
        Ebulk2=E2/len(atoms2)

        write(atoms1,"bulk1_opted"); write(atoms2,"bulk2_opted")

        ###########################################################
        # Surface stability and layer thickness convergence tests #
        ###########################################################
        # These are done before the creation of the slabs, so that the selected Miller indices (-tsur)
        # and numbers of layers (-tlay) are actually used for the slabs/interfaces.
        # creps argument has no effect on symmetric surfaces, as min thickness is used 
        creps1=args.creps1 if args.creps1 else 1
        creps2=args.creps2 if args.creps2 else 1

        if args.checkSurfs:
                print('\nChecking the minimum-energy surface for Material 1 (%s), considering the Miller indices: %s'%(atoms1.get_chemical_formula(empirical=1),miller_list));stdout.flush()
                Ws1,_,_,miller1=check_surfaces(atoms1, miller_list,vac=args.vac_slab,Ebulk=Ebulk1,creps=creps1,engine=engine,kpgrid=KPgrid,outdir=outdir)
                print('\nChecking the minimum-energy surface for Material 2 (%s), considering the Miller indices: %s'%(atoms2.get_chemical_formula(empirical=1),miller_list));stdout.flush()
                Ws2,_,_,miller2=check_surfaces(atoms2, miller_list,vac=args.vac_slab,Ebulk=Ebulk2,creps=creps2,engine=engine,kpgrid=KPgrid,outdir=outdir)
                print('Selected surfaces: %s for material 1 and %s for material 2'%(miller1,miller2));stdout.flush()

        if args.convLayers:#Run layer thickness
                print('\n\nConverging layer thickness...');stdout.flush()
                results=[]
                for n,(atoms,miller) in enumerate(((atoms1,miller1),(atoms2,miller2)),1):
                        slab = make_slab(miller,atoms,repeat=(1,1,1),square=False)
                        slab.center(vacuum=0, axis=2) #to delete the vacuum padding in the slabs
                        slab.name="%s_%d%d%d"%(slab.get_chemical_formula(empirical=1),miller[0],miller[1],miller[2])
                        print ('\nRunning Material %d (%s)'%(n,slab.get_chemical_formula(empirical=1)))
                        results.append(conv_layers(slab,vac=args.vac_slab,lay_min=args.lay_min-1,lay_max=args.lay_max-1,engine=engine,outdir=outdir,ifPlot=1))
                (nl1,Eslab1,_),(nl2,Eslab2,_)=results
                print('Converged thickness (creps):');stdout.flush()
                print('\tfor slab1  is %d with E=%.2f eV'%(nl1,Eslab1));stdout.flush()
                print('\tfor slab2  is %d with E=%.2f eV'%(nl2,Eslab2));stdout.flush()
                creps1=nl1
                creps2=nl2

        #######################################################################
        #Create the intial slabs with given Miller indices (before alignment).#
        #######################################################################
        #If surface structures are provided as inputs, the structures are read from the files, otherwise they are generated using make_slab
        #(or permute_surface for symmetric slabs, cut from the optimised bulk).
        def build_slab_list(n,atoms,miller,creps,surface,sym):
                surf_list=[]
                if surface:
                        slab=standardise_slab(ase.io.read(surface)) #surface normal along z (rotation only)
                        print('Surface %d is read from %s'%(n,surface));stdout.flush()
                        surf_list.append(slab)
                elif sym: #If symmetric slabs are requested, the slabs are created using permute_surface
                        print('\nBulk%d is being symmetrized'%n);stdout.flush()
                        surf_list = permute_surface(atoms.copy(), miller, T, vacuum=args.vac_slab)
                        if len(surf_list) == 0:
                                surf_list.append(make_slab(miller,atoms,repeat=(1,1,creps),square=0))
                else:
                        surf_list.append(make_slab(miller,atoms,repeat=(1,1,creps),square=0))
                for i,slab in enumerate(surf_list): slab.name='slab%d_cut%s'%(n,i)
                return surf_list

        surf_list_slab1 = build_slab_list(1,atoms1,miller1,creps1,args.surface1,args.symSlabs1)
        surf_list_slab2 = build_slab_list(2,atoms2,miller2,creps2,args.surface2,args.symSlabs2)

        ######################################################################
        # Pre-optimisation of the slabs and surface formation energies (SFE) #
        ######################################################################
        def read_decomps(n,files):
                #Read the decomposition products; if they have no energy, optimise them (as bulk phases).
                decomps = []
                for no_d, d in enumerate(files):
                        decomp = ase.io.read(d, index=-1)
                        if decomp.calc is None:
                                decomp.name = 'decomp%d_%s'%(n,no_d)
                                _,decomp=engine.run(decomp,'geom','decomp%d_%s'%(n,no_d),dipolCorr='None',relax_cell=True)
                        decomps.append(decomp)
                return decomps

        def preopt_and_sfe(n,surf_list,bulk,chem_pots,decomp_files):
                #Pre-optimise each slab (a copy; the unoptimised slabs are used for the alignment) and compute its SFE.
                SFE_list=[]
                pas={1:'bot',2:'top'}[n] if args.pas else None
                decomps = read_decomps(n,decomp_files) if (decomp_files and not chem_pots) else None
                for i, slab in enumerate(surf_list):
                        slab_opt = slab.copy()
                        slab_opt.name='slab%d_cut%s'%(n,i)
                        slab_opt.center(vacuum=args.vac_slab, axis=2)
                        write(slab_opt,"slab%d_cut%s_pre"%(n,i))

                        print("Pre-optimizing %s (before alignment)."%slab_opt.name);stdout.flush()
                        _,slab_opt=engine.run(slab_opt,'geom','slab%d_cut%s_pre'%(n,i),dipolCorr='sc',FixCell=True,FixVol=0,passivate=pas,KPgrid_=KPgrid,
                                              fix_bottom_layers=args.fix_layers) #fixed cell: keeps the vacuum padding
                        slab_opt.name='slab%d_cut%s'%(n,i)
                        write(slab_opt,"slab%d_cut%s_pre_opted"%(n,i))

                        # check whether chem_pots or decomps are provided
                        if args.dry:
                                print("Dry run: surface formation energy not computed (set to 0).");stdout.flush()
                                SFE_list.append(0.0)
                                continue
                        elif chem_pots:
                                cp = {k: float(v) for k, v in ast.literal_eval(chem_pots).items()}
                                SFE = calculate_SFE(slab_opt, bulk, chem_pots=cp)
                        elif decomps:
                                SFE = calculate_SFE(slab_opt, bulk, decomps=decomps)
                        # if nothing is supplied try SFE calculation for stoichiometric slabs (zero chemical potentials for ASE test runs)
                        else:
                                SFE = calculate_SFE(slab_opt, bulk, ase=(args.prog=='ase'))
                        print("Surface formation energy of %s: %.2f eV/nm^2 = %.2f J/m^2\n"%(slab_opt.name, SFE, SFE*cfactor));stdout.flush()
                        SFE_list.append(SFE)
                return SFE_list

        SFE_list_slab1 = preopt_and_sfe(1,surf_list_slab1,atoms1,args.chem_pots1,args.decomps1)
        SFE_list_slab2 = preopt_and_sfe(2,surf_list_slab2,atoms2,args.chem_pots2,args.decomps2)

        # sort the slabs by their SFE values and keep the args.noSlabs lowest energy slabs
        low_SFEs_slab1 = sorted(enumerate(SFE_list_slab1), key=lambda x: x[1])[:args.noSlabs]
        low_SFEs_slab2 = sorted(enumerate(SFE_list_slab2), key=lambda x: x[1])[:args.noSlabs]
        le_surf_slab1 = [surf_list_slab1[i] for i, _ in low_SFEs_slab1]
        le_surf_slab2 = [surf_list_slab2[i] for i, _ in low_SFEs_slab2]
        indices_le_s1 = [i for i,_ in low_SFEs_slab1] 
        indices_le_s2 = [i for i,_ in low_SFEs_slab2]

        for n,sym,surfs,low in ((1,args.symSlabs1,le_surf_slab1,low_SFEs_slab1),(2,args.symSlabs2,le_surf_slab2,low_SFEs_slab2)):
                if sym: print("\nThe %s lowest energy slabs for material %d are:"%(args.noSlabs,n));stdout.flush()
                else: print("\nThe surface formation energy of slab%d_cut0 is:"%n);stdout.flush()
                for surf, (_,val) in zip(surfs, low):
                        print("%s: %.2f eV/nm^2"%(surf.name, val));stdout.flush()

        # if keyword provided, exit safely after slab calculations
        if args.surf_only: 
                print('Exiting code after the optimisation of the surface slabs');stdout.flush()
                outf.close()
                return 0

        for i, slab1_in in enumerate(le_surf_slab1):
                for j, slab2_in in enumerate(le_surf_slab2):
                        #Work on copies, so that each pair starts from the original (unaligned, unoptimised) slabs.
                        slab1=slab1_in.copy(); slab1.name=slab1_in.name
                        slab2=slab2_in.copy(); slab2.name=slab2_in.name
                        ci=indices_le_s1[i]; cj=indices_le_s2[j]

                        print("\nMisfit (mu) of %s and %s (before alignment): %.2f%%"%(slab1.name, slab2.name, misfit(slab1,slab2)*100));stdout.flush()

                        ######################
                        # Alignment of Slabs #
                        ######################
                        print("\nAligning the two slabs...");stdout.flush()
                        try:
                                slab1,slab2=slab_aligner(slab1,slab2,L,Lmax,Lstep,ptol,T,atoms1,atoms2)
                        except RuntimeError as err:
                                print("Slab alignment failed for %s and %s: %s"%(slab1.name,slab2.name,err));stdout.flush()
                                continue
                        slab1.name='slab1_cut%s_aligned'%ci
                        slab2.name='slab2_cut%s_aligned'%cj
                        print("\nMisfit (mu) of %s and %s (after alignment): %.2f%%"%(slab1.name, slab2.name, misfit(slab1,slab2)*100));stdout.flush()

                        if args.view: view(slab1);view(slab2)
                        if args.flip2: #Flip the slab2 (top slab) upside-down. still the cell angles cannot be preserved !! (does not work reliably)
                                slab2.rotate(a=180,v='x',rotate_cell=0,center = (0, 0, 0))
                                slab2.center()
                                if args.view: view(slab2)

                        write(slab1,"slab1_cut%s_aligned"%ci); write(slab2,"slab2_cut%s_aligned"%cj)

                        #Interface before optimizing the individual slabs.
                        sep=args.sep
                        vac_used=args.vac_int #vacuum padding (on each side) of the final interface
                        interface=make_interface(slab1,slab2,sep,args.vac_int,reorder=True)
                        if args.view:view(interface)
                        write(interface,"interface_cuts_%s_%s"%(ci,cj))

                        #Calculate energies/structures of the actual slabs (after alignment).
                        print("\nOptimizing %s and %s (after alignment)."%(slab1.name, slab2.name));stdout.flush()
                        slab1.center(vacuum=args.vac_slab, axis=2) #to add vacuum to the slabs (needed)
                        slab2.center(vacuum=args.vac_slab, axis=2)

                        Eslab1,slab1=engine.run(slab1,'geom','slab1_cut%s_aligned'%ci,dipolCorr='sc',FixCell=True,FixVol=0,passivate=('bot' if args.pas else None),
                                                nosymm=1,vac=args.vac_slab,KPgrid_=KPgrid,fix_bottom_layers=args.fix_layers)
                        Eslab2,slab2=engine.run(slab2,'geom','slab2_cut%s_aligned'%cj,dipolCorr='sc',FixCell=True,FixVol=0,passivate=('top' if args.pas else None),
                                                nosymm=1,vac=args.vac_slab,KPgrid_=KPgrid,fix_bottom_layers=args.fix_layers)

                        #Compute the surface energies (using bulk energy per atom).
                        Ws1=(Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
                        Ws2=(Eslab2-len(slab2)*Ebulk2)/2/surf_area(slab2)/0.01 #A2 to nm2

                        str1='\nCalculating the W_surf (surface formation energies / surface area) of the optimised aligned slabs...\n'
                        str1+='%s: %.3f eV/atom\n' % ('Ebulk_1',Ebulk1)
                        str1+='%s: %.3f eV/atom\n' % ('Ebulk_2', Ebulk2)
                        str1+='%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('Wsurf_1', slab1.get_chemical_formula(empirical=1),miller1[0],miller1[1],miller1[2],Ws1,Ws1*cfactor)
                        str1+='%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('Wsurf_2', slab2.get_chemical_formula(empirical=1),miller2[0],miller2[1],miller2[2],Ws2,Ws2*cfactor)
                        print(str1);stdout.flush()
                        outf.writelines(str1)

                        write(slab1,"slab1_cut%s_aligned_opted"%ci); write(slab2,"slab2_cut%s_aligned_opted"%cj)
                        if args.view:view(slab1);view(slab2)

                        #Create the interface from the optimised slabs.
                        interface=make_interface(slab1,slab2,sep,args.vac_int,reorder=True)
                        if args.verb: 
                                a,b,c,alpha,beta,gamma=interface.cell.cellpar()
                                print (a,b,c,alpha,beta,gamma);stdout.flush()
                        if args.view:view(interface)
                        write(interface,"interface-pre_opt_cuts_%s_%s"%(ci,cj))

                        if args.convSep: #Run optimisation on the interface speration distance
                                print("\n\nFinding the optimal interface (interlayer) separation.");stdout.flush()
                                #separation/2 is used as vacuum padding to have a symmetric seperation at all interfaces.
                                interface,sep=convSep(slab1,slab2,vac=None,sep_init=args.sep_init,sep_final=args.sep_final,sep_step=args.sep_step,view=0,ifPlot=1,
                                                      engine=engine,kpgrid=KPgrid,outdir=outdir)
                                vac_used=sep/2

                        if args.test_hori: #Optimise the horizontal stacking of the layers by translating slab
                                print("\n\nFinding the optimial interface lateral shift.");stdout.flush()
                                interface,Eint,minxy=checkHorizontal(slab1,slab2,sep=sep,steps=args.hor_steps,ifPlot=1,engine=engine,kpgrid=KPgrid,outdir=outdir)
                                vac_used=sep/2

                        if args.niggli: print ("Niggli reduce the interface...");niggli_reduce(interface);stdout.flush()
                        if args.view:view(interface)

                        #Optimise the final interface geometry
                        print("\nOptimizing the final interface geometry.");stdout.flush()
                        interface.name='interface_cuts_%s_%s'%(ci,cj)
                        dipC='SC' if vac_used>4 else 'None'
                        #CHECK: Should we fix the volume or not?? fix vol works for some cases (YIG) but not for some (Al2O3//Si)
                        Eint,interface=engine.run(interface,'geom',interface.name,dipolCorr=dipC,FixVol=0,nosymm=1,KPgrid_=KPgrid)

                        #Work of adhesion: W_ad = (E_slab1 + E_slab2 - E_interface) / (n_int * A)  (positive for a bound interface).
                        #With a small vacuum padding (2*vac_used across the periodic boundary, e.g. vac=sep/2 as used in -tsep/-thor),
                        #the cell contains two interfaces (and no free surfaces), so n_int=2; with a large vacuum (>4 A on each side,
                        #where also the dipole correction is switched on) there is one interface and two free surfaces, so n_int=1.
                        n_int = 2 if vac_used<=4 else 1
                        Wad=(Eslab1+Eslab2-Eint)/(n_int*surf_area(interface))/0.01 #A2 to nm2
                        str1='W_ad (work of adhesion, %d interface(s) in the cell) for the final optimised interface: %.2f eV/nm^2 = %.2f J/m^2\n'%(n_int,Wad,Wad*cfactor)
                        print(str1);stdout.flush()
                        outf.writelines(str1)

                        write(interface,"interface_cuts_%s_%s_opted"%(ci,cj))
                        ase.io.write(os.path.join(outdir,"interface_cuts_%s_%s_opted.cif"%(ci,cj)),interface,format='cif')

        outf.close()
        print ("Elapsed time: %.2f sec."%( time.time()-initT));stdout.flush()
        return 0


if __name__ == "__main__":
        raise SystemExit(main())

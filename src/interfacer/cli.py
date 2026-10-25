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

import ase.io

from . import __version__
from .calculators import CalcSettings, get_engine
from .calculators.classical import POTENTIALS
from .calculators.mlip import BACKENDS as MLIP_BACKENDS
from .calculators.optimise import OPTIMISERS
from .convergence import set_default_engine
from .log import say, setup_logging, warn
from .workflow import InterfaceWorkflow, config_from_args


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
    parser.add_argument("-i1", "--infile1", required=1, help="Bulk structure of material 1 (bottom slab); any format ASE can read (cif, POSCAR, res, xyz, ...).")
    parser.add_argument("-i2", "--infile2", required=1, help="Bulk structure of material 2 (top slab).")
    parser.add_argument("-o", "--outfile", default="interfaces.out", help="Summary file with the surface energies and works of adhesion. Def: interfaces.out")
    parser.add_argument("-m1", "--miller1", default=(1,0,0), nargs="+", type=int, help="Miller indices of the surface of material 1, e.g. -m1 1 1 0. Def: 1 0 0")
    parser.add_argument("-m2", "--miller2", default=(1,0,0), nargs="+", type=int, help="Miller indices of the surface of material 2. Def: 1 0 0")
    parser.add_argument("-surf1", "--surface1", default=False, help="extra input file for a surface structure of i1")
    parser.add_argument("-surf2", "--surface2", default=False, help="extra input file for a surface structure of i2")

    #Keywords related to structure building
    parser.add_argument("-cr1", "--creps1", default=None, type=int, help='Repetitions in c direction (number of layers) for surface model 1')
    parser.add_argument("-cr2", "--creps2", default=None, type=int, help='Repetitions in c direction (number of layers) for surface model 2')
    parser.add_argument("-msd", "--max_slab_dimension", default=50, type=float, help='Maximum length (A) of the in-plane cell vectors in the search for a commensurate supercell. Def: 50 A')
    parser.add_argument("-th", "--thickness", default=7, type=float, help='Minimum thickness of the symmetric slabs (-sym_slabs1/2), def: 7A. The thickness of the other slabs is set by -cr1/-cr2.')
    parser.add_argument("-pt", "--percentage_tolerance", default=4, type=float, help='Tolerance (%%) for matching the in-plane angles and lengths of the two slabs (i.e. the maximum strain). Def: 4')
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
    parser.add_argument("--sym_method", default="pymatgen", choices=["pymatgen","internal"], help="Generator of the symmetric slabs (-sym_slabs1/2): pymatgen's SlabGenerator or INTERFACER's own implementation. Def: pymatgen")
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
    parser.add_argument("-v", "--verb", default=False, action='store_true', help='Verbose output: also show details (supercell search, optimiser steps, passivation, ...). Default: False.')
    parser.add_argument("-q", "--quiet", default=False, action='store_true', help='Only show warnings and errors. Default: False.')
    parser.add_argument("-quick", "--quick", default=False, action='store_true', help='Setup a quick test run using ASE/LJ potentials to check functionality. Default: False.')

    #Options added in the refactored version (ASE-based engines)
    parser.add_argument("--optimiser", "--ase_opt", dest="optimiser", default="BFGS", type=str.upper, choices=list(OPTIMISERS), help="ASE optimiser for the ASE-based engines. Def: BFGS")
    parser.add_argument("--relax_cell_bulk", dest="relax_cell_bulk", default=None, action="store_true", help="ASE/MLIP engines: also relax the cell of the bulk structures (and decomposition products). Def: yes for -prog mlip, no for -prog ase (VASP/CASTEP always relax the bulk cell)")
    parser.add_argument("--no_relax_cell_bulk", dest="relax_cell_bulk", default=None, action="store_false", help="ASE/MLIP engines: keep the cell of the bulk structures fixed.")

    #Machine-learning potentials (-prog mlip)
    parser.add_argument("--mlip", default="mace", type=str, help="Machine-learning potential for -prog mlip: %s, or custom:package.module:function (any function returning an ASE calculator). Def: mace"%", ".join(MLIP_BACKENDS))
    parser.add_argument("--mlip_model", default=None, type=str, help="Model name or path for --mlip, e.g. 'medium-mpa-0' or 'my.model' (MACE), '7net-0' (SevenNet), 'uma-s-1' (fairchem). Def: the backend default")
    parser.add_argument("--device", default="auto", type=str, help="Device for the machine-learning potential: cpu, cuda, mps or auto. Def: auto (GPU if available)")
    parser.add_argument("--mlip_kwargs", default=None, type=str, help="Extra keyword arguments for the ML potential calculator as a Python dict, e.g. \"{'default_dtype': 'float32'}\" or \"{'task_name': 'omat'}\". Def: none")
    parser.add_argument("--fix_layers", default=0, type=int, help="Keep the lowest N atomic layers of the slabs fixed in the slab optimisations (bulk-like region). Def: 0")
    restart=parser.add_mutually_exclusive_group()
    restart.add_argument("--restart", dest="restart", default=None, action="store_true", help="Reuse finished calculations, also of the convergence tests (VASP/CASTEP: default; ASE/MLIP: results are kept in ASE-calcs/ and reused if the input structure and settings are unchanged).")
    restart.add_argument("--overwrite", default=False, action="store_true", help="Redo all calculations; existing VASP run directories and CASTEP outputs are moved to *_old<n> (not deleted).")
    parser.add_argument("--outdir", default="OUTPUT", type=str, help="Directory for the output structures (the previous one is kept as <outdir>_prev). Def: OUTPUT")
    parser.add_argument("--version", action="version", version="INTERFACER %s"%__version__)

    #Input files
    parser.add_argument("--input", "-inp", default=None, type=str, help="Input file (TOML, YAML or JSON) with options; keys are the long option names (e.g. infile1, miller1, prog, convSep). Options given on the command line override the file.")
    parser.add_argument("--save_input", default=None, type=str, help="Write the effective options of this run to a TOML/YAML/JSON file (by extension) that can be used with --input.")
    return parser


INPUT_EXCLUDE = {"help", "version", "input", "save_input"}

def load_input_file(path):
        """Read an input file (TOML, YAML or JSON, by extension) into a dict."""
        ext=os.path.splitext(path)[1].lower()
        if ext in (".yaml",".yml"):
                try: import yaml
                except ImportError: raise ValueError("YAML input files need PyYAML (pip install pyyaml); TOML and JSON work without it.")
                with open(path) as f: data=yaml.safe_load(f) or {}
        elif ext==".json":
                import json
                with open(path) as f: data=json.load(f)
        else:
                try: import tomllib
                except ImportError: import tomli as tomllib  # Python < 3.11
                with open(path,"rb") as f: data=tomllib.load(f)
        if not isinstance(data,dict): raise ValueError("Input file %s must contain a table/mapping of options."%path)
        #allow optional sections, e.g. [structures], [calculation] -> flatten
        flat={}
        for k,v in data.items():
                if isinstance(v,dict) and k not in ("chem_pots1","chem_pots2","mlip_kwargs","cp1","cp2"): flat.update(v)
                else: flat[k]=v
        return flat

def _option_index(parser):
        #name (dest, long or short option without dashes) -> argparse action
        idx={}
        for a in parser._actions:
                if a.dest in INPUT_EXCLUDE or a.dest==argparse.SUPPRESS: continue
                idx.setdefault(a.dest,a) #first action wins (e.g. relax_cell_bulk vs. no_relax_cell_bulk)
                for opt in a.option_strings: idx.setdefault(opt.lstrip("-"),a)
        return idx

def _convert(action,value):
        """Convert a value from an input file to what argparse would produce for this option."""
        if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction)):
                if not isinstance(value,bool): raise ValueError("option '%s' expects true or false, got %r"%(action.option_strings[-1].lstrip("-"),value))
                return value if isinstance(action,argparse._StoreTrueAction) else (not value)
        def one(v):
                if isinstance(v,dict): v=repr(v) #e.g. chem_pots1, mlip_kwargs (parsed later with ast.literal_eval)
                if action.type is None: return v if not isinstance(v,(list,tuple)) else v
                if action.type is parse_miller: return parse_miller(v if isinstance(v,str) else ",".join(str(x) for x in v))
                if isinstance(v,bool) and action.type in (int,float): raise ValueError("option '%s' expects a number"%action.dest)
                return action.type(v) if not isinstance(v,(list,tuple)) else action.type(" ".join(str(x) for x in v))
        if action.nargs in ("+","*") or isinstance(action.nargs,int):
                if isinstance(value,str): value=value.split()
                elif not isinstance(value,(list,tuple)): value=[value]
                return [one(v) for v in value]
        return one(value)

def apply_input_file(parser,path):
        """Use the options of an input file as defaults of the parser (command-line options still override them)."""
        values=load_input_file(path)
        idx=_option_index(parser)
        unknown=sorted(k for k in values if k not in idx)
        if unknown: raise ValueError("Unknown option(s) in %s: %s"%(path,", ".join(unknown)))
        defaults={}
        for k,v in values.items():
                a=idx[k]
                defaults[a.dest]=_convert(a,v)
                a.required=False #given in the file
        parser.set_defaults(**defaults)
        return defaults

def save_input_file(args,path,parser=None):
        """Write the parsed options (those differing from the defaults, plus the required ones) to a TOML/YAML/JSON file."""
        parser=parser or build_parser()
        defaults={a.dest:parser.get_default(a.dest) for a in parser._actions} #get_default: as argparse resolves shared dests
        data={}
        for k,v in sorted(vars(args).items()):
                if k in INPUT_EXCLUDE or v is None: continue
                if k in ("infile1","infile2","prog") or v!=defaults.get(k):
                        if isinstance(v,tuple): v=list(v)
                        if isinstance(v,list): v=[list(x) if isinstance(x,tuple) else x for x in v]
                        data[k]=v
        ext=os.path.splitext(path)[1].lower()
        with open(path,"w") as f:
                if ext==".json":
                        import json; json.dump(data,f,indent=1)
                elif ext in (".yaml",".yml"):
                        import yaml; yaml.safe_dump(data,f,sort_keys=True)
                else:
                        f.write("# INTERFACER input file (use with: interfacer --input %s)\n"%os.path.basename(path))
                        for k,v in data.items(): f.write("%s = %s\n"%(k,_toml_value(v)))
        return data

def _toml_value(v):
        if isinstance(v,bool): return "true" if v else "false"
        if isinstance(v,(int,float)): return repr(v)
        if isinstance(v,str): return '"%s"'%v.replace("\\","\\\\").replace('"','\\"')
        if isinstance(v,(list,tuple)): return "[%s]"%", ".join(_toml_value(x) for x in v)
        raise ValueError("Cannot write %r to TOML"%(v,))


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
                              outdir=args.outdir,restart=args.restart,overwrite=args.overwrite,mlip=args.mlip,mlip_model=args.mlip_model,device=args.device,mlip_kwargs=mlip_kwargs)
        if args.prog == 'castep': #read the cutoff/k-point spacing from the CASTEP input files, if present
                try:
                        import ase.calculators.castep
                        calc = ase.calculators.castep.Castep()
                        calc.merge_param(args.castep_param)
                        if calc.param.cut_off_energy.value is not None: settings.encut=calc.param.cut_off_energy.value
                        settings.kpspacing=float(grep('kpoints_mp_spacing',args.castep_cell).split()[-1])
                except Exception as err: warn("Can't read the param file:", err)
        return settings


def main(argv=None):
        """Run INTERFACER with command-line arguments (sys.argv if argv is None). Returns 0 on success."""
        parser=build_parser()
        #An input file (--input) provides defaults; options on the command line override them.
        pre=argparse.ArgumentParser(add_help=False); pre.add_argument("--input","-inp",default=None)
        known,_=pre.parse_known_args(argv)
        if known.input:
                try: apply_input_file(parser,known.input)
                except (OSError,ValueError,TypeError,argparse.ArgumentTypeError) as err: parser.error("Cannot use input file %s: %s"%(known.input,err))
        args = parser.parse_args(argv)
        setup_logging(1 if args.verb else (-1 if args.quiet else 0))
        if args.save_input:
                save_input_file(args,args.save_input,build_parser())
                say("Options written to %s"%args.save_input)

        if args.quick: #Set the parameters for quick testing 
               args.prog='ase'; 
               args.ase_pot='LJ'
               args.ase_gsteps=1 
               args.ase_fmax=0.01

        if len(args.miller1)!=3 or len(args.miller2)!=3: parser.error("-m1/-m2 need three Miller indices, e.g. -m1 1 0 0")
        try:
                settings=make_settings(args)
                config=config_from_args(args)
        except (ValueError,SyntaxError) as err: parser.error(str(err))
        if settings.hubU: say("Hubbard U values to be used: ",settings.hubU)

        engine=get_engine(args.prog,settings)
        set_default_engine(engine)
        if args.prog=='mlip': #load the model now, so that a missing package or model is reported before any work is done
                try: engine.get_calculator()
                except (ImportError,ValueError) as err: parser.error(str(err))

        say("Reading data from %s and %s."%(args.infile1,args.infile2))
        atoms1 = ase.io.read(args.infile1)
        atoms2 = ase.io.read(args.infile2)

        InterfaceWorkflow(engine,config).run(atoms1,atoms2)
        return 0


if __name__ == "__main__":
        raise SystemExit(main())

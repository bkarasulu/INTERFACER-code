"""CASTEP driver (formerly tools/external.py): call_castep and the CASTEP output parsers."""
import os
from os import chdir, getenv, system
from re import search

import numpy as np
import ase
import ase.io
import ase.calculators.castep
from ase import Atoms
from ..log import debug, say, warn

def call_castep(atoms,calc=None, typ="sp",PP='',wDir='./CASTEP-tmp',name='try',param='opt.param',resDir="",dipolCorr=False,dipolDir='z',KPgrid="1 1 1",KPspacing="", ifPrint=False,ifDryRun=False,ENCUT=0,ifManualRun=True,FixCell=False,FixList=None,hubU=None,slowConv=0):
    #For applying constraints (FixList) atom numbering starts from 1.
    FixList=FixList or []; hubU=hubU or {}

        
    #exe="mpirun -n 4 castep";PP_path='/rscratch/bk393/pspots/CASTEP'
    #exe="mpirun -n 20 castep";PP_path='/u/fs1/bk393/pspots/CASTEP'

    #system("export CASTEP_COMMAND='%s'"%exe)
    #system("export CASTEP_COMMAND='mpirun -n 4 castep'")
    #system("export CASTEP_PP_PATH='%s'"%PP_path)

    #exe=popen('echo "$CASTEP_COMMAND"',"r").read()[0:-1]
    #PP_path=popen('echo "$CASTEP_PP_PATH"',"r").read()[0:-1]
    exe=getenv('CASTEP_COMMAND')
    PP_path=getenv('CASTEP_PP_PATH')

    if calc!=None:
            #This is good for assiging the settings from a previously created Calculator object (i.e. not to repeat all the settings the second time). None of the other options will be taken into account.
            atoms.calc=calc
            #manual CASTEP run icin ayri bir function yap ve burda ve asagida cagir.
            return atoms.get_potential_energy(),atoms #same (E, atoms) return format as the other paths

    calc = ase.calculators.castep.Castep()

    #Assign the environment variables.
    calc._castep_command=exe
    calc._castep_pp_path=PP_path
    
    # include interface settings in .param file
    calc._export_settings = True

    # reuse the same directory
    calc._directory = wDir
    calc._rename_existing_dir = False
    calc._label = name

    
    if param:
        #Read paramters from param file input.
        calc.merge_param(param)
    else:        
        # Use default param settings (depreceated)
        calc.param.xc_functional = 'PBE'
        calc.param.cut_off_energy = 100 #500
        calc.param.num_dump_cycles = 0
        calc.param.geom_method = "lbfgs"
        calc.param.geom_max_iter= 10
        calc.param.write_cell_structure=True
        calc.param.spin_polarised=False
        calc.param.opt_strategy="speed"
        calc.param.mix_history_length=20
        calc.param.max_scf_cycles=100
        calc.param.calculate_stress=True
        #calc.param.finite_basis_corr=0
    
    # Cell settings
    #
    calc.cell.symmetry_generate=True
    calc.cell.snap_to_symmetry=True
    if KPspacing:calc.cell.kpoints_mp_spacing = str(KPspacing) #default=0.05 2*pi/A
    else: 
            if not KPgrid: raise ValueError("call_castep: either KPspacing or KPgrid must be given.")
            calc.cell.kpoint_mp_grid = KPgrid #def='1 1 1'
            kpgrid=[float(x) for x in KPgrid.split()]
            calc.cell.kpoints_mp_offset='%.4f %.4f %.4f'%(1./2./kpgrid[0],1./2./kpgrid[1],1./2./kpgrid[2]) #for placing Gamma point in the center.

    
    #calc.cell.fix_com = False
    if FixCell: calc.cell.fix_all_cell = True
    if len(FixList)!=0:
            #c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu'])
            #c = FixAtoms(indices=FixList)
            #atoms.set_constraint(c) #This only work if the CASTEP is called by ASE (not for manual runs).

            #CASTEP expects the ion index within its species (1-based), not the global atom index.
            symbols=atoms.get_chemical_symbols()
            str1=[]
            i=1
            for at in FixList:
                    atom=atoms[at]
                    sp_ind=symbols[:atom.index+1].count(atom.symbol)
                    str1.append("%d %s %d %.8f %.8f %.8f"%(i,atom.symbol,sp_ind,1,0,0))
                    str1.append("%d %s %d %.8f %.8f %.8f"%(i+1,atom.symbol,sp_ind,0,1,0))
                    str1.append("%d %s %d %.8f %.8f %.8f"%(i+2,atom.symbol,sp_ind,0,0,1))
                    i+=3

            calc.cell.ionic_constraints=str1 #a list object needed as input.
            calc.cell.snap_to_symmetry=False
            calc.cell.symmetry_generate=False

    if len(hubU)!=0: #is not None:
            ats=sorted(hubU.keys())
            #str1="%BLOCK HUBBARD_U \n  eV"
            str2=["eV"]
            asyms=atoms.get_chemical_symbols()
            for at in ats:
                    if at not in asyms: continue
                    str1="%3s "%at
                    for orb in sorted(hubU[at].keys()):
                            str1+="%2s: %s "%(orb,hubU[at][orb])
                    str2.append(str1)
            #str1+='\n %ENDBLOCK HUBBARD_U'
            #print(str1)
            calc.cell.hubbard_u=str2
     
    #This overwrites the task paramter from the param input.
    if typ.lower()=="sp":    calc.param.task = 'SinglePoint'
    elif typ.lower()=="geom":calc.param.task = 'GeometryOptimization'
    else: raise ValueError("call_castep: typ must be 'sp' or 'geom', got '%s'"%typ)
    
    if dipolCorr: #def: No dipole corrections. 
        if dipolCorr.upper()=="SC": calc.param.dipole_correction= "SELFCONSISTENT"
        elif dipolCorr=="static": calc.param.dipole_correction= "static"
        else: calc.param.dipole_correction= "None"
        calc.param.dipole_dir=dipolDir #options: x,y,z and a (only energy-corr)

    if slowConv:
            calc.param.mix_charge_amp=0.1
        
    
    #calc.initialize()#Creates all needed input in the _directory. (Not needed for actual run.)
    atoms.calc=calc  #Set for the previously created interface

    if ENCUT!=0: calc.param.cut_off_energy=ENCUT
    
    if PP!="":#otherwise on-the-fly(OTF) is used as default
        fnc=str(calc.param.xc_functional).split("\n")[1].upper()
        #print fnc
        PP=PP.upper()
        #print PP.lower().find(str(calc.param.xc_functional).lower())
        if PP != "OTF" and  PP.find(fnc)== -1:
                raise ValueError("Selected CASTEP pseudopotential (%s) does not match the XC functional being used (%s)."%(PP,fnc))
        elif PP=="OTF": None #default is OTF anyway.
        else: atoms.calc.set_pspot(PP)  #This automatically sets the pseudo-potential for all present species to <Species>_<library>.usp. Make sure that _castep_pp_path is set correctly in the shell.


                            

    # Or Read all settings from previous calculation
    if resDir != "": #Does not read from checkpoint file for some reason, needs to be checked !!
        # Reset to CASTEP default
        #atoms.calc.param.task.clear()
        atoms = ase.io.castep.read_seed('%s/%s' % (wDir,name))
        calc.param.reuse=True
        calc.param._try_reuse=True
        calc._export_settings = True
        #print atoms.calc,"bora\n\n",calc
        
    if ifPrint:say(calc) #prints calculation summary.

    # necessary for tasks with changing positions
    # such as GeometryOptimization or MolecularDynamics (This option does not work as deisgnated !!! The atomic coords are not updated at the end of the geom. opt. unlike the energy.
    calc._set_atoms = True
    atoms.calc._set_atoms = True
    
    
    # Check for correct input
    if not ifDryRun:
            if ifManualRun: #If a manual run of the CASTEP is needed.
                    #str1="%s %s/%s"%(exe,wDir,name)
                    str1="%s %s"%(exe,name)  
                    #TODO: Add here the run3 option !
                    say("Running ",str1)

                    calc._copy_pspots=True
                    calc.initialize()

                    cwd=os.getcwd()
                    chdir(wDir)
                    
                    system(str1) #PSPOT missing in the folder
                    #x=parseCASTEP("%s/%s.geom"%(wDir,name),atoms=atoms)
                    task=str(atoms.calc.param.task).split()[-1]
                    debug(task)
                    if task=='SinglePoint' : #use <seed>.castep file
                            x=parseCASTEP("%s.castep"%(name),atoms=atoms)
                    elif task=='GeometryOptimization': #use <seed>.geom file.
                            try:x=parseCASTEP("%s.geom"%(name),atoms=atoms) #in case no ptimisation is done (that could happen for 2D systems, where the slab is equivalent to the bulk str, so no geom opt is needed/possible.)
                            except:x=None

                            if x == None: x=parseCASTEP("%s.castep"%(name),atoms=atoms);#print ("bk",atoms)
                            if x == None:
                                    chdir(cwd)
                                    raise RuntimeError("call_castep: could not parse the results of %s (see %s/%s.castep)"%(name,wDir,name))

                            if x[-2]==False: warn("parseCASTEP: WARNING: Geometry optimization in %s.geom is not converged!"%name)
                    else:
                            chdir(cwd)
                            raise ValueError("call_castep: calculation type %s is not supported."%task)
                    chdir(cwd)
                    return x
                    
            else: #CASTEP calculation is not done here. It will be called in the main script, when trying to reach the attributes, e.g. atoms.get_potential_energy().
                    return None,atoms
    else:
            if calc.dryrun_ok():
                    return 0.0,atoms #dry run: no energy
            else:
                    raise RuntimeError("CASTEP dry run: found error in input: %s"%calc._error)
            
def parseCASTEP(fname,atoms=None):      #This function has become obselete
        #TODO: fix wrong atomic order (if the atoms object is ordered).
        #Read the CASTEP output to retrieve final 0K energy, atomic coords (Cart. and fractional), and forces.
        bohr2ang=0.52917721
        Ha2eV=27.211386
        atP2GPa=0.29421912e5 #P in atomic units to GPa.
        E=0.0;H=0.0 #Total Energy and Enthalpy
        h=[] #unit cell vectors
        s=[] # stress tensor
        xyz=[] #Cartesian atomic coords.
        forces=[] #forces in au (i.e. Ha/bohrradius).
        ifConv=0 #if the geometry is converged.
        fract=[]
        ids=[] #atomic id's

        debug("Parsing ",fname)
        if fname.split(".")[-1]=="geom":
                with open(fname) as f: tSteps=sum(1 for ln in f if "<-- c" in ln)-1

                flag=0
                for ln in open(fname,'r'):#.readlines(): #readlines is outdated and slow.
                        ln=ln[0:-1]
                        if not flag and search("<-- c",ln):
                                if int(ln.split()[0])==tSteps:
                                        #print 'here'
                                        flag=1
                                        if search("T   T   T   T",ln): ifConv=1

                        elif flag:
                                if search("<-- E",ln):E=float(ln.split()[1])*Ha2eV; H=float(ln.split()[1])*Ha2eV
                                elif search("<-- h",ln): h.append([float(i)*bohr2ang for i in ln.split()[0:3]])
                                elif search("<-- S",ln): s.append([float(i)*atP2GPa for i in ln.split()[0:3]])
                                elif search("<-- R",ln): 
                                        x= ln.split(); 
                                        xyz.append([float(i)*bohr2ang for i in x[2:5]]); 
                                        ids.append(x[0]); 
                                elif search("<-- F",ln): forces.append([float(i)*(Ha2eV/bohr2ang) for i in ln.split()[2:5]])
                                
                                
       
        elif fname.split(".")[-1]=="castep":
                tSteps=0;latFlag=0;forces=[];s=[];h=[];latFlag=0;fract=[];ids=[]
                for ln in open(fname,'r'):#.readlines(): #readlines is outdated and slow.
                        ln=ln[0:-1]
                        #if ln=="":continue
                        if len(ln)<=2: continue
                        elif "Unit Cell" in ln: #for getting only the last one
                                tSteps+=1
                                forces=[];s=[];h=[];latFlag=0;fract=[];ids=[]
                        elif search(r"Final free energy \(E-TS\)    =",ln):H=float(ln.split("=")[1].split()[0])
                        elif search(r"Final energy, E",ln): E=float(ln.split("=")[1].split()[0])
                        elif search(r"\*  x ",ln) or search(r"\*  y ",ln) or search(r"\*  z ",ln): s.append([float(i) for i in ln.split()[2:5]]) #Stress tensor already in GPa.
                        elif ln.lstrip().startswith("*") and len(ln.split())==7: #force block lines: '*  Li  1  fx fy fz  *'
                                forces.append([float(i) for i in ln.split()[2:5]]) #already in eV/A
                        elif "            x" in ln and len(ln.split())==7:
                                fract.append([float(i) for i in ln.split()[3:6]])
                                ids.append(ln.split()[1])
                        elif "Real Lattice" in ln: latFlag=1;continue
                        elif "Lattice parameters" in ln: latFlag=0
                        elif  "Geometry optimization completed successfully." in ln: ifConv=1

                        if latFlag:  
                                if ln=="":latFlag=0;continue
                                h.append([float(i) for i in ln.split()[0:3]])
                                

                #Assuming it is a SP calculation (see call_CASTEP() function), so initial positions are not changed.
                
                #if len(h)==0: print("Hey !!"); h=atoms.get_cell()#.tolist()  #if no Lattice infoin castep file (not likely).
                
                #xyz=atoms.get_positions()#.tolist()
                
        if len(ids)==0: 
                warn(ids);warn("parseCASTEP: WARNING: No chemical symbol info in %s."%fname); return None
        elif len(h)==0: 
                warn("parseCASTEP: WARNING: No cell info in %s."%fname); return None
        elif len(xyz)==0 and len(fract)==0: 
                warn("parseCASTEP: WARNING: No Cartesian or fractional coordinate info in %s."%fname); return None
        
                                                
        if atoms != None:
                #print(ids,h,atoms.calc,xyz,fract)
                #print (atoms.get_chemical_symbols())
                atoms.set_chemical_symbols(ids)
                atoms.set_cell(np.array(h))
                if len(xyz)!=0:atoms.set_positions(np.array(xyz))
                elif len(fract)!=0:atoms.set_scaled_positions(np.array(fract))

                #atoms.set_positions(np.array(xyz))
                
                
                #atoms2=Atoms(symbols=ids,cell=h,pbc=True,calculator=atoms.calc)#positions=xyz)

        else:
                atoms=Atoms(symbols=ids,cell=h,pbc=True,calculator=None)#positions=xyz)
                #return atoms

        
        #if len(xyz)!=0:atoms2.set_positions(np.array(xyz))
        #elif len(fract)!=0:atoms2.set_scaled_positions(np.array(fract))

        #Energy, Enthalpy, stress tensor, Cart coords, fractional coords, atoms object.
        return E, H, s, fract, forces, ifConv, atoms
     
def parseCASTEP_ASE(fname,atoms=None):
        atoms=ase.io.read(fname,index=-1)
        return atoms.get_potential_energy(),atoms #this gets the final enthalpy rather than E-0.5TS

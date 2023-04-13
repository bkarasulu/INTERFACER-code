#!/usr/bin/env python3

from __future__ import print_function
import numpy as np
import argparse
import ase.io
import ase
from ase.visualize import view
from os import system,popen,chdir,getcwd
import os.path
from re import search
from sys import exit
from ase.build import *
import ase.build
from ase import Atoms
import fractions
import copy
import time
import ase.build.tools
import matplotlib.pyplot as plt
import ase.calculators.castep
from spglib import find_primitive,standardize_cell,get_spacegroup #niggli_reduce from ASE-tools conflicts with that from spglib.

from ase.constraints import FixAtoms
#from ase.spacegroup import crystal

from ase import *
from ase.optimize.basin import BasinHopping

try:
        from os import popen4 #as popen
except:
        from os import popen #as popen

#Surface creation and alignment tool by James P. Darby.
#from join8 import *
from int_new import *

def misfit(slab1,slab2,ptol=2,ifPlot=False): #Computes the misfit parameter between two slabs. (see Surf. Interface Anal. 2003, 35, 835-841. )
        #Assuimng that two slabs are well aligned, overlap area should be equalt to smaller one's surface area.  !!FIND A MORE ACCURATE WAY OF DETERMINING OVERLAPPING SURFACE.
        #align_slab_axes(slab1,slab2,ptol)
        A1=surf_area(slab1);A2=surf_area(slab2)
        if ifPlot: plot_slab_axes(slab1,slab2)
        return 1 - 2*min(A1,A2)/(A1+A2)

def surf_area(slab1):
        return np.linalg.norm(np.cross(slab1.cell[0],slab1.cell[1]))

def volume(cell):
        return np.abs(np.dot(cell[2], np.cross(cell[0], cell[1])))

def get_fu(atoms):
        aTypes=atoms.get_chemical_symbols()
        atoms={}
        #print aTypes
        for at in aTypes:
            #print ln
            #at=ln.split()[0]
            if at not in atoms:atoms[at]=1
            else: atoms[at]+=1
        

        keys=list(atoms.keys())
        keys.sort()
        content=""
        for key in keys:
                content+=key
                if atoms[key]!=1:content+=str(atoms[key])

        #Determine the formula unit.

        try:
                fu=1
                vals=list(atoms.values())
                for i in range(2,min(vals)+1):
                        fl=1
                        for j in vals:
                                if j%i!=0:fl=0
                        if fl:fu=i
        #print fu
        except: print("Error in fu determination, fu set to 1");   fu=1
    
        return fu


                

def find_prim(atoms,tol=1e-4):#using SPGlib find primitive cell of a given atoms object.
        scaled_positions= atoms.get_scaled_positions()#(wrap=True) 
        cell=(atoms.cell, scaled_positions, atoms.numbers)
        #print cell
        print("Space group of the given cell using tolerance=%f: %s"%(tol,get_spacegroup(cell,symprec=tol)))
        lattice, scaled_positions, numbers = find_primitive(cell, symprec=tol)
        #lattice, scaled_positions, numbers = standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=tol)
        atoms2=Atoms(symbols=numbers,cell=lattice,scaled_positions=scaled_positions,pbc=True)
        return atoms2

def grep(key,fname,n=-1): 
    #Uses grep to get the nth line with the keyword from a given file. By default the last occurence is returned.
        try:
                return popen4('grep -m %d "%s" %s '%(n,key,fname),"r")[1].readlines()[-1][0:-1] #don't take \n at the end.  #Fastest one!!!
        except: 
                return popen('grep -m %d "%s" %s '%(n,key,fname),"r").read()[0:-1]
        else:
                return ""

def fix_atoms(atoms, width, vacuum):
   #fixes the coordinates of all atoms within not within width of the edge of the slab 
   a3 = atoms.cell[2] 
   n = a3/np.linalg.norm(a3)
   u3min =  (vacuum/2 + width)/np.linalg.norm(a3)
   u3max = 1 - u3min
   
   indices = []
   for i,u3 in enumerate(atoms.get_scaled_positions()):
      #print(i,u3, atoms[i].index) 
      if u3[2] <= u3min or u3[2] >= u3max:#modified
         indices.append(i)
   c = FixAtoms(indices=indices) # ase.constraints.Fix_Atoms
   #print(c)
   #atoms.set_constraint(c)
   return indices

def call_castep(atoms,calc=None, typ="sp",PP='',wDir='./CASTEP-tmp',name='try',param='opt.param',resDir="",dipolCorr=None,dipolDir='z',KPgrid="1 1 1",KPspacing="", ifPrint=False,ifDryRun=False,ENCUT=0,ifManualRun=True,FixCell=False,FixList=[]):
    #For applying constraints (FixList) atom numbering starts from 1.


    #exe="mpirun -n 4 castep";PP_path='/rscratch/bk393/pspots/CASTEP'
    #exe="mpirun -n 20 castep";PP_path='/u/fs1/bk393/pspots/CASTEP'

    #system("export CASTEP_COMMAND='%s'"%exe)
    #system("export CASTEP_COMMAND='mpirun -n 4 castep'")
    #system("export CASTEP_PP_PATH='%s'"%PP_path)

    exe=popen('echo "$CASTEP_COMMAND"',"r").read()[0:-1]
    PP_path=popen('echo "$CASTEP_PP_PATH"',"r").read()[0:-1]

    if calc!=None:
            #This is good for assiging the settings from a previously created Calculator object (i.e. not to repeat all the settings the second time). None of the other options will be taken into account.
            calc._label = name
            calc._directory = wDir
            calc._export_settings = True

            if ifPrint:print (calc) #prints calculation summary.

            if len(FixList)!=0:
                    #atoms.set_constraint(None);
                    c = FixAtoms(indices=FixList)
                    atoms.set_constraint(c) #This actually works.

            calc.initialize()
            atoms.set_calculator(calc)

            return runCASTEP(atoms,exe,name,wDir)


    #if  atoms.get_calculator() is not None: calc=atoms.get_calculator()
    #else: calc = ase.calculators.castep.Castep()

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

    #New user input within the interfacer code.
    if KPspacing:calc.cell.kpoints_mp_spacing = str(KPspacing) #default=0.05 2pi*eV/A
    else: calc.cell.kpoint_mp_grid = KPgrid #def='1 1 1'
    
    #calc.cell.fix_com = False
    if FixCell: calc.cell.fix_all_cell = True
    if len(FixList)!=0:
            #c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu'])
            c = FixAtoms(indices=FixList)
            atoms.set_constraint(c) #This only work if the CASTEP is called by ASE (not for manual runs). This actually works.

            """#This is not needed anymore.
            str1=[]
            i=1
            for at in FixList: 
                    str1.append("%d %s %d %d %d %d"%(i,atom.symbol,atom.index,1,0,0))
                    str1.append("%d %s %d %d %d %d"%(i+1,atom.symbol,atom.index,0,1,0))
                    str1.append("%d %s %d %d %d %d"%(i+2,atom.symbol,atom.index,0,0,1))
                    i+=3

            #calc.cell.ionic_constraints=str1 #a list object needed as input.
            """
            calc.cell.snap_to_symmetry=False
            calc.cell.symmetry_generate=False
            
     
    #This overwrites the task paramter from the param input.
    if typ.lower() in ["sp","single"]:    calc.param.task = 'SinglePoint'
    elif typ.lower() in ['opt','geom']:calc.Task = 'GeometryOptimization'
    
    if dipolCorr: #def: No dipole corrections. 
        if dipolCorr.lower()=="sc": calc.param.dipole_correction= "SELFCONSISTENT"
        elif "st" in dipolCorr.lower(): calc.param.dipole_correction= "static"
        else: calc.param.dipole_correction= "None"
        calc.param.dipole_dir=dipolDir #options: x,y,z and a (only energy-corr)

        
    
    #calc.initialize()#Creates all needed input in the _directory. (Not needed for actual run.)
    atoms.set_calculator(calc)  #Set for the previously created interface

    if ENCUT!=0: calc.param.cut_off_energy=ENCUT
    
    if PP!="":#otherwise on-the-fly(OTF) is used as default
        fnc=str(calc.param.xc_functional).split("\n")[1].upper()
        #print fnc
        PP=PP.upper()
        #print PP.lower().find(str(calc.param.xc_functional).lower())
        if PP != "OTF" and  PP.find(fnc)== -1:
                print("There is a problem with the pseudopotential choice. \nSelected PP does not match with XC functional being used: ",PP,fnc)
                exit()
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
        
    if ifPrint:print (calc) #prints calculation summary.

    # necessary for tasks with changing positions
    # such as GeometryOptimization or MolecularDynamics (This option does not work as deisgnated !!! The atomic coords are not updated at the end of the geom. opt. unlike the energy.
    calc._set_atoms = True
    atoms.calc._set_atoms = True
    
    
    # Check for correct input
    if not ifDryRun:
            if ifManualRun: #If a manual run of the CASTEP is needed.
                    return runCASTEP(atoms,exe,name,wDir)

                    
            else: #CASTEP calculation is not done here. It will be called in the main script, when trying to reach the attributes, e.g. atoms.get_potential_energy().
                    return atoms
    else:
            if calc.dryrun_ok():

                    return atoms
            else:
                    print("CASTEP run: Found error in input")
                    print((calc._error))
                    return None

def runCASTEP(atoms,exe,name,wDir,ifPrint=False):
        calc=atoms.get_calculator()
        str1="%s %s"%(exe,name)
        print("Running ",str1)
        if ifPrint:print (calc) #prints calculation summary. 
        calc._copy_pspots=True
        calc.initialize()

        CWD=getcwd()
        chdir(wDir)

        system(str1) #PSPOT missing in the folder
        #x=parseCASTEP("%s/%s.geom"%(wDir,name),atoms=atoms)
        task=str(atoms.calc.param.task).split()[-1]
        print(task)
        if task=='SinglePoint' : #use <seed>.castep file
                x=parseCASTEP("%s.castep"%(name),atoms=atoms)
        elif task=='GeometryOptimization': #use <seed>.geom file.
                try:
                        x=parseCASTEP("%s.geom"%(name),atoms=atoms)
                except:
                        x=parseCASTEP("%s.castep"%(name),atoms=atoms)

                if x[-2]==False: print("parseCASTEP: WARNING: Geometry optimization in %s.geom is not converged!"%name)
        else:
                print("parseCASTEP: ERROR: Calculation type is not supported.")
                x=None
        chdir(CWD)
        return x

def parseCASTEP(fname,atoms=None):  #TODO: fix wrong atomic order (if the atoms object is ordered).
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

        print("Parsing ",fname)
        if fname.split(".")[-1]=="geom":
                try: tSteps=len(popen("grep '<-- c' %s"%fname).readlines())-1
                except:tSteps=0

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
                        elif search("Final free energy \(E-TS\)    =",ln):H=float(ln.split("=")[1].split()[0])
                        elif search("Final energy, E",ln): E=float(ln.split("=")[1].split()[0])
                        elif search("\*  x ",ln) or search("\*  y ",ln) or search("\*  z ",ln): s.append([float(i) for i in ln.split()[2:5]]) #Stress tensor already in GPa.
                        elif len(ln)>2 and ln[1]=="\*" and len(ln.split())==7:
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
                
        if len(ids)==0: 
                print ("parseCASTEP: WARNING: No chemical symbol info in %s."%fname); return None
        elif len(h)==0: 
                print ("parseCASTEP: WARNING: No cell info in %s."%fname); return None
        elif len(xyz)==0 and len(fract)==0: 
                print ("parseCASTEP: WARNING: No Cartesian or fractional coordinate info in %s."%fname); return None
        
                                                
        if atoms != None:
                atoms.set_chemical_symbols(ids)
                atoms.set_cell(np.array(h))
                if len(xyz)!=0:atoms.set_positions(np.array(xyz))
                elif len(fract)!=0:atoms.set_scaled_positions(np.array(fract))
        else:
                atoms=Atoms(symbols=ids,cell=h,pbc=True,calculator=None)#positions=xyz)

        #Energy, Enthalpy, stress tensor, Cart coords, fractional coords, atoms object.
        return E, H, s, fract, forces, ifConv, atoms
        
def conv_layers(atoms,ifPlot=False,ifPrim=False,vac=4.0,Etol=5e-2,Ws_tol=5e-2,Ftol=5e-2,Stol=0.1,outf=None):#layer convergence test (Input atoms with a calc object). 
        #print "Convergence of E/atom vs. #layers"
        #Etol=1e-1 #eV/atom
        #Ws_tol=1e-1 #eV
        #Ftol=5e-2 #eV/Angstroem
        #Stol=0.1 #GPa
        str1="Layer (thickness) convergence test for %s. Target deltaE= %.2f eV/atom.\n"%(atoms.get_chemical_formula(),Etol)
        print(str1)
        if outf is not None: outf.writelines(str1);outf.flush()
        #Initial values
        E=[0]; F=[0]; S=[0];Ws=[0]

        calc=atoms.get_calculator()
        #calc._label=
        #find the primitive cells to reduce comp. efforts.
        if ifPrim: print('Using primitve cell in layer convergence...'); atoms=find_prim(atoms);atoms.set_calculator(calc) #Check: using prim cell gives twice Wsurf values. #Do not use it !!!
        atoms_orig=atoms.copy()
        atoms.center(vacuum=vac, axis=2)
        
        #view(atoms)

        nAt=atoms.get_number_of_atoms()
        cname=atoms.get_chemical_formula()

        x=call_castep(atoms,calc,name='%s-layer%d'%(cname,1))
        slab1=x[-1]
        Eslab1=x[0]
        E.append(Eslab1/nAt)
        Ws.append((Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01) #A2 to nm2
        #E.append(Ws1)

        str1="Iter. #%d, #layers: %d, #atoms: %d, E/atom=%.4e, Wsurf=%.2f\n"%(0,1,nAt,E[-1],Ws[-1])
        str1+="deltaE: %.3e eV/atom; target: %.3e eV.\n"%(abs(E[1]-E[0]),Etol)
        print(str1)
        if outf is not None: outf.writelines(str1);outf.flush()

        i=1;layers=[1];
        while abs(E[i]-E[i-1]) > Etol:
                layers.append(1+1*i)
                atoms=atoms_orig.copy()
                atoms=atoms.repeat((1,1,layers[-1]))
                atoms.center(vacuum=vac, axis=2)
                atoms.set_calculator(calc)
                #view(atoms)
                nAt=atoms.get_number_of_atoms()
                x=call_castep(atoms,calc,name='%s-layer%d'%(cname,layers[-1]))
                slab1=x[-1]
                Eslab1=x[0]
                E.append(Eslab1/nAt)
                Ws.append((Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01)#A2 to nm2
                str1="Iter. #%d, #layers: %d, #atoms: %d, E/atom=%.4f, Wsurf=%.2f\n"%(i,layers[-1],nAt,E[-1],Ws[-1])
                str1+="deltaE: %.3e eV/atom; target: %.3e eV.\n"%(abs(E[-1]-E[-2]),Etol)
                print(str1)
                if outf is not None: outf.writelines(str1);outf.flush()

                i += 1

        str1="conv_layers: E/atom converged to %.2f eV with %d layers."%(Etol,layers[-1])
        print(str1)
        if outf is not None: outf.writelines(str1);outf.flush()

        if ifPlot: #Do plotting of E/atom vs. #layers
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                #plt.plot(layers,E[1:], 'ko-',layers,Ws[1:], 'ro-')
                ax1.plot(layers,E[1:], 'ko-')
                ax2.plot(layers,Ws[1:], 'ro-')
                ax1.set_xlabel('Number of layers')
                ax1.set_ylabel('Energy per atom (eV/atom)')
                ax2.set_ylabel('Surface formation Energy (Ws) (eV)')
                
                plt.savefig('%s-layer-conv.png'%cname,format='png')
                #plt.show()

        return layers[-1],E[-1]*nAt,atoms   

def make_slab(miller1,atoms1,max_atoms_1,thickness,vac=0,creps_1=None):#New, adapted from from int.py
        #Only needed for surface analysis of a single input geom!!
        cell1 = Cell(atoms1)

        #get the new cell vectors needed to create the desired surface
        slab1 = Slab(miller1, cell1)
        slab1.set_Rmax(max_atoms_1)

	#only need to check out as far as the shorter Rmax	
	#Rmax = min(slab1.Rmax,slab2.Rmax)
        Rmax=slab1.Rmax

        #find all vectors in slabs with length <= Rmax
        #returns shortest vector in each direction, none are parallel
        vecs1 = find_vecs(slab1,Rmax)

        #construct pairs
        pairs1 = build_pairs(vecs1,slab1)
        pairs1 = angle_filter(pairs1)

        pair1 = best_pair(pairs1)			
        out_slab = make_cells(pair=pair1)


        #convert thicknesses to creps
        if creps_1 is not None:
                cr1 = args.creps_1
        else:
                c = np.linalg.norm(out_slab.cell[2])	
                cr1 = int(np.ceil(thickness/c))

        #repeat the slab, replace with function
        out_slab.crep(cr1)	
        if vac > 0.001:			
                out_slab.square()
                out_slab.add_vac(0.5* vac) 

        #write out the slab 
        at1 = ase.atoms.Atoms(symbols=out_slab.symbols, scaled_positions=out_slab.frac, cell=out_slab.cell)

        return at1 #,out_slab

def slab_aligner(slab1,slab2,miller1,miller2,percentage_tolerance,max_atoms_1, max_atoms_2,thickness,creps_1=None,creps_2=None):
        
        cell1 = Cell(atoms1)
        cell2 = Cell(atoms2)


        #get the new cell vectors needed to create the desired surface
        slab1 = Slab(miller1, cell1)
        slab1.set_Rmax(max_atoms_1)
        slab2 = Slab(miller2,cell2)
        slab2.set_Rmax(max_atoms_2)
        
        #only need to check out as far as the shorter Rmax	
        Rmax = min(slab1.Rmax,slab2.Rmax)	


        #find all vectors in slabs with length <= Rmax
        #returns shortest vector in each direction, none are parallel
        vecs1 = find_vecs(slab1,Rmax)
        vecs2 = find_vecs(slab2,Rmax)	

        #construct pairs
        pairs1 = build_pairs(vecs1,slab1)
        pairs2 = build_pairs(vecs2,slab2)


        #TODO insert a symmetry reduction for pairs

        #filter pairs, all angles should be between 60 and 150 degrees
        #if a match could be made outside of this range then gauss reduction on the match
        #would bring theta into the desired range and with shorter lattice vectors
        pairs1 = angle_filter(pairs1)
        pairs2 = angle_filter(pairs2)
        matches = match_pairs(pairs1,pairs2,percentage_tolerance, Rmax)	



        #TODO symmetry reduce matches, do it here or might be easier/more efficient to do
        #it earlier, e.g. symmetry reduce the vectors before pairs are constructed
        #that won't get all symetry equivalents though... maybe it is best just to reduce
        #matches...??
        match = find_best_match(matches)

        #match = matches[0]
        print("there are ",len(matches), "matches")

        #pair 1 is bottom cell, pair 2 is top cell
        top_cell, bot_cell = make_cells(match=match)
        print("made cells")

        #convert thicknesses to creps
        if args.creps_1 is not None:
                cr1 = args.creps_1
        else:
                c = np.linalg.norm(top_cell.cell[2])	
                cr1 = int(np.ceil(thickness/c))


        if args.creps_2 is not None:
                cr2 = args.creps_2
        else:
                c = np.linalg.norm(bot_cell.cell[2])	
                cr2 = int(np.ceil(thickness/c))

        top_cell.crep(cr1)
        bot_cell.crep(cr2)

        print("c repeated cells: %d %d"%(cr1,cr2))

        #square the slabs and then stack themn		
        bot_cell.square()
        top_usq  = copy_cell(top_cell)
        top_cell.square()

        """
        #repeat the slab, replace with function
        out_slab.crep(cr1)	
        if vac > 0.001:			
                out_slab.square()
                out_slab.add_vac(0.5* vac) 
        """

        #write the top and bottom cell out as .cell files
        at1 = ase.atoms.Atoms(symbols=top_cell.symbols, positions=top_cell.positions,cell=top_cell.cell)
        at1.set_pbc([True,True, False])

        #ase.io.write("top.cell",at1,format="castep-cell")
        at2 = ase.atoms.Atoms(symbols=bot_cell.symbols, scaled_positions=bot_cell.frac,cell=bot_cell.cell)
        #ase.io.write("bot.cell",at2,format="castep-cell")
        at2.set_pbc([True, True, False])
        #symmetry = spglib.get_symmetry(at1)	


        #check number densities of the slabs
        n1 = compare_density(at1, atoms1)
        n2 = compare_density(at2, atoms2)

        return at1,at2

def shifter(top_cell,bot_cell,shift_grid,vac,args):
        #get the symmetry translation vectors for each slab
        tt1, tt2, tA = get_symmetry_translations(top_cell)
        bt1, bt2, bA = get_symmetry_translations(bot_cell)	
        #as shifts are relative, take the shifts with smaller area
        #BUT can always shift the top cell w.l.o.g
        if tA < bA:
                shifts = [tt1,tt2]
        else:
                shifts = [bt1,bt2]

        #	shift the final section into a function called stack_and_write
        #then create list of shifted cells to pass to stack_and_write 
        #so will end up with one call generating all the required shifts	
        n,m = shift_grid
        n = int(n);		m = int(m)
        for i in range(0,n):
                for j in range(0,m):
                        t1 = shifts[0]/n * i
                        t2 = shifts[1]/m * j


                        #apply the shifts and then create the interface
                        bot_cell0 = copy_cell(bot_cell)	
                        top_cell0 = copy_cell(top_cell)
                        bot_cell0.shift(t1+t2)
                        interface,M = stack(top_cell0, bot_cell0, vac=args.vac)

                        write_cell_2_res(interface, args, i,j,n,m)

def cell2Atoms(top_cell):
        return Atoms(symbols=top_cell.symbols, positions=top_cell.positions,cell=top_cell.cell,pbc=[True, True,True])



##################################################
#                  MAIN METHOD                   #
##################################################
               
if __name__== '__main__':
        #read in arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("-i1", "--infile1", default="Li.cell")
        parser.add_argument("-o", "--outfile")
        parser.add_argument("-i2", "--infile2", default="Al.cell")
        parser.add_argument("-m1","--miller1", default=(1,0,0), nargs="+", type=int)
        parser.add_argument("-m2","--miller2", default=(1,0,0), nargs="+", type=int)
        #parser.add_argument("-msd","--max_slab_dimension",default=50)
        parser.add_argument("-th","--thickness",default=10,type=float)
        #parser.add_argument("-pt","--percentage_tolerance",default=4)
        parser.add_argument("-conv", "--convLayers",action="store_true",default=False,help="To run a convergence test for slab(s) generated.")

        parser.add_argument("-t","--type",choices=['s','i'],help="s: check different slabs of given bulk structure(s), i: create interface from the slabs with -m1 and -m2 Miller indices from input bulk structures.")


        parser.add_argument("-n1","--max_atoms_1",default=120,type=int,help="max atoms in slab 1")	
        parser.add_argument("-n2","--max_atoms_2",default=120,type=int,help="max atoms in slab 1")	
        #parser.add_argument("--slab_only", action="store_true")
	#parser.add_argument("-t","--thickness",default=7,type=float)
        parser.add_argument("-cr1","--creps_1",default=None,type=int,help="overrides thickness")
        parser.add_argument("-cr2","--creps_2",default=None,type=int)
        parser.add_argument("-pt","--percentage_tolerance",default=15.0, type=float)
        parser.add_argument("-d","--distance",default=1, type=float)
        parser.add_argument("-v","--vac",default=15.0, type=float)
        parser.add_argument("-sg","--shift_grid",default=(1,1,), type=float, nargs="+", help = "2 integers to specify the size of the shift grid" 	)
        parser.add_argument("--res", action="store_true")

        args = parser.parse_args()

        infile1 = args.infile1
        miller1 = tuple(args.miller1)
        infile2= args.infile2
        miller2 = tuple(args.miller2)

        #2 is going on the bottom so m2 ->-m2 to get right orientation of the surface (in James' new version)
        miller2 = [-x for x in miller2]


        if args.outfile:
                outfile = args.outfile
        else:
                outfile ="interfaces.out"

        system("echo >> %s"%outfile)
        system("date >> %s"%outfile)
        outf=open(outfile,'a')


        miller_list=[(0,1,0),(1,1,0),(1,0,0),(1,1,1)]

        #Common DFT-related parameters.
        ecut=300 #cutoff energy in eV (convergence prob. with lower cutoffs).
        pp="00PBE" #pseudopt to use in CASTEP calcs.Def (pp=""): OTF
        pp="OTF"
        dirr="./CASTEP-tmp"
        dC=None #options: 'sc':self-consistent,'st':static, None (default).

        #read in atoms and construct slab, need to repeat atoms to make view work
        print("Reading data from %s and %s."%(infile1,infile2))
        if infile1.split(".")[1]=="cell":
                atoms1 = ase.io.read(infile1, format='castep-cell')
        elif infile1.split(".")[1]=="cif":
                atoms1 = ase.io.read(infile1, format='cif')
        else:
                atoms1 = ase.io.read(infile1)

        if infile2.split(".")[1]=="cell":
                atoms2 = ase.io.read(infile2, format='castep-cell')
        elif infile2.split(".")[1]=="cif":
                atoms2 = ase.io.read(infile2, format='cif')            
        else:
                atoms2 = ase.io.read(infile2)

        initT=time.time()

        #if len(atoms1) < 10:
        #        atoms1=atoms1.repeat((2,2,2))
        #if len(atoms2) < 10:
        #        atoms2=atoms2.repeat((2,2,2))

        print("Structure 1: %s with fu=%d"%(atoms1.get_chemical_formula(),get_fu(atoms1)))
        print("Structure 2: %s with fu=%d"%(atoms2.get_chemical_formula(),get_fu(atoms2)))

        if 0: atoms1=find_prim(atoms1); atoms2=find_prim(atoms2) #Whether to use primitive cells of input structures.

        #Delete calculation files from previous CASTEP run.
        #system("rm -f %s/*"%dirr)
        system("rm -f interface.cell  interface0a.cell  slab1.cell  slab1_aligned.cell  slab1_aligned_opted.cell  slab2.cell  slab2_aligned.cell  slab2_aligned_opted.cell interface_opted.cell interface_opted.cif CASTEP-tmp/slab* CASTEP-tmp/interface*")
        print()


        ########################
        # Do bulk calculations #
        ########################
        #Check if data from a previous run is available (not to repeat the same calcs for bulk).
        fn1="%s/bulk1"%dirr; fn2="%s/bulk2"%dirr
        kp=0.10
        #calc_tup=(None,name='',typ="geom",dipolCorr=dC,ENCUT=ecut,PP=pp,KPspacing=kp,ifPrint=1)
        x1=None;x2=None
        if os.path.exists(fn1+".castep"):
                print ("%s was located, reading data..."%fn1)
                x1=parseCASTEP(fn1+".castep",atoms1)
                if x1==None:
                        x1=parseCASTEP(fn1+".geom",atoms1)
                
        if x1==None:  # whether to compute bulk energies/structures
                print("Computing bulk 1 energy.")
                #calc_tup[0]=atoms1;calc_tup[1]='bulk1'
                x1=call_castep(atoms1,name='bulk1',typ="geom",dipolCorr=dC,ENCUT=ecut,PP=pp,KPspacing=kp,ifPrint=1) #normally use K-point spacing.

        #atoms1=x1[-1]
        #calc=atoms1.get_calculator()
        #print ("bora: ",calc)

        if os.path.exists(fn2+".castep"):
                print ("%s was located, reading data..."%fn2)
                x2=parseCASTEP(fn2+".castep",atoms2)
                if x2==None:
                        x2=parseCASTEP(fn2+".geom",atoms2)

        if x2==None:  
                print("Computing bulk 2 energy.")
                #calc_tup[0]=atoms2;calc_tup[1]='bulk2'
                x2=call_castep(atoms2,name='bulk2',typ="geom",dipolCorr=dC,ENCUT=ecut,PP=pp,KPspacing=kp,ifPrint=1) #normally use K-point spacing.
                #x2=call_castep(atoms2,typ="geom",dipolCorr=None,name='bulk2',ENCUT=ecut,PP=pp,KPspacing=0.10) #normally use K-point spacing.


        atoms1=x1[-1]
        Ebulk1=x1[0]

        fu1=get_fu(atoms1)
        sa1=surf_area(atoms1)
        #Ebulk1 /= fu1 #Get the bulk energy per formula unit.
        Ebulk1 /= len(atoms1) #Get the bulk energy per atom.

        atoms2=x2[-1]
        Ebulk2=x2[0]

        fu2=get_fu(atoms2)
        #Ebulk2 /= fu2
        Ebulk2 /= len(atoms2) #Get the bulk energy per atom.
        sa2=surf_area(atoms2)

        del x1,x2

        if args.type=="s":
                print("Performing a surface analysis as requested.")
                #str1=""
                min_Ws=1e8; min_mil=();min_slab=None
                for mil in miller_list: #fix the old functions.
                        #slab1 = make_slab(mil,atoms1,repeat=(1,1,1),square=False)
                        #slab1.center(vacuum=4.0, axis=2)
                        #try:
                        slab1=make_slab(mil,atoms1,args.max_atoms_1,thickness=args.thickness,vac=args.vac, creps_1=args.creps_1) #new version.
                        #slab1,slab2=slab_aligner(atoms1,atoms1,mil,mil,args.max_atoms_1, args.max_atoms_2,args.percentage_tolerance,args.thickness,args.creps_1,args.creps_2)
                        #except:
                                #print ("Error in make_slab!!");exit()
                        #slab1=cell2Atoms(slab1)
                        print ("burda")
                        name='%s_%d%d%d'%(slab1.get_chemical_formula(),mil[0],mil[1],mil[2])
                        x=call_castep(slab1,typ="geom",dipolCorr=dC,name=name,ENCUT=ecut,KPgrid='4 4 1',PP=pp,FixCell=True)
                        slab1=x[-1]
                        Eslab1=x[0]
                        Ws1=(Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
                        str1='%s (%s_%d%d%d): %.2f eV/nm^2\n' % ('Wsurf 1', slab1.get_chemical_formula(),mil[0],mil[1],mil[2],Ws1)
                        if Ws1<min_Ws: min_Ws=Ws1; min_mil=mil;min_slab=slab1
                        outf.writelines(str1);outf.flush()
                        print (str1)
                str1="Most stable surface is %s_%d%d%d with sigma=%.2f eV.\n"%(slab1.get_chemical_formula(),min_mil[0],min_mil[1],min_mil[2],min_Ws)
                print(str1)
                outf.writelines(str1);outf.flush()

                if args.convLayers:#Run layer thickness convergence test.
                        calc=min_slab.get_calculator()
                        #print (calc)
                        slab1=make_slab(min_mil,atoms1,args.max_atoms_1,thickness=args.thickness,vac=0, creps_1=1)
                        slab1.set_calculator(calc)
                        nl1,Eslab1,slab1=conv_layers(slab1,ifPrim=0,ifPlot=1,Etol=5e-2,vac=args.vac,outf=outf)  #Vacuum layer is added automatically within the function.

                exit()

    
       

        ######################
        # Alignment of Slabs #
        ######################
        print("\nAligning the two slabs...")
        #slab1,slab2=slab_aligner(slab1,slab2,L,Lmax,Lstep,ptol,T)
        slab1,slab2=slab_aligner(atoms1,atoms2,miller1,miller2,args.max_atoms_1, args.max_atoms_2,args.percentage_tolerance,args.thickness,args.creps_1,args.creps_2)
        print("\nMisfit (mu) of slabs 1 and 2 (after alignment): %.2f%%"%(misfit(slab1,slab2)*100))#,ifPlot=1)
        print ("Elapsed time: %.2f sec."%( time.time()-initT))
        

        ase.io.write("slab1_aligned.cell",slab1,format='castep-cell')
        ase.io.write("slab2_aligned.cell",slab2,format='castep-cell')


        if 1: #to calculate energies/structures of the actual slabs (after alignment).
                print("\nOptimizing the slabs 1 and 2 (after alignment).")

                fixList1=[]; fixList2=[];
                if 1: #to add vacuum to the slabs (needed) make_slab does not add it automatically. James' stack function does not support slabs with vacuum!!
                        slab1.center(vacuum=args.vac/2, axis=2)
                        slab2.center(vacuum=args.vac/2, axis=2)
                        ase.io.write("slab1_aligned.cell",slab1,format='castep-cell')
                        ase.io.write("slab2_aligned.cell",slab2,format='castep-cell')

                        if 1:#this works well
                                fixList1=fix_atoms(slab1,width=(args.vac/2)-2.0,vacuum=args.vac)
                                fixList2=fix_atoms(slab2,width=(args.vac/2)-2.0,vacuum=args.vac)
                                print("Fixed atom list for slab1: %s"%fixList1)
                                print("Fixed atom list for slab2: %s"%fixList2)

                #x=call_castep(slab1,typ="geom",dipolCorr=dC,name='slab1-aligned',ENCUT=ecut,KPgrid='1 1 1',PP=pp,FixCell=True,FixList=fixList1)
                x=call_castep(slab1,typ="geom",dipolCorr=dC,name='slab1-aligned',ENCUT=ecut,KPspacing=0.10,PP=pp,FixCell=1,FixList=fixList1)
                slab1=x[-1]
                Eslab1=x[0]

                calc=slab1.get_calculator() #CHECK if this works !!
                x=call_castep(slab2,calc, name='slab2-aligned',FixList=fixList2) #CHECK if the fixatom assignment work for slab2.
                #x=call_castep(slab2,typ="geom",dipolCorr=dC,name='slab2-aligned',ENCUT=ecut,KPgrid='1 1 1',PP=pp,FixCell=True)
                #x=call_castep(slab2,typ="sp",dipolCorr=dC,name='slab2-aligned',ENCUT=ecut,KPspacing=0.10,PP=pp)
                slab2=x[-1]
                Eslab2=x[0]


                #Compute the surfafce energies.
                #Ws1=(Eslab1-fu1*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
                #Ws2=(Eslab2-fu2*Ebulk2)/2/surf_area(slab2)/0.01 #A2 to nm2
                Ws1=(Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
                Ws2=(Eslab2-len(slab2)*Ebulk2)/2/surf_area(slab2)/0.01 #A2 to nm2

                str1=''
                str1+='%s: %s eV\n' % ('Ebulk 1', Ebulk1)
                str1+='%s: %s eV\n' % ('Ebulk 2', Ebulk2)
                str1+='%s (%s_%d%d%d): %.2f eV/nm^2\n' % ('Wsurf 1', slab1.get_chemical_formula(),miller1[0],miller1[1],miller1[2],Ws1)
                str1+='%s (%s_%d%d%d): %.2f eV/nm^2\n' % ('Wsurf 2', slab2.get_chemical_formula(),miller2[0],miller2[1],miller2[2],Ws2)
                name1="%s_%d%d%d"%(slab1.get_chemical_formula(),miller1[0],miller1[1],miller1[2])
                name2="%s_%d%d%d"%(slab2.get_chemical_formula(),miller2[0],miller2[1],miller2[2])

                print(str1)
                outf.writelines(str1);outf.flush()

                #Remove the constraints before writing files (ASE cannot handle the constraints during writing properly).
                slab1.set_constraint(None); slab1.set_constraint(None);
                #ase.io.write("slab1_aligned_opted.cell",slab1,format='castep-cell')
                #ase.io.write("slab2_aligned_opted.cell",slab2,format='castep-cell')
                ase.io.write(name1+"_opted.cell",slab1,format='castep-cell')
                ase.io.write(name2+"_opted.cell",slab2,format='castep-cell')
                

                if 0:
                        slab1.center(vacuum=0, axis=2)
                        slab2.center(vacuum=0, axis=2)
                        view(slab1)
        #exit()


        #Create the interface.
        if 0: 
                interface=ase.build.stack(slab1, slab2, axis=2, maxstrain=None, distance=2.0,cell=None,reorder=True)  #using 0 distance btw slabs gives CASTEP error.
                #interface = wrap_coords(interface)
                #interface.wrap()
                interface.center(vacuum=args.vac/2, axis=2) #Vacuum on both sides. For dipole corrections at least 8A vacuum is needed.

        else: #use James' stack function.

                top_cell = Cell(slab1)
                bot_cell = Cell(slab2)

                interface,M=stack(top_cell,bot_cell,vac=0) #,args.vac)
                interface=cell2Atoms(interface)

        view(interface)

        ase.io.write("interface.cell",interface.repeat((1,1,1)),format='castep-cell')
        ase.io.write("int_"+name1+"_"+name2+".cell",interface.repeat((1,1,1)),format='castep-cell')
        #ase.io.write("interface.cif",interface,format='cif')



        #if niggli: niggli_reduce(interface)
        if 1:
                print("\n Single point run for the final interface geometry.")
                x=call_castep(interface,typ="sp",dipolCorr=dC,name='interface',ENCUT=ecut,PP=pp,KPspacing=0.1)# KPgrid='2 2 1')#
                interface=x[-1]
                Eint=x[0]


                Wad=(Eslab1+Eslab2-Eint)/surf_area(interface)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf??
                str1='W_ad before optimisation: %.2f eV/nm^2\n'%Wad
                print(str1)
                outf.writelines(str1);outf.flush()

        if 1:
                print("\nOptimizing the final interface geometry.")
                x=call_castep(interface,typ="opt",dipolCorr=dC,name='interface',ENCUT=ecut,PP=pp,KPspacing=0.1)
                interface=x[-1]
                Eint=x[0]


                Wad=(Eslab1+Eslab2-Eint)/surf_area(interface)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf??
                str1='W_ad after optimisation: %.2f eV/nm^2\n'%Wad
                print(str1)
                outf.writelines(str1);outf.flush()

                #ase.io.write("interface_opted.cell",interface,format='castep-cell')
                #ase.io.write("interface_opted.cif",interface,format='cif')
                ase.io.write("int_"+name1+"_"+name2+"_opted.cell",interface,format='castep-cell')

        outf.close()

        print ("Elapsed time: %.2f sec."%( time.time()-initT))

        exit()


##################
# Excluded Parts #
##################
def slab_aligner_old(slab1,slab2,L,Lmax,Lstep,ptol,thickness):
        #find and repeat slabs as specified,
        #One should use only 1 layer for a layer convergence test.
        choice = find_commensurate_supercell(slab1,slab2,Lmax,Lstep,ptol)
        crep = np.ceil(abs(thickness/np.dot(slab1.cell[2],(0,0,1))))
        slab1 = cut_cell(slab1,choice[2],choice[3],(0,0,crep))
        slab1 = square_slab(slab1)
        crep = np.ceil(abs(thickness/np.dot(slab2.cell[2],(0,0,1))))
        slab2 = cut_cell(slab2,choice[4],choice[5],(0,0,crep))
        slab2 = square_slab(slab2)

        #rotate slab2 so that it is alligned with slab1
        ase.build.rotate(slab2,slab1.cell[2],slab2.cell[2],slab2.cell[0],slab1.cell[0])

        #confirm that atom densities are the same as at the start
        atom_density_check(atoms1,slab1)
        atom_density_check(atoms2,slab2)
        
        return slab1,slab2

        """
        if 0: #to calculate energies/structures of the initial slabs (before alignment).
                print("Pre-optimizing the initial slabs (before alignment).")
                if 1: #to add vacuum to the slabs (needed)
                        slab1.center(vacuum=4.0, axis=2)
                        slab2.center(vacuum=4.0, axis=2)

                x=call_castep(slab1,typ="geom",dipolCorr='sc',name='slab1',ENCUT=ecut,KPgrid='1 1 1',PP=pp)
                slab1=x[-1]
                Eslab1=x[0]

                x=call_castep(slab2,typ="geom",dipolCorr='sc',name='slab2',ENCUT=ecut,KPgrid='1 1 1',PP=pp)
                slab2=x[-1]
                Eslab2=x[0]
                #calc=at.get_calculator()
"""

        """
        #######################################################################
        #Create the intial slabs with given Miller indices (before alignment).#
        #######################################################################
        print("\nCreating the initial slabs.")
        print (miller1,miller2)
        slab1,sl1 = make_slab(miller1,atoms1,args.max_atoms_1,vac=10) #check the Cell and Slab function by James
        slab2,sl2 = make_slab(miller2,atoms2,args.max_atoms_2,vac=10)


        if 0: #to add vacuum to the slabs (for demonstration)
                slab1_vac=slab1
                slab2_vac=slab2
                slab1_vac.center(vacuum=4.0, axis=2)
                slab2_vac.center(vacuum=4.0, axis=2)

                ase.io.write("slab1.cell",slab1_vac.repeat((1,1,1)),format='castep-cell')
                ase.io.write("slab2.cell",slab2_vac.repeat((1,1,1)),format='castep-cell')
        else:
                ase.io.write("slab1.cell",slab1.repeat((1,1,1)),format='castep-cell')
                ase.io.write("slab2.cell",slab2.repeat((1,1,1)),format='castep-cell')


        print("\nMisfit (mu) of slabs 1 and 2 (before alignment): %.2f%%"%(misfit(slab1,slab2)*100))#,ifPlot=1)
        """
        

def conv_layers_old(atoms,ifPlot=False,ifPrim=False):#layer convergence test (Input atoms with a calc object). 
        #print "Convergence of E/atom vs. #layers"
        Etol=1e-2 #eV/atom
        Ftol=5e-2 #eV/Angstroem
        Estr=0.1 #GPa

        #Initial values
        E=[0]; F=[0]; S=[0]
        atoms_orig=atoms.copy()
        calc=atoms.get_calculator()
        
        #find the primitive cells to reduce comp. efforts.
        if ifPrim: atoms=find_prim(atoms);atoms.set_calculator(calc)
        
        atoms.center(vacuum=4.0, axis=2)
        nAt=atoms.get_number_of_atoms()
        E.append(atoms.get_potential_energy()/nAt)
        i=1;layers=[1]
        while abs(E[i]-E[i-1]) > Etol:
                layers.append(1+1*i)
                atoms=atoms_orig.copy()
                atoms=atoms.repeat((1,1,layers[-1]))
                atoms.center(vacuum=4.0, axis=2)
                atoms.set_calculator(calc)
                #view(atoms)
                nAt=atoms.get_number_of_atoms()
                E.append(atoms.get_potential_energy()/nAt)
                print("Iter. #%d, #layers: %d, #atoms: %d "%(i,layers[-1],nAt))
                print("deltaE: %.3e eV/atom; target: %.3e eV."%(abs(E[i]-E[i-1]),Etol))
                i += 1
        print("conv_layers: E/atom converged to %.2f eV with %d layers."%(Etol,layers[-1]))
        if ifPlot: #Do plotting of E/atom vs. #layers
                plt.plot(layers,E[1:], 'ko-')
                plt.xlabel('Number of layers')
                plt.ylabel('Energy per atom (eV/atom)')
                #plt.savefig('layer-conv.png')
                plt.show()
                
        return layers[-1],E[-1]*nAt,atoms               


def conv_layers(atoms,ifPlot=False,ifPrim=False,vac=4.0,Etol=1e-1,Ws_tol=5e-2,Ftol=5e-2,Stol=0.1,outf=None):#layer convergence test (Input atoms with a calc object). 
        #print "Convergence of E/atom vs. #layers"
        #Etol=1e-1 #eV/atom
        #Ws_tol=1e-1 #eV
        #Ftol=5e-2 #eV/Angstroem
        #Stol=0.1 #GPa
        str1="Layer (thickness) convergence test for %s. Target deltaE= %.2f eV/atom.\n"%(atoms.get_chemical_formula(),Etol)
        print(str1)
        if outf is not None: outf.writelines(str1);outf.flush()
        #Initial values
        E=[0]; F=[0]; S=[0];Ws=[0]

        calc=atoms.get_calculator()
        #calc._label=
        #find the primitive cells to reduce comp. efforts.
        if ifPrim: print('Using primitve cell in layer convergence...'); atoms=find_prim(atoms);atoms.set_calculator(calc)
        atoms_orig=atoms.copy()
        atoms.center(vacuum=vac, axis=2)
        
        #view(atoms)

        nAt=atoms.get_number_of_atoms()
        cname=atoms.get_chemical_formula()
        x=call_castep(atoms,calc,name='%s-layer%d'%(cname,1))#,typ="sp",dipolCorr='sc',name=name,ENCUT=ecut,KPgrid='4 4 1',PP=pp)
        slab1=x[-1]
        Eslab1=x[0]
        E.append(Eslab1/nAt)
        Ws.append((Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01) #A2 to nm2
        #E.append(Ws1)

        str1="Iter. #%d, #layers: %d, #atoms: %d, E/atom=%.4e, Wsurf=%.2f\n"%(0,1,nAt,E[-1],Ws[-1])
        str1+="deltaE: %.3e eV/atom; target: %.3e eV.\n"%(abs(E[1]-E[0]),Etol)
        print(str1)
        if outf is not None: outf.writelines(str1);outf.flush()

        i=1;layers=[1];
        while abs(E[i]-E[i-1]) > Etol:
                layers.append(1+1*i)
                atoms=atoms_orig.copy()
                atoms=atoms.repeat((1,1,layers[-1]))
                atoms.center(vacuum=vac, axis=2)
                atoms.set_calculator(calc)
                #view(atoms)
                nAt=atoms.get_number_of_atoms()
                x=call_castep(atoms,calc,name='%s-layer%d'%(cname,layers[-1]))
                #x=call_castep(atoms,calc)#,typ="sp",dipolCorr='sc',name=name,ENCUT=ecut,KPgrid='4 4 1',PP=pp)   #CHECK if this works correctly!!
                slab1=x[-1]
                Eslab1=x[0]
                E.append(Eslab1/nAt)

                Ws.append((Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01)#A2 to nm2
                #E.append(Ws1)

                str1="Iter. #%d, #layers: %d, #atoms: %d, E/atom=%.4f, Wsurf=%.2f\n"%(i,layers[-1],nAt,E[-1],Ws[-1])
                str1+="deltaE: %.3e eV/atom; target: %.3e eV.\n"%(abs(E[i]-E[i-1]),Etol)
                print(str1)
                if outf is not None: outf.writelines(str1);outf.flush()

                i += 1

        str1="conv_layers: E/atom converged to %.2f eV with %d layers."%(Etol,layers[-1])
        print(str1)
        if outf is not None: outf.writelines(str1);outf.flush()

        if ifPlot: #Do plotting of E/atom vs. #layers
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                #plt.plot(layers,E[1:], 'ko-',layers,Ws[1:], 'ro-')
                ax1.plot(layers,E[1:], 'ko-')
                ax2.plot(layers,Ws[1:], 'ro-')
                ax1.xlabel('Number of layers')
                ax1.ylabel('Energy per atom (eV/atom)')
                ax2.ylabel('Surface formation Energy (Ws) (eV)')
                #plt.savefig('layer-conv.png')
                plt.show()
                plt.savefig('%s-layer-conv.png'%cname,format='png')
                
        return layers[-1],E[-1]*nAt,atoms   


def get_interface_energy(slab1,slab2,Ebulk1,Ebulk2,dist=1.0,convLayers=False): #Input is slab1,slab2 as Atoms object with a calculator property assigned.
        if not slab2.has(calc):
                calc=slab1.get_calculator()
                slab2.set_calculator(calc)
        
        if convLayers: #A layer convergence test is done here!.
                print("Layer convergence test is switched on.")
                #slab1.set_calculator(calc)
                slab1=call_castep(slab1,typ="SP",dipolCorr='SC',name='slab1',ENCUT=ecut,KPgrid='1 1 1',PP=pp) #Use SP for efficiecy.
                calc=slab1.get_calculator()       
                print("Working on slab 1.")
                nl1,Eslab1,slab1=conv_layers(slab1)#,ifPrim=1)  #Vacuum layer is added automatically within the function.
                fu1=get_fu(slab1)
                
                #slab2=call_castep(slab2,typ="SP",dipolCorr='SC',name='slab2',ENCUT=ecut,KPgrid='1 1 1',PP=pp)
                
                slab2.set_calculator(calc)
                slab2.calc._label="slab2"    #change name for slab2
                print("Working on slab 2.")                
                nl2,Eslab2,slab2=conv_layers(slab2)#,ifPrim=1)

                fu2=get_fu(slab2)
                print(nl1,nl2,Eslab1,Eslab2,fu1,fu2)

        else:
                slab1_vac=slab1.copy();slab2_vac=slab2.copy();
                slab1_vac.center(vacuum=4.0, axis=2)
                slab2_vac.center(vacuum=4.0, axis=2)
                #atoms=call_castep(slab1_vac,typ="SP",dipolCorr='SC',name='slab1',ENCUT=ecut,KPgrid='1 1 1',PP=pp)
                atoms=slab1_vac
                atoms.set_calculator(calc)
                Eslab1=atoms.get_potential_energy()
                fu1=get_fu(atoms)
                
                #atoms=call_castep(slab2_vac,typ="SP",dipolCorr='SC',name='slab2',ENCUT=ecut,KPgrid='1 1 1',PP=pp)
                atoms=slab2_vac
                atoms.set_calculator(calc)
                atoms.calc._label="slab2"
                Eslab2=atoms.get_potential_energy()
                fu2=get_fu(atoms)

        ase.io.write("slab1.cell",slab1.repeat((1,1,1)),format='castep-cell')
        ase.io.write("slab2.cell",slab2.repeat((1,1,1)),format='castep-cell')
        #ase.io.write("slab11.cell",slab1a,format='castep-cell')

        Ws1=(Eslab1-fu1*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
        Ws2=(Eslab2-fu2*Ebulk2)/2/surf_area(slab2)/0.01 #A2 to nm2
        
        print(('%s: %s eV' % ('Ebulk 1', Ebulk1)))
        print(('%s: %s eV' % ('Ebulk 2', Ebulk2)))
        print(('%s: %s eV' % ('Eslab 1', Eslab1)))
        print(('%s: %s eV' % ('Eslab 2', Eslab2)))
        print(('%s: %.2f eV/nm^2' % ('Wsurf 1', Ws1)))
        print(('%s: %.2f eV/nm^2' % ('Wsurf 2', Ws2)))

        #exit()
        
        #Interface before alignment.
        print("Creating the interface with given slabs.")
        int1=ase.build.stack(slab1, slab2, axis=2, maxstrain=None, distance=dist,cell=None,reorder=True)
        int1.center(vacuum=4.0,axis=2)
        view(int1)
        ase.io.write("interface1.cell",int1.repeat((1,1,1)),format='castep-cell')
        atoms=call_castep(slab1_vac,typ="SP",dipolCorr=dC,name='int1',ENCUT=500,KPgrid='1 1 1',PP=pp)
        #atoms=int1
        #atoms.set_calculator(calc)
        Eint=atoms.get_potential_energy()

        #Wad=(Ws1+Ws2-Eint)/surf_area(int1)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf?? This is wrong
        #print('Wad before alingment: %.2f eV/nm^2'%Wad)

        Wad=(Eslab1+Eslab2-Eint)/surf_area(int1)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf??
        
        print(('W_ad before alingment: %.2f eV/nm^2'%Wad))
        return Eslab1,Eslab2,Ws1,Ws2,int1#surf_area(int1)

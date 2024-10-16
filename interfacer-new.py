#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import argparse,math, os.path,fractions,time
import matplotlib.pyplot as plt

from os import system,popen,chdir,getcwd,getenv,putenv,listdir
from re import search
from sys import exit, stdout
from copy import deepcopy

#from spglib import find_primitive,standardize_cell,get_spacegroup #, niggli_reduce #niggli_reduce from ASE-tools conflicts with that from spglib.

#ASE related imports
#from ase.build import *
import ase, ase.io, ase.build
from ase.visualize import view
from ase import Atoms,Atom
import ase.calculators.castep,ase.calculators.vasp
from ase.calculators.vasp import Vasp #,Vasp2 #requires ase 3.17.0
from ase.build import niggli_reduce, cut
from ase.geometry import find_mic  #min dist through PBC using the minimum-image covnention
from ase.constraints import FixAtoms
#from ase.spacegroup import crystal
#import ase.build.tools
 

#Import other functions from the INTERFACER package
from tools.builder import *
from tools.external import *
from tools.analysis import *


"""
TODO:

-Name the interface and sla1/2-aligned folders with proper naming, e.g. Na_111-Xlayers and Na_111-Na3Ocl_100 (-sep_3A)

-Check output slab1_aligned1/2.vasp, the structures used in calculations are correct but the printed ones do not look good for Li3PS4-LiNbO3 interfaces. (ASE related)
-Surpress too detailed printing (unless -v // --verb is switched on) in slab/interface building (build.py)

-CHECK energy calculations and picking the most stable surface/interface (lowest/highest?)
- FIX the reading of Miller indices from user input in args.miller_list
- Make sure that all interfaces slabs are not flipped randomly (due to bugs in ase.build.stack and cut functions)
- Make sure that the ase. build cut and stack are working correctly. For interlayer sepration in stack does not work at times (two surface slab layers clash at low separation/distance!!)
-ASE surface slab (Cut function) is problematic for Na3OCL (110) and (111) surfaces (O clashes with the upper layer)

- Add horizontal/lateral translation optimisation with scipy/BFGS
- Add support for 3-,4-layer interfaces
- Take args.parser (command-line) argument from an input file (interfacer.inp), args.parser will overwrite the file input.
- Add the finite surface slab energy correction scheme as given in Lepley et al 2015 paper.
- Add support fixing atoms (e.g. middle layers, for a bulk like region in surface slabs.) during geom optimsiations
- Add ONETEP and GPAW calculator support for plane-wave DFT (can be installed using conda install gpaw)
- Add Grain-boundary modelling tools
- ?? QM/MM support ??

-Add support for CASTEP restart (use ase.io.castep.read_seed())

-Add support for minkowski_reduce
-Check from ase.geometry: 
        minkowski_reduce, standard_form and get_layers()
        handedness: triple product, VASP can work with right-handed ones, so the +1 value
        VASP can work with -ve z lattice paramter, but not z coords, also not with cells with positive triple product of a,b,c

---------
DONE: Changes done as compared to old version (v4.1)):
- Refactoring the code making it modular.
- LJ/Morse/EMT from ASE with BFGS optimiser 
- conv layer test
- surface stability test
- interlayer spacing (vertical translation) test  (Keeping slabs rigid and using single points)
- horizontal tranlation   (Keeping slabs rigid and using single points)
- gamma-only version KPgrid 1 1 1
- VASP Runs for surface analysis in seprate directories (e.g. Li_111), so support restart!!
- No dipole correction when no vacuum in interface.

"""

def grep(key,fname,n=-1): 
    #Uses grep to get the nth line with the keyword from a given file. By default the last occurence is returned.
        #try:   return popen4('grep -m %d "%s" %s '%(n,key,fname),"r")[1].readlines()[-1][0:-1] #don't take \n at the end.  #Fastest one!!!
        #except:   
        try: return popen('grep -m %d "%s" %s '%(n,key,fname),"r").read()[0:-1]
        except:  return ""

from subprocess import Popen,PIPE # as popen # check_output
from sys import stdout,version_info
def Popen4(cmd,timeout=None):
    proc=Popen(cmd,shell=True, stdin=PIPE, stdout=PIPE, close_fds=True)
    out,err=proc.communicate(timeout=timeout)
    #out=proc.stdout.readline() #does not work.

    if version_info[0] < 3: out2=[x for x in str(out).replace("b'",'').replace("'",'').split("\n") if x!=""] #python3 adds \\n to the end, python2 adds \n
    else:out2=[x for x in str(out).replace("b'",'').replace("'",'').split("\\n") if x!=""]

    return out2,err

def flip_cell(interface): #TODO: check if the resulting cross (triple) product of the resulting cell vectors is positive (whhich a an issue for VASP). THen it should not flip the cell
        #interface.rotate(a=180,v='x',rotate_cell=1,center = (0, 0, 0)) #this does not work as it switches the top layer/slab with bottom
        cell=interface.get_cell()
        cell[2][2]=-cell[2][2]
        interface.set_cell(cell,scale_atoms=0)
        interface.translate([0.,0.,cell[2][2]])
        a,b,c=interface.get_cell()
        #print (a,b,c,np.cross(b,c))
        # if np.dot(a,np.cross(b,c))>0:
        #         print('the resulting flipped cell is left-handed, which is not suitable for VASP');stdout.flush()
        #         #TODO: flip it in x and y axes
        return interface


def check_surfaces(atoms, miller_list,vac,Ebulk,creps):
        min_Ws=1e8; min_mil=();min_slab=None
        atoms_orig=atoms.copy()
        calc=atoms_orig.get_calculator()
        str1=''
        for i,mil in enumerate(miller_list):
                print ('Doing the (%d,%d,%d) surface...'%(mil[0],mil[1],mil[2]));stdout.flush()
                slab1 = make_slab(mil,atoms,repeat=(1,1,creps),square=False)
                #slab1 = make_slab(mil,atoms,repeat=(1,1,1),square=False) #orig
                slab1.center(vacuum=vac, axis=2)
                # if (slab1.get_cell()[2][2]<0): #VASP can't work with upside-down cells (with negative z-coordiantes)
                #         print ("check_surfaces: Cell extends towards -z direction, VASP cannot work with -ve z coords, rotating about x-axis by 180 degree to fix it");stdout.flush()
                #         #interface.rotate(a=180,v='x',rotate_cell=1,center = (0, 0, 0))
                #         slab1=flip_cell(slab1)
                #         if (slab1.get_cell()[2][2]<0): print("check_surfaces: That didn't seem to work");stdout.flush()

                slab1.set_calculator(calc)
                name="%s_%d%d%d"%(slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])
                #???Which one better: Single point or geometry opt???
                if args.prog=='castep':x=call_castep(slab1,typ="geom",dipolCorr='sc',name="CASTEP-tmp/"+name,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,PP=pp) #KPgrid='4 4 1'
                elif args.prog=='vasp':x=call_vasp(slab1,typ="geom",dipolCorr='sc',name="VASP-tmp/"+name,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,xc=xc,magmom=args.magmoms,sigma=sigma,exe=exe)
                elif args.prog=='ase':x=call_ase(slab1,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=0)

                calc=slab1.get_calculator()
                slab1=x[-1]
                Eslab1=x[0]
                #-ve / +ve
                Ws1=(Eslab1-len(slab1)*Ebulk)/2/surf_area(slab1)/0.01 #A2 to nm2 #Normally this is defined with a +ve sign, here we do it with -ve on purpose to determine the lowest (i.e. highest) Wad, converted back to +ve at the end.
                #TODO:Add surface energy
                str1+='%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('W_surf ', slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2],Ws1,Ws1*cfactor)
                if i==0 or Ws1<min_Ws: min_Ws=Ws1; min_mil=mil;min_slab=slab1.copy()
                #outf.writelines(str1)
                #print (str1);stdout.flush()

                #TODO: print the structure if the current surface
                if args.prog=="castep":ase.io.write("OUTPUT/%s.cell"%name,slab1,format='castep-cell'); 
                elif args.prog=="vasp":ase.io.write("OUTPUT/%s.vasp"%name,slab1,format='vasp',vasp5=1);
                elif args.prog=="ase": ase.io.write("OUTPUT/%s.xyz" %name,slab1,format='extxyz');     

#(slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])
#(slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])
#(slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])

        print(str1);stdout.flush()
        #Ws1=-Ws1 #convert back to +ve surface formation energy
        Ws1=-min_Ws #convert back to +ve surface formation energy
        print('check_surfaces: Minimum-energy surface for material %s is %s with %d atoms and W_surf=%.2f eV'%(min_slab.get_chemical_formula(empirical=1),min_mil,min_slab.get_global_number_of_atoms(),Ws1));stdout.flush()

        #slab1=min_slab.copy()
        min_slab.set_calculator(calc)
        return Ws1, min_slab,calc,min_mil

#def conv_layers(atoms,vac=2.0,ifPlot=0,ifPrim=False, Etol=1e-2,view=0,lay_min=0,lay_max=5):#layer convergence test (Input atoms with a calc object). 
def conv_layers(atoms,vac=2.0,ifPlot=0,ifPrim=False, Etol=1e-2,view=0,lay_min=0,lay_max=5):#layer convergence test (Input atoms with a calc object).
        #print "Convergence of E/atom vs. #layers"
        #TODO: Take from user
        #Etol=1e-2 #eV/atom
        #Ftol=5e-2 #eV/Angstroem
        #Estr=0.1 #GPa

        #Initial values
        E=[0]; F=[0]; S=[0]
        name=atoms.name
        #name=atoms.get_chemical_formula(empirical=1)
        atoms_orig=atoms.copy()
        calc=atoms.get_calculator()

        #find the primitive cells to reduce comp. efforts.
        if ifPrim: atoms=find_prim(atoms);atoms.set_calculator(calc)

        atoms.center(vacuum=vac, axis=2)
        #nAt=atoms.get_global_number_of_atoms()
        #atoms.set_calculator(calc)
        #E.append(atoms.get_potential_energy()/nAt)
        i=lay_min;layers=[1]
        flag=0 #converged?
        while  i<=lay_max:
                layers.append(1+1*i) #increase 2 layers at a time
                atoms=atoms_orig.copy()
                atoms=atoms.repeat((1,1,layers[-1]))
                atoms.center(vacuum=vac, axis=2)
                #atoms.set_calculator(calc)
                #if args.view:xview(atoms)
                #nAt=atoms.get_number_of_atoms()
                nAt=atoms.get_global_number_of_atoms()
                if args.prog=='castep':x=call_castep(atoms,typ="sp",dipolCorr='sc',name='CASTEP-tmp/%s_%d-layer'%(name,layers[-1]),ENCUT=ecut,KPspacing=KP,PP=pp) #KPgrid='4 4 1'
                elif args.prog=='vasp':x=call_vasp(atoms,typ="sp",dipolCorr='sc',name='VASP-tmp/%s_%d-layer'%(name,layers[-1]),ENCUT=ecut,KPspacing=KP,xc=xc,magmom=args.magmoms,sigma=sigma,exe=exe)
                elif args.prog=='ase':x=call_ase(atoms,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=0)
                atoms=x[-1]
                Ecurr=x[0]
                #E.append(atoms.get_potential_energy()/nAt)
                E.append(Ecurr/nAt)
                print("Iter. #%d, #layers: %d, #atoms: %d "%(i+1,layers[-1],nAt));stdout.flush()
                print("E_total: %.5f; deltaE: %.3e eV/atom; target: %.3e eV."%(E[i],abs(E[i]-E[i-1]),Etol));stdout.flush()
                if view: view(atoms)

                #Writing layer steps
                if i==lay_min: app=0 #append
                else:app=1
                if args.prog=="castep":ase.io.write("OUTPUT/%s-layer.cell"%(name),atoms,format='castep-cell',append=app);
                elif args.prog=="vasp":ase.io.write("OUTPUT/%s-layer.vasp"%(name),atoms,format='vasp-xdatcar',append=app);
                elif args.prog=="ase": ase.io.write("OUTPUT/%s-layer.xyz" %(name),atoms,format='extxyz',append=app);
                if i!=0 and abs(E[i]-E[i-1]) <= Etol:
                       #print('Layer thickness of %d converged to %.3f eV'%(Etol);stdout.flush()
                       flag=1
                       break
                i += 1

        if flag:
               print("conv_layers: E/atom converged to %.2e eV with %d layers."%(Etol,layers[-1]));stdout.flush()
        else:
               print('conv_layers: Layer thickness not converged to %.2e eV within %d layers '%(Etol,layers[-1]));stdout.flush()

        if ifPlot: #Do plotting of E/atom vs. #layers
                #print(layers[1:],E[1:])
                plt.plot(layers[1:],E[1:], 'ko-')
                plt.xlabel('Number of layers')
                plt.ylabel('Energy per atom (eV/atom)')
                plt.savefig('OUTPUT/conv_layers.png')
                #plt.show()

        return layers[-1],E[-1]*nAt,atoms

def conv_layers_new(mil,atoms,vac=2.0,ifPlot=1,ifPrim=False, Etol=1e-2,view=0,lay_min=0,lay_max=5):#layer convergence test (Input atoms with a calc object).         
        print ("Convergence of E/atom vs. #layers")
        #TODO: Take from user
        #Etol=1e-2 #eV/atom
        #Ftol=5e-2 #eV/Angstroem
        #Estr=0.1 #GPa

        #Initial values
        E=[0]; F=[0]; S=[0]
        name=atoms.name
        #name=atoms.get_chemical_formula(empirical=1)
        name="%s_%d%d%d"%(atoms.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])

        atoms_orig=atoms.copy()
        calc=atoms.get_calculator()

        #find the primitive cells to reduce comp. efforts.
        if ifPrim: atoms=find_prim(atoms);atoms.set_calculator(calc)
        
        atoms.center(vacuum=vac, axis=2)
        #nAt=atoms.get_global_number_of_atoms()
        #atoms.set_calculator(calc)
        #E.append(atoms.get_potential_energy()/nAt)
        i=lay_min;layers=[1]
        flag=0 #converged?
        while  i<=lay_max:
                layers.append(1+1*i) #increase 2 layers at a time
                #atoms=atoms_orig.copy()
                #atoms=atoms.repeat((1,1,layers[-1]))
                #atoms.center(vacuum=vac, axis=2)

                #atoms.set_calculator(calc)
                nAt=atoms.get_global_number_of_atoms()
                slab = make_slab(mil,atoms,repeat=(1,1,i+1),square=False)
                slab.center(vacuum=vac, axis=2)

                if args.prog=='castep':x=call_castep(slab,typ="sp",dipolCorr='sc',name='CASTEP-tmp/%s_%d-layer'%(name,layers[-1]),ENCUT=ecut,KPspacing=KP,PP=pp) #KPgrid='4 4 1'
                elif args.prog=='vasp':x=call_vasp(slab,typ="sp",dipolCorr='sc',name='VASP-tmp/%s_%d-layer'%(name,layers[-1]),ENCUT=ecut,KPspacing=KP,xc=xc,magmom=args.magmoms,sigma=sigma,exe=exe)
                elif args.prog=='ase':x=call_ase(slab,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=0)
                slab=x[-1]
                Ecurr=x[0]
                #E.append(atoms.get_potential_energy()/nAt)
                E.append(Ecurr/nAt)
                print("Iter. #%d, #layers: %d, #atoms: %d "%(i+1,layers[-1],nAt));stdout.flush()
                print("E_total: %.5f; deltaE: %.3e eV/atom; target: %.3e eV."%(E[i],abs(E[i-lay_min+1]-E[i-lay_min]),Etol));stdout.flush()
                if view: view(slab)

                #Writing layer steps
                if i==lay_min: app=0 #append
                else:app=1
                if args.prog=="castep":ase.io.write("OUTPUT/%s-layer.cell"%(name),slab,format='castep-cell',append=app); 
                elif args.prog=="vasp":ase.io.write("OUTPUT/%s-layer.vasp"%(name),slab,format='vasp-xdatcar',append=app);
                elif args.prog=="ase": ase.io.write("OUTPUT/%s-layer.xyz" %(name),slab,format='extxyz',append=app);      

                if i!=0 and abs(E[i-lay_min+1]-E[i-lay_min]) <= Etol:
                       #print('Layer thickness of %d converged to %.3f eV'%(Etol);stdout.flush()
                       flag=1
                       break
                i += 1

        if flag:
               print("conv_layers: E/atom converged to %.2e eV with %d layers."%(Etol,layers[-1]));stdout.flush()
        else:
               print('conv_layers: Layer thickness not converged to %.2e eV within %d layers '%(Etol,layers[-1]));stdout.flush()
                     
        if ifPlot: #Do plotting of E/atom vs. #layers
                #print(layers[1:],E[1:])
                plt.plot(layers[1:],E[1:], 'ko-')
                plt.xlabel('Number of layers')
                plt.ylabel('Energy per atom (eV/atom)')
                plt.savefig('OUTPUT/conv_layers.png')
                #plt.show()

                
                
        return layers[-1],E[-1]*nAt,slab               


def convSep(slab1,slab2,vac=None,sep_init=1.5,sep_final=4.0,sep_step=0.5,view=0,ifPlot=1, Etol=1e-2): #Converge the sepration bet ween the two surface slabs (keeping them rigid)
        # TODO: find a way to check whether two slabs are clashing (i.e. <2.0A), 
        # ase.build.stack function can give interfaces (with distance option), clashing surface slabs!!! 
        # Find a way to initially create a safe interface (with proper seperation) and adjust the interlayer spacing manually

        #Initial values
        E=[0]; F=[0]; S=[0]
        #name=atoms.name

        calc=slab1.get_calculator()
        ln1=len(slab1); ln2=len(slab2)

       # delete the vacuum padding in the slabs !!!! Realised that this prevents the application of correct separation
        if 0:
                slab1.center(vacuum=0, axis=2)
                slab2.center(vacuum=0, axis=2)

        minSep=sep_init; Emin=1e6; minInt=None
        seps=np.arange(sep_init,sep_final,sep_step)
        str1=''
        for i,sep in enumerate(seps):
                print ('Doing the %.2f A separation between two layers'%sep);stdout.flush()
                # CHECK whether the deleting vacuum earlier has an adverse effect!!
                # Check min z of min slab and max z of bottom slab 
                interface=ase.build.stack(slab1, slab2, axis=2, maxstrain=None, distance=sep, cell=None,reorder=1)  #using 0 distance btw slabs gives CASTEP error.
                if vac:vacc=vac
                else:vacc=sep/2
                interface.center(vacuum=vacc, axis=2) #Vacuum on both sides. For dipole corrections at least 8A vacuum is needed.
                if (interface.get_cell()[2][2]<0): #VASP can't work with upside-down cells (with negative z-coordiantes)
                        print ("convSep: Cell extends towards -z direction, VASP cannot work with -ve z coords, fixing it...");stdout.flush()
                        interface=flip_cell(interface)
                interface.set_calculator(calc)
                a,b,c,alpha,beta,gamma=interface.cell.cellpar()
                
                nAt=interface.get_global_number_of_atoms()
                if args.prog=='castep':x=call_castep(interface,typ="sp",dipolCorr='None',name='CASTEP-tmp/interface-sep_%.2fA'%sep,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,PP=pp) #KPgrid='4 4 1'
                elif args.prog=='vasp':x=call_vasp(interface,typ="sp",dipolCorr='None',name='VASP-tmp/interface-sep_%.2fA'%sep,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,xc=xc,magmom=args.magmoms,sigma=sigma,exe=exe)
                elif args.prog=='ase':x=call_ase(interface,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=0)
                interface=x[-1]
                Ecurr=x[0]
                #E.append(Ecurr)
                #E.append(atoms.get_potential_energy()/nAt)
                E.append(Ecurr/nAt)
                if i==0 or  E[-1]<Emin: minSep=sep;Emin=E[-1];minInt=interface.copy()
                str1+="Iter. #%d, Seperation: %.2f A, E / atom : %.5f eV, z-thickness: %.2f A \n"%(i+1,sep,E[-1],c)
                print("Iter. #%d, Seperation: %.2f A, E / atom : %.5f eV, z-thickness: %.2f A \n"%(i+1,sep,E[-1],c));stdout.flush()
                #print("E_total: %.5f; deltaE: %.3e eV/atom; target: %.3e eV."%(E[i],abs(E[i]-E[i-1]),Etol));stdout.flush()
                if view: view(interface)

                #Save structures
                app=0
                if args.prog=="castep":ase.io.write("OUTPUT/int-sep_%.2fA.cell"%sep,interface,format='castep-cell',append=app); 
                elif args.prog=="vasp":ase.io.write("OUTPUT/int-sep_%.2fA.vasp"%sep,interface,format='vasp',vasp5=1,append=app);
                elif args.prog=="ase": ase.io.write("OUTPUT/int-sep_%.2fA.xyz" %sep,interface,format='extxyz',append=app);      

        #print(str1)
        print('\nconvSep: Minimum-energy separation is %.2f A with E= %.5f eV'%(minSep,Emin));stdout.flush()

        if minInt: interface=minInt.copy()

        if ifPlot: #Do plotting of E/atom vs. #layers
                #print(seps,E[1:])
                plt.plot(seps,E[1:], 'ko-')
                plt.xlabel('Interlayer Separation [A]')
                plt.ylabel('Energy per atom [eV/atom]')
                plt.savefig('OUTPUT/conv_htrans.png')


        if view:view(interface)

        #interface.set_calculator(calc)
       
        return interface,minSep

def checkHorizontal(slab1,slab2,sep,vac=None,steps=[0.25,0.25],ifPlot=0): #This checks the optimum horizontal stacking of two slabs through translating the slab2
        #TODO: complete this
        #Initial values
        E=[0]; F=[0]; S=[0]

        if 0: #delete vacuum paddings
          slab1.center(vacuum=0, axis=2)
          slab2.center(vacuum=0, axis=2)

        #1. Form a grid (get dimensions/steps from user)
        #2. translate top layer (slab1) using slab2.translate
        calc=slab2.get_calculator()
        #slab2_orig=slab2.copy()

        if vac:vacc=vac
        else:vacc=sep/2
        ln1=len(slab1); ln2=len(slab2)

        a,b,c=slab1.cell.lengths()
        #a,b,c,_,_,_=slab1.cell.cellpar() #same as above
        print('Cell lengths: %.2f A %.2f A %.2f A'%(a,b,c));stdout.flush()
        #print('Cell disp:', slab1.get_celldisp());stdout.flush()

        interface=ase.build.stack(slab1, slab2, axis=2, maxstrain=None, distance=sep,cell=None,reorder=0)  
        #interface.wrap() #works
        interface.center(vacuum=vacc, axis=2) #Vacuum on both sides. For dipole corrections at least 8A vacuum is needed.
        int_orig=interface.copy()

        xrng=np.arange(steps[0],1.00,steps[0])
        yrng=np.arange(steps[1],1.00,steps[1])
        i=0; minE=1e6;minInt=None;minxy=[0.,0.]
        for x,y in zip(xrng,yrng):      #work in fractional coords. Convert to Cartesian for translation
                #slab2=slab2_orig.copy()
                interface=int_orig.copy()

                cart_disp=interface.cell.cartesian_positions([x,y,0])
                #print('Displacement in Fractionals: x= %.2f y= %.2f   in Cartesians: x= %.2f A , y= %.2f A'%(x,y,cart_disp[0],cart_disp[1]))
                #slab2.translate(cart_disp) #translate it as a whole
                disp=[cart_disp for j in range(ln1)]
                disp.extend([[0., 0., 0.] for j in range(ln2)])
                interface.translate(disp)
                interface.wrap()
                #view(interface)

                nAt=interface.get_global_number_of_atoms()
                interface.name='htrans_x%.2f-y%.2f'%(x,y)
                if args.prog=='castep':xx=call_castep(interface,typ="sp",dipolCorr='None',name='CASTEP-tmp/htrans_x%.2f-y%.2f'%(x,y),ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,PP=pp) #KPgrid='4 4 1'
                elif args.prog=='vasp':xx=call_vasp(interface,typ="sp",dipolCorr='None',name='VASP-tmp/htrans_x%.2f-y%.2f'%(x,y),ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,xc=xc,magmom=args.magmoms,sigma=sigma,exe=exe)
                elif args.prog=='ase':xx=call_ase(interface,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=0)
                interface=xx[-1]
                Ecurr=xx[0]
                E.append(Ecurr/nAt)

                if i==0 or  E[-1]<minE: minE=E[-1];minInt=interface.copy();minxy=[x,y]
                print("Displacement: x= %.2f, y= %.2f (x= %.2f A , y= %.2f A); E/atom: %.5f eV, "%(x,y,cart_disp[0],cart_disp[1],E[-1]));stdout.flush();stdout.flush()

                #Save structures
                app=0
                if args.prog=="castep":ase.io.write("OUTPUT/htrans_x%.2f-y%.2f.cell"%(x,y),interface,format='castep-cell',append=app); 
                elif args.prog=="vasp":ase.io.write("OUTPUT/htrans_x%.2f-y%.2f.vasp"%(x,y),interface,format='vasp',vasp5=1,append=app);
                elif args.prog=="ase": ase.io.write("OUTPUT/htrans_x%.2f-y%.2f.xyz"%(x,y),interface,format='extxyz',append=app);      

                i+=1

        print('Minimum-energy horizontal translation: x= %.2f, y= %.2f with E= %.5f eV/atom'%(minxy[0],minxy[1],minE));stdout.flush()

        if ifPlot: #Do plotting of E/atom vs. #x-y translation
                #TODO: COMPLETE this! Do a 2D colour map
                #print(xrng,yrng,E[1:])
                #plt.plot(seps,E[1:], 'ko-')
                None
                # plt.xlabel('x displacement [A]')
                # plt.ylabel('y displacement [A]')
                # plt.zlabel('Energy per atom [eV/atom]')

                # plt.savefig('OUTPUT/conv_htrans.png')

        return minInt,minE,minxy

def checkHorizontal2(slab1,slab2,sep,vac=None): #This checks the optimum horizontal stacking of two slabs through translating the slab2
        #TODO: complete this!! This is for doing scipy opt with respect to horizantal/vertical translations, getting SP energies.

        print("checkHorizontal: Not implemented yet");stdout.flush()
        exit() 
        ln1=len(slab1); ln2=len(slab2)

        if 0: #delete vacuum paddings
          slab1.center(vacuum=0, axis=2)
          slab2.center(vacuum=0, axis=2)

        calc=slab2.get_calculator()

        if vac:vacc=vac
        else:vacc=sep/2         
        interface=ase.build.stack(slab1, slab2, axis=2, maxstrain=None, distance=sep,cell=None,reorder=0)  #using 0 distance btw slabs gives CASTEP error.
        #interface.wrap() #works
        interface.center(vacuum=vac, axis=2) #Vacuum on both sides. For dipole corrections at least 8A vacuum is needed.

        #interface[0:ln1].write('bk.xyz')
        #interface[ln1:].write('bk2.xyz')

#indices=[interface.index for atom in atoms if atom.symbol == 'Cu'],
        """
        from ase.constraints import FixedPlane
        c1 = FixedPlane( 
        a=list(range(0,ln1)),
        #direction=[0, 0, 1], #xy-plane
        direction=np.array([[0, 0, 1] for i in range(0,ln1)]), #xy-plane
        #problem with dimensions??
        )
        c2 = FixedPlane( 
        a=list(range(ln1,ln1+ln2)),
        #direction=[0, 0, 1], #xy-plane
        direction=[[0, 0, 1] for i in range(0,ln2)], #xy-plane
        )
        """

        """
        from ase.constraints import FixInternals
        indices1 = range(0,ln1)
        #bond1 = [[interface.get_distance(ind) for ind in indices1], indices1]
        #angle1 =[[interface.get_angle(ind) for ind in indices1], indices1]
        bond1=interface.get_distances(a=,indices=indices1,mic=1)
        #bond1=[]
        #angle1=interface.get_angles(indices1,mic=1)
        #c1=[bond1,angle1]
        c1 = FixInternals(bonds=[bond1], angles_deg=[angle1])
        print(c1);stdout.flush()

        indices2 = range(0,ln1)
        bond2 = [interface.get_bond(*indices2), indices2]
        angle2 = [interface.get_angle(*indices2), indices2]
        c2=[bond2,angle2]
        """

        from ase.constraints import FixBondLengths
        pairs=[]
        for i in range(ln1):
               for j in (i+1,ln1):
                      if i!=j: pairs.append([i,j])

        c1=FixBondLengths(pairs)
        c2=None
        interface.set_constraint([c1,c2])

        interface.name='lat_shift'
        #TODO:need to write a seprate fnc for the VASP and CASTEP as it does not support the contraints directly. Use an ASEoptimiser with VASP/CASTEP SP calls
        if args.prog=='castep':x=call_castep(interface,typ="sp",dipolCorr='None',name='slab1',ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,PP=pp) #KPgrid='4 4 1'
        elif args.prog=='vasp':x=call_vasp(interface,typ="sp",dipolCorr='None',name='slab1',ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,xc=xc,magmom=args.magmoms,sigma=sigma,exe=exe)
        elif args.prog=='ase':x=call_ase(interface,ctype=args.ase_pot,fmax=args.ase_fmax,steps=1000,opt=1) #steps=args.ase_gsteps
 
        interface=x[-1]
        Ecurr=x[0]

        #Delete the constraints
        interface.set_constraint()

        return interface,Ecurr
        




##################################################
#                  MAIN METHOD                   #
##################################################
try:    
        vasppp = os.environ['VASP_PP_PATH']
        ppdirs=os.listdir(vasppp)
        ppdirs.sort()
except: ppdirs=""

cfactor=6.2415093 #eV / nm^2 to J / m^2

if __name__== '__main__':
        #read in arguments
        parser = argparse.ArgumentParser()

        #Basic input/output related keywords
        parser.add_argument("-i1", "--infile1", required=1)
        parser.add_argument("-i2", "--infile2", required=1)
        parser.add_argument("-o", "--outfile",default="interfaces.out")
        parser.add_argument("-m1","--miller1", default=(1,0,0), nargs="+", type=int)
        parser.add_argument("-m2","--miller2", default=(1,0,0), nargs="+", type=int)
        parser.add_argument("-surf1","--surface1", default=False,help="extra input file for a surface structure of i1")
        parser.add_argument("-surf2","--surface2", default=False,help="extra input file for a surface structure of i2")

        #Keywords related to structure building
        parser.add_argument("-cr1","--creps1",default=None,type=int,help='Repetitions in c direction (number of layers) for surface model 1')
        parser.add_argument("-cr2","--creps2",default=None,type=int,help='Repetitions in c direction (number of layers) for surface model 2')
        parser.add_argument("-msd","--max_slab_dimension",default=50, type=float,help='max length of cell sides')
        parser.add_argument("-th","--thickness",default=7,type=float,help='Thickness of the slab (in c-direction), def: 7A')
        parser.add_argument("-pt","--percentage_tolerance",default=4,type=float,help='percentage tolerances for angle and length matching between two slabs')
        parser.add_argument("-Linit","--initLength",default=3,type=float,help='Initial (minimum) length of surface slabs in a/b direction, def: 3A') #orig def: 5A
        parser.add_argument("-Lstep","--stepLength",default=2,type=float,help='step size for the increasing the length in a/b direction, def: 2A') #orig def: 5A

        parser.add_argument("-vac_slab","--vac_slab",default=6.,type=float,help='Size of vacuum padding in z direction (only for slabs), def: 6 A on both sites')
        parser.add_argument("-vac_int","--vac_int",default=1.,type=float,help='Size of vacuum padding in z direction (only for interface), def: 1.0 A on both sites') #should be equal to args.sep/2 by default!       
        parser.add_argument("-sep","--sep",default=2.0,type=float,help='Seperation between the surface slabs forming the interface, def: 2.0 A')
        parser.add_argument("-pass", "--pas",action="store_true",default=False,help="To passivate the top and bottom surface of the interface slab based on dangling bonds.")
        parser.add_argument('-prim','--prim', default=False,action='store_true', help='Use primitive cell for bulk 1 and 2. Def: no')
        parser.add_argument('-niggli','--niggli', default=False,action='store_true', help='Niggli reduce the resulting interface. Def: no')
        parser.add_argument("-flip2", "--flip2",action="store_true",default=False,help="Flip the slab2 upside-down.")
        parser.add_argument("-sym_slabs", "--symSlabs",action="store_true",default=False,help="Use to create symmetric surface slabs. Def: no")

        #Convergence tests related keywords:
        parser.add_argument('-tsur',"-test_surf", "--checkSurfs",action="store_true",default=False,help="To check the stbility of different surfaces for the given materials. Def: No")
        parser.add_argument('-tlay',"-test_layer", "--convLayers",action="store_true",default=False,help="To run a convergence test on the layer thickness for slab(s) generated. Def: No")
        parser.add_argument('-tsep',"-test_sep", "--convSep",action="store_true",default=False,help="To run a convergence test on the interlayer separation for the interface generated. Def: No")
        parser.add_argument('-thor',"-test_hori", "--test_hori",action="store_true",default=False,help="To run a convergence test on the interlayer separation for the interface generated. Def: No")
        #TODO: parsing of the miller indices are not working right!! Fix needed; wok with the default value though
        parser.add_argument("-mlist","--miller_list", default=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,1,1)], nargs="*", type=tuple,help='list of Miller indices to check during the surface stability analysis, i.e. --checkSurfs')  #(1,1,1),
        #parser.add_argument("-mlist","--miller_list", default="[1 1 1], [1 1 0], [1 0 0],[0 1 0]", nargs="*", type=str,help='list of Miller indices to check during the surface stability analysis, i.e. --checkSurfs')
        parser.add_argument("-sep_init","--sep_init"  ,default=1.0,type=float,help='Initial interface separation to be used in seperation convergence test (--convSep), def: 1.0 A')
        parser.add_argument("-sep_final","--sep_final",default=3.0,type=float,help='Final interface separation to be used in seperation convergence test (--convSep), def: 3.0 A')
        parser.add_argument("-sep_step","--sep_step"  ,default=0.25,type=float,help='Step size in interface separation to be used in seperation convergence test (--convSep), def: 0.25 A')
        parser.add_argument("-lay_min","--lay_min"  ,default=2,type=int,help='Min layers to include n layer thickness convergence test (--convLay), def: 2 layers')
        parser.add_argument("-lay_max","--lay_max"  ,default=8,type=int,help='Max layers to include n layer thickness convergence test (--convLay), def: 8 layers')
        parser.add_argument("-hor_steps","--hor_steps"  ,default=[0.25,0.25],nargs=2,type=float,help='Test the energy as a function of horizontal translations of bottom layer in x and y directions (--test_hori), def: [0.25, 0.25]]')



        #Calculation related keywords
        parser.add_argument("-prog","--prog",choices=['castep','vasp','ase'],required=1,help="Code to be used: CASTEP, VASP or ASE (LJ/Morse, default) (def)")
        parser.add_argument('-xc','--xc', type=str,required=False, default='PBE',help='Exchange correlation functional to use (pseudopots will be selected accordingly, e.g. LDA, PBE,PBEsol, etc. Def: PBE')
        parser.add_argument('-ec','--encut', type=float,required=False, default=520,help='Cutoff energy for the planewaves Def: 520 eV')
        parser.add_argument('-kp_sp','--kpspacing', type=float,required=False, default=0.32,help='KP grid spacing, Def: 0.05 * 2 pi = 0.32 1/A')
        parser.add_argument('-kp_gr','--kpgrid', type=str,required=False, default=None,help='KP grid dimensions (Using KP spacing is recommended), Def: None')
        parser.add_argument('-gamma','--gamma_only', default=False,action='store_true', help='Set KP grid to 1 1 1 and use only the Gamma point for surface slabs and interface. This switches on the gamma_oonly version of VASP. Default: srun is used')
        parser.add_argument('-sig','--sigma', type=float,required=False, default=0.05,help='Gaussian smearing width, Def:0.05 eV')
        parser.add_argument("-mgm","--magmoms", default=None, nargs="*", required=False, help="Magnetic moments for a collinear calculation. Eg, 'Fe 5.0 Nb 0.6 O 0.6' Def.: None. If a defined element is not present in the POSCAR, no MAGMOM will be set for it.")
        parser.add_argument("-hubU", "--hubU", type=str,nargs="*",help="For defining the Hubbard U parameters (in eV) for specific atom and orbital types, e.g. -hubU Fe d 2.7, Sm f 6.1")
        parser.add_argument('-ase_pot','--ase_pot', type=str,required=False, default='LJ',choices=['LJ','Morse','EMT'], help="Potential type to be used through the ASE infrastructure, only relevant when using -prog 'ASE'. Def: LJ")
        parser.add_argument('-ase_fmax','--ase_fmax', type=float,required=False, default=0.05, help='Force convergence criterion when using the ASE built-in geometry optimiser. Def: 0.05 eV/A')
        parser.add_argument('-ase_gsteps','--ase_gsteps', type=int,required=False, default=200, help='Max number of BFGS steps allowed in the ASE built-in geometry optimiser. Def: 200 geometry steps')
        parser.add_argument('-mp','-makepotcar','--makepotcar', default=False,action='store_true', help='to compile POTCAR (for VASP) using actual atomic content. The environment variable ($VASP_PP_PATH) should be defined beforehand. Default: False.')
        parser.add_argument('--potcarxc', default="potpaw_PBE",type=str,choices=ppdirs,help='XC functional to use in making of POTCAR. Def: potpaw_PBE')
        parser.add_argument('-cparam', '--castep_param',default="input.param",type=str,help='CASTEP parameter file to read (optional). Def: input.param')
        parser.add_argument('-ccell', '--castep_cell',default="input.cell",type=str,help='CASTEP cell file to read (optional). Def: input.cell')
        parser.add_argument('-cpp', '--castep_PP',default="OTF",type=str,help='CASTEP pseudopotential (PP, options are OTF, 00PBE, etc.). Def: OTF')
        

        #TODO: add support for  KPGrid determination
        
        #Parallelisation related
        #TODO: construct the CASTEP/VASP exe using this info, rather than relying on reading the environmen variables.
        parser.add_argument('-np', '--nprocs',type=int, default=32,help="No of processes to start for each CASTEP/VASP calculation through srun/mpirun. Def:32")
        parser.add_argument('-nt', '--ntasks',type=int, default=1,help="No of CASTEP/VASP tasks to run simultaneously. Def:1")
        parser.add_argument('-nn', '--nnodes',type=int, default=1,help="No of nodes to run CASTEP/VASP runs through srun. Def:1")
        parser.add_argument('-mpi','--mpirun', default=False,action='store_true', help='Use mpirun for parallel runs. Default: srun is used')
        parser.add_argument('-vexe','--vexe', type=str,required=False, default='vasp_std',help='Vasp exectuable. Def: vasp_std')
        parser.add_argument('-mpiargs','--mpiargs', type=str,required=False, default='',help='Vasp exectuable. No default value. "--partition=solbatsims" needed on zx81')

        #Other general keywords
        parser.add_argument("-dry", "--dry",action="store_true",default=False,help="Make a dry run, do not run CASTEP/VASP calculation. Good for checking the slabs and the interfacing algorithm.")
        parser.add_argument("-view", "--view",action="store_true",default=False,help="To view the generated structures.")
        parser.add_argument('-v','--verb', default=False,action='store_true', help='Verbose output. Default: False.')
        parser.add_argument('-quick','--quick', default=False,action='store_true', help='Setup a quick test run using ASE/LJ potentials to check functionality. Default: False.')


        #TODO: Add the -restart/overwrite options for when bulk1/2,slab1/2 folders exist. 
        #TODO: Also add restart support for the convergence tests!


        args = parser.parse_args()
        

        args.prog=args.prog.lower()

        if args.quick: #Set the parameters for quick testing 
               args.prog='ase'; 
               args.ase_pot='LJ'
               #args.ase_pot='Morse'
               args.ase_gsteps=1 
               args.ase_fmax=0.01


        if args.prog=='vasp':
                if args.mpirun: exe="mpirun -n %d %s"%(args.nprocs,args.vexe)
                else: exe="srun -n %d %s %s"%(args.nprocs,args.mpiargs,args.vexe)

        #Parse the Hubbard_U info and make the dict
        hubU={}
        if args.hubU:
                U=" ".join(args.hubU)
                xxx=U.split(',')
                for xx in xxx:
                        x=xx.split()
                        if not x[0] in hubU:  hubU[x[0]]={x[1]:x[2]}
                        else: hubU[x[0]][x[1]]=x[2]
                print ("Hubbard U values to be used: ",hubU);stdout.flush()

        KPgrid=args.kpgrid
        if args.gamma_only:   KPgrid="1 1 1"
        else:                 KPgrid=None

        #2 is going on the bottom so m2 ->-m2 to get right orientation of the surface (in James' new version)
        #miller2 = [-x for x in miller2] #It's better than using the atoms.rotate function, as the latter could change the surface slabs alignment. However this have clashin atoms issue with the ASE stack function.  Bu halen daha tam calismiyor, emin omak lazim !!!
        infile1 = args.infile1
        miller1 = tuple(args.miller1)
        infile2= args.infile2
        miller2 = tuple(args.miller2)
        
        miller_list=args.miller_list
        #print(miller_list)
        #miller_list=[]
        #mils = args.miller_list.split(',')
        #print (mils)
        #for mil in mils:
        #        print(mil.tolist())
                #mil=mil.replace('[','')
                #mil=mil.replace(']','')
                #millermil.split()

        #read in atoms and construct slab, need to repeat atoms to make view work
        print("Reading data from %s and %s."%(infile1,infile2));stdout.flush()
        atoms1 = ase.io.read(infile1) 
        atoms2 = ase.io.read(infile2) 

        outfile = args.outfile
        system("\n date > %s"%outfile)
        outf=open(outfile,'a')

        print("Structure 1: %-8s with fu=%d and %s lattice"%(atoms1.get_chemical_formula(empirical=1),get_fu(atoms1),atoms1.get_cell().get_bravais_lattice()));stdout.flush()
        print("Structure 2: %-8s with fu=%d and %s lattice"%(atoms2.get_chemical_formula(empirical=1),get_fu(atoms2),atoms2.get_cell().get_bravais_lattice()));stdout.flush()

        if args.prim: 
                print('Using primitive cells as input as requested');stdout.flush()
                atoms1=find_prim(atoms1); atoms2=find_prim(atoms2) #Whether to use primitive cells of input structures.

        #print("Structure 1: %-8s with fu=%d and %s lattice"%(atoms1.get_chemical_formula(empirical=1),get_fu(atoms1),atoms1.get_cell().get_bravais_lattice()));stdout.flush()
        #print("Structure 2: %-8s with fu=%d and %s lattice"%(atoms2.get_chemical_formula(empirical=1),get_fu(atoms2),atoms2.get_cell().get_bravais_lattice()));stdout.flush()

        #if args.prim: 
        #        print('Using primitive cells as input as requested');stdout.flush()
        #        atoms1=find_prim(atoms1); atoms2=find_prim(atoms2) #Whether to use primitive cells of input structures.


        #tolerances for creating slabs and matching two slabs as taken from user
        Lmax = args.max_slab_dimension 	#max length of cell sides
        L = args.initLength # initial length
        Lstep = args.stepLength # step size for the increasing the length in a/b direction 
        T=args.thickness #thickness of slabs
        ptol = args.percentage_tolerance	#percentage tolerances for angle and length matching


        #Common DFT-related parameters, taken from user
        KP=args.kpspacing  #KP spacing
        ecut= args.encut #cutoff energy in eV (convergence prob. with lower cutoffs).
        xc=args.xc
        sigma=args.sigma

        if args.prog == 'castep':
                pp="OTF"
                #pp="00PBE" #pseudopt to use in CASTEP calcs.Def (pp=""): OTF
                dirr="./CASTEP-tmp"

                try:
                        calc = ase.calculators.castep.Castep()
                        calc.merge_param(args.castep_param)
                        ecut=calc.param.cut_off_energy
                        #xc=calc.param.xc_functional
                        #KP=calc.cell.kpoints_mp_spacing #calc.cell is not defined
                        KP=float(grep('kpoints_mp_spacing',args.castep_cell).split()[-1])
                except Exception as err: print("Can't read the param file:", err);stdout.flush()
        elif args.prog == 'vasp':
                dirr="./VASP-tmp" 
                """
                calc=Vasp()
                calc.read_incar(filename='INCAR')
                ecut=calc.encut
                #KP=calc.get_property('kspacing') #does not work, need to read it fro INCAR using grep
                """
        elif args.prog=='ase':
               dirr='./ASE-tmp'


        #Delete calculation files from previous DFT run.
        #system("rm -f %s/*"%dirr)
        #system("rm -f bulk*_opted.* interface_opted.* interface.*  interface-pre_opt.*  slab1.*  slab1_aligned.*  slab1_aligned_opted.*  slab2.cell  slab2_aligned.cell  slab2_aligned_opted.* interface-orig-slabs.* *.traj") #interfaces.out
        print('Cleaning previous output structures in OUTPUT...');stdout.flush()
        system("rm -rf OUTPUT;mkdir -p OUTPUT")
        


        #Time keeping
        initT=time.time()

        ########################
        # Do bulk calculations #
        ########################
        #Check if data from a previous run is available (not to repeat the same calcs for bulk).
        fn1="%s/bulk1"%dirr; fn2="%s/bulk2"%dirr
        x1=None;x2=None
        if args.prog=='castep' and os.path.exists(fn1+".castep"):
                print ("%s was located, reading data..."%fn1);stdout.flush()
                x1=parseCASTEP_ASE(fn1+".geom",atoms1)
                if x1==None:
                        x1=parseCASTEP_ASE(fn1+".castep",atoms1)

        if args.prog=='vasp' and os.path.exists("bulk1/OUTCAR"):
                print ("%s was located, reading data..."%fn1);stdout.flush()
                atoms1=ase.io.read("bulk1/OUTCAR",index=-1)
                x1=[atoms1.get_potential_energy(),atoms1]

        if x1==None:  # whether to compute bulk energies/structures
                print("Computing bulk 1 energy.");stdout.flush()
                atoms1.name='bulk1'
                if args.prog=='castep':x1=call_castep(atoms1,typ="geom",dipolCorr='None',name='bulk1',ENCUT=ecut,PP=pp,KPspacing=KP,hubU=hubU) #normally use K-point spacing.
                elif args.prog=='vasp':x1=call_vasp(atoms1,typ="geom",dipolCorr='None',name='bulk1',ENCUT=ecut,KPspacing=KP,hubU=hubU,slowConv=0,gamma=1,xc=xc,magmom=args.magmoms,sigma=sigma,exe=exe)
                elif args.prog=='ase':x1=call_ase(atoms1,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=True)
                
        if args.prog=='castep' and os.path.exists(fn2+".castep"):
                print ("bulk1 was located, reading data...");stdout.flush()
                x2=parseCASTEP_ASE(fn2+".geom",atoms2)
                if x2==None:
                        x2=parseCASTEP_ASE(fn2+".castep",atoms2)

        if args.prog=='vasp'and os.path.exists("bulk2/OUTCAR"):
                print ("bulk2 was located, reading data...");stdout.flush()
                atoms2=ase.io.read("bulk2/OUTCAR",index=-1)
                x2=[atoms2.get_potential_energy(),atoms2]

        if x2==None:  
                print("Computing bulk 2 energy.");stdout.flush()
                atoms2.name='bulk2'
                if args.prog=='castep':x2=call_castep(atoms2,typ="geom",dipolCorr='None',name='bulk2',ENCUT=ecut,PP=pp,KPspacing=KP,hubU=hubU) #normally use K-point spacing.
                elif args.prog=='vasp':x2=call_vasp(atoms2,typ="geom",dipolCorr='None',name='bulk2',ENCUT=ecut,KPspacing=KP,hubU=hubU,slowConv=0,gamma=1,xc=args.xc,magmom=args.magmoms,sigma=sigma,exe=exe) #normally use K-point spacing.
                elif args.prog=='ase':x2=call_ase(atoms2,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=True)


        atoms1=x1[-1]
        Ebulk1=x1[0]

        fu1=get_fu(atoms1)
        #sa1=surf_area(atoms1) #???? Does it make sense here, it should be more suitable for surface slab??
        #Ebulk1 /= fu1 #Get the bulk energy per formula unit.
        Ebulk1 /= len(atoms1) #Get the bulk energy per atom.

        atoms2=x2[-1]
        Ebulk2=x2[0]

        fu2=get_fu(atoms2)
        #sa2=surf_area(atoms2)
        #Ebulk2 /= fu2
        Ebulk2 /= len(atoms2) #Get the bulk energy per atom.

        #Clean up to save mem
        del x1,x2

        if args.prog=="castep":ase.io.write("OUTPUT/bulk1_opted.cell",atoms1.repeat((1,1,1)),format='castep-cell');  ase.io.write("OUTPUT/bulk2_opted.cell",atoms2.repeat((1,1,1)),format='castep-cell')
        elif args.prog=="vasp":ase.io.write("OUTPUT/bulk1_opted.vasp",atoms1.repeat((1,1,1)),format='vasp',vasp5=1); ase.io.write("OUTPUT/bulk2_opted.vasp",atoms2.repeat((1,1,1)),format='vasp',vasp5=1)
        elif args.prog=="ase": ase.io.write("OUTPUT/bulk1_opted.xyz" ,atoms1.repeat((1,1,1)),format='extxyz');       ase.io.write("OUTPUT/bulk2_opted.xyz" ,atoms2.repeat((1,1,1)),format='extxyz')



        #######################################################################
        #Create the intial slabs with given Miller indices (before alignment).#
        #######################################################################
 

        #TODO: the thickness argument apparently has no effect on the created surface slabs, but only on the interface.
        if args.creps1: creps1=args.creps1
        else:creps1=1
        if args.creps2: creps2=args.creps2
        else:creps2=1

        #This is needed here, in case surface stability and conv layer test are not run.
        #???TODO:try ASE cut function instead of make_slab !! Non need make_slab also uses ase.build.cut.
        #If surface structures are provided as inputs, the structures are read from the files, otherwise they are generated using make_slab.
        if args.surface1:
                surf1=args.surface1
                slab1=ase.io.read(surf1)
                print('Surface 1 is read from %s'%surf1);stdout.flush()
        else:
                slab1 = make_slab(miller1,atoms1,repeat=(1,1,creps1),square=0)
        if args.surface2:
                surf2=args.surface2
                slab2=ase.io.read(surf2)
                print('Surface 2 is read from %s'%surf2);stdout.flush()
        else:
                slab2 = make_slab(miller2,atoms2,repeat=(1,1,creps2),square=0)

        #Do surface stability and layer thickness analysis here
        if args.checkSurfs:

                if 0: #to delete the vacuum padding in the slabs (needed for correct stacking !!) This should not be done for the no-vacuum calculations !!!
                        slab1.center(vacuum=0, axis=2)
                        slab2.center(vacuum=0, axis=2)
                #slab1.name='slab1';slab2.name='slab2'
                

                print('\nChecking the minimum-energy surface for Material 1 (%s), considering the Miller indices: %s'%(atoms1.get_chemical_formula(empirical=1),miller_list));stdout.flush()
                Ws1,slab1,calc,miller1=check_surfaces(atoms1, miller_list,vac=args.vac_slab,Ebulk=Ebulk1,creps=creps1)
                print('\nChecking the minimum-energy surface for Material 2 (%s), considering the Miller indices: %s'%(atoms2.get_chemical_formula(empirical=1),miller_list));stdout.flush()
                Ws2,slab2,calc,miller2=check_surfaces(atoms2, miller_list,vac=args.vac_slab,Ebulk=Ebulk2,creps=creps2)

                slab1.set_calculator(calc);slab2.set_calculator(calc)

        if args.convLayers:#Run layer thickness
                print('\n\nConverging layer thickness...');stdout.flush()
                #slab1 = make_slab(min_mil,atoms1,repeat=(1,1,1),square=False)
                #slab1=min_slab.copy()
                slab1 = make_slab(miller1,atoms1,repeat=(1,1,1),square=False)
                slab2 = make_slab(miller2,atoms2,repeat=(1,1,1),square=False)
                if 1: #to delete the vacuum padding in the slabs (needed for correct stacking !!) This should not be done for the no-vacuum calculations !!!
                        slab1.center(vacuum=0, axis=2)
                        slab2.center(vacuum=0, axis=2)
                #slab1.set_calculator(calc);slab2.set_calculator(calc)
                #slab1.name='slab1';slab2.name='slab2'
                slab1.name="%s_%d%d%d"%(slab1.get_chemical_formula(empirical=1),miller1[0],miller1[1],miller1[2])
                slab2.name="%s_%d%d%d"%(slab2.get_chemical_formula(empirical=1),miller2[0],miller2[1],miller2[2])
                print ('Running Material 1 (%s)'%(slab1.get_chemical_formula(empirical=1)))
                nl1,Eslab1,slab1=conv_layers(slab1,vac=args.vac_slab,lay_min=args.lay_min-1,lay_max=args.lay_max-1)#,ifPrim=1)  #Vacuum layer is added automatically within the function.
                #nl1,Eslab1,slab1=conv_layers(miller1,atoms1,vac=args.vac_slab,lay_min=args.lay_min-1,lay_max=args.lay_max-1)#,ifPrim=1)  #Vacuum layer is added automatically within the function.
                print ('\nRunning Material 2 (%s)'%(slab2.get_chemical_formula(empirical=1)))
                nl2,Eslab2,slab2=conv_layers(slab2,vac=args.vac_slab,lay_min=args.lay_min-1,lay_max=args.lay_max-1)#,ifPrim=1)  #Vacuum layer is added automatically within the function.
                #nl2,Eslab2,slab2=conv_layers(miller2,atoms2,vac=args.vac_slab,lay_min=args.lay_min-1,lay_max=args.lay_max-1)#,ifPrim=1)  #Vacuum layer is added automatically within the function.

                #TODO: complete this!!!
                print('Minimum-energy thickness (creps):');stdout.flush()
                print('\tfor slab1  is %d with E=%.2f eV'%(nl1,Eslab1));stdout.flush()
                print('\tfor slab2  is %d with E=%.2f eV'%(nl2,Eslab2));stdout.flush()
                creps1=nl1
                creps2=nl2

        surf_list_slab1 = []
        surf_list_slab2 = []
        if args.symSlabs:
                # if symmSurf argument is True, obtain list of symmetric syrfaces for given Miller indices
                # check whether slab1 is already symmetric
                rotational_symmetry = get_symmetry(slab1, symprec=1e-5)["rotations"]
                z_rotational_operations = np.stack(rotational_symmetry)
                
                el_bulk1 = atoms1.get_atomic_numbers()
                if all(el == el_bulk1[0] for el in el_bulk1):
                        print('Bulk1 is already symmetric')
                        surf_list_slab1.append(slab1)
                else:
                        # if not symmetric, find symmetric surface cuts
                        print('Bulk1 is being symmetrized')
                        surf_list_slab1 = permute_surface(atoms1, miller1, T, vacuum=args.vac_slab)
                        if len(surf_list_slab1) == 0:
                                surf_list_slab1.append(slab1)
                # check whether slab2 is already symmetric
                el_bulk2 = atoms2.get_atomic_numbers()
                if all(el == el_bulk2[0] for el in el_bulk2):
                        print('Bulk2 is already symmetric')
                        surf_list_slab2.append(slab2)
                else:
                        # if not symmetric, find symmetric surface cuts
                        print('Bulk2 is being symmetrized')
                        surf_list_slab2 = permute_surface(atoms2, miller2, T, vacuum=args.vac_slab)
                        if len(surf_list_slab2) == 0:
                                surf_list_slab2.append(slab2) 
        else:
                surf_list_slab1.append(slab1)
                surf_list_slab2.append(slab2)

        for i, slab1 in enumerate(surf_list_slab1):
                for j, slab2 in enumerate(surf_list_slab2):
        #!!! Need to re-activate this!
                        if args.symSlabs:
                                typ = "sp"
                                opt = False
                        else:
                                typ = "geom"
                                opt = True
                        print(typ, opt)
                        if 0: #to calculate energies/structures of the initial slabs (before alignment). NOT NEEDED
                                print("Pre-optimizing the initial slabs (before alignment).");stdout.flush()
                                if 1: #to add vacuum to the slabs (needed) 
                                        slab1.center(vacuum=args.vac_slab, axis=2)
                                        slab2.center(vacuum=args.vac_slab, axis=2)

                                # for vasp calculations, checks whether folder slab1-pre already exists and takes the output from there
                                if args.prog=='vasp' and os.path.exists("slab1-pre/OUTCAR"):
                                        print ("slab1-pre was located, reading data...");stdout.flush()
                                        slab1=ase.io.read("slab1-pre/OUTCAR",index=-1)
                                        x=[slab1.get_potential_energy(),slab1]
                                # otherwise, calculates does a geometry optimization of slab1
                                else:
                                        slab1.name='slab1_cut%s'%i
                                        if args.prog=='castep':x=call_castep(slab1,typ=typ,dipolCorr='sc',name='slab1_cut%s-pre'%i,ENCUT=ecut,KPgrid=KPgrid,PP=pp,FixCell=0)
                                        elif args.prog=='vasp': 
                                                if args.pas:pas='bot' 
                                                else: pas=None
                                                x=call_vasp(slab1,typ=typ,dipolCorr='sc',name='slab1_cut%s-pre'%i,ENCUT=ecut,KPspacing=KP,KPgrid=KPgrid,FixCell=0,FixVol=0,xc=xc,passivate=pas,magmom=args.magmoms,sigma=sigma,exe=exe)
                                        elif args.prog=='ase':x=call_ase(slab1,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=opt)
                                slab1=x[-1]
                                Eslab1=x[0]

                                # for vasp calculations, checks whether folder slab2-pre already exists and takes the output from there 
                                if args.prog=='vasp' and os.path.exists("slab2-pre/OUTCAR"):
                                        print ("slab2-pre was located, reading data...");stdout.flush()
                                        slab2=ase.io.read("slab2-pre/OUTCAR",index=-1)
                                        x=[slab2.get_potential_energy(),slab2]
                                # otherwise, calculates does a geometry optimization of slab2
                                else:
                                        slab2.name='slab2_cut%s'%j
                                        if args.prog=='castep':x=call_castep(slab2,typ=typ,dipolCorr='sc',name='slab2_cut%s-pre'%j,ENCUT=ecut,KPgrid=KPgrid,PP=pp,FixCell=0)
                                        elif args.prog=='vasp':
                                                if args.pas:pas='top'  
                                                else: pas=None 
                                                x=call_vasp(slab2,typ=typ,dipolCorr='sc',name='slab2_cut%s-pre'%j,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,FixCell=0,FixVol=0,xc=xc,passivate=pas,magmom=args.magmoms,sigma=sigma,exe=exe)
                                        elif args.prog=='ase':x=call_ase(slab2,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=opt)
                                slab2=x[-1]
                                Eslab2=x[0]



                        #Not originally here. (Helps increase the overlap between surfaces. i.e. lower lattice misfit).
                        niggli=0
                        if niggli: niggli_reduce(slab1);niggli_reduce(slab2)

                        if args.prim: slab1=find_prim(slab1);slab2=find_prim(slab2) #does not work.

                        if 1: #to add vacuum to the slabs (for demonstration)
                                slab1_vac=slab1
                                slab2_vac=slab2
                                slab1_vac.center(vacuum=args.vac_slab, axis=2) #TODO: add vacuum
                                slab2_vac.center(vacuum=args.vac_slab, axis=2)

                        if args.prog=="castep":ase.io.write("OUTPUT/slab1_pre-align_cut%s.cell"%i,slab1.repeat((1,1,1)),format='castep-cell');  ase.io.write("OUTPUT/slab2_pre-align_cut%s.cell"%j,slab2.repeat((1,1,1)),format='castep-cell')
                        elif args.prog=="vasp":ase.io.write("OUTPUT/slab1_pre-align_cut%s.vasp"%i,slab1.repeat((1,1,1)),format='vasp',vasp5=1);  ase.io.write("OUTPUT/slab2_pre-align_cut%s.vasp"%j,slab2.repeat((1,1,1)),format='vasp',vasp5=1)
                        elif args.prog=="ase":ase.io.write("OUTPUT/slab1_pre-align_cut%s.xyz"%i,slab1.repeat((1,1,1)),format='extxyz');  ase.io.write("OUTPUT/slab2_pre-align_cut%s.xyz"%j,slab2.repeat((1,1,1)),format='extxyz')


                        print("\nMisfit (mu) of slabs 1 and 2 (before alignment): %.2f%%"%(misfit(slab1,slab2)*100));stdout.flush()#,ifPlot=1)
                        print
                        #exit()


        ######################
        # Alignment of Slabs #
        ######################
                        print("\nAligning the two slabs...");stdout.flush()
                        slab1,slab2=slab_aligner(slab1,slab2,L,Lmax,Lstep,ptol,T,atoms1,atoms2)
                        print("\nMisfit (mu) of slabs 1 and 2 (after alignment): %.2f%%"%(misfit(slab1,slab2)*100));stdout.flush()#,ifPlot=1)

                        if args.view: view(slab1);view(slab2)
                        if args.flip2: #Flip the slab2 (top slab) upside-down.still the cell angles cannot be preserved !! using -h,k,-l could solve this. Or can be moved to after optimizations of the slabs (pasivation settings should be changed, from top to bottom)
                                #This does not work!! try the new flip_cell fnc isntead
                                slab2.rotate(a=180,v='x',rotate_cell=0,center = (0, 0, 0)) #burda yapinca stack calismiyor.
                                slab2.center()
                                if args.view: view(slab2)
                                #a,b,c,alpha,beta,gamma=slab2.get_cell_lengths_and_angles()
                                #print (a,b,c,alpha,beta,gamma)
                                #slab2.rotate(a=alpha/2,v='z',rotate_cell=0,center = (0, 0, 0)) #burda yapinca stack calismiyor.
                                #slab2.center()

                                #slab2.center(vacuum=args.vac,axis=2)
                                if args.view:view(slab2)

                        if args.prog=="castep":ase.io.write("OUTPUT/slab1_aligned_cut%s.cell"%i,slab1.repeat((1,1,1)),format='castep-cell');  ase.io.write("OUTPUT/slab2_aligned_cut%s.cell"%j,slab2.repeat((1,1,1)),format='castep-cell')
                        elif args.prog=="vasp":ase.io.write("OUTPUT/slab1_aligned_cut%s.vasp"%i,slab1.repeat((1,1,1)),format='vasp',vasp5=1);  ase.io.write("OUTPUT/slab2_aligned_cut%s.vasp"%j,slab2.repeat((1,1,1)),format='vasp',vasp5=1)
                        elif args.prog=="ase":ase.io.write("OUTPUT/slab1_aligned_cut%s.xyz"%i,slab1.repeat((1,1,1)),format='extxyz');  ase.io.write("OUTPUT/slab2_cut%s_aligned.xyz"%j,slab2.repeat((1,1,1)),format='extxyz')

                        if 1:
                        #Interface before optimizing the individual slabs.
                                if 1: #to delete the vacuum padding in the slabs (needed for correct stacking !!) This should not be done for the no-vacuum calculations !!!
                                        slab1.center(vacuum=0, axis=2)
                                        slab2.center(vacuum=0, axis=2)
                                interface=ase.build.stack(slab1, slab2, axis=2, maxstrain=None, distance=args.sep,cell=None,reorder=1)  #using 0 distance btw slabs gives CASTEP error.
                                interface.center(vacuum=args.vac_int, axis=2) #Vacuum on both sides. For dipole corrections at least 8A vacuum is needed.
                                if args.view:view(interface)

                                
                                if 0 and (interface.get_cell()[2][2]<0): #VASP can't work with upside-down cells (with negative z-coordiantes)
                                        print ("Cell extends towards -z direction, VASP cannot work with -ve z coords, rotating about x-axis by 180 degree to fix it");stdout.flush()
                                        #interface.rotate(a=180,v='x',rotate_cell=1,center = (0, 0, 0))
                                        interface=flip_cell(interface)
                                        if args.view:view(interface)
                                

                                if args.prog=="castep": ase.io.write("OUTPUT/interface-orig-slabs_cuts_%s_%s.cell"%(i,j),interface,format='castep-cell')
                                elif args.prog=="vasp": ase.io.write("OUTPUT/interface-orig-slabs_cuts_%s_%s.vasp"%(i,j),interface,format='vasp',vasp5=1)
                                elif args.prog=="ase": ase.io.write("OUTPUT/interface-orig-slabs_cuts_%s_%s.xyz"%(i,j),interface,format='extxyz')

                        if 1: #to calculate energies/structures of the actual slabs (after alignment).
                                print("\nOptimizing the slabs 1 and 2 (after alignment).");stdout.flush()

                                if 1: #to add vacuum to the slabs (needed)
                                        slab1.center(vacuum=args.vac_slab, axis=2)
                                        slab2.center(vacuum=args.vac_slab, axis=2)

                                if args.prog=='vasp' and os.path.exists("slab1/OUTCAR"):
                                        print ("slab1-aligned was located, reading data...");stdout.flush()
                                        slab1=ase.io.read("slab1-aligned/OUTCAR",index=-1)
                                        x=[slab1.get_potential_energy(),slab1]
                                
                                else:
                                        slab1.name='slab1_cut%s'%i
                                        if args.prog=='castep': x=call_castep(slab1,typ=typ,dipolCorr='sc',name='slab1_cut%s-aligned'%i,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,PP=pp,FixCell=1,hubU=hubU)#,FixList=[1,2]) #Optimizer TPSD or FIRE can be used for fixed cell opts.
                                        elif args.prog=='vasp': 
                                                if args.pas:pas='bot' 
                                                else: pas=None
                                                #x=call_vasp(slab1,typ="geom",dipolCorr='sc',name='slab1-aligned',ENCUT=ecut,KPgrid='1 1 1',FixCell=True,hubU=hubU,FixVol=0,xc=xc,passivate=pas,nosymm=1,vac=args.vac,magmom=args.magmoms)#,FixList=[1,2])
                                                x=call_vasp(slab1,typ=typ,dipolCorr='sc',name='slab1_cut%s-aligned'%i,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,FixCell=True,hubU=hubU,FixVol=0,xc=xc,passivate=pas,nosymm=1,vac=args.vac_slab,magmom=args.magmoms,sigma=sigma,exe=exe)#,FixList=[1,2])
                                        elif args.prog=='ase':x=call_ase(slab1,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=opt)

                                slab1=x[-1]
                                Eslab1=x[0]

                                if args.prog=='vasp' and os.path.exists("slab2/OUTCAR"):
                                        print ("slab2-aligned was located, reading data...");stdout.flush()
                                        slab2=ase.io.read("slab2-aligned/OUTCAR",index=-1)
                                        x=[slab2.get_potential_energy(),slab2]
                                
                                else:
                                        slab2.name='slab2_cut%s'%j
                                        if args.prog=='castep':
                                                #x=call_castep(slab2,typ="geom",dipolCorr='sc',name='slab2-aligned',ENCUT=ecut,KPgrid='1 1 1',PP=pp,FixCell=True,hubU=hubU)#,FixList=[1,2])
                                                x=call_castep(slab2,typ=typ,dipolCorr='sc',name='slab2_cut%s-aligned'%j,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,PP=pp,FixCell=1,hubU=hubU)#,FixList=[1,2]) #Optimizer TPSD or FIRE can be used for fixed cell opts.
                                        elif args.prog=='vasp':
                                                if args.pas:pas='top'  
                                                else: pas=None 
                                                #x=call_vasp(slab2,typ="geom",dipolCorr='sc',name='slab2-aligned',ENCUT=ecut,KPgrid='1 1 1',FixCell=True,hubU=hubU,FixVol=0,xc=xc,passivate=pas,nosymm=1,vac=args.vac,magmom=args.magmoms)#,FixList=[1,2])
                                                x=call_vasp(slab2,typ=typ,dipolCorr='sc',name='slab2_cut%s-aligned'%j,ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,FixCell=True,hubU=hubU,FixVol=0,xc=xc,passivate=pas,nosymm=1,vac=args.vac_slab,magmom=args.magmoms,sigma=sigma,exe=exe)#,FixList=[1,2])
                                        elif args.prog=='ase':x=call_ase(slab2,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=opt)

                                slab2=x[-1]
                                Eslab2=x[0]

                                
                                #Compute the surfafce energies.
                                if 1: #use energy per atom
                                        Ws1=(Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
                                        Ws2=(Eslab2-len(slab2)*Ebulk2)/2/surf_area(slab2)/0.01 #A2 to nm2
                                else: #use energy per fu
                                        Ws1=(Eslab1-fu1*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
                                        Ws2=(Eslab2-fu2*Ebulk2)/2/surf_area(slab2)/0.01 #A2 to nm2

                                str1='\nCalculating the W_surf (surface formation energies / surface area) of the optimised aligned slabs...\n'
                                str1+='%s: %.3f eV/atom\n' % ('Ebulk_1',Ebulk1)
                                str1+='%s: %.3f eV/atom\n' % ('Ebulk_2', Ebulk2)
                                str1+='%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('Wsurf_1', slab1.get_chemical_formula(empirical=1),miller1[0],miller1[1],miller1[2],Ws1,Ws1*cfactor)
                                str1+='%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('Wsurf_2', slab2.get_chemical_formula(empirical=1),miller2[0],miller2[1],miller2[2],Ws2,Ws2*cfactor)

                                print(str1);stdout.flush()
                                outf.writelines(str1)

                        
                        if args.prog=="castep":
                                ase.io.write("OUTPUT/slab1_aligned_opted_cut%s.cell"%i,slab1,format='castep-cell')
                                ase.io.write("OUTPUT/slab2_aligned_opted_cut%s.cell"%j,slab2,format='castep-cell')
                        elif args.prog=='vasp':
                                ase.io.write("OUTPUT/slab1_aligned_opted_cut%s.vasp"%i,slab1,format='vasp',vasp5=1)
                                ase.io.write("OUTPUT/slab2_aligned_opted_cut%s.vasp"%j,slab2,format='vasp',vasp5=1)
                        elif args.prog=='ase':
                                ase.io.write("OUTPUT/slab1_aligned_opted_cut%s.xyz"%i,slab1,format='extxyz')
                                ase.io.write("OUTPUT/slab2_aligned_opted_cut%s.xyz"%j,slab2,format='extxyz')
                
                        if args.view:view(slab1);view(slab2)

                        #if args.vac>2: #to delete the vacuum padding in the slabs (needed for correct stacking !!)
                        if 1: 
                                slab1.center(vacuum=0, axis=2)
                                slab2.center(vacuum=0, axis=2)

                        #Create the interface.
                        interface=ase.build.stack(slab1, slab2, axis=2, maxstrain=None, distance=args.sep,cell=None,reorder=True)  #using 0 distance btw slabs gives CASTEP error.
                        #interface.wrap() #works
                        interface.center(vacuum=args.vac_int, axis=2) #Vacuum on both sides. For dipole corrections at least 8A vacuum is needed.

                        
                        if args.verb: 
                                a,b,c,alpha,beta,gamma=interface.cell.cellpar()
                                print (a,b,c,alpha,beta,gamma);stdout.flush()
                                #a,b,c,alpha,beta,gamma=interface.get_cell_lengths_and_angles() #old deprecated
                        if args.view:view(interface)

                        if (interface.get_cell()[2][2]<0): #VASP can't work with upside-down cells (with negative z-coordiantes)
                                print ("Cell extends towards -z direction, VASP cannot work with -ve z coords, rotating about x-axis by 180 degree to fix it");stdout.flush()
                                #interface.rotate(a=180,v='x',rotate_cell=1,center = (0, 0, 0))
                                interface=flip_cell(interface)
                                if args.view:view(interface)


                        if args.prog=="castep": ase.io.write("OUTPUT/interface-pre_opt_cuts_%s_%s.cell"%(i,j),interface,format='castep-cell')
                        elif args.prog=="vasp": ase.io.write("OUTPUT/interface-pre_opt_cuts_%s_%s.vasp"%(i,j),interface,format='vasp',vasp5=1)
                        elif args.prog=='ase':  ase.io.write("OUTPUT/interface-pre_opt_cuts_%s_%s.xyz"%(i,j),interface,format='extxyz')
                                

                        if args.convSep: #Run optimisation on the interface speration distance
                                print("\n\nFinding the optimal interface (interlayer) separation.");stdout.flush()
                                if 1: #to add vacuum to the slabs (needed) 
                                        slab1.center(vacuum=args.vac_slab, axis=2)
                                        slab2.center(vacuum=args.vac_slab, axis=2)
                                slab1.name='slab1'
                                slab2.name='slab2'

                                if 0: vacc=args.vac_int
                                else: vacc=None #separation/2 will be used as default to have a symmetric seperation at all interfaces.
                                interface,minSep=convSep(slab1,slab2,vac=vacc,sep_init=args.sep_init,sep_final=args.sep_final,sep_step=args.sep_step,view=0,ifPlot=0)
                                args.sep=minSep

                        if args.test_hori: #Optimise the horizontal stacking of the layers by translating slab
                                print("\n\nFinding the optimial interface lateral shift.");stdout.flush()

                                interface,Eint,minxy=checkHorizontal(slab1,slab2,sep=args.sep,steps=args.hor_steps)
                                #print(Eint);stdout.flush()

                        #niggli=1
                        if args.niggli: print ("Niggli reduce the interface...");niggli_reduce(interface);stdout.flush()
                        #if niggli: interface=niggli_reduce(interface)
                        if args.view:view(interface)

                        if 1: #Single point interface energy before optimisation
                                print("\n Single point run for the interface geometry.");stdout.flush()
                                interface.name='interface_cuts_%s_%s'%(i,j)
                                if args.vac_int>4:dipC='SC'
                                else:dipC='None'

                                if args.prog=="castep":
                                        #x=call_castep(interface,typ="sp",dipolCorr='SC',name='interface',ENCUT=ecut,PP=pp,KPspacing=KP,hubU=hubU)
                                        x=call_castep(interface,typ="sp",dipolCorr=dipC,name='interface_cuts_%s_%s-pre'%(i,j),ENCUT=ecut,PP=pp,KPgrid=KPgrid,KPspacing=KP,hubU=hubU) #KPgrid='1 1 1'               
                                elif args.prog=="vasp": x=call_vasp(interface,typ="sp",dipolCorr=dipC,name='interface_cuts_%s_%s-pre'%(i,j),ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,hubU=hubU,FixVol=0,xc=xc,nosymm=1,sigma=sigma,exe=exe) #vac=args.vac
                                elif args.prog=='ase':x=call_ase(interface,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=0)

                                interface=x[-1]
                                Eint=x[0]


                                Wad=(Eslab1+Eslab2-Eint)/surf_area(interface)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf??
                                str1='W_ad (formation energy/area) for the final interface before optimisation: %.2f eV/nm^2\n'%Wad
                                print(str1);stdout.flush()
                                outf.writelines(str1)

                        if 1: #Optimise the final interface geometry
                                print("\nOptimizing the final interface geometry.");stdout.flush()
                                interface.name='interface_cuts_%s_%s'%(i,j)
                                if args.vac_int>4:dipC='SC'
                                else:dipC='None'
                                #CHECK: Should we fix the volume or not?? fix vol works for some cases (YIG) but not for some (Al2O3//Si)
                                if args.prog=="castep": x=call_castep(interface,typ=typ,dipolCorr=dipC,name='interface_cuts_%s_%s'%(i,j),ENCUT=ecut,PP=pp,KPgrid=KPgrid,KPspacing=KP,hubU=hubU)
                                elif args.prog=="vasp": x=call_vasp(interface,typ=typ,dipolCorr=dipC,name='interface_cuts_%s_%s'%(i,j),ENCUT=ecut,KPgrid=KPgrid,KPspacing=KP,hubU=hubU,FixVol=0,xc=xc,nosymm=1,magmom=args.magmoms) 
                                elif args.prog=='ase': x=call_ase(interface,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=opt)

                                interface=x[-1]
                                Eint=x[0]

                                #-ve sign???
                                #Wad=(Eslab1+Eslab2-Eint)/surf_area(interface)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf??
                                Wad=(Eint-Eslab1-Eslab2)/surf_area(interface)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf??
                                #cfactor=6.2415093 #eV / nm^2 to J / m^2
                                str1='W_ad (formation energy/area) for the final optimised interface: %.2f eV/nm^2 = %.2f J/m^2\n'%(Wad,Wad*cfactor)
                                print(str1);stdout.flush()
                                outf.writelines(str1)

                                if args.prog=='castep': ase.io.write("OUTPUT/interface_opted_cuts_%s_%s.cell"%(i,j),interface,format='castep-cell')
                                elif args.prog=='vasp': ase.io.write("OUTPUT/interface_opted_cuts_%s_%s.vasp"%(i,j),interface,format='vasp',vasp5=1)
                                elif args.prog=='ase':  ase.io.write("OUTPUT/interface_opted_cuts_%s_%s.xyz"%(i,j),interface,format='extxyz')

                                ase.io.write("OUTPUT/interface_opted_cuts_%s_%s.cif"%(i,j),interface,format='cif')
                                

        outf.close()

        print ("Elapsed time: %.2f sec."%( time.time()-initT));stdout.flush()

        exit()


##################
# Excluded Parts #
##################



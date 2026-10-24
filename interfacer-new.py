#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import argparse,math, os.path,fractions,time,shutil,re
import matplotlib.pyplot as plt
import ast

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

- Make sure that all interfaces slabs are not flipped randomly (due to bugs in ase.build.stack and cut functions)
- Make sure that the ase. build cut and stack are working correctly. For interlayer sepration in stack does not work at times (two surface slab layers clash at low separation/distance!!)
-ASE surface slab (Cut function) is problematic for Na3OCL (110) and (111) surfaces (O clashes with the upper layer)

- Add horizontal/lateral translation optimisation with scipy/BFGS
- Add support for 3-,4-layer interfaces
- Take args.parser (command-line) argument from an input file (interfacer.inp), args.parser will overwrite the file input.
- Add the finite surface slab energy correction scheme as given in Lepley et al 2015 paper.
- Add support fixing atoms (e.g. middle layers, for a bulk like region in surface slabs.) during geom optimsiations
- Add ONETEP and GPAW calculator support for plane-wave DFT (can be installed using conda install gpaw)
- Add machine-learning potentials (MACE, CHGNet, ...) through ASE (see docs/ROADMAP.md)
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
- horizontal tranlation   (Keeping slabs rigid and using single points), on a full 2D grid
- gamma-only version KPgrid 1 1 1
- VASP Runs for surface analysis in seprate directories (e.g. Li_111), so support restart!!
- No dipole correction when no vacuum in interface.
- Bug fixes (see docs/known_issues.md): -mlist/-kp_gr parsing, W_ad sign, surface tests feeding into the
  slab creation, 2D lateral scan, layer convergence indexing, interface built from the optimised slabs, ...

"""

def grep(key,fname,n=-1):
    #Returns the lines containing the keyword in a given file (all matches if n<0, otherwise the first n), joined by newlines.
        try:
                with open(fname) as f: lines=[ln.rstrip('\n') for ln in f if key in ln]
        except OSError: return ""
        if n>=0: lines=lines[:n]
        return "\n".join(lines)

from subprocess import Popen,PIPE # as popen # check_output
from sys import stdout,version_info
def Popen4(cmd,timeout=None):
    proc=Popen(cmd,shell=True, stdin=PIPE, stdout=PIPE, close_fds=True, universal_newlines=True)
    out,err=proc.communicate(timeout=timeout)
    out2=[x for x in out.split("\n") if x!=""]
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

def parse_miller(s):
        """Parse a Miller index given on the command line, e.g. '111', '1-10', '1,1,0', '(1,1,0)' or '[1 1 0]'."""
        s=str(s).strip().strip('()[]').strip()
        if ',' in s or ' ' in s: vals=[int(x) for x in re.split(r'[,\s]+',s) if x!='']
        else: vals=[int(x) for x in re.findall(r'-?\d',s)] #single-digit indices, e.g. 1-10 -> (1,-1,0)
        if len(vals)!=3: raise argparse.ArgumentTypeError("Cannot parse Miller index '%s' (expected e.g. 111, 1-10 or 1,1,0)"%s)
        return tuple(vals)

def struct_ext():
        #Output file extension and ASE format for the structures, depending on the code used.
        if args.prog=="castep": return "cell",{'format':'castep-cell'}
        elif args.prog=="vasp": return "vasp",{'format':'vasp','vasp5':True}
        else: return "xyz",{'format':'extxyz'}

def write_struct(atoms,base,outdir="OUTPUT"):
        #Write a structure to OUTPUT/<base>.<ext> in the format of the code used (cell/vasp/xyz).
        ext,kw=struct_ext()
        fname="%s/%s.%s"%(outdir,base,ext)
        ase.io.write(fname,atoms,**kw)
        return fname

def run_calc(atoms,typ,name,dipolCorr='None',FixCell=False,FixVol=0,passivate=None,nosymm=0,vac=0.0,KPgrid_=None,ase_opt=None):
        """Run a single point (typ='sp') or geometry optimisation (typ='geom') with the code selected by -prog.
        All calculations use the same common settings (cutoff, k-points, XC, smearing, U, magnetic moments, executable).
        KPgrid_ overrides the k-point grid (e.g. '1 1 1'); otherwise the k-point spacing is used.
        ase_opt overrides whether ASE runs an optimisation (default: typ=='geom').
        Returns (E, atoms)."""
        if args.prog=='castep':
                if KPgrid_: kpkw={'KPgrid':KPgrid_}
                else: kpkw={'KPspacing':KP}
                return call_castep(atoms,typ=typ,dipolCorr=dipolCorr,name=name,ENCUT=ecut,PP=pp,hubU=hubU,FixCell=FixCell,ifDryRun=args.dry,**kpkw)
        elif args.prog=='vasp':
                return call_vasp(atoms,typ=typ,dipolCorr=dipolCorr,name=name,ENCUT=ecut,KPgrid=KPgrid_,KPspacing=KP,hubU=hubU,FixCell=FixCell,FixVol=FixVol,
                                 xc=xc,passivate=passivate,nosymm=nosymm,vac=vac,magmom=args.magmoms,sigma=sigma,exe=exe,dry=args.dry,slowConv=0,gamma=1)
        elif args.prog=='ase':
                opt=(typ.lower()=='geom') if ase_opt is None else ase_opt
                return call_ase(atoms,ctype=args.ase_pot,fmax=args.ase_fmax,steps=args.ase_gsteps,opt=opt)
        raise ValueError("Unknown code: %s"%args.prog)


def check_surfaces(atoms, miller_list,vac,Ebulk,creps):
        min_Ws=1e8; min_mil=();min_slab=None
        calc=atoms.calc
        str1=''
        for i,mil in enumerate(miller_list):
                print ('Doing the (%d,%d,%d) surface...'%(mil[0],mil[1],mil[2]));stdout.flush()
                slab1 = make_slab(mil,atoms,repeat=(1,1,creps),square=False)
                slab1.center(vacuum=vac, axis=2)

                name="%s_%d%d%d"%(slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])
                slab1.name=name
                #???Which one better: Single point or geometry opt???
                tmpdir={'castep':'CASTEP-tmp','vasp':'VASP-tmp'}.get(args.prog,'')
                x=run_calc(slab1,"geom",os.path.join(tmpdir,name) if tmpdir else name,dipolCorr='sc',FixCell=True,KPgrid_=KPgrid,ase_opt=False) #fixed cell: keeps the vacuum padding

                slab1=x[-1]
                Eslab1=x[0]
                Ws1=(Eslab1-len(slab1)*Ebulk)/2/surf_area(slab1)/0.01 #A2 to nm2
                str1+='%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('W_surf ', slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2],Ws1,Ws1*cfactor)
                if i==0 or Ws1<min_Ws: min_Ws=Ws1; min_mil=mil;min_slab=slab1.copy()

                write_struct(slab1,name)

        print(str1);stdout.flush()
        #The surface with the lowest (positive) surface energy is the most stable one.
        Ws1=min_Ws
        print('check_surfaces: Minimum-energy surface for material %s is %s with %d atoms and W_surf=%.2f eV/nm^2'%(min_slab.get_chemical_formula(empirical=1),min_mil,min_slab.get_global_number_of_atoms(),Ws1));stdout.flush()

        min_slab.calc=calc
        return Ws1, min_slab,calc,min_mil

def conv_layers(atoms,vac=2.0,ifPlot=0,ifPrim=False, Etol=1e-2,view=0,lay_min=0,lay_max=5):#layer convergence test (Input atoms with a calc object).
        #Converges the energy per atom w.r.t. the number of repetitions (layers) of the given slab in the c direction.
        #Returns the number of layers, the total energy and the final slab.

        #Initial values
        E=[]
        name=getattr(atoms,'name','slab')
        atoms_orig=atoms.copy()

        #find the primitive cells to reduce comp. efforts.
        if ifPrim: atoms_orig=find_prim(atoms_orig)

        i=lay_min;layers=[]
        flag=0 #converged?
        while  i<=lay_max:
                layers.append(1+1*i) #increase 1 layer at a time
                atoms=atoms_orig.copy()
                atoms=atoms.repeat((1,1,layers[-1]))
                atoms.center(vacuum=vac, axis=2)
                atoms.name='%s_%d-layer'%(name,layers[-1])
                nAt=atoms.get_global_number_of_atoms()
                tmpdir={'castep':'CASTEP-tmp/','vasp':'VASP-tmp/'}.get(args.prog,'')
                x=run_calc(atoms,"sp",'%s%s_%d-layer'%(tmpdir,name,layers[-1]),dipolCorr='sc')
                atoms=x[-1]
                Ecurr=x[0]
                E.append(Ecurr/nAt)
                print("Iter. #%d, #layers: %d, #atoms: %d "%(len(E),layers[-1],nAt));stdout.flush()
                if len(E)>1: print("E_total: %.5f; deltaE: %.3e eV/atom; target: %.3e eV."%(E[-1],abs(E[-1]-E[-2]),Etol));stdout.flush()
                else: print("E_total: %.5f eV/atom"%(E[-1]));stdout.flush()
                if view: ase_view(atoms)

                #Writing layer steps
                app=(len(E)>1) #append
                if args.prog=="castep":ase.io.write("OUTPUT/%s-layer.cell"%(name),atoms,format='castep-cell',append=app);
                elif args.prog=="vasp":ase.io.write("OUTPUT/%s-layer.vasp"%(name),atoms,format='vasp-xdatcar',append=app);
                elif args.prog=="ase": ase.io.write("OUTPUT/%s-layer.xyz" %(name),atoms,format='extxyz',append=app);
                if len(E)>1 and abs(E[-1]-E[-2]) <= Etol:
                       flag=1
                       break
                i += 1

        if flag:
               print("conv_layers: E/atom converged to %.2e eV with %d layers."%(Etol,layers[-1]));stdout.flush()
        else:
               print('conv_layers: Layer thickness not converged to %.2e eV within %d layers '%(Etol,layers[-1]));stdout.flush()

        if ifPlot: #Do plotting of E/atom vs. #layers
                plt.figure()
                plt.plot(layers,E, 'ko-')
                plt.xlabel('Number of layers')
                plt.ylabel('Energy per atom (eV/atom)')
                plt.savefig('OUTPUT/conv_layers_%s.png'%name)
                plt.close()

        return layers[-1],E[-1]*nAt,atoms

def conv_layers_new(mil,atoms,vac=2.0,ifPlot=1,ifPrim=False, Etol=1e-2,view=0,lay_min=0,lay_max=5):#layer convergence test (Input atoms with a calc object).
        #Same as conv_layers, but the slabs are re-created from the bulk (atoms) with make_slab for each number of layers.
        print ("Convergence of E/atom vs. #layers")

        #Initial values
        E=[]
        name="%s_%d%d%d"%(atoms.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])

        #find the primitive cells to reduce comp. efforts.
        if ifPrim: atoms=find_prim(atoms)

        i=lay_min;layers=[]
        flag=0 #converged?
        while  i<=lay_max:
                layers.append(1+1*i) #increase 1 layer at a time
                slab = make_slab(mil,atoms,repeat=(1,1,layers[-1]),square=False)
                slab.center(vacuum=vac, axis=2)
                slab.name='%s_%d-layer'%(name,layers[-1])
                nAt=slab.get_global_number_of_atoms()

                tmpdir={'castep':'CASTEP-tmp/','vasp':'VASP-tmp/'}.get(args.prog,'')
                x=run_calc(slab,"sp",'%s%s_%d-layer'%(tmpdir,name,layers[-1]),dipolCorr='sc')
                slab=x[-1]
                Ecurr=x[0]
                E.append(Ecurr/nAt)
                print("Iter. #%d, #layers: %d, #atoms: %d "%(len(E),layers[-1],nAt));stdout.flush()
                if len(E)>1: print("E_total: %.5f; deltaE: %.3e eV/atom; target: %.3e eV."%(E[-1],abs(E[-1]-E[-2]),Etol));stdout.flush()
                if view: ase_view(slab)

                #Writing layer steps
                app=(len(E)>1) #append
                if args.prog=="castep":ase.io.write("OUTPUT/%s-layer.cell"%(name),slab,format='castep-cell',append=app);
                elif args.prog=="vasp":ase.io.write("OUTPUT/%s-layer.vasp"%(name),slab,format='vasp-xdatcar',append=app);
                elif args.prog=="ase": ase.io.write("OUTPUT/%s-layer.xyz" %(name),slab,format='extxyz',append=app);

                if len(E)>1 and abs(E[-1]-E[-2]) <= Etol:
                       flag=1
                       break
                i += 1

        if flag:
               print("conv_layers: E/atom converged to %.2e eV with %d layers."%(Etol,layers[-1]));stdout.flush()
        else:
               print('conv_layers: Layer thickness not converged to %.2e eV within %d layers '%(Etol,layers[-1]));stdout.flush()

        if ifPlot: #Do plotting of E/atom vs. #layers
                plt.figure()
                plt.plot(layers,E, 'ko-')
                plt.xlabel('Number of layers')
                plt.ylabel('Energy per atom (eV/atom)')
                plt.savefig('OUTPUT/conv_layers_%s.png'%name)
                plt.close()

        return layers[-1],E[-1]*nAt,slab


def make_interface(slab1,slab2,sep,vac,reorder=True):
        #Stack two slabs (slab1 at the bottom) with the given separation and vacuum padding (on both sides).
        #sep is the gap (in A) along the surface normal between the topmost atom of slab1 and the lowest atom of slab2.
        #NOTE: ase.build.stack(distance=sep) is not used any more: there 'distance' is the minimum interatomic
        #distance, obtained by also translating the slabs laterally and scaling the cells, which gave clashing
        #layers at small separations and interfered with the lateral translation test (-thor).
        #If reorder is False, the atoms of slab1 come first (needed by checkHorizontal). The input slabs are not modified.
        s1=slab1.copy(); s2=slab2.copy()
        s1.center(vacuum=0, axis=2) #to delete the vacuum padding in the slabs (needed for correct stacking !!)
        s2.center(vacuum=0, axis=2)
        interface=ase.build.stack(s1, s2, axis=2, maxstrain=None, distance=None,cell=None,reorder=False) #slabs touching (zero gap)
        normal=np.cross(interface.cell[0],interface.cell[1]); normal/=np.linalg.norm(normal)
        c=interface.cell[2]
        shift=c*sep/abs(np.dot(c,normal)) #vector along c that increases the height along the normal by sep
        pos=interface.get_positions(); pos[len(s1):]+=shift
        interface.set_positions(pos)
        interface.set_cell([interface.cell[0],interface.cell[1],c+shift],scale_atoms=False)
        if reorder: interface=ase.build.sort(interface)
        interface.center(vacuum=vac, axis=2) #Vacuum on both sides. For dipole corrections at least 8A vacuum is needed.
        if (interface.get_cell()[2][2]<0): #VASP can't work with upside-down cells (with negative z-coordiantes)
                print ("Cell extends towards -z direction, VASP cannot work with -ve z coords, fixing it...");stdout.flush()
                interface=flip_cell(interface)
        return interface

def convSep(slab1,slab2,vac=None,sep_init=1.5,sep_final=4.0,sep_step=0.5,view=0,ifPlot=1, Etol=1e-2): #Converge the sepration bet ween the two surface slabs (keeping them rigid)
        # TODO: find a way to check whether two slabs are clashing (i.e. <2.0A),
        # ase.build.stack function can give interfaces (with distance option), clashing surface slabs!!!

        E=[]
        minSep=sep_init; Emin=1e6; minInt=None
        seps=np.arange(sep_init,sep_final+0.5*sep_step,sep_step) #sep_final is included
        for i,sep in enumerate(seps):
                print ('Doing the %.2f A separation between two layers'%sep);stdout.flush()
                if vac:vacc=vac
                else:vacc=sep/2
                interface=make_interface(slab1,slab2,sep,vacc,reorder=True)
                a,b,c,alpha,beta,gamma=interface.cell.cellpar()
                interface.name='interface-sep_%.2fA'%sep

                nAt=interface.get_global_number_of_atoms()
                tmpdir={'castep':'CASTEP-tmp/','vasp':'VASP-tmp/'}.get(args.prog,'')
                x=run_calc(interface,"sp",'%sinterface-sep_%.2fA'%(tmpdir,sep),dipolCorr='None',KPgrid_=KPgrid)
                interface=x[-1]
                Ecurr=x[0]
                E.append(Ecurr/nAt)
                if i==0 or  E[-1]<Emin: minSep=sep;Emin=E[-1];minInt=interface.copy()
                print("Iter. #%d, Seperation: %.2f A, E / atom : %.5f eV, z-thickness: %.2f A \n"%(i+1,sep,E[-1],c));stdout.flush()
                if view: ase_view(interface)

                #Save structures
                write_struct(interface,"int-sep_%.2fA"%sep)

        print('\nconvSep: Minimum-energy separation is %.2f A with E= %.5f eV/atom'%(minSep,Emin));stdout.flush()

        if minInt: interface=minInt.copy()

        if ifPlot: #Do plotting of E/atom vs. separation
                plt.figure()
                plt.plot(seps,E, 'ko-')
                plt.xlabel('Interlayer Separation [A]')
                plt.ylabel('Energy per atom [eV/atom]')
                plt.savefig('OUTPUT/conv_sep.png')
                plt.close()

        if view:ase_view(interface)

        return interface,minSep

def checkHorizontal(slab1,slab2,sep,vac=None,steps=[0.25,0.25],ifPlot=0): #This checks the optimum horizontal stacking of two slabs through translating the bottom slab
        #The bottom slab (slab1) is translated on a 2D grid of fractional shifts (x,y) in [0,1) along a and b,
        #keeping both slabs rigid, and the single-point energies are compared.

        if vac:vacc=vac
        else:vacc=sep/2
        ln1=len(slab1); ln2=len(slab2)

        a,b,c=slab1.cell.lengths()
        print('Cell lengths: %.2f A %.2f A %.2f A'%(a,b,c));stdout.flush()

        int_orig=make_interface(slab1,slab2,sep,vacc,reorder=False)

        xrng=np.arange(0.0,1.0-1e-8,steps[0])
        yrng=np.arange(0.0,1.0-1e-8,steps[1])
        Egrid=np.zeros((len(xrng),len(yrng)))
        minE=1e6;minInt=None;minxy=[0.,0.]
        first=True
        for ix,x in enumerate(xrng):      #work in fractional coords. Convert to Cartesian for translation
          for iy,y in enumerate(yrng):    #(previously only the diagonal x=y was scanned)
                interface=int_orig.copy()

                cart_disp=interface.cell.cartesian_positions([x,y,0])
                disp=[cart_disp for j in range(ln1)]
                disp.extend([[0., 0., 0.] for j in range(ln2)])
                interface.translate(disp)
                interface.wrap()

                nAt=interface.get_global_number_of_atoms()
                interface.name='htrans_x%.2f-y%.2f'%(x,y)
                tmpdir={'castep':'CASTEP-tmp/','vasp':'VASP-tmp/'}.get(args.prog,'')
                xx=run_calc(interface,"sp",'%shtrans_x%.2f-y%.2f'%(tmpdir,x,y),dipolCorr='None',KPgrid_=KPgrid)
                interface=xx[-1]
                Ecurr=xx[0]
                Egrid[ix,iy]=Ecurr/nAt

                if first or  Egrid[ix,iy]<minE: minE=Egrid[ix,iy];minInt=interface.copy();minxy=[x,y];first=False
                print("Displacement: x= %.2f, y= %.2f (x= %.2f A , y= %.2f A); E/atom: %.5f eV, "%(x,y,cart_disp[0],cart_disp[1],Egrid[ix,iy]));stdout.flush()

                #Save structures
                write_struct(interface,"htrans_x%.2f-y%.2f"%(x,y))

        print('Minimum-energy horizontal translation: x= %.2f, y= %.2f with E= %.5f eV/atom'%(minxy[0],minxy[1],minE));stdout.flush()

        if ifPlot: #2D colour map of E/atom vs. x-y translation
                plt.figure()
                plt.imshow(Egrid.T,origin='lower',extent=[xrng[0],xrng[-1]+steps[0],yrng[0],yrng[-1]+steps[1]],aspect='auto')
                plt.colorbar(label='Energy per atom [eV/atom]')
                plt.xlabel('fractional shift along a')
                plt.ylabel('fractional shift along b')
                plt.savefig('OUTPUT/conv_htrans.png')
                plt.close()

        return minInt,minE,minxy

def checkHorizontal2(slab1,slab2,sep,vac=None): #This checks the optimum horizontal stacking of two slabs through translating the slab2
        #TODO: complete this!! This is for doing scipy opt with respect to horizantal/vertical translations, getting SP energies.
        raise NotImplementedError("checkHorizontal2: Not implemented yet, use checkHorizontal (-thor) instead.")


##################################################
#                  MAIN METHOD                   #
##################################################
try:
        vasppp = os.environ['VASP_PP_PATH']
        ppdirs=os.listdir(vasppp)
        ppdirs.sort()
except Exception: ppdirs=None

#eV / nm^2 to J / m^2: 1 eV/nm^2 = 1.602e-19 J / 1e-18 m^2 = 0.1602 J/m^2.
#(The previous value, 6.2415093, is the inverse factor, i.e. J/m^2 to eV/nm^2.)
cfactor=0.160217663

if __name__== '__main__':
        #read in arguments
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
        parser.add_argument("-prog", "--prog", choices=['castep','vasp','ase'], type=str.lower, required=1, help="Code to be used: CASTEP, VASP or ASE (LJ/Morse/EMT)")
        parser.add_argument("-xc", "--xc", default='PBE', type=str, required=False, help='Exchange correlation functional to use (pseudopots will be selected accordingly, e.g. LDA, PBE,PBEsol, etc. Def: PBE')
        parser.add_argument("-ec", "--encut", default=520, type=float, required=False, help='Cutoff energy for the planewaves Def: 520 eV')
        parser.add_argument("-kp_sp", "--kpspacing", default=0.32, type=float, required=False, help='KP grid spacing, Def: 0.05 * 2 pi = 0.32 1/A')
        parser.add_argument("-kp_gr", "--kpgrid", default=None, type=str, required=False, help="KP grid dimensions for the slabs and interfaces, e.g. '4 4 1' (Using KP spacing is recommended), Def: None")
        parser.add_argument("-gamma", "--gamma_only", default=False, action='store_true', help='Set KP grid to 1 1 1 and use only the Gamma point for surface slabs and interface. This switches on the gamma_only version of VASP.')
        parser.add_argument("-sig", "--sigma", default=0.05, type=float, required=False, help='Gaussian smearing width, Def:0.05 eV')
        parser.add_argument("-mgm", "--magmoms", default=None, nargs="*", required=False, help="Magnetic moments for a collinear calculation. Eg, 'Fe 5.0 Nb 0.6 O 0.6' Def.: None. If a defined element is not present in the POSCAR, no MAGMOM will be set for it.")
        parser.add_argument("-hubU", "--hubU", type=str, nargs="*", help="For defining the Hubbard U parameters (in eV) for specific atom and orbital types, e.g. -hubU Fe d 2.7, Sm f 6.1")
        parser.add_argument("-ase_pot", "--ase_pot", default='LJ', type=str, required=False, choices=['LJ','Morse','EMT'], help="Potential type to be used through the ASE infrastructure, only relevant when using -prog 'ASE'. Def: LJ")
        parser.add_argument("-ase_fmax", "--ase_fmax", default=0.05, type=float, required=False, help='Force convergence criterion when using the ASE built-in geometry optimiser. Def: 0.05 eV/A')
        parser.add_argument("-ase_gsteps", "--ase_gsteps", default=200, type=int, required=False, help='Max number of BFGS steps allowed in the ASE built-in geometry optimiser. Def: 200 geometry steps')
        parser.add_argument("-mp", "-makepotcar", "--makepotcar", default=False, action='store_true', help='to compile POTCAR (for VASP) using actual atomic content. The environment variable ($VASP_PP_PATH) should be defined beforehand. Default: False.')
        parser.add_argument("--potcarxc", default="potpaw_PBE", type=str, choices=ppdirs, help='XC functional to use in making of POTCAR. Def: potpaw_PBE')
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


        #TODO: Add the -restart/overwrite options for when bulk1/2,slab1/2 folders exist.
        #TODO: Also add restart support for the convergence tests!


        args = parser.parse_args()


        if args.quick: #Set the parameters for quick testing
               args.prog='ase';
               args.ase_pot='LJ'
               args.ase_gsteps=1
               args.ase_fmax=0.01

        exe=None
        if args.prog=='vasp':
                if args.mpirun: exe="mpirun -n %d %s %s"%(args.nprocs,args.mpiargs,args.vexe)
                else: exe="srun -n %d %s %s"%(args.nprocs,args.mpiargs,args.vexe)
                exe=" ".join(exe.split())

        #Parse the Hubbard_U info and make the dict
        hubU={}
        if args.hubU:
                U=" ".join(args.hubU)
                xxx=U.split(',')
                for xx in xxx:
                        x=xx.split()
                        if len(x)!=3: parser.error("Cannot parse -hubU entry '%s' (expected e.g. 'Fe d 2.7')"%xx.strip())
                        if not x[0] in hubU:  hubU[x[0]]={x[1]:x[2]}
                        else: hubU[x[0]][x[1]]=x[2]
                print ("Hubbard U values to be used: ",hubU);stdout.flush()

        #k-point grid for the slabs and interfaces (None: use the k-point spacing).
        KPgrid=args.kpgrid
        if args.gamma_only:   KPgrid="1 1 1"

        #2 is going on the bottom so m2 ->-m2 to get right orientation of the surface (in James' new version)
        infile1 = args.infile1
        miller1 = tuple(args.miller1)
        infile2= args.infile2
        miller2 = tuple(args.miller2)
        if len(miller1)!=3 or len(miller2)!=3: parser.error("-m1/-m2 need three Miller indices, e.g. -m1 1 0 0")

        miller_list=[tuple(m) for m in args.miller_list]

        #read in atoms and construct slab, need to repeat atoms to make view work
        print("Reading data from %s and %s."%(infile1,infile2));stdout.flush()
        atoms1 = ase.io.read(infile1)
        atoms2 = ase.io.read(infile2)

        outfile = args.outfile
        outf=open(outfile,'w')
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


        #Common DFT-related parameters, taken from user
        KP=args.kpspacing  #KP spacing
        ecut= args.encut #cutoff energy in eV (convergence prob. with lower cutoffs).
        xc=args.xc
        sigma=args.sigma
        pp=args.castep_PP

        if args.prog == 'castep':
                dirr="./CASTEP-tmp"

                try:
                        calc = ase.calculators.castep.Castep()
                        calc.merge_param(args.castep_param)
                        if calc.param.cut_off_energy.value is not None: ecut=calc.param.cut_off_energy.value
                        KP=float(grep('kpoints_mp_spacing',args.castep_cell).split()[-1])
                except Exception as err: print("Can't read the param file:", err);stdout.flush()
        elif args.prog == 'vasp':
                dirr="./VASP-tmp"
        elif args.prog=='ase':
               dirr='./ASE-tmp'


        #Keep the output structures of the previous run in OUTPUT_prev (instead of deleting them).
        if os.path.exists("OUTPUT"):
                print('Moving previous output structures from OUTPUT to OUTPUT_prev...');stdout.flush()
                shutil.rmtree("OUTPUT_prev",ignore_errors=True)
                shutil.move("OUTPUT","OUTPUT_prev")
        os.makedirs("OUTPUT",exist_ok=True)



        #Time keeping
        initT=time.time()

        ########################
        # Do bulk calculations #
        ########################

        # VASP automatically checks if OUTCAR alrady exists and if run is converged (does continuation run of not)
        print("Computing bulk 1 energy.");stdout.flush()
        atoms1.name='bulk1'
        x1=run_calc(atoms1,"geom",'bulk1',dipolCorr='None')

        print("Computing bulk 2 energy.");stdout.flush()
        atoms2.name='bulk2'
        x2=run_calc(atoms2,"geom",'bulk2',dipolCorr='None')

        atoms1=x1[-1]
        Ebulk1=x1[0]

        fu1=get_fu(atoms1)
        Ebulk1 /= len(atoms1) #Get the bulk energy per atom.

        atoms2=x2[-1]
        Ebulk2=x2[0]

        fu2=get_fu(atoms2)
        Ebulk2 /= len(atoms2) #Get the bulk energy per atom.

        #Clean up to save mem
        del x1,x2

        write_struct(atoms1,"bulk1_opted"); write_struct(atoms2,"bulk2_opted")


        ###########################################################
        # Surface stability and layer thickness convergence tests #
        ###########################################################
        # These are done before the creation of the slabs, so that the selected Miller indices (-tsur)
        # and numbers of layers (-tlay) are actually used for the slabs/interfaces.

        #TODO: the thickness argument apparently has no effect on the created surface slabs, but only on the interface.
        # creps argument has no effect on symmetric surfaces, as min thickness is used
        if args.creps1: creps1=args.creps1
        else:creps1=1
        if args.creps2: creps2=args.creps2
        else:creps2=1

        if args.checkSurfs:
                print('\nChecking the minimum-energy surface for Material 1 (%s), considering the Miller indices: %s'%(atoms1.get_chemical_formula(empirical=1),miller_list));stdout.flush()
                Ws1,_,_,miller1=check_surfaces(atoms1, miller_list,vac=args.vac_slab,Ebulk=Ebulk1,creps=creps1)
                print('\nChecking the minimum-energy surface for Material 2 (%s), considering the Miller indices: %s'%(atoms2.get_chemical_formula(empirical=1),miller_list));stdout.flush()
                Ws2,_,_,miller2=check_surfaces(atoms2, miller_list,vac=args.vac_slab,Ebulk=Ebulk2,creps=creps2)
                print('Selected surfaces: %s for material 1 and %s for material 2'%(miller1,miller2));stdout.flush()

        if args.convLayers:#Run layer thickness
                print('\n\nConverging layer thickness...');stdout.flush()
                slab1 = make_slab(miller1,atoms1,repeat=(1,1,1),square=False)
                slab2 = make_slab(miller2,atoms2,repeat=(1,1,1),square=False)
                slab1.center(vacuum=0, axis=2) #to delete the vacuum padding in the slabs
                slab2.center(vacuum=0, axis=2)
                slab1.name="%s_%d%d%d"%(slab1.get_chemical_formula(empirical=1),miller1[0],miller1[1],miller1[2])
                slab2.name="%s_%d%d%d"%(slab2.get_chemical_formula(empirical=1),miller2[0],miller2[1],miller2[2])
                print ('Running Material 1 (%s)'%(slab1.get_chemical_formula(empirical=1)))
                nl1,Eslab1,slab1=conv_layers(slab1,vac=args.vac_slab,lay_min=args.lay_min-1,lay_max=args.lay_max-1)  #Vacuum layer is added automatically within the function.
                print ('\nRunning Material 2 (%s)'%(slab2.get_chemical_formula(empirical=1)))
                nl2,Eslab2,slab2=conv_layers(slab2,vac=args.vac_slab,lay_min=args.lay_min-1,lay_max=args.lay_max-1)  #Vacuum layer is added automatically within the function.

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


        ########################################################################
        # Pre-optimisation of the slabs and surface formation energies (SFE) #
        ########################################################################
        def read_decomps(n,files):
                #Read the decomposition products; if they have no energy, optimise them (as bulk phases).
                decomps = []
                for no_d, d in enumerate(files):
                        decomp = ase.io.read(d, index=-1)
                        if decomp.calc is None:
                                decomp.name = 'decomp%d_%s'%(n,no_d)
                                x=run_calc(decomp,'geom','decomp%d_%s'%(n,no_d),dipolCorr='None')
                                decomp=x[-1]
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
                        write_struct(slab_opt,"slab%d_cut%s_pre"%(n,i))

                        print("Pre-optimizing %s (before alignment)."%slab_opt.name);stdout.flush()
                        x=run_calc(slab_opt,'geom','slab%d_cut%s_pre'%(n,i),dipolCorr='sc',FixCell=True,FixVol=0,passivate=pas,KPgrid_=KPgrid) #fixed cell: keeps the vacuum padding
                        slab_opt=x[-1]
                        slab_opt.name='slab%d_cut%s'%(n,i)
                        write_struct(slab_opt,"slab%d_cut%s_pre_opted"%(n,i))

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

        # sort the slabs by their SFE values and save the args.noSlabs lowest energy slabs
        low_SFEs_slab1 = sorted(enumerate(SFE_list_slab1), key=lambda x: x[1])[:args.noSlabs]
        le_surf_slab1 = [surf_list_slab1[i] for i, _ in low_SFEs_slab1]
        values_SFE_s1 = [v for _, v in low_SFEs_slab1]
        indices_le_s1 = [i for i,_ in low_SFEs_slab1]

        low_SFEs_slab2 = sorted(enumerate(SFE_list_slab2), key=lambda x: x[1])[:args.noSlabs]
        le_surf_slab2 = [surf_list_slab2[i] for i, _ in low_SFEs_slab2]
        values_SFE_s2 = [v for _, v in low_SFEs_slab2]
        indices_le_s2 = [i for i,_ in low_SFEs_slab2]

        for n,sym,surfs,vals in ((1,args.symSlabs1,le_surf_slab1,values_SFE_s1),(2,args.symSlabs2,le_surf_slab2,values_SFE_s2)):
                if sym: print("\nThe %s lowest energy slabs for material %d are:"%(args.noSlabs,n));stdout.flush()
                else: print("\nThe surface formation energy of slab%d_cut0 is:"%n);stdout.flush()
                for surf, val in zip(surfs, vals):
                        print("%s: %.2f eV/nm^2"%(surf.name, val));stdout.flush()

        # exit functionality
        # if keyword provided, exit safely after slab calculations
        if args.surf_only:
                print('Exiting code after the optimisation of the surface slabs');stdout.flush()
                outf.close()
                exit()


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
                        if args.flip2: #Flip the slab2 (top slab) upside-down.still the cell angles cannot be preserved !! using -h,k,-l could solve this. Or can be moved to after optimizations of the slabs (pasivation settings should be changed, from top to bottom)
                                #This does not work!! try the new flip_cell fnc isntead
                                slab2.rotate(a=180,v='x',rotate_cell=0,center = (0, 0, 0)) #burda yapinca stack calismiyor.
                                slab2.center()
                                if args.view: view(slab2)

                        write_struct(slab1,"slab1_cut%s_aligned"%ci); write_struct(slab2,"slab2_cut%s_aligned"%cj)

                        #Interface before optimizing the individual slabs.
                        sep=args.sep
                        vac_used=args.vac_int #vacuum padding (on each side) of the final interface
                        interface=make_interface(slab1,slab2,sep,args.vac_int,reorder=True)
                        if args.view:view(interface)
                        write_struct(interface,"interface_cuts_%s_%s"%(ci,cj))

                        #Calculate energies/structures of the actual slabs (after alignment).
                        print("\nOptimizing %s and %s (after alignment)."%(slab1.name, slab2.name));stdout.flush()
                        slab1.center(vacuum=args.vac_slab, axis=2) #to add vacuum to the slabs (needed)
                        slab2.center(vacuum=args.vac_slab, axis=2)

                        x1=run_calc(slab1,'geom','slab1_cut%s_aligned'%ci,dipolCorr='sc',FixCell=True,FixVol=0,passivate=('bot' if args.pas else None),nosymm=1,vac=args.vac_slab,KPgrid_=KPgrid)
                        slab1=x1[-1]
                        Eslab1=x1[0]

                        x2=run_calc(slab2,'geom','slab2_cut%s_aligned'%cj,dipolCorr='sc',FixCell=True,FixVol=0,passivate=('top' if args.pas else None),nosymm=1,vac=args.vac_slab,KPgrid_=KPgrid)
                        slab2=x2[-1]
                        Eslab2=x2[0]
                        del x1,x2

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

                        write_struct(slab1,"slab1_cut%s_aligned_opted"%ci); write_struct(slab2,"slab2_cut%s_aligned_opted"%cj)

                        if args.view:view(slab1);view(slab2)

                        #Create the interface from the optimised slabs.
                        interface=make_interface(slab1,slab2,sep,args.vac_int,reorder=True)
                        if args.verb:
                                a,b,c,alpha,beta,gamma=interface.cell.cellpar()
                                print (a,b,c,alpha,beta,gamma);stdout.flush()
                        if args.view:view(interface)
                        write_struct(interface,"interface-pre_opt_cuts_%s_%s"%(ci,cj))

                        if args.convSep: #Run optimisation on the interface speration distance
                                print("\n\nFinding the optimal interface (interlayer) separation.");stdout.flush()
                                vacc=None #separation/2 will be used as default to have a symmetric seperation at all interfaces.
                                interface,sep=convSep(slab1,slab2,vac=vacc,sep_init=args.sep_init,sep_final=args.sep_final,sep_step=args.sep_step,view=0,ifPlot=1)
                                vac_used=sep/2

                        if args.test_hori: #Optimise the horizontal stacking of the layers by translating slab
                                print("\n\nFinding the optimial interface lateral shift.");stdout.flush()
                                interface,Eint,minxy=checkHorizontal(slab1,slab2,sep=sep,steps=args.hor_steps,ifPlot=1)
                                vac_used=sep/2

                        if args.niggli: print ("Niggli reduce the interface...");niggli_reduce(interface);stdout.flush()
                        if args.view:view(interface)

                        #Optimise the final interface geometry
                        print("\nOptimizing the final interface geometry.");stdout.flush()
                        interface.name='interface_cuts_%s_%s'%(ci,cj)
                        if vac_used>4:dipC='SC'
                        else:dipC='None'
                        #CHECK: Should we fix the volume or not?? fix vol works for some cases (YIG) but not for some (Al2O3//Si)
                        x=run_calc(interface,'geom',interface.name,dipolCorr=dipC,FixVol=0,nosymm=1,KPgrid_=KPgrid)
                        interface=x[-1]
                        Eint=x[0]

                        #Work of adhesion: W_ad = (E_slab1 + E_slab2 - E_interface) / (n_int * A)  (positive for a bound interface).
                        #With a small vacuum padding (2*vac_used across the periodic boundary, e.g. vac=sep/2 as used in -tsep/-thor),
                        #the cell contains two interfaces (and no free surfaces), so n_int=2; with a large vacuum (>4 A on each side,
                        #where also the dipole correction is switched on) there is one interface and two free surfaces, so n_int=1.
                        n_int = 2 if vac_used<=4 else 1
                        Wad=(Eslab1+Eslab2-Eint)/(n_int*surf_area(interface))/0.01 #A2 to nm2
                        str1='W_ad (work of adhesion, %d interface(s) in the cell) for the final optimised interface: %.2f eV/nm^2 = %.2f J/m^2\n'%(n_int,Wad,Wad*cfactor)
                        print(str1);stdout.flush()
                        outf.writelines(str1)

                        write_struct(interface,"interface_cuts_%s_%s_opted"%(ci,cj))
                        ase.io.write("OUTPUT/interface_cuts_%s_%s_opted.cif"%(ci,cj),interface,format='cif')


        outf.close()

        print ("Elapsed time: %.2f sec."%( time.time()-initT));stdout.flush()

        exit()

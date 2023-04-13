from __future__ import print_function
import numpy as np
import argparse,math, os.path,fractions,time
import matplotlib.pyplot as plt

from os import system,popen,chdir,getcwd,getenv,putenv,listdir
from re import search
from sys import exit
from copy import deepcopy

from spglib import find_primitive,standardize_cell,get_spacegroup #, niggli_reduce #niggli_reduce from ASE-tools conflicts with that from spglib.

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

#Import other functions from the INTERFACER package
from tools.builder import *
#from tools.builder import plot_slab_axes
from tools.external import *
#from tools.analysis import *


def get_thickness(atoms):
        zcoords=[at.position[2] for at in atoms]
        print (max(zcoords)-min(zcoords))
        return max(zcoords)-min(zcoords)

def compute_rdf(dists,cellVol,r_range=None, bin_width=0.005, n_bins=None, norm_by_V=True):
    """Compute radial distribution functions for interatomic distances along a given trajectory. Adapted from mdtraj pacakge. Returns, bin centre points, g(r) and the coordination(r)

    Parameters
    ----------
    dists: list or array, shape (1,N)
    r_range : array-like, shape=(2,), optional, default=(min(dists), max(dists))
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.

    Returns
    -------
    r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        Radii values corresponding to the centers of the bins.
    g_r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        Radial distribution function values at r.
    coord : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        coordination number at r.
    """

    dists=np.array(dists)

    if r_range is None:
        r_range = np.array([min(dists), max(dists)])
    #r_range = ensure_type(r_range, dtype=np.float64, ndim=1, name='r_range',
    #                      shape=(2,), warn_on_cast=False)  #This is a mdtraj property as wel !!
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError('n_bins must be a positive integer')
        bins = np.linspace(r_range[0], r_range[1], n_bins)
    else:
        bins = np.arange(r_range[0], r_range[1] + bin_width, bin_width)

    #distances = compute_distances(traj, pairs, periodic=periodic, opt=opt) #Replace this with ASE !!
    g_r, edges = np.histogram(dists, bins=bins)
    r = 0.5 * (edges[1:] + edges[:-1])


    if norm_by_V:
    # Normalize by volume of the spherical shell.
    # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
    # a less biased way to accomplish this. The conclusion was that this could
    # be interesting to try, but is likely not hugely consequential. This method
    # of doing the calculations matches the implementation in other packages like
    # AmberTools' cpptraj and gromacs g_rdf. 
    # VMD and LAMMPS also seem to do this !!!
        V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        #norm = len(pairs) * np.sum(1.0 / traj.unitcell_volumes) * V
        norm = len(dists) * np.sum(1.0 / cellVol ) * V
        g_r = g_r.astype(np.float64) / norm  # From int64.
    else:
        norm = len(dists) 
        g_r = g_r.astype(np.float64) / norm  # From int64.

    coord=np.array([np.sum(g_r[0:i]) for i in range(len(g_r))])

    return r, g_r , coord

def minDist_ASE(arr1,arr2,latt): #needs np.array inputs
    if len(arr1)==4: arr1=arr1[1:4]
    if len(arr2)==4: arr2=arr2[1:4]
    D,x=find_mic(D=np.array([arr1-arr2]),cell=latt,pbc=True) 
    return x[0]
    
def minDist(arr1,arr2,latt):
    #Finds the minimum distance from the images applying the PBC. (Minimum image convention).
    #This function works fine for ortho and non-ortho cells, however the find_mic() from ase.geometry is upto 2x faster (using a different algorithm).
    if len(arr1)==4: arr1=arr1[1:4]
    if len(arr2)==4: arr2=arr2[1:4]
    #print arr1,arr2
    minD=dist(arr1,arr2)
    #newarr2=deepcopy(arr2)
    newarr2=[0.0, 0.0, 0.0]
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            for k in range(-1,2,1):
                #newarr2=deepcopy(arr2)
                newarr2=[0.0, 0.0, 0.0]
                newarr2[0]=arr2[0]+i*latt[0][0]+j*latt[1][0]+k*latt[2][0]
                newarr2[1]=arr2[1]+i*latt[0][1]+j*latt[1][1]+k*latt[2][1]
                newarr2[2]=arr2[2]+i*latt[0][2]+j*latt[1][2]+k*latt[2][2]
                currD=dist(arr1,newarr2)
                if currD<minD: minD=currD ;#print minD
    return minD

def dist(arr1,arr2):
    #Distance btw two atoms. Works with two 1x3 arrays.
    if len(arr1)==4: arr1=arr1[1:4]
    if len(arr2)==4: arr2=arr2[1:4]
    return math.sqrt((arr1[0]-arr2[0])**2+(arr1[1]-arr2[1])**2+(arr1[2]-arr2[2])**2)

def misfit(slab1,slab2,ptol=2,ifPlot=False): #Computes the misfit parameter between two slabs. (see Surf. Interface Anal. 2003, 35, 835-841. )
        #Assuimng that two slabs are well aligned, overlap area should be equalt to smaller one's surface area.  !!FIND A MORE ACCURATE WAY OF DETERMINING OVERLAPPING SURFACE.
        #align_slab_axes(slab1,slab2,ptol)
        A1=surf_area(slab1);A2=surf_area(slab2)
        #if ifPlot: plot_slab_axes(slab1,slab2)
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


#Done in the main method and currently not called; probably not needed to put it in a separate function.
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
                slab1_vac.center(vacuum=args.vac_slab, axis=2)
                slab2_vac.center(vacuum=args.vac_slab, axis=2)
                atoms=call_castep(slab1_vac,typ="SP",dipolCorr='SC',name='slab1',ENCUT=ecut,KPgrid='1 1 1',PP=pp)
                atoms=slab1_vac
                atoms.set_calculator(calc)
                Eslab1=atoms.get_potential_energy()
                fu1=get_fu(atoms)
                
                atoms=call_castep(slab2_vac,typ="SP",dipolCorr='SC',name='slab2',ENCUT=ecut,KPgrid='1 1 1',PP=pp)
                atoms=slab2_vac
                atoms.set_calculator(calc)
                atoms.calc._label="slab2"
                Eslab2=atoms.get_potential_energy()
                fu2=get_fu(atoms)

        #ase.io.write("slab1.cell",slab1.repeat((1,1,1)),format='castep-cell')
        #ase.io.write("slab2.cell",slab2.repeat((1,1,1)),format='castep-cell')

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
        int1.center(vacuum=args.vac_int,axis=2)
        if args.view:view(int1)
        ase.io.write("interface1.cell",int1.repeat((1,1,1)),format='castep-cell')
        atoms=call_castep(slab1_vac,typ="SP",dipolCorr='SC',name='int1',ENCUT=ecut,KPgrid='1 1 1',PP=pp) #TODO: use general ENCUT setting
        #TODO: add call_VASP
        #atoms=int1
        #atoms.set_calculator(calc)
        Eint=atoms.get_potential_energy()

        #Wad=(Ws1+Ws2-Eint)/surf_area(int1)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf?? This is wrong
        #print('Wad before alingment: %.2f eV/nm^2'%Wad)

        Wad=(Eslab1+Eslab2-Eint)/surf_area(int1)/0.01 #A2 to nm2 #check the formula Ea isntead of Wsurf??
        
        print(('W_ad before alingment: %.2f eV/nm^2\n'%Wad))
        return Eslab1,Eslab2,Ws1,Ws2,int1#surf_area(int1)
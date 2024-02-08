import numpy as np
import argparse
import ase.io
import ase
from ase.visualize import view
from os import system
from sys import exit
#import fractions
import math
import ase.build
#import copy
#import time
import ase.build.tools
import matplotlib.pyplot as plt

from ase import Atoms
from ase.build import surface
from ase.io import read, write
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from spglib import find_primitive, get_symmetry
from tqdm import tqdm

#Import other functions from the INTERFACER package
#from tools.builder import *
from tools.external import *
from tools.analysis import *

#Surface creation and alignment tool by James P. Darby.
#This is the former join8.py

def calc_R1(x1,y1):
    #gives a rotation matrix that will rotate x1 onto y1
    Id = np.array([[1,0,0],[0,1,0],[0,0,1]])
    u = np.cross(x1,y1)

    tol = 0.001
    if np.linalg.norm(u) < tol:
            return Id

    sin_th = np.linalg.norm(u)/(np.linalg.norm(x1)*np.linalg.norm(y1))
    cos_th = np.dot(x1,y1)/(np.linalg.norm(x1)*np.linalg.norm(y1))
    u = u/np.linalg.norm(u)

    #use wiki formula for R1
    ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    ut = np.tensordot(u,u,axes=0)
    R1 = cos_th*Id + sin_th*ux + (1-cos_th)*ut
    return(R1)

def rotate_slab(slab):
    d = slab.cell[0]
    e = slab.cell[1]
    f = slab.cell[2]
    frac_pos = slab.get_scaled_positions()
    R1 = calc_R1(np.cross(d,e),[0,0,1])
    
    slab.cell[0] = np.dot(R1,d)
    slab.cell[1] = np.dot(R1,e)
    slab.cell[2] = np.dot(R1,f)
    slab.set_scaled_positions(frac_pos)
    return slab

def lcm(a,b):
    #return (a*b)//fractions.gcd(a,b)
    return (a*b)//math.gcd(a,b)

def get_cut_vectors(miller, atoms_cell):
    #from miller indicies construct 2 lattice vectors in the required plane
    h,k,l = miller
    a = atoms_cell[0]
    b = atoms_cell[1]
    c = atoms_cell[2]
    zero_inds = np.argwhere(np.array(miller)==0)
    
    #no zero miler indicies
    if len(zero_inds) == 0:
        lm = lcm(h,l);    d = (lm/h,0,-lm/l)    #vector from c intersection to a
        lm = lcm(k,l);    e = (0,lm/k,-lm/l)     #vector from c` intersection to b
        #pick f so that cell is left handed
        f = (0,0,1)
        origin = c*1.0/l
    
    elif len(zero_inds)==3:
        print("miller not a valid plane")
        raise ValueError

    #one zero milller index
    elif len(zero_inds)==1:
        #set d to be the axis that is in the plane    
        d = [0,0,0]
        d[int(zero_inds[0])] = 1

        #construct e from the other 2 vectors
        s = set((0,1,2))
        s.remove(int(zero_inds[0]))
        print("s is",s)
        H = s.pop()
        K = s.pop()
        lm = lcm(miller[H],miller[K])
        print(H,miller[H],K,miller[K],lm)
        e = [0,0,0]
        e[H] = lm/miller[H]
        e[K] = -lm/miller[K]
        print("e is", e)
        
        #pick f - could be improved...
        f = [0,0,0]
        f[H] = 1
        
        #pick origin
        origin = 1.0/miller[H] * atoms_cell[H]    
    

    #two zero miller indicies
    elif len(zero_inds)==2:
        d = [0,0,0]
        #print zero_inds
        d[zero_inds[0][0]] = 1
        e = [0,0,0]
        e[zero_inds[1][0]] = 1
        f_ind = np.argwhere(np.array(miller)!=0)[0][0]
        f = [0,0,0]
        f[f_ind] = 1
        origin = atoms_cell[f_ind]*1.0/miller[f_ind]    
        #print d,e,f
        #print origin

    #ensure that slab is always left handed    
    if np.dot(np.cross(d,e),f) < 0:
        f = -np.array(f)
    """
    #pick f so that it points from the plane, away from the surface
    #origin in cartesians, e,f,d in indicies
    """
    fc = f[0]*a + f[1]*b + f[2]*c
    if np.linalg.norm(origin+fc) < np.linalg.norm(origin):
        f  = np.array(f)
        f = -f
    
    dc = d[0]*a + d[1]*b + d[2]*c
    ec = e[0]*a + e[1]*b + e[2]*c
    if np.dot(np.cross(dc,ec),fc) < 0:
        t = d
        d = e
        e = t
    
    return d,e,f,origin

def wrap_coords(slab):
    sp = slab.get_scaled_positions()
    sp %= 1.0
    slab.set_scaled_positions(sp)
    return slab

### Code for symmetric surfaces by Julian Holland: https://github.com/julianholland/Symmetric-Surface-Finder

def find_smallest_sym_surface(surf, surface_layers, vacuum, position_accuracy):
    """For any given surface this function will produce a symmetric surface should one exist"""
    cell = surf.cell  # find cell parameters
    frac_cell_value = (
        ((cell[2, 2] - 2 * vacuum) * surface_layers) / (surface_layers + 1)
    ) + vacuum  # find a value for minimum space we want to explore for this cut
    z_coords = surf.positions[:, 2]  # create list of all z-coords
    unique_surfaces = np.unique(
        z_coords.round(decimals=position_accuracy)
    )  # find all unique cuts
    mask = (
        unique_surfaces > frac_cell_value)  # find unique cuts for region we wish to explore
    frac_unique_surfaces = unique_surfaces[mask]
    for i in range(len(frac_unique_surfaces)):
        mask = z_coords < frac_unique_surfaces[i]
        new_surf = surf[mask]
        new_surf.center()
        rotational_symmetry = get_symmetry(new_surf, symprec=1e-5)["rotations"]
        z_rotational_operations = np.stack(rotational_symmetry)
        if np.any(z_rotational_operations[:, 2, 2] == -1):
            sym_surf = new_surf
            sym_surf.center(vacuum=vacuum, axis=2)
            break
        else:
            sym_surf = False
    return sym_surf

def make_surface(thickness, miller_plane, bulk, vacuum):
    """Create a surface with 10 angstrom of vacuum provided a bulk, miller plane and layer thickness"""
    surf = surface(bulk, miller_plane, thickness)
    surf.center(vacuum=vacuum, axis=2)  # centre the surface and add vacuum
    single_layer_z_length = (surf.cell[2, 2] - (2 * vacuum)) / thickness
    return surf, single_layer_z_length

def permute_surface(bulk, miller_plane, minimum_thickness, vacuum):
    """Permutes a given surface for all unique z-axis coordinates and creates an xyz file"""
    position_accuracy = 5  # modify this depending on how accurate your atomic measurements are. A higher value generally gives more surfaces as less atoms lie on the 'same' z-plane
    surface_layers = 0
    nudge = 10 ** (-1 * position_accuracy)
    dummy_surf, single_cell_z_length = make_surface(1, miller_plane, bulk, vacuum) # get the single cell value with vacuum
    min_thick_w_buffer = minimum_thickness  # add a buffer to make it more likely that we don't go less than the desired minimum thickness
    thickness_check=0
    while thickness_check <= min_thick_w_buffer: # create a sufficiently thick surface
        surface_layers += 1
        surf, single_cell_z_length = make_surface(
            thickness=(surface_layers + 2),
            miller_plane=miller_plane,
            bulk=bulk,
            vacuum=vacuum,
        )
        thickness_check = ((single_cell_z_length) * (surface_layers))

    z_coords = surf.positions[:, 2]  # create list of all z-coords
    unique_surfaces = np.unique(
        z_coords.round(decimals=position_accuracy)
    )  # create list of all unique z coordinates
    mask = unique_surfaces < single_cell_z_length + vacuum
    single_cell_unique_surfaces = unique_surfaces[mask]
    surf_list = []
    surf_lengths_list = []
    print(
        "There are "
        + str(len(single_cell_unique_surfaces))
        + " unique surface cuts for the "
        + str(miller_plane)
        + " miller plane. One layer of this surface is "
        + str(single_cell_z_length.round(position_accuracy))
        + " Angstrom thick"
    )
    for i in tqdm(range(len(single_cell_unique_surfaces))):
        # define where we cut the surface on the left hand side
        surf_cut = (unique_surfaces[i] + nudge) 
        # integer value of number of unique values required to maintain minimum thickness
        ensure_thickness = int(i + (((surface_layers + 2 - 1) / (surface_layers + 2) ) * len(unique_surfaces))) 
        # define where we cut the surface on the right hand side
        surf_term = (unique_surfaces[ensure_thickness] + nudge)  
        # create T/F matrix for appropriate slice
        newmask = (z_coords > surf_cut) & (z_coords < surf_term)  
        cut_surf = surf[newmask]  # apply mask to surface
        cut_surf.center(vacuum=vacuum, axis=2)  # centre and keep vacuum the same

        # find symmetric surfaces
        sym_surf = find_smallest_sym_surface(
            cut_surf, surface_layers, vacuum, position_accuracy)  
        if sym_surf == False:
            continue
        surf_lengths_list.append(sym_surf.cell[2, 2] - (2 * vacuum))
        if check_surf_unique(surf_list, sym_surf, position_accuracy) > 0:
            continue
        surf_list.append(sym_surf)
    if len(surf_list) == 0:
        print("Found no symmetric structures for this Miller plane.\n")
    else:
        print(
            "Found "
            + str(len(surf_list))
            + " unique, symmetric surfaces for "
            + str(miller_plane)
            + " between "
            + str(min(surf_lengths_list).round(decimals=position_accuracy))
            + " and "
            + str(max(surf_lengths_list).round(decimals=position_accuracy))
            + " Angstrom thick ("
            + str(surface_layers)
            + " to "
            + str(surface_layers + 1 )
            + " layers thick)"
            + "\n" 
        )
    return surf_list

def check_surf_unique(surf_list, surf, position_accuracy):
    """Checks all surfaces in a given directory are unique"""
    symprec=1*10**-(position_accuracy)
    if len(surf_list) == 0:
        return 0
    for i in range(len(surf_list)):
        if surf_list[i].get_chemical_formula(
            mode="metal", empirical=False
        ) == surf.get_chemical_formula(mode="metal", empirical=False):
            new_surf_sym = get_symmetry(surf, symprec=symprec)
            old_surf_sym = get_symmetry(surf_list
            [i], symprec=symprec)
            if old_surf_sym["rotations"].all() == new_surf_sym["rotations"].all():
                return 1
            else:
                continue
    return 0

##########################################################################################################

def make_slab(miller, atoms,repeat=(1,1,1),square=False):
    #construct slab
    d,e,f,origin = get_cut_vectors(miller, atoms.cell)
    atoms = wrap_coords(atoms)
    #slab = ase.build.cut(atoms, d, e, f, origo=origin)
    slab = cut_cell(atoms, d, e, f, origin=origin)
    slab = slab.repeat(repeat)
    slab = rotate_slab(slab)
    slab = AG_slab(slab)
    
    if square:
        slab = square_slab(slab)
    #gives the slabs a consistent handedness
    #swaps a1,a2 after AG if necessary
    #maintains a3.(0,0,1) > 0
    if np.dot(slab.cell[2],np.cross(slab.cell[0],slab.cell[1])) < 0:
        slab = flip_handedness(slab)
    return slab

def square_slab(slab):
    #now set f perpendicular to surface, means can't make it thicker any more!
    #assumes that the slab is aligned with a3 perpendicular to (0,0,1)
    slab.cell[2][1] = 0
    slab.cell[2][0] = 0    
    slab = wrap_coords(slab)
    return slab

def acute_Gauss(u,v):
        u = np.array(u)
        v = np.array(v)
        #start by making sure that u is smaller than v
        if np.linalg.norm(v) < np.linalg.norm(u):
                t = u;  u = v;  v = t;
        #calculate the shortest distance from v + a u to the origin
        epsilon = 10**-5
        while True:
                uhat = u/np.linalg.norm(u)
                vhat = v/np.linalg.norm(v)
                a = np.dot(uhat,vhat)
                a = round(a)
                v = v - a*u
                if (1-epsilon)*np.linalg.norm(u) < np.linalg.norm(v):
                        return(u,v)
                else:
                        t = u;  u = v;  v = t;

        return u,v
    
def basis_change(ab,uv,ruv):
    #ab is final basis, uv is current, ruv is r in uv
    #ab are tuples of basis vectors as rows
    #rab are the components of r in the ab basis
    A = np.transpose(ab);    #print(A)
    U = np.transpose(uv);    #print(U)
    M = np.dot(np.linalg.inv(A),U);    #print(M)
    rab = np.dot(M,ruv)
    return rab

def AG_slab(slab):
    a = [slab.cell[0][i] for i in range(0,2)]
    b = [slab.cell[1][i] for i in range(0,2)]
    a2,b2 = acute_Gauss(a,b)
    a2_ab = basis_change([a,b],[a2,b2],[1,0])
    b2_ab = basis_change([a,b],[a2,b2],[0,1])
    #add the 0 to them so that they can be cut out
    a2_ab = np.append(a2_ab,0)    
    b2_ab = np.append(b2_ab,0)
    slab = cut_cell(slab,a2_ab,b2_ab,(0,0,1))
    return slab

def reasnoble_slab_angles(Lmax=10,slab=None):
    def filter_to_snp(abrep,a,b):
        #gets passed a list of vectors
        #filtere it down to non-parallel
        #taking the shortes option
        angtol = 0.000001
        lentol = 0.000001
        ahat = a*1.0/np.linalg.norm(a)
        vecs = []
        for x in abrep:
            v = a*x[0]+b*x[1]
            vn = np.linalg.norm(v)
            if abs(vn) <= lentol:
                continue
            ctheta = np.dot(v,ahat)/vn
            theta = np.arccos(ctheta)
            tested = False
            for i,y in enumerate(vecs):
                if (y[2]-angtol <= theta <= y[2] +angtol) or (y[2]-angtol <= np.pi - theta <= y[2]+angtol):
                    tested = True
                    if vn < y[3]:
                        vecs[i] = [x[0],x[1],theta,vn,v]
                    break
            if not tested:
                vecs.append([x[0],x[1],theta,vn,v])
            
        return vecs
    #creates a list of reasnoble slab angles,areas of those slabs
    #and slab vectors in terms of a,b
    #considering slab lengths below Lmax
    angles = []
    vecs = []
    
    a = [slab.cell[0][i] for i in range(0,2)]
    b = [slab.cell[1][i] for i in range(0,2)]
    a = np.array(a);    b = np.array(b);
    naMax = int(np.floor(Lmax/np.linalg.norm(a)))
    for na in range(0,naMax+1):
        A = na*a
        d_vec = A - (np.dot(A,b)*1.0/np.linalg.norm(b)**2)*b
        d = np.linalg.norm(d_vec)    

        if d < Lmax:
            #print  a, b, A, d_vec
            d_ab = basis_change([a,b],[[1,0],[0,1]],d_vec)
            nbf = np.round(d_ab[1])
            nbc = nbf + 1
            #add from nbc, until not within Lmax
            while np.linalg.norm(A + nbc*b) <= Lmax:
                vecs.append([na,nbc])
                nbc +=1
            #subtract from bvf, until not within Lmax
            while np.linalg.norm(A +nbf*b) <= Lmax:
                vecs.append([na,nbf])
                nbf -=1
                
    #filter vecs, removing those which are parallel
    vecs = filter_to_snp(vecs,a,b)
    for i,c1 in enumerate(vecs):
        for j in range(i,len(vecs)):
            c2 = vecs[j]
            v1 = c1[0]*a+c1[1]*b    
            v2 = c2[0]*a+c2[1]*b    
        
            if np.linalg.norm(v1) < 0.0001 or np.linalg.norm(v2) < 0.001:
                continue
            
            #make right handed
            v13 = [x for x in v1]; v13.append(0);    v13 = np.array(v13)
            v23 = [x for x in v2]; v23.append(0);    v23 = np.array(v23)
            
            if np.dot(slab.cell[2],np.cross(v13,v23)) > 0:
                d1 = [x    for x in c1]
                d2 = [x for x in c2]
            else:
                d1 = [x for x in c2]
                d2 = [x for x in c1]
                v1 = d1[0]*a+d1[1]*b    
                v2 = d2[0]*a+d2[1]*b    
            
            ctheta = np.dot(v1,v2)*1.0/(np.linalg.norm(v1)*np.linalg.norm(v2))
            if ctheta > 1-0.001 or ctheta < -1 + 0.001:
                continue
                
            Area = np.linalg.norm(np.cross(v1,v2))
            theta = np.arccos(ctheta)
            
            #add so that the slab is acute and right-handed
            #f right angle add both options    
            if theta < np.pi/2 - 0.01:
                angles.append([theta,Area,[d1[0],d1[1]],[d2[0],d2[1]]])
            elif theta > np.pi/2 + 0.01:
                angles.append([np.pi-theta,Area,[-d2[0],-d2[1]],[d1[0],d1[1]]])
            else:
                angles.append([theta,Area,[d1[0],d1[1]],[d2[0],d2[1]]])
                angles.append([np.pi-theta,Area,[-d2[0],-d2[1]],[d1[0],d1[1]]])
    return angles

def LCM_float(x,y,Lmax,ptol):
    #find nx -ny <=tol
    swap = False
    tol = 0.001
    if x < y:
        t = x;    x = y; y = t;    
        swap = True
    Nxmax = int(np.ceil(Lmax*1.0/x))
    Nymax = int(np.ceil(Lmax*1.0/y))
    for nx in range(1,Nxmax):
        ny = 1;
        X = nx*x; Y = y
        while X >= Y-tol and ny < Nymax:
            Y = ny*y
            pd = abs(X-Y)*100.0/X
            if pd < ptol:
                if not swap:
                    return True,nx,ny
                else:
                    return True,ny,nx
        
            ny += 1
    return False,0,0

def flip_handedness(atoms):
    pos = atoms.positions
    sp = atoms.get_scaled_positions()
    cell = [line for line in atoms.cell]
    #swap a and b then swap scaled positions
    temp = cell[0]
    cell[0] = cell[1];    cell[1] = temp
    atoms.cell = cell
    swsp = [[x[1], x[0], x[2]] for x in sp]
    atoms.set_scaled_positions(swsp)
    return atoms

def match_angles2(slab1_cell, angles1, slab2_cell, angles2, ptol, Lmax):
    matches = []
    for ang1 in angles1:
        for ang2 in angles2:
            pd = 100*abs(ang1[0]-ang2[0])/float(ang1[0])
            #print('Current pd: ',pd)
            if pd > ptol:
                continue
            
            a1 = ang1[2][0]*slab1_cell[0] + ang1[2][1]*slab1_cell[1]
            a2 = ang1[3][0]*slab1_cell[0] + ang1[3][1]*slab1_cell[1]
            b1 = ang2[2][0]*slab2_cell[0] + ang2[2][1]*slab2_cell[1]
            b2 = ang2[3][0]*slab2_cell[0] + ang2[3][1]*slab2_cell[1]
            theta_check1 = np.arccos(np.dot(a1,a2)/(np.linalg.norm(a1)*np.linalg.norm(a2)))    
            theta_check2 = np.arccos(np.dot(b1,b2)/(np.linalg.norm(b1)*np.linalg.norm(b2)))    
            
            match1,na1,nb1 = LCM_float(np.linalg.norm(a1),np.linalg.norm(b1),Lmax,ptol)
            if match1:
                match2, na2,nb2 = LCM_float(np.linalg.norm(a2),np.linalg.norm(b2),Lmax,ptol)
                if match2:
                    Area = np.linalg.norm(np.cross(a1,a2))*na1*na2
                    theta = 0.5*(ang1[0]+ang2[0])
                    #if theta <0.75: 
                    #print (theta);# continue
                    #print(ang1[0],ang2[0],theta_check1,theta_check2)
                    A1 = (ang1[2][0]*na1,ang1[2][1]*na1,0)
                    A2 = (ang1[3][0]*na2,ang1[3][1]*na2,0)
                    B1 = (ang2[2][0]*nb1,ang2[2][1]*nb1,0)
                    B2 = (ang2[3][0]*nb2,ang2[3][1]*nb2,0)
                    match = [Area,theta,A1,A2,B1,B2]
                    matches.append(match)
    
    #filter the matches based on area on and theta    
    matches = sorted(matches,key=lambda a_entry: a_entry[0])
            
    def filter_matches(matches):
        print ("selecting best match based on Area and sin(theta) from {} choices".format(len(matches)))
        A = matches[0][0]
        stheta = np.sin(matches[0][1])
        best  = matches.pop(0)
        ftol = 0.001
        for x in matches:
            if x[0] <= A + ftol and np.sin(x[1]) > stheta:
                A = x[0]
                stheta = np.sin(x[1])
                best = x
        return best
    best = filter_matches(matches)    
    return best
                    
def plot_slab_axes(slab1,slab2):
    ax = plt.axes()
    def plot_arrow(Offset,vec,ax,c):
        ax.arrow(Offset,Offset,vec[0],vec[1],head_width=0.05,head_length=0.1,fc=c,ec=c)
        ax.set_xlim((-40,40))
        ax.set_ylim((-40,40))            
        return ax

    ax = plot_arrow(1,slab1.cell[0],ax,"b")
    ax = plot_arrow(1,slab1.cell[1],ax,"k")
    ax = plot_arrow(0,slab2.cell[0],ax,"r")
    ax = plot_arrow(0,slab2.cell[1],ax,"g")
    plt.show()

def cut_cell(atoms,a1,a2,a3,origin=(0,0,0)):
    tol = 10**-7
    sp = atoms.get_scaled_positions()

    for i,x in enumerate(sp):
        for j,y in enumerate(x):
            y %= 1
            if y >= 1 - tol:
                sp[i][j] = tol
            elif y <= tol:
                sp[i][j] = tol
            
    atoms.set_scaled_positions(sp)
    atoms2 = ase.build.cut(atoms,a1,a2,a3,origo=origin,tolerance=10**-3*tol)
    
    atom_density_check(atoms,atoms2)    
    return atoms2

def number_density(slab):
    N = len(slab.positions)
    a1 = slab.cell[0]
    a2 = slab.cell[1]
    a3 = slab.cell[2]
    V = abs(np.dot(a3,np.cross(a1,a2)))
    return N*1.0/V

def RH(slab):
    #swaps a1 and a2 to make the slab right handed
    a = [line for line in slab.cell]
    fp = slab.get_scaled_positions()
    fp2 = [[u[1],u[0],u[2]] for u in fp]
    if np.dot(a[2],np.cross(a[0],a[1])) < 0:
        t = a[0];    a[0]= a[1];    a[1] = t;
    slab.cell = a
    slab.set_scaled_positions(fp2)
    return slab

def invert_axis(slab,axis):
    #flips the sign of one of the slabs axis
    #scaled coords adjusted accordingly
    sp = slab.get_scaled_positions()
    sp2 = []
    for line in sp:
        l2 = [x for x in line]
        l2[axis] = 1-l2[axis]
        sp2.append(l2)
    slab.cell[axis] = -slab.cell[axis]
    slab.set_scaled_positions(sp2)
    return slab

def bot_to_top(slab):
    #rotates by 180 degress then inverts and makes right handed
    R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    fp = slab.get_scaled_positions()
    cell = [np.dot(R,slab.cell[i]) for i in range(0,3)]
    slab.cell = cell
    slab.set_scaled_positions(fp)        

    #invert the z axis
    slab = invert_axis(slab,2)

    #force to be right handed
    slab = RH(slab)        
    return slab

def print_cell(slab1):
    #print out the cells for the slabs
    for line in slab1.cell:
        print(line,np.linalg.norm(line))

def atom_density_check(atoms,slab):
    #check that the atom densities haven't been changed
    ri = number_density(atoms)
    rf = number_density(slab)
    pd = abs((ri-rf)/ri*100)
    if pd > 10.0: #10**-4:
        #print ("number density of slab has changed by {:.2} percent".format(pd))
        print ("number density of slab has changed by %%%.2f"%(pd))
        #print ("exiting due to change in atom density...")

def find_commensurate_supercell(slab1,slab2,L,Lmax,Lstep,ptol):
    #loop over Ls increasing util finding a commensurate supercell
    success = False
    #L = Lstep
    print('burda Lmax=',Lmax)
    print('burda ptol=',ptol)

    while L <= Lmax and success == False:
        print ("trying with L = %.2f A"%L)
        angles2 = reasnoble_slab_angles(Lmax=L,slab=slab2)
        angles1 = reasnoble_slab_angles(Lmax=L,slab=slab1)    
        print ("comparing {} possible pairs".format(len(angles1)*len(angles2))    )
        #angles contains all pairs of cell vectors where both are shorter than Lma
        if len(angles1) > 0 and len(angles2) > 0:
            try:
                choice =  match_angles2(slab1.cell, angles1, slab2.cell, angles2, ptol=ptol, Lmax=L)
                success = True
            except:
                pass    
        if success == False:
            L += Lstep
    if success == False:
        print ("not possible within these tolerances")
        exit()                
    return choice

def slab_aligner(slab1,slab2,L,Lmax,Lstep,ptol,thickness,atoms1,atoms2):
        #find and repeat slabs as specified,
        #One should use only 1 layer for a layer convergence test.
        print('surda lmax',Lmax, L)

        choice = find_commensurate_supercell(slab1,slab2,L,Lmax,Lstep,ptol)
        crep = np.ceil(abs(thickness/np.dot(slab1.cell[2],(0,0,1)))) #This does not work properly, doesn't give the thickness desired by the user!!
        #crep=1
        #crep=  np.ceil((args.thickness/get_thickness(slab1)))
        #slab1 = cut_cell(slab1,choice[2],choice[3],(0,0,crep))
        slab1 = cut_cell(slab1,choice[2],choice[3],(0,0,1))
        slab1 = square_slab(slab1)
        crep = np.ceil(abs(thickness/np.dot(slab2.cell[2],(0,0,1))))
        #crep=1
        #crep=  np.ceil((args.thickness/get_thickness(slab2)))
        #slab2 = cut_cell(slab2,choice[4],choice[5],(0,0,crep))
        slab2 = cut_cell(slab2,choice[4],choice[5],(0,0,1))    
        slab2 = square_slab(slab2)

        #rotate slab2 so that it is alligned with slab1
        ase.build.rotate(slab2,slab1.cell[2],slab2.cell[2],slab2.cell[0],slab1.cell[0])

        #confirm that atom densities are the same as at the start
        atom_density_check(atoms1,slab1)
        atom_density_check(atoms2,slab2)
        
        return slab1,slab2

#NOT NEEDED PROBABLY:
"""
       #move this to slab aligner.
        if args.creps1: creps1=args.creps1
        else:                 creps1= int((args.thickness/get_thickness(slab1))) #SLAB1 is not defined yet.
        if creps1==0:creps1=1
        if args.creps2: creps2=args.creps2
        else:                 creps2= int((args.thickness/get_thickness(slab2)))
        if creps2==0:creps2=1

        #slab1=slab1.repeat((1,1,creps1));         slab2=slab2.repeat((1,1,creps2))

        slab1 = make_slab(miller1,atoms1,repeat=(1,1,creps1),square=False) #TODO:replace with creps
        slab2 = make_slab(miller2,atoms2,repeat=(1,1,creps2),square=False)
        """

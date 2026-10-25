"""Slab construction and alignment: cutting slabs from bulk structures (make_slab), symmetric slabs (permute_surface),
finding a commensurate supercell of two slabs (slab_aligner) and stacking them into an interface (make_interface)."""
import math
from sys import stdout

import ase
import ase.build
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from .log import debug, say, warn
from .symmetry import symmetry_rotations

#Surface creation and alignment tool by James P. Darby.
#This is the former join8.py

def calc_R1(x1,y1):
    #gives a rotation matrix that will rotate x1 onto y1
    Id = np.array([[1,0,0],[0,1,0],[0,0,1]])
    u = np.cross(x1,y1)

    tol = 0.001
    if np.linalg.norm(u) < tol*np.linalg.norm(x1)*np.linalg.norm(y1):
            if np.dot(x1,y1) > 0:
                    return Id
            #antiparallel: rotate by 180 degrees about an axis perpendicular to x1 (previously no rotation was done, so
            #slabs with the surface normal along -z kept it and the second slab was stacked below the first one)
            axis = np.cross(x1,[1,0,0])
            if np.linalg.norm(axis) < tol*np.linalg.norm(x1): axis = np.cross(x1,[0,1,0])
            axis = axis/np.linalg.norm(axis)
            return 2*np.outer(axis,axis) - Id

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

def orient_up(slab, wrap=True):
    """Orient a slab so that the surface normal (a x b) and c point along +z, keeping a right-handed cell.

    If c points down after rotating the normal to +z, -c (an equivalent lattice vector) is used; the atoms keep their
    Cartesian positions and are wrapped along c (wrap=True, periodic cells as from make_slab) or shifted into the cell
    (wrap=False, slabs with vacuum)."""
    slab = rotate_slab(slab)
    if slab.cell[2][2] < 0:
        cell = slab.cell[:].copy()
        cell[2] = -cell[2]
        slab.set_cell(cell, scale_atoms=False)
        sp = slab.get_scaled_positions(wrap=False)
        if wrap:
            sp[:, 2] %= 1.0
        else:
            sp[:, 2] -= np.floor(sp[:, 2].min())
        slab.set_scaled_positions(sp)
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
        d[int(zero_inds[0][0])] = 1

        #construct e from the other 2 vectors
        s = set((0,1,2))
        s.remove(int(zero_inds[0][0]))
        H = s.pop()
        K = s.pop()
        lm = lcm(miller[H],miller[K])
        e = [0,0,0]
        e[H] = lm/miller[H]
        e[K] = -lm/miller[K]
        
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

def permute_surface(bulk_ase, miller_plane, minimum_thickness, vacuum, method="pymatgen"):
    """Creates all unique symmetric surface slabs (both surfaces equivalent; generally non-stoichiometric).
    Args:
        bulk_ase (ase.Atoms): The bulk structure in ASE format.
        miller_plane (tuple): The Miller indices of the surface.
        minimum_thickness (float): The minimum thickness of the slab.
        vacuum (float): The vacuum size to add to the slab. 
        method (str): 'pymatgen' (SlabGenerator.nonstoichiometric_symmetrized_slab, default) or 'internal'
            (INTERFACER's own implementation, see symmetric_slabs).
    Returns:
        list: A list of unique, symmetric surfaces in ASE format."""
    if method == "internal":
        return symmetric_slabs(bulk_ase, miller_plane, minimum_thickness, vacuum)
    if method != "pymatgen":
        raise ValueError("permute_surface: method must be 'pymatgen' or 'internal', got '%s'" % method)

    position_accuracy = 5

    # convert ase bulk to pymatgen structure
    bulk =  AseAtomsAdaptor.get_structure(bulk_ase) 

    # create supercell, which the number of atoms of the slabs will be compared to
    # supercell = bulk.copy()
    # supercell = supercell.make_supercell([1,1,2])

    # create slab generator objects with bulk structure and supercell
    slabgen = SlabGenerator(bulk, miller_index=miller_plane, min_slab_size=minimum_thickness*2, min_vacuum_size=vacuum, center_slab=True, primitive=False)
    # slabgen_super = SlabGenerator(supercell, miller_index=miller_plane, min_slab_size=minimum_thickness*2, min_vacuum_size=vacuum, center_slab=True)
    slabs = slabgen.get_slabs()

    # create symmetric slabs
    # need to use the supercell slab with the bulk SlabGeneratore object 
    # to ensure that the number of atoms in the non-symmetric slab is large enough
    surf_list = []
    for i, slab in enumerate (slabs):
        sym_slabs = slabgen.nonstoichiometric_symmetrized_slab(slab)
        for j, sym_slab in enumerate(sym_slabs):
            if sym_slab.is_symmetric():
                sym_slab_ase = standardise_slab(AseAtomsAdaptor.get_atoms(sym_slab))
                # check that slab has the minimum thickness (atomic extent along the surface normal;
                # the c-vector of pymatgen slabs is not necessarily perpendicular to the surface)
                if slab_thickness(sym_slab_ase) < minimum_thickness:
                    continue
                # check if the slab is unique
                if check_surf_unique(surf_list, sym_slab_ase, position_accuracy) > 0:
                    continue
                surf_list.append(sym_slab_ase)
    if len(surf_list) == 0:
        say("Found no symmetric structures for this Miller plane.\n");stdout.flush()
    else:
        say(f"Found {str(len(surf_list))} unique, symmetric surfaces for {str(miller_plane)} \n");stdout.flush()
        
    return surf_list

def standardise_slab(slab):
    """Bring a slab (e.g. from pymatgen or read from a file) into the orientation used by INTERFACER:
    surface normal (a x b) along z, so that a and b lie in the xy-plane, and a right-handed cell with a3.z > 0.
    Only the chemical symbols, positions, cell and pbc are kept (pymatgen adds string arrays such as
    'bulk_wyckoff' that break the stacking/extxyz output)."""
    slab = Atoms(symbols=slab.get_chemical_symbols(), positions=slab.get_positions(), cell=slab.get_cell(), pbc=True)
    slab = rotate_slab(slab)
    if np.dot(slab.cell[2],np.cross(slab.cell[0],slab.cell[1])) < 0:
        slab = flip_handedness(slab)
    return orient_up(slab, wrap=False) #surface normal and c along +z (vacuum slab: atoms shifted, not wrapped)

def atomic_planes(slab, tol=0.3):
    """Group the atoms of a slab into planes along the surface normal (a x b).
    Atoms whose heights differ by less than tol (A) from the lowest atom of a plane belong to that plane.
    Returns (list of index arrays from bottom to top, plane heights)."""
    normal = np.cross(slab.cell[0], slab.cell[1])
    normal = normal / np.linalg.norm(normal)
    h = slab.positions @ normal
    order = np.argsort(h)
    planes, heights = [], []
    for i in order:
        if planes and h[i] - heights[-1] < tol:
            planes[-1].append(int(i))
        else:
            planes.append([int(i)]); heights.append(h[i])
    return [np.array(p) for p in planes], np.array(heights)


def is_symmetric_slab(slab, symprec=1e-3):
    """True if the two surfaces of a slab are equivalent, i.e. its symmetry contains an operation that maps the surface
    normal onto itself reversed (inversion, a mirror plane parallel to the surface or a two-fold axis in the surface)."""
    rot = symmetry_rotations(slab, symprec=symprec)
    if rot is None:
        return False
    return bool(np.any(np.asarray(rot)[:, 2, 2] == -1))


def symmetric_slabs(bulk, miller, minimum_thickness, vacuum, plane_tol=0.3, symprec=1e-3, max_extra=2):
    """INTERFACER's own generator of symmetric surface slabs (no pymatgen).

    A thick slab is cut along the Miller plane (make_slab) and its atoms are grouped into planes. Every plane within one
    bulk repeat along the normal is tried as bottom termination; for each, the slab is extended plane by plane until it
    is at least `minimum_thickness` thick and both surfaces are equivalent (is_symmetric_slab), up to `max_extra` further
    bulk repeats. Unique slabs (check_surf_unique) are returned with `vacuum` on both sides, surface normal along z.
    The slabs are in general non-stoichiometric: their stability is compared with surface formation energies.
    """
    one = make_slab(tuple(miller), bulk, repeat=(1, 1, 1))
    planes1, _ = atomic_planes(one, plane_tol)
    d_rep = abs(np.dot(one.cell[2], np.cross(one.cell[0], one.cell[1]) / np.linalg.norm(np.cross(one.cell[0], one.cell[1]))))
    nrep = int(np.ceil(minimum_thickness / d_rep)) + max_extra + 2
    thick = make_slab(tuple(miller), bulk, repeat=(1, 1, nrep))
    planes, heights = atomic_planes(thick, plane_tol)
    n_per_rep = len(planes1)
    found = []
    for i in range(min(n_per_rep, len(planes))):
        for j in range(i + 1, len(planes)):
            if heights[j] - heights[i] < minimum_thickness:
                continue
            if heights[j] - heights[i] > minimum_thickness + (max_extra + 1) * d_rep:
                break
            idx = np.concatenate(planes[i:j + 1])
            cand = standardise_slab(thick[idx])
            cand.center(vacuum=vacuum, axis=2)
            if is_symmetric_slab(cand, symprec):
                if check_surf_unique(found, cand, 5) == 0:
                    found.append(cand)
                break  # the thinnest symmetric slab for this termination
    if found:
        say("Found %d unique, symmetric surfaces for %s (internal generator)\n" % (len(found), str(tuple(miller))))
    else:
        say("Found no symmetric structures for this Miller plane (internal generator).\n")
    return found


def slab_thickness(slab):
    """Thickness of a slab (in A), i.e. the extent of the atomic positions along the surface normal (a x b)."""
    normal = np.cross(slab.cell[0], slab.cell[1])
    normal = normal/np.linalg.norm(normal)
    proj = np.dot(slab.positions, normal)
    return proj.max() - proj.min()

def _rotation_set(rotations):
    return sorted(tuple(np.asarray(r).flatten()) for r in rotations)

def check_surf_unique(surf_list, surf, position_accuracy):
    """Checks whether a surface is already present in a list of surfaces.
    Two surfaces are considered equivalent if they have the same chemical formula
    and the same set of symmetry rotations (as determined by spglib).
    Args:
        surf_list (list): List of existing symmetric surfaces to compare to.
        surf (ase.Atoms): The surface to check for uniqueness.
        position_accuracy (int): The accuracy for the get_symmetry spglib function (symprec=10^-position_accuracy).
    Returns:
        int: 1 if an equivalent surface is already in surf_list (i.e. a duplicate), 0 if it is unique."""
    
    symprec=1*10**-(position_accuracy)
    if len(surf_list) == 0:
        return 0
    for i in range(len(surf_list)):
        if surf_list[i].get_chemical_formula(
            mode="metal", empirical=False
        ) == surf.get_chemical_formula(mode="metal", empirical=False):
            new_rot = symmetry_rotations(surf, symprec=symprec)
            old_rot = symmetry_rotations(surf_list[i], symprec=symprec)
            if new_rot is not None and old_rot is not None and _rotation_set(old_rot) == _rotation_set(new_rot):
                return 1
            else:
                continue
    return 0

##########################################################################################################

def make_slab(miller, atoms,repeat=(1,1,1),square=False):
    #construct slab
    d,e,f,origin = get_cut_vectors(miller, atoms.cell)
    atoms = wrap_coords(atoms.copy()) #do not modify the input structure
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
    return orient_up(slab, wrap=True) #surface normal and c along +z (the slab cell is periodic along c)

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
            theta = np.arccos(np.clip(ctheta,-1.0,1.0))
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
                
            Area = abs(v1[0]*v2[1]-v1[1]*v2[0]) #2D cross product (np.cross of 2D vectors is removed in numpy 2)
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
        debug("selecting best match based on Area and sin(theta) from {} choices".format(len(matches)))
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
    atoms = atoms.copy() #do not modify the input structure
    sp = atoms.get_scaled_positions(wrap=False)
    sp %= 1.0 #wrap into [0,1) (previously only a local copy was wrapped)
    sp[(sp >= 1 - tol) | (sp <= tol)] = tol
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
    if np.dot(a[2],np.cross(a[0],a[1])) < 0:
        #only swap the scaled positions when the cell vectors are swapped
        t = a[0];    a[0]= a[1];    a[1] = t;
        fp = [[u[1],u[0],u[2]] for u in fp]
    slab.cell = a
    slab.set_scaled_positions(fp)
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
        warn("number density of slab has changed by %%%.2f"%(pd))
        #print ("exiting due to change in atom density...")

def find_commensurate_supercell(slab1,slab2,L,Lmax,Lstep,ptol):
    #loop over Ls increasing util finding a commensurate supercell
    success = False
    #L = Lstep
    choice = None

    while L <= Lmax and success == False:
        debug("trying with L = %.2f A"%L)
        angles2 = reasnoble_slab_angles(Lmax=L,slab=slab2)
        angles1 = reasnoble_slab_angles(Lmax=L,slab=slab1)    
        debug("comparing {} possible pairs".format(len(angles1)*len(angles2))    )
        #angles contains all pairs of cell vectors where both are shorter than Lma
        if len(angles1) > 0 and len(angles2) > 0:
            try:
                choice =  match_angles2(slab1.cell, angles1, slab2.cell, angles2, ptol=ptol, Lmax=L)
                success = True
            except IndexError: #no matching cell pairs found for this L (empty list of matches)
                pass    
        if success == False:
            L += Lstep
    if success == False:
        raise RuntimeError("No commensurate supercell found with L <= %.2f A and %.1f%% tolerance. "
                           "Try increasing -msd (max slab dimension) or -pt (percentage tolerance)."%(Lmax,ptol))
    return choice

def slab_aligner(slab1,slab2,L,Lmax,Lstep,ptol,thickness,atoms1,atoms2):
        #find and repeat slabs as specified,
        #One should use only 1 layer for a layer convergence test.
        #NOTE: 'thickness' is currently not used here; the slab thickness is set by the number of
        #repetitions in c (-cr1/-cr2) when the slabs are created (or by -th for symmetric slabs).
        #The input slabs are not modified.

        slab1_in=slab1; slab2_in=slab2
        choice = find_commensurate_supercell(slab1,slab2,L,Lmax,Lstep,ptol)
        slab1 = cut_cell(slab1,choice[2],choice[3],(0,0,1))
        slab1 = square_slab(slab1)
        slab2 = cut_cell(slab2,choice[4],choice[5],(0,0,1))    
        slab2 = square_slab(slab2)

        #rotate slab2 so that it is alligned with slab1
        ase.build.rotate(slab2,slab1.cell[2],slab2.cell[2],slab2.cell[0],slab1.cell[0])

        #confirm that atom densities are the same as at the start
        #(compared with the input slabs; the bulk (atoms1/2) cannot be used as reference for slabs with vacuum padding)
        atom_density_check(slab1_in,slab1)
        atom_density_check(slab2_in,slab2)
        
        return slab1,slab2



def prepare_cell_for_vasp(atoms):
        """Make a cell safe for VASP: right-handed (positive triple product a.(b x c); otherwise a and b are swapped) and
        no negative fractional coordinates along c (atoms are shifted into the cell). Returns the (possibly modified) atoms."""
        changed=[]
        if np.linalg.det(atoms.cell[:])<0:
                atoms=flip_handedness(atoms); changed.append("made right-handed (a and b swapped)")
        sp=atoms.get_scaled_positions(wrap=False)
        if sp[:,2].min()<-1e-8 or sp[:,2].max()>=1.0:
                pbc_c=bool(atoms.pbc[2])
                if sp[:,2].max()-sp[:,2].min()<1.0:
                        atoms.translate(-atoms.cell[2]*np.floor(sp[:,2].min()))
                        changed.append("shifted along c (no negative z coordinates)")
                elif pbc_c:
                        atoms.wrap(); changed.append("wrapped into the cell")
        if changed: debug("prepare_cell_for_vasp: cell "+", ".join(changed))
        return atoms

def flip_cell(interface): #TODO: check if the resulting cross (triple) product of the resulting cell vectors is positive (whhich a an issue for VASP). THen it should not flip the cell
        #Flip a cell that extends towards -z (VASP cannot work with negative z coordinates).
        cell=interface.get_cell()
        cell[2][2]=-cell[2][2]
        interface.set_cell(cell,scale_atoms=0)
        interface.translate([0.,0.,cell[2][2]])
        return interface

def make_interface(slab1,slab2,sep,vac,reorder=True,check_clashes=True):
        """Stack two aligned slabs (slab1 at the bottom) into an interface.

        Args:
            slab1, slab2 (ase.Atoms): slabs with (nearly) the same in-plane cell, e.g. from slab_aligner.
            sep (float): gap (A) along the surface normal between the topmost atom of slab1 and the lowest atom of slab2.
            vac (float): vacuum padding (A) on both sides along c. With vac=sep/2 the periodic images form a second,
                equivalent interface.
            reorder (bool): sort the atoms by element (otherwise the atoms of slab1 come first).
            check_clashes (bool): warn if atoms are too close (see analysis.find_clashes).
        Returns:
            ase.Atoms: the interface (the input slabs are not modified). The in-plane cell is the average of both cells.

        NOTE: ase.build.stack(distance=sep) is not used: there 'distance' is the minimum interatomic distance, obtained by
        also translating the slabs laterally and scaling the cells, which gave clashing layers at small separations.
        """
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
                debug("Cell extends towards -z direction, VASP cannot work with -ve z coords, fixing it...");stdout.flush()
                interface=flip_cell(interface)
        if check_clashes:
                from .analysis import warn_clashes
                warn_clashes(interface,"the interface (sep=%.2f A)"%sep)
        return interface

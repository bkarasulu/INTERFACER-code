"""Analysis tools: surface area, misfit, formula units, primitive cells, RDFs, surface formation energies (SFE)
and chemical potentials."""
import math
import warnings

import ase
import ase.build
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from .log import debug, say, warn
from .symmetry import primitive_cell, spacegroup


def get_thickness(atoms): #replace cell[2][2] or vol/surf_area (better)
        zcoords=[at.position[2] for at in atoms]
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
        except: warn("Error in fu determination, fu set to 1");   fu=1
    
        return fu        

def find_prim(atoms,tol=1e-4):#using SPGlib find primitive cell of a given atoms object.
        debug("Space group of the given cell using tolerance=%f: %s"%(tol,spacegroup(atoms,symprec=tol)))
        prim = primitive_cell(atoms, symprec=tol)
        if prim is None:
                warn("find_prim: spglib could not find a primitive cell (tol=%g), using the input cell."%tol)
                return atoms.copy()
        lattice, scaled_positions, numbers = prim
        #lattice, scaled_positions, numbers = standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=tol)
        atoms2=Atoms(symbols=numbers,cell=lattice,scaled_positions=scaled_positions,pbc=True)
        return atoms2


def interface_energy(gamma1, gamma2, W_ad):
        """Interface (formation) energy from the Dupre relation: gamma_int = gamma_1 + gamma_2 - W_ad (same units)."""
        return gamma1 + gamma2 - W_ad

def sfe_vs_chempot(slab, bulk, chem_pots, element, partner, dmu):
        """Surface formation energy (eV/nm^2) of a slab as a function of the chemical potential of one element.

        mu_element = chem_pots[element] + dmu; to keep the bulk in equilibrium (sum_i n_i mu_i = E_bulk), the chemical
        potential of `partner` changes by -dmu * n_element / n_partner (n: numbers of atoms in the bulk).
        Args:
            slab, bulk (ase.Atoms): with energies.
            chem_pots (dict): reference chemical potentials of all elements of the bulk (eV).
            element, partner (str): the varied element and the compensating one.
            dmu (array): changes of mu_element (eV).
        Returns:
            numpy array of SFEs (eV/nm^2), one per value of dmu.
        """
        sym=bulk.get_chemical_symbols()
        if element not in chem_pots or partner not in chem_pots:
                raise ValueError("sfe_vs_chempot: %s and %s need reference chemical potentials (have %s)"%(element,partner,sorted(chem_pots)))
        if element==partner: raise ValueError("sfe_vs_chempot: element and partner must differ")
        ratio=sym.count(element)/float(sym.count(partner))
        out=[]
        for d in np.atleast_1d(dmu):
                cp=dict(chem_pots); cp[element]+=d; cp[partner]-=d*ratio
                out.append(calculate_SFE(slab,bulk,chem_pots=cp))
        return np.array(out)

def find_clashes(atoms, factor=0.6):
        """Pairs of atoms closer than factor*(r_cov,i + r_cov,j) (periodic images included).
        Returns a list of (i, j, distance, threshold) with i < j, sorted by distance."""
        from ase.data import covalent_radii
        from ase.neighborlist import neighbor_list
        r=covalent_radii[atoms.numbers]
        if len(atoms)==0: return []
        cutoff=float(2*factor*r.max())
        i,j,d=neighbor_list('ijd',atoms,cutoff)
        out=[]
        for a,b,dist in zip(i,j,d):
                thr=factor*(r[a]+r[b])
                if a<b and dist<thr: out.append((int(a),int(b),float(dist),float(thr)))
        return sorted(out,key=lambda x:x[2])

def warn_clashes(atoms, label="structure", factor=0.6):
        """Print a warning if atoms clash (see find_clashes); returns the clashes."""
        cl=find_clashes(atoms,factor)
        if cl:
                a,b,dist,thr=cl[0]
                warn("WARNING: %d atom pair(s) too close in %s, e.g. %s%d-%s%d at %.2f A (< %.2f A = %.1f x sum of covalent radii). "
                     "Check the separation (-sep) and the slab terminations."%(len(cl),label,atoms[a].symbol,a,atoms[b].symbol,b,dist,thr,factor))
        return cl

def is_stoichiometric(slab, bulk, tol=1e-6):
        """True if the slab has the composition of the bulk (same elements in the same ratios)."""
        els=sorted(set(bulk.get_chemical_symbols()))
        if sorted(set(slab.get_chemical_symbols()))!=els: return False
        nb=np.array([bulk.get_chemical_symbols().count(e) for e in els],float)
        ns=np.array([slab.get_chemical_symbols().count(e) for e in els],float)
        return bool(np.allclose(ns/ns.sum(), nb/nb.sum(), atol=tol))

def get_interface_energy(slab1,slab2,Ebulk1,Ebulk2,dist=1.0,convLayers=False,vac_slab=6.0,vac_int=1.0,calc=None):
        """Single-point surface and adhesion energies for two (already aligned) slabs.

        Not used by the main workflow (which does the same step by step), kept for scripting.
        Args:
            slab1, slab2 (ase.Atoms): aligned slabs. A calculator is taken from `calc`, or else from slab1 (or slab2).
            Ebulk1, Ebulk2 (float): bulk energies per atom (eV/atom).
            dist (float): separation between the slabs in the interface (A).
            convLayers (bool): not supported any more (use conv_layers in the main script).
            vac_slab, vac_int (float): vacuum padding (A, on each side) for the slabs and the interface.
        Returns:
            Eslab1, Eslab2 (eV), Ws1, Ws2 (eV/nm^2), interface (ase.Atoms with W_ad in interface.info['W_ad'], eV/nm^2)
        """
        if convLayers:
                raise NotImplementedError("get_interface_energy: layer convergence is done by conv_layers() in interfacer.py")
        if calc is None: calc = slab1.calc if slab1.calc is not None else slab2.calc
        if calc is None: raise ValueError("get_interface_energy: no calculator given or attached to the slabs.")

        energies=[]
        for slab in (slab1,slab2):
                s=slab.copy(); s.center(vacuum=vac_slab, axis=2); s.calc=calc
                energies.append(s.get_potential_energy())
        Eslab1,Eslab2=energies

        Ws1=(Eslab1-len(slab1)*Ebulk1)/2/surf_area(slab1)/0.01 #A2 to nm2
        Ws2=(Eslab2-len(slab2)*Ebulk2)/2/surf_area(slab2)/0.01 #A2 to nm2

        s1=slab1.copy(); s2=slab2.copy()
        s1.center(vacuum=0, axis=2); s2.center(vacuum=0, axis=2)
        int1=ase.build.stack(s1, s2, axis=2, maxstrain=None, distance=dist,cell=None,reorder=True)
        int1.center(vacuum=vac_int,axis=2)
        int1.calc=calc
        Eint=int1.get_potential_energy()

        #Work of adhesion (positive for a bound interface).
        Wad=(Eslab1+Eslab2-Eint)/surf_area(int1)/0.01 #A2 to nm2
        int1.info['W_ad']=Wad
        say('Wsurf 1: %.2f eV/nm^2, Wsurf 2: %.2f eV/nm^2, W_ad: %.2f eV/nm^2'%(Ws1,Ws2,Wad))
        return Eslab1,Eslab2,Ws1,Ws2,int1

################# functions to determine surface formation energies of symmetric surfaces ###################################

def calculate_chem_pots(bulk, decomps):
    """Calculate the chemical potential of each species based on the decomposition products.
    Args:
        slab (ase.Atoms): The slab structure.
        decomps (list) : A list of the optimised decomposition products, each as an ase.Atoms object.
    Returns:
        np.ndarray: An array of chemical potentials for each element in the bulk structure.
    """
    toa_bulk = sorted(set(bulk.symbols))
    noa_bulk = [bulk.get_chemical_symbols().count(symbol) for symbol in toa_bulk]
    e_bulk = bulk.get_potential_energy() #in eV (same energy definition as in calculate_SFE)

    noa_matrix = np.zeros((len(decomps)+1, len(toa_bulk)))
    noa_matrix[-1] = noa_bulk
    energies = np.zeros(len(decomps)+1) 
    energies[-1] = e_bulk

    for i, decomp in enumerate(decomps):
        toa = sorted(set(decomp.symbols))
        noa = [decomp.get_chemical_symbols().count(symbol) for symbol in toa]
        for j, el in enumerate(toa_bulk):
            ix = np.argwhere(el == np.array(toa))[:,0].tolist()
            if ix:
                noa_matrix[i,j] = noa[ix[0]]
            else:
                noa_matrix[i,j] = 0
        e = decomp.get_potential_energy() #in eV
        energies[i] = e 

    if noa_matrix.shape[0] != noa_matrix.shape[1]:
        raise ValueError("calculate_chem_pots: need exactly (number of elements - 1) = %d decomposition products, got %d."%(len(toa_bulk)-1, len(decomps)))
    try:
        cp = np.linalg.solve(noa_matrix, energies)
    except np.linalg.LinAlgError:
        raise ValueError("calculate_chem_pots: the compositions of the decomposition products (%s) and the bulk (%s) are linearly "
                         "dependent (e.g. Na3OCl = Na2O + NaCl), so the chemical potentials are not determined. Use products that "
                         "fix one limit, e.g. an elemental phase such as Na metal (Na-rich) together with NaCl."
                         % (", ".join(d.get_chemical_formula() for d in decomps), bulk.get_chemical_formula()))
    assert (np.allclose(np.dot(noa_matrix, cp), energies)), "Chemical potentials calculation failed."
    chem_pots = {toa_bulk[i]: cp[i] for i in range(len(toa_bulk))}

    return chem_pots

def calculate_SFE(slab, bulk, decomps=None, chem_pots=None, ase=False):
    """Calculate the surface formation energy of a slab structure.
    Args:
        slab (ase.Atoms): The slab structure.
        bulk (ase.Atoms): The bulk structure.
        decomps (list, optional): A list of the optimised decomposition products, each as an ase.Atoms object.
        chem_pots (dict, optional): A dictionary of chemical potentials for each element in the slab structure.
    Returns:
        float: The surface formation energy in eV/nm^2.
    """
    
    A = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))/100 # surface area in nm^2

    E_bulk = bulk.get_potential_energy() # in eV
    E_slab = slab.get_potential_energy() # in eV
    
    # get the type of atoms and the number of atoms for each type in the bulk and slab structures
    toa_bulk = sorted(set(bulk.symbols))
    toa_slab = sorted(set(slab.symbols))
    if toa_bulk != toa_slab:
        raise ValueError("The bulk (%s) and slab (%s) structures do not contain the same elements."%(toa_bulk, toa_slab))
    noa_bulk = [bulk.get_chemical_symbols().count(symbol) for symbol in toa_bulk]
    noa_slab = [slab.get_chemical_symbols().count(symbol) for symbol in toa_slab]
    # calculate the number of formula units in the bulk structure
    # (the 'formula unit' here is normalised to the least abundant element, e.g. Al4O6 -> 4 x AlO1.5)
    noa_bulk_fu = noa_bulk/np.min(noa_bulk)
    no_fu_bulk = np.min(noa_bulk)/np.min(noa_bulk_fu)
    
    # calculate number of formula units of bulk in the slab
    fu_all = np.array(noa_slab)/np.array(noa_bulk/no_fu_bulk)

    # if all formula units in the slab are the same, then the slab is stoichiometric 
    if np.allclose(fu_all, fu_all[0]):
        no_fu_bis = fu_all[0]          
        # calculate the surface formation energy
        SFE = 1/(2*A) * (E_slab - ((E_bulk/no_fu_bulk)*no_fu_bis))
    # if the slab is not stoichiometric, then we need to calculate the difference number and the chemical potentials
    else:
        is_int = np.isclose(fu_all, np.round(fu_all))
        # number of formula units of bulk in the slab (reference); the SFE does not depend on this
        # choice when the chemical potentials are consistent with the bulk energy.
        no_fu_bis = np.round(np.max(fu_all[is_int])) if np.any(is_int) else np.min(fu_all)
        # calculate the difference in the number of atoms of each type between the slab and the bulk
        diff = {}
        for i, symbol in enumerate(toa_bulk):
            diff[symbol] = noa_slab[i] - (noa_bulk[i]/no_fu_bulk) * no_fu_bis
        # check if chemical potentials or decomposition products are provided to calculate the chemical potentials
        if chem_pots is not None:
            if not isinstance(chem_pots, dict):
                raise ValueError("chem_pots should be a dictionary with chemical potentials for each element in the slab structure.")
            assert set(chem_pots.keys()) == set(toa_bulk), "The keys of chem_pots do not match the elements in the bulk structure."
            chem_pots = chem_pots 
        elif decomps is not None:
            if not isinstance(decomps, list):
                raise ValueError("decomps should be a list of optimised decomposition products, each as an ase.Atoms object.")
            assert len(decomps) == len(toa_slab) - 1, "The number of decomposition products should match the number of elements in the slab structure minus one."
            chem_pots = calculate_chem_pots(bulk, decomps) # calculate the chemical potential of each species based on the decomposition products
        elif ase:
            chem_pots = {toa_bulk[i]: 0 for i in range(len(toa_bulk))}
            warnings.warn("At least one slab is not stoichiometric; either chem_pots or decomps needs to be provided to calculate the surface formation energy." +
                          "Using zero chemical potentials for all elements in the slab structure. This may lead to incorrect results.", UserWarning)
        else:
            raise ValueError(f"Slab {getattr(slab, 'name', slab.get_chemical_formula())} is not stoichiometric; either chem_pots or decomps needs to be provided to calculate the surface formation energy.")
        # calculate the surface formation energy of a non-stoichiometric slab
        SFE = 1/(2*A) * (E_slab - ((E_bulk/no_fu_bulk)*no_fu_bis) - (sum([diff[el]*chem_pots[el] for el in toa_bulk])))
    
    return SFE

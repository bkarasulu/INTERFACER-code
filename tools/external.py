from __future__ import print_function
import sys
import numpy as np
import argparse,math, os.path,fractions,time
import matplotlib.pyplot as plt

from os import system,popen,chdir,getcwd,getenv,putenv,listdir
from re import search
from sys import exit, stdout
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


from ase.calculators.lj import LennardJones as LJ
from ase.calculators.morse import MorsePotential as MP
from ase.calculators.emt import EMT
from ase.optimize import BFGS

#Import other functions from the INTERFACER package
from tools.builder import *
#from tools.external import *
from tools.analysis import *

def handle_magmoms(atoms, magmoms):
    """
    The magamoms variable should be a list parsed by argparse in the form:
    [...] -mgm Fe 5 Nb 0.6 O 0.6 [...]
    which is then converted to a dictionary:
    d = {
        'Fe': 5.,
        'Nb': 0.6,
        'O': 0.6
        }
    """
    if magmoms is None:   return atoms
    else: print("Magnetic moments are set with ASE: %s"%magmoms)
    
    elements = magmoms[::2]
    values = magmoms[1::2]
    d = dict(zip(elements, values))
    init_mgm = []
    #for s in atoms.symbols:
    for s in atoms.get_chemical_symbols():
        if s not in elements:
            init_mgm.append(0)
        else:
            init_mgm.append(d[s])
    #print([[atoms.get_chemical_symbols()[i],init_mgm[i]] for i in range(len(init_mgm))])
    atoms.set_initial_magnetic_moments(init_mgm)
    return atoms

def set_hubU(atoms,hubU):
    #atoms.calc.set(ldau_luj={'Si': {'L': 1, 'U': 3, 'J': 0}})

    if len(hubU)>0: #is not None:
        atoms.calc.set(ldau=".True.",ldauprint=0,ldautype=2,lmaxmix=6)
        ats=sorted(hubU.keys())
        ldauu=[];ldaul=[]
        #asyms=np.unique(atoms.get_chemical_symbols())#this automatically sorts the atom order
        asyms = list(dict.fromkeys(atoms.get_chemical_symbols()))
        #print (atoms.get_chemical_symbols(),asyms)
        for at in asyms:
            if at in ats: #continue
                if len(hubU[at].keys())>1: print("WARNING: cannot set multiple U values for the same atom type (%s), setting to 0... "%at);ldauu.append(0);ldaul.append(0);continue
                #print (hubU[at].keys()[0])
                orb="".join(hubU[at].keys())#[0] indexing in dict_key format is a problem in Python3.
                ldauu.append(float(hubU[at][orb]))
                if orb=="p":ldaul.append(1)
                elif orb=="d":ldaul.append(2)
                elif orb=="f":ldaul.append(3)
            else:ldaul.append(0);ldauu.append(0)
        atoms.calc.set(ldauj=[ 0 for i in range(len(asyms)) ])
        atoms.calc.set(ldauu=ldauu,ldaul=ldaul)
        atoms.calc.set(lasph=1)#This is essential for accurate total energies and band structure calculations for f-elements (e.g. ceria), all 3d-elements (transition metal oxides), and magnetic atoms in the 2nd row (B-F atom), in particular if LDA+U or hybrid functionals or meta-GGAs are used, since these functionals often result in aspherical charge densities.
    return atoms

def get_potcar(elements, xc):
    try:    vasppp = os.environ['VASP_PP_PATH']
    except: print ("VASP_PP_PATH is not defined, but neededor VASP calculation setup. Terminating. ");exit()

    ppdir = os.path.join(vasppp, xc)

    pp = ''
    for e in elements:
        #print os.path.join(ppdir, e, 'POTCAR')
        pp += open(os.path.join(ppdir, e, 'POTCAR')).read()

    return pp

def make_potcar(xc="potpaw_PBE",elements=None,wDir='./'):
    try:    vasppp = os.environ['VASP_PP_PATH']
    except: print ("VASP_PP_PATH is not defined, but neededor VASP calculation setup. Terminating. ");exit()

    print('Making potcar using %s'%vasppp);stdout.flush();


    if elements==None:#Try to read it from POSCAR if not explicitly given.
        try: #VASP5 format.
            elements = open(wDir+"/POSCAR").readlines()[5].strip().split()
        except:
            print ('Elements not given on command line and POSCAR not found')
            sys.exit(-1)

    pp = get_potcar(elements, xc)

    p = open(wDir+"/POTCAR", 'w')
    p.write(pp);    p.close()

def vasp_continue(ifDel=0):
        fl = listdir(".")
        fl.sort()
        #print fl
        flag=0
        ind=0
        for fn in fl:
                if fn[0:3]=="RUN" :
                        ind = int(fn[-1])

        run="RUN%d"%(ind+1)
        print ("Previous data stored in %s."%run)
        system('mkdir %s'%run)
        system('cp * %s 2> /dev/null '%run)
        if ifDel: system('rm WAVECAR CHGCAR')
 
def call_vasp(atoms,calc=None, typ="sp",xc='PBE',
              name='./VASP-tmp',resDir="",
              dipolCorr=False,dipolDir='z',KPgrid=None,KPspacing=None, 
              ifPrint=False,ENCUT=0,ifManualRun=True,FixCell=False,
              FixVol=False,FixList=[],hubU={},ispin=0,restart=None,nelm=150,
              nsw=800,etol=1E-8,ediffg=-0.01,slowConv=0,ismear=0,sigma=0.01,
              passivate=None,gamma=1,algo="Fast",nosymm=0,exe=None,
              mpiexe='mpirun',nproc=None,vac=0.0,magmom=None,view=False,
              dry=False, readINCAR=1,setups='recommended'):

        #TODO: Read an external INCAR file for ENCUT, XC, KSPACING options
        if not exe: exe=getenv('VASP_COMMAND') ;# print ('bk')
        #if not exe: exe='vasp_std'
        if not exe: 
               if mpiexe=='mpirun': exe = "mpirun -n %d vasp_std > vasp.out"%(int(getenv('SLURM_NTASKS')))
               elif mpiexe=='srun': exe = "srun -n %d vasp_std > vasp.out"%(int(getenv('SLURM_NTASKS')))
        #exe="mpirun -np 128 ~/APPS/vasp.5.4.4/bin/vasp_std > vasp.out&"
        os.environ['VASP_COMMAND']=exe
        print("VASP command: ", exe) #, os.environ['VASP_COMMAND'])

        cwd=os.getcwd()


        try: ncores=int(getenv('SLURM_NTASKS'))  #ncores=int(popen('echo "$SLURM_NTASKS"',"r").read()[0:-1]) #add ones for PBS
        except:ncores=1
        if len(atoms)<20: ncores=1 #NPAR>1 can be problematic with small systems due to partioning of the bands.
 

        if calc!=None:
                #This is good for assiging the settings from a previously created Calculator object (i.e. not to repeat all the settings the second time). None of the other options will be taken into account.
                atoms.set_calculator(calc)
                #manual VASP run icin ayri bir function yap ve burda ve asagida cagir.
                chdir(calc.directory)
                E=atoms.get_potential_energy()
                chdir(cwd)
                return E,atoms


        wDir=name
        os.system('mkdir -p %s'%wDir)
        chdir(wDir)

        asyms=[]
        for at in atoms:  #To guarantee the same order as in POSCAR.
                if at.symbol not in asyms:asyms.append(at.symbol)

        if 1 and os.path.exists('OUTCAR'):
                print("A previous calculation found in %s"%name)
                print(getcwd())
                restart=True

        try: 
                calc = Vasp(atoms,restart=restart, directory="./", label='vasp', command=exe, txt=None) #ignore_bad_restart_file=False,
                print ("Previous run was read successfully, checking if converged...")
                #if Vasp.read_convergence(calc): print('Geom opt already converged...')
                #else: print('Geom opt not converged; running a continuation job...')
        except: #if restarting fails
                print("Problem with restarting from the previous run. Starting a new calculation.")
                calc = Vasp(atoms,restart=None, directory="./", label='vasp', command=exe, txt=None);restart=False

        if restart: #if restarted no need to assign the variables and use the ones read from INCAR and OUTCAR by ASE.
                import ase, ase.io
                if Vasp.read_convergence(calc):#calc.converged:#read_convergence('OUTCAR'):
                        print ("Calculation has converged, reading data from vasprun.xml/OUTCAR... ")
                        try:atoms=ase.io.read('vasprun.xml',index=-1)
                        except:atoms=ase.io.read('OUTCAR',index=-1)
                else:
                        #Standard ASE impletation for VASP-restart does not read the last geom for some reason, this is a simple trick to get the CONTCAR and same settigns from the prevoious run.
                        print ("Not converged, restarting from last point")
                        # #atoms=ase.io.read('CONTCAR') #no energy so must be repeated.
                        # #atoms.set_calculator(calc)
                        # vasp_continue() #copies prev. run  files into RUNX folder.
                        # #atoms.write('POSCAR',format='vasp',vasp5=1)
                        # #calc = Vasp(atoms,restart=restart, directory="./", label='vasp', command=exe, txt=None) #ignore_bad_restart_file=False,
                        # calc = Vasp(restart=True)
                        # #atoms.set_calculator(calc)
                        # atoms = calc.get_atoms()

                        calc=Vasp()
                        calc.read_incar(filename='INCAR')
                        if os.path.exists("./OUTCAR"):   vasp_continue()
                        atoms=ase.io.read("POSCAR",format="vasp")
                        #atoms=ase.io.read(fname)

                        calc.set(xc=xc)
                        #setups='recommended'
                        #setups='minimal'
                        calc.set(setups=setups)
                        calc.directory="."#cdir
                        atoms.set_calculator(calc)


                E=atoms.get_potential_energy()
                chdir(cwd)
                return E,atoms


        #Read INCAR file if requested/provided:
        # if readINCAR:
        #         try:
        #              print('Reading ./INCAR '); calc.read_incar(filename='INCAR')
        #         except:
        #               print('Reading ../INCAR '); calc.read_incar(filename='../INCAR')
        #         else:
        #               print("INCAR file can't be read from ./ or ../")
 
        if typ.lower()=='geom':IBRION=2 #CG algo
        elif typ.lower()=="sp": IBRION=-1;nsw=0

        if FixCell:isif=2 #fixed cell optimization
        elif FixVol:isif=4 #cell dimensions can vary but total volume is kept fixed (good for keeping the vacuum padding).
        else:isif=3 #full cell relaxation.
        #if vac<=2:isif=3 #this must be set to relax the cell only in the vacuum direction (i.e. c-axis), but not possible in VASP.

        #calc.initialize()
        ncore=1
        npar=int(np.sqrt(ncores))
        if ncores%npar!=0 or npar==1:npar=8
        if len(atoms)<20:ncore=16 #but NPAR should be unset/commented, this is neded for the Sub-Space-Matrix is not hermitian in the Davidson algorithm.
        ncore=16 #only 8 works for small systems like 2-atom graphene.
        calc.set(xc=xc.lower(),encut=ENCUT,prec="Normal",ediff=etol, ediffg=ediffg,sigma=sigma,reciprocal=0,algo=algo,ispin=ispin,lreal="AUTO",nelm=nelm,ibrion=IBRION,gamma=gamma,isif=isif,nsw=nsw,ismear=ismear,npar=npar) #ncore=ncore)#,npar=npar)
        calc.set(nwrite=1, lcharg=0 , lwave=0) #Wavecar could be used for next run, specially for ISPIN=2
        if ispin==2:calc.set(lwave=1) 

        calc.set(lmaxmix=6,lasph=1) #need when there is a trans metal.

        if nosymm: calc.set(isym=0)
        #TODO add support for reading INCAR file if exists.
        #calc.set(setups='recommended')


        if KPgrid: #KPgrid given
                #calc.cell.kpoint_mp_grid = KPgrid #def='1 1 1'
                kpgrid=[int(x) for x in KPgrid.split()]
                import ase.dft.kpoints
                kpts = ase.dft.kpoints.monkhorst_pack(kpgrid) #+  [1./2./kpgrid[0],1./2./kpgrid[1],1./2./kpgrid[2]] #for placing Gamma point in the center.  #CHECK if works CORRECTLY!!
                calc.set(kpts=kpts)
                if KPgrid=='1 1 1': 
                        print ("VASP gamma-only implementation (vasp_gam) is activated.")
                        cmd=exe.split('vasp_')[0]+"vasp_gam >vasp.out"
                else:
                        print ("Multiple KPOINTS: VASP standard implementation (vasp_std) is activated.")
                        cmd=exe.split('vasp_')[0]+"vasp_std >vasp.out"

                calc.command=cmd
                #print(cmd)

        elif KPspacing: 
                calc.set(kspacing=KPspacing,kgamma=gamma)#calc.cell.kpoints_mp_spacing = str(KPspacing) #default=0.05 2*pi/A
                print ("Multiple KPOINTS: VASP standard implementation (vasp_std) is activated.")
                cmd=exe.split('vasp_')[0]+"vasp_std >vasp.out"
                calc.command=cmd
        
        else:
               print('call_vasp: Either KPspacing or KPgrid keyword must be defined! Terminating... ')
               exit()

        a,b,c,_,_,_=atoms.cell.cellpar()
        if a >35 or b>35 or c>35: calc.set(amin=0.01) #gor super large cells to avoid the charge sloshing along the long lattice vector. 
        if slowConv:
                calc.set(amix=0.1,bmix=3.0,lmaxmix=6)

        if (dipolCorr and dipolCorr.lower()!='none'):
                calc.set(ldipol = ".TRUE.") #this one switches on  the corrections to the potential and thus forces
                if dipolDir=='x':idipol=1 #IDIPOL switches on the single point energy corrections.
                elif dipolDir=='y':idipol=2
                elif dipolDir=='z':idipol=3
                calc.set(idipol = idipol) #no need to set DIPOL keyword (giving the charge center, if no charged dipole in the system (i.e. total charge=0) 

                #One needs to set the DIPOL keyword (i.e. the center of the net charge in scaled/fractional coordinates), as the automated algorithm leads to terrible convergence. It should be the centre of charge (that requires the CHG file to be created and analysed to be determined), instead  the centre of mass could be used as a good approximation. Setting DIPOL helps even the veryslow convergence when the LDIPOL=true. It even affects the total spin of the system (when ISPIN=2).

                calc.set(dipol=atoms.get_center_of_mass(scaled=1)) #Does this need to be updated for continuation run or better to stick with initial value for better covnergence ??? (CHECK)


        if 1 and len(hubU)!=0: #is not None:
                #if len(hubU)>0: atoms=set_hubU(atoms,hubU)
                calc.set(ldau=".True.",ldauprint=0,ldautype=2,lmaxmix=6)
                ats=sorted(hubU.keys())
                #asyms=np.unique(atoms.get_chemical_symbols())
                #asyms=popen('head -n1 POSCAR',"r").read()[0:-1].split() #this doen't work always well.

                #if len(asyms)==0: print ("DFT-U Warning: Atomic info cannot be read from the head line of POSCAR")
                ldauu=[];ldaul=[]
                for at in asyms:
                        if at in ats: #continue
                                if len(hubU[at].keys())>1: print("WARNING: cannot set multiple U values for the same atom type (%s), setting to 0... "%at);ldauu.append(0);ldaul.append(0);continue
                                #print (hubU[at].keys()[0])
                                orb="".join(hubU[at].keys())#[0] indexing in dict_key format is a problem in Python3.
                                ldauu.append(float(hubU[at][orb]))
                                if orb=="p":ldaul.append(1)
                                elif orb=="d":ldaul.append(2)
                                elif orb=="f":ldaul.append(3)
                        else:ldaul.append(0);ldauu.append(0)
                calc.set(ldauj=[ 0 for i in range(len(asyms)) ])
                calc.set(ldauu=ldauu,ldaul=ldaul)
                calc.set(lasph=1)#This is essential for accurate total energies and band structure calculations for f-elements (e.g. ceria), all 3d-elements (transition metal oxides), and magnetic atoms in the 2nd row (B-F atom), in particular if LDA+U or hybrid functionals or meta-GGAs are used, since these functionals often result in aspherical charge densities.

        if magmom: 
                # Adding MAGMOM and ISPIN to INCAR if -mgm or --magmoms is defined.
                atoms = handle_magmoms(atoms=atoms, magmoms=magmom) 
                calc.set(ispin=2)
                

        if passivate: #compute the coordination numbers for each atom, do statistics and then add passivating (pseudo)-Hydrogen atoms. 
                #from ase.build import molecule
                
                #Preparations
                setups={'base': 'recommended'}
                cov_radii={}
                for ln in open(os.environ['HOME']+"/covalent_radii.dat",'r'):
                        x=ln.split()
                        cov_radii[x[0]]=float(x[1])

                #Get the valance electron for each atom type from corr. POTCAR file (ZVAL)


                from ase.neighborlist import NeighborList, neighbor_list
                from ase.utils import \
                    natural_cutoffs  # distance cutoff based on covalent radii.

                #ASEs neighborlist class has bugs, it counts number of coordination/neighbors for atoms located at the 0.0, 0.5 and 1.0 in fractional coordinates (a,b,c does not matter) !! Use rdf calculation instead. Actually this is a bug in natural_cutoff which cannot capture the different oxidation states of e.g. Fe. It's defined too short for capturing Fe6+.
                #natural_cutoff of ASE does not work properly for finding the orrect no of neighbours.
                cov_cutoff={}
                for asym in asyms:
                        for asym2 in asyms:
                                key=(asym,asym2)
                                if key not in cov_cutoff: cov_cutoff[key]=(cov_radii[asym]+cov_radii[asym2])*1.1
                if 0: #This does not work properly as one cannot deine the pair distances explicitly for each atom type pair, instead a radius defiend for each atom type (so same as natural_cutoffs) . USE neighbour_list instead.
                        #nl = NeighborList(cutoffs=natural_cutoffs(atoms), skin=0.3, sorted=False, self_interaction=0, bothways=1)
                        #nl = NeighborList(cutoffs=cov_cutoff, skin=0.3, sorted=False, self_interaction=0, bothways=1)
                        nl = NeighborList(cutoffs=[cov_radii[at.symbol] for at in atoms], skin=0.3, sorted=False, self_interaction=0, bothways=1)
                        nl.update(atoms)
                        #print(nl.npbcneighbors)
                        coord=[]
                        for at in atoms:   coord.append(len(nl.get_neighbors(at.index)[0]))
                else:  # bothways is on by default in this. This works well. #THIS DOES NOT WORK WELL EITHER, USE OVITO ISNTEAD (GIVES THE BEST RESULTS).
                        i,j=neighbor_list('ij',atoms,cutoff=cov_cutoff)#natural_cutoffs(atoms)) #there is a bug for determining the neighbours for atoms located at the origin. #i: index of the central atom, j: index of the neighbour.
                        coord = np.bincount(i) #Nx1 ; coutns no of occurences of each value in the input array. #one could use np.unique() for counts as well.

                        #unique, counts = np.unique(i, return_counts=True)
                        #print(unique, counts)   

                        #The dictionary of neighbour types for each atom index.
                        ntypes={}
                        tmp=[]
                        pkey=""
                        for k in range(len(i)):
                                #print(i[k],j[k])
                                key=str(i[k])
                                #if key not in ntypes: 
                                if k!=0 and key!=pkey: ntypes[pkey]=tmp;tmp=[];pkey=key#[atoms[j[k]].type]
                                elif k==0:pkey=key
                                elif k==len(i)-1: ntypes[key]=tmp
                                tmp.append(atoms[j[k]].symbol)

                        for k in sorted(ntypes.keys()): 
                                #k =int(k)
                                print (k,ntypes[k],coord[int(k)])
                        #print (sorted(ntypes))
                  
                        #unique, counts = np.unique(ntypes.values(), return_counts=True)
                        #print(unique, counts)

                print(coord,len(coord))

                #exit()
   
                #Get layers from the structure based on z coordinates. #Use 0.7 A bin_width
                zcoords=[at.position[2] for at in atoms]
                bin_width=0.5 #Looks like a reasonable binning but Determine this from the avg distances btw atoms in z direction !!  original:0.7
                n_bins=None
                r_range = np.array([min(zcoords), max(zcoords)])
                if n_bins is not None:
                        n_bins = int(n_bins)
                        if n_bins <= 0:
                                raise ValueError('n_bins must be a positive integer')
                        bins = np.linspace(r_range[0], r_range[1], n_bins)
                else:
                        bins = np.arange(r_range[0], r_range[1] + bin_width, bin_width)
                #plt.hist(zcoords,bins=bins)
                #plt.show()
                counts,bin_edges=np.histogram(zcoords,bins=bins) #the layers in terms of c-axis sepration.

                #Instead of histogram of zcoords, compare the zcoords of each atom and classify the based on being in close vicinity. More precise than histogram in determining distinct layers.
                layers={}
                tol=1e-2 #tolerance for determining if atom is in a partiular layer. in A
                #for at in atoms:
                #flag=0
                for i,zc in enumerate(zcoords):
                        at=atoms[i]
                        #if i==0: layers[zc]=[i]
                        keys=layers.keys()
                        flag=0
                        for key in keys:
                                if abs(zc-key)<tol: 
                                        layers[key].append(i);flag=1;break
                        if not flag:
                                layers[zc]=[i]

                for key in sorted(layers.keys()):                     print (key,layers[key])
                layers=[layers[key] for key in sorted(layers.keys())]
                print (layers)

                #Determine the matching top and layers for ensuring symmetric slab termination
                #Move layer determination to  a function.
                #Do also the reverse way (so matching topmost layer to bottom layer).
                tol=1e-5
                for i in range(4):
                        #flag=0
                        #at1=atoms[i]
                        lay1=layers[i]
                        for j in range(-1,-5,-1):
                                #flag=0
                                lay2=layers[j]
                                #at2=atoms
                                if len(lay1)!=len(lay2): continue #does not match
                                for k in range(len(lay1)):
                                        pos1=atoms[lay1[k]].position  ; pos2=atoms[lay2[k]].position
                                        if abs(pos1[0]-pos2[0]) > tol or abs(pos1[1]-pos2[1]) > tol: break#layers do not match.
                                #flag=1
                                print ("Layer #%d matches layer #%d"%(i,j))


                #Max occurence of coordination no for a given atom type can also be used, instead of avg coord no.
                if 1:
                    crds={};common_crd={};nty={};common_ntypes={};valance={};dangling={}
                    for aty in asyms:
                            #Determine the no of coordination for each atom.
                            crds[aty]=[coord[at.index] for at in atoms if at.symbol==aty]
                            common_crd[aty]=np.argmax(np.bincount(crds[aty]))
                            #Determine the neighbour types.
                            nty[aty]=[ntypes[str(at.index)] for at in atoms if at.symbol==aty] #Do we ned to store this??
                            unique, counts = np.unique(nty[aty], return_counts=True)
                            counts, unique =zip(*sorted(zip(counts, unique),reverse=1))
                            #print(unique, counts)
                            common_ntypes[aty]=unique[0]
                            #Get the valence electron no from the POTCAR files.
                            potfile="%s/potpaw/%s/POTCAR"%(getenv('VASP_PP_PATH'),aty)
                            with open(potfile,'r') as f:
                                    for ln in f:
                                            if  'ZVAL' in ln:  val=float(ln.split('ZVAL')[-1].split()[1]);  break
                            if val>8: val-=8
                            valance[aty]=val
                            dangling[aty]=val/common_crd[aty]
                            print (aty,common_crd[aty],common_ntypes[aty],valance[aty],dangling[aty])
                            
                    data=common_crd
                else:
                    #Get the average coord no for each atom type.
                    data={}
                    for i,at in enumerate(atoms): #Get stats on coordination no
                            #i=at.index
                            typ=at.symbol
                            if typ in data: data[typ]+=coord[i]
                            else: data[typ]=coord[i]


                    for key in asyms: #atom types; 
                            data[key]=data[key]/float(len([at for at in atoms if at.symbol==key]))
                            print (key,data[key])

                scpos=atoms.get_scaled_positions()
                undercoord=[]; overcoord=[]; normal=[]
                for i,at in enumerate(atoms):
                        if coord[i]<data[at.symbol]:undercoord.append(i)
                        elif coord[i]>data[at.symbol]:overcoord.append(i)
                        else:normal.append(i)
                        #print (i,at.symbol,at.position,scpos[i],coord[i])

                print ("Undercoordinated:",undercoord)
                print("Overcoordinated: ",overcoord)
                print("Standard coordination:",normal)
                #print(len(bin_edges))

                #Decide on which layers to passivate (top or bottom)
                if passivate.lower()=="bot":slayers=[0,1,2] #layer numbering starts from 0 
                elif passivate.lower()=="top":slayers=[-1,-2,-3]

                #Switch to layers determined earlier instead of using zcoords histogram.
                Hcnt=0
                for sl in slayers:
                        layer=layers[sl]
                        for i in layer:
                                if i not in undercoord: continue
                                at=atoms[i]
                                crd=coord[i]

                                print (at,crd, sl)

                                Hcnt+=1
                                offset=(cov_radii[at.symbol]+cov_radii['H'])*1.0
                                if passivate=='top': offset=-offset
                                atoms.extend(Atom("H",(at.position[0],at.position[1],at.position[2]-offset)))
                                #Determine the amont of missing e-.
                                missing=[];zval=0.0
                                c1=deepcopy(ntypes[str(i)])
                                c2=deepcopy(common_ntypes[at.symbol])

                                for c in c2:
                                        flag=0
                                        for cc in c1:
                                                if c ==cc: flag=1;c1.remove(cc);break
                                        if not flag: missing.append(c);zval+=dangling[at.symbol]
                                print ("Missing atoms: %s No of e- on pseudohydrogen: %.1f "%(missing,zval))
                                setups[len(atoms)-1]='H%.1f'%zval #when applied the added pseuodo-H atom is moved to the top of the atom list. 
                                #this can be done alltogether to prevent fractioning of the pseudohydrogens and limit number of atom enetries in POSCAR.
                                #print (setups)


                """
                #Using zcoords histogram.
                #Decide on which layers to passivate (top or bottom)
                if passivate.lower()=="bot":slayers=[1,2,3] #bin numbering starts from 1 
                elif passivate.lower()=="top":slayers=[len(bin_edges)-1, len(bin_edges)-2,len(bin_edges)-3]

                Hcnt=0
                for i in undercoord: #can be combined with the above atoms loop.
                        at=atoms[i]
                        crd=coord[i]
                        for j,be in enumerate(bin_edges): #Zcoord histogram.
                                if at.position[2] <be:
                                        if j  in slayers: #This check is needed for avaoiding passivation of  a defect in the middle of the slab rather than the surface, and passivating only the target surface.
                                                #TODO:check fro clashes with already-existing  atoms.
                                                print (at,crd, j)
                                                Hcnt+=1
                                                offset=(cov_radii[at.symbol]+cov_radii['H'])*1.0
                                                if passivate=='top': offset=-offset
                                                atoms.extend(Atom("H",(at.position[0],at.position[1],at.position[2]-offset)))
                                                #Determine the amont of missing e-.
                                                missing=[];zval=0.0
                                                c1=deepcopy(ntypes[str(i)])
                                                c2=deepcopy(common_ntypes[at.symbol])
                                                
                                                for c in c2:
                                                        flag=0
                                                        for cc in c1:
                                                                if c ==cc: flag=1;c1.remove(cc);break
                                                        if not flag: missing.append(c);zval+=dangling[at.symbol]
                                                print ("Missing atoms: %s No of e- on pseudohydrogen: %.1f "%(missing,zval))
                                                setups[len(atoms)-1]='H%.1f'%zval #when applied the added pseuodo-H atom is moved to the top of the atom list.
                                                #print (setups)
                                        else:
                                                print ('other surface:',at,crd, j)
                                        break
                """
                print (setups)
                calc.set(setups=setups)
                print ("\nAdded %d (pseudo)hydrogens to saturate the dangling bonds."%Hcnt)
                print(atoms)
                #atoms=ase.build.sort(atoms)
                #atoms.sort()
                if view:view(atoms)

                if dipolCorr and dipolCorr.lower()!='none':  calc.set(dipol=atoms.get_center_of_mass(scaled=1))


        atoms.set_calculator(calc)      
        

        if dry: #do not run VASP calculation.
                chdir(cwd)
                return 0.0,atoms

        #TODO: Call VASP manually for doing the geom opt and read the output for the final enerfy and geom.
        #atoms.calc.initialize(atoms) #this does not create the input files for a manual run.
        atoms.get_potential_energy()
        #atoms.calc.set(command="");atoms.get_potential_energy();atoms.calc.set(command=exe)


        #no need to run it manually as first get_energy does run the geom opt as well, depending on the IBRION.
        #system(exe) 

        import ase.io
        #atoms=ase.io.read('OUTCAR',format="vasp",index=-1)
        atoms=ase.io.read('vasprun.xml',index=-1)
        #atoms=ase.io.read(wDir+'/OUTCAR',index=-1)
        
        chdir(cwd)
        return atoms.get_potential_energy(),atoms      

def call_vasp_v2(fname='',exe=None,xc='pbe',mgms=None,hubU={},makepotcar=0,potcarxc='potpaw_PBE'): #atoms=None,

    if not exe: exe=getenv('VASP_COMMAND') ; 
    if not exe: exe='vasp_std'
    
    os.environ['VASP_COMMAND']=exe
    print("VASP command: ", exe) #, os.environ['VASP_COMMAND'])

    cwd=os.getcwd()

    seed=fname.split('.')[0]
    try:system();chdir(seed)
    except:print('Cannot change to %s directory, using %s instead'%(seed,cwd))

    print('Working dir: %s'%getcwd());stdout.flush();

    if not os.path.exists('POTCAR') or makepotcar: make_potcar(xc=potcarxc,wDir='.')

    flag=0 #whether to start a new/continuation run 
    try:
        calc = Vasp(restart=True)
        atoms = calc.get_atoms()
        print ("\nVASP run was read succesfully from OUTCAR.")
        if Vasp.read_convergence(calc): print('Geom opt already converged...')
        else:
            print('Geom opt not converged; running a continuation job...')
            flag=1

    except:
        print ("VASP run could not be read, starting a new run...")
        flag=1

    if flag:
        calc=Vasp()
        calc.read_incar(filename='INCAR')
        if os.path.exists("./OUTCAR"):   vasp_continue()
        atoms=ase.io.read("POSCAR",format="vasp")
        #atoms=ase.io.read(fname)
        #calc.set(xc="pbe",ibrion=2,setups='recommended')#TODO: POTCAR: add others or take from the user
        calc.set(xc=xc)#,ibrion=2)
        calc.directory="."#cdir
        setups='recommended'
        #setups='minimal'
        calc.set(setups=setups)
        atoms.set_calculator(calc)

    # Adding MAGMOM and ISPIN to INCAR if -mgm or --magmoms is defined.
    atoms = handle_magmoms(atoms=atoms, magmoms=mgms) 
    if len(hubU)>0: atoms=set_hubU(atoms,hubU)

    Ep = atoms.get_potential_energy() 
    chdir(cwd)

    return Ep,atoms

def call_castep(atoms,calc=None, typ="sp",PP='',wDir='./CASTEP-tmp',name='try',param='opt.param',resDir="",dipolCorr=False,dipolDir='z',KPgrid="1 1 1",KPspacing="", ifPrint=False,ifDryRun=False,ENCUT=0,ifManualRun=True,FixCell=False,FixList=[],hubU={},slowConv=0):
    #For applying constraints (FixList) atom numbering starts from 1.

        
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
            atoms.set_calculator(calc)
            #manual CASTEP run icin ayri bir function yap ve burda ve asagida cagir.
            return atoms 

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
            calc.cell.kpoint_mp_grid = KPgrid #def='1 1 1'
            kpgrid=[float(x) for x in KPgrid.split()]
            calc.cell.kpoints_mp_offset='%.4f %.4f %.4f'%(1./2./kpgrid[0],1./2./kpgrid[1],1./2./kpgrid[2]) #for placing Gamma point in the center.

    
    #calc.cell.fix_com = False
    if FixCell: calc.cell.fix_all_cell = True
    if len(FixList)!=0:
            #c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu'])
            #c = FixAtoms(indices=FixList)
            #atoms.set_constraint(c) #This only work if the CASTEP is called by ASE (not for manual runs).

            str1=[]
            i=1
            for at in FixList:
                    atom=atoms[at]
                    #for j in range(1,4):
                    str1.append("%d %s %d %.8f %.8f %.8f"%(i,atom.symbol,atom.index,1,0,0))
                    str1.append("%d %s %d %.8f %.8f %.8f"%(i+1,atom.symbol,atom.index,0,1,0))
                    str1.append("%d %s %d %.8f %.8f %.8f"%(i+2,atom.symbol,atom.index,0,0,1))
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
    elif typ.lower()=="geom":calc.Task = 'GeometryOptimization'
    
    if dipolCorr: #def: No dipole corrections. 
        if dipolCorr.upper()=="SC": calc.param.dipole_correction= "SELFCONSISTENT"
        elif dipolCorr=="static": calc.param.dipole_correction= "static"
        else: calc.param.dipole_correction= "None"
        calc.param.dipole_dir=dipolDir #options: x,y,z and a (only energy-corr)

    if slowConv:
            calc.param.mix_charge_amp=0.1
        
    
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
                    #str1="%s %s/%s"%(exe,wDir,name)
                    str1="%s %s"%(exe,name)  
                    #TODO: Add here the run3 option !
                    print("Running ",str1)

                    calc._copy_pspots=True
                    calc.initialize()

                    chdir(wDir)
                    
                    system(str1) #PSPOT missing in the folder
                    #x=parseCASTEP("%s/%s.geom"%(wDir,name),atoms=atoms)
                    task=str(atoms.calc.param.task).split()[-1]
                    print(task)
                    if task=='SinglePoint' : #use <seed>.castep file
                            x=parseCASTEP("%s.castep"%(name),atoms=atoms)
                    elif task=='GeometryOptimization': #use <seed>.geom file.
                            try:x=parseCASTEP("%s.geom"%(name),atoms=atoms) #in case no ptimisation is done (that could happen for 2D systems, where the slab is equivalent to the bulk str, so no geom opt is needed/possible.)
                            except:x=None

                            if x == None: x=parseCASTEP("%s.castep"%(name),atoms=atoms);#print ("bk",atoms)

                            if x[-2]==False: print("parseCASTEP: WARNING: Geometry optimization in %s.geom is not converged!"%name)
                    else:
                            print("parseCASTEP: ERROR: Calculation type is not supported.")
                            x=None
                    chdir("..")
                    return x
                    
            else: #CASTEP calculation is not done here. It will be called in the main script, when trying to reach the attributes, e.g. atoms.get_potential_energy().
                    return atoms
    else:
            if calc.dryrun_ok():
                    return atoms
            else:
                    print("CASTEP run: Found error in input")
                    print((calc._error))
                    return None
            
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
                                

                #Assuming it is a SP calculation (see call_CASTEP() function), so initial positions are not changed.
                
                #if len(h)==0: print("Hey !!"); h=atoms.get_cell()#.tolist()  #if no Lattice infoin castep file (not likely).
                
                #xyz=atoms.get_positions()#.tolist()
                
        if len(ids)==0: 
                print(ids);print ("parseCASTEP: WARNING: No chemical symbol info in %s."%fname); return None
        elif len(h)==0: 
                print ("parseCASTEP: WARNING: No cell info in %s."%fname); return None
        elif len(xyz)==0 and len(fract)==0: 
                print ("parseCASTEP: WARNING: No Cartesian or fractional coordinate info in %s."%fname); return None
        
                                                
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

def call_ase(atoms,opt=False,ctype=LJ,fmax=0.05,steps=200):
        #THe dfault paramters are completely irrelvant aand provided to be used in test calculations.
        #TODO: add support for user to define their own LJ/Morse params.
        if ctype.upper()=='LJ': calc=LJ()
        elif ctype.lower() in ['morse','m','mor']:calc=MP()
        elif ctype.upper()=='EMT':calc=EMT()
        else: print('ASE calculator type not supported, try Morse or LJ..');return None

        #if len(atoms)==2:atoms=atoms.repeat((2,2,2)) #if two atom system, then LJ energy is zero
        del(atoms.calc)
        atoms.set_calculator(calc)
        
        if opt:
                print('Optimising the %s structure using ASE/%s'%(atoms.get_chemical_formula(empirical=1),ctype))
                try:
                        dyn = BFGS(atoms,trajectory='OUTPUT/%s_opt.traj'%atoms.name)
                        dyn.run(fmax=fmax,steps=steps)
                        return atoms.get_potential_energy(),atoms 
                except Exception as err: 
                        print("call_ase: Error in geometry opt using ASE:  ", err)
                        return 0.0,atoms
        else:
               print('Getting SP energy using ASE/%s'%ctype)
               return atoms.get_potential_energy(),atoms 
        
#TODO: Add DFTB and GPAW!
def call_dftb(atoms):
       #This requires rigorous testing, as I  only took the example directly from ASE website.
       #https://wiki.fysik.dtu.dk/ase/ase/calculators/dftb.html
       #requires
       calc = Dftb(Hamiltonian_='DFTB',  # this line is included by default
            Hamiltonian_SCC='Yes',
            Hamiltonian_SCCTolerance=1e-8,
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_H='s',
            Hamiltonian_MaxAngularMomentum_O='p')
       #or read the 'dftb_in.hsd' file
       atoms.set_calculator(calc)
       return atoms.get_potential_energy(), atoms


# For installing GPAW, one could use:
# conda install gpaw
# which would probably require UCX package, which you would need to install using:
# conda install -c conda-forge ucx
#
"""VASP driver (formerly tools/external.py): call_vasp and helpers (POTCAR, Hubbard U, magnetic moments, restarts)."""
import os
import re
import shutil
from collections import Counter
from copy import deepcopy
from os import chdir, getcwd, getenv, listdir
from sys import stdout

import numpy as np
import ase
import ase.io
from ase import Atom
from ase.calculators.vasp import Vasp
from ase.visualize import view as ase_view #'view' is shadowed by a keyword argument in call_vasp

LDAU_L={'s':0,'p':1,'d':2,'f':3} #orbital -> LDAUL value

def _ldau_lists(asyms,hubU):
    #Build the LDAUU/LDAUL lists (in the order of atom types in POSCAR) from a hubU dict, e.g. {'Fe':{'d':'2.7'}}.
    ldauu=[];ldaul=[]
    for at in asyms:
        if at in hubU:
            if len(hubU[at].keys())>1: print("WARNING: cannot set multiple U values for the same atom type (%s), setting to 0... "%at);ldauu.append(0);ldaul.append(0);continue
            orb="".join(hubU[at].keys()).lower()
            if orb not in LDAU_L: raise ValueError("Hubbard U: unknown orbital '%s' for %s (use s, p, d or f)."%(orb,at))
            ldauu.append(float(hubU[at][orb] if orb in hubU[at] else list(hubU[at].values())[0]))
            ldaul.append(LDAU_L[orb])
        else:ldaul.append(0);ldauu.append(0)
    return ldauu,ldaul

def _vasp_cmd(exe,variant='std'):
    #Return the VASP command using the requested executable variant (vasp_std/vasp_gam), with output redirected to vasp.out.
    base=exe.split('>')[0].strip()
    if re.search(r'vasp_(std|gam|ncl)',base): base=re.sub(r'vasp_(std|gam|ncl)','vasp_%s'%variant,base)
    return base+" > vasp.out"

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
        atoms.calc.set(ldau=True,ldauprint=0,ldautype=2,lmaxmix=6)
        asyms = list(dict.fromkeys(atoms.get_chemical_symbols()))
        ldauu,ldaul=_ldau_lists(asyms,hubU)
        atoms.calc.set(ldauj=[ 0 for i in range(len(asyms)) ])
        atoms.calc.set(ldauu=ldauu,ldaul=ldaul)
        atoms.calc.set(lasph=True)#This is essential for accurate total energies and band structure calculations for f-elements (e.g. ceria), all 3d-elements (transition metal oxides), and magnetic atoms in the 2nd row (B-F atom), in particular if LDA+U or hybrid functionals or meta-GGAs are used, since these functionals often result in aspherical charge densities.
    return atoms

def get_potcar(elements, xc):
    vasppp = os.environ.get('VASP_PP_PATH')
    if not vasppp: raise EnvironmentError("VASP_PP_PATH is not defined, but needed for VASP calculation setup.")

    ppdir = os.path.join(vasppp, xc)

    pp = ''
    for e in elements:
        #print os.path.join(ppdir, e, 'POTCAR')
        pp += open(os.path.join(ppdir, e, 'POTCAR')).read()

    return pp

def make_potcar(xc="potpaw_PBE",elements=None,wDir='./'):
    vasppp = os.environ.get('VASP_PP_PATH')
    if not vasppp: raise EnvironmentError("VASP_PP_PATH is not defined, but needed for VASP calculation setup.")

    print('Making potcar using %s'%vasppp);stdout.flush();


    if elements==None:#Try to read it from POSCAR if not explicitly given.
        try: #VASP5 format.
            elements = open(wDir+"/POSCAR").readlines()[5].strip().split()
        except:
            raise ValueError('Elements not given on command line and POSCAR not found in %s'%wDir)

    pp = get_potcar(elements, xc)

    p = open(wDir+"/POTCAR", 'w')
    p.write(pp);    p.close()

def vasp_continue(ifDel=0):
        #Back up the files of the previous run into the next free RUN<n> folder.
        ind=0
        for fn in listdir("."):
                m=re.match(r'^RUN(\d+)$',fn)
                if m: ind=max(ind,int(m.group(1))) #RUN10 was previously read as RUN0

        run="RUN%d"%(ind+1)
        print ("Previous data stored in %s."%run)
        os.makedirs(run,exist_ok=True)
        for fn in listdir("."):
                if os.path.isfile(fn): shutil.copy2(fn,run)
        if ifDel:
                for fn in ('WAVECAR','CHGCAR'):
                        if os.path.exists(fn): os.remove(fn)
 
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
        if not exe: exe=getenv('VASP_COMMAND')
        if not exe:
               nprocs=nproc or getenv('SLURM_NTASKS')
               if nprocs and mpiexe in ('mpirun','srun'): exe = "%s -n %d vasp_std > vasp.out"%(mpiexe,int(nprocs))
               else: exe = "vasp_std > vasp.out"
        #exe="mpirun -np 128 ~/APPS/vasp.5.4.4/bin/vasp_std > vasp.out&"
        os.environ['VASP_COMMAND']=exe
        print("VASP command: ", exe) #, os.environ['VASP_COMMAND'])

        cwd=os.getcwd()


        try: ncores=int(getenv('SLURM_NTASKS'))  #ncores=int(popen('echo "$SLURM_NTASKS"',"r").read()[0:-1]) #add ones for PBS
        except:ncores=1
        if len(atoms)<20: ncores=1 #NPAR>1 can be problematic with small systems due to partioning of the bands.
 

        if calc!=None:
                #This is good for assiging the settings from a previously created Calculator object (i.e. not to repeat all the settings the second time). None of the other options will be taken into account.
                atoms.calc=calc
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
                if restart: print ("Previous run was read successfully, checking if converged...")
        except Exception: #if restarting fails
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
                        # #atoms.calc=calc
                        # vasp_continue() #copies prev. run  files into RUNX folder.
                        # #atoms.write('POSCAR',format='vasp',vasp5=1)
                        # #calc = Vasp(atoms,restart=restart, directory="./", label='vasp', command=exe, txt=None) #ignore_bad_restart_file=False,
                        # calc = Vasp(restart=True)
                        # #atoms.calc=calc
                        # atoms = calc.get_atoms()

                        #The settings of the previous run are kept by reading its INCAR (and KPOINTS), and the run
                        #is continued from the last geometry (CONTCAR) rather than the initial one (POSCAR).
                        calc=Vasp(directory="./", label='vasp', command=exe, txt=None)
                        calc.read_incar(filename='INCAR')
                        if os.path.exists('KPOINTS'):
                                try: calc.read_kpoints(filename='KPOINTS')
                                except Exception as err: print("Could not read KPOINTS from the previous run:",err)
                        vasp_continue()
                        if os.path.exists("CONTCAR") and os.path.getsize("CONTCAR")>0: atoms=ase.io.read("CONTCAR",format="vasp")
                        else: atoms=ase.io.read("POSCAR",format="vasp")

                        calc.set(xc=xc)
                        calc.set(setups=setups)
                        atoms.calc=calc
                        if not dry:
                                atoms.get_potential_energy()
                                atoms=ase.io.read('vasprun.xml',index=-1)

                E=atoms.get_potential_energy() if not dry else 0.0
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
        else: chdir(cwd); raise ValueError("call_vasp: typ must be 'sp' or 'geom', got '%s'"%typ)

        if FixCell:isif=2 #fixed cell optimization
        elif FixVol:isif=4 #cell dimensions can vary but total volume is kept fixed (good for keeping the vacuum padding).
        else:isif=3 #full cell relaxation.
        #if vac<=2:isif=3 #this must be set to relax the cell only in the vacuum direction (i.e. c-axis), but not possible in VASP.

        npar=int(np.sqrt(ncores))
        if ncores%npar!=0 or npar==1:npar=8
        calc.set(xc=xc.lower(),encut=ENCUT,prec="Normal",ediff=etol, ediffg=ediffg,sigma=sigma,reciprocal=False,algo=algo,ispin=ispin,lreal="AUTO",nelm=nelm,ibrion=IBRION,gamma=bool(gamma),isif=isif,nsw=nsw,ismear=ismear,npar=npar) #ncore=ncore)#,npar=npar)
        calc.set(nwrite=1, lcharg=False, lwave=False) #Wavecar could be used for next run, specially for ISPIN=2
        if ispin==2:calc.set(lwave=True) 

        calc.set(lmaxmix=6,lasph=True) #need when there is a trans metal.

        if nosymm: calc.set(isym=0)
        #TODO add support for reading INCAR file if exists.
        #calc.set(setups='recommended')


        if KPgrid: #KPgrid given
                #calc.cell.kpoint_mp_grid = KPgrid #def='1 1 1'
                kpgrid=[int(x) for x in KPgrid.split()]
                #Pass the grid itself (written as a Gamma-centred/Monkhorst-Pack KPOINTS file depending on 'gamma').
                #Previously an explicit list of Monkhorst-Pack points (in reciprocal units) was passed with reciprocal=False,
                #so VASP interpreted them as Cartesian coordinates.
                calc.set(kpts=kpgrid)
                if kpgrid==[1,1,1]: 
                        print ("VASP gamma-only implementation (vasp_gam) is activated.")
                        cmd=_vasp_cmd(exe,'gam')
                else:
                        print ("Multiple KPOINTS: VASP standard implementation (vasp_std) is activated.")
                        cmd=_vasp_cmd(exe,'std')

                calc.command=cmd
                #print(cmd)

        elif KPspacing: 
                calc.set(kspacing=KPspacing,kgamma=bool(gamma))#calc.cell.kpoints_mp_spacing = str(KPspacing) #default=0.05 2*pi/A
                print ("Multiple KPOINTS: VASP standard implementation (vasp_std) is activated.")
                cmd=_vasp_cmd(exe,'std')
                calc.command=cmd
        
        else:
               chdir(cwd)
               raise ValueError('call_vasp: Either KPspacing or KPgrid keyword must be defined!')

        a,b,c,_,_,_=atoms.cell.cellpar()
        if a >35 or b>35 or c>35: calc.set(amin=0.01) #gor super large cells to avoid the charge sloshing along the long lattice vector. 
        if slowConv:
                calc.set(amix=0.1,bmix=3.0,lmaxmix=6)

        if (dipolCorr and dipolCorr.lower()!='none'):
                calc.set(ldipol = True) #this one switches on  the corrections to the potential and thus forces
                if dipolDir.lower() not in ('x','y','z'): chdir(cwd); raise ValueError("call_vasp: dipolDir must be x, y or z")
                idipol={'x':1,'y':2,'z':3}[dipolDir.lower()] #IDIPOL switches on the single point energy corrections.
                calc.set(idipol = idipol) #no need to set DIPOL keyword (giving the charge center, if no charged dipole in the system (i.e. total charge=0) 

                #One needs to set the DIPOL keyword (i.e. the center of the net charge in scaled/fractional coordinates), as the automated algorithm leads to terrible convergence. It should be the centre of charge (that requires the CHG file to be created and analysed to be determined), instead  the centre of mass could be used as a good approximation. Setting DIPOL helps even the veryslow convergence when the LDIPOL=true. It even affects the total spin of the system (when ISPIN=2).

                calc.set(dipol=atoms.get_center_of_mass(scaled=1)) #Does this need to be updated for continuation run or better to stick with initial value for better covnergence ??? (CHECK)


        if 1 and len(hubU)!=0: #is not None:
                #if len(hubU)>0: atoms=set_hubU(atoms,hubU)
                calc.set(ldau=True,ldauprint=0,ldautype=2,lmaxmix=6)
                ldauu,ldaul=_ldau_lists(asyms,hubU)
                calc.set(ldauj=[ 0 for i in range(len(asyms)) ])
                calc.set(ldauu=ldauu,ldaul=ldaul)
                calc.set(lasph=True)#This is essential for accurate total energies and band structure calculations for f-elements (e.g. ceria), all 3d-elements (transition metal oxides), and magnetic atoms in the 2nd row (B-F atom), in particular if LDA+U or hybrid functionals or meta-GGAs are used, since these functionals often result in aspherical charge densities.

        if magmom: 
                # Adding MAGMOM and ISPIN to INCAR if -mgm or --magmoms is defined.
                atoms = handle_magmoms(atoms=atoms, magmoms=magmom) 
                calc.set(ispin=2)
                

        if passivate: #compute the coordination numbers for each atom, do statistics and then add passivating (pseudo)-Hydrogen atoms. 
                #from ase.build import molecule
                
                #Preparations
                if passivate.lower() not in ('bot','top'): chdir(cwd); raise ValueError("call_vasp: passivate must be 'bot' or 'top'")
                setups={'base': setups if isinstance(setups,str) else 'recommended'}
                #Covalent radii: read from $HOME/covalent_radii.dat if present, otherwise use ASE's values.
                from ase.data import covalent_radii as ase_cov_radii, atomic_numbers
                cov_radii={}
                for el in set(asyms+['H']): cov_radii[el]=float(ase_cov_radii[atomic_numbers[el]])
                crfile=os.path.join(os.path.expanduser('~'),"covalent_radii.dat")
                if os.path.exists(crfile):
                        for ln in open(crfile,'r'):
                                x=ln.split()
                                if len(x)>=2: cov_radii[x[0]]=float(x[1])

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
                        coord = np.bincount(i,minlength=len(atoms)) #no of neighbours of each atom (also those without neighbours)

                        #The dictionary of neighbour types for each atom index (every atom has an entry).
                        ntypes={str(k):[] for k in range(len(atoms))}
                        for k in range(len(i)): ntypes[str(i[k])].append(atoms[j[k]].symbol)

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
                nl=min(4,len(layers))
                for i in range(nl):
                        lay1=layers[i]
                        for j in range(-1,-nl-1,-1):
                                lay2=layers[j]
                                if len(lay1)!=len(lay2): continue #does not match
                                for k in range(len(lay1)):
                                        pos1=atoms[lay1[k]].position  ; pos2=atoms[lay2[k]].position
                                        if abs(pos1[0]-pos2[0]) > tol or abs(pos1[1]-pos2[1]) > tol: break#layers do not match.
                                else:
                                        print ("Layer #%d matches layer #%d"%(i,j))


                #Max occurence of coordination no for a given atom type can also be used, instead of avg coord no.
                if 1:
                    crds={};common_crd={};nty={};common_ntypes={};valance={};dangling={}
                    for aty in asyms:
                            #Determine the no of coordination for each atom.
                            crds[aty]=[coord[at.index] for at in atoms if at.symbol==aty]
                            common_crd[aty]=np.argmax(np.bincount(crds[aty]))
                            #Determine the neighbour types.
                            nty[aty]=[tuple(sorted(ntypes[str(at.index)])) for at in atoms if at.symbol==aty] #Do we ned to store this??
                            common_ntypes[aty]=list(Counter(nty[aty]).most_common(1)[0][0]) #most common neighbour environment
                            #Get the valence electron no from the POTCAR files.
                            val=None
                            for ppd in ('potpaw','potpaw_PBE'):
                                    potfile="%s/%s/%s/POTCAR"%(getenv('VASP_PP_PATH'),ppd,aty)
                                    if os.path.exists(potfile): break
                            with open(potfile,'r') as f:
                                    for ln in f:
                                            if  'ZVAL' in ln:  val=float(ln.split('ZVAL')[-1].split()[1]);  break
                            if val is None: chdir(cwd); raise ValueError("call_vasp: ZVAL not found in %s"%potfile)
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
                                if passivate.lower()=='top': offset=-offset
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
                                                if passivate.lower()=='top': offset=-offset
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
                if view:ase_view(atoms)

                if dipolCorr and dipolCorr.lower()!='none':  calc.set(dipol=atoms.get_center_of_mass(scaled=1))


        atoms.calc=calc      
        

        if dry: #do not run VASP calculation; write the input files (if possible) for checking.
                try: calc.write_input(atoms)
                except Exception as err: print("call_vasp (dry run): could not write all VASP input files in %s: %r"%(wDir,err))
                atoms.calc=None #no energy available; prevents an accidental VASP run later on
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
    try:chdir(seed)
    except Exception:print('Cannot change to %s directory, using %s instead'%(seed,cwd))

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
        atoms.calc=calc

    # Adding MAGMOM and ISPIN to INCAR if -mgm or --magmoms is defined.
    atoms = handle_magmoms(atoms=atoms, magmoms=mgms) 
    if len(hubU)>0: atoms=set_hubU(atoms,hubU)

    Ep = atoms.get_potential_energy() 
    chdir(cwd)

    return Ep,atoms

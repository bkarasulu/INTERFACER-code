"""Convergence and screening tests: surface stability (-tsur), layer thickness (-tlay), interface separation (-tsep)
and lateral translation of the slabs (-thor).

All functions keep their original positional arguments; the code used is given by the keyword `engine`
(an :class:`interfacer.calculators.Engine`). If it is omitted, the engine set with :func:`set_default_engine` is used.
"""
import os
from sys import stdout

import numpy as np
from matplotlib.figure import Figure

from .analysis import find_prim, surf_area
from .builder import make_interface, make_slab
from .io import write_struct

from .log import debug, say

#eV / nm^2 to J / m^2: 1 eV/nm^2 = 1.602e-19 J / 1e-18 m^2 = 0.1602 J/m^2.
cfactor = 0.160217663

_default_engine = None


def set_default_engine(engine):
    """Set the engine used when the functions in this module are called without `engine`."""
    global _default_engine
    _default_engine = engine


def _engine(engine):
    engine = engine if engine is not None else _default_engine
    if engine is None:
        raise ValueError("No engine given (use engine=... or set_default_engine()).")
    return engine


def _calc_name(engine, name):
    return os.path.join(engine.tmpdir, name) if engine.tmpdir else name


def check_surfaces(atoms, miller_list, vac, Ebulk, creps, engine=None, kpgrid=None, outdir="OUTPUT"):
        """Compute the surface energies of the given Miller planes and return the most stable one.

        Args:
            atoms (ase.Atoms): (optimised) bulk structure.
            miller_list (list): Miller indices, e.g. [(1,0,0),(1,1,0),(1,1,1)].
            vac (float): vacuum padding (A, each side).
            Ebulk (float): bulk energy per atom (eV/atom).
            creps (int): number of repetitions (layers) of the slabs along c.
        Returns:
            (W_surf in eV/nm^2, slab, calculator, Miller indices) of the lowest-energy surface.
        """
        engine=_engine(engine)
        min_Ws=1e8; min_mil=();min_slab=None
        calc=atoms.calc
        str1=''
        for i,mil in enumerate(miller_list):
                say('Doing the (%d,%d,%d) surface...'%(mil[0],mil[1],mil[2]));stdout.flush()
                slab1 = make_slab(mil,atoms,repeat=(1,1,creps),square=False)
                slab1.center(vacuum=vac, axis=2)

                name="%s_%d%d%d"%(slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])
                slab1.name=name
                #???Which one better: Single point or geometry opt??? (ASE engines: single point, as in the original code)
                x=engine.run(slab1,"geom",_calc_name(engine,name),dipolCorr='sc',FixCell=True,KPgrid_=kpgrid,ase_opt=False) #fixed cell: keeps the vacuum padding

                slab1=x[-1]
                Eslab1=x[0]
                Ws1=(Eslab1-len(slab1)*Ebulk)/2/surf_area(slab1)/0.01 #A2 to nm2
                str1+='%s (%s_%d%d%d): %.2f eV/nm^2 = %.2f J/m^2\n' % ('W_surf ', slab1.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2],Ws1,Ws1*cfactor)
                if i==0 or Ws1<min_Ws: min_Ws=Ws1; min_mil=mil;min_slab=slab1.copy()

                write_struct(slab1,name,engine,outdir)

        say(str1);stdout.flush()
        #The surface with the lowest (positive) surface energy is the most stable one.
        Ws1=min_Ws
        say('check_surfaces: Minimum-energy surface for material %s is %s with %d atoms and W_surf=%.2f eV/nm^2'%(min_slab.get_chemical_formula(empirical=1),min_mil,min_slab.get_global_number_of_atoms(),Ws1));stdout.flush()

        min_slab.calc=calc
        return Ws1, min_slab,calc,min_mil

def _layer_loop(make, name, engine, vac, Etol, lay_min, lay_max, view, outdir):
        #Shared loop of conv_layers/conv_layers_new: make(n) returns the slab with n layers.
        E=[];layers=[]
        i=lay_min
        flag=0 #converged?
        atoms=None;nAt=0
        while  i<=lay_max:
                layers.append(1+1*i) #increase 1 layer at a time
                atoms=make(layers[-1])
                atoms.center(vacuum=vac, axis=2)
                atoms.name='%s_%d-layer'%(name,layers[-1])
                nAt=atoms.get_global_number_of_atoms()
                x=engine.run(atoms,"sp",_calc_name(engine,'%s_%d-layer'%(name,layers[-1])),dipolCorr='sc')
                atoms=x[-1]
                E.append(x[0]/nAt)
                say("Iter. #%d, #layers: %d, #atoms: %d "%(len(E),layers[-1],nAt));stdout.flush()
                if len(E)>1: say("E_total: %.5f; deltaE: %.3e eV/atom; target: %.3e eV."%(E[-1],abs(E[-1]-E[-2]),Etol));stdout.flush()
                else: say("E_total: %.5f eV/atom"%(E[-1]));stdout.flush()
                if view:
                        from ase.visualize import view as ase_view
                        ase_view(atoms)

                #Writing layer steps
                write_struct(atoms,"%s-layer"%name,engine,outdir,trajectory=True,append=(len(E)>1))
                if len(E)>1 and abs(E[-1]-E[-2]) <= Etol:
                       flag=1
                       break
                i += 1

        if flag:
               say("conv_layers: E/atom converged to %.2e eV with %d layers."%(Etol,layers[-1]));stdout.flush()
        else:
               say('conv_layers: Layer thickness not converged to %.2e eV within %d layers '%(Etol,layers[-1]));stdout.flush()
        return layers,E,atoms,nAt

def _plot_layers(layers,E,fname):
        fig=Figure(); ax=fig.add_subplot()  #no pyplot: works without a display and does not change the backend
        ax.plot(layers,E, 'ko-')
        ax.set_xlabel('Number of layers')
        ax.set_ylabel('Energy per atom (eV/atom)')
        fig.savefig(fname)

def conv_layers(atoms,vac=2.0,ifPlot=0,ifPrim=False, Etol=1e-2,view=0,lay_min=0,lay_max=5,engine=None,outdir="OUTPUT"):
        """Layer convergence test: converge the energy per atom w.r.t. the number of repetitions of the given slab along c
        (lay_min+1 ... lay_max+1 layers). Returns (number of layers, total energy, final slab)."""
        engine=_engine(engine)
        name=getattr(atoms,'name','slab')
        atoms_orig=atoms.copy()
        if ifPrim: atoms_orig=find_prim(atoms_orig) #find the primitive cells to reduce comp. efforts.
        layers,E,atoms,nAt=_layer_loop(lambda n: atoms_orig.copy().repeat((1,1,n)), name, engine, vac, Etol, lay_min, lay_max, view, outdir)
        if ifPlot: _plot_layers(layers,E,os.path.join(outdir,'conv_layers_%s.png'%name))
        return layers[-1],E[-1]*nAt,atoms

def conv_layers_new(mil,atoms,vac=2.0,ifPlot=1,ifPrim=False, Etol=1e-2,view=0,lay_min=0,lay_max=5,engine=None,outdir="OUTPUT"):
        """Same as conv_layers, but the slabs are re-created from the bulk (atoms) with make_slab for each number of layers."""
        engine=_engine(engine)
        say("Convergence of E/atom vs. #layers")
        name="%s_%d%d%d"%(atoms.get_chemical_formula(empirical=1),mil[0],mil[1],mil[2])
        if ifPrim: atoms=find_prim(atoms)
        layers,E,slab,nAt=_layer_loop(lambda n: make_slab(mil,atoms,repeat=(1,1,n),square=False), name, engine, vac, Etol, lay_min, lay_max, view, outdir)
        if ifPlot: _plot_layers(layers,E,os.path.join(outdir,'conv_layers_%s.png'%name))
        return layers[-1],E[-1]*nAt,slab

def lateral_shift(slab, xy):
        """Copy of a slab translated laterally by the fractional shift (x, y) of its in-plane cell vectors (wrapped)."""
        s=slab.copy()
        s.translate(s.cell.cartesian_positions([xy[0],xy[1],0]))
        s.wrap(pbc=[True,True,False])
        return s

def convSep(slab1,slab2,vac=None,sep_init=1.5,sep_final=4.0,sep_step=0.5,view=0,ifPlot=1, Etol=1e-2,engine=None,kpgrid=None,outdir="OUTPUT",shift=None):
        """Scan the separation between two rigid slabs (sep_init ... sep_final, included) with single points.
        The vacuum padding is vac (default sep/2, i.e. two equivalent interfaces). shift=(x, y) translates slab1 laterally
        (fractions of its cell vectors, e.g. the best registry from checkHorizontal) before stacking.
        Returns (lowest-energy interface, separation)."""
        # TODO: find a way to check whether two slabs are clashing (i.e. <2.0A)
        engine=_engine(engine)
        if shift is not None and any(abs(v)>1e-12 for v in shift): slab1=lateral_shift(slab1,shift)
        E=[]
        minSep=sep_init; Emin=1e6; minInt=None
        seps=np.arange(sep_init,sep_final+0.5*sep_step,sep_step) #sep_final is included
        interface=None
        for i,sep in enumerate(seps):
                say('Doing the %.2f A separation between two layers'%sep);stdout.flush()
                vacc=vac if vac else sep/2
                interface=make_interface(slab1,slab2,sep,vacc,reorder=True)
                a,b,c,alpha,beta,gamma=interface.cell.cellpar()
                interface.name='interface-sep_%.2fA'%sep

                nAt=interface.get_global_number_of_atoms()
                x=engine.run(interface,"sp",_calc_name(engine,'interface-sep_%.2fA'%sep),dipolCorr='None',KPgrid_=kpgrid)
                interface=x[-1]
                E.append(x[0]/nAt)
                if i==0 or  E[-1]<Emin: minSep=sep;Emin=E[-1];minInt=interface.copy()
                say("Iter. #%d, Seperation: %.2f A, E / atom : %.5f eV, z-thickness: %.2f A \n"%(i+1,sep,E[-1],c));stdout.flush()

                write_struct(interface,"int-sep_%.2fA"%sep,engine,outdir)

        say('\nconvSep: Minimum-energy separation is %.2f A with E= %.5f eV/atom'%(minSep,Emin));stdout.flush()

        if minInt: interface=minInt.copy()
        if outdir:
                os.makedirs(outdir,exist_ok=True)
                np.savetxt(os.path.join(outdir,'conv_sep.csv'),np.column_stack([seps,E]),delimiter=",",header="sep(A),E_per_atom(eV)",comments="")

        if ifPlot: #Do plotting of E/atom vs. separation
                fig=Figure(); ax=fig.add_subplot()
                ax.plot(seps,E, 'ko-')
                ax.set_xlabel('Interlayer Separation [A]')
                ax.set_ylabel('Energy per atom [eV/atom]')
                fig.savefig(os.path.join(outdir,'conv_sep.png'))

        return interface,minSep

def checkHorizontal(slab1,slab2,sep,vac=None,steps=(0.25,0.25),ifPlot=0,engine=None,kpgrid=None,outdir="OUTPUT"):
        """Lateral translation test: translate the bottom slab (slab1) on a 2D grid of fractional shifts (x,y) in [0,1)
        along a and b, keeping both slabs rigid, and compare single-point energies.
        Returns (lowest-energy interface, its energy per atom, [x, y])."""
        engine=_engine(engine)
        vacc=vac if vac else sep/2
        ln1=len(slab1); ln2=len(slab2)

        a,b,c=slab1.cell.lengths()
        debug('Cell lengths: %.2f A %.2f A %.2f A'%(a,b,c));stdout.flush()

        int_orig=make_interface(slab1,slab2,sep,vacc,reorder=False) #atoms of slab1 first

        xrng=np.arange(0.0,1.0-1e-8,steps[0])
        yrng=np.arange(0.0,1.0-1e-8,steps[1])
        Egrid=np.zeros((len(xrng),len(yrng)))
        minE=1e6;minInt=None;minxy=[0.,0.]
        first=True
        for ix,x in enumerate(xrng):      #work in fractional coords. Convert to Cartesian for translation
          for iy,y in enumerate(yrng):
                interface=int_orig.copy()

                cart_disp=interface.cell.cartesian_positions([x,y,0])
                disp=[cart_disp for j in range(ln1)]
                disp.extend([[0., 0., 0.] for j in range(ln2)])
                interface.translate(disp)
                interface.wrap()

                nAt=interface.get_global_number_of_atoms()
                interface.name='htrans_x%.2f-y%.2f'%(x,y)
                xx=engine.run(interface,"sp",_calc_name(engine,'htrans_x%.2f-y%.2f'%(x,y)),dipolCorr='None',KPgrid_=kpgrid)
                interface=xx[-1]
                Egrid[ix,iy]=xx[0]/nAt

                if first or  Egrid[ix,iy]<minE: minE=Egrid[ix,iy];minInt=interface.copy();minxy=[x,y];first=False
                say("Displacement: x= %.2f, y= %.2f (x= %.2f A , y= %.2f A); E/atom: %.5f eV, "%(x,y,cart_disp[0],cart_disp[1],Egrid[ix,iy]));stdout.flush()

                write_struct(interface,"htrans_x%.2f-y%.2f"%(x,y),engine,outdir)

        say('Minimum-energy horizontal translation: x= %.2f, y= %.2f with E= %.5f eV/atom'%(minxy[0],minxy[1],minE));stdout.flush()
        if outdir:
                os.makedirs(outdir,exist_ok=True)
                X,Y=np.meshgrid(xrng,yrng,indexing='ij')
                np.savetxt(os.path.join(outdir,'conv_htrans.csv'),np.column_stack([X.ravel(),Y.ravel(),Egrid.ravel()]),delimiter=",",header="x,y,E_per_atom(eV)",comments="")

        if ifPlot: #2D colour map of E/atom vs. x-y translation
                fig=Figure(); ax=fig.add_subplot()
                im=ax.imshow(Egrid.T,origin='lower',extent=[xrng[0]-steps[0]/2,xrng[-1]+steps[0]/2,yrng[0]-steps[1]/2,yrng[-1]+steps[1]/2],aspect='auto')
                fig.colorbar(im,ax=ax,label='Energy per atom [eV/atom]')
                ax.set_xlabel('fractional shift along a')
                ax.set_ylabel('fractional shift along b')
                fig.savefig(os.path.join(outdir,'conv_htrans.png'))

        return minInt,minE,minxy

def optimise_placement(slab1,slab2,sep0,xy0=(0.0,0.0),vac=None,engine=None,kpgrid=None,outdir="OUTPUT",maxiter=60,
                       method="Nelder-Mead",sep_bounds=(0.5,6.0)):
        """Optimise the placement of two rigid slabs (lateral shift x, y of slab1 in fractions of the cell vectors and the
        separation sep) with scipy.optimize.minimize on single-point energies ("key-lock" model).

        The vacuum padding is vac (default sep/2, i.e. two equivalent interfaces). Starts from (xy0, sep0), e.g. the best
        point of checkHorizontal/convSep. Returns (interface, (x, y, sep), energy per atom)."""
        from scipy.optimize import minimize
        engine=_engine(engine)
        history=[]
        nat=len(slab1)+len(slab2)

        def build(p):
                x,y,sep=p
                return make_interface(lateral_shift(slab1,(x,y)),slab2,sep,vac if vac else sep/2,reorder=True,check_clashes=False)

        def energy(p):
                x,y,sep=p
                if not (sep_bounds[0] <= sep <= sep_bounds[1]):
                        return 1e3+abs(sep-np.clip(sep,*sep_bounds))*1e3   #outside the allowed separations
                interface=build(p)
                interface.name='topt_%03d'%len(history)
                E=engine.run(interface,"sp",_calc_name(engine,'topt_%03d'%len(history)),dipolCorr='None',KPgrid_=kpgrid)[0]/nat
                history.append((x%1.0,y%1.0,sep,E))
                debug("placement x= %.3f y= %.3f sep= %.3f A: E/atom= %.5f eV"%(x%1.0,y%1.0,sep,E))
                return E

        p0=np.array([xy0[0],xy0[1],sep0],float)
        opts={"maxiter":maxiter,"xatol":1e-3,"fatol":1e-5} if method=="Nelder-Mead" else {"maxiter":maxiter}
        if method=="Nelder-Mead":
                opts["initial_simplex"]=np.array([p0,p0+[0.1,0,0],p0+[0,0.1,0],p0+[0,0,0.25]])
        res=minimize(energy,p0,method=method,options=opts)
        x,y,sep=res.x[0]%1.0,res.x[1]%1.0,float(res.x[2])
        best=build((x,y,sep))
        say("optimise_placement: x= %.3f, y= %.3f, sep= %.3f A, E= %.5f eV/atom after %d single points"%(x,y,sep,res.fun,len(history)))
        write_struct(best,"interface-placement_opt",engine,outdir)
        if history:
                np.savetxt(os.path.join(outdir,"placement_opt.csv"),np.array(history),delimiter=",",header="x,y,sep(A),E_per_atom(eV)",comments="")
        return best,(x,y,sep),float(res.fun)

def checkHorizontal2(slab1,slab2,sep,vac=None,**kw): #This checks the optimum horizontal stacking of two slabs through translating the slab2
        """Optimisation of the slab placement with scipy (see optimise_placement); returns (interface, energy per atom, [x, y])."""
        interface,(x,y,s),E=optimise_placement(slab1,slab2,sep,vac=vac,**kw)
        return interface,E,[x,y]

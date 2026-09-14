"""Conservative 2:1 D3Q19 transfer and coupled wall-block reference.

A parent stores population MASS in units shared with its eight children: its
population density scale is eight times the children's scale. This avoids
rounding a flux mismatch by dividing it by eight. Restriction and splitting
conserve the actual stored populations, hence all conserved lattice moments.

Before collision, continuity of strain requires
 Pi_target = (dt_target*tau_target)/(dt_source*tau_source) * Pi_source.
This is the incoming-population convention; it must not be applied to a
post-collision distribution with the same factor. See:
https://www.mdpi.com/2311-5521/8/3/103
"""
import numpy as np
from .lbm3d_mrt_les import C,P2_PROJECTOR,equilibrium


def split_parent_mass(parent):
    """Integer eight-way split; signed remainders use fixed octant order."""
    parent=np.asarray(parent)
    if parent.dtype!=np.int64 or parent.ndim!=4 or parent.shape[0]!=19:
        raise ValueError('expected int64 parent population masses')
    shape=parent.shape[1:];out=np.empty((19,2*shape[0],2*shape[1],2*shape[2]),np.int64)
    base=parent//8;remainder=parent%8
    for octant in range(8):
        dx,dy,dz=octant//4,(octant//2)%2,octant%2
        out[:,dx::2,dy::2,dz::2]=base+(remainder>octant)
    return out


def restrict_child_mass(children):
    """Exact population restriction, with a conservative overflow bound."""
    children=np.asarray(children)
    if children.dtype!=np.int64 or children.ndim!=4 or children.shape[0]!=19 or any(n%2 for n in children.shape[1:]):
        raise ValueError('expected even-sized int64 child grid')
    bound=max(abs(int(children.min())),abs(int(children.max())))
    if bound>=(2**63)//8:raise OverflowError('eight-child reduction may overflow')
    n,m,p=(v//2 for v in children.shape[1:])
    return children.reshape(19,n,2,m,2,p,2).sum(axis=(2,4,6),dtype=np.int64)


def rescale_incoming_stress(f,*,tau_source,tau_target,dt_ratio):
    """Unforced Hermite population transfer; density and velocity preserved.

Higher non-equilibrium moments are removed, consistent with the selected
Hermite MRT. This function takes density populations, not volume-weighted
integer masses. Forcing and space/time interpolation are not implemented here.
"""
    f=np.asarray(f,dtype=np.float64)
    if f.ndim!=4 or f.shape[0]!=19 or not np.isfinite(f).all():raise ValueError('invalid populations')
    if not np.isfinite([tau_source,tau_target,dt_ratio]).all() or min(tau_source,tau_target)<=.5 or dt_ratio<=0:
        raise ValueError('invalid relaxation or time scale')
    rho=f.sum(0)
    if np.any(rho<=0):raise ValueError('positive density required')
    u=np.einsum('qd,q...->d...',C,f)/rho
    eq=equilibrium(rho,u)
    return eq+(dt_ratio*tau_target/tau_source)*np.einsum('pq,q...->p...',P2_PROJECTOR,f-eq)


def reflux_population_mass(coarse_cell,incoming_fine_net,coarse_net):
    """Replace a coarse interface flux by the two fine-step fluxes.

All three arrays must use the same physical mass unit, including volume.
Inputs are accumulated signed NET transfers INTO the active coarse region,
so adding fine_net - coarse_net closes its per-population interface balance.
This is algebra only: callers must collect actual streamed fluxes at matching
locations and times; this primitive does not make interpolation conservative.
"""
    arrays=[np.asarray(v) for v in (coarse_cell,incoming_fine_net,coarse_net)]
    if any(v.dtype!=np.int64 for v in arrays) or any(v.shape!=arrays[0].shape for v in arrays):
        raise ValueError('matching int64 arrays required')
    bound=sum(max(abs(int(v.min())),abs(int(v.max()))) for v in arrays)
    if bound>=2**63:raise OverflowError('reflux sum may overflow')
    return arrays[0]+arrays[1]-arrays[2]


def checked_int64_sum(values):
    """Reduce bounded chunks and combine in Python before the int64 range gate.

    Opposing wall-pressure terms may have intermediate sums beyond int64,
    even when their exact net is representable. No NumPy reduction may wrap.
    """
    a=np.asarray(values)
    if a.dtype!=np.int64 or not a.size:
        raise ValueError('nonempty int64 array required')
    bound=max(abs(int(a.min())),abs(int(a.max())))
    chunk=max(1,(2**63-1)//max(1,bound))
    flat=a.ravel()
    total=sum(int(flat[i:i+chunk].sum(dtype=np.int64)) for i in range(0,flat.size,chunk))
    if total<-(2**63) or total>=2**63:
        raise OverflowError('exact sum exceeds int64 range')
    return total


def sgs_transfer_taus(rho,stress_norm,tau_source,tau_target,cs_source,cs_target,dt_ratio):
    """Effective relaxation times for incoming forced-stress transfer.

    Pi_target = r*tau_eff_target/tau_eff_source * Pi_source. Substituting
    this into the target Smagorinsky quadratic gives its explicit value.
    For Cs_target=Cs_source/r and acoustic molecular scaling, physical
    eddy viscosity and strain are continuous across the interface.
    """
    if not np.isfinite([tau_source,tau_target,cs_source,cs_target,dt_ratio]).all() or min(tau_source,tau_target)<=.5 or min(cs_source,cs_target)<0 or dt_ratio<=0:
        raise ValueError('invalid SGS transfer parameters')
    rho=np.asarray(rho);stress_norm=np.asarray(stress_norm)
    if np.any(rho<=0) or np.any(stress_norm<0) or not np.isfinite(rho).all() or not np.isfinite(stress_norm).all():
        raise ValueError('invalid density or stress norm')
    source=.5*(tau_source+np.sqrt(tau_source**2+18*np.sqrt(2)*cs_source**2*stress_norm/rho))
    target=tau_target+4.5*np.sqrt(2)*cs_target**2*dt_ratio*stress_norm/(rho*source)
    return source,target


class ReferenceRefinedChannel:
    """CPU-orchestrated two-wall-block reference, fixed 2:1 acoustic scaling.

    This intentionally exposes every transfer for conservation checks. It is
    not the GPU performance path. Fine blocks advance twice per coarse step;
    interface ghost populations use trilinear space interpolation and linear
    time interpolation of the coarse predictor. Reflux replaces coarse fluxes
    by measured fine fluxes. The physical Smagorinsky filter is fixed across
    levels: Cs_coarse = Cs_fine / 2 under acoustic scaling.
    """
    def __init__(self,nx=8,height=32,nz=8,wall_cells=8,tau_fine=.8,force_fine=1e-5,device='cpu',cs_fine=0.0,ledger_mode='int64'):
        import warp as wp
        from .lbm3d_mrt_les import quantize
        from .lbm3d_channel import ChannelSimulation
        if any(not isinstance(n,int) or n<2 or n%2 for n in (nx,height,nz,wall_cells)) or wall_cells<2 or 2*wall_cells>=height:
            raise ValueError('even dimensions and a nonempty coarse core required')
        if ledger_mode not in ('int64','int64_limbs'):raise ValueError('invalid ledger mode')
        self.ledger_mode=ledger_mode
        if ledger_mode=='int64' and nx*height*nz*2**40>=2**63:
            raise OverflowError('volume-weighted int64 inventory exceeds range')
        if not np.isfinite([tau_fine,force_fine,cs_fine]).all() or tau_fine<=.5 or cs_fine<0:
            raise ValueError('invalid relaxation, force or Smagorinsky coefficient')
        self.csf=cs_fine;self.csc=cs_fine/2
        self.nx,self.h,self.nz,self.nf=nx,height,nz,wall_cells
        self.tf=tau_fine;self.tc=.5+(tau_fine-.5)/2
        self.force_units=int(np.rint(force_fine*2**40));self.gf=self.force_units/2**40
        self.wp=wp;self.steps=0;self.b=wall_cells//2+1;self.t=height//2-wall_cells//2
        self.fine=[]
        for top in (False,True):
            shape=(nx,wall_cells+2,nz);mask=np.zeros(shape,np.int32)
            mask[:,-1 if top else 0,:]=1
            y=np.arange(wall_cells+2)-.5+(height-wall_cells if top else 0)
            u=np.zeros((3,)+shape);u[0]=self.gf/(2*((tau_fine-.5)/3))*y[None,:,None]*(height-y[None,:,None])-.5*self.gf
            u[:,mask!=0]=0
            q=quantize(equilibrium(np.ones(shape),u),40,ledger_mode=ledger_mode)
            self.fine.append(ChannelSimulation(q,tau=self.tf,force_density=self.gf,cs=self.csf,solid=mask,device=device,ledger_mode=ledger_mode))
        shape=(nx//2,height//2+2,nz//2);mask=np.zeros(shape,np.int32);mask[:,0,:]=1;mask[:,-1,:]=1
        y=2*np.arange(height//2+2)-1
        u=np.zeros((3,)+shape);u[0]=self.gf/(2*((tau_fine-.5)/3))*y[None,:,None]*(height-y[None,:,None])-self.gf
        u[:,mask!=0]=0
        q=quantize(equilibrium(np.ones(shape),u),43,ledger_mode=ledger_mode)
        self.coarse=ChannelSimulation(q,tau=self.tc,force_density=2*self.gf,cs=self.csc,bits=43,solid=mask,device=device,ledger_mode=ledger_mode)
        self.initial_ledger=self.ledger()

    def _replace(self,sim,q):
        self.wp.copy(sim.a,self.wp.array(q,dtype=self.wp.int64,device=sim.device))

    def _convert(self,f,source_tau,target_tau,dt_ratio,source_force,target_force,bits):
        from .lbm3d_mrt_les import quantize
        rho=f.sum(0);u=np.einsum('qd,q...->d...',C,f)/rho;u[0]+=.5*source_force/rho
        eq=equilibrium(rho,u)
        pi=np.einsum('qa,qb,q...->ab...',C,C,f-eq)
        for a in range(3):
            pi[0,a]+=.5*u[a]*source_force;pi[a,0]+=.5*u[a]*source_force
        source_cs=self.csf if dt_ratio==2 else self.csc
        target_cs=self.csc if dt_ratio==2 else self.csf
        effective_source,effective_target=sgs_transfer_taus(
            rho,np.sqrt(np.sum(pi*pi,axis=(0,1))),source_tau,target_tau,
            source_cs,target_cs,dt_ratio)
        pi*=dt_ratio*effective_target/effective_source
        for a in range(3):
            pi[0,a]-=.5*u[a]*target_force;pi[a,0]-=.5*u[a]*target_force
        h2=np.einsum('qa,qb->qab',C,C)-np.eye(3)[None,:,:]/3
        from .lbm3d_mrt_les import W
        target=eq+4.5*W[:,None,None,None]*np.einsum('qab,ab...->q...',h2,pi)
        target-=1.5*W[:,None,None,None]*C[:,0,None,None,None]*target_force
        return quantize(target,bits,ledger_mode=self.ledger_mode)

    def _covered(self,coarse):
        from .lbm3d_mrt_les import quantize
        for side,sim in enumerate(self.fine):
            q=sim.numpy()[:,:,1:-1,:]
            density=restrict_child_mass(q).astype(float)/2**43
            target=self._convert(density,self.tf,self.tc,2,self.gf,2*self.gf,43)
            if side==0:coarse[:,:,1:self.b,:]=target
            else:coarse[:,:,self.t+1:-1,:]=target
        return coarse

    def _ghost(self,old,new,alpha,side):
        f=((1-alpha)*old.astype(float)+alpha*new.astype(float))/2**43
        y=self.nf+.5 if side==0 else self.h-self.nf-.5
        cy=(y+1)/2;j=int(np.floor(cy));fy=cy-j
        x=np.arange(self.nx)/2-.25;z=np.arange(self.nz)/2-.25
        ix=np.floor(x).astype(int);iz=np.floor(z).astype(int);fx=x-ix;fz=z-iz
        out=np.zeros((19,self.nx,1,self.nz))
        for dx in (0,1):
            for dz in (0,1):
                wx=fx if dx else 1-fx;wz=fz if dz else 1-fz
                vals=(1-fy)*f[:,(ix+dx)%(self.nx//2),j,:]+fy*f[:,(ix+dx)%(self.nx//2),j+1,:]
                out[:,:,0,:]+=vals[:,:,(iz+dz)%(self.nz//2)]*wx[None,:,None]*wz[None,None,:]
        return self._convert(out,self.tc,self.tf,.5,2*self.gf,self.gf,40)

    def _fine_flux(self,q,side):
        net=np.zeros((19,self.nx//2,self.nz//2),np.int64)
        for h,(cx,cy,cz) in enumerate(C):
            if cy==0:continue
            outgoing=(cy==1) if side==0 else (cy==-1)
            plane=q[h,:,-1 if side==0 else 0,:] if outgoing else q[h,:,-2 if side==0 else 1,:]
            if not outgoing:plane=-np.roll(plane,(-int(cx),-int(cz)),axis=(0,1))
            net[h]=plane.reshape(self.nx//2,2,self.nz//2,2).sum(axis=(1,3),dtype=np.int64)
        return net

    def _coarse_flux(self,q,side):
        j=self.b if side==0 else self.t
        net=np.zeros((19,self.nx//2,self.nz//2),np.int64)
        for h,(cx,cy,cz) in enumerate(C):
            if cy==0:continue
            incoming=(cy==1) if side==0 else (cy==-1)
            if incoming:net[h]=q[h,:,j,:]
            else:net[h]=-np.roll(q[h,:,j+int(cy),:],(-int(cx),-int(cz)),axis=(0,1))
        return net

    def step(self):
        old=self._covered(self.coarse.numpy());self._replace(self.coarse,old)
        self.coarse.step();new=self.coarse.numpy()
        coarse_net=[self._coarse_flux(new,side) for side in (0,1)]
        fine_net=[np.zeros_like(v) for v in coarse_net]
        for substep in range(2):
            for side,sim in enumerate(self.fine):
                q=sim.numpy()
                if side==0:q[:,:,-1:,:]=self._ghost(old,new,substep/2,side)
                else:q[:,:,:1,:]=self._ghost(old,new,substep/2,side)
                self._replace(sim,q);sim.step();fine_net[side]+=self._fine_flux(sim.numpy(),side)
        for side,j in enumerate((self.b,self.t)):
            new[:,:,j,:]=reflux_population_mass(new[:,:,j,:],fine_net[side],coarse_net[side])
        self._replace(self.coarse,new);self.steps+=2

    def ledger(self):
        arrays=[s.numpy()[:,:,1:-1,:] for s in self.fine]
        arrays.append(self.coarse.numpy()[:,:,self.b:self.t+1,:])
        if self.ledger_mode=='int64_limbs':
            from .lbm3d_ledger import population_ledger
            return population_ledger(arrays,C)
        totals=[sum(checked_int64_sum(a[q]) for a in arrays) for q in range(19)]
        if sum(abs(v) for v in totals)>=2**63:
            raise OverflowError('combined int64 population ledger exceeds range')
        return [sum(totals),*[sum(int(C[q,k])*totals[q] for q in range(19)) for k in range(3)]]

    def wall_impulse(self):
        from .lbm3d_ledger import exact_sum,int64_limbs
        out=[0,0,0]
        for side,sim in enumerate(self.fine):
            a=sim.wall_impulse.numpy().reshape((3,)+sim.shape)
            for k in range(3):
                out[k]+=exact_sum(a[k,:,1 if side==0 else -2,:])
        if self.ledger_mode=='int64_limbs':
            for value in out:int64_limbs(value)
            return out
        if any(v<-(2**63) or v>=2**63 for v in out):
            raise OverflowError('combined wall impulse exceeds int64 range')
        return np.asarray(out,dtype=np.int64)

    def profile(self):
        from .lbm3d_mrt_les import fields
        values=[];positions=[]
        for q,bits,force,y in [
            (self.fine[0].numpy()[:,:,1:-1,:],40,self.gf,np.arange(self.nf)+.5),
            (self.coarse.numpy()[:,:,self.b:self.t+1,:],43,2*self.gf,np.arange(self.nf+1,self.h-self.nf,2)),
            (self.fine[1].numpy()[:,:,1:-1,:],40,self.gf,np.arange(self.h-self.nf,self.h)+.5)]:
            rho,u=fields(q,bits);values.extend((u[0]+.5*force/rho).mean(axis=(0,2)));positions.extend(y)
        return np.array(positions),np.array(values)

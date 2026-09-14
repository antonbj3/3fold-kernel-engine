"""Device-resident 2:1 channel coupling, initially checked for laminar flow.

Uses the same volume-weighted integer population convention as the CPU
reference. Each flux element has one writer; reflux uses actual streamed
population masses. No host readback is required between coarse steps.
"""
import numpy as np
import warp as wp
from .lbm3d_mrt_les import V19,I19
from .lbm3d_refinement import ReferenceRefinedChannel

Mat3=wp.types.matrix(shape=(3,3),dtype=wp.float64)
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.func
def convert(f:V19,c:wp.array2d(dtype=wp.int32),w:wp.array(dtype=wp.float64),
            ts:wp.float64,tt:wp.float64,dt:wp.float64,fs:wp.float64,ft:wp.float64,
            scale:wp.float64,cs_source:wp.float64,cs_target:wp.float64,bs:wp.float64,bt:wp.float64,recursive:int,failure:wp.array(dtype=wp.int32)):
    rho=wp.float64(0.0);jx=wp.float64(0.0);jy=wp.float64(0.0);jz=wp.float64(0.0)
    for q in range(19):
        rho+=f[q];jx+=wp.float64(c[q,0])*f[q];jy+=wp.float64(c[q,1])*f[q];jz+=wp.float64(c[q,2])*f[q]
    out=I19()
    if rho<=wp.float64(0.0):
        wp.atomic_max(failure,0,1)
        return out
    ux=jx/rho+wp.float64(.5)*fs/rho;uy=jy/rho;uz=jz/rho
    u=wp.vec3d(ux,uy,uz);usq=ux*ux+uy*uy+uz*uz
    eq=V19();pi=Mat3()
    for q in range(19):
        cu=wp.float64(c[q,0])*ux+wp.float64(c[q,1])*uy+wp.float64(c[q,2])*uz
        eq[q]=w[q]*rho*(wp.float64(1.0)+wp.float64(3.0)*cu+wp.float64(4.5)*cu*cu-wp.float64(1.5)*usq)
    for a in range(3):
        for b in range(3):
            value=wp.float64(0.0)
            for q in range(19):value+=wp.float64(c[q,a]*c[q,b])*(f[q]-eq[q])
            if a==0:value+=wp.float64(.5)*u[b]*fs
            if b==0:value+=wp.float64(.5)*u[a]*fs
            pi[a,b]=value
    norm2=wp.float64(0.0)
    for a in range(3):
        for b in range(3):norm2+=pi[a,b]*pi[a,b]
    norm=wp.sqrt(norm2)
    effective_source=wp.float64(.5)*(ts+wp.sqrt(ts*ts+wp.float64(18.0)*wp.sqrt(wp.float64(2.0))*cs_source*cs_source*norm/rho))
    effective_target=tt+wp.float64(4.5)*wp.sqrt(wp.float64(2.0))*cs_target*cs_target*dt*norm/(rho*effective_source)
    trace=wp.float64(0.0)
    if bs>wp.float64(0.0):
        trace=(pi[0,0]+pi[1,1]+pi[2,2])/wp.float64(3.0)
        devnorm=wp.float64(0.0)
        for a in range(3):
            for b in range(3):
                v=pi[a,b]
                if a==b:v-=trace
                devnorm+=v*v
        ac=dt*dt/(effective_source*effective_source)*devnorm
        target_trace=dt*bt/bs*trace
        bc=wp.float64(3.0)*target_trace*target_trace
        factor=wp.float64(4.5)*wp.sqrt(wp.float64(2.0))*cs_target*cs_target/rho
        lo=tt;hi=tt+factor*wp.sqrt(ac+bc/(tt*tt))
        for iteration in range(48):
            mid=wp.float64(.5)*(lo+hi)
            if mid-tt-factor*wp.sqrt(ac+bc/(mid*mid))>wp.float64(0.0):hi=mid
            else:lo=mid
        effective_target=wp.float64(.5)*(lo+hi)
    physical_pi=Mat3()
    for a in range(3):
        for b in range(3):
            value=pi[a,b]*(dt*effective_target/effective_source)
            if bs>wp.float64(0.0) and a==b:
                value=(pi[a,b]-trace)*(dt*effective_target/effective_source)+dt*bt/bs*trace
            physical_pi[a,b]=value
            if a==0:value-=wp.float64(.5)*u[b]*ft
            if b==0:value-=wp.float64(.5)*u[a]*ft
            pi[a,b]=value
    om=wp.int64(0);ox=wp.int64(0);oy=wp.int64(0);oz=wp.int64(0)
    for q in range(19):
        correction=wp.float64(0.0)
        for a in range(3):
            for b in range(3):
                h=wp.float64(c[q,a]*c[q,b])
                if a==b:h-=wp.float64(1.0)/wp.float64(3.0)
                correction+=h*pi[a,b]
        value=eq[q]+wp.float64(4.5)*w[q]*correction-wp.float64(1.5)*w[q]*wp.float64(c[q,0])*ft
        if recursive != 0:
            cx=wp.float64(c[q,0]);cy=wp.float64(c[q,1]);cz=wp.float64(c[q,2])
            cu=cx*ux+cy*uy+cz*uz
            cpc=wp.float64(0.0);cpu=wp.float64(0.0);trace=wp.float64(0.0)
            for a in range(3):
                trace+=physical_pi[a,a]
                for b in range(3):
                    cpc+=wp.float64(c[q,a]*c[q,b])*physical_pi[a,b]
                    cpu+=wp.float64(c[q,a])*physical_pi[a,b]*u[b]
            value+=w[q]*rho*(wp.float64(4.5)*cu*cu*cu-wp.float64(4.5)*cu*usq)
            value+=w[q]*(wp.float64(13.5)*cu*cpc-wp.float64(4.5)*cu*trace-wp.float64(9.0)*cpu)
            value-=wp.float64(.5)*w[q]*ft*((wp.float64(13.5)*cu*cu-wp.float64(4.5)*usq)*cx-wp.float64(9.0)*cu*ux)
        if not wp.isfinite(value) or wp.abs(value)>wp.float64(32.0):
            wp.atomic_max(failure,0,1)
            return I19()
        fq=wp.int64(wp.round(value*scale));out[q]=fq
        om+=fq;ox+=fq*wp.int64(c[q,0]);oy+=fq*wp.int64(c[q,1]);oz+=fq*wp.int64(c[q,2])
    mass=wp.int64(wp.round(rho*scale))
    dx=wp.int64(wp.round((rho*ux-wp.float64(.5)*ft)*scale))-ox
    dy=wp.int64(wp.round(rho*uy*scale))-oy;dz=wp.int64(wp.round(rho*uz*scale))-oz
    out[1]+=dx;out[3]+=dy;out[5]+=dz;out[0]+=mass-om-dx-dy-dz
    return out


@wp.kernel
def restrict_covered(fine:wp.array4d(dtype=wp.int64),coarse:wp.array4d(dtype=wp.int64),
                     c:wp.array2d(dtype=wp.int32),w:wp.array(dtype=wp.float64),
                     offset:int,tf:wp.float64,tc:wp.float64,g:wp.float64,csf:wp.float64,csc:wp.float64,bf:wp.float64,bc:wp.float64,recursive:int,failure:wp.array(dtype=wp.int32)):
    i,j,k=wp.tid();f=V19()
    for q in range(19):
        total=wp.int64(0)
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):total+=fine[q,2*i+dx,1+2*j+dy,2*k+dz]
        f[q]=wp.float64(total)/wp.float64(8796093022208.0)
    out=convert(f,c,w,tf,tc,wp.float64(2.0),g,wp.float64(2.0)*g,wp.float64(8796093022208.0),csf,csc,bf,bc,recursive,failure)
    for q in range(19):coarse[q,i,j+offset,k]=out[q]


@wp.kernel
def fill_ghost(old:wp.array4d(dtype=wp.int64),predicted:wp.array4d(dtype=wp.int64),fine:wp.array4d(dtype=wp.int64),
               c:wp.array2d(dtype=wp.int32),w:wp.array(dtype=wp.float64),
               ncx:int,ncz:int,jc:int,jf:int,fy:wp.float64,alpha:wp.float64,
               tf:wp.float64,tc:wp.float64,g:wp.float64,csf:wp.float64,csc:wp.float64,bf:wp.float64,bc:wp.float64,recursive:int,failure:wp.array(dtype=wp.int32)):
    i,k=wp.tid();ix=i/2;iz=k/2;fx=wp.float64(.25);fz=wp.float64(.25)
    if i%2==0:ix-=1;fx=wp.float64(.75)
    if k%2==0:iz-=1;fz=wp.float64(.75)
    f=V19()
    for dx in range(2):
        for dz in range(2):
            wx=wp.float64(1.0)-fx;wz=wp.float64(1.0)-fz
            if dx==1:wx=fx
            if dz==1:wz=fz
            ci=(ix+dx+ncx)%ncx;ck=(iz+dz+ncz)%ncz
            for q in range(19):
                lo=((wp.float64(1.0)-alpha)*wp.float64(old[q,ci,jc,ck])+alpha*wp.float64(predicted[q,ci,jc,ck]))/wp.float64(8796093022208.0)
                hi=((wp.float64(1.0)-alpha)*wp.float64(old[q,ci,jc+1,ck])+alpha*wp.float64(predicted[q,ci,jc+1,ck]))/wp.float64(8796093022208.0)
                f[q]+=((wp.float64(1.0)-fy)*lo+fy*hi)*wx*wz
    out=convert(f,c,w,tc,tf,wp.float64(.5),wp.float64(2.0)*g,g,wp.float64(1099511627776.0),csc,csf,bc,bf,recursive,failure)
    for q in range(19):fine[q,i,jf,k]=out[q]


@wp.kernel
def initialize_flux(coarse:wp.array4d(dtype=wp.int64),delta:wp.array3d(dtype=wp.int64),
                    c:wp.array2d(dtype=wp.int32),j:int,side:int,ncx:int,ncz:int):
    q,i,k=wp.tid();cx=c[q,0];cy=c[q,1];cz=c[q,2]
    value=wp.int64(0)
    if cy!=0:
        incoming=(cy==1 and side==0) or (cy==-1 and side==1)
        if incoming:value=-coarse[q,i,j,k]
        else:value=coarse[q,(i+cx+ncx)%ncx,j+cy,(k+cz+ncz)%ncz]
    delta[q,i,k]=value


@wp.kernel
def collect_flux(fine:wp.array4d(dtype=wp.int64),delta:wp.array3d(dtype=wp.int64),
                 c:wp.array2d(dtype=wp.int32),side:int,nf:int,nx:int,nz:int):
    q,i,k=wp.tid();cx=c[q,0];cy=c[q,1];cz=c[q,2]
    if cy==0:return
    outgoing=(cy==1 and side==0) or (cy==-1 and side==1)
    j=int(0)
    if side==0:j=nf+1
    if not outgoing:
        j=1
        if side==0:j=nf
    total=wp.int64(0)
    for dx in range(2):
        for dz in range(2):
            fi=2*i+dx;fk=2*k+dz
            if not outgoing:fi=(fi+cx+nx)%nx;fk=(fk+cz+nz)%nz
            total+=fine[q,fi,j,fk]
    if outgoing:delta[q,i,k]+=total
    else:delta[q,i,k]-=total


@wp.kernel
def project_reflux(delta:wp.array3d(dtype=wp.int64),c:wp.array2d(dtype=wp.int32),w:wp.array(dtype=wp.float64),second_order:int):
    i,k=wp.tid();mass=wp.int64(0);jx=wp.int64(0);jy=wp.int64(0);jz=wp.int64(0)
    for q in range(19):
        v=delta[q,i,k];mass+=v;jx+=v*wp.int64(c[q,0]);jy+=v*wp.int64(c[q,1]);jz+=v*wp.int64(c[q,2])
    stress=Mat3()
    if second_order!=0:
        for a in range(3):
            for b in range(3):
                value=wp.float64(0.0)
                for q in range(19):
                    h=wp.float64(c[q,a]*c[q,b])
                    if a==b:h-=wp.float64(1.0)/wp.float64(3.0)
                    value+=h*wp.float64(delta[q,i,k])
                stress[a,b]=value
    out=I19();dm=mass;dx=jx;dy=jy;dz=jz
    for q in range(19):
        value=w[q]*(wp.float64(mass)+wp.float64(3.0)*(wp.float64(c[q,0])*wp.float64(jx)+wp.float64(c[q,1])*wp.float64(jy)+wp.float64(c[q,2])*wp.float64(jz)))
        if second_order!=0:
            correction=wp.float64(0.0)
            for a in range(3):
                for b in range(3):
                    h=wp.float64(c[q,a]*c[q,b])
                    if a==b:h-=wp.float64(1.0)/wp.float64(3.0)
                    correction+=h*stress[a,b]
            value+=wp.float64(4.5)*w[q]*correction
        v=wp.int64(wp.round(value));out[q]=v;dm-=v;dx-=v*wp.int64(c[q,0]);dy-=v*wp.int64(c[q,1]);dz-=v*wp.int64(c[q,2])
    out[1]+=dx;out[3]+=dy;out[5]+=dz;out[0]+=dm-dx-dy-dz
    for q in range(19):delta[q,i,k]=out[q]


@wp.kernel
def apply_reflux(coarse:wp.array4d(dtype=wp.int64),delta:wp.array3d(dtype=wp.int64),j:int):
    q,i,k=wp.tid();coarse[q,i,j,k]+=delta[q,i,k]


@wp.kernel
def apply_balanced_reflux(coarse:wp.array4d(dtype=wp.int64),fine:wp.array4d(dtype=wp.int64),delta:wp.array3d(dtype=wp.int64),jc:int,jf:int):
    q,i,k=wp.tid();d=delta[q,i,k]
    half=d/wp.int64(2)
    if d<wp.int64(0) and d%wp.int64(2)!=wp.int64(0):half-=wp.int64(1)
    rest=d-half;base=rest/wp.int64(8)
    if rest<wp.int64(0) and rest%wp.int64(8)!=wp.int64(0):base-=wp.int64(1)
    remainder=rest-wp.int64(8)*base
    coarse[q,i,jc,k]+=half
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                value=base
                if wp.int64(4*dx+2*dy+dz)<remainder:value+=wp.int64(1)
                fine[q,2*i+dx,jf+dy,2*k+dz]+=value


class RefinedChannelGPU(ReferenceRefinedChannel):
    """Resident coupling path; device='cpu' is useful for reference checks."""
    def __init__(self,*,device='cuda:0',**kwargs):
        super().__init__(device=device,**kwargs)
        self.device=device
        self.delta=[wp.zeros((19,self.nx//2,self.nz//2),dtype=wp.int64,device=device) for _ in range(2)]
    def step(self):
        c,w=self.coarse.args[1],self.coarse.args[2]
        shared=[wp.float64(self.tf),wp.float64(self.tc),wp.float64(self.gf),wp.float64(self.csf),wp.float64(self.csc),wp.float64(self.bf or 0.),wp.float64(self.bc or 0.),int(self.recursive)]
        for side,sim in enumerate(self.fine):
            offset=1 if side==0 else self.t+1
            wp.launch(restrict_covered,(self.nx//2,self.nf//2,self.nz//2),[sim.a,self.coarse.a,c,w,offset,*shared,sim.failure],device=self.device)
        self.coarse.step()
        for side,j in enumerate((self.b,self.t)):
            wp.launch(initialize_flux,(19,self.nx//2,self.nz//2),[self.coarse.a,self.delta[side],c,j,side,self.nx//2,self.nz//2],device=self.device)
        for substep in range(2):
            for side,sim in enumerate(self.fine):
                jc=self.b-1 if side==0 else self.t;jf=self.nf+1 if side==0 else 0
                fy=.75 if side==0 else .25
                wp.launch(fill_ghost,(self.nx,self.nz),[self.coarse.b,self.coarse.a,sim.a,c,w,self.nx//2,self.nz//2,jc,jf,wp.float64(fy),wp.float64(substep/2),*shared,sim.failure],device=self.device)
                sim.step()
                wp.launch(collect_flux,(19,self.nx//2,self.nz//2),[sim.a,self.delta[side],c,side,self.nf,self.nx,self.nz],device=self.device)
        for side,j in enumerate((self.b,self.t)):
            if self.conserved_reflux:wp.launch(project_reflux,(self.nx//2,self.nz//2),[self.delta[side],c,w,int(self.stress_reflux)],device=self.device)
            if self.balanced_reflux:
                jf=self.nf-1 if side==0 else 1
                wp.launch(apply_balanced_reflux,(19,self.nx//2,self.nz//2),[self.coarse.a,self.fine[side].a,self.delta[side],j,jf],device=self.device)
            else:
                wp.launch(apply_reflux,(19,self.nx//2,self.nz//2),[self.coarse.a,self.delta[side],j],device=self.device)
        self.steps+=2

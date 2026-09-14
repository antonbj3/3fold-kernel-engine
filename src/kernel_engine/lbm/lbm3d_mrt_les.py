"""Experimental D3Q19 raw-moment MRT/Smagorinsky, fused collision and streaming.

One thread owns one voxel. Populations are signed int64 multiples of 2**-40;
collision arithmetic is float64. Quantization is part of this numerical model,
not a rounded diagnostic: residual momentum goes to the three positive axial
populations and residual mass to the rest population before streaming. Thus
unforced collision conserves integer mass AND momentum locally. Periodic push
streaming is a permutation; optional solid links use halfway bounce-back.

The basis follows the raw polynomial construction in Li et al. (2019), equations
7–10, with an equivalent ordering of third-order moments:
https://discovery.ucl.ac.uk/10076000/1/Luo%202019%20Computers%20and%20Fluids%20Accepted.pdf
The optional hermite_mrt mode projects the nine higher raw-moment rows onto
the complement of the conserved and second-order Hermite subspaces. It is
algebraically equivalent to a second-order Hermite MRT with higher rates one
(projected regularization); see https://arxiv.org/abs/physics/0506157.
The original raw-moment mode remains available as the failed control.
All six second-order moments share the effective viscosity relaxation rate;
the nine higher moments relax at one. This is MRT, not cumulant collision.
Smagorinsky uses the Frobenius norm of the nonequilibrium stress, with
 tau_eff = (tau + sqrt(tau**2 + 18*sqrt(2)*Cs**2*||Pi||/rho))/2.
See OpenLB collisionLES.h, SmagorinskyEffectiveOmega (no extra sqrt(2) in norm):
https://www.openlb.net/DoxyGen/html/d7/dcf/collisionLES_8h_source.html

No forcing, inlet/outlet, turbulence accuracy or production certification is
implied by this first collision experiment. Existing aero solvers are unchanged.
"""
import numpy as np
import warp as wp

C = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
              [1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[1,0,1],[-1,0,-1],
              [1,0,-1],[-1,0,1],[0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]],dtype=np.int32)
W = np.array([1/3]+[1/18]*6+[1/36]*12, dtype=np.float64)
OPP = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17],dtype=np.int32)
x,y,z = C.T.astype(np.float64)
r2 = x*x+y*y+z*z
M = np.array([np.ones(19),x,y,z,r2,3*x*x-r2,y*y-z*z,x*y,x*z,y*z,
              x*y*y,x*z*z,y*x*x,y*z*z,z*x*x,z*y*y,x*x*y*y,x*x*z*z,y*y*z*z])
MI = np.linalg.inv(M)
# Population-space projectors at rest. Orthogonalize only the ghost rows;
# the first ten moment definitions used by the stress reconstruction stay exact.
E_PROJECTOR = W[:,None]*(1+3*(C@C.T))
P2_PROJECTOR = 4.5*W[:,None]*((C@C.T)**2-(r2[:,None]+r2[None,:])/3+1/3)
M_HERMITE = M.copy()
M_HERMITE[10:] = M[10:]@(np.eye(19)-E_PROJECTOR-P2_PROJECTOR)
MI_HERMITE = np.linalg.inv(M_HERMITE)
V19 = wp.types.vector(length=19,dtype=wp.float64)
I19 = wp.types.vector(length=19,dtype=wp.int64)
Mat19 = wp.types.matrix(shape=(19,19),dtype=wp.float64)
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def collide_stream(a: wp.array4d(dtype=wp.int64), b: wp.array4d(dtype=wp.int64),
                   solid: wp.array3d(dtype=wp.int32), c: wp.array2d(dtype=wp.int32),
                   w: wp.array(dtype=wp.float64), opp: wp.array(dtype=wp.int32),
                   m: Mat19, mi: Mat19, tau: wp.float64, cs: wp.float64,
                   scale: wp.float64, mode: int, nx: int, ny: int, nz: int,
                   failure: wp.array(dtype=wp.int32)):
    i,j,k = wp.tid()
    g = V19()
    qout = I19()
    mass = wp.int64(0)
    jx = wp.int64(0); jy = wp.int64(0); jz = wp.int64(0)
    for q in range(19):
        fq = a[q,i,j,k]
        qout[q] = fq
        g[q] = wp.float64(fq)/scale
        mass += fq
        jx += fq*wp.int64(c[q,0]); jy += fq*wp.int64(c[q,1]); jz += fq*wp.int64(c[q,2])
    if solid[i,j,k] != 0:
        for q in range(19):
            b[q,i,j,k] = a[q,i,j,k]
        return
    rho = wp.float64(mass)/scale
    valid = mass > wp.int64(0)
    if valid:
        ux = wp.float64(jx)/wp.float64(mass)
        uy = wp.float64(jy)/wp.float64(mass)
        uz = wp.float64(jz)/wp.float64(mass)
        eq = V19()
        usq = ux*ux+uy*uy+uz*uz
        for q in range(19):
            cu = wp.float64(c[q,0])*ux+wp.float64(c[q,1])*uy+wp.float64(c[q,2])*uz
            eq[q] = w[q]*rho*(wp.float64(1.0)+wp.float64(3.0)*cu+wp.float64(4.5)*cu*cu-wp.float64(1.5)*usq)
        delta = g-eq
        dm = m*delta
        # Recover the six stress components from trace/deviatoric raw moments.
        pxx = (dm[4]+dm[5])/wp.float64(3.0)
        pyy = (dm[4]-pxx+dm[6])/wp.float64(2.0)
        pzz = (dm[4]-pxx-dm[6])/wp.float64(2.0)
        norm = wp.sqrt(pxx*pxx+pyy*pyy+pzz*pzz+wp.float64(2.0)*(dm[7]*dm[7]+dm[8]*dm[8]+dm[9]*dm[9]))
        teff = wp.float64(0.5)*(tau+wp.sqrt(tau*tau+wp.float64(18.0)*wp.sqrt(wp.float64(2.0))*cs*cs*norm/rho))
        omega = wp.float64(1.0)/teff
        post = g-omega*delta
        if mode == 1:
            relaxed = V19()
            for h in range(19):
                rate = wp.float64(0.0)
                if h >= 4:
                    rate = wp.float64(1.0)
                    if h < 10:
                        rate = omega
                relaxed[h] = rate*dm[h]
            post = g-mi*relaxed
        for q in range(19):
            if not wp.isfinite(post[q]) or wp.abs(post[q]) > wp.float64(32.0):
                valid = False
        if valid:
            outmass = wp.int64(0)
            ox = wp.int64(0); oy = wp.int64(0); oz = wp.int64(0)
            for q in range(19):
                fq = wp.int64(wp.round(post[q]*scale))
                qout[q] = fq
                outmass += fq
                ox += fq*wp.int64(c[q,0]); oy += fq*wp.int64(c[q,1]); oz += fq*wp.int64(c[q,2])
            dx = jx-ox; dy = jy-oy; dz = jz-oz
            qout[1] += dx; qout[3] += dy; qout[5] += dz
            qout[0] += mass-outmass-dx-dy-dz
    if not valid:
        # Mark the run failed before any unsafe float-to-int conversion. No
        # repaired result can pass: the caller must inspect this sticky flag.
        wp.atomic_max(failure,0,1)
    for q in range(19):
        ti = (i+c[q,0]+nx)%nx
        tj = (j+c[q,1]+ny)%ny
        tk = (k+c[q,2]+nz)%nz
        if solid[ti,tj,tk] != 0:
            b[opp[q],i,j,k] = qout[q]
        else:
            b[q,ti,tj,tk] = qout[q]


def equilibrium(rho, velocity):
    rho = np.asarray(rho,dtype=np.float64)
    velocity = np.asarray(velocity,dtype=np.float64)
    if velocity.shape != (3,)+rho.shape:
        raise ValueError('velocity must have shape (3, *rho.shape)')
    cu = np.einsum('qd,d...->q...',C,velocity)
    return W.reshape((19,)+(1,)*rho.ndim)*rho*(1+3*cu+4.5*cu*cu-1.5*np.sum(velocity**2,axis=0))


def quantize(f, bits=40):
    f = np.asarray(f,dtype=np.float64)
    if bits not in (32,36,40,43,44) or f.ndim != 4 or f.shape[0] != 19:
        raise ValueError('expected D3Q19 populations and bits in (32,36,40,43,44)')
    if not np.isfinite(f).all() or np.max(np.abs(f)) > 2 or int(np.prod(f.shape[1:]))*2**bits >= 2**63:
        raise ValueError('initial population/ledger range exceeded')
    scale = 2**bits
    q = np.rint(f*scale).astype(np.int64)
    mass = np.rint(f.sum(axis=0)*scale).astype(np.int64)
    target = np.rint(np.einsum('qd,q...->d...',C,f)*scale).astype(np.int64)
    current = np.einsum('qd,q...->d...',C.astype(np.int64),q)
    d = target-current
    q[1] += d[0]; q[3] += d[1]; q[5] += d[2]
    q[0] += mass-q.sum(axis=0,dtype=np.int64)
    ledger(q)  # Validate actual integer totals as well as the cell-count bound.
    return q


def fields(q,bits=40):
    f = np.asarray(q,dtype=np.float64)/2**bits
    rho = f.sum(axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        u = np.einsum('qd,q...->d...',C,f)/rho
    return rho,u


def ledger(q):
    """Exact integer totals, with explicit absolute-sum overflow protection."""
    q = np.asarray(q)
    if q.dtype != np.int64 or q.shape[0] != 19:
        raise ValueError("expected int64 D3Q19 populations")
    totals = []
    for plane in q:
        bound = max(abs(int(plane.min())), abs(int(plane.max()))) * plane.size
        if bound >= 2**63:
            raise OverflowError("population-plane reduction range exceeded")
        totals.append(int(plane.sum(dtype=np.int64)))
    if sum(abs(v) for v in totals) >= 2**63:
        raise OverflowError('int64 ledger range exceeded')
    t = np.array(totals,dtype=np.int64)
    return [int(t.sum()),*[int(v) for v in C.astype(np.int64).T@t]]


class Simulation:
    def __init__(self,q,*,tau,cs=0.1,mode='mrt',bits=40,solid=None,device='cpu'):
        q = np.asarray(q)
        if q.dtype != np.int64 or q.ndim != 4 or q.shape[0] != 19 or min(q.shape[1:])<2:
            raise ValueError('expected int64 (19,nx,ny,nz), all dimensions >=2')
        if not np.isfinite([tau,cs]).all() or tau<=0.5 or cs<0 or bits not in (32,36,40,43,44) or mode not in ('mrt','bgk','hermite_mrt'):
            raise ValueError('invalid collision parameters')
        if np.max(np.abs(q.astype(np.float64)))>32*2**bits or int(np.prod(q.shape[1:]))*2**bits>=2**63:
            raise ValueError('population/ledger range exceeded')
        ledger(q)  # Reject unsafe reductions before device allocation.
        if solid is None: solid=np.zeros(q.shape[1:],dtype=np.int32)
        solid=np.asarray(solid,dtype=np.int32)
        if solid.shape!=q.shape[1:] or not np.isin(solid,[0,1]).all(): raise ValueError('invalid solid mask')
        self.device=device; self.bits=bits; self.shape=q.shape[1:]
        self.a=wp.array(q,dtype=wp.int64,device=device);self.b=wp.empty_like(self.a)
        self.failure=wp.zeros(1,dtype=wp.int32,device=device)
        self.args=[wp.array(solid,dtype=wp.int32,device=device),wp.array(C,dtype=wp.int32,device=device),
                   wp.array(W,dtype=wp.float64,device=device),wp.array(OPP,dtype=wp.int32,device=device),
                   Mat19(M_HERMITE if mode=='hermite_mrt' else M),
                   Mat19(MI_HERMITE if mode=='hermite_mrt' else MI),wp.float64(tau),wp.float64(cs),wp.float64(2**bits),
                   int(mode!='bgk'),*self.shape,self.failure]
    def step(self,steps=1):
        if not isinstance(steps,int) or steps<0: raise ValueError('steps must be nonnegative integer')
        for _ in range(steps):
            wp.launch(collide_stream,self.shape,[self.a,self.b,*self.args],device=self.device)
            self.a,self.b=self.b,self.a
    def numpy(self):
        return self.a.numpy()

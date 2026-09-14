"""Forced 3D channel sibling of the checked D3Q19 Hermite MRT collision.

Guo forcing is transformed with (I-S/2) in moment space. The prescribed body
force is a constant integer momentum-density increment per fluid voxel/step;
the physical velocity includes its half-step shift. Halfway bounce-back records
the exact opposite wall impulse, allowing a closed integer momentum balance.
No channel DNS or refinement validation is implied by these building blocks.
"""
import numpy as np
import warp as wp
from .lbm3d_mrt_les import Simulation,V19,I19,Mat19

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def forced_collide_stream(a: wp.array4d(dtype=wp.int64), b: wp.array4d(dtype=wp.int64),
                   solid: wp.array3d(dtype=wp.int32), c: wp.array2d(dtype=wp.int32),
                   w: wp.array(dtype=wp.float64), opp: wp.array(dtype=wp.int32),
                   m: Mat19, mi: Mat19, tau: wp.float64, cs: wp.float64,
                   scale: wp.float64, mode: int, nx: int, ny: int, nz: int,
                   failure: wp.array(dtype=wp.int32), force_x: wp.int64,
                   wall_impulse: wp.array(dtype=wp.int64)):
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
        ux = (wp.float64(jx)+wp.float64(0.5)*wp.float64(force_x))/wp.float64(mass)
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
        # Guo strain correction Pi_neq + (uF+Fu)/2, PRE86,016705 eq32.
        force = wp.float64(force_x)/scale
        pxx += ux*force
        pxy = dm[7]+wp.float64(0.5)*uy*force
        pxz = dm[8]+wp.float64(0.5)*uz*force
        norm = wp.sqrt(pxx*pxx+pyy*pyy+pzz*pzz+wp.float64(2.0)*(pxy*pxy+pxz*pxz+dm[9]*dm[9]))
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
        if force_x != wp.int64(0):
            source = V19()
            force = wp.float64(force_x)/scale
            for q in range(19):
                cx = wp.float64(c[q,0]); cy = wp.float64(c[q,1]); cz = wp.float64(c[q,2])
                cu = cx*ux+cy*uy+cz*uz
                source[q] = w[q]*force*(wp.float64(3.0)*(cx-ux)+wp.float64(9.0)*cu*cx)
            if mode == 1:
                ms = m*source
                for h in range(19):
                    rate = wp.float64(0.0)
                    if h >= 4:
                        rate = wp.float64(1.0)
                        if h < 10:
                            rate = omega
                    ms[h] = (wp.float64(1.0)-wp.float64(0.5)*rate)*ms[h]
                post += mi*ms
            else:
                post += (wp.float64(1.0)-wp.float64(0.5)*omega)*source
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
            dx = jx+force_x-ox; dy = jy-oy; dz = jz-oz
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
            for axis in range(3):
                # One counter per owning voxel: no contested global atomics.
                idx = axis*nx*ny*nz+(i*ny+j)*nz+k
                change = wp.int64(2)*qout[q]*wp.int64(c[q,axis])
                previous = wall_impulse[idx]
                safe = True
                if change > wp.int64(0) and previous > wp.int64(9223372036854775807)-change:
                    safe = False
                if change < wp.int64(0) and previous < wp.int64(-9223372036854775807)-change:
                    safe = False
                if safe:
                    wall_impulse[idx] = previous+change
                else:
                    wp.atomic_max(failure,0,1)
        else:
            b[q,ti,tj,tk] = qout[q]

class ChannelSimulation(Simulation):
    def __init__(self,q,*,force_density,tau,cs=.1,bits=40,solid=None,device='cpu',ledger_mode='int64'):
        if not np.isfinite(force_density) or abs(force_density)>.01:
            raise ValueError('invalid body force')
        super().__init__(q,tau=tau,cs=cs,mode='hermite_mrt',bits=bits,solid=solid,device=device,ledger_mode=ledger_mode)
        self.force_units=int(np.rint(force_density*2**bits))
        self.force_density=self.force_units/2**bits
        self.wall_impulse=wp.zeros(3*int(np.prod(self.shape)),dtype=wp.int64,device=device)
        self.args += [wp.int64(self.force_units),self.wall_impulse]
    def step(self,steps=1):
        if not isinstance(steps,int) or steps<0:raise ValueError('steps must be nonnegative integer')
        for _ in range(steps):
            wp.launch(forced_collide_stream,self.shape,[self.a,self.b,*self.args],device=self.device)
            self.a,self.b=self.b,self.a

    def wall_impulse_numpy(self):
        # Individual int64 counters can have large opposing normal impulses.
        # Sum only nonzero slots with exact Python integers, then range-check
        # the physical net vector before representing it in the int64 ledger.
        out=[]
        for row in self.wall_impulse.numpy().reshape(3,-1):
            total=sum(map(int,row[row!=0]))
            if not -(2**63)<total<2**63:raise OverflowError('net wall impulse exceeds int64')
            out.append(total)
        return np.array(out,dtype=np.int64)

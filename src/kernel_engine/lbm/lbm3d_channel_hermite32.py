"""Experimental FP32 recursive H3 correction only; remaining collision FP64.
Derived source SHA256: 4befcbd9dd31de11fe5f54a3af4f9734d7155a0f8c08c1a15cae34d4f0f91a94
"""
import warp as wp
from .lbm3d_mrt_les import V19,I19,Mat19,M_HERMITE,MI_HERMITE
from .lbm3d_channel import ChannelSimulation
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
FIXED_M=wp.constant(Mat19(M_HERMITE))
FIXED_MI=wp.constant(Mat19(MI_HERMITE))

@wp.kernel
def hermite32_collide_stream(a: wp.array4d(dtype=wp.int64), b: wp.array4d(dtype=wp.int64),
                   solid: wp.array3d(dtype=wp.int32), c: wp.array2d(dtype=wp.int32),
                   w: wp.array(dtype=wp.float64), opp: wp.array(dtype=wp.int32),
                   tau: wp.float64, cs: wp.float64,
                   scale: wp.float64, mode: int, nx: int, ny: int, nz: int,
                   failure: wp.array(dtype=wp.int32), force_x: wp.int64,
                   wall_impulse: wp.array(dtype=wp.int64), bulk_rate: wp.float64, recursive: int):
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
            if recursive != 0:
                eq[q] += w[q]*rho*(wp.float64(4.5)*cu*cu*cu-wp.float64(4.5)*cu*usq)
        delta = g-eq
        dm = FIXED_M*delta
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
                        if h == 4 and bulk_rate > wp.float64(0.0):
                            rate = bulk_rate
                relaxed[h] = rate*dm[h]
            post = g-FIXED_MI*relaxed
        if recursive != 0:
            # Recursive third Hermite nonequilibrium from the post-collision
            # physical stress. Guo half-force correction is already in pxx,
            # pxy and pxz. This is not a cumulant collision.
            factor = wp.float64(1.0)-omega
            axx=factor*pxx;ayy=factor*pyy;azz=factor*pzz
            axy=factor*pxy;axz=factor*pxz;ayz=factor*dm[9]
            if bulk_rate > wp.float64(0.0):
                trace_shift=(omega-bulk_rate)*(pxx+pyy+pzz)/wp.float64(3.0)
                axx+=trace_shift;ayy+=trace_shift;azz+=trace_shift
            atrace=axx+ayy+azz
            pux=axx*ux+axy*uy+axz*uz
            puy=axy*ux+ayy*uy+ayz*uz
            puz=axz*ux+ayz*uy+azz*uz
            ux_low=wp.float32(ux)
            uy_low=wp.float32(uy)
            uz_low=wp.float32(uz)
            axx_low=wp.float32(axx)
            ayy_low=wp.float32(ayy)
            azz_low=wp.float32(azz)
            axy_low=wp.float32(axy)
            axz_low=wp.float32(axz)
            ayz_low=wp.float32(ayz)
            atrace_low=wp.float32(atrace)
            pux_low=wp.float32(pux)
            puy_low=wp.float32(puy)
            puz_low=wp.float32(puz)
            for q in range(19):
                cx_low=wp.float32(c[q,0]);cy_low=wp.float32(c[q,1]);cz_low=wp.float32(c[q,2])
                cu_low=cx_low*ux_low+cy_low*uy_low+cz_low*uz_low
                cpc_low=cx_low*cx_low*axx_low+cy_low*cy_low*ayy_low+cz_low*cz_low*azz_low+wp.float32(2.0)*(cx_low*cy_low*axy_low+cx_low*cz_low*axz_low+cy_low*cz_low*ayz_low)
                cpu_low=cx_low*pux_low+cy_low*puy_low+cz_low*puz_low
                post[q]+=wp.float64(wp.float32(w[q])*(wp.float32(13.5)*cu_low*cpc_low-wp.float32(4.5)*cu_low*atrace_low-wp.float32(9.0)*cpu_low))
        if force_x != wp.int64(0):
            source = V19()
            force = wp.float64(force_x)/scale
            for q in range(19):
                cx = wp.float64(c[q,0]); cy = wp.float64(c[q,1]); cz = wp.float64(c[q,2])
                cu = cx*ux+cy*uy+cz*uz
                source[q] = w[q]*force*(wp.float64(3.0)*(cx-ux)+wp.float64(9.0)*cu*cx)
                if recursive != 0:
                    source[q]+=w[q]*force*((wp.float64(13.5)*cu*cu-wp.float64(4.5)*usq)*cx-wp.float64(9.0)*cu*ux)
            if mode == 1:
                ms = FIXED_M*source
                for h in range(19):
                    rate = wp.float64(0.0)
                    if h >= 4:
                        rate = wp.float64(1.0)
                        if h < 10:
                            rate = omega
                            if h == 4 and bulk_rate > wp.float64(0.0):
                                rate = bulk_rate
                    ms[h] = (wp.float64(1.0)-wp.float64(0.5)*rate)*ms[h]
                post += FIXED_MI*ms
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

class Hermite32ChannelSimulation(ChannelSimulation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.specialized_args=self.args[:4]+self.args[6:]
    def step(self,steps=1):
        if not isinstance(steps,int) or steps<0:raise ValueError('steps must be nonnegative integer')
        for _ in range(steps):
            wp.launch(hermite32_collide_stream,self.shape,[self.a,self.b,*self.specialized_args],device=self.device)
            self.a,self.b=self.b,self.a

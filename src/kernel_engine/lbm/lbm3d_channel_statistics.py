"""Deterministic plane statistics for the forced D3Q19 channel.

Each instantaneous scalar is rounded to 2^-32 (ties away from zero) before
int64 accumulation.
Caller bounds sample count and resets between blocks. These are diagnostic
moments; the simulation's populations and conservation ledger remain at2^-40.
"""
import warp as wp
V10=wp.types.vector(length=10,dtype=wp.float64)
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def accumulate(a:wp.array4d(dtype=wp.int64),solid:wp.array3d(dtype=wp.int32),
               c:wp.array2d(dtype=wp.int32),scale:wp.float64,force:wp.int64,
               out:wp.array2d(dtype=wp.int64),failure:wp.array(dtype=wp.int32)):
    i,j,k=wp.tid()
    if solid[i,j,k] != 0:return
    mass=wp.int64(0);jx=wp.int64(0);jy=wp.int64(0);jz=wp.int64(0)
    for q in range(19):
        v=a[q,i,j,k];mass+=v
        jx+=v*wp.int64(c[q,0]);jy+=v*wp.int64(c[q,1]);jz+=v*wp.int64(c[q,2])
    if mass<=wp.int64(0):
        wp.atomic_max(failure,0,1)
        return
    rho=wp.float64(mass)/scale
    ux=(wp.float64(jx)+wp.float64(.5)*wp.float64(force))/wp.float64(mass)
    uy=wp.float64(jy)/wp.float64(mass);uz=wp.float64(jz)/wp.float64(mass)
    if not wp.isfinite(ux+uy+uz+rho) or rho<wp.float64(.5) or rho>wp.float64(1.5) or ux*ux+uy*uy+uz*uz>wp.float64(.25):
        wp.atomic_max(failure,0,1)
        return
    moments=V10(rho,ux,uy,uz,ux*ux,uy*uy,uz*uz,ux*uy,ux*uz,uy*uz)
    for h in range(10):
        wp.atomic_add(out,h,j,wp.int64(wp.round(moments[h]*wp.float64(4294967296.0))))

"""Fixed-order gather adjoint beside the unchanged 3-D wave forward kernels."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from kernel_engine.wave_fdtd import diff_wave_3d as baseline


@wp.kernel
def seed_gradient(p:wp.array3d(dtype=float),gp:wp.array3d(dtype=float),lo:int,hi:int):
    i,j,k=wp.tid()
    value=float(0.)
    if i>=lo and i<hi and j>=lo and j<hi and k>=lo and k<hi:
        value=2.0*p[i,j,k]
    gp[i,j,k]=value


@wp.kernel
def reverse_pressure(gp:wp.array3d(dtype=float),gx:wp.array3d(dtype=float),
                     gy:wp.array3d(dtype=float),gz:wp.array3d(dtype=float),
                     csq:wp.array3d(dtype=float),vx:wp.array3d(dtype=float),
                     vy:wp.array3d(dtype=float),vz:wp.array3d(dtype=float),
                     hx:wp.array3d(dtype=float),hy:wp.array3d(dtype=float),
                     hz:wp.array3d(dtype=float),gc:wp.array3d(dtype=float),cs:float):
    i,j,k=wp.tid()
    n=gp.shape[0]
    x=gx[i,j,k];y=gy[i,j,k];z=gz[i,j,k]
    if i>=1 and j>=1 and k>=1:
        coefficient=cs*csq[i,j,k]*gp[i,j,k]
        x=x-coefficient;y=y-coefficient;z=z-coefficient
        div=cs*(vx[i,j,k]-vx[i-1,j,k])+cs*(vy[i,j,k]-vy[i,j-1,k])+cs*(vz[i,j,k]-vz[i,j,k-1])
        gc[i,j,k]=gc[i,j,k]-gp[i,j,k]*div
    if i+1<n and j>=1 and k>=1:
        x=x+cs*csq[i+1,j,k]*gp[i+1,j,k]
    if j+1<n and i>=1 and k>=1:
        y=y+cs*csq[i,j+1,k]*gp[i,j+1,k]
    if k+1<n and i>=1 and j>=1:
        z=z+cs*csq[i,j,k+1]*gp[i,j,k+1]
    hx[i,j,k]=x;hy[i,j,k]=y;hz[i,j,k]=z


@wp.kernel
def reverse_velocity(gp:wp.array3d(dtype=float),hx:wp.array3d(dtype=float),
                     hy:wp.array3d(dtype=float),hz:wp.array3d(dtype=float),
                     previous:wp.array3d(dtype=float),gx:wp.array3d(dtype=float),
                     gy:wp.array3d(dtype=float),gz:wp.array3d(dtype=float),cs:float):
    i,j,k=wp.tid()
    n=gp.shape[0]
    value=gp[i,j,k]
    x=float(0.);y=float(0.);z=float(0.)
    if i<n-1:
        x=hx[i,j,k];value=value+cs*x
    if i>=1:
        value=value-cs*hx[i-1,j,k]
    if j<n-1:
        y=hy[i,j,k];value=value+cs*y
    if j>=1:
        value=value-cs*hy[i,j-1,k]
    if k<n-1:
        z=hz[i,j,k];value=value+cs*z
    if k>=1:
        value=value-cs*hz[i,j,k-1]
    previous[i,j,k]=value
    gx[i,j,k]=x;gy[i,j,k]=y;gz[i,j,k]=z


def solve(csq):
    n=baseline.N
    shape=(n,n,n)
    a=np.arange(n)
    seed=np.zeros(shape,np.float32)
    seed+=np.exp(-(((a[:,None,None]-5)**2+(a[None,:,None]-n//2)**2
                   +(a[None,None,:]-n//2)**2)/7.0)).astype(np.float32)
    p=wp.array(seed,dtype=float,device=baseline.DEV)
    def zero():
        return wp.zeros(shape,dtype=float,device=baseline.DEV)
    vx,vy,vz=zero(),zero(),zero()
    history=[]
    for _ in range(baseline.T):
        nx,ny,nz,pn=zero(),zero(),zero(),zero()
        wp.launch(baseline.upd_v,shape,inputs=[p,vx,vy,vz,nx,ny,nz,.4],device=baseline.DEV)
        wp.launch(baseline.upd_p,shape,inputs=[p,nx,ny,nz,pn,csq,.4],device=baseline.DEV)
        p,vx,vy,vz=pn,nx,ny,nz
        history.append((vx,vy,vz))
    gp,previous=zero(),zero()
    gx,gy,gz=zero(),zero(),zero()
    hx,hy,hz=zero(),zero(),zero()
    gc=zero()
    wp.launch(seed_gradient,shape,inputs=[p,gp,2*n//5,3*n//5],device=baseline.DEV)
    for vx,vy,vz in reversed(history):
        wp.launch(reverse_pressure,shape,inputs=[gp,gx,gy,gz,csq,vx,vy,vz,hx,hy,hz,gc,.4],device=baseline.DEV)
        wp.launch(reverse_velocity,shape,inputs=[gp,hx,hy,hz,previous,gx,gy,gz,.4],device=baseline.DEV)
        gp,previous=previous,gp
    return p,gc


def main():
    if baseline.DEV!="cuda:0":
        raise RuntimeError("CUDA required")
    shape=(baseline.N,)*3
    host=np.ones(shape,np.float32)
    legs=[]
    for _ in range(2):
        material=wp.array(host,dtype=float,device=baseline.DEV)
        p,g=solve(material)
        final,gradient=p.numpy(),g.numpy()
        loss,reference=baseline.forward(material)
        value=baseline.energy_value(loss,p)
        rows=[]
        for cell in ((baseline.N//2,)*3,(baseline.N//3,baseline.N//2,baseline.N//2),
                     (baseline.N//2,2*baseline.N//3,baseline.N//2)):
            plus=host.copy();minus=host.copy()
            plus[cell]+=.01;minus[cell]-=.01
            jp=baseline.energy_value(*baseline.forward(wp.array(plus,dtype=float,device=baseline.DEV)))
            jm=baseline.energy_value(*baseline.forward(wp.array(minus,dtype=float,device=baseline.DEV)))
            fd=(jp-jm)/.02
            ad=float(gradient[cell])
            rows.append({"cell":cell,"adjoint":ad,"finite_difference":fd,"relative_error":abs(ad-fd)/(abs(fd)+1e-12)})
        row={"energy":value,"forward_byte_identity":final.tobytes()==reference.numpy().tobytes(),
             "gradient_sha256":hashlib.sha256(gradient.tobytes()).hexdigest(),
             "forward_sha256":hashlib.sha256(final.tobytes()).hexdigest(),
             "finite":bool(np.isfinite(gradient).all()),"finite_difference":rows,
             "max_fd_relative_error":max(r["relative_error"] for r in rows)}
        legs.append(row);print(json.dumps(row),flush=True)
    gates={"two_run_identity":legs[0]==legs[1],"frozen_forward_identity":all(r["forward_byte_identity"] for r in legs),
           "finite":all(r["finite"] for r in legs),"unchanged_fd_gate":all(r["max_fd_relative_error"]<.05 for r in legs)}
    result={"legs":legs,"gates":gates,"performance_measured":False,
            "scope":"one fixed synthetic 3D wave fixture; other selftests unchanged"}
    (ROOT/"reports"/"diff_wave_3d_gather.json").write_text(json.dumps(result,indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())

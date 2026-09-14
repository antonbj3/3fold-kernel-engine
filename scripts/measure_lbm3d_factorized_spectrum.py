"""Screen a D3Q19 represented-moment Gaussian closure before GPU work.

Second central moments relax at the shear rate. Represented third cumulants
and mixed fourth cumulants relax fully (rate one); missing D3Q27 moments are
not claimed. The mixed fourth central moments are covariance products.
"""
import json
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb


def reconstruct(rho,u,a):
    x,y,z=u;xx,yy,zz=a[0,0],a[1,1],a[2,2];xy,xz,yz=a[0,1],a[0,2],a[1,2]
    rxx=rho*(x*x+xx);ryy=rho*(y*y+yy);rzz=rho*(z*z+zz)
    moments=np.array([rho,rho*x,rho*y,rho*z,rxx+ryy+rzz,2*rxx-ryy-rzz,ryy-rzz,
        rho*(x*y+xy),rho*(x*z+xz),rho*(y*z+yz),
        rho*(x*y*y+x*yy+2*y*xy),rho*(x*z*z+x*zz+2*z*xz),
        rho*(y*x*x+y*xx+2*x*xy),rho*(y*z*z+y*zz+2*z*yz),
        rho*(z*x*x+z*xx+2*x*xz),rho*(z*y*y+z*yy+2*y*yz),
        rho*(x*x*y*y+x*x*yy+y*y*xx+4*x*y*xy+xx*yy+2*xy*xy),
        rho*(x*x*z*z+x*x*zz+z*z*xx+4*x*z*xz+xx*zz+2*xz*xz),
        rho*(y*y*z*z+y*y*zz+z*z*yy+4*y*z*yz+yy*zz+2*yz*yz)])
    return lb.MI@moments


def collision(f,tau=.5021333333333333):
    rho=f.sum();u=lb.C.T@f/rho
    covariance=np.einsum('qa,qb,q->ab',lb.C,lb.C,f)/rho-np.outer(u,u)
    a=np.eye(3)/3+(1-1/tau)*(covariance-np.eye(3)/3)
    return reconstruct(rho,u,a)


def main():
    k=np.array(np.meshgrid(*([np.linspace(0,np.pi,17)]*3),indexing='ij')).reshape(3,-1).T
    phase=np.exp(-1j*(k@lb.C.T))[:,:,None];results=[]
    for u in [np.array([v,0.,0.]) for v in [0.,.04,.07,.1]]+[np.array([.07,.02,.01])]:
        f=reconstruct(1.,u,np.eye(3)/3);jac=np.empty((19,19))
        for j in range(19):
            pert=f.astype(complex);pert[j]+=1e-30j;jac[:,j]=collision(pert).imag/1e-30
        radius=np.max(np.abs(np.linalg.eigvals(phase*jac)),axis=1);worst=np.argmax(radius)
        results.append({'u':u.tolist(),'max_radius':float(radius[worst]),'unstable_modes':int(np.sum(radius>1+1e-10)),
            'worst_k_over_pi':(k[worst]/np.pi).tolist(),'equilibrium_residual':float(np.max(np.abs(collision(f)-f)))})
    r={'scope':'complex-step Jacobian of represented D3Q19 cumulant closure; not GPU/DNS validation','modes':len(k),'tau':.5021333333333333,'results':results}
    p=Path('reports/lbm3d_factorized_spectrum_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()

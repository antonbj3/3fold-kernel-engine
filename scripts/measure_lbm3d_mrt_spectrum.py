"""Linearized Fourier diagnosis at rest; excludes finite-amplitude LES dynamics."""
import json
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb


def projectors():
    c=lb.C.astype(float);c2=np.sum(c*c,axis=1)
    eq=lb.W[:,None]*(1+3*(c@c.T))
    stress=4.5*lb.W[:,None]*((c@c.T)**2-(c2[:,None]+c2[None,:])/3+1/3)
    return eq,stress


def measure():
    eq,p=projectors();k=np.array(np.meshgrid(*([np.linspace(0,np.pi,17)]*3),indexing='ij')).reshape(3,-1).T
    phase=np.exp(-1j*(k@lb.C.T))[:,:,None]
    rates=np.r_[np.zeros(4),np.full(6,1/.5001),np.ones(9)]
    raw=np.eye(19)-lb.MI@np.diag(rates)@lb.M@(np.eye(19)-eq)
    rates[4]=1
    bulk=np.eye(19)-lb.MI@np.diag(rates)@lb.M@(np.eye(19)-eq)
    kernels={'raw_mrt':raw,'bulk_rate_one':bulk,'hermite_projected_mrt':eq+(1-1/.5001)*p}
    results={}
    for name,collision in kernels.items():
        radius=np.max(np.abs(np.linalg.eigvals(phase*collision)),axis=1);i=np.argmax(radius)
        results[name]={'max_spectral_radius':float(radius[i]),'worst_k_over_pi':(k[i]/np.pi).tolist(),
                       'unstable_vectors_tolerance_1e_minus10':int(np.sum(radius>1+1e-10))}
    return {'scope':'4913 Fourier vectors in [0,pi]^3, infinitesimal perturbations about rho=1,u=0; Cs term is higher order and absent in this Jacobian',
            'tau':.5001,'modes':4913,'results':results,'projector_rank':int(np.linalg.matrix_rank(p)),
            'projector_idempotence_max_error':float(np.max(np.abs(p@p-p))),
            'conserved_stress_cross_error':float(np.max(np.abs(eq@p)))}


if __name__=='__main__':
    r=measure();out=Path('reports/lbm3d_mrt_spectrum_v1/report.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

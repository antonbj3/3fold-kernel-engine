"""Linear moving-state screen; distinguishes bounded LES from kinetic stability."""
import json
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb


def equilibrium_jacobian(u,third_order=False):
    c=lb.C.astype(float);cu=c@u;u2=u@u
    value=1+3*cu+4.5*cu**2-1.5*u2
    grad=(3+9*cu)[:,None]*c-3*u[None,:]
    if third_order:
        value+=4.5*cu**3-4.5*cu*u2
        grad+=(13.5*cu**2-4.5*u2)[:,None]*c-9*cu[:,None]*u[None,:]
    return lb.W[:,None]*(value[:,None]+grad@(c-u).T)


def main():
    k=np.array(np.meshgrid(*([np.linspace(0,np.pi,17)]*3),indexing='ij')).reshape(3,-1).T
    phase=np.exp(-1j*(k@lb.C.T))[:,:,None];results=[]
    for speed in [0.,.04,.07,.10]:
        for bulk_tau in [None,1.]:
            for third in [False,True]:
                eq=equilibrium_jacobian(np.array([speed,0.,0.]),third)
                rates=np.r_[np.zeros(4),np.full(6,1/.5021333333333333),np.ones(9)]
                if bulk_tau:rates[4]=1/bulk_tau
                collision=np.eye(19)-lb.MI_HERMITE@np.diag(rates)@lb.M_HERMITE@(np.eye(19)-eq)
                radius=np.max(np.abs(np.linalg.eigvals(phase*collision)),axis=1);worst=np.argmax(radius)
                results.append({'u_x':speed,'bulk_tau':bulk_tau,'third_order_equilibrium':third,'max_radius':float(radius[worst]),'unstable_modes':int(np.sum(radius>1+1e-10)),'worst_k_over_pi':(k[worst]/np.pi).tolist()})
    r={'scope':'infinitesimal unforced moving equilibrium, Cs contribution higher order; no nonlinear/DNS gate',
       'tau_shear':.5021333333333333,'wavevectors':len(k),'results':results}
    p=Path('reports/lbm3d_moving_spectrum_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()

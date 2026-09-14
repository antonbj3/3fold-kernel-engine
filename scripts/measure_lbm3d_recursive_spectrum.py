"""Recursive third-order Hermite nonequilibrium screen on D3Q19."""
import json
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb
from measure_lbm3d_moving_spectrum import equilibrium_jacobian


def recursive_projector(u):
    c=lb.C.astype(float);cu=c@u
    # Each input population column contributes c_a*c_b to Pi_neq.
    dots=c@c.T;trace=np.sum(c*c,axis=1)
    third=13.5*lb.W[:,None]*(cu[:,None]*dots**2-cu[:,None]*trace[None,:]/3-2*dots*(c@u)[None,:]/3)
    return lb.P2_PROJECTOR+third


def main():
    k=np.array(np.meshgrid(*([np.linspace(0,np.pi,17)]*3),indexing='ij')).reshape(3,-1).T
    phase=np.exp(-1j*(k@lb.C.T))[:,:,None];results=[]
    for vector in [[0.,0.,0.],[.04,0.,0.],[.07,0.,0.],[.1,0.,0.],[.07,.02,.01]]:
        u=np.array(vector);eq=equilibrium_jacobian(u,True)
        collision=eq+(1-1/.5021333333333333)*recursive_projector(u)@(np.eye(19)-eq)
        radius=np.max(np.abs(np.linalg.eigvals(phase*collision)),axis=1);i=np.argmax(radius)
        results.append({'u':vector,'max_radius':float(radius[i]),'unstable_modes':int(np.sum(radius>1+1e-10)),'worst_k_over_pi':(k[i]/np.pi).tolist()})
    r={'scope':'recursive third-Hermite nonequilibrium plus cubic equilibrium, unforced linear screen only','modes':len(k),'results':results}
    p=Path('reports/lbm3d_recursive_spectrum_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()

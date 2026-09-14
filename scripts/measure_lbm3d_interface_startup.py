"""Small 3D low-viscosity interface screen; no turbulent validation claim."""
import json,time
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU


def initialize(sim,y,speed):
    x,_,z=np.indices(sim.shape)
    envelope=np.maximum(0,np.sin(np.pi*y/96))[None,:,None]
    u=np.zeros((3,)+sim.shape);u[0]=speed*envelope
    u[1]=1e-6*np.sin(2*np.pi*x/sim.shape[0])*np.cos(2*np.pi*z/sim.shape[2])*envelope
    u[:,sim.args[0].numpy()!=0]=0
    return lb.quantize(lb.equilibrium(np.ones(sim.shape),u),sim.bits)


def main(skip_uniform=False,output='reports/lbm3d_interface_startup_v1',conserved_reflux=False):
    report={'scope':'small3D low-viscosity startup, identical smooth physical initialization; not DNS','cases':[]}
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    for speed in [0.,.07]:
        refined=RefinedChannelGPU(device='cpu',nx=8,height=96,nz=8,wall_cells=12,tau_fine=.5032,force_fine=0.,cs_fine=.1,bulk_tau_fine=1.,recursive=True,conserved_reflux=conserved_reflux)
        for s,y in zip([*refined.fine,refined.coarse],[np.arange(14)-.5,np.arange(14)+83.5,2*np.arange(50)-1]):
            refined._replace(s,initialize(s,y,speed))
        solid=np.zeros((8,98,8),np.int32);solid[:,0,:]=1;solid[:,-1,:]=1
        q=lb.quantize(lb.equilibrium(np.ones(solid.shape),np.zeros((3,)+solid.shape)))
        uniform=ChannelSimulation(q,solid=solid,force_density=0.,tau=.5032,cs=.1,bulk_tau=1.,recursive=True)
        refined._replace(uniform,initialize(uniform,np.arange(98)-.5,speed))
        for name,sim in [('uniform',uniform),('refined',refined)]:
            if skip_uniform and name=='uniform':continue
            rows=[];start=time.perf_counter()
            for done in range(100,4001,100):
                if name=='uniform':sim.step(100);blocks=[sim]
                else:
                    for _ in range(50):sim.step()
                    blocks=[*sim.fine,sim.coarse]
                values=[]
                for block in blocks:
                    a=block.numpy();rho=a.sum(0,dtype=float)/2**block.bits
                    values.append({'rho_min':float(rho.min()),'rho_max':float(rho.max()),'flag':int(block.failure.numpy()[0])})
                rows.append({'step':done,'blocks':values})
                if any(v['flag'] or v['rho_min']<.5 or v['rho_max']>1.5 for v in values):break
            report['cases'].append({'speed':speed,'grid_kind':name,'seconds':time.perf_counter()-start,'history':rows})
            (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'speed':speed,'grid_kind':name,'last':rows[-1]}),flush=True)


if __name__=='__main__':main()

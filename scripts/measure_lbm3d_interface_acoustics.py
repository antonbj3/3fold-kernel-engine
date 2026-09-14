"""3D standing-wave interface gate against a uniform-grid numerical control.

A weak, long-wavelength pressure mode isolates pressure transmission without
the expense or ambiguity of turbulent averaging. All grids have the same
physical bulk/shear viscosity; fine time and space units are shared.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU


def main(wavelength=96,wall_cells=12,output="reports/lbm3d_interface_acoustics_v1"):
    h=96;nf=wall_cells;amplitude=1e-4;shape=(8,h+2,8)
    solid=np.zeros(shape,np.int32);solid[:,0,:]=1;solid[:,-1,:]=1
    def populations(s,y):
        rho=np.broadcast_to(1+amplitude*np.cos(2*np.pi*y/wavelength)[None,:,None],s.shape).copy()
        return lb.quantize(lb.equilibrium(rho,np.zeros((3,)+s.shape)),s.bits)
    q=lb.quantize(lb.equilibrium(np.ones(shape),np.zeros((3,)+shape)))
    uniform=ChannelSimulation(q,solid=solid,force_density=0.,tau=.5032,cs=.1,bulk_tau=1.,recursive=True,device='cuda:0')
    models={}
    for name,balanced in [('conserved',False),('balanced_conserved',True)]:
        s=RefinedChannelGPU(nx=8,height=h,nz=8,wall_cells=nf,tau_fine=.5032,force_fine=0.,cs_fine=.1,bulk_tau_fine=1.,recursive=True,conserved_reflux=True,balanced_reflux=balanced)
        for block,y in zip([*s.fine,s.coarse],[np.arange(nf+2)-.5,np.arange(nf+2)+h-nf-.5,2*np.arange(h//2+2)-1]):s._replace(block,populations(block,y))
        models[name]=s
    models['conserved']._replace(uniform,populations(uniform,np.arange(98)-.5))
    initial={name:s.ledger() for name,s in models.items()}
    rows=[];start=time.perf_counter();yfine=np.arange(h)+.5
    for done in range(50,2001,50):
        uniform.step(50);rho,u=lb.fields(uniform.numpy());truth=np.array([rho[:,1:-1,:].mean((0,2))-1,u[1,:,1:-1,:].mean((0,2))*np.sqrt(3)])
        row={'step':done,'models':{}}
        for name,s in models.items():
            for _ in range(25):s.step()
            positions=[];values=[]
            for block,ys,selection in [(s.fine[0],np.arange(nf)+.5,slice(1,-1)),(s.coarse,np.arange(nf+1,h-nf,2),slice(s.b,s.t+1)),(s.fine[1],np.arange(h-nf,h)+.5,slice(1,-1))]:
                rho,u=lb.fields(block.numpy(),block.bits)
                positions.extend(ys);values.append(np.array([rho[:,selection,:].mean((0,2))-1,u[1,:,selection,:].mean((0,2))*np.sqrt(3)]))
            measured=np.concatenate(values,axis=1);interpolated=np.array([np.interp(yfine,positions,v) for v in measured])
            error=float(np.sqrt(np.mean((interpolated-truth)**2))/amplitude)
            ledger=s.ledger();wall=s.wall_impulse();flags=[int(b.failure.numpy()[0]) for b in [*s.fine,s.coarse]]
            exact=ledger[0]==initial[name][0] and all(ledger[k+1]-initial[name][k+1]+int(wall[k])==0 for k in range(3))
            row['models'][name]={'relative_wave_error':error,'exact_ledger':exact,'flags':flags}
        rows.append(row)
        if any(not v['exact_ledger'] or any(v['flags']) or v['relative_wave_error']>100 for v in row['models'].values()):break
    maxima={name:max(r['models'][name]['relative_wave_error'] for r in rows) for name in models}
    report={'scope':'small3D acoustic interface diagnostic; not turbulent DNS','amplitude':amplitude,'wavelength_cells':wavelength,'wall_cells':nf,'active_cell_fraction':2*nf/h+(1-2*nf/h)/8,'steps':done,'seconds':time.perf_counter()-start,
            'maximum_wave_errors':maxima,'history':rows,'gates':{'complete':done==2000,'exact_ledgers_and_flags':all(v['exact_ledger'] and not any(v['flags']) for r in rows for v in r['models'].values()),
            'balanced_error_below_5percent':maxima['balanced_conserved']<=.05,'balanced_improves_control':maxima['balanced_conserved']<maxima['conserved']},
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/kernel_engine/lbm/lbm3d_refinement.py'),Path('src/kernel_engine/lbm/lbm3d_refinement_gpu.py'),Path('scripts/measure_lbm3d_interface_acoustics.py')]}}
    report['passed']=all(report['gates'].values())
    p=Path(output)/'report.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['gates','maximum_wave_errors','seconds']}),flush=True)


if __name__=='__main__':main()

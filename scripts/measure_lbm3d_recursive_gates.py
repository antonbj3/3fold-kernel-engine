"""Recursive D3Q19 candidate: full repeats, finer precision and shear gate."""
import hashlib,json
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
from measure_lbm3d_mrt_les import initial,metrics


def main(*,bulk_tau=1.,output='reports/lbm3d_recursive_gates_v1/report.json'):
    sims=[];ledgers=[]
    for bits in [40,44,40]:
        q=lb.quantize(initial(),bits);ledgers.append(lb.ledger(q))
        sims.append(ChannelSimulation(q,force_density=0,tau=.5001,cs=.1,bits=bits,bulk_tau=bulk_tau,recursive=True,device='cuda:0'))
    history=[];repeat=True;conserved=True;stable=True;maximum=0.
    for step in range(64,4097,64):
        states=[];row=[]
        for s,bits,ledger in zip(sims,[40,44,40],ledgers):
            s.step(64);q=s.numpy();states.append(q);m=metrics(q,bits)
            valid=bool(not int(s.failure.numpy()[0]) and .5<=m['rho_min'] and m['rho_max']<=1.5 and m['max_speed']<.5 and m['energy']<=1.05*.005)
            stable=stable and valid;conserved=conserved and lb.ledger(q)==ledger
            row.append({'bits':bits,**m,'sha256':hashlib.sha256(q.tobytes()).hexdigest(),'valid':valid})
        error=float(np.max(np.abs(lb.fields(states[0],40)[1]-lb.fields(states[1],44)[1])))
        maximum=max(maximum,error);repeat=repeat and np.array_equal(states[0],states[2]);history.append({'step':step,'states':row,'velocity_error_40_44':error})
        if not stable:break
    n=32;k=2*np.pi/n;y,z=np.meshgrid(np.arange(n),np.arange(n),indexing='ij');u=np.zeros((3,n,n,n));u[0]=.01*np.cos(k*y)[None,:,:]*np.cos(k*z)[None,:,:]
    q=lb.quantize(lb.equilibrium(np.ones((n,n,n)),u));s=ChannelSimulation(q,force_density=0,tau=.8,cs=0,bulk_tau=bulk_tau,recursive=True,device='cuda:0');s.step(500)
    out=s.numpy();uf=lb.fields(out)[1];amp=float(np.sum(uf[0]*u[0])/np.sum(u[0]**2));truth=float(np.exp(-2*.1*k*k*500));shear_error=abs(amp/truth-1)
    gates={'4096_steps_stable':bool(stable and step==4096),'full_40bit_repeat_every_checkpoint':bool(repeat),
        'all_integer_ledgers_exact':bool(conserved and lb.ledger(out)==lb.ledger(q)),
        '40_vs44_velocity_error_below_1e_5':maximum<=1e-5,'shear_decay_error_below_2percent':shear_error<.02 and int(s.failure.numpy()[0])==0}
    report={'scope':'recursive third-order Hermite, unchanged TGV64^3 and precision thresholds','bulk_tau':bulk_tau,
        'gates':gates,'passed':all(gates.values()),'maximum_velocity_error_40_44':maximum,'shear_decay_relative_error':shear_error,'history':history,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/kernel_engine/lbm/lbm3d_channel.py'),Path('scripts/measure_lbm3d_recursive_gates.py')]}}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'gates':gates,'precision_error':maximum,'shear_error':shear_error}),flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

"""Bounded local GPU timing and repeat probe; not a DNS gate."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation


def main():
    n=64;steps=200
    x,y,z=np.meshgrid(*([np.arange(n)*2*np.pi/n]*3),indexing='ij')
    u=np.array([.04*np.sin(x)*np.cos(y)*np.cos(z),-.04*np.cos(x)*np.sin(y)*np.cos(z),np.zeros_like(x)])
    q=lb.quantize(lb.equilibrium(np.ones((n,n,n)),u));ledger=lb.ledger(q)
    modes={}
    for recursive in [False,True]:
        hashes=[];times=[];valid=True
        for repeat in range(2):
            s=ChannelSimulation(q,force_density=0,tau=.8,cs=.1,recursive=recursive,device='cuda:0')
            # Warm on a separate state so measured trajectories stay identical.
            s.step(2);wp.synchronize_device('cuda:0');s=ChannelSimulation(q,force_density=0,tau=.8,cs=.1,recursive=recursive,device='cuda:0')
            start=time.perf_counter();s.step(steps);wp.synchronize_device('cuda:0');times.append(time.perf_counter()-start)
            out=s.numpy();hashes.append(hashlib.sha256(out.tobytes()).hexdigest());valid=valid and lb.ledger(out)==ledger and int(s.failure.numpy()[0])==0
        modes[str(recursive)]={'seconds':times,'MLUPS':[n**3*steps/t/1e6 for t in times],'hashes':hashes,'repeat_exact':hashes[0]==hashes[1],'ledger_and_flags_pass':valid}
    r={'scope':'200 steps64^3 warm local throughput, not DNS or old/new numerical equivalence','device':wp.get_device('cuda:0').name,'modes':modes,
       'passed':all(v['repeat_exact'] and v['ledger_and_flags_pass'] for v in modes.values()),
       'source_sha256':hashlib.sha256(Path('src/kernel_engine/lbm/lbm3d_channel.py').read_bytes()).hexdigest()}
    p=Path('reports/lbm3d_local_probe_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

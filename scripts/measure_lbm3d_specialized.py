"""Compile-time matrix candidate: exact full-trajectory parity and paired timing."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation
from measure_lbm3d_mrt_les import initial


def main():
    q=lb.quantize(initial());ledger=lb.ledger(q)
    kwargs=dict(force_density=0,tau=.5001,cs=.1,bulk_tau=1.,recursive=True,device='cuda:0')
    sims=[cls(q,**kwargs) for cls in [ChannelSimulation,SpecializedChannelSimulation,SpecializedChannelSimulation]]
    exact=True;conserved=True;flags=True;history=[];started=time.perf_counter()
    for step in range(64,4097,64):
        out=[]
        for s in sims:s.step(64);out.append(s.numpy())
        exact=exact and np.array_equal(out[0],out[1]) and np.array_equal(out[1],out[2])
        conserved=conserved and all(lb.ledger(a)==ledger for a in out)
        flags=flags and all(int(s.failure.numpy()[0])==0 for s in sims)
        history.append({'step':step,'states_exact':exact,'ledgers_exact':conserved,'flags_clear':flags})
        if not exact or not conserved or not flags:break
    hashes=[hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in sims]
    timings={};n=96;steps=100
    q=lb.quantize(lb.equilibrium(np.ones((n,n,n)),np.zeros((3,n,n,n))))
    timed=[cls(q,force_density=0,tau=.8,recursive=True,device='cuda:0') for cls in [ChannelSimulation,SpecializedChannelSimulation]]
    for s in timed:s.step(2)
    wp.synchronize_device('cuda:0')
    # Alternate A/B then B/A to expose ordering or thermal bias.
    durations=[[],[]]
    for order in [[0,1],[1,0]]:
        for i in order:
            start=time.perf_counter();timed[i].step(steps);wp.synchronize_device('cuda:0');durations[i].append(time.perf_counter()-start)
    for name,t in zip(['runtime_matrices','compile_time_matrices'],durations):timings[name]={'seconds':t,'MLUPS':[n**3*steps/v/1e6 for v in t]}
    gains=[durations[0][i]/durations[1][i] for i in range(2)]
    gates={'full4096step_states_exact_including_repeat':bool(exact and step==4096),'integer_ledgers_exact':bool(conserved),'flags_clear':bool(flags),'faster_in_both_pairs':all(v>1 for v in gains)}
    r={'device':wp.get_device('cuda:0').name,'scope':'same recursive physics, matrix specialization only; no new DNS claim',
       'gates':gates,'passed':all(gates.values()),'speedup':gains,'timings':timings,'history':history,'final_hashes':hashes,
       'elapsed_s':time.perf_counter()-started,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/kernel_engine/lbm/lbm3d_channel.py'),Path('src/kernel_engine/lbm/lbm3d_channel_specialized.py')]}}
    p=Path('reports/lbm3d_specialized_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'gates':gates,'speedup':gains,'timings':timings}),flush=True)
    return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

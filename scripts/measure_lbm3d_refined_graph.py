"""Exact state/ledger and paired complete-step CUDA graph experiment."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
from measure_lbm3d_refinement_sgs import initialize


def main():
    opts=dict(recursive=True,bulk_tau_fine=1.,conserved_reflux=True,balanced_reflux=True,
              tau_fine=.5032,force_fine=1e-8,channel_factory=SpecializedChannelSimulation)
    sims=[initialize(RefinedChannelGPU,**opts) for _ in range(2)]
    history=[]
    for count in (1,3,2,49,50,895):
        for i,s in enumerate(sims):s.step_many(count,use_graph=bool(i))
        arrays=[[b.numpy() for b in [*s.fine,s.coarse]] for s in sims]
        exact=all(np.array_equal(a,b) for a,b in zip(*arrays))
        ledgers=[];valid=True
        for s in sims:
            ledger=s.ledger();wall=s.wall_impulse();initial=s.initial_ledger
            residual=[ledger[k+1]-initial[k+1]+int(wall[k])-(s.steps*s.nx*s.h*s.nz*s.force_units if k==0 else 0) for k in range(3)]
            valid=valid and ledger[0]==initial[0] and residual==[0,0,0] and not any(int(b.failure.numpy()[0]) for b in [*s.fine,s.coarse])
            ledgers.append(ledger)
        history.append({'fine_steps':sims[0].steps,'states_exact':exact,'conservation_and_flags':valid,'counters_exact':sims[0].steps==sims[1].steps})
        if not exact or not valid:break
    passed=all(all(r[k] for k in ('states_exact','conservation_and_flags','counters_exact')) for r in history) and sims[0].steps==2000
    durations=[[],[]];capture_s=None;timed_exact=False
    if passed:
        sims=[RefinedChannelGPU(nx=64,height=96,nz=64,wall_cells=24,cs_fine=.1,**opts) for _ in range(2)]
        for i,s in enumerate(sims):
            start=time.perf_counter();s.step_many(2,use_graph=bool(i));wp.synchronize_device('cuda:0')
            if i:capture_s=time.perf_counter()-start
        for order in ((0,1),(1,0),(0,1)):
            for i in order:
                start=time.perf_counter();sims[i].step_many(100,use_graph=bool(i));wp.synchronize_device('cuda:0');durations[i].append(time.perf_counter()-start)
        timed_exact=all(np.array_equal(a.numpy(),b.numpy()) for a,b in zip([*sims[0].fine,sims[0].coarse],[*sims[1].fine,sims[1].coarse]))
    gains=[a/b for a,b in zip(*durations)]
    gates={'interleaved_2000step_states_ledgers_counters_exact':passed,'timed_full_states_exact':timed_exact,'faster_all_three_pairs':len(gains)==3 and all(g>1 for g in gains)}
    p=Path('reports/lbm3d_refined_graph_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True)
    result={'scope':'complete refined-step graph versus specialized launch loop, not DNS validation','device':wp.get_device('cuda:0').name,'gates':gates,'passed':all(gates.values()),'history':history,'seconds':durations,'speedups':gains,'capture_and_first_four_fine_steps_seconds':capture_s,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/kernel_engine/lbm/lbm3d_refinement_gpu.py')]}}
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'gates':gates,'speedups':gains}),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

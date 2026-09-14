"""Hardware-conditioned collision launch tuning, complete refinement retained."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation,specialized_collide_stream
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
from measure_lbm3d_refinement_sgs import initialize
from measure_lbm3d_refinement_gpu import run


def factory(block):
    class Tuned(SpecializedChannelSimulation):
        def step(self,steps=1):
            if not isinstance(steps,int) or steps<0:raise ValueError('invalid steps')
            for _ in range(steps):
                wp.launch(specialized_collide_stream,self.shape,[self.a,self.b,*self.specialized_args],device=self.device,block_dim=block)
                self.a,self.b=self.b,self.a
    return Tuned


def main():
    choices=(256,128,64)
    opts=dict(recursive=True,bulk_tau_fine=1.,conserved_reflux=True,balanced_reflux=True,tau_fine=.5032,force_fine=1e-8)
    records=[];states=[]
    for block in choices:
        r,s=run(initialize(RefinedChannelGPU,channel_factory=factory(block),**opts));records.append(r);states.append(s)
    exact=all(np.array_equal(a,b) for s in states[1:] for a,b in zip(states[0],s))
    valid=all(len(r['history'])==20 and all(h['mass_exact'] and h['momentum_residual']==[0,0,0] and not any(h['flags']) for h in r['history']) for r in records)
    durations=[[],[],[]];timed_exact=False
    if exact and valid:
        sims=[RefinedChannelGPU(nx=288,height=96,nz=144,wall_cells=24,cs_fine=.1,channel_factory=factory(b),**opts) for b in choices]
        for s in sims:s.step_many(2)
        wp.synchronize_device('cuda:0')
        for order in ((0,1,2),(2,1,0),(1,0,2)):
            for i in order:
                start=time.perf_counter();sims[i].step_many(20);wp.synchronize_device('cuda:0');durations[i].append(time.perf_counter()-start)
        timed_exact=all(np.array_equal(a.numpy(),b.numpy()) for s in sims[1:] for a,b in zip([*sims[0].fine,sims[0].coarse],[*s.fine,s.coarse]))
    gains={str(b):[a/v for a,v in zip(durations[0],t)] for b,t in zip(choices,durations)}
    result={'scope':'paired full-size refined steps; smooth timing state, no new turbulent validation','device':wp.get_device('cuda:0').name,'block_dimensions':choices,'fine_steps_per_timing':40,'seconds':durations,'speedups_over256':gains,
            'gates':{'full2000step_states_exact':exact,'ledgers_and_flags':valid,'timed_full_states_exact':timed_exact},
            'faster_all_pairs':{b:len(g)==3 and all(v>1 for v in g) for b,g in gains.items() if b!='256'},
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).relative_to(Path.cwd()),Path('src/kernel_engine/lbm/lbm3d_channel_specialized.py')]}}
    p=Path('reports/lbm3d_launch_tuning_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0 if all(result['gates'].values()) else 1

if __name__=='__main__':raise SystemExit(main())

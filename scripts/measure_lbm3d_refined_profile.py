"""Per-kernel device timings for the complete accepted refined geometry."""
import json,time
from pathlib import Path
import warp as wp
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU


def main():
    sim=RefinedChannelGPU(nx=288,height=96,nz=144,wall_cells=24,cs_fine=.1,recursive=True,bulk_tau_fine=1.,conserved_reflux=True,balanced_reflux=True,tau_fine=.5032,force_fine=1e-8,channel_factory=SpecializedChannelSimulation)
    sim.step_many(4);wp.synchronize_device('cuda:0')
    start=time.perf_counter();sim.step_many(100);wp.synchronize_device('cuda:0');plain=time.perf_counter()-start
    events=[]
    with wp.ScopedTimer('refined100',print=False,cuda_filter=wp.TIMING_KERNEL) as timer:
        sim.step_many(100)
    events=timer.timing_results
    if not events:raise RuntimeError('device timing collection is empty')
    rows={}
    for event in events:
        row=rows.setdefault(event.name,{'calls':0,'milliseconds':0.})
        row['calls']+=1;row['milliseconds']+=event.elapsed
    result={'scope':'full-size geometry, smooth initial state, kernel attribution only; not a turbulent throughput benchmark','device':wp.get_device('cuda:0').name,'macro_steps':100,'plain_seconds':plain,'kernels':dict(sorted(rows.items(),key=lambda item:-item[1]['milliseconds']))}
    p=Path('reports/lbm3d_refined_profile_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()

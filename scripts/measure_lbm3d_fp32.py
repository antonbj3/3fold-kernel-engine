"""FP32 arithmetic admission against FP64: same int64 state and physics."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation
from kernel_engine.lbm.lbm3d_channel_fp32 import FP32ChannelSimulation
from measure_lbm3d_mrt_les import initial,metrics


def main(mixed=False,hermite=False):
    from kernel_engine.lbm.lbm3d_channel_hermite32 import Hermite32ChannelSimulation
    from kernel_engine.lbm.lbm3d_channel_mixed import MixedChannelSimulation
    candidate=Hermite32ChannelSimulation if hermite else MixedChannelSimulation if mixed else FP32ChannelSimulation
    q=lb.quantize(initial());original=lb.ledger(q);initial_energy=metrics(q,40)['energy']
    opts=dict(force_density=0.,tau=.5001,cs=.1,bulk_tau=1.,recursive=True,device='cuda:0')
    sims=[cls(q,**opts) for cls in (SpecializedChannelSimulation,candidate,candidate)]
    history=[];started=time.perf_counter()
    for step in range(64,4097,64):
        arrays=[]
        for sim in sims:sim.step(64);arrays.append(sim.numpy())
        fields=[lb.fields(a) for a in arrays];m=[metrics(a,40) for a in arrays]
        error=float(np.max(np.abs(fields[0][1]-fields[1][1])))
        row={'step':step,'velocity_max_abs_error':error,'density_max_abs_error':float(np.max(np.abs(fields[0][0]-fields[1][0]))),
             'repeat_exact':bool(np.array_equal(arrays[1],arrays[2])),
             'ledgers_exact':all(lb.ledger(a)==original for a in arrays),
             'flags_clear':not any(int(s.failure.numpy()[0]) for s in sims),
             'stable':all(.5<=r['rho_min'] and r['rho_max']<=1.5 and r['max_speed']<.5 and r['energy']<=1.05*initial_energy for r in m)}
        history.append(row)
        if error>1e-5 or not all(row[k] for k in ('repeat_exact','ledgers_exact','flags_clear','stable')):break
    numerical=step==4096 and all(r['velocity_max_abs_error']<=1e-5 and all(r[k] for k in ('repeat_exact','ledgers_exact','flags_clear','stable')) for r in history)
    # Timing is diagnostic even when precision fails; no accepted speedup then.
    q=lb.quantize(initial(96));timed=[cls(q,**opts) for cls in (SpecializedChannelSimulation,candidate)]
    for s in timed:s.step(2)
    wp.synchronize_device('cuda:0');durations=[[],[]]
    for order in ((0,1),(1,0)):
        for i in order:
            start=time.perf_counter();timed[i].step(64);wp.synchronize_device('cuda:0');durations[i].append(time.perf_counter()-start)
    result={'scope':'64^3 TGV u0=.2 tau=.5001 Cs=.1, FP32 collision against FP64, int64 40fractionbit storage and repair retained; not turbulent channel validation',
            'candidate':'hermite32' if hermite else 'mixed_transforms' if mixed else 'fp32_collision','device':wp.get_device('cuda:0').name,'velocity_absolute_limit':1e-5,'passed':numerical,'history':history,
            'timing_scope':'96^3 TGV,64steps per timing, diagnostic only if precision fails','seconds':durations,'diagnostic_speedups':[a/b for a,b in zip(*durations)],
            'elapsed_seconds':time.perf_counter()-started,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('scripts/measure_lbm3d_fp32.py'),Path('src/kernel_engine/lbm/lbm3d_channel_hermite32.py' if hermite else 'src/kernel_engine/lbm/lbm3d_channel_mixed.py' if mixed else 'src/kernel_engine/lbm/lbm3d_channel_fp32.py')]}}
    p=Path('reports/lbm3d_hermite32_v1/report.json' if hermite else 'reports/lbm3d_mixed_v1/report.json' if mixed else 'reports/lbm3d_fp32_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':numerical,'last':history[-1],'speedups':result['diagnostic_speedups']}),flush=True)
    return 0 if numerical else 1

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mixed',action='store_true');parser.add_argument('--hermite',action='store_true')
    args=parser.parse_args()
    raise SystemExit(main(mixed=args.mixed,hermite=args.hermite))

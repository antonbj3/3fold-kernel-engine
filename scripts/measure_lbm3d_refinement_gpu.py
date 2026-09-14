"""Resident 2:1 coupling: H100 repeat, CPU reference and laminar gates."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm.lbm3d_refinement import ReferenceRefinedChannel
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU


def run(sim):
    initial=sim.initial_ledger
    history=[]
    start=time.perf_counter()
    for macro in range(1,1001):
        sim.step()
        if macro%50==0:
            ledger=sim.ledger();wall=sim.wall_impulse()
            residual=[ledger[a+1]-initial[a+1]+int(wall[a])-(sim.steps*sim.nx*sim.h*sim.nz*sim.force_units if a==0 else 0) for a in range(3)]
            flags=[int(s.failure.numpy()[0]) for s in [*sim.fine,sim.coarse]]
            history.append({'steps':sim.steps,'mass_exact':ledger[0]==initial[0],'momentum_residual':residual,'flags':flags})
            if any(flags):break
    wp.synchronize_device(sim.coarse.device)
    elapsed=time.perf_counter()-start
    y,u=sim.profile()
    truth=sim.gf/(2*((sim.tf-.5)/3))*y*(sim.h-y)
    states=[s.numpy() for s in [*sim.fine,sim.coarse]]
    return {'history':history,'seconds':elapsed,'profile':u.tolist(),'y':y.tolist(),
            'profile_relative_L2':float(np.linalg.norm(u-truth)/np.linalg.norm(truth)),
            'state_sha256':[hashlib.sha256(q.tobytes()).hexdigest() for q in states]},states


def main():
    reference,_=run(ReferenceRefinedChannel())
    first,a=run(RefinedChannelGPU())
    second,b=run(RefinedChannelGPU())
    error=float(np.max(np.abs(np.asarray(first['profile'])-reference['profile'])))
    gates={'full_gpu_repeat_exact':all(np.array_equal(x,y) for x,y in zip(a,b)),
           'repeat_history_exact':first['history']==second['history'],
           'cpu_profile_absolute_error_at_most_1e_10':error<=1e-10,
           'laminar_profile_relative_L2_below_2_percent':all(r['profile_relative_L2']<.02 for r in [reference,first,second]),
           'all_ledgers_exact_and_states_valid':all(len(r['history'])==20 and all(h['mass_exact'] and h['momentum_residual']==[0,0,0] and h['flags']==[0,0,0] for h in r['history']) for r in [reference,first,second])}
    paths=[Path(__file__),*Path('src/kernel_engine/lbm').glob('lbm3d_*.py')]
    report={'scope':'3D two wall blocks, Cs=0, 2000 fine steps; no turbulent validation',
            'device':wp.get_device('cuda:0').name,'cpu_reference':reference,'gpu_first':first,'gpu_second':second,
            'cpu_gpu_profile_max_absolute_error':error,'gates':gates,'passed':all(gates.values()),
            'source_sha256':{str(p.relative_to(Path.cwd()) if p.is_absolute() else p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    p=Path('reports/lbm3d_refinement_gpu_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gates':gates,'cpu_gpu_profile_max_absolute_error':error,'gpu_seconds':[first['seconds'],second['seconds']]}),flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

"""Hardware-specific paired specialization gate for the complete refined step.

Only the collision matrix representation changes. All coupling launches,
substeps and flux corrections remain in the timing. This is not a DNS gate.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
from measure_lbm3d_refinement_sgs import initialize
from measure_lbm3d_refinement_gpu import run


def main():
    options=dict(recursive=True,bulk_tau_fine=1.,conserved_reflux=True,balanced_reflux=True,
                 tau_fine=.5032,force_fine=1e-8)
    records=[];states=[]
    for factory in (ChannelSimulation,SpecializedChannelSimulation,SpecializedChannelSimulation):
        report,state=run(initialize(RefinedChannelGPU,channel_factory=factory,**options))
        records.append(report);states.append(state)
    exact=all(np.array_equal(a,b) for other in states[1:] for a,b in zip(states[0],other))
    history_exact=all(r['history']==records[0]['history'] for r in records[1:])
    valid=all(len(r['history'])==20 and all(h['mass_exact'] and h['momentum_residual']==[0,0,0] and not any(h['flags']) for h in r['history']) for r in records)
    durations=[[],[]]
    if exact and history_exact and valid:
        sims=[RefinedChannelGPU(nx=64,height=96,nz=64,wall_cells=24,cs_fine=.1,channel_factory=f,**options)
              for f in (ChannelSimulation,SpecializedChannelSimulation)]
        for sim in sims:
            for _ in range(2):sim.step()
        wp.synchronize_device('cuda:0')
        for order in ((0,1),(1,0),(0,1)):
            for index in order:
                started=time.perf_counter()
                for _ in range(100):sims[index].step()
                wp.synchronize_device('cuda:0')
                durations[index].append(time.perf_counter()-started)
        exact=exact and all(np.array_equal(a.numpy(),b.numpy()) for a,b in zip([*sims[0].fine,sims[0].coarse],[*sims[1].fine,sims[1].coarse]))
        valid=valid and all(not int(b.failure.numpy()[0]) for s in sims for b in [*s.fine,s.coarse])
    gains=[a/b for a,b in zip(*durations)]
    gates={'small_full_states_and_timed_states_exact':exact,'repeated_ledger_history_exact':history_exact,
           'all_2000step_ledgers_exact_and_flags_clear':valid,'faster_in_all_three_pairs':len(gains)==3 and all(v>1 for v in gains)}
    paths=[Path(__file__),*Path('src/kernel_engine/lbm').glob('lbm3d_*.py')]
    result={'scope':'full refined-step implementation parity and timing; no turbulent accuracy claim',
            'device':wp.get_device('cuda:0').name,'fine_equivalent_extent':[64,96,64],'wall_cells':24,
            'fine_steps_per_timing':200,'durations_seconds':{'runtime':durations[0],'specialized':durations[1]},
            'speedups':gains,'gates':gates,'passed':all(gates.values()),'numerical_runs':records,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    p=Path('reports/lbm3d_refined_specialized_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'gates':gates,'speedups':gains}),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

"""Three-level fixed-point convergence of the unchanged Hermite MRT trajectory.

This is a new, declared precision-refinement experiment. Historical 36-vs40
failures remain failures. The 40-bit solution is compared against a finer
44-bit reference at every saved checkpoint, not only at the last step.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm import lbm3d_mrt_les as lb
from measure_lbm3d_mrt_les import initial,metrics,digest

OUT=Path('reports/lbm3d_precision_convergence_v1/report.json')
EXPECTED={36:'e8e9bf6ae38b063c32fefabde34a85221c1a48b13431d6921c066e19bfa4e90e',
          40:'170c393995e4b9cae8d56b3b694e38c02176b922796a04ee70da0f6152b4551a'}


def main():
    report={'source_sha256':hashlib.sha256(Path(lb.__file__).read_bytes()).hexdigest(),
            'preregistered':{'grid':[64,64,64],'steps':4096,'checkpoint':64,'u0':.2,'tau':.5001,'Cs':.1,
                            'mode':'hermite_mrt','bits':[36,40,44],'repeat_bits':44,
                            'velocity_atol':1e-5,'error_contraction_at_least':4,
                            'require_prior_36_40_final_hashes':EXPECTED,
                            'scope':'40-vs44 max velocity difference over all checkpoints; 36-vs40 failure retained'},
            'history':[]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    def save(): OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    sims=[];original=[]
    for bits in [36,40,44,44]:
        q=lb.quantize(initial(),bits);original.append(lb.ledger(q))
        sims.append(lb.Simulation(q,tau=.5001,cs=.1,mode='hermite_mrt',bits=bits,device='cuda:0'))
    report['initial_ledgers']=original;save()
    exact=True;conserved=True;stable=True;repeat_exact=True
    max_errors=np.zeros(2);elapsed=[]
    for step in range(64,4097,64):
        row={'step':step,'states':[]};states=[];velocities=[]
        for idx,(sim,bits) in enumerate(zip(sims,[36,40,44,44])):
            t=time.perf_counter();sim.step(64);q=sim.numpy();elapsed.append(time.perf_counter()-t)
            m=metrics(q,bits);l=lb.ledger(q);flag=int(sim.failure.numpy()[0]);h=digest(q)
            conserved=conserved and l==original[idx]
            stable=stable and all(np.isfinite(v) for v in m.values()) and .5<=m['rho_min'] and m['rho_max']<=1.5 and m['max_speed']<.5 and m['energy']<=1.05*.005 and not flag
            row['states'].append({'bits':bits,**m,'ledger':l,'flag':flag,'sha256':h})
            states.append(q)
            if idx<3:velocities.append(lb.fields(q,bits)[1])
        repeat_exact=repeat_exact and np.array_equal(states[2],states[3])
        errors=[float(np.max(np.abs(velocities[0]-velocities[1]))),float(np.max(np.abs(velocities[1]-velocities[2])))]
        max_errors=np.maximum(max_errors,errors)
        row['velocity_max_abs_difference_36_40_and_40_44']=errors
        report['history'].append(row);save()
        if step%512==0:print(json.dumps({'step':step,'errors':errors,'stable':bool(stable),'ledger_exact':bool(conserved)}),flush=True)
        if not stable:break
    final=report['history'][-1]['states']
    prior=all(final[i]['sha256']==EXPECTED[b] for i,b in enumerate([36,40]))
    ratio=float(max_errors[0]/max_errors[1]) if max_errors[1] else None
    gates={'complete_stable':bool(stable and step==4096),'integer_ledgers_exact':bool(conserved),
           '44_bit_full_repeat_exact_at_every_checkpoint':bool(repeat_exact),
           'old_36_and_40_states_unchanged':bool(prior),'40_vs44_velocity_within_1e_minus5':bool(max_errors[1]<=1e-5),
           'error_contracts_at_least_4x':bool(max_errors[1]<=max_errors[0]/4)}
    report.update(gates=gates,passed=all(gates.values()),maximum_checkpoint_errors=max_errors.tolist(),
                  error_contraction=ratio,step_and_readback_s=sum(elapsed),
                  final_step=step,warp=wp.config.version,device=str(wp.get_device('cuda:0')))
    save();print(json.dumps({k:v for k,v in report.items() if k not in ['history','preregistered']}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

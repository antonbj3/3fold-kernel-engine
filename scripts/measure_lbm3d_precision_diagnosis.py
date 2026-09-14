"""Separate repeated rounding injection, perturbation growth and repair bias."""
import hashlib,json
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation as F64
from kernel_engine.lbm.lbm3d_channel_fp32 import FP32ChannelSimulation as F32
from kernel_engine.lbm.lbm3d_channel_symmetric32 import Symmetric32ChannelSimulation as Balanced
from measure_lbm3d_mrt_les import initial,metrics


def main():
    opts=dict(force_density=0.,tau=.5001,cs=.1,bulk_tau=1.,recursive=True,device='cuda:0')
    q=lb.quantize(initial());ledger=lb.ledger(q)
    sims={'reference':F64(q,**opts),'fp32':F32(q,**opts),'symmetric_repair':Balanced(q,**opts)}
    for s in sims.values():s.step()
    # Same first-step FP32 perturbation, subsequently evolved only by FP64.
    sims['single_injection_then_fp64']=F64(sims['fp32'].numpy(),**opts)
    def difference(q,ref):
        rho,u=lb.fields(q);rr,v=lb.fields(ref);delta=u-v
        return {'velocity_max':float(np.max(np.abs(delta))),'velocity_RMS':float(np.sqrt(np.mean(delta**2))),
                'density_max':float(np.max(np.abs(rho-rr))),
                'relative_energy_difference':float(metrics(q,40)['energy']/metrics(ref,40)['energy']-1),
                'max_error_component_and_cell':list(map(int,np.unravel_index(np.argmax(np.abs(delta)),delta.shape)))}
    rows=[];done=1
    for target in (1,64,256,512,1024):
        for s in sims.values():s.step(target-done)
        done=target;arrays={k:s.numpy() for k,s in sims.items()};ref=arrays['reference']
        row={'step':target,'differences':{k:difference(a,ref) for k,a in arrays.items() if k!='reference'},
             'all_ledgers_exact':all(lb.ledger(a)==ledger for a in arrays.values()),'flags_clear':not any(int(s.failure.numpy()[0]) for s in sims.values())}
        if target in (1,512,1024):
            scratch={k:cls(ref,**opts) for k,cls in [('fp64',F64),('fp32',F32),('symmetric_repair',Balanced)]}
            for s in scratch.values():s.step()
            truth=scratch['fp64'].numpy()
            row['one_step_from_identical_state']={k:difference(s.numpy(),truth) for k,s in scratch.items() if k!='fp64'}
            del scratch
        rows.append(row)
        if not row['all_ledgers_exact'] or not row['flags_clear']:break
    result={'scope':'diagnostic only, fixed precision admission limits unchanged; single-injection control is not proof of chaos','device':sims['reference'].device,'history':rows,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('scripts/measure_lbm3d_precision_diagnosis.py'),Path('src/kernel_engine/lbm/lbm3d_channel_symmetric32.py')]}}
    p=Path('reports/lbm3d_precision_diagnosis_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)

if __name__=='__main__':main()

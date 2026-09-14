"""Spatially varying 3D SGS refinement parity and conservation gate."""
import hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm.lbm3d_refinement import ReferenceRefinedChannel
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
from measure_lbm3d_refinement_gpu import run


def initialize(cls,**kwargs):
    sim=cls(nx=8,height=32,nz=8,wall_cells=8,cs_fine=.1,**kwargs)
    for block in [*sim.fine,sim.coarse]:
        q=block.numpy();x,y,z=np.indices(block.shape)
        amplitude=2**(block.bits-16)
        perturb=np.rint(amplitude*np.sin(2*np.pi*x/block.shape[0])*np.cos(2*np.pi*z/block.shape[2])*(1+y)).astype(np.int64)
        perturb[block.args[0].numpy()!=0]=0
        for positive,negative in [(1,2),(3,4),(5,6)]:q[positive]+=perturb;q[negative]-=perturb
        sim._replace(block,q)
    sim.initial_ledger=sim.ledger()
    return sim


def main(*,recursive=False,bulk_tau_fine=None,conserved_reflux=False,output='reports/lbm3d_refinement_sgs_v1/report.json'):
    reference,a=run(initialize(ReferenceRefinedChannel,recursive=recursive,bulk_tau_fine=bulk_tau_fine,conserved_reflux=conserved_reflux))
    first,b=run(initialize(RefinedChannelGPU,recursive=recursive,bulk_tau_fine=bulk_tau_fine,conserved_reflux=conserved_reflux))
    second,c=run(initialize(RefinedChannelGPU,recursive=recursive,bulk_tau_fine=bulk_tau_fine,conserved_reflux=conserved_reflux))
    errors=[float(np.max(np.abs(x.astype(float)-y.astype(float)))/2**bits) for x,y,bits in zip(a,b,[40,40,43])]
    gates={'full_gpu_repeat_exact':all(np.array_equal(x,y) for x,y in zip(b,c)),
           'repeat_history_exact':first['history']==second['history'],
           'all_population_density_cpu_gpu_error_at_most_1e_10':max(errors)<=1e-10,
           'every_mass_impulse_ledger_exact_and_flags_clear':all(len(r['history'])==20 and all(h['mass_exact'] and h['momentum_residual']==[0,0,0] and h['flags']==[0,0,0] for h in r['history']) for r in [reference,first,second])}
    # The inherited observer's laminar-profile distance is diagnostic only:
    # this case has nonzero SGS and three-dimensional initial perturbations.
    report={'scope':'SGS level coupling numerical gate; not turbulent DNS validation',
            'physical_filter':'fine Cs=0.1, coarse Cs=0.05; acoustic 2:1','recursive':recursive,'bulk_tau_fine':bulk_tau_fine,'conserved_reflux':conserved_reflux,'device':wp.get_device('cuda:0').name,
            'population_density_max_absolute_errors':errors,'gates':gates,'passed':all(gates.values()),
            'cpu':reference,'gpu_first':first,'gpu_second':second,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('src/kernel_engine/lbm').glob('lbm3d_*.py'),Path('scripts/measure_lbm3d_refinement_sgs.py'),Path('scripts/measure_lbm3d_refinement_gpu.py')]}}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'gates':gates,'errors':errors}),flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

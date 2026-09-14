"""Large physical-volume gate with explicit two-int64-word exact ledgers."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
from kernel_engine.lbm.lbm3d_ledger import int64_limbs


def run():
    sim=RefinedChannelGPU(nx=384,height=128,nz=192,wall_cells=16,
                         force_fine=1e-7,cs_fine=.1,ledger_mode='int64_limbs')
    initial=sim.initial_ledger;history=[];start=time.perf_counter()
    for macro in range(1,251):
        sim.step()
        if macro%50==0:
            ledger=sim.ledger();wall=sim.wall_impulse()
            residual=[ledger[k+1]-initial[k+1]+int(wall[k])-(sim.steps*sim.nx*sim.h*sim.nz*sim.force_units if k==0 else 0) for k in range(3)]
            flags=[int(s.failure.numpy()[0]) for s in [*sim.fine,sim.coarse]]
            history.append({'steps':sim.steps,'mass_exact':ledger[0]==initial[0],
                'ledger_int64_limbs':[int64_limbs(v) for v in ledger],
                'wall_impulse_int64_limbs':[int64_limbs(v) for v in wall],
                'momentum_residual':residual,'flags':flags})
            if any(flags):break
    wp.synchronize_device('cuda:0');elapsed=time.perf_counter()-start
    hashes=[hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in [*sim.fine,sim.coarse]]
    return {'initial_mass':initial[0],'initial_mass_int64_limbs':int64_limbs(initial[0]),
            'history':history,'seconds_including_diagnostics':elapsed,'full_state_sha256':hashes}


def main():
    a=run();b=run()
    gates={'inventory_exceeds_single_signed_int64':a['initial_mass']>=2**63,
           'repeated_full_state_hashes_exact':a['full_state_sha256']==b['full_state_sha256'],
           'repeated_histories_exact':a['history']==b['history'],
           'every_mass_momentum_ledger_exact_and_flags_clear':all(len(r['history'])==5 and all(h['mass_exact'] and h['momentum_residual']==[0,0,0] and h['flags']==[0,0,0] for h in r['history']) for r in [a,b])}
    report={'scope':'500 fine-step large inventory gate; not long-time turbulent DNS',
            'fine_equivalent_extent':[384,128,192],'fine_wall_layers_each':16,
            'active_cells':384*192*32+192*96*48,'uniform_fine_cells':384*128*192,
            'population_fraction_bits':{'fine':40,'coarse':43},
            'ledger_encoding':'signed int64 high * 2**63 + nonnegative int64 low',
            'first':a,'second':b,'gates':gates,'passed':all(gates.values()),
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('src/kernel_engine/lbm').glob('lbm3d_*.py'),Path('scripts/measure_lbm3d_refinement_wide_ledger.py')]}}
    p=Path('reports/lbm3d_refinement_wide_ledger_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gates':gates,'initial_mass':a['initial_mass'],'seconds':[a['seconds_including_diagnostics'],b['seconds_including_diagnostics']]}),flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

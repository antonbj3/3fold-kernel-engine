"""CPU coupling-reference gate; not a refined GPU/DNS claim."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_refinement as amr


def main():
    sim=amr.ReferenceRefinedChannel();initial=sim.initial_ledger
    start=time.perf_counter();history=[];conservative=True;valid=True
    for macro in range(1,1001):
        sim.step()
        if macro%50==0:
            l=sim.ledger();wall=sim.wall_impulse()
            residual=[l[a+1]-initial[a+1]+int(wall[a])-(sim.steps*sim.nx*sim.h*sim.nz*sim.force_units if a==0 else 0) for a in range(3)]
            flags=[int(s.failure.numpy()[0]) for s in [*sim.fine,sim.coarse]]
            conservative=conservative and l[0]==initial[0] and residual==[0,0,0]
            valid=valid and flags==[0,0,0]
            y,u=sim.profile();truth=sim.gf/(2*((sim.tf-.5)/3))*y*(sim.h-y)
            error=float(np.linalg.norm(u-truth)/np.linalg.norm(truth))
            history.append({'fine_steps':sim.steps,'mass_exact':l[0]==initial[0],'momentum_residual':residual,'flags':flags,'profile_relative_L2':error})
            if not valid:break
    elapsed=time.perf_counter()-start
    passed=bool(valid and conservative and sim.steps==2000 and error<.02)
    r={'scope':'CPU two-wall-block coupling reference, Cs=0; not GPU or turbulent validation',
       'preregistered':{'fine_extent':[8,32,8],'wall_fine_layers_each':8,'fine_tau':.8,'coarse_tau':.65,'steps_fine':2000,'profile_relative_L2_limit':.02},
       'history':history,'passed':passed,'seconds':elapsed,'y':y.tolist(),'profile':u.tolist(),'analytic':truth.tolist(),
       'source_sha256':hashlib.sha256(Path(amr.__file__).read_bytes()).hexdigest()}
    p=Path('reports/lbm3d_refinement_laminar_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r),flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())

"""Analytical forced-channel gate before turbulent or refined-grid runs."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm import lbm3d_channel as ch


def main():
    nx,h,nz=8,32,8;tau=.8;nu=(tau-.5)/3;steps=12000;bits=40
    shape=(nx,h+2,nz);mask=np.zeros(shape,np.int32);mask[:,0,:]=1;mask[:,-1,:]=1
    q=lb.quantize(lb.equilibrium(np.ones(shape),np.zeros((3,)+shape)))
    runs=[];last=None
    for repeat in range(2):
        t=time.perf_counter();s=ch.ChannelSimulation(q,force_density=1e-5,tau=tau,cs=0,solid=mask,device='cuda:0')
        s.step(steps);out=s.numpy();seconds=time.perf_counter()-t
        rho,u=lb.fields(out);physical=u[0]+.5*s.force_density/rho
        profile=physical[:,1:-1,:].mean(axis=(0,2));y=np.arange(h)+.5
        truth=s.force_density/(2*nu)*y*(h-y)
        l0=lb.ledger(q);l1=lb.ledger(out);wall=s.wall_impulse_numpy()
        balance=[l1[i+1]-l0[i+1]+int(wall[i])-(steps*nx*h*nz*s.force_units if i==0 else 0) for i in range(3)]
        r={'seconds':seconds,'L2_relative':float(np.linalg.norm(profile-truth)/np.linalg.norm(truth)),
           'max_profile_error_over_peak':float(np.max(np.abs(profile-truth))/max(truth)),
           'mass_exact':l0[0]==l1[0],'momentum_balance_residual':balance,
           'flag':int(s.failure.numpy()[0]),'sha256':hashlib.sha256(out.tobytes()).hexdigest(),
           'profile':profile.tolist(),'analytic':truth.tolist(),'force_units':s.force_units,'force_density':s.force_density}
        runs.append(r)
        if last is not None:assert np.array_equal(last,out)
        last=out
    passed=all(r['L2_relative']<.01 and r['max_profile_error_over_peak']<.02 and r['mass_exact'] and r['momentum_balance_residual']==[0,0,0] and r['flag']==0 for r in runs)
    report={'preregistered':{'grid':[nx,h+2,nz],'fluid_height':h,'tau':tau,'Cs':0,'steps':steps,'profile_L2_limit':.01,'peak_normalized_limit':.02},
            'runs':runs,'full_repeat_exact':True,'passed':passed,'source_sha256':hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest()}
    p=Path('reports/lbm3d_channel_laminar_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())

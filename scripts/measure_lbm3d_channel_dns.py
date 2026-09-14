"""Uniform-grid Re_tau180 control before block refinement; fixed averaging gates."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import warp as wp
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm import lbm3d_channel as ch
from kernel_engine.lbm.lbm3d_channel_statistics import accumulate

ROOT=Path('reports/lbm3d_channel_dns_uniform_v1')
NX,H,NZ=192,64,96
BURN,AVERAGE,BLOCK,SAMPLE=80000,160000,20000,200
UTAU=.004;NU=UTAU*(H/2)/180;TAU=.5+3*NU
INITIAL_FILTER_CELLS=2.
THRESHOLDS={'log_mean_relative_max':.05,'whole_mean_relative_L2':.10,'stress_peak_normalized_RMS':.20,
            'friction_velocity_relative':.05,'mean_half_window_relative_max':.05,'stress_half_window_peak_RMS':.15}


def initial():
    shape=(NX,H+2,NZ);mask=np.zeros(shape,np.int32);mask[:,0,:]=1;mask[:,-1,:]=1
    eta=(np.arange(H+2)-.5)/H;envelope=np.maximum(0,4*eta*(1-eta))[None,:,None]
    rng=np.random.default_rng(20260914);u=np.zeros((3,)+shape)
    for axis in range(3):
        noise=gaussian_filter(rng.normal(size=shape),INITIAL_FILTER_CELLS,mode=('wrap','reflect','wrap'))
        u[axis]=noise/noise.std()*(2*UTAU)*envelope
    u[0]+=18*UTAU*np.maximum(0,1-(2*eta-1)**8)[None,:,None]
    force=UTAU**2/(H/2)
    u[0]-=.5*force;u[:,mask!=0]=0
    return lb.quantize(lb.equilibrium(np.ones(shape),u)),mask,force


def folded(raw,count):
    # Reflect normal velocity and shear moments when folding the upper wall.
    values=raw.astype(float)/(2**32*count)
    sign=np.array([1,1,-1,1,1,1,1,-1,1,-1])[:,None]
    v=.5*(values[:,1:H//2+1]+sign*values[:,H:H//2:-1])
    stresses=np.array([v[4]-v[1]**2,v[5]-v[2]**2,v[6]-v[3]**2,
                       v[7]-v[1]*v[2],v[8]-v[1]*v[3],v[9]-v[2]*v[3]])
    return v[1],stresses,v[0]


def main(final_observer=None,bulk_tau=None,recursive=False,simulation_class=None):
    ROOT.mkdir(parents=True,exist_ok=True)
    ref=Path('tests/data/lbm_channel');means=np.loadtxt(ref/'chan180.means');stress=np.loadtxt(ref/'chan180.reystress')
    report={'grid':[NX,H+2,NZ],'fluid_cells':NX*H*NZ,'wall_locations':[-.5,H-.5],
            'Re_tau_target':180,'reference_Re_tau':178.12,'first_cell_wall_units_nominal':.5*UTAU/NU,
            'domain_over_half_height':[NX/(H/2),2,NZ/(H/2)],'tau':TAU,'bulk_tau':bulk_tau,'recursive_third_order':recursive,'Cs':.1,'bits':40,
            'burn_steps':BURN,'averaging_steps':AVERAGE,'block_steps':BLOCK,'sample_every':SAMPLE,
            'burn_outer_times':BURN*UTAU/(H/2),'averaging_outer_times':AVERAGE*UTAU/(H/2),
            'initial_filter_cells':INITIAL_FILTER_CELLS,
            'thresholds':THRESHOLDS,'source_sha256':hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest(),
            'reference_provenance':json.loads((ref/'provenance.json').read_text()),'blocks':[]}
    def save(): (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    simulation_class=ch.ChannelSimulation if simulation_class is None else simulation_class
    report['simulation_class']=simulation_class.__name__
    q,mask,force=initial();s=simulation_class(q,force_density=force,tau=TAU,solid=mask,device='cuda:0',bulk_tau=bulk_tau,recursive=recursive)
    first_ledger=lb.ledger(q);report['initial_ledger']=first_ledger
    report['force_density_applied']=s.force_density;report['force_integer_units']=s.force_units;save()
    stats=wp.zeros((10,H+2),dtype=wp.int64,device='cuda:0')
    t=time.perf_counter();block_data=[];wall_at_burn=None;valid=True
    for done in range(BLOCK,BURN+AVERAGE+1,BLOCK):
        stat_time=0.;stats.zero_()
        for _ in range(BLOCK//SAMPLE):
            s.step(SAMPLE)
            if done>BURN:
                a=time.perf_counter()
                wp.launch(accumulate,s.shape,[s.a,s.args[0],s.args[1],wp.float64(2**40),wp.int64(s.force_units),stats,s.failure],device='cuda:0')
                stat_time+=time.perf_counter()-a
        out=s.numpy();l=lb.ledger(out);wall=s.wall_impulse_numpy();flag=int(s.failure.numpy()[0])
        rho,u=lb.fields(out);u[0]+=.5*s.force_density/rho
        fluid=mask==0;speed=np.sqrt(np.sum(u*u,axis=0))
        balance=[l[i+1]-first_ledger[i+1]+int(wall[i])-(done*NX*H*NZ*s.force_units if i==0 else 0) for i in range(3)]
        valid=bool(not flag and l[0]==first_ledger[0] and balance==[0,0,0] and np.isfinite(speed).all() and speed[fluid].max()<.5)
        r={'step':done,'elapsed_s':time.perf_counter()-t,'mass_exact':l[0]==first_ledger[0],
           'momentum_residual':balance,'wall_impulse':wall.tolist(),'flag':flag,
           'rho_min':float(rho[fluid].min()),'rho_max':float(rho[fluid].max()),'max_speed':float(speed[fluid].max()),
           'bulk_velocity':float(u[0,fluid].mean()),'state_sha256':hashlib.sha256(out.tobytes()).hexdigest(),
           'statistics_launch_host_s':stat_time}
        if done==BURN:wall_at_burn=wall.copy()
        if done>BURN:
            raw=stats.numpy();block_data.append(raw);r['plane_integer_statistics']=raw.tolist()
        report['blocks'].append(r);save();print(json.dumps({k:v for k,v in r.items() if k!='plane_integer_statistics'}),flush=True)
        if not valid:break
    report['total_s']=time.perf_counter()-t
    if not valid or done!=BURN+AVERAGE:
        report.update(passed=False,blocker='state or integer conservation failed before complete averaging');save();return 1
    count=(AVERAGE//SAMPLE)*NX*NZ
    raw=np.sum(block_data,axis=0,dtype=np.int64);mean,reynolds,density=folded(raw,count)
    wall_shear=float((int(wall[0])-int(wall_at_burn[0]))/2**40/(2*NX*NZ*AVERAGE))
    utau=np.sqrt(abs(wall_shear));yp=(np.arange(H//2)+.5)*utau/NU
    up=mean/utau;rp=reynolds/utau**2
    ref_u=np.interp(yp,means[:,1],means[:,2])
    ref_r=np.array([np.interp(yp,stress[:,1],stress[:,i]) for i in range(2,8)])
    log=(yp>=30)&(yp<=100)
    mean_error=float(np.max(np.abs(up[log]/ref_u[log]-1)))
    whole_error=float(np.linalg.norm(up-ref_u)/np.linalg.norm(ref_u))
    peaks=np.max(np.abs(ref_r[:4]),axis=1)
    stress_errors=np.sqrt(np.mean((rp[:4]-ref_r[:4])**2,axis=1))/peaks
    half=[]
    for b in [block_data[:len(block_data)//2],block_data[len(block_data)//2:]]:
        half.append(folded(np.sum(b,axis=0,dtype=np.int64),count//2)[:2])
    mean_drift=float(np.max(np.abs((half[0][0]-half[1][0])/mean)))
    stress_drift=np.sqrt(np.mean(((half[0][1][:4]-half[1][1][:4])/utau**2)**2,axis=1))/peaks
    gates={'conservation_and_state':valid,'log_mean':mean_error<=THRESHOLDS['log_mean_relative_max'],
           'whole_mean':whole_error<=THRESHOLDS['whole_mean_relative_L2'],
           'reynolds_stresses':bool(np.all(stress_errors<=THRESHOLDS['stress_peak_normalized_RMS'])),
           'friction_velocity':bool(abs(utau/UTAU-1)<=THRESHOLDS['friction_velocity_relative']),
           'mean_stationarity':mean_drift<=THRESHOLDS['mean_half_window_relative_max'],
           'stress_stationarity':bool(np.all(stress_drift<=THRESHOLDS['stress_half_window_peak_RMS']))}
    report.update(gates=gates,passed=all(gates.values()),wall_shear=wall_shear,u_tau_measured=float(utau),
                  Re_tau_measured=float(utau*(H/2)/NU),first_cell_wall_units=float(.5*utau/NU),
                  y_plus=yp.tolist(),U_plus=up.tolist(),reference_U_plus=ref_u.tolist(),
                  reynolds_plus=rp.tolist(),reference_reynolds_plus=ref_r.tolist(),
                  log_mean_relative_max_error=mean_error,whole_mean_relative_L2_error=whole_error,
                  stress_peak_normalized_RMS=stress_errors.tolist(),mean_half_window_relative_max=mean_drift,
                  stress_half_window_peak_RMS=stress_drift.tolist(),
                  whole_run_MLUPS=NX*H*NZ*(BURN+AVERAGE)/report['total_s']/1e6,
                  note='Uniform control only. No block refinement or draft-tube validation claimed. Sampled integer moments use2^-32; actual populations use2^-40.')
    save();print(json.dumps({'gates':gates,'mean_error':mean_error,'stress_errors':stress_errors.tolist(),'u_tau':utau,'passed':report['passed']}),flush=True)
    if final_observer is not None:final_observer(s,mask,report)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

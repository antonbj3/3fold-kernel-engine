"""Refined channel gate admitted only with a passed matching uniform control."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
import measure_lbm3d_channel_dns as dns
from kernel_engine.lbm.lbm3d_refinement import restrict_child_mass
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
from kernel_engine.lbm.lbm3d_channel_statistics import accumulate
from kernel_engine.lbm.lbm3d_refined_statistics import fold_wall_blocks
from kernel_engine.lbm.lbm3d_ledger import int64_limbs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--uniform-report',type=Path,required=True);parser.add_argument('--sensitivity-report',type=Path,required=True);parser.add_argument('--wall-cells',type=int,default=8)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--conserved-reflux',action='store_true')
    parser.add_argument('--balanced-reflux',action='store_true')
    parser.add_argument('--specialized',action='store_true')
    args=parser.parse_args();control=json.loads(args.uniform_report.read_text());sensitivity=json.loads(args.sensitivity_report.read_text())
    if control.get('grid')==[288,98,144]:
        from measure_lbm3d_recursive_resolution import configure
        configure()
    if not control.get('passed') or not control.get('recursive_third_order') or control.get('bulk_tau') not in (None,1.) or control['grid']!=[dns.NX,dns.H+2,dns.NZ]:
        raise ValueError('passed matching recursive uniform control required')
    if sensitivity['base_state_sha256']!=control['blocks'][-1]['state_sha256'] or not all(sensitivity['gates'].values()):
        raise ValueError('converged sensitivity measured at this uniform endpoint required')
    if args.diagnostic:dns.BURN,dns.AVERAGE,dns.BLOCK,dns.SAMPLE=0,2000,100,100
    nx,h,nz=dns.NX,dns.H,dns.NZ;nf=args.wall_cells
    start=time.perf_counter();q,mask,force=dns.initial()
    from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation
    sim=RefinedChannelGPU(nx=nx,height=h,nz=nz,wall_cells=nf,tau_fine=dns.TAU,force_fine=force,cs_fine=.1,recursive=True,bulk_tau_fine=control.get("bulk_tau"),conserved_reflux=args.conserved_reflux,balanced_reflux=args.balanced_reflux,channel_factory=SpecializedChannelSimulation if args.specialized else None)
    sim._replace(sim.fine[0],q[:,:,:nf+2,:]);sim._replace(sim.fine[1],q[:,:,h-nf:h+2,:])
    coarse=sim.coarse.numpy();density=restrict_child_mass(q[:,:,1:-1,:]).astype(float)/2**43
    coarse[:,:,1:-1,:]=sim._convert(density,sim.tf,sim.tc,2,sim.gf,2*sim.gf,43)
    sim._replace(sim.coarse,coarse);initial=sim.ledger();sim.initial_ledger=initial
    blocks=[*sim.fine,sim.coarse];stats=[wp.zeros((10,s.shape[1]),dtype=wp.int64,device='cuda:0') for s in blocks]
    active=2*nx*nf*nz+(nx//2)*((h-2*nf)//2)*(nz//2)
    report={'scope':'recursive two-wall-block DNS and uniform-profile gate','fine_equivalent_extent':[nx,h,nz],'wall_fine_layers_each':nf,
        'active_cells':active,'uniform_cells':nx*h*nz,'cell_fraction':active/(nx*h*nz),'allocated_cells':sum(np.prod(s.shape).item() for s in blocks),
        'setup_s':time.perf_counter()-start,'fine_tau':sim.tf,'coarse_tau':sim.tc,'fine_bulk_tau':sim.bf,'coarse_bulk_tau':sim.bc,'Cs_fine':sim.csf,'Cs_coarse':sim.csc,'population_fraction_bits':[40,43],'burn_steps':dns.BURN,'averaging_steps':dns.AVERAGE,'sample_every':dns.SAMPLE,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).relative_to(Path.cwd()),Path('src/kernel_engine/lbm/lbm3d_channel.py'),Path('src/kernel_engine/lbm/lbm3d_refinement_gpu.py'),Path('src/kernel_engine/lbm/lbm3d_refinement.py')]},'uniform_report_sha256':hashlib.sha256(args.uniform_report.read_bytes()).hexdigest(),
        'sensitivity_report_sha256':hashlib.sha256(args.sensitivity_report.read_bytes()).hexdigest(),'thresholds':dns.THRESHOLDS,
        'uniform_parity_limits':{'mean_relative_L2':.05,'stress_peak_RMS':.10},'blocks':[]}
    report['diagnostic_only']=args.diagnostic
    report['conserved_reflux']=args.conserved_reflux
    report['balanced_reflux']=args.balanced_reflux
    report['specialized']=args.specialized
    if args.specialized:
        p=Path('src/kernel_engine/lbm/lbm3d_channel_specialized.py')
        report['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    root=Path('reports/lbm3d_refined_startup_v1' if args.diagnostic else 'reports/lbm3d_refined_dns_v1');root.mkdir(parents=True,exist_ok=True)
    def save(): (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    samples_per_block=dns.BLOCK//dns.SAMPLE;records=[];wall_at_burn=None;valid=True;started=time.perf_counter();save()
    def folded(selected):
        raw=[np.sum([record[i] for record in selected],axis=0,dtype=np.int64) for i in range(3)]
        return fold_wall_blocks(raw,nx=nx,height=h,nz=nz,wall_cells=nf,samples=len(selected)*samples_per_block)
    for done in range(dns.BLOCK,dns.BURN+dns.AVERAGE+1,dns.BLOCK):
        for a in stats:a.zero_()
        for _ in range(samples_per_block):
            for _ in range(dns.SAMPLE//2):sim.step()
            if done>dns.BURN:
                for s,a in zip(blocks,stats):wp.launch(accumulate,s.shape,[s.a,s.args[0],s.args[1],wp.float64(2**s.bits),wp.int64(s.force_units),a,s.failure],device='cuda:0')
        flags=[int(s.failure.numpy()[0]) for s in blocks]
        if args.diagnostic or any(flags):
            extrema=[]
            for s in blocks:
                q=s.numpy();rho=q.sum(axis=0,dtype=np.float64)/2**s.bits
                fluid=s.args[0].numpy()==0
                extrema.append({'rho_min':float(rho[fluid].min()),'rho_max':float(rho[fluid].max()),
                                'population_min':float(q.min()/2**s.bits),'population_max':float(q.max()/2**s.bits)})
            report.setdefault('state_checks',[]).append({'step':done,'flags':flags,'blocks':extrema});save()
        if any(flags):
            report.update(passed=False,blocker='collision or statistics state guard failed before ledger reduction');save();return 1
        ledger=sim.ledger();wall=sim.wall_impulse()
        residual=[ledger[k+1]-initial[k+1]+int(wall[k])-(done*nx*h*nz*sim.force_units if k==0 else 0) for k in range(3)]
        valid=valid and ledger[0]==initial[0] and residual==[0,0,0] and flags==[0,0,0]
        row={'step':done,'mass_exact':ledger[0]==initial[0],'momentum_residual':residual,'flags':flags,'wall_int64_limbs':[int64_limbs(int(v)) for v in wall],'elapsed_s':time.perf_counter()-started}
        if done==dns.BURN:wall_at_burn=list(map(int,wall))
        if done>dns.BURN:
            raw=[a.numpy() for a in stats];records.append(raw);row['plane_integer_statistics']=[a.tolist() for a in raw]
        report['blocks'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='plane_integer_statistics'}),flush=True)
        if not valid:break
    report['seconds']=time.perf_counter()-started
    if args.diagnostic:
        report.update(passed=bool(valid),scope='startup diagnostics only; no turbulent DNS validation');save();return 0 if valid else 1
    if not valid or done!=dns.BURN+dns.AVERAGE:
        report.update(passed=False,blocker='state or conservation failed');save();return 1
    y,mean,stress,rho=folded(records);shear=(int(wall[0])-wall_at_burn[0])/2**40/(2*nx*nz*dns.AVERAGE);utau=np.sqrt(abs(shear));yp=y*utau/dns.NU;up=mean/utau;rp=stress/utau**2
    reference=Path('tests/data/lbm_channel');m=np.loadtxt(reference/'chan180.means');rs=np.loadtxt(reference/'chan180.reystress')
    ru=np.interp(yp,m[:,1],m[:,2]);rr=np.array([np.interp(yp,rs[:,1],rs[:,i]) for i in range(2,8)]);peaks=np.max(np.abs(rr[:4]),axis=1)
    # Interpolate onto the same fine-wall-distance sample positions for L2.
    uniform_yp=(np.arange(h//2)+.5)*utau/dns.NU
    interp_u=np.interp(uniform_yp,yp,up);ref_u=np.interp(uniform_yp,m[:,1],m[:,2]);log=(yp>=30)&(yp<=100)
    log_error=float(np.max(np.abs(up[log]/ru[log]-1)));mean_error=float(np.linalg.norm(interp_u-ref_u)/np.linalg.norm(ref_u))
    stress_error=np.sqrt(np.mean((rp[:4]-rr[:4])**2,axis=1))/peaks
    halves=[folded(part) for part in [records[:len(records)//2],records[len(records)//2:]]]
    mean_drift=float(np.max(np.abs((halves[0][1]-halves[1][1])/mean)));stress_drift=np.sqrt(np.mean(((halves[0][2][:4]-halves[1][2][:4])/utau**2)**2,axis=1))/peaks
    control_u=np.interp(yp,control['y_plus'],control['U_plus']);control_r=np.array([np.interp(yp,control['y_plus'],r) for r in control['reynolds_plus']])
    control_mean_error=float(np.linalg.norm(up-control_u)/np.linalg.norm(control_u));control_stress_error=np.sqrt(np.mean((rp[:4]-control_r[:4])**2,axis=1))/peaks
    gates={'state_and_conservation':valid,'log_mean':log_error<=.05,'whole_mean':mean_error<=.10,'Reynolds_stresses':bool(np.all(stress_error<=.20)),
        'wall_friction':bool(abs(utau/dns.UTAU-1)<=.05),'mean_stationarity':mean_drift<=.05,'stress_stationarity':bool(np.all(stress_drift<=.15)),
        'uniform_mean_parity':control_mean_error<=.05,'uniform_stress_parity':bool(np.all(control_stress_error<=.10)),'fewer_active_cells':active<nx*h*nz}
    report.update(passed=all(gates.values()),gates=gates,y_plus=yp.tolist(),U_plus=up.tolist(),reynolds_plus=rp.tolist(),u_tau=float(utau),first_cell_y_plus=float(yp[0]),
        log_mean_error=log_error,mean_L2_error=mean_error,stress_peak_RMS=stress_error.tolist(),mean_drift=mean_drift,stress_drift=stress_drift.tolist(),
        uniform_mean_L2_error=control_mean_error,uniform_stress_peak_RMS=control_stress_error.tolist(),fine_equivalent_MLUPS=nx*h*nz*(dns.BURN+dns.AVERAGE)/report['seconds']/1e6)
    save();print(json.dumps(gates),flush=True);return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

"""Finite-horizon coarsening sensitivity at a measured uniform DNS endpoint.

This diagnostic cannot certify a refinement map until its uniform reference
passes the DNS gate. Perturbations are balanced within each 2x2x2 parent.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import warp as wp
import measure_lbm3d_channel_dns as dns
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
from kernel_engine.lbm.lbm3d_channel_statistics import accumulate
from kernel_engine.lbm.lbm3d_refinement import restrict_child_mass,split_parent_mass
from kernel_engine.allocation.d_poxel_waterfilling_unification import unified_alloc,err2

ROOT=Path('reports/lbm3d_channel_sensitivity_v1')


def probe_layout(height):
    if height < 32 or height % 32:
        raise ValueError('height must be a positive multiple of32')
    return height//16, 4*height, height//4


def observer(sim,mask,dns_report):
    band_width,steps,sample=probe_layout(dns.H)
    ROOT.mkdir(parents=True,exist_ok=True)
    base=sim.numpy();rho,u=lb.fields(base);u[0]+=.5*sim.force_density/rho
    fluctuations=u[:,:,1:-1,:]-u[:,:,1:-1,:].mean(axis=(1,3),keepdims=True)
    spectrum=np.abs(np.fft.rfftn(fluctuations,axes=(1,3)))**2
    # Explicit Fourier fractions per velocity component, including exact
    # lattice-alternating modes; high-k energy is not removed from the data.
    kx=np.abs(np.fft.fftfreq(dns.NX));kz=np.fft.rfftfreq(dns.NZ)
    weights=np.ones(kz.size);weights[1:-1]=2
    spectrum*=weights[None,None,None,:]
    high=(kx[:,None]>=.25)|(kz[None,:]>=.25)
    alternating=(kx[:,None]==.5)|(kz[None,:]==.5)
    sums=spectrum.sum(axis=(1,2,3))
    diagnostic={'high_wavenumber_energy_fraction':(np.sum(spectrum*high[None,:,None,:],axis=(1,2,3))/sums).tolist(),
                'exact_alternating_energy_fraction':(np.sum(spectrum*alternating[None,:,None,:],axis=(1,2,3))/sums).tolist(),
                'wall_normal_plane_mean_rms':float(np.sqrt(np.mean(u[1,:,1:-1,:].mean(axis=(0,2))**2)))}
    del rho,u,fluctuations,spectrum
    full=base[:,:,1:-1,:]
    defect=split_parent_mass(restrict_child_mass(full))-full
    target_utau=dns.UTAU
    ref_u=np.asarray(dns_report['reference_U_plus']);ref_r=np.asarray(dns_report['reference_reynolds_plus'])[:4]
    scaling=np.r_[np.maximum(.05*ref_u,.1),np.repeat(.2*np.max(np.abs(ref_r),axis=1),dns.H//2)]
    report={'scope':'finite-horizon coarsening diagnostic; map not certified','recursive':sim.recursive,'bulk_tau':sim.bulk_tau,
            'uniform_DNS_passed':dns_report['passed'],'base_state_sha256':hashlib.sha256(base.tobytes()).hexdigest(),
            'diagnostic':diagnostic,'steps_per_probe':steps,'sample_every':sample,'average_last_steps':steps//2,
            'epsilon':[.125,.0625],'bands':8,'linearity_relative_L2_limit':.05,'probes':[]}
    def save(): (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    def observe(q):
        s=type(sim)(q,force_density=sim.force_density,tau=dns.TAU,solid=mask,device='cuda:0',recursive=sim.recursive,bulk_tau=sim.bulk_tau)
        initial=lb.ledger(q);stats=wp.zeros((10,dns.H+2),dtype=wp.int64,device='cuda:0')
        for step in range(sample,steps+1,sample):
            s.step(sample)
            if step>steps//2:wp.launch(accumulate,s.shape,[s.a,s.args[0],s.args[1],wp.float64(2**40),wp.int64(s.force_units),stats,s.failure],device='cuda:0')
        out=s.numpy();ledger=lb.ledger(out);wall=s.wall_impulse_numpy()
        balance=[ledger[k+1]-initial[k+1]+int(wall[k])-(steps*dns.NX*dns.H*dns.NZ*s.force_units if k==0 else 0) for k in range(3)]
        valid=ledger[0]==initial[0] and balance==[0,0,0] and int(s.failure.numpy()[0])==0
        mean,stress,_=dns.folded(stats.numpy(),8*dns.NX*dns.NZ)
        return np.r_[mean/target_utau,(stress[:4]/target_utau**2).ravel()]/scaling,valid
    derivatives=[];start=time.perf_counter()
    for band in range(8):
        responses=[];all_valid=True
        for epsilon in [.125,.0625]:
            delta=np.zeros_like(base);d=np.rint(defect*epsilon).astype(np.int64)
            # Remove each rounded parent's residual, keeping all 19 population
            # totals exact independently; band limits align to parent pairs.
            residual=restrict_child_mass(d);d[:,::2,::2,::2]-=residual
            lo=band*band_width;hi=lo+band_width
            delta[:,:,lo+1:hi+1,:]=d[:,:,lo:hi,:]
            delta[:,:,dns.H-hi+1:dns.H-lo+1,:]=d[:,:,dns.H-hi:dns.H-lo,:]
            positive,valid_p=observe(base+delta);negative,valid_n=observe(base-delta)
            responses.append((positive-negative)/(2*epsilon));all_valid=all_valid and valid_p and valid_n
        relative=float(np.linalg.norm(responses[0]-responses[1])/max(np.linalg.norm(responses[1]),1e-30))
        derivatives.append(responses[1]);report['probes'].append({'band':band,'distance_from_wall_cells':[lo,hi],
            'linearity_relative_L2':relative,'ledgers_and_flags_pass':all_valid,'weighted_sensitivity_L2':float(np.linalg.norm(responses[1]))})
        save();print(json.dumps(report['probes'][-1]),flush=True)
    sensitivities=np.linalg.norm(derivatives,axis=1)
    # Reuse the existing continuous mesh allocation law. Its error model is
    # conditional on this measured local derivative, not a PDE certificate.
    budget=float(np.sum(sensitivities**2)/4)
    levels,waterlevel=unified_alloc(sensitivities,1,max(budget,1e-30))
    report.update(weighted_jacobian=np.asarray(derivatives).T.tolist(),continuous_levels=levels.tolist(),
        waterlevel=waterlevel,model_error_budget_squared=budget,model_error_squared=err2(levels,sensitivities),
        sensitivity_order=np.argsort(-sensitivities,kind='stable').tolist(),seconds=time.perf_counter()-start,
        candidate_wall_layers=2*band_width,candidate_active_cell_fraction=.25+.75/8,
        candidate_map_note='Two contiguous wall blocks spanning the first two bands are a candidate only; ranking may reject this geometry.',
        gates={'all_probe_ledgers_and_flags':all(p['ledgers_and_flags_pass'] for p in report['probes']),
               'finite_difference_linearity':all(p['linearity_relative_L2']<=.05 for p in report['probes'])},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('scripts/measure_lbm3d_channel_sensitivity.py'),Path('scripts/measure_lbm3d_channel_dns.py')]})
    save()

if __name__=='__main__':
    dns.main(final_observer=observer)

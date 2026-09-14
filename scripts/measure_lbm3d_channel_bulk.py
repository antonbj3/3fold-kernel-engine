"""One independent-bulk-relaxation candidate, unchanged uniform DNS gates."""
from pathlib import Path
import json
import numpy as np
import measure_lbm3d_channel_dns as dns
from kernel_engine.lbm import lbm3d_mrt_les as lb


def diagnose(sim,mask,report):
    rho,u=lb.fields(sim.numpy());u[0]+=.5*sim.force_density/rho
    v=u[:,:,1:-1,:]-u[:,:,1:-1,:].mean(axis=(1,3),keepdims=True)
    energy=np.abs(np.fft.rfftn(v,axes=(1,3)))**2
    kx=np.abs(np.fft.fftfreq(dns.NX));kz=np.fft.rfftfreq(dns.NZ)
    weights=np.ones(kz.size);weights[1:-1]=2;energy*=weights[None,None,None,:]
    high=(kx[:,None]>=.25)|(kz[None,:]>=.25)
    alternating=(kx[:,None]==.5)|(kz[None,:]==.5)
    total=energy.sum(axis=(1,2,3))
    report['endpoint_spectrum']={'high_wavenumber_energy_fraction':(np.sum(energy*high[None,:,None,:],axis=(1,2,3))/total).tolist(),
        'exact_alternating_energy_fraction':(np.sum(energy*alternating[None,:,None,:],axis=(1,2,3))/total).tolist()}
    (dns.ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    dns.ROOT=Path('reports/lbm3d_channel_dns_bulk_v1')
    raise SystemExit(dns.main(final_observer=diagnose,bulk_tau=1.0))

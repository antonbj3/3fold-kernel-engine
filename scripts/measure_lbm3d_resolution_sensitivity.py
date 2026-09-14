"""Reproduce all accepted resolution checkpoints before probing the endpoint."""
import hashlib,json
from pathlib import Path
import measure_lbm3d_channel_dns as dns
import measure_lbm3d_channel_sensitivity as sensitivity
from measure_lbm3d_recursive_resolution import configure
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation


def verify_replay(control,report):
    keys=('grid','tau','bulk_tau','Cs','bits','burn_steps','averaging_steps',
          'block_steps','sample_every','initial_filter_cells','source_sha256')
    return (control.get('passed') is True and report.get('passed') is True
            and all(control[k]==report[k] for k in keys)
            and len(control['blocks'])==len(report['blocks'])
            and all(a['step']==b['step'] and a['state_sha256']==b['state_sha256']
                    and a.get('plane_integer_statistics')==b.get('plane_integer_statistics')
                    for a,b in zip(control['blocks'],report['blocks'])))


def main():
    configure()
    reference=Path('data/lbm3d_resolution_control.json')
    control=json.loads(reference.read_text())
    if not control.get('passed'):raise ValueError('passed resolution control required')
    dns.ROOT=Path('reports/lbm3d_resolution_replay_v1')
    sensitivity.ROOT=Path('reports/lbm3d_resolution_sensitivity_v1')
    def finish(sim,mask,report):
        exact=verify_replay(control,report)
        report['accepted_control_replay_exact']=exact
        report['control_report_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
        (dns.ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not exact:raise RuntimeError('accepted control replay mismatch; probes not admitted')
        sensitivity.observer(sim,mask,report)
        result=json.loads((sensitivity.ROOT/'report.json').read_text())
        if not all(result['gates'].values()):raise RuntimeError('sensitivity gate failed')
    return dns.main(final_observer=finish,bulk_tau=1.,recursive=True,
                    simulation_class=SpecializedChannelSimulation)


if __name__=='__main__':raise SystemExit(main())

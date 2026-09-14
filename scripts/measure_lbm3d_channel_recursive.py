"""Recursive third-order candidate with the original uniform DNS thresholds."""
from pathlib import Path
import measure_lbm3d_channel_dns as dns
from measure_lbm3d_channel_bulk import diagnose
from measure_lbm3d_channel_sensitivity import observer


def finalize(sim,mask,report):
    diagnose(sim,mask,report)
    if report["passed"]:observer(sim,mask,report)

if __name__=='__main__':
    dns.ROOT=Path('reports/lbm3d_channel_dns_recursive_v1')
    raise SystemExit(dns.main(final_observer=finalize,bulk_tau=1.,recursive=True))

"""Check whether recursive reconstruction removes the need for separate bulk tau."""
from pathlib import Path
from measure_lbm3d_recursive_gates import main as gates
import measure_lbm3d_channel_dns as dns
from measure_lbm3d_channel_recursive import finalize

if __name__=='__main__':
    status=gates(bulk_tau=None,output='reports/lbm3d_recursive_shared_bulk_gates_v1/report.json')
    if status:raise SystemExit(status)
    dns.ROOT=Path('reports/lbm3d_channel_dns_recursive_shared_bulk_v1')
    raise SystemExit(dns.main(final_observer=finalize,bulk_tau=None,recursive=True))

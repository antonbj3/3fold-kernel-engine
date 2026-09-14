"""Recursive third-order candidate with the original uniform DNS thresholds."""
from pathlib import Path
import measure_lbm3d_channel_dns as dns
from measure_lbm3d_channel_bulk import diagnose

if __name__=='__main__':
    dns.ROOT=Path('reports/lbm3d_channel_dns_recursive_v1')
    raise SystemExit(dns.main(final_observer=diagnose,bulk_tau=1.,recursive=True))

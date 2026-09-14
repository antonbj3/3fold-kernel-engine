"""H100 bit-exact specialization gate before the recursive bulk-one DNS case."""
from pathlib import Path
from measure_lbm3d_specialized import main as specialize
from measure_lbm3d_recursive_gates import main as precision
from measure_lbm3d_channel_recursive import finalize
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation
import measure_lbm3d_channel_dns as dns

if __name__=='__main__':
    if specialize():raise SystemExit(1)
    if precision():raise SystemExit(1)
    dns.ROOT=Path('reports/lbm3d_channel_dns_recursive_v1')
    raise SystemExit(dns.main(final_observer=finalize,bulk_tau=1.,recursive=True,simulation_class=SpecializedChannelSimulation))

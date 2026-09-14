"""A priori 1.5x spatial/acoustic refinement of the recursive channel control.

Reynolds number, domain proportions, outer-time windows and all DNS thresholds
are unchanged. The random initialization has the same seed and physical filter
width, but is a different realization on the larger grid, not state interpolation.
"""
from pathlib import Path
import measure_lbm3d_channel_dns as dns
from measure_lbm3d_channel_bulk import diagnose
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation


def configure():
    dns.NX,dns.H,dns.NZ=288,96,144
    dns.BURN,dns.AVERAGE,dns.BLOCK,dns.SAMPLE=120000,240000,30000,300
    dns.NU=dns.UTAU*(dns.H/2)/180
    dns.TAU=.5+3*dns.NU
    dns.INITIAL_FILTER_CELLS=3.
    dns.ROOT=Path('reports/lbm3d_recursive_resolution_v1')


if __name__=='__main__':
    configure()
    raise SystemExit(dns.main(final_observer=diagnose,bulk_tau=1.,recursive=True,
                             simulation_class=SpecializedChannelSimulation))

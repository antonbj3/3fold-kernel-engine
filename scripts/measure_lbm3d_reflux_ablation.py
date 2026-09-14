"""Nonconservative diagnostic only: omit reflux to isolate interface growth.

The launch override is confined to this standalone, single-threaded observer
and is restored before exit. This is not a solver option or an admissible DNS.
"""
import warp as wp
from kernel_engine.lbm.lbm3d_refinement_gpu import apply_reflux
from measure_lbm3d_interface_startup import main


if __name__=='__main__':
    launch=wp.launch
    def diagnostic_launch(kernel,*args,**kwargs):
        if kernel is apply_reflux:return
        return launch(kernel,*args,**kwargs)
    try:
        wp.launch=diagnostic_launch
        main(skip_uniform=True,output='reports/lbm3d_reflux_ablation_v1')
    finally:
        wp.launch=launch

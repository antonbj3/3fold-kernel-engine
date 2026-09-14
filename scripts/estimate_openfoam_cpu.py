"""Traceable, conditional CPU cost model; not an OpenFOAM benchmark.

The only timing anchor is a published 2017 tutorial run. Hardware uplift,
required target mesh and strong-scaling efficiency are assumptions. The
scenario spread is not a statistical confidence interval or accuracy proof.
"""
import json,math
from pathlib import Path


def main():
    anchor={'cpu':'Intel Core i7-2600 3.40GHz','ranks':4,'cells':60000,'wall_seconds':777.06,'physical_end_time':1000,'delta_t':.2,'steps':5000,
            'timing_url':'https://www.xsim.info/articles/OpenFOAM/en-US/tutorials/incompressible-pimpleFoam-channel395.html',
            'control_url':'https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-4.x/master/tutorials/incompressible/pimpleFoam/channel395/system/controlDict',
            'mesh_url':'https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-4.x/master/tutorials/incompressible/pimpleFoam/channel395/system/blockMeshDict',
            'solution_url':'https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-4.x/master/tutorials/incompressible/pimpleFoam/channel395/system/fvSolution',
            'limitations':'Different Reynolds number, 2017 hardware/software, single reported timing, no raw solver log; mapping to5000steps assumes the documented stock tutorial settings. Timing may include output/other overhead.'}
    k=anchor['wall_seconds']*anchor['ranks']/(anchor['cells']*anchor['steps'])
    assumptions={'outer_times':30,'domain_x_over_h':6,'representative_advecting_velocity_over_u_tau':20,'CFL':.5,
                 'modern_per_core_speedup_over_anchor':3,'hardware_speedup_sensitivity':[2,4],
                 'parallel_efficiency_at_target':.7,'efficiency_sensitivity':[.5,.85],
                 'target_seconds':660,'CFL_and_solver_source':'https://journal.openfoam.com/index.php/ofj/article/download/159/186/4425',
                 'target_meshes_are_not_accuracy_validated':True,
                 'hardware_note':'A modern EPYC-class CPU is intended, but 2-4x is an assumed sensitivity range, not a measured architectural conversion or GHz ratio.',
                 'timestep_note':'Streamwise advective CFL estimate; ignores transient velocity peaks and other-direction flux contributions. No claim all grids meet accuracy at this timestep.',
                 'scaling_note':'Assumed efficiency can fail at high rank counts; calculated ranks do not establish that the deadline is attainable.'}
    rows=[]
    for label,shape in [('coarse_screen_only',(64,64,64)),('near_DNS_wall_parallel_spacing',(128,96,96)),('LBM_uniform_cell_count_control',(288,96,144))]:
        cells=math.prod(shape);steps=math.ceil(30*20*shape[0]/(6*.5));old_work=k*cells*steps
        central_work=old_work/3
        rows.append({'label':label,'shape':shape,'cells':cells,'estimated_OpenFOAM_steps':steps,
                     'dx_plus':6*180/shape[0],'dz_plus':3*180/shape[2],
                     'historical_equivalent_core_hours':old_work/3600,
                     'assumed_modern_core_hours':central_work/3600,
                     'idealized_32core_minutes':central_work/32/60,
                     'conditional_cores_for11min':central_work/(660*.7),
                     'cores_sensitivity_hardware_and_efficiency_only':[old_work/(4*660*.85),old_work/(2*660*.5)]})
    result={'scope':__doc__,'timing_anchor':anchor,'historical_core_seconds_per_cell_step':k,'assumptions':assumptions,'scenarios':rows,
            'conclusion':'Useful for selecting benchmark resources, not a speedup claim. The target mesh that meets the requested error bars is the dominant unresolved choice; CPU pilot or independent matching validated logs still required.',
            'published_mesh_guidance':'Mukha/Parsani2026 discuss DNS wall-parallel dx+~10,dz+~5 and LES~25,10. This motivates scenarios only, not acceptance under our exact profile/stationarity limits.'}
    p=Path('reports/openfoam_cpu_estimate_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()

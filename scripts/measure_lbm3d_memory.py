"""Resident array inventory and device allocation delta for the refined case.

Device free-memory deltas include allocator/kernel overhead and can include
background activity. They are sampled, not a lifetime peak measurement.
"""
import argparse,json
from pathlib import Path
import warp as wp
from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation


def main(wall_cells):
    device=wp.get_device('cuda:0');before=device.free_memory
    sim=RefinedChannelGPU(nx=288,height=96,nz=144,wall_cells=wall_cells,cs_fine=.1,recursive=True,bulk_tau_fine=1.,conserved_reflux=True,balanced_reflux=True,tau_fine=.5032,force_fine=1e-8,channel_factory=SpecializedChannelSimulation)
    wp.synchronize_device(device);after_setup=device.free_memory
    sim.step_many(2);wp.synchronize_device(device);after_steps=device.free_memory
    arrays={}
    def collect(value,name):
        if isinstance(value,wp.array) and value.device.is_cuda:
            arrays.setdefault(value.ptr,{'name':name,'bytes':value.capacity,'shape':list(value.shape),'dtype':str(value.dtype)})
        elif isinstance(value,(list,tuple)):
            for i,item in enumerate(value):collect(item,f'{name}[{i}]')
    for label,obj in [('refined',sim),('fine0',sim.fine[0]),('fine1',sim.fine[1]),('coarse',sim.coarse)]:
        for key,value in vars(obj).items():collect(value,f'{label}.{key}')
    result={'scope':__doc__,'device':device.name,'wall_layers':wall_cells,'total_device_bytes':device.total_memory,
            'resident_array_bytes':sum(a['bytes'] for a in arrays.values()),'device_used_increase_after_setup':before-after_setup,'device_used_increase_after_four_fine_steps':before-after_steps,
            'arrays':sorted(arrays.values(),key=lambda a:-a['bytes'])}
    p=Path(f'reports/lbm3d_memory_wall{wall_cells}_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='arrays'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wall-cells',type=int,default=18)
    main(parser.parse_args().wall_cells)

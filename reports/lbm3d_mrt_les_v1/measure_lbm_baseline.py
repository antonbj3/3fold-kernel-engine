import json,time,hashlib
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm import lbm3d_gpu as b
n=64; steps=200; shape=(19,n,n,n)
t=time.perf_counter()
a=wp.array(np.broadcast_to(b.WI[:,None,None,None],shape).copy(),device='cuda:0'); z=wp.empty_like(a)
args=[wp.array(b.EI[:,j],dtype=wp.int32,device='cuda:0') for j in range(3)]
w=wp.array(b.WI,device='cuda:0'); op=wp.array(b.OPPI,device='cuda:0'); wp.synchronize()
setup=time.perf_counter()-t
def step(a,z):
    wp.launch(b.collide_stream,(n,n,n),[a,z,*args,w,op,1.0,0.0,0.5,n,n,n],device='cuda:0')
t=time.perf_counter();step(a,z);wp.synchronize();build=time.perf_counter()-t
runs=[]
for _ in range(2):
    t=time.perf_counter()
    for i in range(steps): step(a,z);a,z=z,a
    wp.synchronize();runs.append(time.perf_counter()-t)
t=time.perf_counter(); f=a.numpy(); read=time.perf_counter()-t
copy=[]
for _ in range(2):
    t=time.perf_counter()
    for i in range(steps): wp.copy(z,a)
    wp.synchronize();copy.append(time.perf_counter()-t)
r=dict(shape=shape,steps=steps,setup_s=setup,first_launch_build_s=build,step_s=runs,readback_s=read,mlups=[n**3*steps/x/1e6 for x in runs],logical_bandwidth_GBs=[2*f.nbytes*steps/x/1e9 for x in runs],copy_GBs=[2*f.nbytes*steps/x/1e9 for x in copy],finite=bool(np.isfinite(f).all()),source_sha256=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest())
Path('reports').mkdir(exist_ok=True);Path('reports/lbm_baseline_profile.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

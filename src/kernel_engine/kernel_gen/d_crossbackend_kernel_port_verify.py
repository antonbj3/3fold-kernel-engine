import numpy as np, wgpu, torch
N=1_000_000
x=np.linspace(0.5,2.0,N,dtype=np.float32); c0,c1=0.6,0.2
WGSL=f"""
@group(0) @binding(0) var<storage,read> inp: array<f32>;
@group(0) @binding(1) var<storage,read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let i=g.x; let n=arrayLength(&inp); if(i>=n){{return;}}
  let xc=inp[i]; let xm=inp[select(i-1u,0u,i==0u)]; let xp=inp[select(i+1u,n-1u,i==n-1u)];
  var s={c0}*xc; s=fma({c1},xm,s); s=fma({c1},xp,s); outp[i]=s;
}}
"""
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
ib=dev.create_buffer_with_data(data=x,usage=wgpu.BufferUsage.STORAGE)
ob=dev.create_buffer(size=x.nbytes,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
pipe=dev.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=WGSL),"entry_point":"main"})
bg=dev.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[{"binding":0,"resource":{"buffer":ib,"offset":0,"size":x.nbytes}},{"binding":1,"resource":{"buffer":ob,"offset":0,"size":x.nbytes}}])
enc=dev.create_command_encoder(); cp=enc.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0,bg); cp.dispatch_workgroups((N+63)//64); cp.end(); dev.queue.submit([enc.finish()])
vk=np.frombuffer(dev.queue.read_buffer(ob),dtype=np.float32).copy()
xt=torch.from_numpy(x).cuda(); xm=torch.cat([xt[:1],xt[:-1]]); xp=torch.cat([xt[1:],xt[-1:]])
s=(np.float32(c0))*xt; s=torch.addcmul(s,torch.full_like(xt,c1),xm); s=torch.addcmul(s,torch.full_like(xt,c1),xp)
cu=s.cpu().numpy()
be=float((vk==cu).mean()); md=float(np.abs(vk.astype(np.float64)-cu.astype(np.float64)).max())
print(f"★REAL VERIFIED KERNEL PORT — 3-point fma-stencil, CUDA vs Vulkan (same RTX5070):")
print(f"   bit-exact CUDA==Vulkan = {be:.4f} | max abs delta = {md:.2e}")
print("   => a MULTI-OP real kernel ports BIT-IDENTICAL cross-backend when composed of explicit-fma (the portable recipe)." if be>0.999 else "   => still diverges: interior bit-exact matters (boundary select may differ).")
# interior-only check (exclude boundaries)
bei=float((vk[1:-1]==cu[1:-1]).mean())
print(f"   interior (excl boundaries) bit-exact = {bei:.4f}")

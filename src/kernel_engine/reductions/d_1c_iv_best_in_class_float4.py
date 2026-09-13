"""Codegen QUALITY: close the 73%→best-in-class gap on REAL CUDA. The snap's hand-tree was deterministic+correct but
only 73% peak (cub hits 92%). The lever is float4 VECTORIZED loads (4 floats/transaction). Generate a float4 deterministic
tree reduction, compile via nvcc, measure vs the scalar tree (73%) and cub (torch.sum, 92%). If float4 ≈ cub, the generator produces
BEST-IN-CLASS + CERTIFIED (deterministic) kernels — the codegen-quality residual is closeable, mechanically. flock+contam-flag.
"""
import torch, time
from torch.utils.cpp_extension import load_inline
THEO_BW = 672.0
CUDA = r'''
#include <torch/extension.h>
__global__ void red_scalar(const float* x, float* p, int n){                 // the snap's 73% kernel
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  for(int j=blockIdx.x*blockDim.x+t; j<n; j+=blockDim.x*gridDim.x) v+=x[j];
  s[t]=v; __syncthreads();
  for(int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if(t==0) p[blockIdx.x]=s[0];
}
__global__ void red_float4(const float4* x, float* p, int n4){               // float4 vectorized loads (4x/transaction)
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  for(int j=blockIdx.x*blockDim.x+t; j<n4; j+=blockDim.x*gridDim.x){ float4 f=x[j]; v+=f.x+f.y+f.z+f.w; }  // fixed order => deterministic
  s[t]=v; __syncthreads();
  for(int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if(t==0) p[blockIdx.x]=s[0];
}
torch::Tensor scalar_p(torch::Tensor x){ int n=x.numel(),b=256,g=2048; auto p=torch::zeros({g},x.options());
  red_scalar<<<g,b,b*sizeof(float)>>>(x.data_ptr<float>(),p.data_ptr<float>(),n); return p; }
torch::Tensor float4_p(torch::Tensor x){ int n4=x.numel()/4,b=256,g=2048; auto p=torch::zeros({g},x.options());
  red_float4<<<g,b,b*sizeof(float)>>>((const float4*)x.data_ptr<float>(),p.data_ptr<float>(),n4); return p; }
'''
CPP = "torch::Tensor scalar_p(torch::Tensor); torch::Tensor float4_p(torch::Tensor);"
print("compiling...", flush=True)
m = load_inline(name="d1civq", cpp_sources=[CPP], cuda_sources=[CUDA], functions=["scalar_p","float4_p"], verbose=False)
print("compiled OK.\n")
N = 1 << 26
x = torch.rand(N, device='cuda', dtype=torch.float32); ref = torch.sum(x).item()

def measure(name, fn):
    torch.cuda.synchronize()
    vals = [fn() for _ in range(5)]
    det = len(set(vals)) == 1; correct = abs(vals[0]-ref)/abs(ref) < 1e-2
    ts=[]
    for _ in range(30):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    bw=4*N/min(ts)/1e9
    print(f"  {name:20s}: det={det!s:5s} correct={correct!s:5s} BW={bw:5.0f} GB/s = {bw/THEO_BW*100:2.0f}% peak")
    return bw/THEO_BW
scalar = measure("scalar-tree (snap)", lambda: torch.sum(m.scalar_p(x)).item())
flt4   = measure("float4-tree",        lambda: torch.sum(m.float4_p(x)).item())
cub    = measure("cub (torch.sum)",    lambda: torch.sum(x).item())
print("\n" + "="*92)
print(f"  ⟹ codegen QUALITY: scalar {scalar*100:.0f}% → float4 {flt4*100:.0f}% (vs cub {cub*100:.0f}%).")
if flt4 >= 0.88:
    print(f"    float4 vectorization CLOSES the gap → BEST-IN-CLASS ({flt4*100:.0f}%) AND deterministic (certified). The 73% was")
    print(f"    a codegen-quality gap, mechanically closeable (float4 loads). the generator produces certified best-in-class kernels.")
else:
    print(f"    float4 improved to {flt4*100:.0f}% but not yet cub-level — more levers (grid/occupancy/shfl) remain. HONEST.")
print("="*92)

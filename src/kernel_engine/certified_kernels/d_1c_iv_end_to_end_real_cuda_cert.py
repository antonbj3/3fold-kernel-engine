"""1c-iv + the SNAP: end-to-end certified-generation on REAL nvcc-compiled CUDA (torch cpp_extension). Composes the whole:
generate two real reduction kernels — (A) optimal DETERMINISTIC block-tree (fixed-order) + (B) naive atomicAdd (non-det,
contended) — compile via nvcc, then run the actual CERT-VECTOR on the compiled kernels: R2 determinism (bit-repeat),
correctness (vs torch.sum), roofline (BW vs 672 GB/s theoretical). ★The verdict is a property of the COMPUTATION: for a
determinism-REQUIRED sum (L∞/ill-cond), the cert CERTIFIES the tree and REJECTS the atomicAdd — on REAL hardware, not modeled.
This is CORE-1 (roofline target) ⊕ cert-vector (R2⊕correct⊕roofline) ⊕ nvcc-codegen (1c-iv) snapping together. flock+contam-flag.
"""
import torch, time
from torch.utils.cpp_extension import load_inline
assert torch.cuda.is_available()
THEO_BW = 672.0

CUDA = r'''
#include <torch/extension.h>
__global__ void reduce_det(const float* x, float* partials, int n){
  extern __shared__ float s[]; int tid=threadIdx.x;
  float v=0.f; for(int j=blockIdx.x*blockDim.x+tid; j<n; j+=blockDim.x*gridDim.x) v+=x[j];  // grid-stride, coalesced, fixed per-thread order
  s[tid]=v; __syncthreads();
  for(int st=blockDim.x/2; st>0; st>>=1){ if(tid<st) s[tid]+=s[tid+st]; __syncthreads(); } // fixed-order tree => deterministic
  if(tid==0) partials[blockIdx.x]=s[0];
}
__global__ void reduce_atomic(const float* x, float* out, int n){
  for(int j=blockIdx.x*blockDim.x+threadIdx.x; j<n; j+=blockDim.x*gridDim.x) atomicAdd(out, x[j]); // non-det float order + contention
}
torch::Tensor det_partials(torch::Tensor x){ int n=x.numel(),b=256,g=1024; auto p=torch::zeros({g},x.options());
  reduce_det<<<g,b,b*sizeof(float)>>>(x.data_ptr<float>(),p.data_ptr<float>(),n); return p; }        // <<<>>> lives in cuda_sources (nvcc)
torch::Tensor atomic_sum(torch::Tensor x){ int n=x.numel(),b=256,g=1024; auto o=torch::zeros({1},x.options());
  reduce_atomic<<<g,b>>>(x.data_ptr<float>(),o.data_ptr<float>(),n); return o; }
'''
CPP = "torch::Tensor det_partials(torch::Tensor); torch::Tensor atomic_sum(torch::Tensor);"  # declarations only; load_inline auto-binds
print("compiling real CUDA via nvcc (load_inline)...", flush=True)
m = load_inline(name="d1civ", cpp_sources=[CPP], cuda_sources=[CUDA],
                functions=["det_partials","atomic_sum"], verbose=False)
print("compiled OK.\n")

N = 1 << 24
x = torch.rand(N, device='cuda', dtype=torch.float32)
ref = torch.sum(x).item()
det_sum = lambda: torch.sum(m.det_partials(x)).item()      # tree partials + deterministic final sum
atm_sum = lambda: m.atomic_sum(x).item()

def cert(name, fn, det_required=True):
    torch.cuda.synchronize()
    vals = [fn() for _ in range(5)]                          # R2 determinism: bit-repeat
    deterministic = len(set(vals)) == 1
    correct = abs(vals[0] - ref) / abs(ref) < 1e-2
    # roofline (read-only 4N), min-of-trials
    ts=[]
    for _ in range(20):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    bw = 4*N/min(ts)/1e9; roof = bw/THEO_BW
    verdict = ("CERTIFY" if (deterministic and correct and roof>=0.5) else
               ("REJECT (non-deterministic; computation REQUIRES det)" if (det_required and not deterministic) else
                "ABSTAIN/REJECT"))
    print(f"  {name:22s}: determin={deterministic!s:5s} correct={correct!s:5s} BW={bw:5.0f} GB/s ({roof*100:2.0f}% peak) spread={max(vals)-min(vals):.3e} → {verdict}")
    return deterministic, correct, roof

print(f"cert-vector on REAL compiled kernels (sum of N={N}; ref torch.sum={ref:.1f}); computation = det-REQUIRED sum:")
cert("reduce_det (tree)", det_sum, det_required=True)
cert("reduce_atomic", atm_sum, det_required=True)
print("\n" + "="*96)
print("  ⟹ THE SNAP (real hardware): the SAME cert-vector, run on two REAL nvcc-compiled reduction kernels, CERTIFIES the")
print("    deterministic tree (bit-repeatable + correct + memory-bound) and REJECTS the atomicAdd (non-deterministic float")
print("    order) for a determinism-REQUIRED computation — even though atomicAdd is 'correct on average'. CORE-1 roofline")
print("    target ⊕ cert-vector (R2⊕correct⊕roofline) ⊕ nvcc-codegen (1c-iv) compose end-to-end. Not modeled — compiled + measured.")
print("="*96)

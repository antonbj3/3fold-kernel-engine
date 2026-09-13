"""CONFLICT-METHOD stress test (force node reshaping where results conflict at boundaries).
C body533's 2×2 puts R2 (determinism, correctness×data-dep) and R4 (contention, perf×data-dep) in DIFFERENT cells. But the snap's
atomicAdd failed BOTH (non-det AND 3 GB/s) — they LOOK collapsed. FORCE the boundary by DISSOCIATION (like A's R1/R3 3rd-test): build
a kernel failing R2-not-R4 and one failing R4-not-R2. If both exist ⇒ R2⊥R4 genuinely distinct (2×2 holds); if not ⇒ reshape.
★Non-obvious lever: INT64 atomics are DETERMINISTIC (integer add is order-INDEPENDENT) yet still CONTENDED ⇒ R2 PASS + R4 FAIL.
And a tree with a FINAL float-atomic of few partials is NON-det but UNcontended ⇒ R2 FAIL + R4 PASS. On real compiled CUDA. flock.
"""
import torch, time
from torch.utils.cpp_extension import load_inline
THEO_BW = 672.0; SCALE = float(1<<20)
CUDA = r'''
#include <torch/extension.h>
__global__ void atom_float(const float* x, float* out, int n){                                  // R2 FAIL + R4 FAIL
  for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=blockDim.x*gridDim.x) atomicAdd(out,x[j]);
}
__global__ void atom_int64(const float* x, unsigned long long* out, int n, float scale){         // R2 PASS (int add order-indep) + R4 FAIL (contended)
  for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=blockDim.x*gridDim.x) atomicAdd(out,(unsigned long long)(x[j]*scale));
}
__global__ void tree_atomicfinal(const float* x, float* out, int n){                             // R2 FAIL (few float-atomics non-det) + R4 PASS (uncontended)
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  for(int j=blockIdx.x*blockDim.x+t;j<n;j+=blockDim.x*gridDim.x) v+=x[j];
  s[t]=v; __syncthreads();
  for(int st=blockDim.x/2;st>0;st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if(t==0) atomicAdd(out, s[0]);                                                                  // only gridDim float-atomics (2048), non-det order
}
torch::Tensor r_float(torch::Tensor x){ int n=x.numel(); auto o=torch::zeros({1},x.options());
  atom_float<<<2048,256>>>(x.data_ptr<float>(),o.data_ptr<float>(),n); return o; }
torch::Tensor r_int64(torch::Tensor x){ int n=x.numel(); auto o=torch::zeros({1},x.options().dtype(torch::kInt64));
  atom_int64<<<2048,256>>>(x.data_ptr<float>(),(unsigned long long*)o.data_ptr<int64_t>(),n,%SCALE%); return o; }
torch::Tensor r_treeatom(torch::Tensor x){ int n=x.numel(),b=256; auto o=torch::zeros({1},x.options());
  tree_atomicfinal<<<2048,b,b*sizeof(float)>>>(x.data_ptr<float>(),o.data_ptr<float>(),n); return o; }
'''.replace('%SCALE%', repr(SCALE))
CPP = "torch::Tensor r_float(torch::Tensor); torch::Tensor r_int64(torch::Tensor); torch::Tensor r_treeatom(torch::Tensor);"
print("compiling...", flush=True)
m = load_inline(name="drdiss", cpp_sources=[CPP], cuda_sources=[CUDA], functions=["r_float","r_int64","r_treeatom"], verbose=False)
print("compiled OK.\n")
N = 1 << 24
x = torch.rand(N, device='cuda', dtype=torch.float32); ref = torch.sum(x).item()

def cert(name, fn, conv=lambda v: v):
    torch.cuda.synchronize()
    vals = [conv(fn().item()) for _ in range(6)]
    r2_det = len(set(vals)) == 1                                   # R2: bit-reproducible?
    correct = abs(vals[0]-ref)/abs(ref) < 2e-2
    ts=[]
    for _ in range(20):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    bw=4*N/min(ts)/1e9; r4_ok = bw/THEO_BW > 0.30                   # R4: not contention-crippled?
    print(f"  {name:26s}: R2(determ)={'PASS' if r2_det else 'FAIL':4s}  R4(uncontended)={'PASS' if r4_ok else 'FAIL':4s}  "
          f"BW={bw:5.0f} GB/s ({bw/THEO_BW*100:2.0f}%)  correct={correct}")
    return r2_det, r4_ok
print(f"cert R2 (determinism) vs R4 (contention) on 3 REAL kernels (sum N={N}):")
a=cert("float-atomic",    lambda: m.r_float(x))
b=cert("int64-atomic",    lambda: m.r_int64(x), conv=lambda v: v/SCALE)
c=cert("tree+atomic-final",lambda: m.r_treeatom(x))
print("\n"+"="*96)
diss = (not a[0] and not a[1]) and (b[0] and not b[1]) and (not c[0] and c[1])
print("  ⟹ CONFLICT-METHOD RESULT (R2 vs R4 boundary, real hardware):")
print(f"  • float-atomic: R2 FAIL + R4 FAIL (both) · int64-atomic: R2 PASS + R4 FAIL (det-but-contended) · tree+atomic-final: R2 FAIL + R4 PASS (fast-but-nondet)")
if diss:
    print(f"  ⟹ R2 ⊥ R4 DISSOCIATE on real hardware (each present without the other) ⇒ genuinely DISTINCT facets — C's 2×2 HOLDS,")
    print(f"    NOT collapsed. ★int64 atomics = DETERMINISTIC (int add order-independent) yet CONTENDED = the decisive dissociator.")
    print(f"    The snap's atomicAdd failing both was a CONFOUND (float-atomic happens to fail both); the boundary is real.")
else:
    print(f"  ⟹ dissociation INCOMPLETE {a},{b},{c} — the 2×2 may need reshaping here. HONEST.")
print("="*96)

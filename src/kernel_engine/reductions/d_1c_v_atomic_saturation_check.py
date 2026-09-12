"""Cheap forced follow-up (OODA Orient step) on d_1c_v run3's surprise: atomic_scalar at N=67M, grid_mult=1 came back
det=True (bit-identical x5) but correct=False -- OPPOSITE of the classic "non-deterministic but correct on average"
story (which the ORIGINAL d_1c_iv_end_to_end_real_cuda_cert.py showed at N=16M: spread=62, non-det, correct). Adversary
check: is det=True at N=67M robust (>5 reps), and is the wrong value consistent with catastrophic cancellation (67M
serial atomicAdds into ONE float32 accumulator -- Higham's naive-summation bound (N-1)*eps*sum(|x|) is >>1 at N=67M,
i.e. naive serial fp32 summation of 67M terms is expected to be badly wrong REGARDLESS of order, which would also
explain determinism-in-practice: once deep in saturation, order-dependent noise is far below the rounding floor).
Reuses the cached compiled module (same name+source as d_1c_v_autonomy_loop_end_to_end_real_cuda.py -> ~0s recompile).
"""
import time
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r'''
#include <torch/extension.h>
__global__ void k_atomic_scalar(const float* __restrict__ x, float* out, int n){
  int stride = blockDim.x*gridDim.x;
  for (int j = blockIdx.x*blockDim.x+threadIdx.x; j<n; j+=stride) atomicAdd(out, x[j]);
}
torch::Tensor atomic_scalar(torch::Tensor x, int64_t block, int64_t grid){
  int n=x.numel(); auto out=torch::zeros({1}, x.options());
  k_atomic_scalar<<<(int)grid,(int)block>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
'''
m = load_inline(name="d1cv_atomic_check", cpp_sources=["torch::Tensor atomic_scalar(torch::Tensor, int64_t, int64_t);"],
                cuda_sources=[CUDA_SRC], functions=["atomic_scalar"], verbose=False)

N = 1 << 26
SM = torch.cuda.get_device_properties(0).multi_processor_count
x = torch.rand(N, device='cuda', dtype=torch.float32)
ref = torch.sum(x).item()
ref_fp64 = x.double().sum().item()
print(f"N={N}  ref(fp32 cub)={ref:.2f}  ref(fp64, ~exact)={ref_fp64:.4f}")

vals = [m.atomic_scalar(x, 256, 1*SM).item() for _ in range(20)]
print(f"atomic_scalar grid_mult=1, 20 reps: {vals}")
print(f"  unique values: {sorted(set(vals))}  all-identical={len(set(vals))==1}")
relerr = abs(vals[0] - ref) / abs(ref)
print(f"  val={vals[0]:.2f}  relerr-vs-fp32-ref={relerr*100:.1f}%")

# float32 ULP at the magnitude where the accumulator would sit partway through accumulation
import math
for mag in [2**23, 2**24, 2**25, 2**26]:
    ulp = math.ulp(float(mag))
    print(f"  fp32 ULP at magnitude {mag:.3e} = {ulp:.3f}  (mean per-element value ~0.5 -> "
          f"{'BELOW rounding floor (lost)' if ulp>1.0 else 'still representable'})")
higham_bound_relerr = (N-1) * 2**-23  # Higham naive-summation worst-case relative error bound, eps_fp32=2^-23
print(f"  Higham naive-summation worst-case relative-error bound: (N-1)*eps = {higham_bound_relerr:.2f} "
      f"(>>1 => naive serial fp32 accumulation of N={N} terms is THEORETICALLY unbounded-relative-error; "
      f"a ~{relerr*100:.0f}% observed error is well within a mechanism this loose, i.e. NOT surprising)")

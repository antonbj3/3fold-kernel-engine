"""Certified kernel loop on real compiled CUDA (nvcc via torch.utils.cpp_extension.load_inline).

Starts from a deliberately weak kernel (naive atomicAdd, scalar loads, poor grid), measures a
certificate vector on the real GPU (bit-repeat determinism, correctness against torch.sum, achieved
bandwidth over the roofline), and picks the next mutation from the largest certificate deficit over the
knob set {mode, width, grid_mult}. Deficit-guided selection is compared against random mutation and a
fixed-order heuristic, and an injected wrong-tile bug must be caught.

Requires a CUDA GPU and a working nvcc. Prints the loop trace; writes no evidence file.

  python d_1c_v_autonomy_loop_end_to_end_real_cuda.py
"""
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

assert torch.cuda.is_available()
T_START = time.time()
dev = torch.cuda.get_device_properties(0)
SM = dev.multi_processor_count
THEO_BW = 672.0  # GB/s: RTX 5070, 192-bit GDDR7 @14001MHz-class (nvidia-smi clocks.max.memory=14001MHz); reused
                 # unmodified from d_1c_iv_end_to_end_real_cuda_cert.py / d_1c_iv_best_in_class_float4.py, both run on
                 # this SAME GPU (external anchor: consistent across independent prior real runs, not asserted fresh).
print(f"=== device={dev.name} SM_count={SM} THEO_BW={THEO_BW}GB/s torch={torch.__version__} cuda={torch.version.cuda} ===")
print("CONTAM-FLAG: GPU is SHARED (desktop compositor/Xorg resident ~1.7GB, ~7%% util at rest per nvidia-smi). Roofline"
      " %% below may carry a light negative bias vs a fully-idle GPU; NOT corrected for, flagged honestly.\n")

# -----------------------------------------------------------------------------------------------------------------
# REAL CUDA: the whole small mutation-set as ONE nvcc-compiled library. grid & block are RUNTIME params (int64_t) --
# no recompilation to vary launch config; recompilation IS required to change mode/width (genuine source-level axes).
# -----------------------------------------------------------------------------------------------------------------
CUDA_SRC = r'''
#include <torch/extension.h>

__global__ void k_atomic_scalar(const float* __restrict__ x, float* out, int n){
  int stride = blockDim.x*gridDim.x;
  for (int j = blockIdx.x*blockDim.x+threadIdx.x; j<n; j+=stride) atomicAdd(out, x[j]);
}
__global__ void k_atomic_float4(const float4* __restrict__ x4, float* out, int n4){
  int stride = blockDim.x*gridDim.x;
  for (int j = blockIdx.x*blockDim.x+threadIdx.x; j<n4; j+=stride){
    float4 f = x4[j]; atomicAdd(out, f.x+f.y+f.z+f.w);
  }
}
__global__ void k_tree_scalar(const float* __restrict__ x, float* p, int n){
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  int stride=blockDim.x*gridDim.x;
  for (int j=blockIdx.x*blockDim.x+t; j<n; j+=stride) v+=x[j];
  s[t]=v; __syncthreads();
  for (int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if (t==0) p[blockIdx.x]=s[0];
}
__global__ void k_tree_float4(const float4* __restrict__ x4, float* p, int n4){
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  int stride=blockDim.x*gridDim.x;
  for (int j=blockIdx.x*blockDim.x+t; j<n4; j+=stride){ float4 f=x4[j]; v+=f.x+f.y+f.z+f.w; }
  s[t]=v; __syncthreads();
  for (int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if (t==0) p[blockIdx.x]=s[0];
}
__global__ void k_tree_wrongtile_scalar(const float* __restrict__ x, float* p, int n_half){
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  int stride=blockDim.x*gridDim.x;
  for (int j=blockIdx.x*blockDim.x+t; j<n_half; j+=stride) v+=x[j];
  s[t]=v; __syncthreads();
  for (int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if (t==0) p[blockIdx.x]=s[0];
}

torch::Tensor atomic_scalar(torch::Tensor x, int64_t block, int64_t grid){
  int n=x.numel(); auto out=torch::zeros({1}, x.options());
  k_atomic_scalar<<<(int)grid,(int)block>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
torch::Tensor atomic_float4(torch::Tensor x, int64_t block, int64_t grid){
  int n4=x.numel()/4; auto out=torch::zeros({1}, x.options());
  k_atomic_float4<<<(int)grid,(int)block>>>((const float4*)x.data_ptr<float>(), out.data_ptr<float>(), n4);
  return out;
}
torch::Tensor tree_scalar(torch::Tensor x, int64_t block, int64_t grid){
  int n=x.numel(); auto p=torch::zeros({grid}, x.options());
  size_t shmem=(size_t)block*sizeof(float);
  k_tree_scalar<<<(int)grid,(int)block,shmem>>>(x.data_ptr<float>(), p.data_ptr<float>(), n);
  return p;
}
torch::Tensor tree_float4(torch::Tensor x, int64_t block, int64_t grid){
  int n4=x.numel()/4; auto p=torch::zeros({grid}, x.options());
  size_t shmem=(size_t)block*sizeof(float);
  k_tree_float4<<<(int)grid,(int)block,shmem>>>((const float4*)x.data_ptr<float>(), p.data_ptr<float>(), n4);
  return p;
}
torch::Tensor tree_wrongtile_scalar(torch::Tensor x, int64_t block, int64_t grid){
  int n=x.numel(); int n_half=n/2; auto p=torch::zeros({grid}, x.options());
  size_t shmem=(size_t)block*sizeof(float);
  k_tree_wrongtile_scalar<<<(int)grid,(int)block,shmem>>>(x.data_ptr<float>(), p.data_ptr<float>(), n_half);
  return p;
}
'''
CPP_DECL = ("torch::Tensor atomic_scalar(torch::Tensor, int64_t, int64_t);\n"
            "torch::Tensor atomic_float4(torch::Tensor, int64_t, int64_t);\n"
            "torch::Tensor tree_scalar(torch::Tensor, int64_t, int64_t);\n"
            "torch::Tensor tree_float4(torch::Tensor, int64_t, int64_t);\n"
            "torch::Tensor tree_wrongtile_scalar(torch::Tensor, int64_t, int64_t);\n")

print("compiling real CUDA (nvcc via load_inline) -- the whole mutation-set library, ONE compile unit...", flush=True)
t0 = time.time()
m = load_inline(name="d1cv_autoloop", cpp_sources=[CPP_DECL], cuda_sources=[CUDA_SRC],
                 functions=["atomic_scalar", "atomic_float4", "tree_scalar", "tree_float4", "tree_wrongtile_scalar"],
                 verbose=False)
T_COMPILE_LIB = time.time() - t0
print(f"compiled OK in {T_COMPILE_LIB:.1f}s (real nvcc).\n")

BLOCK = 256
TOL = 1e-2
TARGET_ROOF = 0.85
# N=1<<26 (67M elem, 268MB), NOT 1<<24: FORCED CORRECTION (OODA, run 1) -- at N=1<<24 (16M elem, 67MB) EVERY kernel
# variant (atomic/tree x scalar/float4 x all grid_mult 1..64) plateaus at 74-76% roof regardless of quality, because
# the ~130-190us kernel body is small enough that fixed per-launch overhead (Python dispatch+kernel launch+alloc,
# ~20-50us) is a non-negligible (~15-25%) fraction of measured time -- a measurement-regime artifact that makes 85%
# UNREACHABLE by any config (verified: real cross-check at N=67M on the IDENTICAL best config jumped 76.0%->88.7%).
# At N=67M the kernel body (~530-600us) dwarfs the fixed overhead (~3-5%), so genuine kernel-quality differences show.
N = 1 << 26
x = torch.rand(N, device='cuda', dtype=torch.float32)
ref = torch.sum(x).item()
print(f"N={N} ({N*4/1e6:.0f} MB); ref (torch.sum/cub) = {ref:.1f}\n")

RAW = {('atomic', 'scalar'): m.atomic_scalar, ('atomic', 'float4'): m.atomic_float4,
       ('tree', 'scalar'): m.tree_scalar, ('tree', 'float4'): m.tree_float4}


def _fn(mode, width, block, grid, xx):
    if mode == 'atomic':
        return lambda: RAW[(mode, width)](xx, block, grid).item()
    return lambda: torch.sum(RAW[(mode, width)](xx, block, grid)).item()


def cert(mode, width, grid_mult, det_reps=5, time_reps=7, xx=None, rr=None, n=None, block=BLOCK):
    xx = x if xx is None else xx
    rr = ref if rr is None else rr
    n = N if n is None else n
    grid = grid_mult * SM
    fn = _fn(mode, width, block, grid, xx)
    torch.cuda.synchronize()
    vals = [fn() for _ in range(det_reps)]
    deterministic = len(set(vals)) == 1
    correct = abs(vals[0] - rr) / abs(rr) < TOL
    ts = []
    for _ in range(time_reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    bw = 4 * n / min(ts) / 1e9
    roof = bw / THEO_BW
    return dict(det=deterministic, correct=correct, bw=bw, roof=roof, spread=max(vals) - min(vals), grid=grid)


def deficits(r):
    return np.array([0.0 if r['det'] else 1.0, 0.0 if r['correct'] else 1.0, max(0.0, TARGET_ROOF - r['roof'])])


def is_certified(r):
    return r['det'] and r['correct'] and r['roof'] >= TARGET_ROOF


# -----------------------------------------------------------------------------------------------------------------
# PILOT SWEEP (real GPU measurements, BEFORE deciding the weak start): roofline vs grid_mult, tree-mode, both widths.
# This gives the GEOMETRY (not an assumed formula) that the "cross-aware" fix and the weak-start choice are DERIVED
# from -- and reveals whether {width, grid_mult} genuinely couple on real hardware (both move the SAME roof number).
# -----------------------------------------------------------------------------------------------------------------
GRID_MULTS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
print("--- PILOT SWEEP (real GPU): roofline vs grid_mult, tree-mode, scalar vs float4 ---")
sweep = {}
for width in ['scalar', 'float4']:
    for gm in GRID_MULTS:
        r = cert('tree', width, gm, det_reps=2, time_reps=7)
        sweep[(width, gm)] = r
        print(f"  tree/{width:6s} grid_mult={gm:3d} grid={gm*SM:5d}: det={str(r['det']):5s} correct={str(r['correct']):5s} "
              f"BW={r['bw']:6.1f} GB/s roof={r['roof']*100:5.1f}%")

best_f4 = max(GRID_MULTS, key=lambda gm: sweep[('float4', gm)]['roof'])
max_roof_f4 = sweep[('float4', best_f4)]['roof']
GRID_OPT = min(gm for gm in GRID_MULTS if sweep[('float4', gm)]['roof'] >= max_roof_f4 - 0.01)
worst_gm_scalar = min(GRID_MULTS, key=lambda gm: sweep[('scalar', gm)]['roof'])
width_effect = sweep[('float4', GRID_OPT)]['roof'] - sweep[('scalar', GRID_OPT)]['roof']
grid_effect = max(sweep[('scalar', gm)]['roof'] for gm in GRID_MULTS) - sweep[('scalar', worst_gm_scalar)]['roof']
print(f"\n  => GRID_OPT (measured): grid_mult={GRID_OPT} (grid={GRID_OPT*SM}, smallest within 1pp of the swept max "
      f"{max_roof_f4*100:.1f}%); worst measured grid_mult (scalar)={worst_gm_scalar}")
print(f"  => measured Jacobian on roofline: width_effect(scalar->float4 @ GRID_OPT)={width_effect*100:+.1f}pp   "
      f"grid_effect(worst->best @ scalar)={grid_effect*100:+.1f}pp  "
      f"=> {'WIDTH dominates' if width_effect>=grid_effect else 'GRID_MULT dominates'} (real, measured, not assumed)\n")

# -----------------------------------------------------------------------------------------------------------------
# Weak start: EMPIRICALLY the worst grid_mult (from the sweep, not asserted) + atomic (non-det) + scalar (low BW).
# -----------------------------------------------------------------------------------------------------------------
WEAK_START = dict(mode='atomic', width='scalar', grid_mult=worst_gm_scalar)
print(f"WEAK START (deliberately bad on all 3 axes, worst-grid empirically chosen): {WEAK_START}")
r0 = cert(**WEAK_START, det_reps=5, time_reps=7)
print(f"  cert(weak start): det={r0['det']} correct={r0['correct']} BW={r0['bw']:.1f} GB/s roof={r0['roof']*100:.1f}% "
      f"spread={r0['spread']:.3e}  deficits={np.round(deficits(r0),3)}\n")


# ----------------------------- policies -----------------------------
def fix_det(c):
    c = dict(c); c['mode'] = 'tree'; return c


def fix_correct_fallback(c):
    c = dict(c); c['mode'] = 'tree'; return c


def fix_roof_crossaware(c):
    c = dict(c)
    need_width = c['width'] != 'float4'
    need_grid = c['grid_mult'] != GRID_OPT
    if not need_width and not need_grid:
        return c  # no lever left in this mutation set
    if need_width and need_grid:
        if width_effect >= grid_effect:
            c['width'] = 'float4'
        else:
            c['grid_mult'] = GRID_OPT
    elif need_width:
        c['width'] = 'float4'
    else:
        c['grid_mult'] = GRID_OPT
    return c


def step_deficit_guided(c, r):
    d = deficits(r)
    i = int(np.argmax(d))
    if d[i] <= 0:
        return dict(c)
    return [fix_det, fix_correct_fallback, fix_roof_crossaware][i](c)


def step_fixed_order(c, r):
    # NAIVE fixed order, IGNORES the measured deficit vector: mode->tree, then width->float4, then grid_mult->LARGEST
    # swept value (a plausible "more parallelism = better" heuristic that does NOT check the real optimum).
    c = dict(c)
    if c['mode'] != 'tree':
        c['mode'] = 'tree'; return c
    if c['width'] != 'float4':
        c['width'] = 'float4'; return c
    if c['grid_mult'] != GRID_MULTS[-1]:
        c['grid_mult'] = GRID_MULTS[-1]; return c
    return c


def step_random(c, rng):
    c2 = dict(c)
    knob = int(rng.integers(0, 3))
    if knob == 0:
        c2['mode'] = 'atomic' if c['mode'] == 'tree' else 'tree'
    elif knob == 1:
        c2['width'] = 'scalar' if c['width'] == 'float4' else 'float4'
    else:
        opts = [g for g in GRID_MULTS if g != c['grid_mult']]
        c2['grid_mult'] = int(rng.choice(opts))
    return c2


def run_deterministic(policy_fn, start, cap, det_reps=3, time_reps=5):
    c = dict(start); path = []
    for it in range(cap + 1):
        r = cert(**c, det_reps=det_reps, time_reps=time_reps)
        path.append((dict(c), r))
        if is_certified(r):
            return dict(iters=it, config=dict(c), path=path, certified=True, stuck=False)
        c2 = policy_fn(c, r)
        if c2 == c:
            return dict(iters=it, config=dict(c), path=path, certified=False, stuck=True)
        c = c2
    return dict(iters=cap, config=dict(c), path=path, certified=False, stuck=False)


def run_random(start, seed, cap, det_reps=3, time_reps=5):
    rng = np.random.default_rng(seed)
    c = dict(start)
    r = cert(**c, det_reps=det_reps, time_reps=time_reps)
    total = deficits(r).sum()
    for it in range(cap + 1):
        if is_certified(r):
            return it
        c2 = step_random(c, rng)
        r2 = cert(**c2, det_reps=det_reps, time_reps=time_reps)
        t2 = deficits(r2).sum()
        if t2 <= total:
            c, r, total = c2, r2, t2
    return cap + 1  # did not solve within cap


# -----------------------------------------------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------------------------------------------
CAP_GUIDED = 15
CAP_RANDOM = 40
N_SEEDS = 20

print("=== DEFICIT-GUIDED (cross-aware) loop ===")
res_guided = run_deterministic(step_deficit_guided, WEAK_START, cap=CAP_GUIDED)
for i, (cc, rr) in enumerate(res_guided['path']):
    print(f"  it{i}: {cc}  det={str(rr['det']):5s} correct={str(rr['correct']):5s} roof={rr['roof']*100:5.1f}%  "
          f"deficits={np.round(deficits(rr),3)}")
print(f"  -> iters={res_guided['iters']} certified={res_guided['certified']} stuck={res_guided['stuck']}")

repeat_iters = [res_guided['iters']]
for rep in range(2):
    rr = run_deterministic(step_deficit_guided, WEAK_START, cap=CAP_GUIDED)
    repeat_iters.append(rr['iters'])
print(f"  stability check (3x from same start, same policy): iters={repeat_iters} "
      f"(flags measurement-noise-driven path divergence -- {'STABLE' if len(set(repeat_iters))==1 else 'UNSTABLE'})\n")

print("=== FIXED-ORDER-NAIVE loop [adversary: does the MEASURED optimum matter vs a dumb fixed heuristic?] ===")
res_naive = run_deterministic(step_fixed_order, WEAK_START, cap=CAP_GUIDED)
for i, (cc, rr) in enumerate(res_naive['path']):
    print(f"  it{i}: {cc}  det={str(rr['det']):5s} correct={str(rr['correct']):5s} roof={rr['roof']*100:5.1f}%")
print(f"  -> iters={res_naive['iters']} certified={res_naive['certified']} stuck={res_naive['stuck']}\n")

print(f"=== RANDOM-MUTATION baseline ({N_SEEDS} seeds, greedy-accept on total deficit, cap={CAP_RANDOM}) ===")
rand_iters = np.array([run_random(WEAK_START, seed=s, cap=CAP_RANDOM) for s in range(N_SEEDS)])
rand_solved = rand_iters <= CAP_RANDOM
print(f"  raw iters per seed: {rand_iters.tolist()}")
print(f"  solved-rate={rand_solved.mean()*100:.0f}%  "
      f"iters(solved only): mean={rand_iters[rand_solved].mean() if rand_solved.any() else float('nan'):.1f} "
      f"median={np.median(rand_iters[rand_solved]) if rand_solved.any() else float('nan'):.1f} "
      f"min={rand_iters[rand_solved].min() if rand_solved.any() else '-'} "
      f"max={rand_iters[rand_solved].max() if rand_solved.any() else '-'}\n")

# -----------------------------------------------------------------------------------------------------------------
# FINAL RIGOROUS CERT on the deficit-guided converged config: 10 independent full-rigor reps (stability band, guards
# the "0.85 pass is a noise fluke" adversary).
# -----------------------------------------------------------------------------------------------------------------
fc = res_guided['config']
print(f"=== FINAL RIGOROUS CERT (10x, det_reps=5 time_reps=25) on deficit-guided's converged config {fc} ===")
final_reps = [cert(**fc, det_reps=5, time_reps=25) for _ in range(10)]
roofs = np.array([rr['roof'] for rr in final_reps])
all_det = all(rr['det'] for rr in final_reps)
all_correct = all(rr['correct'] for rr in final_reps)
print(f"  across 10 independent reps: det=ALL-TRUE:{all_det}  correct=ALL-TRUE:{all_correct}  "
      f"roof min={roofs.min()*100:.1f}% mean={roofs.mean()*100:.1f}% max={roofs.max()*100:.1f}% "
      f"(spread={((roofs.max()-roofs.min())*100):.2f}pp)")
robust_certified = all_det and all_correct and (roofs.min() >= TARGET_ROOF)
print(f"  ROBUST certified (min-of-10 >= 85%): {robust_certified}\n")

# -----------------------------------------------------------------------------------------------------------------
# CROSS-CHECK at a SECOND independent large N (1<<25, 33.5M elem, ~134MB) -- diverse instance-space, not one lucky
# size. Also re-affirms (forensic, not re-run) run-1's measured finding: at N=1<<24 the SAME final config plateaued
# at 76.0% (see d1cv_run2.log) -- the small-N overhead-floor is real and reproducible, not a one-off.
# -----------------------------------------------------------------------------------------------------------------
print("=== CROSS-CHECK at a SECOND independent large N=1<<25 (33.5M elem, 134MB) -- diverse instance-space ===")
N2 = 1 << 25
x2 = torch.rand(N2, device='cuda', dtype=torch.float32)
ref2 = torch.sum(x2).item()
r2 = cert(**fc, det_reps=5, time_reps=15, xx=x2, rr=ref2, n=N2)
print(f"  N=33.5M: det={r2['det']} correct={r2['correct']} BW={r2['bw']:.0f} GB/s roof={r2['roof']*100:.1f}% "
      f"(vs N=67M roof={roofs.mean()*100:.1f}%) -- {'CONSISTENT' if abs(r2['roof']-roofs.mean())<0.05 else 'DIVERGENT (N-size confound flagged)'}\n")

# -----------------------------------------------------------------------------------------------------------------
# SIDE-CHECK: injected correctness bug (wrong-tile, drops back half). Does the cert-vector CATCH it end-to-end?
# -----------------------------------------------------------------------------------------------------------------
print("=== SIDE-CHECK: injected wrong-tile correctness bug -- does the cert-vector CATCH it? ===")
grid_wt = GRID_OPT * SM
vals_wt = [torch.sum(m.tree_wrongtile_scalar(x, BLOCK, grid_wt)).item() for _ in range(5)]
det_wt = len(set(vals_wt)) == 1
relerr_wt = abs(vals_wt[0] - ref) / abs(ref)
correct_wt = relerr_wt < TOL
print(f"  tree_wrongtile_scalar: det={det_wt} val={vals_wt[0]:.1f} vs ref={ref:.1f} relerr={relerr_wt*100:.1f}% "
      f"correct(<{TOL*100:.0f}%)={correct_wt} -> verdict={'CERTIFY (BUG MISSED!)' if det_wt and correct_wt else 'REJECT (correctness) -- CAUGHT'}\n")

# -----------------------------------------------------------------------------------------------------------------
# RECOMPILE-COST CHECK: one FRESH isolated nvcc/load_inline invocation (single kernel, brand-new module name), timed.
# Answers "is recompile-per-iteration too slow" honestly, independent of the amortized 5-kernel-library compile above.
# -----------------------------------------------------------------------------------------------------------------
print("=== RECOMPILE-COST CHECK: one FRESH isolated nvcc/load_inline invocation (single kernel), timed ===")
FRESH_SRC = r'''
#include <torch/extension.h>
__global__ void k_fresh(const float4* __restrict__ x4, float* p, int n4){
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  int stride=blockDim.x*gridDim.x;
  for (int j=blockIdx.x*blockDim.x+t; j<n4; j+=stride){ float4 f=x4[j]; v+=f.x+f.y+f.z+f.w; }
  s[t]=v; __syncthreads();
  for (int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if (t==0) p[blockIdx.x]=s[0];
}
torch::Tensor fresh_fn(torch::Tensor x, int64_t block, int64_t grid){
  int n4=x.numel()/4; auto p=torch::zeros({grid}, x.options());
  size_t shmem=(size_t)block*sizeof(float);
  k_fresh<<<(int)grid,(int)block,shmem>>>((const float4*)x.data_ptr<float>(), p.data_ptr<float>(), n4);
  return p;
}
'''
t0 = time.time()
m_fresh = load_inline(name=f"d1cv_freshcompile_{int(time.time())}", cpp_sources=["torch::Tensor fresh_fn(torch::Tensor, int64_t, int64_t);"],
                       cuda_sources=[FRESH_SRC], functions=["fresh_fn"], verbose=False)
t_fresh_compile = time.time() - t0
print(f"  fresh nvcc compile (1 kernel, brand-new module name, cold): {t_fresh_compile:.1f}s")
print(f"  (for reference: whole 5-kernel library compile above took {T_COMPILE_LIB:.1f}s)\n")

# -----------------------------------------------------------------------------------------------------------------
# VERDICT
# -----------------------------------------------------------------------------------------------------------------
print("=" * 100)
print("VERDICT (pre-registered thresholds, applied to the REAL measurements above):")
adv_ratio = (rand_iters[rand_solved].mean() / max(res_guided['iters'], 1)) if (rand_solved.any() and res_guided['iters'] > 0) else float('inf')
guided_reached = res_guided['certified'] and robust_certified
forced_advantage = guided_reached and ((rand_solved.mean() < 1.0) or (adv_ratio >= 2.0))
bug_caught = not correct_wt
naive_also_certifies = res_naive['certified']

print(f"  deficit-guided: reached CERTIFIED={res_guided['certified']} in {res_guided['iters']} iters; ROBUST "
      f"(min-of-10 roof={roofs.min()*100:.1f}%>=85%)={robust_certified}; path stability={len(set(repeat_iters))==1}")
print(f"  random-mutation: solved-rate={rand_solved.mean()*100:.0f}%/{N_SEEDS} seeds within cap={CAP_RANDOM}; "
      f"mean-iters(solved)={rand_iters[rand_solved].mean() if rand_solved.any() else float('nan'):.1f}; "
      f"advantage ratio (random/guided)={adv_ratio:.1f}x")
print(f"  fixed-order-naive: reached CERTIFIED={res_naive['certified']} in {res_naive['iters']} iters, final roof="
      f"{res_naive['path'][-1][1]['roof']*100:.1f}% (GRID_OPT={GRID_OPT} vs naive's target grid_mult={GRID_MULTS[-1]})")
print(f"  wrongtile bug-catch: {'CAUGHT (REJECT)' if bug_caught else 'MISSED (FALSE CERTIFY -- adversary WINS)'} (relerr={relerr_wt*100:.1f}%)")
print(f"  recompile cost: 5-kernel library={T_COMPILE_LIB:.1f}s one-off; isolated fresh single-kernel={t_fresh_compile:.1f}s")
print(f"  total wall-clock: {time.time()-T_START:.1f}s")

if guided_reached and forced_advantage and bug_caught:
    verdict = "SUPPORTED"
elif guided_reached and bug_caught:
    verdict = "PARTIAL"
else:
    verdict = "REFUTED"
print(f"\n  ⟹ NORTH-STAR AUTONOMY (end-to-end, real compiled CUDA, deficit-guided compile-cert-refine loop): {verdict}")
if not forced_advantage:
    print("    (advantage-over-random NOT forced -- random matched/beat deficit-guided or the state-space was too easy)")
if naive_also_certifies:
    print(f"    HONEST: fixed-order-naive ALSO reached certified ({res_naive['iters']} iters) -- on THIS real hardware/knob-set"
          f" there was no antagonistic coupling (unlike CORE-2's constructed toy R4), so a dumb fixed order sufficed too;"
          f" cross-awareness's real edge here is the {'grid_mult target (GRID_OPT vs naive MAX)' if width_effect<grid_effect else 'ordering choice'}, not cycle-avoidance.")
print("=" * 100)
import sys
sys.exit(0 if verdict == "SUPPORTED" else (2 if verdict == "PARTIAL" else 1))

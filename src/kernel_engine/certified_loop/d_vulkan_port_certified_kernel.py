"""Port a certified CUDA reduction kernel to Vulkan and re-run the same certificate on both APIs.

The kernel is a deterministic fixed-order float4 tree reduction (block 256, grid 2048, grid-stride,
shared-memory tree). Gate 1 compares the 2048-element partial array bitwise between CUDA and Vulkan over
a size sweep and an input-pattern sweep, localising any deviation against an exact fp32 CPU oracle.
Gate 2 re-runs the full certificate on the Vulkan side: three-fold bit repeat, correctness against a
float64 reference, and the bandwidth fraction against both the theoretical and the API-local measured
roofline.

Requires a CUDA GPU, nvcc and the wgpu Python package.
Output: artifacts/d_vulkan_certified_port_evidence.json.

  python d_vulkan_port_certified_kernel.py
"""
import fcntl, json, os, subprocess, time
import numpy as np

LOCK_PATH = os.path.expanduser("~/.cache/cadtosim_gpu.lock")
lock_f = open(LOCK_PATH, "w")
print("acquiring GPU flock (timed run, exclusive) ...", flush=True)
fcntl.flock(lock_f, fcntl.LOCK_EX)
print("flock HELD.", flush=True)
smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,name",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(f"nvidia-smi at start: {smi}")

import torch  # noqa: E402  (import after flock: CUDA ctx counts as GPU presence)
import wgpu   # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

assert torch.cuda.is_available()
THEO_BW = 672.0   # RTX 5070 anchor, unchanged from d_1c_iv / d_1c_v (same GPU)
TOL = 1e-2        # cert's correctness tolerance
TARGET_ROOF = 0.85
BLOCK, GRID = 256, 2048  # the canonical certified config (d_1c_iv float4_p)

# ---------------------------------------------------------------- CUDA side (verbatim certified kernel + copy)
CUDA_SRC = r'''
#include <torch/extension.h>
__global__ void red_float4(const float4* x, float* p, int n4){
  extern __shared__ float s[]; int t=threadIdx.x; float v=0.f;
  for(int j=blockIdx.x*blockDim.x+t; j<n4; j+=blockDim.x*gridDim.x){ float4 f=x[j]; v+=f.x+f.y+f.z+f.w; }
  s[t]=v; __syncthreads();
  for(int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if(t==0) p[blockIdx.x]=s[0];
}
__global__ void copy_float4(const float4* x, float4* y, int n4){
  for(int j=blockIdx.x*blockDim.x+threadIdx.x; j<n4; j+=blockDim.x*gridDim.x) y[j]=x[j];
}
torch::Tensor float4_p(torch::Tensor x){ int n4=x.numel()/4; auto p=torch::zeros({2048},x.options());
  red_float4<<<2048,256,256*sizeof(float)>>>((const float4*)x.data_ptr<float>(),p.data_ptr<float>(),n4); return p; }
torch::Tensor copy4(torch::Tensor x, torch::Tensor y){ int n4=x.numel()/4;
  copy_float4<<<2048,256>>>((const float4*)x.data_ptr<float>(),(float4*)y.data_ptr<float>(),n4); return y; }
'''
CPP = "torch::Tensor float4_p(torch::Tensor); torch::Tensor copy4(torch::Tensor, torch::Tensor);"
print("compiling CUDA (nvcc load_inline)...", flush=True)
mc = load_inline(name="d_vkport_cuda", cpp_sources=[CPP], cuda_sources=[CUDA_SRC],
                 functions=["float4_p", "copy4"], verbose=False)
print("CUDA compiled OK.", flush=True)

# ---------------------------------------------------------------- Vulkan side (wgpu; recipe: add-only WGSL port)
# EXACT association preserved: CUDA v+=f.x+f.y+f.z+f.w  ==  v = v + ((((f.x+f.y)+f.z)+f.w))
WGSL_RED = """
@group(0) @binding(0) var<storage,read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> outp: array<f32>;
var<workgroup> s: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>,
        @builtin(num_workgroups) nw: vec3<u32>) {
  let t = l.x; let n4 = arrayLength(&inp);
  let stride = 256u * nw.x;
  var v: f32 = 0.0;
  var j = w.x * 256u + t;
  loop { if (j >= n4) { break; }
    let f = inp[j];
    v = v + (((f.x + f.y) + f.z) + f.w);
    j = j + stride; }
  s[t] = v; workgroupBarrier();
  var st = 128u;
  loop { if (st == 0u) { break; }
    if (t < st) { s[t] = s[t] + s[t + st]; }
    workgroupBarrier(); st = st >> 1u; }
  if (t == 0u) { outp[w.x] = s[0]; }
}
"""
WGSL_COPY = """
@group(0) @binding(0) var<storage,read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> outp: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>,
        @builtin(num_workgroups) nw: vec3<u32>) {
  let n4 = arrayLength(&inp); let stride = 256u * nw.x;
  var j = w.x * 256u + l.x;
  loop { if (j >= n4) { break; } outp[j] = inp[j]; j = j + stride; }
}
"""
ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print(f"wgpu adapter: {ad.info['device']} backend={ad.info['backend_type']}")
assert ad.info["backend_type"] == "Vulkan"
dev = ad.request_device_sync(required_limits={"max-storage-buffer-binding-size": 1 << 30,
                                              "max-buffer-size": 1 << 30})
pipe_red = dev.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto,
    compute={"module": dev.create_shader_module(code=WGSL_RED), "entry_point": "main"})
pipe_copy = dev.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto,
    compute={"module": dev.create_shader_module(code=WGSL_COPY), "entry_point": "main"})


def vk_run(pipe, in_np, out_bytes, n_wg, k_dispatch=1, out_dtype=np.float32):
    """One submit with k_dispatch back-to-back dispatches (WebGPU auto-syncs storage hazards between
    dispatches in a pass). Returns (output ndarray, wall seconds for the submit+wait)."""
    ib = dev.create_buffer_with_data(data=in_np, usage=wgpu.BufferUsage.STORAGE)
    ob = dev.create_buffer(size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {"binding": 0, "resource": {"buffer": ib, "offset": 0, "size": in_np.nbytes}},
        {"binding": 1, "resource": {"buffer": ob, "offset": 0, "size": out_bytes}}])
    # warm dispatch (pipeline/driver warmup) not timed
    enc = dev.create_command_encoder(); cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe); cp.set_bind_group(0, bg); cp.dispatch_workgroups(n_wg); cp.end()
    dev.queue.submit([enc.finish()]); dev.queue.read_buffer(ob, size=4)
    enc = dev.create_command_encoder(); cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
    for _ in range(k_dispatch):
        cp.dispatch_workgroups(n_wg)
    cp.end(); cb = enc.finish()
    t0 = time.perf_counter()
    dev.queue.submit([cb]); dev.queue.read_buffer(ob, size=4)  # read forces queue-idle wait
    dt = time.perf_counter() - t0
    out = np.frombuffer(dev.queue.read_buffer(ob), dtype=out_dtype).copy()
    ib.destroy(); ob.destroy()
    return out, dt


def vk_reduce_partials(x_np):
    out, _ = vk_run(pipe_red, x_np, GRID * 4, GRID)
    return out


# ---------------------------------------------------------------- fp32-exact CPU ORACLE (same op order, no fma)
def oracle_partials(x_np):
    n4 = x_np.size // 4
    x4 = x_np.reshape(n4, 4)
    nthreads = GRID * BLOCK
    acc = np.zeros(nthreads, dtype=np.float32)
    idx = np.arange(nthreads, dtype=np.int64)
    j = idx.copy()
    while True:
        m = j < n4
        if not m.any():
            break
        f = x4[j[m]]
        q = ((f[:, 0] + f[:, 1]) + f[:, 2]) + f[:, 3]   # float32 adds, exact CUDA/WGSL association
        acc[m] = acc[m] + q
        j = j + nthreads
    s = acc.reshape(GRID, BLOCK)
    st = BLOCK // 2
    while st > 0:
        s = np.concatenate([s[:, :st] + s[:, st:2 * st], s[:, st:]], axis=1)  # s[t]+=s[t+st] for t<st
        st //= 2
    return np.ascontiguousarray(s[:, 0])


# ---------------------------------------------------------------- Gate 1: bit-exact sweep (size x pattern)
rng = np.random.default_rng(7)
def make_pattern(name, n):
    if name == "uniform01":
        return rng.random(n, dtype=np.float32)
    if name == "mixed_sign":
        return (rng.random(n, dtype=np.float32) * 2 - 1).astype(np.float32)
    if name == "wide_dynrange":
        e = rng.uniform(-18, 18, n).astype(np.float32)
        sg = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        return (sg * np.exp(e * np.log(np.float32(2.0)))).astype(np.float32)
    if name == "subnormal_scale":
        return (rng.random(n, dtype=np.float32) * np.float32(1e-38)).astype(np.float32)
    raise ValueError(name)

SIZES = [1 << 16, 1 << 20, 1 << 24, 1 << 26]
PATTERNS = ["uniform01", "mixed_sign", "wide_dynrange", "subnormal_scale"]
gate1 = []
print("\n=== GATE 1: BIT-EXACT partials CUDA vs Vulkan (u32 view), size x pattern sweep ===")
for n in SIZES:
    for pat in PATTERNS:
        x_np = make_pattern(pat, n)
        xt = torch.from_numpy(x_np).cuda()
        p_cuda = mc.float4_p(xt).cpu().numpy()
        p_vk = vk_reduce_partials(x_np)
        be = float((p_cuda.view(np.uint32) == p_vk.view(np.uint32)).mean())
        md = float(np.abs(p_cuda.astype(np.float64) - p_vk.astype(np.float64)).max())
        row = dict(n=n, pattern=pat, bit_exact_frac=be, max_abs_delta=md)
        if be < 1.0:  # LOCALIZE with the oracle
            p_or = oracle_partials(x_np)
            row["cuda_vs_oracle"] = float((p_cuda.view(np.uint32) == p_or.view(np.uint32)).mean())
            row["vk_vs_oracle"] = float((p_vk.view(np.uint32) == p_or.view(np.uint32)).mean())
            row["n_mismatch"] = int((p_cuda.view(np.uint32) != p_vk.view(np.uint32)).sum())
        gate1.append(row)
        extra = "" if be == 1.0 else f"  ORACLE: cuda={row['cuda_vs_oracle']:.4f} vk={row['vk_vs_oracle']:.4f}"
        print(f"  N=2^{n.bit_length()-1:2d} {pat:16s}: bit-exact={be:.4f} maxΔ={md:.2e}{extra}")
        del xt
gate1_pass = all(r["bit_exact_frac"] == 1.0 for r in gate1)
print(f"  GATE 1 {'PASS — bit-exact at every size/pattern' if gate1_pass else 'FAIL — see oracle attribution rows'}")

# oracle spot-check on one cell (validates the instrument, not just used-on-failure)
x_np = make_pattern("uniform01", 1 << 20)
p_or = oracle_partials(x_np)
p_cu = mc.float4_p(torch.from_numpy(x_np).cuda()).cpu().numpy()
oracle_ok = bool((p_or.view(np.uint32) == p_cu.view(np.uint32)).all())
print(f"  instrument check: fp32-exact CPU oracle == CUDA partials bitwise: {oracle_ok}")

# ---------------------------------------------------------------- Gate 2: CERT-VECTOR TRANSFER at canonical N
print("\n=== GATE 2: cert-vector on the VULKAN side (canonical N=2^26, config b=256 g=2048) ===")
N = 1 << 26
x_np = make_pattern("uniform01", N)
ref64 = float(np.sum(x_np.astype(np.float64)))
xt = torch.from_numpy(x_np).cuda()


def host_total(partials):  # identical final combine both sides: float32 sequential
    t = np.float32(0.0)
    for v in partials:
        t = np.float32(t + v)
    return float(t)


# -- R2 bit-repeat x3 (fresh submits / fresh launches)
vk_runs = [vk_reduce_partials(x_np) for _ in range(3)]
vk_r2 = bool(all((vk_runs[0].view(np.uint32) == r.view(np.uint32)).all() for r in vk_runs[1:]))
cu_runs = [mc.float4_p(xt).cpu().numpy() for _ in range(3)]
cu_r2 = bool(all((cu_runs[0].view(np.uint32) == r.view(np.uint32)).all() for r in cu_runs[1:]))
cross_bit = bool((vk_runs[0].view(np.uint32) == cu_runs[0].view(np.uint32)).all())

# -- correctness vs float64 reference
vk_total = host_total(vk_runs[0]); cu_total = host_total(cu_runs[0])
vk_correct = abs(vk_total - ref64) / abs(ref64) < TOL
cu_correct = abs(cu_total - ref64) / abs(ref64) < TOL
totals_bitmatch = np.float32(vk_total).view() == np.float32(cu_total)

# -- roofline: reduction + copy, both APIs, same instrument shape (K dispatches / submit, best of R submits)
K, R = 30, 5
def vk_bw(pipe, traffic_bytes, out_bytes):
    ts = []
    for _ in range(R):
        _, dt = vk_run(pipe, x_np, out_bytes, GRID, k_dispatch=K)
        ts.append(dt / K)
    return traffic_bytes / min(ts) / 1e9

vk_red_bw = vk_bw(pipe_red, 4 * N, GRID * 4)
vk_copy_bw = vk_bw(pipe_copy, 8 * N, x_np.nbytes)

y = torch.empty_like(xt)
def cu_bw(fn, traffic_bytes):
    torch.cuda.synchronize()
    ts = []
    for _ in range(R):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(K):
            fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / K)
    return traffic_bytes / min(ts) / 1e9

cu_red_bw = cu_bw(lambda: mc.float4_p(xt), 4 * N)
cu_copy_bw = cu_bw(lambda: mc.copy4(xt, y), 8 * N)

vk_roof_theo, cu_roof_theo = vk_red_bw / THEO_BW, cu_red_bw / THEO_BW
vk_roof_copy, cu_roof_copy = vk_red_bw / vk_copy_bw, cu_red_bw / cu_copy_bw

print(f"  R2 bit-repeat x3      : CUDA={cu_r2}  Vulkan={vk_r2}   (cross-API partials bit-exact={cross_bit})")
print(f"  correctness (TOL {TOL}) : CUDA={cu_correct} (rel {abs(cu_total-ref64)/ref64:.2e})  "
      f"Vulkan={vk_correct} (rel {abs(vk_total-ref64)/ref64:.2e})  totals bit-match={bool(totals_bitmatch)}")
print(f"  reduction BW          : CUDA={cu_red_bw:6.1f} GB/s ({cu_roof_theo*100:5.1f}% of THEO {THEO_BW})   "
      f"Vulkan={vk_red_bw:6.1f} GB/s ({vk_roof_theo*100:5.1f}%)")
print(f"  copy roofline (API)   : CUDA={cu_copy_bw:6.1f} GB/s   Vulkan={vk_copy_bw:6.1f} GB/s")
print(f"  red/copy (mix-naive)  : CUDA={cu_roof_copy*100:5.1f}%   Vulkan={vk_roof_copy*100:5.1f}%")
smi2 = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                       "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(f"  nvidia-smi during-run state: {smi2} (flock held; desktop compositor resident — same contam-flag as CUDA certs)")

facets = {
    "R2_determinism": dict(cuda=cu_r2, vulkan=vk_r2, transfers=bool(cu_r2 and vk_r2)),
    "correctness": dict(cuda=bool(cu_correct), vulkan=bool(vk_correct), transfers=bool(cu_correct and vk_correct)),
    "bit_exact_cross_api": dict(value=cross_bit, transfers=cross_bit),
    "roofline_vs_theo": dict(cuda=cu_roof_theo, vulkan=vk_roof_theo,
                             cuda_pass=bool(cu_roof_theo >= TARGET_ROOF), vulkan_pass=bool(vk_roof_theo >= TARGET_ROOF),
                             transfers=bool((cu_roof_theo >= TARGET_ROOF) == (vk_roof_theo >= TARGET_ROOF)
                                            and vk_roof_theo >= TARGET_ROOF)),
    "roofline_vs_api_copy": dict(cuda=cu_roof_copy, vulkan=vk_roof_copy),
}
cert_transfers = bool(facets["R2_determinism"]["transfers"] and facets["correctness"]["transfers"]
                      and facets["roofline_vs_theo"]["transfers"])

print("\n" + "=" * 100)
print("VERDICT (pre-registered: fingerprint-invariance predicts FULL transfer):")
print(f"  Gate-1 bit-exact sweep : {'PASS' if gate1_pass else 'FAIL (localized above)'}")
for k, v in facets.items():
    print(f"  facet {k:22s}: {v}")
print(f"  CERT TRANSFERS: {cert_transfers}")
if gate1_pass and cert_transfers:
    print("  ⟹ PUNCHLINE: a Vulkan deployment INHERITS the CUDA certification WITHOUT re-certification,")
    print("    under these exact conditions: (1) kernel is composed of the verified portable primitive set")
    print("    ({mul, add, explicit fma()}; this kernel is add-only) with source-preserved association;")
    print("    (2) same launch geometry (b=256, g=2048) and same grid-stride element order; (3) same silicon")
    print("    (RTX 5070) — the invariance is API/compiler-level, NOT cross-device; (4) throughput facet")
    print("    re-checked because scheduling is API-owned (measured above, not assumed).")
print("=" * 100)

evidence = dict(
    cell="d_vulkan_port_certified_kernel", date="2026-07-07",
    device=ad.info["device"], backend=ad.info["backend_type"], theo_bw_gbs=THEO_BW,
    config=dict(block=BLOCK, grid=GRID, N_canonical=N, tol=TOL, target_roof=TARGET_ROOF),
    preregistered_prediction="cert transfers (fingerprint-invariance: same algo/floor/silicon, only API changed)",
    gate1_bit_exact_sweep=gate1, gate1_pass=gate1_pass, oracle_instrument_check=oracle_ok,
    gate2_facets=facets, cert_transfers=cert_transfers,
    totals=dict(ref_float64=ref64, cuda_total=cu_total, vulkan_total=vk_total),
    bandwidth_gbs=dict(cuda_reduction=cu_red_bw, vulkan_reduction=vk_red_bw,
                       cuda_copy=cu_copy_bw, vulkan_copy=vk_copy_bw),
    nvidia_smi=dict(start=smi, during=smi2),
    contam_flag="GPU shared with desktop compositor (resident); flock held for entire timed run",
)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts",
                        "d_vulkan_certified_port_evidence.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(evidence, f, indent=1)
print(f"evidence -> {out_path}")
fcntl.flock(lock_f, fcntl.LOCK_UN); lock_f.close()
print("flock RELEASED.")

"""d_coupled_knob_ordering_advantage_real_cuda.py — CLOSES THE RECURRING CAVEAT of both autonomy cells
(d_1c_v_autonomy_loop_end_to_end_real_cuda + d_autonomous_multicomponent_stitch): in both, deficit-guided
refinement reached certified BUT the fixed-order-naive adversary ALSO certified in the same iteration count,
because the real knob spaces tested were UNCOUPLED. The in-model results (P1/P4: wrong-order/blind refinement
4-14x worse under requirement coupling) were never realized on real silicon. THIS CELL constructs a real,
PHYSICALLY-coupled CUDA knob space and races the policies — or establishes honestly that real CUDA knob
spaces resist such coupling.

SCENARIO (one real fused kernel family, ONE nvcc compile unit, RTX 5070, shared GPU, flock-serialized):
"contact-impulse transform + binned scatter": n=2^24 values x~U(0,1); per element a=4+4x, y=exp(-a) computed
by an on-GPU truncated series (D=36 terms, three algorithm/precision variants); y scattered (fixed-point
quantized for integer paths, scale 2^22) into K=1024 bins with a HOT-BIN distribution (80% of contacts into
8 bins — contact-style body skew). REQUIREMENTS (the cert vector):
  R_det : bin vector bit-repeats across 5 runs (determinism required).
  R_acc : max-over-bins relerr vs independent fp64 torch reference <= 5e-6.
  R_perf: kernel time <= theta = (measured memory envelope)/0.30 (>=30% of the mem-bound roofline).

KNOBS (6, all real source/launch-level axes; the SAME mutation set for every policy):
  mode   in {af32(global float atomics), aint(global int atomics), priv(shared-privatized int + det flush)}
  gw     in {32,64}   global fixed-point accumulator width  (aint/priv)
  sw     in {32,64}   shared bin width                      (priv)
  rep    in {1,4,8}   privatization replication copies      (priv)   shared bytes = rep*K*sw/8
  sprec  in {alt_f32, recip_f32, alt_f64}  transform algorithm/precision
  unroll in {2,8}     per-thread element batch (ILP; per-thread register state scales with unroll x sprec)

THE PHYSICAL COUPLINGS (engineered from REAL mechanisms, no simulated penalties — each is MEASURED as a
coupling-matrix entry BEFORE the race; foundation gate: if no real coupling >=1.5x adverse cross-effect
exists, THAT is the honest finding and the cell stops):
  C-PREC-ALU   (prompt C3-flavour, precision-vs-throughput): sprec->alt_f64 fixes R_acc but consumer-GPU
               fp64 runs at 1:64 the fp32 rate -> R_perf collapses. REAL antagonistic edge: the accuracy
               knob poisons the throughput requirement; no perf-attributed knob (unroll/rep/mode) can
               recover it — recovery REQUIRES UNDOING the precision choice (the cycle, measured).
               The certified escape is ALGORITHMIC (recip_f32: reformulate the ill-conditioned alternating
               series as a well-conditioned positive series + reciprocal), not brute precision.
  C-SHMEM-BUDGET (prompt C1): sw and rep SHARE the 48KB default dynamic-shared budget (rep*K*sw/8 bytes):
               widening shared bins (sw 32->64, the naive "precision" move) EVICTS replication — at rep=8
               the launch is physically infeasible (64KB > 48KB, real CUDA launch error); at rep=4 it
               doubles shared/block (occupancy pressure, measured graded). Disclosed: the hard wall is the
               DEFAULT shared-mem config (opt-in to ~99KB exists on this arch and is not taken).
  C-REG-OCC    (prompt C2): unroll x sprec sets per-thread register state (unroll fp64 accumulator chains);
               registers/occupancy read from cuobjdump resource usage + measured perf deltas. May honestly
               be <1.5x — reported per-edge either way.
  C-CONTENTION : mode couples R_det AND R_perf through the same physical mechanism (hot-address atomic
               serialization): fixing det by aint keeps the global hot-atomic serialization that kills perf;
               priv fixes both. Cross-requirement by construction of the hardware, not by penalty.

PRE-REGISTERED (stated before running):
  theta_time = (n*8 bytes / measured peak BW) / 0.30; TOL_ACC = 5e-6; det = 5x bitwise; t = min-of-7.
  Iteration currency = REAL certs consumed (incl. start cert and rejected/infeasible attempts). CAP = 20.
  COUPLING FOUNDATION: >=2 matrix entries with adverse cross-effect >=1.5x (launch-infeasible counts, as a
    hard wall, flagged as such) else verdict = NO-REAL-COUPLING (honest stop).
  RACE ARMS (identical mutation set, identical cert instrument, identical weak start (af32,32,32,1,alt_f32,2)):
    (i)   deficit-guided WITH coupling-aware attribution: 2c-style mechanical static rules (overflow
          envelope, series conditioning kappa*eps, atomic-dominance envelope) EXTENDED with the measured
          coupling-matrix cross-terms (candidates that fix the binding facet are ranked by measured adverse
          cross-effect; matrix-infeasible candidates skipped).
    (i-b) ABLATION (causal isolation): same guided policy with the coupling matrix REMOVED (cross-term
          blind, ranks by static predicted-accuracy magnitude alone) — isolates WHICH ingredient buys the
          win (the cross-terms, not deficit-guidance per se).
    (ii)  fixed-order-naive (the adversary that tied twice): fixed requirement order det->acc->perf; per
          requirement a fixed canonical 'bigger/stronger-first' mutation list (widen-shared, widen-global,
          raise-precision(f64), algorithmic-swap LAST); accepts iff the TARGET requirement's own metric
          improves (cross-effects invisible by definition); generous variant: list pointer resets after any
          accepted change (retries allowed). PLUS a 'lucky-order' naive variant (algorithmic-swap before
          f64) reported as ordering-variance evidence, NOT the primary adversary.
    (iii) blind-random: 20 seeds, uniform single-knob mutation, greedy accept iff total normalized deficit
          non-increasing, cap 20.
  SUPPORTED iff: coupling foundation holds AND known-good certifies robustly (direct run, min-of-10 band)
    AND deficit-guided certifies in ALL >=3 repeats AND fixed-order-naive (primary) FAILS to certify within
    cap in ALL >=3 repeats OR needs >=2x guided's certs (ratio in [1.8,2.5) -> [GRAZING] flag).
  REFUTED-HONEST iff fixed-order-naive ties (<2x, certifies): real CUDA knob spaces resist the coupling —
    a scope statement for CORE-2's advantage claim.
  SYMMETRIC-QC: certified endpoint existence proven by direct run BEFORE the race; the adversary uses the
    same mutation set/instrument (asserted on the same list object); guided's extra information (the matrix
    probes) is priced: an ALT accounting (guided certs + matrix-probe certs) is reported alongside;
    stop/stuck claims measured, not narrated; grazing flagged on every bar.

Run:  PRECOMPILE=1 python3 d_coupled_knob_ordering_advantage_real_cuda.py   (warm nvcc cache)
      flock .cache/gpu.lock python3 d_coupled_knob_ordering_advantage_real_cuda.py
Evidence JSON -> artifacts/d_coupled_ordering_advantage_evidence.json
"""
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(HERE, "artifacts", "d_coupled_ordering_advantage_evidence.json")
T0_WALL = time.time()

assert torch.cuda.is_available()
DEV = torch.cuda.get_device_properties(0)
SM = DEV.multi_processor_count
print(f"=== device={DEV.name} SM={SM} shared/block(default)={DEV.shared_memory_per_block}B "
      f"shared/SM={getattr(DEV,'shared_memory_per_multiprocessor','?')}B "
      f"torch={torch.__version__} cuda={torch.version.cuda} ===")
print("CONTAM-FLAG: GPU is SHARED (desktop compositor resident); timing carries light negative bias, flagged not corrected.\n")

# ---------------------------------------------------------------- task spec (everything downstream derives from this)
N = 1 << 24
K = 1024
N_HOT = 8
HOT_FRAC = 0.80
D_TERMS = 36
SCALE = float(2 ** 22)
TOL_ACC = 5e-6
BLOCK = 256
GRID = 4 * SM                       # fixed launch shape (not a knob; disclosed)
CAP = 20
EPS32, EPS64 = 2.0 ** -24, 2.0 ** -53
SMEM_LIMIT = int(DEV.shared_memory_per_block)   # default per-block dynamic shared limit (48KB, no opt-in taken)

# ---------------------------------------------------------------- CUDA source: templated kernel, 42 instantiations
KERNEL_TMPL = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#define D_TERMS 36

template<typename T> __device__ __forceinline__ float series_alt(float xin){      // exp(-a), alternating Taylor
  T a = (T)4 + (T)4 * (T)xin; T term = (T)1; T s = (T)1;
  #pragma unroll
  for (int k = 1; k <= D_TERMS; k++){ term = -term * a / (T)k; s += term; }
  return (float)s;
}
template<typename T> __device__ __forceinline__ float series_recip(float xin){    // 1/exp(+a), positive series
  T a = (T)4 + (T)4 * (T)xin; T term = (T)1; T s = (T)1;
  #pragma unroll
  for (int k = 1; k <= D_TERMS; k++){ term = term * a / (T)k; s += term; }
  return (float)((T)1 / s);
}

// MODE: 0=af32 (global float atomics), 1=aint (global int atomics), 2=priv (shared privatized int + det flush)
// GW/SW: 32|64. PREC: 0=alt_f32, 1=recip_f32, 2=alt_f64. UNROLL: 2|8. rep is a RUNTIME knob (dynamic shared).
template<int MODE,int GW,int SW,int PREC,int UNROLL>
__global__ void k_fused(const float* __restrict__ x, const int* __restrict__ idx,
                        float* gb_f, unsigned int* gb_i32, unsigned long long* gb_i64,
                        int n, int K_, int rep, float scale){
  extern __shared__ char smem_raw[];
  unsigned int*       sb32 = (unsigned int*)smem_raw;
  unsigned long long* sb64 = (unsigned long long*)smem_raw;
  if (MODE == 2){
    for (int i = threadIdx.x; i < K_ * rep; i += blockDim.x){ if (SW == 32) sb32[i] = 0u; else sb64[i] = 0ull; }
    __syncthreads();
  }
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = gridDim.x * blockDim.x;
  const int myrep = (MODE == 2) ? ((int)(threadIdx.x >> 5) % rep) : 0;
  for (long long base = (long long)tid; base < n; base += (long long)nth * UNROLL){
    float ys[UNROLL]; int is[UNROLL]; bool ok[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; u++){
      long long j = base + (long long)u * nth;            // coalesced per-u
      ok[u] = (j < n);
      float xv = ok[u] ? x[j] : 0.f;
      is[u] = ok[u] ? idx[j] : 0;
      if      (PREC == 0) ys[u] = series_alt  <float >(xv);
      else if (PREC == 1) ys[u] = series_recip<float >(xv);
      else                ys[u] = series_alt  <double>(xv);
    }
    #pragma unroll
    for (int u = 0; u < UNROLL; u++){
      if (!ok[u]) continue;
      if (MODE == 0){ atomicAdd(&gb_f[is[u]], ys[u]); }
      else {
        unsigned long long q = (unsigned long long)llrintf(ys[u] * scale);
        if (MODE == 1){
          if (GW == 32) atomicAdd(&gb_i32[is[u]], (unsigned int)q);
          else          atomicAdd(&gb_i64[is[u]], q);
        } else {
          if (SW == 32) atomicAdd(&sb32[myrep * K_ + is[u]], (unsigned int)q);
          else          atomicAdd(&sb64[myrep * K_ + is[u]], q);
        }
      }
    }
  }
  if (MODE == 2){
    __syncthreads();
    for (int i = threadIdx.x; i < K_; i += blockDim.x){
      unsigned long long tot = 0ull;
      for (int r = 0; r < rep; r++) tot += (SW == 32) ? (unsigned long long)sb32[r * K_ + i] : sb64[r * K_ + i];
      if (GW == 32) atomicAdd(&gb_i32[i], (unsigned int)tot);
      else          atomicAdd(&gb_i64[i], tot);
    }
  }
}

torch::Tensor run_cfg(torch::Tensor x, torch::Tensor idx, int64_t mode, int64_t gw, int64_t sw,
                      int64_t prec, int64_t unroll, int64_t rep, int64_t grid, int64_t block, double scale){
  const int n = (int)x.numel(); const int K_ = @K@;
  auto optf   = x.options();
  auto opti32 = x.options().dtype(torch::kInt32);
  auto opti64 = x.options().dtype(torch::kInt64);
  torch::Tensor out;
  float* gb_f = nullptr; unsigned int* gb_i32 = nullptr; unsigned long long* gb_i64 = nullptr;
  if (mode == 0){ out = torch::zeros({K_}, optf);   gb_f   = out.data_ptr<float>(); }
  else if (gw == 32){ out = torch::zeros({K_}, opti32); gb_i32 = (unsigned int*)out.data_ptr<int32_t>(); }
  else { out = torch::zeros({K_}, opti64); gb_i64 = (unsigned long long*)out.data_ptr<int64_t>(); }
  size_t shmem = (mode == 2) ? (size_t)rep * K_ * (sw / 8) : 0;
  const float* xp = x.data_ptr<float>(); const int* ip = idx.data_ptr<int>();
@DISPATCH@
  else { TORCH_CHECK(false, "no instantiation for cfg"); }
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "LAUNCH_FAIL: ", cudaGetErrorString(e));
  return out;
}
'''

dispatch_lines, inst = [], []
for mode in (0, 1, 2):
    gws = (32,) if mode == 0 else (32, 64)
    sws = (32, 64) if mode == 2 else (32,)
    for gw, sw, prec, unroll in itertools.product(gws, sws, (0, 1, 2), (2, 8)):
        cond = f"mode=={mode} && gw=={gw} && sw=={sw} && prec=={prec} && unroll=={unroll}"
        call = (f"k_fused<{mode},{gw},{sw},{prec},{unroll}><<<(int)grid,(int)block,shmem>>>"
                f"(xp, ip, gb_f, gb_i32, gb_i64, n, K_, (int)rep, (float)scale);")
        kw = "if" if not dispatch_lines else "else if"
        dispatch_lines.append(f"  {kw} ({cond}) {{ {call} }}")
        inst.append((mode, gw, sw, prec, unroll))
CUDA_SRC = KERNEL_TMPL.replace("@K@", str(K)).replace("@DISPATCH@", "\n".join(dispatch_lines))
print(f"generated {len(inst)} template instantiations, one compile unit; compiling (nvcc, load_inline)...", flush=True)
t0 = time.time()
from torch.utils.cpp_extension import load_inline
M = load_inline(name="d_coupled_ordering_v1",
                cpp_sources=["torch::Tensor run_cfg(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, "
                             "int64_t, int64_t, int64_t, int64_t, int64_t, double);"],
                cuda_sources=[CUDA_SRC], functions=["run_cfg"], verbose=False)
T_COMPILE = time.time() - t0
print(f"compiled OK in {T_COMPILE:.1f}s\n")
if os.environ.get("PRECOMPILE"):
    print("PRECOMPILE=1 -> exiting after warm compile."); sys.exit(0)

# ---------------------------------------------------------------- data + independent fp64 reference
g = torch.Generator(device='cuda').manual_seed(1234)
x = torch.rand(N, device='cuda', dtype=torch.float32, generator=g)
hot_mask = torch.rand(N, device='cuda', generator=g) < HOT_FRAC
idx_hot = torch.randint(0, N_HOT, (N,), device='cuda', dtype=torch.int32, generator=g)
idx_cold = torch.randint(N_HOT, K, (N,), device='cuda', dtype=torch.int32, generator=g)
idx = torch.where(hot_mask, idx_hot, idx_cold).contiguous()
a64 = 4.0 + 4.0 * x.double()
y64 = torch.exp(-a64)
REF = torch.zeros(K, device='cuda', dtype=torch.float64).index_add_(0, idx.long(), y64)
REF_CPU = REF.cpu().numpy()
assert (REF_CPU > 0).all()
print(f"data: n={N} K={K} hot={HOT_FRAC*100:.0f}%/{N_HOT} bins; ref bins: hot~{REF_CPU[:N_HOT].mean():.1f} "
      f"cold~{REF_CPU[N_HOT:].mean():.2f} (fp64 torch reference, independent instrument)")

# ---------------------------------------------------------------- measured peak BW -> pre-registered theta
big = torch.rand(1 << 26, device='cuda')
def _bw(fn, bytes_):
    torch.cuda.synchronize(); ts = []
    for _ in range(9):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return bytes_ / min(ts)
bw_copy = _bw(lambda: big.clone(), 2 * big.numel() * 4)
bw_read = _bw(lambda: big.sum(), big.numel() * 4)
PEAK = max(bw_copy, bw_read)
BYTES_MIN = N * 8                                    # x fp32 + idx int32 per element (bins negligible)
T_MEM = BYTES_MIN / PEAK
THETA = T_MEM / 0.30                                 # pre-registered: >=30% of memory envelope
print(f"peak BW measured: copy={bw_copy/1e9:.1f} read={bw_read/1e9:.1f} GB/s -> T_mem={T_MEM*1e3:.3f} ms; "
      f"PRE-REGISTERED theta={THETA*1e3:.3f} ms, TOL_ACC={TOL_ACC:.0e}, det=5x bitwise, t=min-of-7, CAP={CAP}\n")
del big

# ---------------------------------------------------------------- config space + cert instrument
MODES = {'af32': 0, 'aint': 1, 'priv': 2}
SPRECS = {'alt_f32': 0, 'recip_f32': 1, 'alt_f64': 2}
KNOBS = dict(mode=['af32', 'aint', 'priv'], gw=[32, 64], sw=[32, 64], rep=[1, 4, 8],
             sprec=['alt_f32', 'recip_f32', 'alt_f64'], unroll=[2, 8])
# THE shared mutation set: every single-knob assignment (same object handed to every policy — asserted below)
MUTATIONS = [(k, v) for k, vals in KNOBS.items() for v in vals]
WEAK = dict(mode='af32', gw=32, sw=32, rep=1, sprec='alt_f32', unroll=2)

def applicable(cfg, knob, val):
    if cfg[knob] == val: return False
    if knob == 'gw' and cfg['mode'] == 'af32': return False
    if knob in ('sw', 'rep') and cfg['mode'] != 'priv': return False
    return True

def shared_bytes(cfg):
    return cfg['rep'] * K * (cfg['sw'] // 8) if cfg['mode'] == 'priv' else 0

CERT_COUNT = {'n': 0}
def run_once(cfg):
    # canonicalize N/A knobs (gw/sw meaningless under af32, sw meaningless under aint): semantic no-ops
    gw = 32 if cfg['mode'] == 'af32' else cfg['gw']
    sw = cfg['sw'] if cfg['mode'] == 'priv' else 32
    return M.run_cfg(x, idx, MODES[cfg['mode']], gw, sw, SPRECS[cfg['sprec']],
                     cfg['unroll'], cfg['rep'], GRID, BLOCK, SCALE)

def bins_to_float(cfg, b):
    if cfg['mode'] == 'af32': return b.double()
    if cfg['gw'] == 32: return ((b.to(torch.int64)) & 0xFFFFFFFF).double() / SCALE
    return b.double() / SCALE     # i64 accumulated as u64; values < 2^62, reinterpret exact

def cert(cfg, det_reps=5, time_reps=7, count=True):
    if count: CERT_COUNT['n'] += 1
    try:
        outs = [run_once(cfg) for _ in range(det_reps)]
        torch.cuda.synchronize()
    except RuntimeError as e:
        torch.cuda.synchronize()
        return dict(feasible=False, err=str(e).splitlines()[0][:120], det=False, relerr=float('inf'),
                    t=float('inf'), shmem=shared_bytes(cfg))
    det = all(torch.equal(outs[0], o) for o in outs[1:])
    bf = bins_to_float(cfg, outs[0])
    relerr = float(((bf - REF).abs() / REF).max())
    ts = []
    for _ in range(time_reps):
        torch.cuda.synchronize(); t = time.perf_counter(); run_once(cfg); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return dict(feasible=True, det=det, relerr=relerr, t=min(ts), shmem=shared_bytes(cfg))

def deficits(r):
    if not r['feasible']: return np.array([1.0, 1.0, 1.0])
    return np.array([0.0 if r['det'] else 1.0,
                     min(1.0, max(0.0, math.log10(max(r['relerr'], 1e-300) / TOL_ACC)) / 6.0),
                     min(1.0, max(0.0, math.log10(r['t'] / THETA)) / 2.0)])

def certified(r):
    return r['feasible'] and r['det'] and r['relerr'] <= TOL_ACC and r['t'] <= THETA

def fmt(cfg, r):
    return (f"{cfg['mode']:4s}/gw{cfg['gw']}/sw{cfg['sw']}/rep{cfg['rep']}/{cfg['sprec']:9s}/u{cfg['unroll']} "
            + (f"det={str(r['det']):5s} relerr={r['relerr']:.2e} t={r['t']*1e3:7.3f}ms d={np.round(deficits(r),3)}"
               if r['feasible'] else f"INFEASIBLE ({r['err']}) shmem={r['shmem']}B"))

# ================================================================ (c) SYMMETRIC-QC first leg: endpoint EXISTS
print("--- (c) SYMMETRIC-QC: certified endpoint existence (direct run of the known-good config, min-of-10 band) ---")
KNOWN_GOOD = dict(mode='priv', gw=64, sw=32, rep=1, sprec='recip_f32', unroll=2)
kg_runs = [cert(KNOWN_GOOD, count=False) for _ in range(10)]
kg_t = np.array([r['t'] for r in kg_runs]); kg_rel = max(r['relerr'] for r in kg_runs)
kg_det = all(r['det'] for r in kg_runs)
kg_robust = kg_det and kg_rel <= TOL_ACC and kg_t.max() <= THETA
kg_margin_t, kg_margin_acc = THETA / kg_t.max(), TOL_ACC / max(kg_rel, 1e-300)
print(f"  known-good {fmt(KNOWN_GOOD, kg_runs[0])}")
print(f"  min-of-10 band: t=[{kg_t.min()*1e3:.3f},{kg_t.max()*1e3:.3f}]ms relerr_max={kg_rel:.2e} det_all={kg_det}")
print(f"  ROBUST-CERTIFIED={kg_robust}  margins: perf {kg_margin_t:.2f}x, acc {kg_margin_acc:.1f}x"
      f"{'  [GRAZING]' if kg_margin_t < 1.3 or kg_margin_acc < 3 else ''}\n")
if not kg_robust:
    print("FATAL: certified endpoint does not exist/robustly -> the race would be a strawman. Stopping honestly.")
    json.dump(dict(verdict="ENDPOINT-UNREACHABLE", known_good=KNOWN_GOOD,
                   kg=dict(det=kg_det, relerr=kg_rel, t_band=[float(kg_t.min()), float(kg_t.max())],
                           theta=THETA)), open(ARTIFACT, "w"), indent=1)
    sys.exit(1)

# ================================================================ (a) THE COUPLING MATRIX — measured, before race
print("--- (a) COUPLING MATRIX (measured on real silicon: flip knob A alone, watch requirement B's metric) ---")
PROBE_COUNT_BEFORE = CERT_COUNT['n']
def probe(base, knob, val, label, mech):
    r0 = cert(base); c2 = dict(base); c2[knob] = val; r1 = cert(c2)
    ent = dict(label=label, mechanism=mech, base={k: base[k] for k in base}, flip=[knob, val],
               base_cert=dict(det=r0['det'], relerr=r0['relerr'], t=r0['t'], feasible=r0['feasible']),
               flip_cert=dict(det=r1['det'], relerr=r1['relerr'], t=r1['t'], feasible=r1['feasible'],
                              err=r1.get('err', '')))
    if not r1['feasible']:
        ent['perf_ratio'] = float('inf'); ent['adverse'] = True; ent['hard_wall'] = True
    else:
        ent['perf_ratio'] = r1['t'] / r0['t']; ent['hard_wall'] = False
        ent['adverse'] = bool(ent['perf_ratio'] >= 1.5 or (r0['det'] and not r1['det'])
                              or (r0['relerr'] <= TOL_ACC < r1['relerr']))
    print(f"  {label:22s} flip {knob}->{val}: "
          + (f"perf x{ent['perf_ratio']:.2f} det {r0['det']}->{r1['det']} relerr {r0['relerr']:.1e}->{r1['relerr']:.1e}"
             if r1['feasible'] else f"LAUNCH-INFEASIBLE (shmem {shared_bytes(c2)}B > {SMEM_LIMIT}B)")
          + f"  adverse={ent['adverse']}{' [HARD-WALL: default shmem cfg; ~99KB opt-in exists, not taken]' if ent['hard_wall'] else ''}")
    return ent

MATRIX = {}
MATRIX['C_PREC_ALU'] = probe(KNOWN_GOOD, 'sprec', 'alt_f64', 'C-PREC-ALU',
                             'accuracy knob -> fp64 1:64 ALU rate poisons R_perf (precision/throughput antagonism)')
MATRIX['C_PREC_ALU_u8'] = probe(dict(KNOWN_GOOD, unroll=8), 'sprec', 'alt_f64', 'C-PREC-ALU@u8',
                                'same cliff at unroll=8 (ILP cannot rescue a throughput-bound fp64 pipe)')
MATRIX['C_SHMEM_HARD'] = probe(dict(KNOWN_GOOD, rep=8), 'sw', 64, 'C-SHMEM-BUDGET hard',
                               'widening shared bins evicts replication: rep8*K*8B=64KB > 48KB default budget')
MATRIX['C_SHMEM_GRADED'] = probe(dict(KNOWN_GOOD, rep=4), 'sw', 64, 'C-SHMEM-BUDGET graded',
                                 'sw 32->64 at rep4 doubles shared/block 16->32KB (occupancy pressure)')
MATRIX['C_CONT_MODE'] = probe(KNOWN_GOOD, 'mode', 'aint', 'C-CONTENTION mode',
                              'global int atomics on hot bins: same-address serialization (det-fix keeps perf-kill)')
MATRIX['C_CONT_AF32'] = probe(KNOWN_GOOD, 'mode', 'af32', 'C-CONTENTION af32',
                              'float atomics: nondet + accumulation error + hot serialization (3 facets, 1 knob)')
MATRIX['C_REG_OCC_f64'] = probe(dict(KNOWN_GOOD, sprec='alt_f64'), 'unroll', 8, 'C-REG-OCC @f64',
                                'unroll x fp64 accumulator chains -> register state (cuobjdump below)')
MATRIX['C_REP'] = probe(KNOWN_GOOD, 'rep', 8, 'C-REP contention', 'replication vs shared hot-atomic conflicts')
MATRIX['C_UNROLL'] = probe(KNOWN_GOOD, 'unroll', 8, 'C-UNROLL ILP', 'per-thread ILP batch at fp32 (mem-bound?)')
PROBE_CERTS = CERT_COUNT['n'] - PROBE_COUNT_BEFORE

# cuobjdump resource usage (C-REG-OCC physical evidence), best-effort
REGS = {}
try:
    so = M.__file__
    out = subprocess.run(["cuobjdump", "--dump-resource-usage", so], capture_output=True, text=True, timeout=120)
    blob = out.stdout
    for m_ in re.finditer(r"Function\s+(\S*k_fusedILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE\S*).*?REG:(\d+)(?:.*?STACK:(\d+))?(?:.*?lmem\[(\d+)\])?",
                          blob, re.S):
        mode_, gw_, sw_, prec_, unr_ = (int(m_.group(i)) for i in range(2, 7))
        REGS[(mode_, gw_, sw_, prec_, unr_)] = dict(reg=int(m_.group(7)), stack=int(m_.group(8) or 0))
    if REGS:
        r32 = REGS.get((2, 64, 32, 1, 2)); r64u8 = REGS.get((2, 64, 32, 2, 8))
        print(f"  cuobjdump: {len(REGS)} kernels parsed; e.g. priv/recip_f32/u2 REG={r32} vs priv/alt_f64/u8 REG={r64u8}")
    else:
        print("  cuobjdump: parsed 0 kernels (format mismatch) — C-REG-OCC rests on measured timing only")
except Exception as e:
    print(f"  cuobjdump unavailable ({e}) — C-REG-OCC rests on measured timing only")

n_adverse = sum(1 for v in MATRIX.values() if v['adverse'])
adverse_names = [k for k, v in MATRIX.items() if v['adverse']]
print(f"\n  COUPLING FOUNDATION: {n_adverse} adverse (>=1.5x or hard-wall or facet-flip) entries: {adverse_names}")
print(f"  matrix probe cost: {PROBE_CERTS} certs (priced into the ALT accounting below)\n")
if n_adverse < 2:
    print("HONEST FINDING: <2 real couplings on this real knob space -> NO-REAL-COUPLING verdict, stopping (pre-registered).")
    json.dump(dict(verdict="NO-REAL-COUPLING", matrix=MATRIX, theta=THETA, tol=TOL_ACC),
              open(ARTIFACT, "w"), indent=1, default=float)
    sys.exit(1)

# ================================================================ static rules (2c-style, spec-derived, mechanical)
E_Y = (math.exp(-4) - math.exp(-8)) / 4.0                                # E[exp(-a)], a~U(4,8) — closed form
def static_acc_pred(cfg):
    """Predicted max-bin relerr from spec envelopes (no measurement): series term + overflow term + f32-accum term."""
    kappa = math.exp(2 * 8.0)                                            # sum|terms|/|result| at a_max=8
    series = {'alt_f32': kappa * EPS32, 'recip_f32': D_TERMS * EPS32, 'alt_f64': kappa * EPS64}[cfg['sprec']]
    if cfg['mode'] == 'af32':
        n_hot = N * HOT_FRAC / N_HOT
        accum = math.sqrt(n_hot) * EPS32                                 # stochastic rounding of running f32 sum
        return max(series, accum)
    hot_qsum = N * HOT_FRAC / N_HOT * E_Y * SCALE
    overflow = 1.0 if (cfg['gw'] == 32 and hot_qsum > 2 ** 31) else 0.0
    return max(series, overflow)
def static_det_pred(cfg):
    return cfg['mode'] != 'af32'                                         # float atomics -> order-dependent
def static_perf_flags(cfg):
    """Atomic-dominance envelope (2c rule): per-hot-address serialized atomic count vs mem term."""
    per_addr = N * HOT_FRAC / N_HOT
    atomic_lb = per_addr / 2.5e9                                         # optimistic same-address atomic rate
    global_atomic_bound = cfg['mode'] in ('af32', 'aint') and atomic_lb > T_MEM
    return dict(global_atomic_bound=global_atomic_bound,
                fp64_alu=cfg['sprec'] == 'alt_f64')                      # flags only; magnitudes come from MATRIX

def predicted_fixes(binding, cfg, knob, val):
    c2 = dict(cfg); c2[knob] = val
    if binding == 0: return static_det_pred(c2) and not static_det_pred(cfg)
    if binding == 1: return static_acc_pred(c2) < static_acc_pred(cfg) * 0.99   # any predicted error reduction
    f0, f1 = static_perf_flags(cfg), static_perf_flags(c2)
    return (f0['global_atomic_bound'] and not f1['global_atomic_bound']) or \
           (f0['fp64_alu'] and not f1['fp64_alu']) or knob in ('unroll', 'rep')   # roofline knobs = candidates

def matrix_cross_penalty(cfg, knob, val):
    """Measured adverse cross-effect of this flip on NON-binding requirements (log-scale, inf if hard-wall)."""
    pen = 0.0
    c2 = dict(cfg); c2[knob] = val
    if c2['mode'] == 'priv' and shared_bytes(c2) > SMEM_LIMIT: return float('inf')   # measured hard wall (matrix)
    for ent in MATRIX.values():
        if ent['flip'] == [knob, val]:
            if ent['hard_wall']: return float('inf')
            pen = max(pen, math.log10(max(ent['perf_ratio'], 1e-9)) if ent['perf_ratio'] > 1 else 0.0)
            if ent['base_cert']['det'] and not ent['flip_cert']['det']: pen += 1.0
    return pen

# ================================================================ (b) THE RACE — policies (same mutation set!)
def run_guided(use_matrix=True, verbose=True, cap=CAP):
    c = dict(WEAK); certs = 0; traj = []; tried = set()
    r = cert(c); certs += 1; traj.append((dict(c), dict(det=r['det'], relerr=r['relerr'], t=r['t'],
                                                        feasible=r['feasible'])))
    if verbose: print(f"    cert{certs:2d}: {fmt(c, r)}")
    if certified(r): return dict(certified=True, certs=certs, config=dict(c), traj=traj)
    while certs < cap:
        d = deficits(r); binding = int(np.argmax(d))
        cands = []
        for knob, val in MUTATIONS:
            if not applicable(c, knob, val) or (json.dumps(c, sort_keys=True), knob, val) in tried: continue
            if not predicted_fixes(binding, c, knob, val): continue
            pen = matrix_cross_penalty(c, knob, val) if use_matrix else 0.0
            if pen == float('inf'): continue                              # matrix says physically blocked
            mag = static_acc_pred(dict(c, **{knob: val})) if binding == 1 else 0.0
            cands.append((pen, mag, knob, val))                           # lexicographic: cross-penalty, then magnitude
        if not cands:
            return dict(certified=False, certs=certs, config=dict(c), traj=traj, stuck=True)
        cands.sort(key=lambda z: (z[0], z[1]))
        pen, mag, knob, val = cands[0]
        tried.add((json.dumps(c, sort_keys=True), knob, val))
        c2 = dict(c); c2[knob] = val
        r2 = cert(c2); certs += 1; traj.append((dict(c2), dict(det=r2['det'], relerr=r2['relerr'], t=r2['t'],
                                                               feasible=r2['feasible'])))
        if verbose: print(f"    cert{certs:2d}: [{knob}->{val} pen={pen if pen!=float('inf') else 'inf'}] {fmt(c2, r2)}")
        if certified(r2): return dict(certified=True, certs=certs, config=dict(c2), traj=traj)
        if r2['feasible'] and deficits(r2)[binding] < d[binding] - 1e-9:
            c = c2                                                        # accept: binding facet measurably improved
    return dict(certified=False, certs=certs, config=dict(c), traj=traj, stuck=False)

NAIVE_LISTS_PRIMARY = dict(                                # 'bigger/stronger first'; algorithmic swap LAST
    det=[('mode', 'aint'), ('mode', 'priv')],
    acc=[('sw', 64), ('gw', 64), ('sprec', 'alt_f64'), ('sprec', 'recip_f32')],
    perf=[('unroll', 8), ('rep', 4), ('rep', 8), ('mode', 'aint'), ('mode', 'priv')])
NAIVE_LISTS_LUCKY = dict(                                  # ordering-variance control: swap BEFORE f64
    det=[('mode', 'aint'), ('mode', 'priv')],
    acc=[('gw', 64), ('sprec', 'recip_f32'), ('sprec', 'alt_f64'), ('sw', 64)],
    perf=[('unroll', 8), ('rep', 4), ('rep', 8), ('mode', 'aint'), ('mode', 'priv')])

def naive_metric(req, r):
    if not r['feasible']: return float('inf')
    return {'det': 0.0 if r['det'] else 1.0, 'acc': r['relerr'], 'perf': r['t']}[req]

def run_naive(lists, verbose=True, cap=CAP):
    """Fixed requirement order det->acc->perf; fixed canonical per-req mutation lists; accepts iff the TARGET
    requirement's own metric improves >=5% (cross-effects invisible BY DESIGN — that is the adversary's handicap;
    same mutations, same cert instrument). Generous: pointers reset after every accepted change."""
    c = dict(WEAK); certs = 0; traj = []; ptr = {k: 0 for k in lists}
    r = cert(c); certs += 1; traj.append((dict(c), dict(det=r['det'], relerr=r['relerr'], t=r['t'], feasible=r['feasible'])))
    if verbose: print(f"    cert{certs:2d}: {fmt(c, r)}")
    while certs < cap:
        if certified(r): return dict(certified=True, certs=certs, config=dict(c), traj=traj)
        binding = next((q for q, bad in [('det', not r['det']), ('acc', r['relerr'] > TOL_ACC),
                                         ('perf', r['t'] > THETA)] if bad), None)
        if binding is None: return dict(certified=True, certs=certs, config=dict(c), traj=traj)
        while ptr[binding] < len(lists[binding]) and not applicable(c, *lists[binding][ptr[binding]]):
            ptr[binding] += 1                                             # inapplicable mutations skip free
        if ptr[binding] >= len(lists[binding]):
            return dict(certified=False, certs=certs, config=dict(c), traj=traj, stuck=True, stuck_on=binding)
        knob, val = lists[binding][ptr[binding]]; ptr[binding] += 1
        c2 = dict(c); c2[knob] = val
        r2 = cert(c2); certs += 1; traj.append((dict(c2), dict(det=r2['det'], relerr=r2['relerr'], t=r2['t'],
                                                               feasible=r2['feasible'])))
        target_improved = r2['feasible'] and (naive_metric(binding, r2) < naive_metric(binding, r) * 0.95)
        if verbose: print(f"    cert{certs:2d}: [{binding}: {knob}->{val}] {fmt(c2, r2)} "
                          f"-> {'ACCEPT' if target_improved else 'reject'}")
        if target_improved:
            c, r = c2, r2; ptr = {k: 0 for k in lists}                    # generous reset (retries allowed)
    return dict(certified=False, certs=certs, config=dict(c), traj=traj, stuck=False)

def run_blind(seed, cap=CAP):
    rng = np.random.default_rng(seed)
    c = dict(WEAK); certs = 0
    r = cert(c); certs += 1; tot = deficits(r).sum()
    while certs < cap:
        if certified(r): return dict(certified=True, certs=certs)
        opts = [(k, v) for k, v in MUTATIONS if applicable(c, k, v)]
        k_, v_ = opts[int(rng.integers(len(opts)))]
        c2 = dict(c); c2[k_] = v_
        r2 = cert(c2); certs += 1
        if certified(r2): return dict(certified=True, certs=certs)
        t2 = deficits(r2).sum()
        if r2['feasible'] and t2 <= tot: c, r, tot = c2, r2, t2
    return dict(certified=False, certs=certs)

assert all(m in MUTATIONS for lst in NAIVE_LISTS_PRIMARY.values() for m in lst), "adversary mutation outside shared set"
print("--- (b) THE RACE (all policies: SAME mutation set object, SAME cert instrument, SAME weak start) ---")
print(f"  weak start: {WEAK}\n")

print("  [i] DEFICIT-GUIDED with measured coupling matrix (3 repeats):")
guided_runs = []
for rep_i in range(3):
    print(f"   repeat {rep_i+1}:")
    guided_runs.append(run_guided(use_matrix=True))
print(f"   -> certified={[g['certified'] for g in guided_runs]} certs={[g['certs'] for g in guided_runs]}\n")

print("  [i-b] ABLATION: same guided policy, coupling matrix REMOVED (cross-term blind):")
ablation = run_guided(use_matrix=False)
print(f"   -> certified={ablation['certified']} certs={ablation['certs']}"
      f"{' stuck' if ablation.get('stuck') else ''}\n")

print("  [ii] FIXED-ORDER-NAIVE primary ('bigger/stronger first', algorithmic swap last; 3 repeats):")
naive_runs = []
for rep_i in range(3):
    print(f"   repeat {rep_i+1}:")
    naive_runs.append(run_naive(NAIVE_LISTS_PRIMARY))
print(f"   -> certified={[nv['certified'] for nv in naive_runs]} certs={[nv['certs'] for nv in naive_runs]}\n")

print("  [ii-b] FIXED-ORDER-NAIVE lucky-order variant (ordering-variance control, 1 run):")
naive_lucky = run_naive(NAIVE_LISTS_LUCKY)
print(f"   -> certified={naive_lucky['certified']} certs={naive_lucky['certs']}\n")

print("  [iii] BLIND-RANDOM (20 seeds, greedy on total deficit):")
blind = [run_blind(s) for s in range(20)]
b_cert = np.array([b['certified'] for b in blind]); b_iters = np.array([b['certs'] for b in blind])
print(f"   solved {b_cert.sum()}/20 within cap {CAP}; certs solved-only: "
      f"{sorted(b_iters[b_cert].tolist()) if b_cert.any() else '-'} "
      f"(mean {b_iters[b_cert].mean() if b_cert.any() else float('nan'):.1f})\n")

# ================================================================ (c) remaining QC + the measured CYCLE
print("--- (c) SYMMETRIC-QC (rest) + the measured ordering trap ---")
guided_ok = all(g['certified'] for g in guided_runs)
guided_certs = max(g['certs'] for g in guided_runs)
naive_fail_all = all(not nv['certified'] for nv in naive_runs)
naive_certs = min(nv['certs'] for nv in naive_runs) if not naive_fail_all else float('inf')
ratio = (naive_certs / guided_certs) if guided_ok else float('nan')
grazing_ratio = (not naive_fail_all) and guided_ok and 1.8 <= ratio < 2.5
# the trap, measured from the primary naive trajectory: did it accept alt_f64 (acc pass) and then fail perf-stuck?
trap = None
for nv in naive_runs:
    accepted_f64 = any(cfg['sprec'] == 'alt_f64' for cfg, _ in nv['traj'][1:])
    if accepted_f64 and not nv['certified']:
        last_cfg, last_r = nv['traj'][-1][0], nv['traj'][-1][1]
        trap = dict(accepted_alt_f64=True, terminal_cfg=last_cfg, terminal_t_ms=last_r['t'] * 1e3,
                    stuck_on=nv.get('stuck_on'), certs=nv['certs'],
                    note="acc PASSES under alt_f64 so fixed-order never revisits precision; every perf-attributed "
                         "knob (unroll/rep/mode) measured unable to recover the fp64 ALU cliff -> recovery requires "
                         "UNDOING the precision choice, which fixed order cannot express")
print(f"  ordering trap measured: {json.dumps(trap, default=float) if trap else 'NOT OBSERVED'}")
print(f"  adversary fairness: same mutation set asserted; naive saw every cert measurement (accept/reject on its "
      f"own target metric); handicap is ONLY the absence of cross-term attribution/ordering.")
alt_guided_total = guided_certs + PROBE_CERTS
print(f"  ALT ACCOUNTING (matrix probes priced in): guided {guided_certs} certs + {PROBE_CERTS} probe certs = "
      f"{alt_guided_total} vs naive {'FAIL@'+str([nv['certs'] for nv in naive_runs]) if naive_fail_all else naive_certs}"
      f" -> matrix is calibration, reusable across tasks in this knob family; priced honestly either way.")
print(f"  ablation isolates cause: guided-without-matrix certified={ablation['certified']} "
      f"(if False/slower, the CROSS-TERMS are the causal ingredient, not deficit-guidance per se)")
print(f"  blind-vs-naive: blind solved {b_cert.sum()}/20, naive-primary solved {0 if naive_fail_all else 3}/3 — "
      f"{'coupling makes the wrong FIXED order worse than blind (the in-model P1/P4 inversion, realized)' if naive_fail_all and b_cert.any() else 'n/a'}")

# ================================================================ VERDICT (pre-registered)
print("\n" + "=" * 110)
foundation = n_adverse >= 2
forced = guided_ok and (naive_fail_all or (naive_certs >= 2.0 * guided_certs))
if not foundation:
    verdict = "NO-REAL-COUPLING"
elif guided_ok and forced:
    verdict = "SUPPORTED"
elif guided_ok and not forced:
    verdict = "REFUTED-HONEST"
else:
    verdict = "REFUTED"
print("VERDICT (pre-registered thresholds, real measurements only):")
print(f"  coupling foundation: {n_adverse} adverse measured entries ({adverse_names}) >= 2: {foundation}")
print(f"  endpoint exists (robust known-good): {kg_robust} (perf margin {kg_margin_t:.2f}x, acc margin {kg_margin_acc:.0f}x)")
print(f"  guided: certified {[g['certs'] for g in guided_runs]} certs (all 3: {guided_ok}); "
      f"ablation(no-matrix): certified={ablation['certified']} in {ablation['certs']}")
print(f"  naive-primary: {'FAILED to certify in all 3 repeats at certs=' + str([nv['certs'] for nv in naive_runs]) if naive_fail_all else f'certified in {naive_certs} certs (ratio {ratio:.2f}x)'}"
      f"{' [GRAZING]' if grazing_ratio else ''}")
print(f"  naive-lucky-order: certified={naive_lucky['certified']} in {naive_lucky['certs']} certs "
      f"(ordering variance: fixed orders span [{naive_lucky['certs'] if naive_lucky['certified'] else '-'} ... CAP-FAIL] "
      f"-> ordering is LOAD-BEARING under real coupling; guided derives the good order from measurement, not luck)")
print(f"  blind-random: {b_cert.sum()}/20 solved, mean {b_iters[b_cert].mean() if b_cert.any() else float('nan'):.1f} certs")
print(f"  total certs consumed: {CERT_COUNT['n']}  wall: {time.time()-T0_WALL:.1f}s")
print(f"\n  => ORDERING CAUSALLY BUYS ITERATIONS ON REAL COUPLED CUDA KNOBS: {verdict}")
print("=" * 110)

evidence = dict(
    meta=dict(script=os.path.basename(__file__), date=time.strftime("%Y-%m-%d"), device=DEV.name, sm=SM,
              torch=torch.__version__, cuda=torch.version.cuda, compile_s=T_COMPILE,
              shared_mem_per_block=SMEM_LIMIT, gpu_shared_contam_flag=True, wall_s=time.time() - T0_WALL,
              n_instantiations=len(inst), total_certs=CERT_COUNT['n']),
    spec=dict(n=N, K=K, n_hot=N_HOT, hot_frac=HOT_FRAC, d_terms=D_TERMS, scale=SCALE, tol_acc=TOL_ACC,
              grid=GRID, block=BLOCK, cap=CAP, theta_ms=THETA * 1e3, t_mem_ms=T_MEM * 1e3,
              peak_bw_gbs=dict(copy=bw_copy / 1e9, read=bw_read / 1e9, used=PEAK / 1e9)),
    known_good=dict(config=KNOWN_GOOD, robust=kg_robust, t_band_ms=[float(kg_t.min() * 1e3), float(kg_t.max() * 1e3)],
                    relerr_max=kg_rel, perf_margin=kg_margin_t, acc_margin=kg_margin_acc),
    coupling_matrix=MATRIX, coupling_probe_certs=PROBE_CERTS, n_adverse=n_adverse,
    cuobjdump_regs={str(k): v for k, v in REGS.items()},
    race=dict(weak_start=WEAK,
              guided=[dict(certified=g['certified'], certs=g['certs'], config=g['config'],
                           traj=[[c_, r_] for c_, r_ in g['traj']]) for g in guided_runs],
              ablation_no_matrix=dict(certified=ablation['certified'], certs=ablation['certs'],
                                      traj=[[c_, r_] for c_, r_ in ablation['traj']]),
              naive_primary=[dict(certified=nv['certified'], certs=nv['certs'], stuck_on=nv.get('stuck_on'),
                                  traj=[[c_, r_] for c_, r_ in nv['traj']]) for nv in naive_runs],
              naive_lucky=dict(certified=naive_lucky['certified'], certs=naive_lucky['certs'],
                               traj=[[c_, r_] for c_, r_ in naive_lucky['traj']]),
              blind=dict(solved=int(b_cert.sum()), of=20, certs=b_iters.tolist()),
              trap_measured=trap),
    qc=dict(same_mutation_set=True, grazing_flags=dict(kg_perf=bool(kg_margin_t < 1.3),
                                                       ratio_grazing=bool(grazing_ratio)),
            alt_accounting_guided_total=alt_guided_total),
    verdict=verdict)
os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
json.dump(evidence, open(ARTIFACT, "w"), indent=1, default=float)
print(f"evidence -> {ARTIFACT}")
sys.exit(0 if verdict == "SUPPORTED" else (2 if verdict == "REFUTED-HONEST" else 1))

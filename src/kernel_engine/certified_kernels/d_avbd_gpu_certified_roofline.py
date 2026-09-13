"""d_avbd_gpu_certified_roofline.py — BUILD-TARGET (§D ★KEY GAP): the FIRST at-roofline CERTIFIED non-fluid
DEFORMABLE (VBD/AVBD-class) GPU kernel, certified the way this project's LBM is (LBM: 94-99% of copy-roofline
at the 72 B/voxel floor). Real compiled CUDA (nvcc via load_inline, ONE compile unit).

REPRESENTATION (chosen so the R1 traffic floor derives cleanly from the stencil, like LBM's 72 B/voxel):
a REGULAR 3D mass-spring lattice (NX x NY x NZ vertices, 6-stencil structural springs, rest length L, per-edge
stiffness k, per-vertex mass m; top plane z==NZ-1 pinned, compiled-in like LBM's wall rows -> no mask read).
Vertex Block Descent (VBD, Chen et al. 2024 class): per vertex, ONE 3x3 Newton step on the incremental potential
  E(x) = sum_i m_i/(2 h^2) |x_i - y_i|^2 + sum_edges (k_e/2)(|x_i - x_j| - L)^2 ,  y_i = x + h v + h^2 g_ext,
with the standard PSD projection (tangential spring-Hessian coefficient clamped at 0), no line search (stated).
fp32 storage+arithmetic in-kernel; ALL correctness/convergence instruments are float64 on CPU (metric-can-be-
artifact discipline: the instrument must not share the kernel's precision).

VARIANTS (one nvcc compile unit; identical per-vertex math; ONLY the update schedule/data-path differs):
  (i)  colored_gs : red-black (6-stencil is bipartite) Gauss-Seidel VBD, COMPACT-BY-COLOR SoA storage
                    (x,y,z component planes per color -> every access unit-strided; parity-interleaved storage
                    would cap at 50% line utilization by construction and was rejected a-priori).
                    DETERMINISTIC BY CONSTRUCTION: within a pass each output is written by exactly one thread and
                    all reads are of the other color (disjoint) or the thread's own vertex -> no atomics, no races.
                    This is the 'AVBD-colored-det' candidate.
  (ii) jacobi     : full-parallel block-Jacobi VBD, SoA, ping-pong gather (deterministic too — pure gather; verified,
                    not assumed). Baseline.
  (iii) atomic    : naive per-edge scatter (atomicAdd of gradient + 6 Hessian components to both endpoints) then a
                    per-vertex solve — the weak start. Expected non-deterministic (float atomic ordering) + slower.
                    Diagnostic only; exempt from cert.

R1 TRAFFIC FLOORS — DERIVED A-PRIORI from the stencil (unique-bytes model, the LBM convention: count each
compulsory DRAM stream once; neighbor RE-reads must be absorbed by L1/L2 and any failure to absorb shows up as
measured B_eff amplification, cross-checked below):
  jacobi     : read x_in 12 B + read y 12 B + write x_out 12 B                     = 36 B/vertex/iteration.
               (6 neighbor re-reads of x_in are cache traffic: z-neighbor reuse distance = 3 planes x 3 components
                x NX*NY*4B = 2.4 MB at 256^2 planes << 48 MB L2 -> absorbed; stated, then measured.)
  colored_gs : per full sweep (=2 passes): pass over color C reads x_C (own, 12 B/updated vtx) + y_C (12) +
               x_{~C} neighbors (unique = all of the other color, 12 B/other vtx) and writes x_C (12).
               Per pass: 4 x 12 x N/2 = 24N  ->  48 B/vertex/full-sweep.  (GS pays +12 B/vertex/sweep over Jacobi
               because the opposite color is re-streamed each pass; it buys ~2x convergence/sweep -> pre-registered
               bytes-to-solution advantage ~ 36*2/48 = 1.5x, measured below.)
  atomic     : >= 132 B unique (36 zero-accum + 12 x + 36 accum re-read + 36 accum stream + 12 y + 12 write) with
               18 L2-RMW atomics/vertex uncounted — no clean floor; diagnostic, no cert claim.
Arithmetic intensity at the jacobi floor: ~250 flops / 36 B ~ 7 flop/B << RTX-5070 ridge (~45) -> memory-bound;
at-roofline is the correct target.

CERT-VECTOR (pre-registered BEFORE running):
  R1: B_model (= derived floor, auditable from kernel source) <= 1.2 x floor  [true by construction, flagged]
      AND achieved BW := floor x N x iters / t_min >= 70% of the FRESH-MEASURED copy roofline (~570 GB/s class,
      re-measured in-run as the denominator). >= 85% is reported as BEATING the pre-reg, not as the pre-reg.
      Cross-check: B_eff := t_min x BW_copy / (N x iters); B_eff/floor = 1/roofline-fraction. The STRICT reading
      (B_eff <= 1.2 x floor <=> >= 83.3% of roofline) is also reported per variant.
  R2: bit-repeat x3 from identical init (sha256 of raw position bytes), per variant. colored_gs MUST pass (its
      whole point); jacobi should pass (gather); atomic expected FAIL — all three MEASURED, none assumed.
  CONVERGENCE: per-sweep error-contraction factor rho for colored_gs vs jacobi on a homogeneous lattice, float64
      position-error-to-reference instrument. Pre-registered: rho_GS ~ rho_J^2 (Young's theorem for consistently-
      ordered/red-black SPD systems) => ~2x fewer sweeps for GS. Measured, incl. the exponent log(rho_GS)/log(rho_J).
  CORRECTNESS ANCHOR (EXTERNAL, per variant): static equilibrium of a hanging Hookean chain (2x1x64 ladder — two
      independent columns; x-rungs carry zero force when columns match) vs the ANALYTIC per-vertex solution
      z(j) = z_top - sum_{s=j}^{N-2} (L + (s+1) m g / k): max per-vertex error < 1% of max displacement.
      Plus a 3D block (16x16x64) against the same analytic per column + lateral-displacement-~0 check.
  ENERGY (dynamic step): free longitudinal oscillation of a pre-stretched block, converged implicit steps at
      h*omega_1 in {0.3, 0.1, 0.03}: energy drift per step MEASURED in float64 (implicit Euler is dissipative,
      NOT symplectic — the decay RATE is reported as measured, with its (h*omega)^2 scaling checked; no
      symplecticity is asserted).
  MASS-RATIO LEG (the inherited AVBD claim, now on real GPU): two-material lattice, pinned soft/light TOP half,
      heavy/stiff BOTTOM half hanging from it (the classic hard case), R in {1,10,100,1000,10000} for BOTH
      (A) stiffness ratio (k x R below the midplane, uniform m) and (B) mass ratio (m x R, uniform k).
      Instrument: sweeps-to-tolerance ||x - x*||/||x0 - x*|| < 1e-3 (float64, reference x* from a long converged
      run; GS/Jacobi reference agreement reported). Jacobi gets an under-relaxation sweep {1.0, 0.7, 0.5}, best
      taken (divergence = inf). PRE-REGISTERED DIRECTION (per task): iters_J/iters_GS GROWS with R.
      A-PRIORI COUNTER-PREDICTION recorded for symmetric QC: Young's theorem says for the red-black-ordered SPD
      system the asymptotic ratio is EXACTLY 2 regardless of conditioning — so the growth, if any, must come from
      non-asymptotic transients / omega-damping / nonlinearity. The measurement decides; either outcome is reported
      against both predictions, with absolute sweep counts (which MUST grow with R) alongside.

CERTIFIED (the §D gap-closing claim) iff: best of {jacobi, colored_gs} clears the R1 bar above AND that variant's
ladder anchor < 1% AND its R2 bit-repeat x3 passes. Gap localization on miss: block-size sweep {64,128,256,512}
(launch-error-gated — never trust a timing without one) + grid-size L2-residency probe. Primary cert grid 256^3
(=16.8M vertices; jacobi working set 604 MB, colored_gs 402 MB >> 48 MB L2 -> pure DRAM regime; sub-L2 points
would inflate and are not used for cert).

GPU IS SHARED — the run is invoked under flock .cache/gpu.lock (compile pass is done
un-flocked first; the build cache makes the flocked run start instantly). No fast-math (IEEE sqrt/div — the
fp32 gradient noise floor is computed in-comments and respected by the instruments). NO run_in_background.

AMENDMENTS RECORDED AFTER RUN 1 / BEFORE RUN 2 (symmetric QC on run 1's own green result; run 1: colored_gs
105.7% / jacobi 100.9% of copy roofline, all anchors + R2 as pre-registered):
  A1  >100%-of-COPY readings are a red flag until the mechanism is MEASURED (L2 inflation is excluded: working
      sets 604/402 MB >> 48 MB L2). Hypothesis: the copy kernel is a 1:1 read:write mix, while jacobi moves 24R:12W
      (2:1) and a colored-GS pass moves 18R:6W (3:1) — read-heavier DRAM mixes can sustain more GB/s than 1:1 (fewer
      write-turnarounds). Run 2 adds MIX-MATCHED streaming denominators to the same compile unit (mix21: o=a+b,
      mix31: o=a+b+c) and reports each variant's fraction of BOTH the copy roof (the pre-registered cert bar,
      unchanged) and its mix-matched roof (the mechanism check: if the mix hypothesis is right, mix-matched
      fractions should fall to <= ~100%).
  A2  [5] rho_GS came back nan in run 1: GS crossed the whole fit band between residual checks (converged in 40
      sweeps, checks every 10). Instrument granularity, not physics — check_every 10 -> 2 in [5].
  A3  [6] run 1's direction-verdict line contradicted its own data: at Rk=1e3 Jacobi CAP-HIT (>=60000 sweeps) vs
      GS 23880 is a LOWER BOUND ratio >= 2.51 (GROWING advantage), but the verdict used finite ratios only and
      printed 'NOT SUPPORTED'. Run 2: conv_run returns a status (tol/cap/div), cap-hits enter the verdict as lower
      bounds, Rk=1e3 gets an extended cap (240k) to try to close the bound, and rows with reference agreement
      > 1e-2 (Rk=1e4: ref-agree 0.41 — beyond the fp32-reliable envelope) are flagged unreliable and excluded,
      reported as the honest envelope edge rather than data.
"""
import hashlib
import json
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------------------------
# ONE nvcc COMPILE UNIT
# ---------------------------------------------------------------------------------------------
CUDA_SRC = r'''
#include <torch/extension.h>

#define LAUNCH_CHECK() do { cudaError_t e_ = cudaGetLastError(); \
  TORCH_CHECK(e_ == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(e_)); } while(0)

// per-vertex mass weight w = (m/h^2) * (Rm if z < NZhalf)   |   per-edge stiffness k = k0 * (Rk if zmax < NZhalf)
__device__ __forceinline__ float wgt(int z, float w0, float Rm, int NZhalf){ return (z < NZhalf) ? w0*Rm : w0; }
__device__ __forceinline__ float ek(int zmax, float k0, float Rk, int NZhalf){ return (zmax < NZhalf) ? k0*Rk : k0; }

// accumulate spring grad + projected Hessian for one edge (xi - xj = d). Exact gradient, PSD-projected Hessian.
__device__ __forceinline__ void spring_acc(float dx, float dy, float dz, float k, float L,
    float& gx, float& gy, float& gz, float& h00, float& h01, float& h02, float& h11, float& h12, float& h22){
  float len2 = dx*dx + dy*dy + dz*dz;
  float len  = sqrtf(len2);
  float s    = 1.f - L/len;                    // (1 - L/|d|), true gradient coefficient
  gx += k*s*dx; gy += k*s*dy; gz += k*s*dz;
  float t = fmaxf(s, 0.f);                     // PSD projection of tangential term
  float a = k*(1.f - t)/len2;                  // H = k[ t I + (1-t) d d^T / |d|^2 ]
  h00 += k*t + a*dx*dx; h11 += k*t + a*dy*dy; h22 += k*t + a*dz*dz;
  h01 += a*dx*dy; h02 += a*dx*dz; h12 += a*dy*dz;
}

// solve H delta = -g, 3x3 symmetric, Cramer/adjugate (H >= w I > 0 so det > 0)
__device__ __forceinline__ void solve3(float gx, float gy, float gz,
    float h00, float h01, float h02, float h11, float h12, float h22, float& dx, float& dy, float& dz){
  float c00 = h11*h22 - h12*h12;
  float c01 = h02*h12 - h01*h22;
  float c02 = h01*h12 - h02*h11;
  float det = h00*c00 + h01*c01 + h02*c02;
  float c11 = h00*h22 - h02*h02;
  float c12 = h01*h02 - h00*h12;
  float c22 = h00*h11 - h01*h01;
  float inv = 1.f/det;
  dx = -(c00*gx + c01*gy + c02*gz)*inv;
  dy = -(c01*gx + c11*gy + c12*gz)*inv;
  dz = -(c02*gx + c12*gy + c22*gz)*inv;
}

// ---------------- (ii) JACOBI: full SoA [3][N], gather from X, write XOUT (ping-pong) ----------------
__global__ void k_jacobi(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ XO,
    float w0, float k0, float L, float om, float Rm, float Rk, int NX, int NY, int NZ, int NZhalf){
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  int N = NX*NY*NZ; if(c >= N) return;
  int x = c % NX; int r = c / NX; int y = r % NY; int z = r / NY;
  float xi = X[c], yi = X[N+c], zi = X[2*N+c];
  if(z == NZ-1){ XO[c]=xi; XO[N+c]=yi; XO[2*N+c]=zi; return; }        // pinned plane, copy-through
  float w = wgt(z, w0, Rm, NZhalf);
  float gx = w*(xi - Y[c]), gy = w*(yi - Y[N+c]), gz = w*(zi - Y[2*N+c]);
  float h00=w, h11=w, h22=w, h01=0.f, h02=0.f, h12=0.f;
  #pragma unroll
  for(int e=0;e<6;e++){
    int nc; int zmax = z;
    if     (e==0){ if(x==0)    continue; nc = c-1; }
    else if(e==1){ if(x==NX-1) continue; nc = c+1; }
    else if(e==2){ if(y==0)    continue; nc = c-NX; }
    else if(e==3){ if(y==NY-1) continue; nc = c+NX; }
    else if(e==4){ if(z==0)    continue; nc = c-NX*NY; }
    else         { if(z==NZ-1) continue; nc = c+NX*NY; zmax = z+1; }
    float k = ek(zmax, k0, Rk, NZhalf);
    spring_acc(xi-X[nc], yi-X[N+nc], zi-X[2*N+nc], k, L, gx,gy,gz, h00,h01,h02,h11,h12,h22);
  }
  float dx,dy,dz; solve3(gx,gy,gz,h00,h01,h02,h11,h12,h22,dx,dy,dz);
  XO[c] = xi + om*dx; XO[N+c] = yi + om*dy; XO[2*N+c] = zi + om*dz;
}

// ---------------- (i) COLORED-GS: compact-by-color SoA [3][N/2] per color, in-place ----------------
// vertex recovery: thread t of color col -> pair (2t, 2t+1) lies in one row (NX even); pick the member of color col.
__global__ void k_gs(float* __restrict__ XC, const float* __restrict__ XN, const float* __restrict__ YC,
    int col, float w0, float k0, float L, float om, float Rm, float Rk, int NX, int NY, int NZ, int NZhalf){
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  int N = NX*NY*NZ; int Nh = N>>1; if(t >= Nh) return;
  int c2 = 2*t;
  int x0 = c2 % NX; int r = c2 / NX; int y = r % NY; int z = r / NY;
  int col0 = (x0 + y + z) & 1;
  int c = c2 + ((col0 == col) ? 0 : 1);
  int x = c % NX;
  if(z == NZ-1) return;                                                // pinned plane (no write, no read needed)
  float xi = XC[t], yi = XC[Nh+t], zi = XC[2*Nh+t];
  float w = wgt(z, w0, Rm, NZhalf);
  float gx = w*(xi - YC[t]), gy = w*(yi - YC[Nh+t]), gz = w*(zi - YC[2*Nh+t]);
  float h00=w, h11=w, h22=w, h01=0.f, h02=0.f, h12=0.f;
  #pragma unroll
  for(int e=0;e<6;e++){
    int nc; int zmax = z;
    if     (e==0){ if(x==0)    continue; nc = c-1; }
    else if(e==1){ if(x==NX-1) continue; nc = c+1; }
    else if(e==2){ if(y==0)    continue; nc = c-NX; }
    else if(e==3){ if(y==NY-1) continue; nc = c+NX; }
    else if(e==4){ if(z==0)    continue; nc = c-NX*NY; }
    else         { if(z==NZ-1) continue; nc = c+NX*NY; zmax = z+1; }
    int nt = nc >> 1;                                                  // other-color compact index
    float k = ek(zmax, k0, Rk, NZhalf);
    spring_acc(xi-XN[nt], yi-XN[Nh+nt], zi-XN[2*Nh+nt], k, L, gx,gy,gz, h00,h01,h02,h11,h12,h22);
  }
  float dx,dy,dz; solve3(gx,gy,gz,h00,h01,h02,h11,h12,h22,dx,dy,dz);
  XC[t] = xi + om*dx; XC[Nh+t] = yi + om*dy; XC[2*Nh+t] = zi + om*dz;
}

// ---------------- (iii) ATOMIC scatter (the weak start): per-edge scatter -> per-vertex solve ----------------
__global__ void k_edge_scatter(const float* __restrict__ X, float* __restrict__ A,
    float k0, float L, float Rk, int NX, int NY, int NZ, int NZhalf){
  int e = blockIdx.x*blockDim.x + threadIdx.x;
  int N = NX*NY*NZ; if(e >= 3*N) return;
  int dir = e / N; int c = e % N;
  int x = c % NX; int r = c / NX; int y = r % NY; int z = r / NY;
  int nc; int zmax = z;
  if     (dir==0){ if(x==NX-1) return; nc = c+1; }
  else if(dir==1){ if(y==NY-1) return; nc = c+NX; }
  else           { if(z==NZ-1) return; nc = c+NX*NY; zmax = z+1; }
  float k = ek(zmax, k0, Rk, NZhalf);
  float gx=0,gy=0,gz=0,h00=0,h01=0,h02=0,h11=0,h12=0,h22=0;
  spring_acc(X[c]-X[nc], X[N+c]-X[N+nc], X[2*N+c]-X[2*N+nc], k, L, gx,gy,gz, h00,h01,h02,h11,h12,h22);
  atomicAdd(&A[c],       gx); atomicAdd(&A[N+c],     gy); atomicAdd(&A[2*N+c],   gz);
  atomicAdd(&A[nc],     -gx); atomicAdd(&A[N+nc],   -gy); atomicAdd(&A[2*N+nc], -gz);
  atomicAdd(&A[3*N+c],  h00); atomicAdd(&A[4*N+c],  h01); atomicAdd(&A[5*N+c],  h02);
  atomicAdd(&A[6*N+c],  h11); atomicAdd(&A[7*N+c],  h12); atomicAdd(&A[8*N+c],  h22);
  atomicAdd(&A[3*N+nc], h00); atomicAdd(&A[4*N+nc], h01); atomicAdd(&A[5*N+nc], h02);
  atomicAdd(&A[6*N+nc], h11); atomicAdd(&A[7*N+nc], h12); atomicAdd(&A[8*N+nc], h22);
}
__global__ void k_vertex_update(const float* __restrict__ X, const float* __restrict__ Y,
    const float* __restrict__ A, float* __restrict__ XO,
    float w0, float om, float Rm, int NX, int NY, int NZ, int NZhalf){
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  int N = NX*NY*NZ; if(c >= N) return;
  int z = c / (NX*NY);
  float xi = X[c], yi = X[N+c], zi = X[2*N+c];
  if(z == NZ-1){ XO[c]=xi; XO[N+c]=yi; XO[2*N+c]=zi; return; }
  float w = wgt(z, w0, Rm, NZhalf);
  float gx = A[c]     + w*(xi - Y[c]);
  float gy = A[N+c]   + w*(yi - Y[N+c]);
  float gz = A[2*N+c] + w*(zi - Y[2*N+c]);
  float dx,dy,dz;
  solve3(gx,gy,gz, A[3*N+c]+w, A[4*N+c], A[5*N+c], A[6*N+c]+w, A[7*N+c], A[8*N+c]+w, dx,dy,dz);
  XO[c] = xi + om*dx; XO[N+c] = yi + om*dy; XO[2*N+c] = zi + om*dz;
}

// ---------------- gradient instruments (full + color layouts). mode: 1 = incremental w(x-y)+dEs, 0 = static dEs - m g ----------------
__global__ void k_grad_full(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ G,
    int mode, float w0, float k0, float L, float fz0, float Rm, float Rk, int NX, int NY, int NZ, int NZhalf){
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  int N = NX*NY*NZ; if(c >= N) return;
  int x = c % NX; int r = c / NX; int y = r % NY; int z = r / NY;
  if(z == NZ-1){ G[c]=0.f; G[N+c]=0.f; G[2*N+c]=0.f; return; }
  float xi = X[c], yi = X[N+c], zi = X[2*N+c];
  float mfac = (z < NZhalf) ? Rm : 1.f;
  float gx, gy, gz;
  if(mode==1){ float w = w0*mfac; gx = w*(xi-Y[c]); gy = w*(yi-Y[N+c]); gz = w*(zi-Y[2*N+c]); }
  else       { gx = 0.f; gy = 0.f; gz = -fz0*mfac; }
  float h00=0,h01=0,h02=0,h11=0,h12=0,h22=0;
  #pragma unroll
  for(int e=0;e<6;e++){
    int nc; int zmax = z;
    if     (e==0){ if(x==0)    continue; nc = c-1; }
    else if(e==1){ if(x==NX-1) continue; nc = c+1; }
    else if(e==2){ if(y==0)    continue; nc = c-NX; }
    else if(e==3){ if(y==NY-1) continue; nc = c+NX; }
    else if(e==4){ if(z==0)    continue; nc = c-NX*NY; }
    else         { if(z==NZ-1) continue; nc = c+NX*NY; zmax = z+1; }
    float k = ek(zmax, k0, Rk, NZhalf);
    spring_acc(xi-X[nc], yi-X[N+nc], zi-X[2*N+nc], k, L, gx,gy,gz, h00,h01,h02,h11,h12,h22);
  }
  G[c] = gx; G[N+c] = gy; G[2*N+c] = gz;
}

// ---------------- copy-roofline kernel (float4 grid-stride) — the cert denominator ----------------
__global__ void k_copy_f4(const float4* __restrict__ in, float4* __restrict__ out, long n4){
  long stride = (long)blockDim.x*gridDim.x;
  for(long j = (long)blockIdx.x*blockDim.x + threadIdx.x; j<n4; j+=stride) out[j] = in[j];
}
// mix-matched streaming denominators (amendment A1): 2 reads : 1 write, and 3 reads : 1 write
__global__ void k_mix21_f4(const float4* __restrict__ a, const float4* __restrict__ b, float4* __restrict__ o, long n4){
  long stride = (long)blockDim.x*gridDim.x;
  for(long j = (long)blockIdx.x*blockDim.x + threadIdx.x; j<n4; j+=stride){
    float4 x = a[j], y = b[j];
    o[j] = make_float4(x.x+y.x, x.y+y.y, x.z+y.z, x.w+y.w);
  }
}
__global__ void k_mix31_f4(const float4* __restrict__ a, const float4* __restrict__ b, const float4* __restrict__ c,
                           float4* __restrict__ o, long n4){
  long stride = (long)blockDim.x*gridDim.x;
  for(long j = (long)blockIdx.x*blockDim.x + threadIdx.x; j<n4; j+=stride){
    float4 x = a[j], y = b[j], z = c[j];
    o[j] = make_float4(x.x+y.x+z.x, x.y+y.y+z.y, x.z+y.z+z.z, x.w+y.w+z.w);
  }
}

// ---------------- runners: whole iteration loop in C++ (no per-iter Python dispatch in timed region) ----------------
void run_jacobi(torch::Tensor X, torch::Tensor Y, torch::Tensor XO, int64_t iters,
    double w0, double k0, double L, double om, double Rm, double Rk,
    int64_t NX, int64_t NY, int64_t NZ, int64_t NZhalf, int64_t block){
  TORCH_CHECK(iters%2==0, "iters must be even (result lands back in X)");
  float* a = X.data_ptr<float>(); float* o = XO.data_ptr<float>(); const float* y = Y.data_ptr<float>();
  long N = NX*NY*NZ; long grid = (N+block-1)/block;
  for(long s=0;s<iters;s++){
    k_jacobi<<<(int)grid,(int)block>>>(a, y, o, (float)w0,(float)k0,(float)L,(float)om,(float)Rm,(float)Rk,
                                       (int)NX,(int)NY,(int)NZ,(int)NZhalf);
    float* t=a; a=o; o=t;
  }
  LAUNCH_CHECK();
}
void run_gs(torch::Tensor X0, torch::Tensor X1, torch::Tensor Y0, torch::Tensor Y1, int64_t sweeps,
    double w0, double k0, double L, double om, double Rm, double Rk,
    int64_t NX, int64_t NY, int64_t NZ, int64_t NZhalf, int64_t block){
  float* x0 = X0.data_ptr<float>(); float* x1 = X1.data_ptr<float>();
  const float* y0 = Y0.data_ptr<float>(); const float* y1 = Y1.data_ptr<float>();
  long Nh = (NX*NY*NZ)>>1; long grid = (Nh+block-1)/block;
  for(long s=0;s<sweeps;s++){
    k_gs<<<(int)grid,(int)block>>>(x0, x1, y0, 0, (float)w0,(float)k0,(float)L,(float)om,(float)Rm,(float)Rk,
                                   (int)NX,(int)NY,(int)NZ,(int)NZhalf);
    k_gs<<<(int)grid,(int)block>>>(x1, x0, y1, 1, (float)w0,(float)k0,(float)L,(float)om,(float)Rm,(float)Rk,
                                   (int)NX,(int)NY,(int)NZ,(int)NZhalf);
  }
  LAUNCH_CHECK();
}
void run_atomic(torch::Tensor X, torch::Tensor Y, torch::Tensor XO, torch::Tensor A, int64_t iters,
    double w0, double k0, double L, double om, double Rm, double Rk,
    int64_t NX, int64_t NY, int64_t NZ, int64_t NZhalf, int64_t block){
  TORCH_CHECK(iters%2==0, "iters must be even");
  float* a = X.data_ptr<float>(); float* o = XO.data_ptr<float>();
  const float* y = Y.data_ptr<float>(); float* acc = A.data_ptr<float>();
  long N = NX*NY*NZ; long gridE = (3*N+block-1)/block; long gridV = (N+block-1)/block;
  for(long s=0;s<iters;s++){
    cudaMemsetAsync(acc, 0, sizeof(float)*9*N);
    k_edge_scatter<<<(int)gridE,(int)block>>>(a, acc, (float)k0,(float)L,(float)Rk,(int)NX,(int)NY,(int)NZ,(int)NZhalf);
    k_vertex_update<<<(int)gridV,(int)block>>>(a, y, acc, o, (float)w0,(float)om,(float)Rm,(int)NX,(int)NY,(int)NZ,(int)NZhalf);
    float* t=a; a=o; o=t;
  }
  LAUNCH_CHECK();
}
void run_grad_full(torch::Tensor X, torch::Tensor Y, torch::Tensor G, int64_t mode,
    double w0, double k0, double L, double fz0, double Rm, double Rk,
    int64_t NX, int64_t NY, int64_t NZ, int64_t NZhalf, int64_t block){
  long N = NX*NY*NZ; long grid = (N+block-1)/block;
  k_grad_full<<<(int)grid,(int)block>>>(X.data_ptr<float>(), Y.data_ptr<float>(), G.data_ptr<float>(),
      (int)mode, (float)w0,(float)k0,(float)L,(float)fz0,(float)Rm,(float)Rk,(int)NX,(int)NY,(int)NZ,(int)NZhalf);
  LAUNCH_CHECK();
}
void run_copy(torch::Tensor a, torch::Tensor b, int64_t block, int64_t grid){
  long n4 = a.numel()/4;
  k_copy_f4<<<(int)grid,(int)block>>>((const float4*)a.data_ptr<float>(), (float4*)b.data_ptr<float>(), n4);
  LAUNCH_CHECK();
}
void run_mix21(torch::Tensor a, torch::Tensor b, torch::Tensor o, int64_t block, int64_t grid){
  long n4 = a.numel()/4;
  k_mix21_f4<<<(int)grid,(int)block>>>((const float4*)a.data_ptr<float>(), (const float4*)b.data_ptr<float>(),
                                       (float4*)o.data_ptr<float>(), n4);
  LAUNCH_CHECK();
}
void run_mix31(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor o, int64_t block, int64_t grid){
  long n4 = a.numel()/4;
  k_mix31_f4<<<(int)grid,(int)block>>>((const float4*)a.data_ptr<float>(), (const float4*)b.data_ptr<float>(),
                                       (const float4*)c.data_ptr<float>(), (float4*)o.data_ptr<float>(), n4);
  LAUNCH_CHECK();
}
'''
CPP_DECL = "\n".join(f"void {f}({sig});" for f, sig in [
    ("run_jacobi", "torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, int64_t, int64_t, int64_t, int64_t, int64_t"),
    ("run_gs", "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, int64_t, int64_t, int64_t, int64_t, int64_t"),
    ("run_atomic", "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, int64_t, int64_t, int64_t, int64_t, int64_t"),
    ("run_grad_full", "torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, double, double, double, double, double, int64_t, int64_t, int64_t, int64_t, int64_t"),
    ("run_copy", "torch::Tensor, torch::Tensor, int64_t, int64_t"),
    ("run_mix21", "torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t"),
    ("run_mix31", "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t"),
])

import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

print("compiling ONE nvcc compile unit (colored-GS + Jacobi + atomic VBD variants + grad instrument + copy kernel)...",
      flush=True)
_t0 = time.time()
M = load_inline(name="d_avbd_gpu_cert", cpp_sources=[CPP_DECL], cuda_sources=[CUDA_SRC],
                functions=["run_jacobi", "run_gs", "run_atomic", "run_grad_full", "run_copy", "run_mix21", "run_mix31"],
                extra_cuda_cflags=["-O3"], verbose=False)
print(f"compiled OK in {time.time()-_t0:.1f}s (real nvcc).\n")
if "--compile-only" in sys.argv:
    print("compile-only pass done (build cached); exiting before touching the GPU.")
    sys.exit(0)

assert torch.cuda.is_available()
T_START = time.time()
DEV = torch.cuda.get_device_properties(0)
SMI = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(f"=== device={DEV.name} SMs={DEV.multi_processor_count} torch={torch.__version__} cuda={torch.version.cuda} ===")
print(f"nvidia-smi @ start: {SMI}")
print("CONTAM-FLAG: GPU shared (desktop resident). flock held for this whole run; roofline % may carry a small "
      "negative bias vs fully-idle — flagged, not corrected.\n")

# ---------------- PRE-REGISTERED CONSTANTS ----------------
FLOOR_J = 36.0          # B/vertex/iter, derived (header)
FLOOR_GS = 48.0         # B/vertex/full-sweep, derived (header)
TRAFFIC_MULT = 1.2
ROOF_CERT = 0.70        # pre-registered cert bar
ROOF_BEAT = 0.85        # "beats the pre-reg" bar
ANCHOR_TOL_PCT = 1.0
CONV_TOL = 1e-3         # position-error-to-reference tolerance (float64 instrument)
BW_MEAS_ANCHOR = 578.0  # GB/s, prior D-measured copy roofline — re-measured fresh below

GVEC = -10.0            # gravity g_z


# =================================================================================================
# lattice helpers (positions [3, N] SoA; color split by (x+y+z)&1, compact order = linear-index order)
# =================================================================================================
def lattice(NX, NY, NZ, L=1.0):
    z, y, x = np.meshgrid(np.arange(NZ), np.arange(NY), np.arange(NX), indexing="ij")
    P = np.stack([x, y, z]).reshape(3, -1).astype(np.float32) * np.float32(L)
    return P


def color_idx(NX, NY, NZ):
    z, y, x = np.meshgrid(np.arange(NZ), np.arange(NY), np.arange(NX), indexing="ij")
    col = ((x + y + z) & 1).reshape(-1)
    return np.where(col == 0)[0], np.where(col == 1)[0]


def split(P, i0, i1):
    return (torch.from_numpy(np.ascontiguousarray(P[:, i0])).cuda(),
            torch.from_numpy(np.ascontiguousarray(P[:, i1])).cuda())


def join(X0, X1, i0, i1, N):
    F = np.empty((3, N), np.float64)
    F[:, i0] = X0.cpu().numpy(); F[:, i1] = X1.cpu().numpy()
    return F


def np_grad(F, dims, k0, Rk, NZhalf, m0, Rm, mode, Y=None, w0=None, L=1.0, gz=GVEC):
    """float64 residual instrument. F: [3,N] float64. mode 'static': dEs - m*g ; 'incr': w(x-y)+dEs. Pinned rows -> 0."""
    NX, NY, NZ = dims
    P = F.reshape(3, NZ, NY, NX)
    G = np.zeros_like(P)
    zidx = np.arange(NZ)
    for ax, zmax_off in ((3, 0), (2, 0), (1, 1)):           # x, y, z axes of P[c, z, y, x]
        d = np.diff(P, axis=ax)                              # upper - lower
        ln = np.sqrt((d * d).sum(0))
        s = 1.0 - L / ln
        if ax == 1:
            zmax = zidx[1:NZ]
            ke = np.where(zmax < NZhalf, k0 * Rk, k0)[:, None, None]
        else:
            ke = np.where(zidx < NZhalf, k0 * Rk, k0)[:, None, None]
        f = ke * s * d
        sl_hi = [slice(None)] * 4; sl_hi[ax] = slice(1, None)
        sl_lo = [slice(None)] * 4; sl_lo[ax] = slice(None, -1)
        G[tuple(sl_hi)] += f
        G[tuple(sl_lo)] -= f
    mfac = np.where(zidx < NZhalf, Rm, 1.0)[:, None, None]
    if mode == "static":
        G[2] -= (m0 * mfac) * gz
    else:
        Yp = Y.reshape(3, NZ, NY, NX)
        G += (w0 * mfac)[None] * (P - Yp)
    G[:, NZ - 1] = 0.0
    return G.reshape(3, -1)


def np_spring_energy(F, dims, k0, Rk, NZhalf, L=1.0):
    NX, NY, NZ = dims
    P = F.reshape(3, NZ, NY, NX)
    zidx = np.arange(NZ)
    E = 0.0
    for ax in (3, 2, 1):
        d = np.diff(P, axis=ax)
        ln = np.sqrt((d * d).sum(0))
        if ax == 1:
            ke = np.where(zidx[1:NZ] < NZhalf, k0 * Rk, k0)[:, None, None]
        else:
            ke = np.where(zidx < NZhalf, k0 * Rk, k0)[:, None, None]
        E += float((0.5 * ke * (ln - L) ** 2).sum())
    return E


def sha(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()


EV = dict(script="d_avbd_gpu_certified_roofline.py", date="",
          device=dict(name=DEV.name, SMs=DEV.multi_processor_count, torch=torch.__version__, cuda=torch.version.cuda),
          preregistration=dict(floor_jacobi_B_per_vtx_iter=FLOOR_J, floor_gs_B_per_vtx_sweep=FLOOR_GS,
                               traffic_mult=TRAFFIC_MULT, roof_cert=ROOF_CERT, roof_beat=ROOF_BEAT,
                               anchor_tol_pct=ANCHOR_TOL_PCT,
                               convergence="rho_GS ~ rho_J^2 (Young, red-black consistently-ordered SPD) => ~2x/sweep",
                               massratio_direction="task pre-reg: iters_J/iters_GS grows with R; "
                                                   "a-priori counter-prediction (Young): asymptotic ratio == 2 flat"))

# =================================================================================================
# [0] MEASURED copy roofline (fresh) — the cert denominator
# =================================================================================================
print("=" * 100)
print("[0] MEASURED copy roofline (float4 grid-stride, 134MB buffer, r+w)")
print("=" * 100)
NCOPY = 1 << 25
ca = torch.rand(NCOPY, device="cuda"); cb = torch.zeros_like(ca)
grid_copy = DEV.multi_processor_count * 32
for _ in range(5):
    M.run_copy(ca, cb, 256, grid_copy)
torch.cuda.synchronize()
cts = []
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    M.run_copy(ca, cb, 256, grid_copy)
    torch.cuda.synchronize(); cts.append(time.perf_counter() - t0)
BW_COPY = 2 * 4 * NCOPY / min(cts) / 1e9
assert torch.equal(ca, cb)
print(f"  copy BW (min of 10): {BW_COPY:.1f} GB/s | prior 578 anchor ratio {BW_COPY/BW_MEAS_ANCHOR:.3f}")
# amendment A1: mix-matched denominators (2R:1W and 3R:1W streaming)
cc = torch.rand(NCOPY, device="cuda"); cd = torch.zeros_like(ca)
for _ in range(5):
    M.run_mix21(ca, cc, cd, 256, grid_copy)
torch.cuda.synchronize()
ts21 = []
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    M.run_mix21(ca, cc, cd, 256, grid_copy)
    torch.cuda.synchronize(); ts21.append(time.perf_counter() - t0)
BW_MIX21 = 3 * 4 * NCOPY / min(ts21) / 1e9
for _ in range(5):
    M.run_mix31(ca, cb, cc, cd, 256, grid_copy)
torch.cuda.synchronize()
ts31 = []
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    M.run_mix31(ca, cb, cc, cd, 256, grid_copy)
    torch.cuda.synchronize(); ts31.append(time.perf_counter() - t0)
BW_MIX31 = 4 * 4 * NCOPY / min(ts31) / 1e9
print(f"  mix-matched roofs (A1): 2R:1W = {BW_MIX21:.1f} GB/s ({BW_MIX21/BW_COPY*100:.1f}% of copy) | "
      f"3R:1W = {BW_MIX31:.1f} GB/s ({BW_MIX31/BW_COPY*100:.1f}% of copy)")
EV["copy_roofline"] = dict(measured_GBs=BW_COPY, anchor_578_ratio=BW_COPY / BW_MEAS_ANCHOR,
                           mix21_GBs=BW_MIX21, mix31_GBs=BW_MIX31)
del ca, cb, cc, cd
torch.cuda.empty_cache()


# =================================================================================================
# [1] EXTERNAL ANCHOR — hanging Hookean ladder (2x1x64) vs analytic, per variant
# =================================================================================================
print("\n" + "=" * 100)
print("[1] EXTERNAL anchor: hanging ladder 2x1x64, static equilibrium vs analytic Hookean chain (<1%)")
print("=" * 100)
AN = dict(NX=2, NY=1, NZ=64, L=1.0, k0=2000.0, m0=1.0)
dimsA = (AN["NX"], AN["NY"], AN["NZ"]); NA = 2 * 1 * 64
w0A = 0.01 * AN["k0"]                       # proximal weight m/h^2; h^2 = m0/w0
h2A = AN["m0"] / w0A
NZc = AN["NZ"]
# analytic (float64): z(j) = z_top - sum_{s=j}^{NZ-2} (L + (s+1) m g / k)
lenz = AN["L"] + (np.arange(1, NZc) * AN["m0"] * (-GVEC)) / AN["k0"]     # len of spring s (s=0..NZ-2), tension (s+1)mg
z_ana = np.zeros(NZc)
z_ana[NZc - 1] = (NZc - 1) * AN["L"]
for j in range(NZc - 2, -1, -1):
    z_ana[j] = z_ana[j + 1] - lenz[j]
max_disp = np.max(np.abs(z_ana - np.arange(NZc) * AN["L"]))
i0A, i1A = color_idx(*dimsA)


def statics_solve(runner_name, dims, params, w0, sweeps_inner=400, cap_outer=3000, tol=3e-4, omega=1.0):
    """proximal statics: repeat { y = x + h^2 g ; inner solve } until float64 static residual < tol."""
    NX, NY, NZ = dims
    N = NX * NY * NZ
    P0 = lattice(NX, NY, NZ, params["L"])
    h2 = params["m0"] / w0
    if runner_name == "gs":
        i0, i1 = color_idx(NX, NY, NZ)
        X0, X1 = split(P0, i0, i1)
        Y0, Y1 = X0.clone(), X1.clone()
        for it in range(cap_outer):
            Y0.copy_(X0); Y1.copy_(X1); Y0[2] += h2 * GVEC; Y1[2] += h2 * GVEC
            M.run_gs(X0, X1, Y0, Y1, sweeps_inner, w0, params["k0"], params["L"], omega, 1.0, 1.0,
                     NX, NY, NZ, 0, 128)
            if (it + 1) % 25 == 0 or it == cap_outer - 1:
                torch.cuda.synchronize()
                F = join(X0, X1, i0, i1, N)
                r = np_grad(F, dims, params["k0"], 1.0, 0, params["m0"], 1.0, "static", L=params["L"])
                rel = np.linalg.norm(r) / (params["m0"] * abs(GVEC) * np.sqrt(N))
                if rel < tol:
                    return F, rel, (it + 1) * sweeps_inner
        return F, rel, cap_outer * sweeps_inner
    else:
        X = torch.from_numpy(P0.copy()).cuda(); Y = X.clone(); XO = torch.empty_like(X)
        A = torch.zeros(9, N, device="cuda") if runner_name == "atomic" else None
        for it in range(cap_outer):
            Y.copy_(X); Y[2] += h2 * GVEC
            if runner_name == "jacobi":
                M.run_jacobi(X, Y, XO, sweeps_inner, w0, params["k0"], params["L"], omega, 1.0, 1.0, NX, NY, NZ, 0, 128)
            else:
                M.run_atomic(X, Y, XO, A, sweeps_inner, w0, params["k0"], params["L"], omega, 1.0, 1.0, NX, NY, NZ, 0, 128)
            if (it + 1) % 25 == 0 or it == cap_outer - 1:
                torch.cuda.synchronize()
                F = X.cpu().numpy().astype(np.float64)
                r = np_grad(F, dims, params["k0"], 1.0, 0, params["m0"], 1.0, "static", L=params["L"])
                rel = np.linalg.norm(r) / (params["m0"] * abs(GVEC) * np.sqrt(N))
                if rel < tol:
                    return F, rel, (it + 1) * sweeps_inner
        return F, rel, cap_outer * sweeps_inner


EV["anchor_ladder"] = {}
for vn in ("gs", "jacobi", "atomic"):
    F, rel, sw = statics_solve(vn, dimsA, AN, w0A, sweeps_inner=400, cap_outer=3000, tol=3e-4)
    zc = F[2].reshape(NZc, 1, 2)
    err = np.abs(zc - z_ana[:, None, None]).max()
    lat = max(np.abs(F[0].reshape(NZc, 1, 2) - np.array([0.0, 1.0])[None, None, :]).max(), np.abs(F[1]).max())
    errpct = 100 * err / max_disp
    ok = errpct < ANCHOR_TOL_PCT
    print(f"  {vn:8s}: static-res(rel,f64)={rel:.2e} after {sw} sweeps | max |z-z_ana| = {err:.4f} "
          f"({errpct:.3f}% of max-disp {max_disp:.2f}) | lateral max {lat:.2e} | {'PASS' if ok else 'FAIL'}")
    EV["anchor_ladder"][vn] = dict(static_rel_res=rel, sweeps=sw, err_pct_of_maxdisp=errpct, lateral_max=lat, ok=bool(ok))

# 3D block anchor (16x16x64, GS): every column must match the same chain analytic; lateral ~ 0
BL = dict(NX=16, NY=16, NZ=64, L=1.0, k0=2000.0, m0=1.0)
dimsB = (16, 16, 64)
F, rel, sw = statics_solve("gs", dimsB, BL, 0.01 * BL["k0"], sweeps_inner=400, cap_outer=3000, tol=3e-4)
zb = F[2].reshape(64, 16, 16)
errB = np.abs(zb - z_ana[:, None, None]).max()
latB = max(np.abs(F[0].reshape(64, 16, 16) - np.arange(16)[None, None, :]).max(),
           np.abs(F[1].reshape(64, 16, 16) - np.arange(16)[None, :, None]).max())
errBpct = 100 * errB / max_disp
print(f"  3D block (16x16x64, gs): res={rel:.2e} | max col err {errBpct:.3f}% | lateral {latB:.2e} | "
      f"{'PASS' if errBpct < ANCHOR_TOL_PCT else 'FAIL'}")
EV["anchor_block3d"] = dict(static_rel_res=rel, err_pct=errBpct, lateral_max=latB, ok=bool(errBpct < ANCHOR_TOL_PCT))


# =================================================================================================
# [2] ENERGY over dynamic steps — dissipation MEASURED (implicit Euler; no symplecticity asserted)
# =================================================================================================
print("\n" + "=" * 100)
print("[2] ENERGY over dynamic steps: pre-stretched 8x8x32 block, free oscillation; drift measured (f64)")
print("=" * 100)
EN = dict(NX=8, NY=8, NZ=32, L=1.0, k0=100.0, m0=1.0)
dimsE = (8, 8, 32); NE = 8 * 8 * 32
om1 = np.sqrt(EN["k0"] / EN["m0"]) * np.pi / (2 * EN["NZ"])       # fundamental longitudinal mode estimate
EV["energy_dynamic"] = dict(omega1_est=om1, runs=[])
i0E, i1E = color_idx(*dimsE)
for h_om in (0.3, 0.1, 0.03):
    h = h_om / om1
    P0 = lattice(*dimsE)
    ztop = (EN["NZ"] - 1) * EN["L"]
    P0[2] = ztop - (ztop - P0[2]) * 1.05                            # 5% uniform stretch from pinned top
    X0, X1 = split(P0, i0E, i1E)
    V0 = torch.zeros_like(X0); V1 = torch.zeros_like(X1)
    Y0 = torch.empty_like(X0); Y1 = torch.empty_like(X1)
    w0E = EN["m0"] / h / h
    Es0 = np_spring_energy(join(X0, X1, i0E, i1E, NE), dimsE, EN["k0"], 1.0, 0)
    Elog = []
    steps = 200
    for s in range(steps):
        Xp0 = X0.clone(); Xp1 = X1.clone()
        Y0.copy_(X0); Y0.add_(V0, alpha=h); Y1.copy_(X1); Y1.add_(V1, alpha=h)
        M.run_gs(X0, X1, Y0, Y1, 100, w0E, EN["k0"], EN["L"], 1.0, 1.0, 1.0, *dimsE, 0, 128)
        V0.copy_(X0).sub_(Xp0).div_(h); V1.copy_(X1).sub_(Xp1).div_(h)
        if (s + 1) % 10 == 0:
            torch.cuda.synchronize()
            F = join(X0, X1, i0E, i1E, NE)
            Vf = join(V0, V1, i0E, i1E, NE)
            E = np_spring_energy(F, dimsE, EN["k0"], 1.0, 0) + 0.5 * EN["m0"] * float((Vf ** 2).sum())
            Elog.append(E)
    Elog = np.array(Elog)
    ratio_end = Elog[-1] / Es0
    # per-step decay from log-linear fit
    stepsx = 10 * np.arange(1, len(Elog) + 1)
    slope = np.polyfit(stepsx, np.log(np.maximum(Elog, 1e-300)), 1)[0]
    print(f"  h*omega1={h_om:5.2f}: E_200/E_0 = {ratio_end:.4f} | decay/step = {100*(1-np.exp(slope)):.4f}% "
          f"(dissipative implicit Euler — measured, not asserted symplectic)")
    EV["energy_dynamic"]["runs"].append(dict(h_omega=h_om, E_end_over_E0=ratio_end,
                                             decay_pct_per_step=100 * (1 - np.exp(slope))))
r_ = EV["energy_dynamic"]["runs"]
if len(r_) == 3 and r_[2]["decay_pct_per_step"] > 0:
    sc = (r_[0]["decay_pct_per_step"] / r_[2]["decay_pct_per_step"])
    print(f"  decay scaling (h*om 0.3 vs 0.03, expect ~(10)^2=100x if ~(h*omega)^2): {sc:.1f}x")
    EV["energy_dynamic"]["decay_scaling_0.3_vs_0.03"] = sc


# =================================================================================================
# [3] R2 bit-repeat x3 per variant at cert scale (256^3), fixed init, 100 iters
# =================================================================================================
print("\n" + "=" * 100)
print("[3] R2 determinism: bit-repeat x3 at 256^3, 100 iters, per variant")
print("=" * 100)
CN = dict(NX=256, NY=256, NZ=256, L=1.0, k0=100.0, m0=1.0)
dimsC = (256, 256, 256); NC = 256 ** 3
w0C = 0.2 * CN["k0"]; h2C = CN["m0"] / w0C
i0C, i1C = color_idx(*dimsC)
rng = np.random.default_rng(7)
P0C = lattice(*dimsC)
P0C += rng.normal(0, 0.01, P0C.shape).astype(np.float32)           # symmetry-breaking noise, fixed seed


def fresh_state(variant):
    if variant == "colored_gs":
        X0, X1 = split(P0C, i0C, i1C)
        Y0 = X0.clone(); Y1 = X1.clone(); Y0[2] += h2C * GVEC; Y1[2] += h2C * GVEC
        return (X0, X1, Y0, Y1)
    X = torch.from_numpy(P0C.copy()).cuda()
    Y = X.clone(); Y[2] += h2C * GVEC
    XO = torch.empty_like(X)
    if variant == "atomic":
        return (X, Y, XO, torch.zeros(9, NC, device="cuda"))
    return (X, Y, XO)


def run_variant(variant, st, iters, block=128):
    if variant == "colored_gs":
        M.run_gs(st[0], st[1], st[2], st[3], iters, w0C, CN["k0"], CN["L"], 1.0, 1.0, 1.0, *dimsC, 0, block)
    elif variant == "jacobi":
        M.run_jacobi(st[0], st[1], st[2], iters, w0C, CN["k0"], CN["L"], 1.0, 1.0, 1.0, *dimsC, 0, block)
    else:
        M.run_atomic(st[0], st[1], st[2], st[3], iters, w0C, CN["k0"], CN["L"], 1.0, 1.0, 1.0, *dimsC, 0, block)


def state_hash(variant, st):
    if variant == "colored_gs":
        return sha(st[0]) + sha(st[1])
    return sha(st[0])


EV["r2_determinism"] = {}
for vn in ("colored_gs", "jacobi", "atomic"):
    hs = []
    for rep in range(3):
        st = fresh_state(vn)
        run_variant(vn, st, 100)
        torch.cuda.synchronize()
        hs.append(state_hash(vn, st))
        del st
        torch.cuda.empty_cache()
    ok = len(set(hs)) == 1
    exp = "expected-PASS (by construction)" if vn != "atomic" else "expected-FAIL (float atomic ordering)"
    print(f"  {vn:10s}: bit-repeat x3 = {'PASS' if ok else 'FAIL'}  [{exp}]  hash={hs[0][:16]}")
    EV["r2_determinism"][vn] = dict(bit_repeat=bool(ok), hashes=[h[:16] for h in hs], expectation=exp)


# =================================================================================================
# [4] ROOFLINE at 256^3 — block sweep, cert per variant
# =================================================================================================
print("\n" + "=" * 100)
print(f"[4] ROOFLINE 256^3 ({NC/1e6:.1f}M vertices): floors jacobi={FLOOR_J:.0f} gs={FLOOR_GS:.0f} B/vtx/iter | "
      f"denominator {BW_COPY:.1f} GB/s | cert bar >= {ROOF_CERT*100:.0f}%")
print("=" * 100)
BLOCKS = [64, 128, 256, 512]
TIMED = 50
EV["roofline"] = {}
for vn, floor, iters_t in (("colored_gs", FLOOR_GS, TIMED), ("jacobi", FLOOR_J, TIMED), ("atomic", None, 10)):
    st = fresh_state(vn)
    run_variant(vn, st, 10)                                   # warm (state relaxes; traffic invariant)
    torch.cuda.synchronize()
    rows = {}
    for blk in BLOCKS:
        try:
            ts = []
            for rep in range(5):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                run_variant(vn, st, iters_t, block=blk)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            tmin = min(ts)
            mvups = NC * iters_t / tmin / 1e6
            if floor is not None:
                bw = floor * NC * iters_t / tmin / 1e9
                beff = tmin * BW_COPY * 1e9 / (NC * iters_t)
                bw_mix = BW_MIX31 if vn == "colored_gs" else BW_MIX21   # A1: GS pass = 3R:1W, jacobi = 2R:1W
                rows[blk] = dict(t_min_s=tmin, MVUPS=mvups, achieved_GBs=bw, roof_frac=bw / BW_COPY,
                                 mix_roof_frac=bw / bw_mix, B_eff=beff)
                print(f"  {vn:10s} block={blk:4d}: {mvups:8.1f} MVUPS | {bw:6.1f} GB/s = {bw/BW_COPY*100:5.1f}% copy-roof"
                      f" | {bw/bw_mix*100:5.1f}% mix-matched-roof | B_eff {beff:6.1f} B/vtx (floor x{beff/floor:.2f})")
            else:
                rows[blk] = dict(t_min_s=tmin, MVUPS=mvups,
                                 B_eff=tmin * BW_COPY * 1e9 / (NC * iters_t))
                print(f"  {vn:10s} block={blk:4d}: {mvups:8.1f} MVUPS | B_eff {rows[blk]['B_eff']:6.1f} B/vtx "
                      f"(diagnostic, no floor cert)")
        except RuntimeError as e:
            rows[blk] = dict(launch_error=str(e)[:120])
            print(f"  {vn:10s} block={blk:4d}: LAUNCH ERROR (gated, not a timing): {str(e)[:80]}")
    # NaN gate: timings on a diverged/NaN state are still traffic-valid but flag it
    finite = bool(torch.isfinite(st[0]).all().item())
    if not finite:
        print(f"  {vn}: WARNING state non-finite after timing — flagged")
    EV["roofline"][vn] = dict(blocks=rows, floor_B=floor, state_finite=finite, timed_iters=iters_t)
    del st
    torch.cuda.empty_cache()

best = {}
for vn in ("colored_gs", "jacobi"):
    ok_rows = {b: r for b, r in EV["roofline"][vn]["blocks"].items() if "roof_frac" in r}
    bb = max(ok_rows, key=lambda b: ok_rows[b]["roof_frac"])
    best[vn] = dict(block=bb, **ok_rows[bb])
    r = best[vn]
    r["cert_R1"] = bool(r["roof_frac"] >= ROOF_CERT)
    r["strict_Beff_le_1.2floor"] = bool(r["B_eff"] <= TRAFFIC_MULT * EV["roofline"][vn]["floor_B"])
    r["beats_prereg_85"] = bool(r["roof_frac"] >= ROOF_BEAT)
    print(f"  BEST {vn:10s}: block={bb} {r['roof_frac']*100:.1f}% of copy roofline "
          f"({r['mix_roof_frac']*100:.1f}% of its mix-matched roof) | R1(>=70%): "
          f"{'PASS' if r['cert_R1'] else 'FAIL'} | strict B_eff<=1.2xfloor (<=>83.3%): "
          f"{'PASS' if r['strict_Beff_le_1.2floor'] else 'FAIL'} | >=85%: {r['beats_prereg_85']}")
mech = all(best[v]["mix_roof_frac"] <= 1.02 for v in ("colored_gs", "jacobi"))
print(f"  A1 mechanism check: mix-matched fractions <= ~100%: {'CONFIRMED (read-heavy mix explains >100%-of-copy)' if mech else 'NOT CONFIRMED — residual unexplained, flagged'}")
EV["roofline"]["a1_mix_mechanism_confirmed"] = bool(mech)
EV["roofline"]["best"] = best

# bytes-to-solution comparison (uses convergence factors measured in [5]; filled after)


# =================================================================================================
# [5] CONVERGENCE: contraction/sweep, colored-GS vs Jacobi (homogeneous), float64 position-error instrument
# =================================================================================================
print("\n" + "=" * 100)
print("[5] CONVERGENCE homogeneous 32^3: contraction factor per sweep, GS vs Jacobi (pre-reg rho_GS ~ rho_J^2)")
print("=" * 100)
HG = dict(NX=32, NY=32, NZ=32, L=1.0, k0=100.0, m0=1.0)
dimsH = (32, 32, 32); NH = 32 ** 3
w0H = 0.2 * HG["k0"]; h2H = HG["m0"] / w0H
i0H, i1H = color_idx(*dimsH)


def one_step_problem(dims, params, w0, Rm=1.0, Rk=1.0, NZhalf=0):
    """fresh implicit step from rest: x0 = rest lattice, y = x0 + h^2 g."""
    NX, NY, NZ = dims
    P0 = lattice(NX, NY, NZ, params["L"])
    Y = P0.copy(); Y[2] += (params["m0"] / w0) * GVEC
    return P0, Y


def conv_run(variant, dims, params, w0, Rm, Rk, NZhalf, xstar, omega, cap, check_every, i0, i1, tol=CONV_TOL):
    NX, NY, NZ = dims; N = NX * NY * NZ
    P0, Ynp = one_step_problem(dims, params, w0, Rm, Rk, NZhalf)
    err0 = np.linalg.norm(P0.astype(np.float64) - xstar)
    curve = []
    if variant == "colored_gs":
        X0, X1 = split(P0, i0, i1)
        Y0, Y1 = split(Ynp, i0, i1)
        for it in range(0, cap, check_every):
            M.run_gs(X0, X1, Y0, Y1, check_every, w0, params["k0"], params["L"], omega, Rm, Rk, NX, NY, NZ, NZhalf, 128)
            torch.cuda.synchronize()
            F = join(X0, X1, i0, i1, N)
            e = np.linalg.norm(F - xstar) / err0
            curve.append((it + check_every, e))
            if not np.isfinite(e) or e > 1e4:
                return curve, float("inf"), "div"
            if e < tol:
                return curve, it + check_every, "tol"
    else:
        X = torch.from_numpy(P0.copy()).cuda(); Y = torch.from_numpy(Ynp.copy()).cuda(); XO = torch.empty_like(X)
        assert check_every % 2 == 0
        for it in range(0, cap, check_every):
            M.run_jacobi(X, Y, XO, check_every, w0, params["k0"], params["L"], omega, Rm, Rk, NX, NY, NZ, NZhalf, 128)
            torch.cuda.synchronize()
            F = X.cpu().numpy().astype(np.float64)
            e = np.linalg.norm(F - xstar) / err0
            curve.append((it + check_every, e))
            if not np.isfinite(e) or e > 1e4:
                return curve, float("inf"), "div"
            if e < tol:
                return curve, it + check_every, "tol"
    return curve, float("inf"), "cap"


def reference_xstar(dims, params, w0, Rm, Rk, NZhalf, i0, i1, sweeps=40000):
    """converged x* via long GS; agreement cross-checked vs long damped Jacobi."""
    NX, NY, NZ = dims; N = NX * NY * NZ
    P0, Ynp = one_step_problem(dims, params, w0, Rm, Rk, NZhalf)
    X0, X1 = split(P0, i0, i1); Y0, Y1 = split(Ynp, i0, i1)
    M.run_gs(X0, X1, Y0, Y1, sweeps, w0, params["k0"], params["L"], 1.0, Rm, Rk, NX, NY, NZ, NZhalf, 128)
    torch.cuda.synchronize()
    Fg = join(X0, X1, i0, i1, N)
    X = torch.from_numpy(P0.copy()).cuda(); Y = torch.from_numpy(Ynp.copy()).cuda(); XO = torch.empty_like(X)
    M.run_jacobi(X, Y, XO, sweeps, w0, params["k0"], params["L"], 0.5, Rm, Rk, NX, NY, NZ, NZhalf, 128)
    torch.cuda.synchronize()
    Fj = X.cpu().numpy().astype(np.float64)
    disp = np.linalg.norm(Fg - P0.astype(np.float64))
    agree = np.linalg.norm(Fg - Fj) / max(disp, 1e-30)
    # float64 incremental residual of the reference (convergedness certificate)
    r = np_grad(Fg, dims, params["k0"], Rk, NZhalf, params["m0"], Rm, "incr", Y=Ynp.astype(np.float64), w0=w0)
    r0 = np_grad(P0.astype(np.float64), dims, params["k0"], Rk, NZhalf, params["m0"], Rm, "incr",
                 Y=Ynp.astype(np.float64), w0=w0)
    relres = np.linalg.norm(r) / np.linalg.norm(r0)
    return Fg, agree, relres


xstarH, agreeH, relresH = reference_xstar(dimsH, HG, w0H, 1.0, 1.0, 0, i0H, i1H, sweeps=20000)
print(f"  reference x*: GS-vs-damped-Jacobi agreement {agreeH:.2e} (rel to disp) | f64 incr-residual rel {relresH:.2e}")
curves = {}
for vn, omg in (("colored_gs", 1.0), ("jacobi", 1.0)):
    curve, it_tol, _st = conv_run(vn, dimsH, HG, w0H, 1.0, 1.0, 0, xstarH, omg, cap=4000, check_every=2,
                                  i0=i0H, i1=i1H)
    if vn == "jacobi" and not np.isfinite(it_tol):
        print("  jacobi omega=1 did not converge within cap — falling back to omega=0.5 (recorded)")
        curve, it_tol, _st = conv_run(vn, dimsH, HG, w0H, 1.0, 1.0, 0, xstarH, 0.5, cap=8000, check_every=2,
                                      i0=i0H, i1=i1H)
    # contraction fit on the tail err in [1e-1, 1e-3]
    seg = [(i, e) for i, e in curve if 1e-3 <= e <= 1e-1]
    rho = np.exp(np.polyfit([s[0] for s in seg], np.log([s[1] for s in seg]), 1)[0]) if len(seg) >= 3 else float("nan")
    curves[vn] = (curve, it_tol, rho)
    print(f"  {vn:10s}: sweeps to err<1e-3 = {it_tol} | contraction/sweep rho = {rho:.5f}")
rho_gs, rho_j = curves["colored_gs"][2], curves["jacobi"][2]
expo = np.log(rho_gs) / np.log(rho_j) if np.isfinite(rho_gs) and np.isfinite(rho_j) else float("nan")
ratio_sweeps = curves["jacobi"][1] / curves["colored_gs"][1]
print(f"  log(rho_GS)/log(rho_J) = {expo:.2f} (pre-reg ~2, Young) | sweeps ratio J/GS = {ratio_sweeps:.2f}")
EV["convergence_homogeneous"] = dict(ref_agreement=agreeH, ref_relres=relresH,
                                     gs=dict(sweeps_to_tol=curves["colored_gs"][1], rho=rho_gs),
                                     jacobi=dict(sweeps_to_tol=curves["jacobi"][1], rho=rho_j),
                                     young_exponent=expo, sweeps_ratio_J_over_GS=ratio_sweeps)
# bytes-to-solution: floor x sweeps-to-tol
b2s_gs = FLOOR_GS * curves["colored_gs"][1]
b2s_j = FLOOR_J * curves["jacobi"][1]
print(f"  bytes-to-solution (floor x sweeps): GS {b2s_gs:.0f} vs Jacobi {b2s_j:.0f} B/vtx -> GS advantage x{b2s_j/b2s_gs:.2f} "
      f"(pre-reg ~1.5x)")
EV["convergence_homogeneous"]["bytes_to_solution_advantage_GS"] = b2s_j / b2s_gs


# =================================================================================================
# [6] MASS-RATIO LEG — two-material lattice, R sweep, iterations-to-tolerance
# =================================================================================================
print("\n" + "=" * 100)
print("[6] MASS-RATIO leg 16x16x32 (soft/light pinned TOP half, heavy/stiff BOTTOM half hanging):")
print("    (A) stiffness ratio Rk, (B) mass ratio Rm; sweeps-to-tol, GS vs best-omega Jacobi")
print("=" * 100)
MR = dict(NX=16, NY=16, NZ=32, L=1.0, k0=100.0, m0=1.0)
dimsM = (16, 16, 32); NM = 16 * 16 * 32
w0M = 0.2 * MR["k0"]
i0M, i1M = color_idx(*dimsM)
NZhalfM = 16
CAP = 60000
EV["massratio"] = dict(config=dict(**MR, w0=w0M, NZhalf=NZhalfM, tol=CONV_TOL, cap=CAP), legs={})
for leg, (rname, rvals) in (("A_stiffness", ("Rk", [1, 10, 100, 1000, 10000])),
                            ("B_mass", ("Rm", [1, 10, 100, 1000, 10000]))):
    rows = []
    for R in rvals:
        Rm = R if rname == "Rm" else 1.0
        Rk = R if rname == "Rk" else 1.0
        capR = 240000 if (rname == "Rk" and R >= 1000) else CAP        # A3: extend cap to close run-1's bound
        xstar, agree, relres = reference_xstar(dimsM, MR, w0M, Rm, Rk, NZhalfM, i0M, i1M, sweeps=capR)
        _, it_gs, st_gs = conv_run("colored_gs", dimsM, MR, w0M, Rm, Rk, NZhalfM, xstar, 1.0, capR, 20, i0M, i1M)
        jbest = float("inf"); jom = None; jst = "cap"
        for omg in (1.0, 0.7, 0.5):
            _, it_j, st_j = conv_run("jacobi", dimsM, MR, w0M, Rm, Rk, NZhalfM, xstar, omg, capR, 20, i0M, i1M)
            if it_j < jbest:
                jbest, jom, jst = it_j, omg, st_j
            if st_j != "div" and jst == "cap" and not np.isfinite(jbest):
                jst = st_j
        reliable = agree < 1e-2 and relres < 1e-2
        gs_eff = it_gs if np.isfinite(it_gs) else capR
        j_eff = jbest if np.isfinite(jbest) else capR
        is_bound = (not np.isfinite(jbest)) and np.isfinite(it_gs) and jst != "div"
        ratio = j_eff / gs_eff
        tag = " [LOWER-BOUND: jacobi cap-hit]" if is_bound else ("" if np.isfinite(jbest) and np.isfinite(it_gs)
                                                                 else " [CAP/DIV]")
        if not reliable:
            tag += " [UNRELIABLE-REF: fp32 envelope edge]"
        print(f"  {leg} {rname}={R:6d}: ref-agree {agree:.1e} relres {relres:.1e} | GS {it_gs} ({st_gs}) | "
              f"Jacobi(best om={jom}) {jbest} ({jst}) | ratio J/GS {'>=' if is_bound else '='} {ratio:.2f}{tag}")
        rows.append(dict(R=R, cap=capR, ref_agreement=agree, ref_relres=relres, sweeps_gs=it_gs, gs_status=st_gs,
                         sweeps_jacobi_best=jbest, jacobi_status=jst, jacobi_omega=jom,
                         ratio_J_over_GS=ratio, ratio_is_lower_bound=bool(is_bound), reliable=bool(reliable)))
    EV["massratio"]["legs"][leg] = rows
    usable = [r for r in rows if r["reliable"] and (np.isfinite(r["sweeps_gs"]))]
    if len(usable) >= 2:
        grew = usable[-1]["ratio_J_over_GS"] > usable[0]["ratio_J_over_GS"] * 1.1
        b = " (last point is a lower bound)" if usable[-1]["ratio_is_lower_bound"] else ""
        print(f"  {leg}: ratio({rname}={usable[-1]['R']}) / ratio({rname}={usable[0]['R']}) = "
              f"{usable[-1]['ratio_J_over_GS']/usable[0]['ratio_J_over_GS']:.2f}{b} -> pre-reg 'advantage grows' "
              f"{'SUPPORTED' if grew else 'NOT SUPPORTED (Young flat-2x counter-prediction)'}")
        EV["massratio"]["legs"][leg + "_direction_supported"] = bool(grew)


# =================================================================================================
# VERDICT + evidence
# =================================================================================================
print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
cands = {}
for vn in ("colored_gs", "jacobi"):
    b = EV["roofline"]["best"][vn]
    anch_name = "gs" if vn == "colored_gs" else "jacobi"
    ok = (b["cert_R1"] and EV["anchor_ladder"][anch_name]["ok"] and EV["r2_determinism"][vn]["bit_repeat"]
          and EV["roofline"][vn]["state_finite"])
    cands[vn] = ok
    print(f"  {vn:10s}: R1 {b['roof_frac']*100:5.1f}% (>=70% {'PASS' if b['cert_R1'] else 'FAIL'}) | anchor "
          f"{'PASS' if EV['anchor_ladder'][anch_name]['ok'] else 'FAIL'} | R2 "
          f"{'PASS' if EV['r2_determinism'][vn]['bit_repeat'] else 'FAIL'} | >=85% beats-pre-reg: {b['beats_prereg_85']} "
          f"-> {'CERTIFIED' if ok else 'not certified'}")
CERT = any(cands.values())
best_vn = max(("colored_gs", "jacobi"), key=lambda v: EV["roofline"]["best"][v]["roof_frac"])
EV["verdict"] = dict(certified=bool(CERT), per_variant=cands, best_variant=best_vn,
                     best_roof_frac=EV["roofline"]["best"][best_vn]["roof_frac"],
                     claim="first at-roofline certified non-fluid deformable (VBD/AVBD-class) kernel"
                           if CERT else "NOT certified — gap localized in roofline block table")
EV["contamination_flag"] = "GPU shared (desktop resident); flock held; small negative roofline bias possible, uncorrected"
EV["wall_s"] = time.time() - T_START
print(f"\n  OVERALL: {'CERTIFIED' if CERT else 'NOT CERTIFIED'} (best={best_vn}, "
      f"{EV['roofline']['best'][best_vn]['roof_frac']*100:.1f}% of {BW_COPY:.0f} GB/s copy roofline)")
print(f"  wall {EV['wall_s']:.0f}s")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


out = "artifacts/d_avbd_gpu_certified_evidence.json"
with open(out, "w") as fh:
    json.dump(_clean(EV), fh, indent=1)
print(f"  evidence -> {out}")

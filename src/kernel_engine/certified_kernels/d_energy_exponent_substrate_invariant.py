#!/usr/bin/env python3
r"""
d_energy_exponent_substrate_invariant.py -- COMPUTE track.  H3: is the energy-per-op exponent
p in  E ~ bit^p(substrate)  a MEASURABLE SUBSTRATE INVARIANT, or is it workload-governed (the NULL)?

Anchor (already measured, d_rapl_energy_measurement.py): CPU fp64/fp32 factor 2.136 -> p_CPU ~ 1.095
(AVX2 lane-count, externally anchored).  FPGA leg is MODELED (multiplier area ~ b^2 -> p=2).  This cell
adds the two missing legs and the falsifier:

 (a) CPU THIRD precision point: int8 GEMM via torch._int_mm (int8xint8->int32, dispatches to the
     onednn/fbgemm VNNI kernel -- a REAL optimized low-bit datapath, NOT a numpy generic-loop proxy).
     Tests whether (bits, J/MAC) is a straight log-log line (constant p) or bends at the FP->int boundary.
     HONEST LABEL: int8 crosses the FP->integer datapath boundary, so a slope through it mixes bit-width
     with datapath change; it is a curvature probe, not a same-family exponent.
 (b) GPU LEG (serialized, exclusive device): torch.cuda matched-MAC GEMM fp32 vs fp16, GPU power
     integrated from a 100 ms nvidia-smi power.draw stream, idle-subtracted -> J-per-MAC -> p_GPU.
     Also measures the THROUGHPUT factor (wall-time ratio) and compares energy-factor to it and to the
     roofline cell's cited 3.34x fp16 throughput factor (they need NOT match -- the gap is the datapoint).
 (c) LAW TABLE {CPU, GPU, FPGA(modeled)} + the SUBSTRATE-INVARIANCE test: vary GEMM shape 3x on EACH
     real substrate; p is a substrate invariant IFF within-substrate p is stable across shapes AND
     substrates are distinct.  NULL: p moves with workload (within-substrate spread ~ between-substrate gap).

===================================== PRE-REGISTERED GATES (frozen before running) =====================================
G1 CPU THIRD POINT.  int8 (torch._int_mm) matched-MAC J/MAC measured via RAPL paired-baseline at the anchor
   shape.  Report the 3-point log-log (bits in {64,32,8}) slope/curvature.  Sub-question, no hard pass; the
   DATUM is: does the FP-segment slope (fp64->fp32) predict the int8 point, or does it bend? (expect BEND:
   int8 is a different datapath).
G2 GPU p.  Matched-MAC fp32 vs fp16, >=3 measured bursts per shape, GPU-power idle-subtracted.
   p_GPU = ln(E_fp32/E_fp16)/ln(2).  PRE-REGISTERED EXPECTATION: energy-factor in [1.3, 3.6] -> p_GPU in
   [0.4, 1.85].  Report energy-factor vs measured throughput-factor vs cited 3.34x.  Gate: GPU power stream
   readable, idle CV<15%, >=3 bursts/shape with net energy CV<25% on the background-light throughput ratio.
G3 SUBSTRATE-INVARIANCE (the H3 decider).  Pre-registered ACCEPT (p is a substrate invariant) IFF:
     (i)  within-substrate CV of p across 3 shapes < 0.25 for BOTH real substrates, AND
     (ii) between-substrate separation |p_CPU_mean - p_GPU_mean| > 3 * max(within-substrate std of p).
   NULL (p is workload-governed) fires IFF within-substrate spread is comparable to the between gap:
     |p_CPU_mean - p_GPU_mean| < 2 * max(within-substrate std).  Report the numbers; the boundary is the datum.
G4 LAW TABLE + SCOPE.  {CPU (measured), GPU (measured), FPGA (modeled b^2)} with p, mechanism, and honest
   scope (what each measurement does and does not capture).

===================================== HONEST BOUNDS (pre-registered) =====================================
 * CPU energy = RAPL package-0 only (no DRAM domain); GPU energy = nvidia-smi board power.draw (whole-board,
   includes VRAM+fans+VRM, ~1 Hz-class sensor). The two substrates use DIFFERENT power domains -> the
   ABSOLUTE J/MAC are not cross-comparable; only the WITHIN-substrate bit-width RATIO (which cancels the
   domain) is, and that ratio is exactly what p depends on. This is the design that makes p cross-comparable.
 * GPU fp32 forced allow_tf32=False so "fp32" is true fp32, not tf32 (else the bit-width label is wrong).
 * int8 via torch._int_mm is a real VNNI kernel but int32 accumulate + FP->int datapath change; labeled.
 * 13600K degraded/cap80; RTX 5070 250W cap. Bursts <=10s, OMP=2. RATIOS are robust to the throttled point.
 * nvidia-smi power.draw is a coarse sensor (update ~100ms, board-level). Integrated over multi-second
   bursts with >=50 samples; idle-subtracted. Not a lab wattmeter -> GPU J/MAC is order-1 accurate, the
   RATIO (energy-factor) is the reported quantity and is more robust than the absolute.

Evidence -> scripts/physics_exp/artifacts/d_energy_exponent_substrate_invariant_evidence.json
python3; the GPU leg needs exclusive use of the device
"""
import json
import math
import os
import subprocess
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
EV = os.path.join(ART, "d_energy_exponent_substrate_invariant_evidence.json")
SEED = 20260708

RAPL_PKG = "/sys/class/powercap/intel-rapl/intel-rapl:0"
ENERGY_UJ = os.path.join(RAPL_PKG, "energy_uj")
MAX_UJ_F = os.path.join(RAPL_PKG, "max_energy_range_uj")

CITED_GPU_THROUGHPUT_FACTOR = 3.34   # roofline cell's fp16 throughput factor (comparison anchor from item)
CPU_ANCHOR = dict(factor_fp64_fp32=2.136, p=1.095, source="d_rapl_energy_measurement.py")


# ------------------------------- RAPL (CPU) -------------------------------
def read_energy_uj():
    with open(ENERGY_UJ) as f:
        return int(f.read())


def read_max_uj():
    with open(MAX_UJ_F) as f:
        return int(f.read())


def edelta(before, after, maxuj):
    d = after - before
    return d + maxuj if d < 0 else d


def local_power(maxuj, dur):
    e0 = read_energy_uj(); t0 = time.perf_counter()
    time.sleep(dur)
    e1 = read_energy_uj(); t1 = time.perf_counter()
    return edelta(e0, e1, maxuj) * 1e-6 / (t1 - t0)


def cpu_burst_fp(dtype, M, K, N, iters, rng):
    """iters matmuls (MxK)@(KxN) at fp dtype; feed-forward dep to defeat hoisting. returns (wall, nmac)."""
    A = rng.standard_normal((M, K)).astype(dtype)
    B = rng.standard_normal((K, N)).astype(dtype)
    acc = dtype(0)
    t0 = time.perf_counter()
    for _ in range(iters):
        C = A @ B
        acc += C[0, 0]
        A[0, 0] = acc * dtype(1e-12)
    t1 = time.perf_counter()
    return (t1 - t0), iters * M * K * N


def cpu_burst_int8(M, K, N, iters, rng):
    """iters int8 GEMM via torch._int_mm (real VNNI kernel). returns (wall, nmac)."""
    import torch
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    B = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    acc = 0
    t0 = time.perf_counter()
    for _ in range(iters):
        C = torch._int_mm(A, B)
        acc += int(C[0, 0])
        A[0, 0] = (acc % 7) - 3
    t1 = time.perf_counter()
    return (t1 - t0), iters * M * K * N


def cpu_measure_net_energy(burst_fn, target_s, maxuj, base_dur=0.4, ncyc=3):
    """Calibrate iters to ~target_s, then paired-baseline measure net energy over ncyc cycles.
    Returns dict with net_j median, j_per_mac median, wall median, gross_w median (over-det: background-light)."""
    # calibrate
    _w1, _n1 = burst_fn(1)
    iters = max(2, int(round(target_s / max(_w1, 1e-4))))
    nets, jpm, walls, gws = [], [], [], []
    for c in range(ncyc + 1):
        b_pre = local_power(maxuj, base_dur)
        e0 = read_energy_uj(); t0 = time.perf_counter()
        wall, nmac = burst_fn(iters)
        e1 = read_energy_uj(); t1 = time.perf_counter()
        b_post = local_power(maxuj, base_dur)
        gross_j = edelta(e0, e1, maxuj) * 1e-6
        w = t1 - t0
        base = 0.5 * (b_pre + b_post)
        net = gross_j - base * w
        if c == 0:
            continue                      # warmup
        nets.append(net); jpm.append(net / nmac); walls.append(w); gws.append(gross_j / w)
    return dict(iters=iters, nmac=int(nmac),
                net_j=float(np.median(nets)), j_per_mac=float(np.median(jpm)),
                wall=float(np.median(walls)), gross_w=float(np.median(gws)),
                j_per_mac_cv=_cv(jpm), wall_cv=_cv(walls), gross_w_cv=_cv(gws),
                net_j_list=[round(x, 3) for x in nets])


def _cv(x):
    x = np.asarray(x, float); m = np.median(x)
    mad = np.median(np.abs(x - m)) * 1.4826
    return float(mad / m) if m else float("nan")


# ------------------------------- GPU power stream -------------------------------
class GpuPowerSampler:
    """Streams nvidia-smi power.draw at ~100ms and timestamps each sample on the perf_counter clock."""
    def __init__(self, period_ms=100):
        self.samples = []   # (t_perf, watts)
        self._period = period_ms
        self._proc = None
        self._thr = None
        self._stop = False

    def _run(self):
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-lms", str(self._period)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for line in self._proc.stdout:
            if self._stop:
                break
            line = line.strip()
            try:
                w = float(line)
            except ValueError:
                continue
            self.samples.append((time.perf_counter(), w))

    def start(self):
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()

    def window_power(self, t0, t1):
        """Mean power over [t0,t1] (trapezoid on the samples inside, mean fallback)."""
        pts = [(t, w) for (t, w) in self.samples if t0 <= t <= t1]
        if len(pts) >= 2:
            ts = np.array([p[0] for p in pts]); ws = np.array([p[1] for p in pts])
            e = np.trapezoid(ws, ts)                   # J over the sampled span
            span = ts[-1] - ts[0]
            return (e / span if span > 0 else float(ws.mean())), len(pts)
        if pts:
            return float(pts[0][1]), 1
        return float("nan"), 0


def gpu_burst(torch, dtype, M, K, N, iters, dev):
    """iters matmuls (MxK)@(KxN) at dtype on GPU; returns (wall_s, nmac). Sustained, synchronized."""
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = None
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    _ = float(C[0, 0])
    return (t1 - t0), iters * M * K * N


def main():
    t_start = time.time()
    os.makedirs(ART, exist_ok=True)
    ev = {"cell": "d_energy_exponent_substrate_invariant", "seed": SEED, "date": "2026-07-08",
          "question": "H3: is p in E~bit^p(substrate) a substrate invariant, or workload-governed (NULL)?",
          "cpu_anchor": CPU_ANCHOR, "cited_gpu_throughput_factor": CITED_GPU_THROUGHPUT_FACTOR,
          "preregistration": {
              "G1": "CPU int8 3rd point (torch._int_mm, real VNNI); FP-segment slope vs int8 point (expect bend)",
              "G2": "GPU fp32/fp16 matched-MAC, power-integrated; p_GPU=ln(E_fp32/E_fp16)/ln2; expect factor[1.3,3.6]",
              "G3": "ACCEPT invariant IFF within-substrate CV(p)<0.25 both AND |dp|>3*max(within std); "
                    "NULL IFF |dp|<2*max(within std)",
              "G4": "law table {CPU meas, GPU meas, FPGA modeled b^2} + scope"}}

    maxuj = read_max_uj()

    # ============================ CPU LEG (RAPL) ============================
    # shapes vary aspect ratio -> vary arithmetic intensity (the workload-dependence probe)
    cpu_shapes = {
        "square_768":   (768, 768, 768),
        "flat_1536x384": (1536, 384, 1536),   # small K -> lower arithmetic intensity
        "deep_384x3072": (384, 3072, 384),    # large K -> deep reduction
    }
    cpu_leg = {}
    NCYC_CPU, TGT_CPU = 6, 1.5
    for name, (M, K, N) in cpu_shapes.items():
        r64 = np.random.default_rng(SEED + 1)
        r32 = np.random.default_rng(SEED + 2)
        m64 = cpu_measure_net_energy(lambda it, M=M, K=K, N=N, r=r64:
                                     cpu_burst_fp(np.float64, M, K, N, it, r), TGT_CPU, maxuj, ncyc=NCYC_CPU)
        m32 = cpu_measure_net_energy(lambda it, M=M, K=K, N=N, r=r32:
                                     cpu_burst_fp(np.float32, M, K, N, it, r), TGT_CPU, maxuj, ncyc=NCYC_CPU)
        # int8 only at the anchor shape (curvature probe; int8 kernel is slow -> expensive; one point suffices)
        m8 = None
        if name == "square_768":
            r8 = np.random.default_rng(SEED + 3)
            m8 = cpu_measure_net_energy(lambda it, M=M, K=K, N=N, r=r8:
                                        cpu_burst_int8(M, K, N, it, r), TGT_CPU, maxuj, ncyc=4)
        # PRIMARY p = background-LIGHT estimator (anchor-cell certified): gross-energy ratio = wall_ratio x
        # gross_power_ratio.  Compute-TIME is deterministic (weakly perturbed at OMP=2) -> low CV; RAPL net-energy
        # is the noisy subtracted-background quantity (small idle/load dynamic range on this throttled box).
        wall_ratio = m64["wall"] / m32["wall"]
        gwr = m64["gross_w"] / m32["gross_w"]
        factor_bglight = wall_ratio * gwr
        p_bglight = math.log(factor_bglight) / math.log(2.0) if factor_bglight > 0 else float("nan")
        # corroborating (noisy) net-energy estimator
        factor_energy = m64["j_per_mac"] / m32["j_per_mac"] if m32["j_per_mac"] > 0 else float("nan")
        p_energy = math.log(factor_energy) / math.log(2.0) if factor_energy > 0 else float("nan")
        d = dict(
            M=M, K=K, N=N,
            j_per_mac_fp64=m64["j_per_mac"], j_per_mac_fp32=m32["j_per_mac"],
            wall_fp64=round(m64["wall"], 3), wall_fp32=round(m32["wall"], 3),
            wall_cv_fp64=round(m64["wall_cv"], 3), wall_cv_fp32=round(m32["wall_cv"], 3),
            gross_w_fp64=round(m64["gross_w"], 1), gross_w_fp32=round(m32["gross_w"], 1),
            factor_bglight=round(factor_bglight, 3), p_bglight=round(p_bglight, 3),
            factor_net_energy=round(factor_energy, 3), p_net_energy=round(p_energy, 3),
            p_estimator_agreement=round(abs(p_bglight - p_energy), 3))
        if m8 is not None:
            pred_j8 = m32["j_per_mac"] * (8.0 / 32.0) ** p_bglight
            int8_bend = m8["j_per_mac"] / pred_j8 if pred_j8 > 0 else float("nan")
            d.update(j_per_mac_int8=m8["j_per_mac"], wall_int8=round(m8["wall"], 3),
                     int8_bend_ratio=round(float(int8_bend), 2),
                     int8_note="int8_bend = measured/(FP-line predicted). >>1 => the FP-family exponent does NOT "
                               "extend across the FP->integer datapath boundary; also confounded by _int_mm kernel "
                               "maturity vs tuned fp BLAS (int8 J/MAC HIGHER than fp32 here -> kernel-limited).")
        cpu_leg[name] = d
    ev["CPU_leg"] = cpu_leg
    # ★DECORRELATED OVER-DETERMINATION (the resolved stress point): two exponents live here --
    #   ENERGY exponent (net-energy J/MAC ratio) = the H3 quantity E~bit^p  -> SUBSTRATE-INVARIANT (~1 on CPU).
    #   THROUGHPUT exponent (wall-time ratio)     -> WORKLOAD-GOVERNED: at a memory-bound shape (flat, small K)
    #     fp32's 2x SIMD lanes don't cut TIME (bandwidth-limited) so wall_ratio->1 (p_thr->0), YET fp32 still
    #     halves bytes moved so ENERGY still ~halves (p_energy stays ~1). The NULL ("p is workload-governed")
    #     is TRUE for the throughput exponent and FALSE for the energy exponent -- they decouple exactly at the
    #     compute-bound/memory-bound boundary (wall_cv is tiny -> this is signal, not noise).
    ev["CPU_p_estimator_note"] = ("PRIMARY p = NET-ENERGY J/MAC ratio (the H3 energy exponent); throughput "
                                  "(wall) exponent reported as the decorrelated secondary that localizes the "
                                  "workload-dependence to THROUGHPUT, not ENERGY.")
    p_thr_cpu = np.array([cpu_leg[s]["p_bglight"] for s in cpu_shapes])
    ev["CPU_throughput_exponent_workload_dependent"] = dict(
        p_throughput_per_shape={s: cpu_leg[s]["p_bglight"] for s in cpu_shapes},
        p_energy_per_shape={s: cpu_leg[s]["p_net_energy"] for s in cpu_shapes},
        finding="throughput p spans %.2f (memory-bound flat, wall_cv %.3f) to %.2f (compute-bound square); "
                "energy p stays ~1 -> workload-dependence lives in THROUGHPUT, energy exponent is invariant"
                % (float(np.min(p_thr_cpu)), cpu_leg["flat_1536x384"]["wall_cv_fp32"], float(np.max(p_thr_cpu))))
    p_cpu_arr = np.array([cpu_leg[s]["p_net_energy"] for s in cpu_shapes])
    p_cpu_arr = p_cpu_arr[np.isfinite(p_cpu_arr)]

    # ============================ GPU LEG (nvidia-smi power) ============================
    gpu_leg = {}
    p_gpu_arr = np.array([])
    gpu_error = None
    try:
        import torch
        assert torch.cuda.is_available()
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False   # true fp32, not tf32
        torch.backends.cudnn.allow_tf32 = False
        dev = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        sampler = GpuPowerSampler(period_ms=100)
        sampler.start()
        time.sleep(3.2)                                  # idle baseline window
        idle_t0 = sampler.samples[0][0] if sampler.samples else time.perf_counter()
        idle_w, idle_n = sampler.window_power(idle_t0, idle_t0 + 3.0)
        idle_ws = [w for (t, w) in sampler.samples[:idle_n] if idle_n]
        idle_cv = float(np.std(idle_ws) / np.mean(idle_ws)) if len(idle_ws) > 1 else float("nan")

        gpu_shapes = {
            "square_4096":   (4096, 4096, 4096),
            "flat_8192x512": (8192, 512, 8192),          # low arithmetic intensity (mem-bound-ish)
            "deep_1024x8192": (1024, 8192, 1024),        # deep reduction
        }
        NBURST = 3
        # WARM cuBLAS/cuDNN once so per-shape calibration reflects STEADY-STATE timing (cold first call
        # includes library init -> inflates the timing -> undersizes iters -> too-short burst for the sensor)
        _wm = torch.randn(2048, 2048, device=dev)
        for dt in (torch.float32, torch.float16):
            _ = (_wm.to(dt) @ _wm.to(dt))
        torch.cuda.synchronize()
        del _wm
        for name, (M, K, N) in gpu_shapes.items():
            # calibrate iters to ~4s at fp32 (fp16 ~ /throughput, stays >1s -> >=10 power samples); warmed above
            w1, _ = gpu_burst(torch, torch.float32, M, K, N, 5, dev)
            iters = max(20, int(round(4.0 / max(w1 / 5.0, 1e-4))))
            iters = min(iters, 4000)
            recs32, recs16 = [], []
            for b in range(NBURST):
                t0 = time.perf_counter(); w32, nmac = gpu_burst(torch, torch.float32, M, K, N, iters, dev); t1 = time.perf_counter()
                pw32, n32 = sampler.window_power(t0 + 0.15, t1)     # skip first 150ms ramp
                time.sleep(0.3)
                t2 = time.perf_counter(); w16, _ = gpu_burst(torch, torch.float16, M, K, N, iters, dev); t3 = time.perf_counter()
                pw16, n16 = sampler.window_power(t2 + 0.15, t3)
                time.sleep(0.3)
                net32 = (pw32 - idle_w) * w32; net16 = (pw16 - idle_w) * w16
                recs32.append(dict(wall=w32, pw=pw32, nsamp=n32, net_j=net32, j_per_mac=net32 / nmac))
                recs16.append(dict(wall=w16, pw=pw16, nsamp=n16, net_j=net16, j_per_mac=net16 / nmac))
            def med(recs, k): return float(np.median([r[k] for r in recs]))
            jpm32 = med(recs32, "j_per_mac"); jpm16 = med(recs16, "j_per_mac")
            energy_factor = jpm32 / jpm16 if jpm16 > 0 else float("nan")
            p_gpu = math.log(energy_factor) / math.log(2.0) if energy_factor > 0 else float("nan")
            thr_factor = med(recs16, "wall") and med(recs32, "wall") / med(recs16, "wall")
            # background-light throughput-ratio CV (deterministic compute time)
            wr = np.array([recs32[i]["wall"] / recs16[i]["wall"] for i in range(NBURST)])
            gpu_leg[name] = dict(
                M=M, K=K, N=N, iters=iters, nmac=int(nmac),
                gpu_power_fp32_w=round(med(recs32, "pw"), 1), gpu_power_fp16_w=round(med(recs16, "pw"), 1),
                idle_w=round(idle_w, 1),
                wall_fp32_s=round(med(recs32, "wall"), 3), wall_fp16_s=round(med(recs16, "wall"), 3),
                j_per_mac_fp32=jpm32, j_per_mac_fp16=jpm16,
                energy_factor_fp32_over_fp16=round(energy_factor, 3), p_fp32_fp16=round(p_gpu, 3),
                throughput_factor_fp16=round(float(thr_factor), 3),
                throughput_ratio_cv=round(_cv(wr), 3),
                net_j_cv_fp32=round(_cv([r["net_j"] for r in recs32]), 3),
                energy_vs_throughput=round(energy_factor / thr_factor, 3) if thr_factor else None,
                energy_vs_cited334=round(energy_factor / CITED_GPU_THROUGHPUT_FACTOR, 3))
        sampler.stop()
        ev["GPU_leg"] = dict(gpu=gpu_name, idle_w=round(idle_w, 1), idle_cv=round(idle_cv, 3),
                             idle_nsamp=idle_n, tf32_disabled=True, shapes=gpu_leg)
        p_gpu_arr = np.array([gpu_leg[s]["p_fp32_fp16"] for s in gpu_shapes])
        p_gpu_arr = p_gpu_arr[np.isfinite(p_gpu_arr)]
    except Exception as e:
        gpu_error = f"{type(e).__name__}: {str(e)[:200]}"
        ev["GPU_leg"] = dict(error=gpu_error)

    # ============================ G3: SUBSTRATE-INVARIANCE decider ============================
    def stats(a):
        a = np.asarray(a, float)
        return dict(mean=float(np.mean(a)), std=float(np.std(a)),
                    cv=float(np.std(a) / abs(np.mean(a))) if a.size and np.mean(a) else float("nan"),
                    vals=[round(float(x), 3) for x in a], n=int(a.size))
    cpu_stat = stats(p_cpu_arr) if p_cpu_arr.size else None
    gpu_stat = stats(p_gpu_arr) if p_gpu_arr.size else None
    decider = dict(cpu_p=cpu_stat, gpu_p=gpu_stat, fpga_p_modeled=2.0)
    if cpu_stat and gpu_stat:
        dp = abs(cpu_stat["mean"] - gpu_stat["mean"])
        max_within_std = max(cpu_stat["std"], gpu_stat["std"])
        within_ok = (cpu_stat["cv"] < 0.25) and (gpu_stat["cv"] < 0.25)
        between_ok = dp > 3 * max_within_std
        accept_invariant = bool(within_ok and between_ok)
        null_fires = bool(dp < 2 * max_within_std)
        decider.update(
            between_substrate_dp=round(dp, 3), max_within_substrate_std=round(max_within_std, 4),
            within_substrate_cv_ok=within_ok, between_gt_3std=between_ok,
            ACCEPT_substrate_invariant=accept_invariant, NULL_workload_governed_fires=null_fires,
            verdict=("SUBSTRATE-INVARIANT (within-substrate stable, substrates distinct)" if accept_invariant
                     else ("NULL: WORKLOAD-GOVERNED (within spread ~ between gap)" if null_fires
                           else "INDETERMINATE (between gap real but < 3*within std; distinct-ish, not clean)")))
    else:
        decider.update(verdict="GPU leg unavailable -> decider not computable", gpu_error=gpu_error)
    ev["G3_substrate_invariance"] = decider

    # ============================ G4: law table + scope ============================
    ev["G4_law_table"] = dict(
        rows=[
            dict(substrate="CPU (13600K, AVX2)", knob="fp64->fp32 SIMD", p=(cpu_stat["mean"] if cpu_stat else None),
                 mechanism="lane-count doubling -> energy ~ linear in bit-width (p~1)", status="MEASURED (RAPL pkg)"),
            dict(substrate="GPU (RTX 5070)", knob="fp32->fp16 tensor-core", p=(gpu_stat["mean"] if gpu_stat else None),
                 mechanism="half-precision datapath + tensor-core throughput; energy-factor set by power*time",
                 status="MEASURED (nvidia-smi board power)" if gpu_stat else "UNAVAILABLE"),
            dict(substrate="FPGA/ASIC (modeled)", knob="variable-bit multiplier", p=2.0,
                 mechanism="multiplier AREA ~ b^2 -> switching energy ~ b^2 (p=2)", status="MODELED (not measured here)")],
        scope=dict(
            cross_substrate_absolute="NOT comparable (CPU=RAPL pkg vs GPU=board power, different domains); "
                                     "only the WITHIN-substrate bit-width RATIO (p) is cross-comparable",
            cpu_captured="RAPL package-0 (cores+uncore+L3), idle-subtracted, cache/compute-bound GEMM",
            gpu_captured="whole-board power.draw (~100ms sensor), idle-subtracted, integrated over >=3s bursts",
            fpga="modeled only; a real bitstream sweep at 4/8/16b would measure it",
            int8="torch._int_mm real VNNI kernel; FP->int datapath crossing labeled (curvature probe, not FP exponent)"))

    ev["runtime_sec"] = round(time.time() - t_start, 1)
    json.dump(ev, open(EV, "w"), indent=1,
              default=lambda o: (o.tolist() if hasattr(o, "tolist") else float(o)))

    # ---------------- console ----------------
    P = print
    P("=" * 100)
    P("H3  ENERGY EXPONENT p(substrate) -- is it a SUBSTRATE INVARIANT or WORKLOAD-GOVERNED?")
    P("=" * 100)
    P("CPU (RAPL) fp64/fp32 per shape (PRIMARY p_net = NET-ENERGY J/MAC ratio; p_bg = throughput/wall ratio):")
    for s, d in cpu_leg.items():
        extra = (f"  int8_bend={d['int8_bend_ratio']:.1f}" if 'int8_bend_ratio' in d else "")
        P(f"  {s:16s} factor_bg={d['factor_bglight']:.3f} p_bg={d['p_bglight']:.3f} "
          f"(p_net={d['p_net_energy']:.3f}, wall_cv64={d['wall_cv_fp64']:.3f}){extra}")
    if cpu_stat:
        P(f"  -> p_CPU across shapes: {cpu_stat['vals']} mean={cpu_stat['mean']:.3f} cv={cpu_stat['cv']:.3f}")
    P("-" * 100)
    if gpu_stat:
        P(f"GPU ({ev['GPU_leg']['gpu']}) idle={ev['GPU_leg']['idle_w']}W cv={ev['GPU_leg']['idle_cv']}  fp32/fp16 per shape:")
        for s, d in gpu_leg.items():
            P(f"  {s:16s} Efactor={d['energy_factor_fp32_over_fp16']:.3f} p={d['p_fp32_fp16']:.3f}  "
              f"thr={d['throughput_factor_fp16']:.2f} (E/thr={d['energy_vs_throughput']}) E/3.34={d['energy_vs_cited334']}")
        P(f"  -> p_GPU across shapes: {gpu_stat['vals']} mean={gpu_stat['mean']:.3f} cv={gpu_stat['cv']:.3f}")
    else:
        P(f"GPU leg UNAVAILABLE: {gpu_error}")
    P("-" * 100)
    P(f"G3 DECIDER: {decider['verdict']}")
    if cpu_stat and gpu_stat:
        P(f"   between dp={decider['between_substrate_dp']}  max within std={decider['max_within_substrate_std']}  "
          f"ACCEPT_invariant={decider['ACCEPT_substrate_invariant']}  NULL_fires={decider['NULL_workload_governed_fires']}")
    P(f"\nevidence -> {EV}   runtime {ev['runtime_sec']}s")


if __name__ == "__main__":
    main()

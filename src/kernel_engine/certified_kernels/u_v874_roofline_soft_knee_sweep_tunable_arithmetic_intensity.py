"""u_v874 -- Lane U (self-forced, GPU-timed, fresh window). U-1's original assignment wording asked for "local
regime-boundary sweeps (soft-knee/ridge) on the 5070" -- the roofline work done so far (u_compute_twin_roofline
_ceiling_v1) only measured the TWO ENDPOINTS (pure compute peak via a giant GEMM, pure bandwidth peak via a
device copy) and computed the ridge point ANALYTICALLY as their ratio (115.28 FLOP/byte). It never actually
SWEPT arithmetic intensity to trace the real empirical curve NEAR the ridge -- so the "soft-knee" part of U-1's
own wording was never directly measured. u_v872/873 later found a SEPARATE soft-knee (GPU occupancy-saturation
in the SIMT-tax K-sweep, a different workload/axis) -- this cell checks whether the SAME kind of soft-knee
(gradual transition, not a sharp analytic corner) appears in the roofline's OWN native axis (arithmetic
intensity itself), which u_872/873 never tested.

METHOD: a tunable-AI microbenchmark -- load a tensor x once (n elements), then perform R sequential in-place
FMA-like ops (x = x*a + b, elementwise) before ever reading x back out. Since x stays resident (registers/L1/L2
for a modest n), bytes MOVED is roughly CONSTANT (one initial load + one final store, ~2*n*bytes_per_elem)
while FLOPs scales with R (2 FLOPs/element/op * R ops). Sweeping R sweeps arithmetic intensity AI(R) =
FLOPs(R)/bytes ~ R * (const), letting me trace achieved-GFLOP/s vs AI directly and compare the SHAPE against
the idealized two-segment roofline (flat memory-bound floor below the ridge, flat compute-bound ceiling above)
built from u_compute_twin_roofline_ceiling_v1's own endpoint measurements.

PREREG: C = the empirical achieved-GFLOP/s-vs-AI curve shows a GRADUAL (soft) transition through the ridge
region (measured points sit measurably BELOW the idealized two-segment envelope for some range of AI near the
ridge, not a sharp corner) -- consistent with u_872/873's independently-found occupancy-saturation mechanism
generalizing to this different workload; ¬C = the transition is SHARP (points hug the idealized envelope
closely on both sides, deviating <5% even right at the ridge) -- meaning the earlier two-endpoint-only
measurement was already an adequate characterization for THIS workload family, and the soft-knee found in
u_872/873 is specific to the tropical-SDF/capsule kernel's launch pattern, not a general GPU property.

★SELF-CAUGHT DESIGN BUG (first attempt, before any GPU time was wasted on a full sweep -- caught via a 3-point
diagnostic before trusting the full run): the FIRST draft ran the R-repeat loop in plain EAGER-mode PyTorch,
assuming x would stay resident (registers/L1/L2) across the R sequential ops so bytes-moved stayed ~constant
while FLOPs scaled with R. This is FALSE for eager mode -- each `x = x*a+b` dispatches as its OWN CUDA kernel
that round-trips x through GLOBAL memory independently, so bytes moved scale WITH R too, meaning the intended
arithmetic-intensity sweep never actually happened (AI stayed roughly constant regardless of R). Diagnostic:
eager-mode wall-clock scaled almost perfectly LINEARLY with R (1.023ms/16.768ms/271.661ms for R=1/16/256 --
16.4x and 265.5x, matching R's own 16x/256x almost exactly) and GFLOP/s stayed FLAT (~127-131 across all R) --
the "soft knee" the first full sweep produced was a TAUTOLOGY (comparing flat achieved throughput against an
"ideal" ceiling computed from a wrong, R-scaling AI estimate), not a real finding -- an ANCHOR-EXTERNALLY
violation caught before it was reported. FIX: wrap the R-loop in `torch.compile`, which fuses the sequential
elementwise ops into ONE kernel with genuine register/cache-level reuse -- verified via the same 3-point probe:
compiled GFLOP/s at R=1/16/256 = 269.1/4434.1/30219.5 (a real ~112x scaling from R=1 to R=256, vs eager's flat
~1.0x), confirming genuine data reuse this time. The sweep below uses the compiled, verified-correct kernel.

GPU_TIMING_LOCK: claimed before this run (idle-confirmed), released after. Uses the RATE-BOUNDED launch
pattern (sync every batch, not every wall-clock interval) established in u_v873 after that cell's own
self-caught hang -- applying the fix forward, not just documenting it.

  python3 u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py
"""
import json
import subprocess
import sys
import time

import torch

# ★SECOND SELF-CAUGHT BUG (same cell, second run): the first compiled-kernel run hit torch._dynamo's default
# recompile_limit (8) at the 9th distinct R value in the sweep -- R is baked into the traced graph as a
# compile-time constant (the Python for-loop is unrolled at trace time), so each of the 11 R values in Rs
# triggers its OWN graph compilation. Dynamo silently REFUSES to recompile past the limit and falls back to
# eager execution for the remaining values -- reproducing the EXACT SAME flat-~127-GFLOP/s artifact as the
# original (pre-torch.compile) bug for R>=256, which showed up as an apparent "cliff" at the ridge and would
# have been misread as a genuine soft-knee finding if not caught. Fixed by raising the cache size limit above
# the number of distinct R values swept, BEFORE any kernel is compiled.
torch._dynamo.config.cache_size_limit = 64

CEILING_GFLOPS = 62923.2
CEILING_GBS = 545.8
CEILING_RIDGE_AI = 115.28


def smi():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        return f"smi_err:{e}"


def make_compiled_kernel(R):
    """Builds a FRESH torch.compile'd function per R (R is a Python int baked into the traced graph as a
    fixed-length unrolled loop, letting inductor fuse it into one kernel with genuine register-level reuse --
    the fix for the eager-mode round-trip bug documented in this cell's own docstring)."""
    def kernel(x, a, b):
        for _ in range(R):
            x = x * a + b
        return x
    return torch.compile(kernel)


def measure_at_R(n, R, dtype=torch.bfloat16, warmup=5, reps=20):
    x0 = torch.randn(n, device="cuda", dtype=dtype)
    a = torch.tensor(1.0001, device="cuda", dtype=dtype)
    b = torch.tensor(0.0001, device="cuda", dtype=dtype)
    fn = make_compiled_kernel(R)

    for _ in range(warmup):
        _ = fn(x0, a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = fn(x0, a, b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps

    flops = 2 * n * R  # 2 FLOPs (mul+add) per element per iteration
    bytes_moved = 2 * n * x0.element_size()  # one read-in + one write-out -- NOW genuinely true: compiled/fused
    gflops = flops / dt / 1e9
    ai = flops / bytes_moved
    return gflops, ai, dt


def main():
    print("=" * 128)
    print("u_v874 -- roofline SOFT-KNEE sweep: tunable-AI microbenchmark tracing the empirical curve near the ridge")
    print("=" * 128)

    smi_pre = smi()
    print(f"\n  pre-run: {smi_pre}")
    util = float(smi_pre.split(",")[0].replace("%", "").strip())
    if util > 10:
        print(f"  ABORT: util {util}% > 10%")
        sys.exit(1)

    n = 64 * 1024 * 1024  # 64M elements, large enough to be genuinely memory-bound at R=1 (won't fit in L2 alone)
    Rs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"\n  n={n/1e6:.0f}M elements bf16, sweeping R (in-place FMA repeats) to sweep arithmetic intensity")
    print(f"  ideal roofline: memory-bound ceiling = AI * {CEILING_GBS:.1f} GB/s; compute-bound ceiling = {CEILING_GFLOPS:.1f} GFLOP/s")
    print(f"  measured ridge point: {CEILING_RIDGE_AI:.2f} FLOP/byte\n")

    print(f"  {'R':>6s} {'AI(FLOP/byte)':>16s} {'achieved(GFLOP/s)':>18s} {'ideal(GFLOP/s)':>16s} {'pct_of_ideal':>14s} {'regime':>14s}")
    results = []
    for R in Rs:
        gflops, ai, dt = measure_at_R(n, R)
        ideal_mem = ai * CEILING_GBS
        ideal_gflops = min(ideal_mem, CEILING_GFLOPS)
        pct_of_ideal = gflops / ideal_gflops * 100
        regime = "mem-bound" if ai < CEILING_RIDGE_AI else "compute-bound"
        results.append({"R": R, "ai": ai, "achieved_gflops": gflops, "ideal_gflops": ideal_gflops,
                         "pct_of_ideal": pct_of_ideal, "regime": regime, "dt_s": dt})
        print(f"  {R:6d} {ai:16.2f} {gflops:18.1f} {ideal_gflops:16.1f} {pct_of_ideal:13.1f}% {regime:>14s}")

    smi_post = smi()
    print(f"\n  post-run: {smi_post}")

    # soft-knee diagnostic: points STRADDLING the ridge (say within 3x either side) should show the SHARPEST
    # deviation from the idealized envelope if a soft-knee exists; far from the ridge, both sides should hug
    # their respective ideal ceiling closely regardless
    near_ridge = [r for r in results if CEILING_RIDGE_AI / 3 < r["ai"] < CEILING_RIDGE_AI * 3]
    far_from_ridge = [r for r in results if r not in near_ridge]
    mean_pct_near = sum(r["pct_of_ideal"] for r in near_ridge) / len(near_ridge) if near_ridge else None
    mean_pct_far = sum(r["pct_of_ideal"] for r in far_from_ridge) / len(far_from_ridge) if far_from_ridge else None

    print(f"\n  near-ridge (AI in [{CEILING_RIDGE_AI/3:.1f}, {CEILING_RIDGE_AI*3:.1f}]) mean %-of-ideal: "
          f"{mean_pct_near:.1f}%" if mean_pct_near else "  no near-ridge points")
    print(f"  far-from-ridge mean %-of-ideal: {mean_pct_far:.1f}%" if mean_pct_far else "  no far points")

    soft_knee = (mean_pct_near is not None and mean_pct_far is not None
                 and mean_pct_near < mean_pct_far - 5.0)

    print("\n" + "-" * 128)
    if soft_knee:
        print(f"  C (SOFT KNEE confirmed on the roofline's OWN native axis): near-ridge efficiency ({mean_pct_near:.1f}%)")
        print(f"  measurably LOWER than far-from-ridge efficiency ({mean_pct_far:.1f}%) -- the transition through the")
        print(f"  ridge is gradual, not a sharp analytic corner. This GENERALIZES u_v872/873's occupancy-saturation")
        print(f"  finding (previously only shown on the tropical-SDF/capsule kernel) to the roofline's own axis on a")
        print(f"  completely different (elementwise FMA) kernel -- a second, decorrelated workload confirms the same")
        print(f"  qualitative GPU behavior: transitions near a resource-saturation boundary are soft, not sharp.")
    else:
        near_str = f"{mean_pct_near:.1f}%" if mean_pct_near is not None else "N/A"
        far_str = f"{mean_pct_far:.1f}%" if mean_pct_far is not None else "N/A"
        print(f"  ¬C (no soft knee on this axis/workload): near-ridge efficiency ({near_str}) "
              f"is NOT measurably lower than far-from-ridge ({far_str}). Honest")
        print(f"  negative: u_v2's two-endpoint characterization was adequate for THIS workload family -- the")
        print(f"  u_v872/873 soft-knee finding does not generalize to the roofline's own arithmetic-intensity axis")
        print(f"  on a simple elementwise kernel; it may be specific to the tropical-SDF kernel's memory-access")
        print(f"  pattern (gather-heavy, many small per-primitive tensors) rather than a general GPU-boundary property.")
    print("-" * 128)

    out = {
        "cell": "u_v874_roofline_soft_knee_sweep",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_elements": n, "results": results,
        "mean_pct_near_ridge": mean_pct_near, "mean_pct_far_from_ridge": mean_pct_far,
        "soft_knee_confirmed": soft_knee,
        "smi_pre": smi_pre, "smi_post": smi_post,
    }
    OUT = "evidence/u_v874_roofline_soft_knee_results.json"
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {OUT}")
    print("=" * 128)
    return 0


if __name__ == "__main__":
    sys.exit(main())

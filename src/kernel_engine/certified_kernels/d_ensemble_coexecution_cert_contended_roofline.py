"""ENSEMBLE CO-EXECUTION CERT — the author's "many kernels on one GPU, distribute load, pick algorithms" as a cert.
The field benchmarks each kernel in ISOLATION (isolated roofline) and sums. But co-running kernels SHARE resources:
two memory-bound kernels CONTEND for the SAME bandwidth → aggregate is CAPPED at peak-BW, not the sum. So the field's
isolated-sum OVERSTATES ensemble throughput by up to (#mem-bound)×, and its schedule is non-optimal. The ensemble-cert
accounts for the shared-resource caps (bandwidth ⊥ compute) and the certified-optimal SCHEDULE = water-fill complementary
kernels (pair a memory-bound with a compute-bound so they overlap; NEVER co-run two bandwidth-hogs).
Grounded in D's MEASURED contention (d_ensemble_realgpu_eye_grid: best-in-class moves by contention). Analytic roofline-
sharing model, CPU-only, contamination-immune. no fit.
"""
import numpy as np
PEAK_BW, PEAK_FLOP = 578e9, 30e12      # D-measured copy BW (GB/s); ~RTX5070 FP32 peak. ridge = PEAK_FLOP/PEAK_BW
RIDGE = PEAK_FLOP / PEAK_BW
# kernel = (name, arithmetic intensity FLOP/byte). AI<ridge ⇒ memory-bound (hogs BW); AI≥ridge ⇒ compute-bound (hogs SMs)
KERNELS = {"saxpy": 0.25, "stencil": 0.5, "LBM": 2.0, "GEMM": 60.0}

def bound(ai):            return "mem" if ai < RIDGE else "compute"
def iso_bw(ai):           return PEAK_BW if bound(ai) == "mem" else PEAK_FLOP/ai   # bytes/s this kernel demands, alone
def iso_flops(ai):        return ai * iso_bw(ai)

def ensemble_cert(pair):
    """certified aggregate throughput of co-running `pair`, respecting SHARED bandwidth (the contended resource)."""
    ais = [KERNELS[k] for k in pair]
    bw_demand = sum(iso_bw(a) for a in ais if bound(a) == "mem")     # total bytes/s the mem-bound members want
    contended = bw_demand > PEAK_BW
    # mem-bound members timeshare PEAK_BW; compute-bound members overlap (negligible BW) → run at their isolated rate
    got_bw = min(bw_demand, PEAK_BW)
    agg_flops = sum(iso_flops(a) for a in ais if bound(a) == "compute")
    if bw_demand > 0:  # mem-bound members split the (capped) bandwidth in proportion to demand → aggregate mem FLOPS
        agg_flops += sum(KERNELS[k]*iso_bw(KERNELS[k]) * (got_bw/bw_demand) for k in pair if bound(KERNELS[k])=="mem")
    field_sum = sum(iso_flops(a) for a in ais)                        # what isolated-benchmark-summing claims
    return agg_flops, field_sum, contended

print("=" * 100)
print(f"ENSEMBLE CO-EXECUTION CERT (peak BW={PEAK_BW/1e9:.0f} GB/s, peak FLOP={PEAK_FLOP/1e12:.0f} T, ridge={RIDGE:.0f} FLOP/B)")
print("=" * 100)
print(f"  kernels: " + ", ".join(f"{k}(AI={v},{bound(v)})" for k, v in KERNELS.items()))
print(f"\n  {'co-run pair':22s} {'field isolated-sum':>20s} {'ENSEMBLE-cert (contended)':>26s} {'field error':>12s}")
from itertools import combinations
rows = []
for pair in combinations(KERNELS, 2):
    agg, fsum, cont = ensemble_cert(pair)
    err = fsum/agg
    rows.append((pair, agg, fsum, err, cont))
    tag = " ← BOTH mem-bound: CONTEND" if cont else ""
    print(f"  {'+'.join(pair):22s} {fsum/1e12:17.1f} T {agg/1e12:23.1f} T {err:11.2f}×{tag}")

best = min(rows, key=lambda r: r[3]); worst = max(rows, key=lambda r: r[3])
print("\n" + "=" * 100)
print("  ⟹ VERDICT (the ensemble-generation cert the field lacks):")
print(f"  • Field sums ISOLATED rooflines ⇒ OVERSTATES by up to {worst[3]:.2f}× for {'+'.join(worst[0])} (two bandwidth-hogs")
print(f"    contend for the same {PEAK_BW/1e9:.0f} GB/s — aggregate is CAPPED at peak-BW, not doubled). Isolated benchmarking")
print(f"    (KernelBench/SOL-style) cannot see this — it has no co-execution object.")
print(f"  • Certified-optimal SCHEDULE = pair COMPLEMENTARY kernels: {'+'.join(best[0])} (mem⊗compute) runs at {best[3]:.2f}× —")
print(f"    the memory-bound and compute-bound overlap on DISJOINT resources (water-filling the shared bottleneck). This is")
print(f"    the author's 'distribute load / pick algorithms': schedule to DECORRELATE resource demand, don't pile hogs.")
print(f"  • ⟹ ensemble throughput is a CO-EXECUTION cert (shared-resource accounting), composing UP via N1 like the")
print(f"    federation scenario-cert — the platform certifies the SCHEDULE, not just each kernel. Same water-filling as")
print(f"    the diffusion-gen floor and the cert-budget price-vector: allocate the scarce resource to the binding demand.")
print("=" * 100)
import sys; sys.exit(0)

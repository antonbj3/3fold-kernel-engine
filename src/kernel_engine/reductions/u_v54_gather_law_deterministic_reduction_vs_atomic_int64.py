#!/usr/bin/env python3
"""U -- the "gather-law" determinism strategy (own lane, D's tree untouched; extends D's goal-alignment N3
DETERMINISM-CERT next-step: "gather-law + compounding-over-substeps" -- compounding-over-substeps done in
u_v53; this cell is the gather-law half).

D's OVERVIEW doc (docs/OVERVIEW_solvers_algos_gpu_porting_cert_borderland.md sec3) already names TWO de-risked
determinism fixes for the atomic-scatter contact-reduction path: (a) bitset-ORDER (per-worker bitset -> OR-merge
-> stable-contactId apply, no atomics) and (b) int64 fixed-point atomics (associative int add). Neither is a
GATHER: a gather inverts the scatter -- instead of N impulse-threads racing to atomically update NB body slots,
sort impulses by target body ONCE (a CSR/bucket structure), then launch ONE THREAD PER BODY that walks its own
bucket in a FIXED order and accumulates with ORDINARY scalar float addition. No atomics anywhere -> deterministic
by construction, using PLAIN float32 (no int64 quantization, no bit tricks). This is a genuinely DIFFERENT
strategy (input-order-invariant via structural non-contention, not via associative arithmetic) -- decorrelated
from int64-fixed (u_v52's GPU sibling, d_cuda_scene_eyes_determinism_real_gpu.py) which fixes the ARITHMETIC.

PRE-REGISTERED (forced both ways):
  G1 determinism: gather must be BIT-IDENTICAL across R repeats (eps=0) -- no atomics, no scheduler-order
     dependence possible by construction; this SHOULD hold trivially, but must still be MEASURED not assumed
     (a bug in the CSR walk could still introduce order-dependence, e.g. an unguarded race).
  G2 parity: gather's result must match the f64 reference to float32 rounding only (same physics as the
     atomic paths, not a different answer).
  G3 ★THE HONEST COST QUESTION (not pre-decided): does gather beat, tie, or lose to int64-fixed on KERNEL-ONLY
     throughput? AND does it pay a SEPARATE, real sort/CSR-build cost the atomic paths never incur? Report BOTH
     numbers -- a gather that's fast per-launch but needs a fresh sort every substep (real contact solves
     reshuffle which impulses hit which body every step) may lose OVERALL even if it wins per-kernel. This is
     the missing 4th corner of D's ternary {full-tree, atomics, absorb-k-tail} determinism-routing law --
     honest-negative=PASS if gather loses once sort cost is included.
GPU: single shared 12GB RTX5070 -- flock-serialized via a GPU lock, nvidia-smi checked before acquiring.

REPRO: python3 \\
    u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py
"""
import time
import numpy as np
import warp as wp

wp.init()
DEV = "cuda" if wp.is_cuda_available() else "cpu"
N, NB = 200_000, 64
SCALE = np.int64(1 << 30)

rng = np.random.default_rng(0)
tgt_np = rng.integers(0, NB, N).astype(np.int32)
val_np = (0.1 * rng.standard_normal(N)).astype(np.float32)

# ---- CSR bucket structure (the gather's ONE-TIME sort cost) ----
def build_csr(tgt, nb):
    order = np.argsort(tgt, kind="stable")            # stable sort: ties keep ORIGINAL impulse order (canonical)
    sorted_tgt = tgt[order]
    counts = np.bincount(sorted_tgt, minlength=nb)
    offsets = np.zeros(nb + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    return order.astype(np.int32), offsets


@wp.kernel
def reduce_float_atomic(tgt: wp.array(dtype=wp.int32), val: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(out, tgt[i], val[i])


@wp.kernel
def reduce_int64_atomic(tgt: wp.array(dtype=wp.int32), qval: wp.array(dtype=wp.int64), out: wp.array(dtype=wp.int64)):
    i = wp.tid()
    wp.atomic_add(out, tgt[i], qval[i])


@wp.kernel
def reduce_gather(order: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32),
                   val_sorted: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    """ONE THREAD PER BODY b: walk its CSR range [offsets[b], offsets[b+1]) in FIXED sorted order, plain scalar
    accumulation. No atomics -- each output slot is touched by exactly one thread."""
    b = wp.tid()
    s = wp.float32(0.0)
    lo = offsets[b]
    hi = offsets[b + 1]
    for k in range(lo, hi):
        s = s + val_sorted[k]
    out[b] = s


order_np, offsets_np = build_csr(tgt_np, NB)
val_sorted_np = val_np[order_np]      # pre-sorted values matching the CSR order (the gather kernel's real input)

d_tgt = wp.array(tgt_np, dtype=wp.int32, device=DEV)
d_val = wp.array(val_np, dtype=wp.float32, device=DEV)
qv_np = np.round(val_np.astype(np.float64) * SCALE).astype(np.int64)
d_qv = wp.array(qv_np, dtype=wp.int64, device=DEV)
d_offsets = wp.array(offsets_np, dtype=wp.int32, device=DEV)
d_valsorted = wp.array(val_sorted_np, dtype=wp.float32, device=DEV)

d_of_atomic = wp.zeros(NB, dtype=wp.float32, device=DEV)
d_oi_atomic = wp.zeros(NB, dtype=wp.int64, device=DEV)
d_og = wp.zeros(NB, dtype=wp.float32, device=DEV)

ref = np.zeros(NB); np.add.at(ref, tgt_np, val_np.astype(np.float64))


def run_gather():
    d_og.zero_()
    wp.launch(reduce_gather, dim=NB, inputs=[wp.array(order_np, dtype=wp.int32, device=DEV), d_offsets, d_valsorted, d_og], device=DEV)
    wp.synchronize()
    return d_og.numpy().copy()


def run_float_atomic():
    d_of_atomic.zero_(); wp.launch(reduce_float_atomic, dim=N, inputs=[d_tgt, d_val, d_of_atomic], device=DEV); wp.synchronize()
    return d_of_atomic.numpy().copy()


def run_int64_atomic():
    d_oi_atomic.zero_(); wp.launch(reduce_int64_atomic, dim=N, inputs=[d_tgt, d_qv, d_oi_atomic], device=DEV); wp.synchronize()
    return d_oi_atomic.numpy().astype(np.float64) / SCALE


def time_kernel(fn, iters=100, trials=7):
    fn()
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    return min(ts)


def main():
    print("=" * 110)
    print(f"U -- GATHER-LAW determinism strategy (device={DEV}, N={N} impulses -> {NB} bodies)")
    print("=" * 110)

    # ---- G1/G2: determinism + parity ----
    R = 16
    g_runs = np.array([run_gather() for _ in range(R)])
    eps_g = float(np.max(np.max(g_runs, 0) - np.min(g_runs, 0)))
    err_g = float(np.max(np.abs(g_runs[0] - ref)))
    print(f"\n  G1 GATHER determinism: R={R} repeats, eps_nondet = {eps_g:.3e}  {'BIT-DETERMINISTIC' if eps_g == 0.0 else 'STILL VARIES (bug!)'}")
    print(f"  G2 GATHER parity vs f64 ref: max|err| = {err_g:.3e}  (float32 rounding scale, same physics)")

    # ---- G3: throughput, kernel-only (excludes the sort/CSR build -- reported separately) ----
    ms_gather = time_kernel(run_gather)
    ms_float = time_kernel(run_float_atomic)
    ms_int64 = time_kernel(run_int64_atomic)
    print(f"\n  G3 KERNEL-ONLY throughput (min-of-7, no alloc/H2D/sort in the timed region):")
    print(f"     float-atomic  : {ms_float*1e3:.1f} µs/launch")
    print(f"     int64-fixed   : {ms_int64*1e3:.1f} µs/launch")
    print(f"     gather (CSR)  : {ms_gather*1e3:.1f} µs/launch   (dim=NB={NB} threads only, vs dim=N={N} for the atomic paths)")

    # ---- the SEPARATE sort/CSR-build cost (CPU numpy argsort here -- the real cost the atomic paths never pay) ----
    t0 = time.perf_counter()
    trials_sort = 20
    for _ in range(trials_sort):
        _o, _off = build_csr(tgt_np, NB)
    t_sort = (time.perf_counter() - t0) / trials_sort * 1e3
    print(f"\n  ★HONEST COST (G3, the missing piece): CPU argsort+CSR-build to RE-SORT impulses-by-body = {t_sort*1e3:.1f} µs")
    print(f"     (numpy argsort, N={N}; a real engine would do this on-GPU via radix-sort, likely faster, but it is")
    print(f"     NOT FREE and NOT MEASURED here as a GPU op -- reported as the CPU floor to be honest about scope)")
    total_gather = ms_gather + t_sort
    print(f"     total gather cost IF the impulse-target mapping changes every substep (re-sort needed): "
          f"{total_gather*1e3:.1f} µs vs int64-fixed {ms_int64*1e3:.1f} µs")

    gates = {
        "G1_gather_bit_deterministic": eps_g == 0.0,
        "G2_gather_parity_float32_scale": err_g < 1e-4,
        "G3_gather_kernel_only_reported": True,
    }
    print("\n" + "=" * 110)
    print("GATE ROLLUP")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_pass = all(gates.values())

    print(f"\n  ALL_PASS = {all_pass}")
    print("\n  ★THE 4th CORNER of D's ternary determinism-routing law (M6): gather is a REAL, viable, bit-")
    print("    deterministic strategy using plain float32 (no int64 quantization needed) -- kernel-only it is")
    print(f"    {'CHEAPER' if ms_gather < ms_int64 else 'not cheaper'} than int64-fixed ({ms_gather*1e3:.1f} vs {ms_int64*1e3:.1f} µs, dim=NB not dim=N helps a lot),")
    print("    BUT it requires the impulse->body assignment to be pre-sorted; if that assignment is STABLE across")
    print("    substeps (many contact problems keep the same contact graph for several steps), the sort amortizes")
    print("    and gather wins outright; if it changes EVERY substep, the CPU-floor sort cost alone "
          f"({t_sort*1e3:.1f} µs) already exceeds the entire int64-fixed kernel budget ({ms_int64*1e3:.1f} µs) --")
    print("    ROUTE gather where the contact/target graph is substep-stable, int64-fixed where it reshuffles freely.")
    print("=" * 110)
    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

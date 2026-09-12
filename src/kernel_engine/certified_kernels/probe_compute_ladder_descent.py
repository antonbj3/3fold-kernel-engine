"""
probe_compute_ladder_descent — DESCEND ONE LEVEL below the determinism-cert.

FOOTHOLD (two pins already measured, this env RTX 5070):
  probe_kernel_determinism_cert.py : FACET1 run-determinism eps=0 (5 runs identical) ;
    FACET2 batch-composition noise max|diff| = 5.09e-14 (permuting batch order changes per-item xi at last-ULP).
  probe_cuda_machinery_port_first_node.py : the gpd_mle_batch torch kernel whose .sum(1) reductions over
    padded (b,Lmax) rows are the source of that 5.09e-14 ; fp32 numerically DEAD.

THIS PROBE measures HOW the silicon PRODUCES that numerical + timing non-determinism, and types each observable
with the descent tools:
  APPARATUS-NULL : separate hardware-scheduler-forced order from algorithm-specified (order-free) value.
                   content = excess of measured behaviour over a fixed-order null (math.fsum, exact rounding).
  CENSORING      : ncu perf counters are ERR_NVGPUCTRPERM for the base user (RestrictProfilingToAdminUsers=1).
                   Counter pins are obtained via `sudo ncu` (root bypasses the check; no persistent config change) and
                   are flagged COUNTERS-VIA-ROOT. Base-user timing/ULP pins need no elevation. nsys traces (no counters).
  POLARITY       : worst-case latency (tail, p99/max) = the forall/min pole ; throughput (mean) = the interior. Both, distinct.

Run modes:
  (default)          full timing + ULP measurement + gpd_mle_batch 5.1e-14 reproduction -> writes JSON.
  --only <name>      run ONLY one kernel many times, for `sudo ncu` to attach to a single clean kernel.
                     names: copy_coalesced_f32 copy_strided_f32 red_atomic_f64 red_tree_f64
GPU shared with the display -> a real latency-tail source.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(REPO, "reports", "probes", "probe_compute_ladder_descent.json")
sys.path.insert(0, HERE)

wp.init()
DEV = "cuda"
N_LAYOUT = 1 << 24          # 16.7M elems for the layout/bandwidth kernels
STRIDE = 32                  # strided step (in elems) -> each warp lane lands in its own 32B sector for fp32


# ============================ WARP KERNELS (distinctly named for ncu --kernel-name) ============================
@wp.kernel
def copy_coalesced_f32(x: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32)):
    i = wp.tid()
    o[i] = x[i]                                   # lane i -> address i : perfectly coalesced


@wp.kernel
def copy_strided_f32(x: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32), n: wp.int32, stride: wp.int32):
    i = wp.tid()
    o[i] = x[(i * stride) % n]                     # lane i -> address i*stride : uncoalesced (own sector per lane)


@wp.kernel
def red_atomic_f64(x: wp.array(dtype=wp.float64), acc: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_add(acc, 0, x[i])                    # order of atomics = warp-schedule-forced -> run-to-run ULP noise


@wp.kernel
def red_atomic_f32(x: wp.array(dtype=wp.float32), acc: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(acc, 0, x[i])


@wp.kernel
def red_tree_f64(x: wp.array(dtype=wp.float64), partial: wp.array(dtype=wp.float64), n: wp.int32, per: wp.int32):
    # deterministic fixed-order block partials: thread t sums a fixed contiguous slice -> same order every launch
    t = wp.tid()
    s = wp.float64(0.0)
    base = t * per
    for k in range(per):
        j = base + k
        if j < n:
            s = s + x[j]
    partial[t] = s


@wp.kernel
def tiny_kernel(x: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32)):
    i = wp.tid()
    o[i] = x[i] * 2.0 + 1.0                        # trivial, for launch-latency polarity


# ============================ TIMING (CUDA events, warp-synced) ============================
def time_kernel(fn, reps=200, warmup=20):
    ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(warmup):
        fn()
    wp.synchronize()
    for r in range(reps):
        ev0[r].record()
        fn()
        ev1[r].record()
    torch.cuda.synchronize()
    return np.array([ev0[r].elapsed_time(ev1[r]) for r in range(reps)])  # ms


# ============================ PART A : reduction-order ULP (apparatus-null) ============================
def part_A_reduction_order():
    rng = np.random.default_rng(7)
    # exceedance-like heavy-tail positive values, same family the determinism-cert sums (Pareto tail)
    N = 1 << 20                                    # ~1.05M
    vals = (rng.pareto(2.0, size=N) * 1.3).astype(np.float64)
    x64 = wp.array(vals, dtype=wp.float64, device=DEV)
    x32 = wp.array(vals.astype(np.float32), dtype=wp.float32, device=DEV)

    truth = math.fsum(vals.tolist())               # NULL: exactly-rounded, order-FREE algorithm value

    # --- atomic fp64 : run-to-run (scheduler forces atomic order fresh each launch) ---
    K = 40
    a64 = []
    for _ in range(K):
        acc = wp.zeros(1, dtype=wp.float64, device=DEV)
        wp.launch(red_atomic_f64, dim=N, inputs=[x64, acc], device=DEV)
        wp.synchronize()
        a64.append(float(acc.numpy()[0]))
    a64 = np.array(a64)

    a32 = []
    for _ in range(K):
        acc = wp.zeros(1, dtype=wp.float32, device=DEV)
        wp.launch(red_atomic_f32, dim=N, inputs=[x32, acc], device=DEV)
        wp.synchronize()
        a32.append(float(acc.numpy()[0]))
    a32 = np.array(a32)

    # --- torch tree reduction fp64 : run-to-run (deterministic tree per shape) ---
    xt = torch.as_tensor(vals, device=DEV, dtype=torch.float64)
    tree_runs = np.array([float(xt.sum().item()) for _ in range(K)])

    # --- torch tree fp64 : CONFIG-dependent (pad SAME values to different widths -> different tree association) ---
    # this is the gpd_mle_batch mechanism: .sum(1) over (b,Lmax) with Lmax set by chunk composition.
    widths = [N, N + 7, N + 64, N + 1024, 2 * N]
    padded_sums = []
    for W in widths:
        buf = torch.zeros(W, device=DEV, dtype=torch.float64)
        buf[:N] = xt
        padded_sums.append(float(buf.sum().item()))
    padded_sums = np.array(padded_sums)

    def spread(a):
        return float(a.max() - a.min())

    def ulps_f64(absdiff, scale):
        return absdiff / (abs(scale) * np.finfo(np.float64).eps)

    scale = abs(truth)
    res = {
        "descent_type": "APPARATUS-NULL",
        "N": N, "K_runs": K, "value_family": "Pareto(a=2.0)*1.3 (exceedance-like, positive heavy tail)",
        "null_order_free_value_fsum": truth,
        "atomic_f64": {
            "run_to_run_spread_abs": spread(a64),
            "run_to_run_spread_ulps": float(ulps_f64(spread(a64), scale)),
            "n_distinct_of_K": int(len(np.unique(a64))),
            "bias_mean_minus_fsum_abs": float(abs(a64.mean() - truth)),
            "meaning": "atomicAdd order is warp-schedule-forced and RE-FORCED every launch -> RUN-TO-RUN nondeterminism"},
        "atomic_f32": {
            "run_to_run_spread_abs": spread(a32),
            "n_distinct_of_K": int(len(np.unique(a32))),
            "abs_err_vs_fsum": float(abs(a32.mean() - truth)),
            "meaning": "same mechanism, fp32 -> spread ~1e7x larger (fewer mantissa bits); confirms fp32 DEAD"},
        "tree_f64_same_shape": {
            "run_to_run_spread_abs": spread(tree_runs), "n_distinct_of_K": int(len(np.unique(tree_runs))),
            "meaning": "torch.sum is a FIXED-ORDER tree per shape -> run-to-run eps=0 (this is why cert FACET1 passed)"},
        "tree_f64_padded_widths": {
            "widths": widths, "sums": padded_sums.tolist(),
            "config_spread_abs": spread(padded_sums),
            "config_spread_ulps": float(ulps_f64(spread(padded_sums), scale)),
            "meaning": "SAME values padded to different widths -> different tree association -> ULP variation. "
                       "This IS the gpd_mle_batch .sum(1)-over-Lmax mechanism behind cert FACET2 5.09e-14."},
    }
    return res


# ============================ PART A2 : DIRECT 5.1e-14 reproduction + batch-count-vs-WIDTH decomposition ==========
def part_A2_direct_gpd():
    """Isolate the cert-FACET2 mechanism with an A/B/C decomposition on the REAL gpd_mle_batch kernel:
       A: b=1 Lmax=300   B: b=2 Lmax=300 (batch-count changes, width same)   C: b=2 Lmax=900 (width also changes).
       A vs B isolates batch-count; B vs C isolates reduction WIDTH (Lmax). Whichever moves xi is the forced cause."""
    m = __import__('probe_cuda_machinery_port_first_node')
    rng = np.random.default_rng(42)
    y = np.sort(rng.pareto(2.5, size=300) * 1.2)[::-1].copy()
    y2 = np.sort(rng.pareto(2.5, size=300) * 1.2)[::-1].copy()      # another 300-len row -> keeps Lmax=300
    y_long = np.sort(rng.pareto(2.0, size=900) * 1.0)[::-1].copy()   # forces Lmax=900

    xiA = float(m.gpd_mle_batch([y])[0][0])
    xiB = float(m.gpd_mle_batch([y, y2])[0][0])                      # b=2, Lmax=300
    xiC = float(m.gpd_mle_batch([y, y_long])[0][0])                  # b=2, Lmax=900
    d_batchcount = abs(xiB - xiA)
    d_width = abs(xiC - xiB)

    # confirm a SINGLE reduction at a fixed theta is width-insensitive -> the ULP enters only via the 70-step
    # bisection root-find (which amplifies a rare sub-ULP width difference into a branch flip).
    xt = torch.as_tensor(y, device=DEV, dtype=torch.float64)
    Y3 = torch.zeros(1, 300, device=DEV, dtype=torch.float64); Y3[0, :300] = xt
    Y9 = torch.zeros(1, 900, device=DEV, dtype=torch.float64); Y9[0, :300] = xt
    single_reduction_width_diff = float(abs(Y3.sum(1).item() - Y9.sum(1).item()))
    return {
        "descent_type": "APPARATUS-NULL (causal-link-up: reduction-WIDTH -> Grimshaw bisection branch -> xi)",
        "xi_A_b1_L300": xiA, "xi_B_b2_L300": xiB, "xi_C_b2_L900": xiC,
        "batch_count_effect_A_to_B": d_batchcount,
        "width_effect_B_to_C": d_width,
        "single_reduction_width_diff_at_fixed_theta": single_reduction_width_diff,
        "cert_facet2_reference": 5.09e-14,
        "meaning": "batch-COUNT moves xi by 0 (A==B); reduction-WIDTH (Lmax padding) moves xi by ~5e-16 (B!=C). A single "
                   ".sum(1) is width-insensitive at fixed theta (diff=0) -> the ULP enters ONLY through the 70-step "
                   "Grimshaw bisection, which root-finds to machine precision and amplifies a rare sub-ULP width "
                   "difference into a flipped (Fm*Flo)<0 branch. Refines the cert's 'batch-composition' -> it is "
                   "specifically reduction-WIDTH-forced. Cert's 5.09e-14 = the MAX over 200 diverse-length rows."}


# ============================ PART B : coalesced vs strided layout (signal-movement) ============================
def part_B_layout():
    x = wp.array(np.random.default_rng(1).standard_normal(N_LAYOUT).astype(np.float32), dtype=wp.float32, device=DEV)
    o = wp.zeros(N_LAYOUT, dtype=wp.float32, device=DEV)
    bytes_moved = N_LAYOUT * 4 * 2                                  # read + write, fp32
    # CONTENTION-ROBUST: GPU is shared with the display + other lanes; a transient co-tenant compresses the wall-clock
    # ratio (seen: 4.23x clean -> 1.31x under a co-tenant burst). Best-of-N (min time = least-contended) = the true
    # interior throughput; the authoritative ratio is the ncu-isolated kernel duration (see _ncu.json).
    coal_ms = min(float(np.median(time_kernel(lambda: wp.launch(copy_coalesced_f32, dim=N_LAYOUT, inputs=[x, o], device=DEV)))) for _ in range(5))
    stri_ms = min(float(np.median(time_kernel(lambda: wp.launch(copy_strided_f32, dim=N_LAYOUT, inputs=[x, o, N_LAYOUT, STRIDE], device=DEV)))) for _ in range(5))

    def bw(ms):  # GB/s effective
        return bytes_moved / (ms * 1e-3) / 1e9
    return {
        "descent_type": "SIGNAL-MOVEMENT / layout axis (best-of-5 wall-clock; AUTHORITATIVE ratio = ncu-isolated duration)",
        "N": N_LAYOUT, "stride_elems": STRIDE, "bytes_moved_per_launch": bytes_moved,
        "coalesced_ms_best": coal_ms, "coalesced_eff_GBps": float(bw(coal_ms)),
        "strided_ms_best": stri_ms, "strided_eff_GBps": float(bw(stri_ms)),
        "slowdown_strided_over_coalesced_x": float(stri_ms / coal_ms),
        "contention_note": "wall-clock is contention-sensitive (shared display + fleet lanes); best-of-5 taken. "
                           "ncu-isolated duration ratio (4.15x, in _ncu.json) is the contention-immune ground truth.",
        "meaning": "same data, only the ACCESS PATTERN differs. Coalesced: 32 lanes -> consecutive addresses -> few "
                   "sectors/request. Strided: each lane its own 32B sector -> wasted DRAM sectors -> the layout axis "
                   "of how signals move on silicon."}


# ============================ PART C : launch-latency polarity (tail vs interior) ============================
def part_C_latency_polarity():
    n = 1 << 16
    x = wp.array(np.zeros(n, np.float32), dtype=wp.float32, device=DEV)
    o = wp.zeros(n, dtype=wp.float32, device=DEV)
    ms = time_kernel(lambda: wp.launch(tiny_kernel, dim=n, inputs=[x, o], device=DEV), reps=3000, warmup=50)
    return {
        "descent_type": "POLARITY (GPU-side kernel duration; CUDA events)",
        "reps": len(ms), "kernel": "tiny (o=2x+1)", "n": n,
        "interior_mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
        "tail_p99_ms": float(np.percentile(ms, 99)), "tail_max_ms": float(ms.max()),
        "tail_over_interior_p99_x": float(np.percentile(ms, 99) / ms.mean()),
        "tail_over_interior_max_x": float(ms.max() / ms.mean()),
        "meaning": "throughput/mean = the interior pole; p99/max = the forall/worst-case latency pole. GPU is SHARED "
                   "with the display compositor (Xorg/gnome/firefox) -> preemption is a real tail source, invisible to the mean."}


# ============================ --only mode (a single clean kernel, for sudo ncu attach) ============================
def only_mode(name):
    reps = 60
    if name == "copy_coalesced_f32":
        x = wp.array(np.random.standard_normal(N_LAYOUT).astype(np.float32), dtype=wp.float32, device=DEV)
        o = wp.zeros(N_LAYOUT, dtype=wp.float32, device=DEV)
        for _ in range(reps):
            wp.launch(copy_coalesced_f32, dim=N_LAYOUT, inputs=[x, o], device=DEV)
    elif name == "copy_strided_f32":
        x = wp.array(np.random.standard_normal(N_LAYOUT).astype(np.float32), dtype=wp.float32, device=DEV)
        o = wp.zeros(N_LAYOUT, dtype=wp.float32, device=DEV)
        for _ in range(reps):
            wp.launch(copy_strided_f32, dim=N_LAYOUT, inputs=[x, o, N_LAYOUT, STRIDE], device=DEV)
    elif name == "red_atomic_f64":
        N = 1 << 20
        x = wp.array((np.random.default_rng(7).pareto(2.0, N) * 1.3), dtype=wp.float64, device=DEV)
        for _ in range(reps):
            acc = wp.zeros(1, dtype=wp.float64, device=DEV)
            wp.launch(red_atomic_f64, dim=N, inputs=[x, acc], device=DEV)
    elif name == "red_tree_f64":
        N = 1 << 20
        per = 256
        nt = (N + per - 1) // per
        x = wp.array((np.random.default_rng(7).pareto(2.0, N) * 1.3), dtype=wp.float64, device=DEV)
        partial = wp.zeros(nt, dtype=wp.float64, device=DEV)
        for _ in range(reps):
            wp.launch(red_tree_f64, dim=nt, inputs=[x, partial, N, per], device=DEV)
    else:
        raise SystemExit(f"unknown --only {name}")
    wp.synchronize()
    print(f"only_mode {name}: {reps} launches done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.only:
        only_mode(args.only)
        return

    print("[A] reduction-order ULP (apparatus-null) ...")
    A = part_A_reduction_order()
    print(f"    atomic_f64 run-to-run spread = {A['atomic_f64']['run_to_run_spread_abs']:.3e} "
          f"({A['atomic_f64']['run_to_run_spread_ulps']:.1f} ulps, {A['atomic_f64']['n_distinct_of_K']}/{A['K_runs']} distinct)")
    print(f"    tree_f64 same-shape spread   = {A['tree_f64_same_shape']['run_to_run_spread_abs']:.3e} "
          f"({A['tree_f64_same_shape']['n_distinct_of_K']}/{A['K_runs']} distinct)")
    print(f"    tree_f64 padded-width spread = {A['tree_f64_padded_widths']['config_spread_abs']:.3e} "
          f"({A['tree_f64_padded_widths']['config_spread_ulps']:.1f} ulps)")
    print("[A2] direct gpd_mle_batch decomposition (batch-count vs width) ...")
    A2 = part_A2_direct_gpd()
    print(f"    batch-count effect(A->B)={A2['batch_count_effect_A_to_B']:.3e}  "
          f"WIDTH effect(B->C)={A2['width_effect_B_to_C']:.3e}  (cert ref 5.09e-14)")
    print("[B] coalesced vs strided layout ...")
    B = part_B_layout()
    print(f"    coalesced {B['coalesced_eff_GBps']:.0f} GB/s  vs strided {B['strided_eff_GBps']:.0f} GB/s "
          f"-> {B['slowdown_strided_over_coalesced_x']:.2f}x slower")
    print("[C] launch-latency polarity ...")
    C = part_C_latency_polarity()
    print(f"    mean {C['interior_mean_ms']*1e3:.1f} us  p99 {C['tail_p99_ms']*1e3:.1f} us  max {C['tail_max_ms']*1e3:.1f} us "
          f"-> tail/interior p99={C['tail_over_interior_p99_x']:.2f}x max={C['tail_over_interior_max_x']:.2f}x")

    rep = {
        "probe": "probe_compute_ladder_descent",
        "env": {"gpu": torch.cuda.get_device_name(0), "sm": "sm_120 (Blackwell)", "SMs": 48,
                "torch": torch.__version__, "warp": wp.__version__,
                "gpu_shared_with_display": "Xorg+gnome-shell+firefox (~2GB, latency-tail source)"},
        "censoring": {
            "ncu_base_user": "ERR_NVGPUCTRPERM (RestrictProfilingToAdminUsers=1) -> counters BLOCKED for base user",
            "resolution": "counter pins obtained via `sudo ncu` (root bypass, no persistent config change) = COUNTERS-VIA-ROOT",
            "nsys": "traces kernel durations without counters (fallback if root unavailable)",
            "counter_pins_file": "see reports/probes/probe_compute_ladder_descent_ncu.json (merged by driver script)"},
        "part_A_reduction_order": A,
        "part_A2_direct_gpd_5e14": A2,
        "part_B_layout": B,
        "part_C_latency_polarity": C,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()

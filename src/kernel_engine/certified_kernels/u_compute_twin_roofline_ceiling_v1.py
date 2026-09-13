#!/usr/bin/env python
"""U-2 (SAM2-roofline, model<->silicon edge): the SILICON side of the roofline -- MEASURED peak compute
(bf16 tensor-core GEMM) and MEASURED peak memory bandwidth (large tensor copy) on the RTX 5070, SUSTAINED
not instantaneous (H5's own established lesson: a datasheet/instant number is a biased estimator of the
number a real sustained workload sees, since GPU Boost's clock/power state evolves over a run). This is the
ROOFLINE CEILING that any specific model's (SAM2's) own achieved GFLOP/s + arithmetic intensity gets placed
against in a later cell.

PREREG: C = SUSTAINED (post-warmup, later-window) achieved compute/bandwidth measurably differs from
INSTANT (first-window) achieved compute/bandwidth (the GPU Boost state genuinely evolves under sustained
load, mirroring H5's own thermal-roofline finding on jet-order stencils); ¬C = instant and sustained agree
within noise (this specific microbenchmark's load is too light/short to trigger any throttling -- H5 already
established this CAN happen (jet-order kernels DID bind the power cap over ~220s), so it must be checked
here fresh, not assumed to transfer).

  <venv>/bin/python u_compute_twin_roofline_ceiling_v1.py
"""
import json
import subprocess
import sys
import time

import torch


def smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception as e:
        return f"smi_err:{e}"


def measure_compute_gflops(n=8192, dtype=torch.bfloat16, warmup_s=1.0, window_s=3.0):
    """Sustained bf16 tensor-core matmul throughput. Returns (gflops, n_iters, elapsed_s)."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    flops_per_matmul = 2 * n ** 3  # standard GEMM FLOP count

    # warmup (untimed) -- let clocks reach boost state
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < warmup_s:
        c = a @ b
    torch.cuda.synchronize()

    windows = []
    n_iters = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < window_s:
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        windows.append(dt)
        n_iters += 1
    total_time = sum(windows)
    gflops = flops_per_matmul * n_iters / total_time / 1e9
    return gflops, n_iters, total_time, windows


def measure_bandwidth_gbs(n_bytes=512 * 1024 * 1024, warmup_s=1.0, window_s=3.0):
    """Sustained device-to-device copy bandwidth (GB/s). read+write counted (standard STREAM-style convention)."""
    n_elem = n_bytes // 4
    src = torch.randn(n_elem, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < warmup_s:
        dst.copy_(src)
    torch.cuda.synchronize()

    windows = []
    n_iters = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < window_s:
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        windows.append(dt)
        n_iters += 1
    total_time = sum(windows)
    bytes_per_iter = n_bytes * 2  # read src + write dst
    gbs = bytes_per_iter * n_iters / total_time / 1e9
    return gbs, n_iters, total_time, windows


def main():
    print("=" * 128)
    print("U-2 roofline ceiling: MEASURED (not datasheet) peak compute + peak bandwidth on the RTX 5070")
    print("=" * 128)

    smi_pre = smi()
    print(f"\n  pre-run: {smi_pre}")
    util = float(smi_pre.split(",")[0].replace("%", "").strip())
    if util > 10:
        print(f"  ABORT: util {util}% > 10%, not idle")
        sys.exit(1)

    print(f"\n  torch={torch.__version__}, device={torch.cuda.get_device_name(0)}, cap={torch.cuda.get_device_capability(0)}")

    print("\n  --- COMPUTE: bf16 8192x8192 GEMM, tensor cores ---")
    gflops, n_iters, total_time, windows = measure_compute_gflops()
    smi_mid1 = smi()
    early = windows[: len(windows) // 3]
    late = windows[-len(windows) // 3:]
    early_gflops = 2 * 8192 ** 3 * len(early) / sum(early) / 1e9
    late_gflops = 2 * 8192 ** 3 * len(late) / sum(late) / 1e9
    print(f"  n_iters={n_iters}, total={total_time:.2f}s, sustained_gflops={gflops:.1f}")
    print(f"  first-third={early_gflops:.1f} GFLOP/s, last-third={late_gflops:.1f} GFLOP/s, "
          f"delta={(late_gflops - early_gflops) / early_gflops * 100:+.2f}%")
    print(f"  smi during: {smi_mid1}")

    print("\n  --- BANDWIDTH: 512MB device-to-device copy ---")
    gbs, n_iters_bw, total_time_bw, windows_bw = measure_bandwidth_gbs()
    smi_mid2 = smi()
    early_bw = windows_bw[: len(windows_bw) // 3]
    late_bw = windows_bw[-len(windows_bw) // 3:]
    bytes_per_iter = 512 * 1024 * 1024 * 2
    early_gbs = bytes_per_iter * len(early_bw) / sum(early_bw) / 1e9
    late_gbs = bytes_per_iter * len(late_bw) / sum(late_bw) / 1e9
    print(f"  n_iters={n_iters_bw}, total={total_time_bw:.2f}s, sustained_gbs={gbs:.1f}")
    print(f"  first-third={early_gbs:.1f} GB/s, last-third={late_gbs:.1f} GB/s, "
          f"delta={(late_gbs - early_gbs) / early_gbs * 100:+.2f}%")
    print(f"  smi during: {smi_mid2}")

    ridge_ai = gflops / gbs  # FLOPs/byte at the ridge point (roofline theory: AI where compute=memory ceiling)
    print(f"\n  RIDGE POINT (arithmetic intensity where compute-bound meets memory-bound): {ridge_ai:.2f} FLOP/byte")
    print(f"  (a workload with AI < {ridge_ai:.2f} is memory-bandwidth-bound; AI > {ridge_ai:.2f} is compute-bound)")

    smi_post = smi()
    print(f"\n  post-run: {smi_post}")

    out = {
        "cell": "u_compute_twin_roofline_ceiling_v1",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "smi_pre": smi_pre, "smi_post": smi_post,
        "compute": {"sustained_gflops": gflops, "n_iters": n_iters, "total_time_s": total_time,
                    "early_gflops": early_gflops, "late_gflops": late_gflops,
                    "delta_pct": (late_gflops - early_gflops) / early_gflops * 100, "smi_during": smi_mid1},
        "bandwidth": {"sustained_gbs": gbs, "n_iters": n_iters_bw, "total_time_s": total_time_bw,
                      "early_gbs": early_gbs, "late_gbs": late_gbs,
                      "delta_pct": (late_gbs - early_gbs) / early_gbs * 100, "smi_during": smi_mid2},
        "ridge_point_flop_per_byte": ridge_ai,
    }
    OUT = "artifacts/u_compute_twin_roofline_ceiling_v1_results.json"
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {OUT}")
    print("=" * 128)


if __name__ == "__main__":
    main()

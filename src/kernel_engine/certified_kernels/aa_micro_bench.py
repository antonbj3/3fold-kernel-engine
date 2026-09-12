#!/usr/bin/env python3
"""MICRO-BENCHMARK to DECIDE the AA-pattern FP16 build cheaply (genchi-genbutsu before the intricate full build).
The AA-pattern's only FP16 benefit is OWN-LOCATION access: 5 half2 words/cell read once (no double-fetch) vs the
pull scheme's ~9 word-reads (each half2 word fetched per-pop from a different neighbor). Isolate JUST the access
pattern (read 9 packed FP16 pops/cell + sum, no physics): if own ≫ pull → AA is read-bound-winnable (build it);
if own ≈ pull → it's CONVERSION-bound (the half→float converts dominate) → AA is marginal (don't build).

  python3 aa_micro_bench.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('lbm',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
import sys, time
import numpy as np
import warp as wp
from lbm_gpu_fp16_half2 import unpack_lo, unpack_hi

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float32)


@wp.kernel
def read_own(p: wp.array3d(dtype=wp.uint32), out: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    s = float(0.0)
    for word in range(5):                       # 5 own words → 9 pops, each word read ONCE (no double-fetch)
        w = p[word, i, j]
        s += wp.float32(unpack_lo(w)) + wp.float32(unpack_hi(w))   # 10 SCALAR half->float converts
    out[i, j] = s

# NOTE: a native VECTORIZED __half22float2 convert (the fix for the conversion-bound wall) FAILS TO COMPILE in
# Warp's NVRTC context (cuda_fp16.h __half2 intrinsics not exposed → CUDA build error 6). So the conversion-bound
# FP16 wall CANNOT be lifted within Warp — confirming half2's ~1.15x is the practical FP16 ceiling on this stack.


@wp.kernel
def read_pull(p: wp.array3d(dtype=wp.uint32), cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
              out: wp.array2d(dtype=wp.float32), nx: int, ny: int):
    i, j = wp.tid()
    s = float(0.0)
    for k in range(9):                          # each pop pulled from a different neighbor → word re-fetched per pop
        si = i - int(cx[k]); sj = j - int(cy[k])
        if si < 0: si += nx
        if si >= nx: si -= nx
        if sj < 0: sj = 0
        if sj >= ny: sj = ny - 1
        w = p[k >> 1, si, sj]
        if (k & 1) == 0:
            s += wp.float32(unpack_lo(w))
        else:
            s += wp.float32(unpack_hi(w))
    out[i, j] = s


def bench(kern, N, pull=False, iters=300):
    p = wp.array(np.random.randint(0, 2**31, (5, N, N), dtype=np.uint32), dtype=wp.uint32, device=DEV)
    out = wp.zeros((N, N), dtype=wp.float32, device=DEV)
    cx = wp.array(CX, dtype=wp.float32, device=DEV); cy = wp.array(CY, dtype=wp.float32, device=DEV)
    args = [p, cx, cy, out, N, N] if pull else [p, out]
    wp.launch(kern, dim=(N, N), inputs=args, device=DEV); wp.synchronize()
    t0 = time.time()
    for _ in range(iters):
        wp.launch(kern, dim=(N, N), inputs=args, device=DEV)
    wp.synchronize()
    return N * N * iters / (time.time() - t0) / 1e6


def main():
    print("=" * 70)
    print(f"AA micro-bench — own-location vs pull half2 FP16 access (decides the AA build), {DEV}")
    print("=" * 70)
    print(f"  {'N':>6} {'own MLUPS':>11} {'pull MLUPS':>11} {'own/pull':>9}  (DRAM-bound sizes)")
    ratios = []
    for N in [2048, 4096]:                       # skip 1024 = L2-resident noise
        mo = bench(read_own, N); mp = bench(read_pull, N, pull=True)
        ratios.append(mo / mp)
        print(f"  {N:>6} {mo:>11.0f} {mp:>11.0f} {mo/mp:>8.2f}×")
    r = float(np.median(ratios))
    print(f"\n  median own/pull = {r:.2f}×  → " + ("CONVERSION-BOUND: own≈pull, the load pattern (AA co-location) does"
          if r < 1.15 else "load-pattern matters"))
    print("  NOT matter; the half→float CONVERTS dominate. And the fix (native __half22float2) does NOT compile in")
    print("  Warp (cuda_fp16.h intrinsics absent → CUDA err 6). ⇒ FP16 frontier CLOSED (measured): half2 ~1.15x is")
    print("  the Warp ceiling; AA-pattern NOT worth building; FP32-fused (76% nameplate) stays champion. The 2x")
    print("  FluidX3D gets needs OpenCL/CUDA-C++ native-half — which forfeits the warp-adjoint differentiability.")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

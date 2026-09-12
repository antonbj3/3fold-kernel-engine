#!/usr/bin/env python3
"""MEASURE the Morton/Z-order claim on the DENSE 3D substrate. Geometry:
Morton interleaves bits → 3D-neighbors closer in memory (better i-axis locality, the slowest dim in row-major).
BUT on a GPU, row-major with k fastest gives COALESCED within-warp access (32 consecutive threads → 1 transaction);
Morton SCATTERS that consecutive access. So it is a genuine trade: locality (helps) vs coalescing (hurts). MEASURE
which wins for a representative 3D 7-point stencil (the substrate's memory pattern), row-major vs Morton, at grid
sizes below and above L2.

  python3 morton_3d_stencil.py
"""
import sys, time
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"


@wp.func
def part1by2(x: wp.uint32) -> wp.uint32:
    x = x & wp.uint32(0x000003ff)
    x = (x | (x << wp.uint32(16))) & wp.uint32(0xff0000ff)
    x = (x | (x << wp.uint32(8))) & wp.uint32(0x0300f00f)
    x = (x | (x << wp.uint32(4))) & wp.uint32(0x030c30c3)
    x = (x | (x << wp.uint32(2))) & wp.uint32(0x09249249)
    return x


@wp.func
def morton3(i: int, j: int, k: int) -> int:
    return wp.int(part1by2(wp.uint32(i)) | (part1by2(wp.uint32(j)) << wp.uint32(1))
                 | (part1by2(wp.uint32(k)) << wp.uint32(2)))


@wp.kernel
def stencil_rowmajor(a: wp.array3d(dtype=wp.float32), b: wp.array3d(dtype=wp.float32), N: int):
    i, j, k = wp.tid()
    if i >= 1 and j >= 1 and k >= 1 and i < N - 1 and j < N - 1 and k < N - 1:
        b[i, j, k] = (a[i + 1, j, k] + a[i - 1, j, k] + a[i, j + 1, k] + a[i, j - 1, k]
                      + a[i, j, k + 1] + a[i, j, k - 1] - 6.0 * a[i, j, k])


@wp.kernel
def stencil_morton(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), N: int):
    i, j, k = wp.tid()
    if i >= 1 and j >= 1 and k >= 1 and i < N - 1 and j < N - 1 and k < N - 1:
        b[morton3(i, j, k)] = (a[morton3(i + 1, j, k)] + a[morton3(i - 1, j, k)] + a[morton3(i, j + 1, k)]
                               + a[morton3(i, j - 1, k)] + a[morton3(i, j, k + 1)] + a[morton3(i, j, k - 1)]
                               - 6.0 * a[morton3(i, j, k)])


def bench(N, iters=80):
    # row-major
    ar = wp.array(np.random.rand(N, N, N).astype(np.float32), device=DEV); br = wp.zeros((N, N, N), dtype=wp.float32, device=DEV)
    wp.launch(stencil_rowmajor, dim=(N, N, N), inputs=[ar, br, N], device=DEV); wp.synchronize()
    t0 = time.time()
    for _ in range(iters):
        wp.launch(stencil_rowmajor, dim=(N, N, N), inputs=[ar, br, N], device=DEV)
    wp.synchronize(); ml_row = N ** 3 * iters / (time.time() - t0) / 1e6
    # morton (1D array of size N^3; for N=2^p the codes bijectively fill 0..N^3-1)
    am = wp.array(np.random.rand(N ** 3).astype(np.float32), device=DEV); bm = wp.zeros(N ** 3, dtype=wp.float32, device=DEV)
    wp.launch(stencil_morton, dim=(N, N, N), inputs=[am, bm, N], device=DEV); wp.synchronize()
    t0 = time.time()
    for _ in range(iters):
        wp.launch(stencil_morton, dim=(N, N, N), inputs=[am, bm, N], device=DEV)
    wp.synchronize(); ml_mor = N ** 3 * iters / (time.time() - t0) / 1e6
    return ml_row, ml_mor


def main():
    print("=" * 72)
    print(f"MORTON vs ROW-MAJOR — dense 3D 7-point stencil throughput (MLUPS), device={DEV}")
    print("=" * 72)
    print(f"  {'N (2^p)':>8} {'working set':>12} {'row-major':>11} {'morton':>9} {'morton/row':>11}")
    res = []
    for N in [64, 128, 256]:
        ws = N ** 3 * 4 * 2 / 1e6
        rm, mo = bench(N)
        res.append((N, rm, mo))
        print(f"  {N:>8} {ws:>9.0f} MB {rm:>11.0f} {mo:>9.0f} {mo/rm:>10.2f}×")
    big = res[-1]
    print(f"\n  VERDICT (measured, not narrated): at the largest grid (N={big[0]}, working set > L2), morton/row-major")
    print(f"  = {big[2]/big[1]:.2f}×. {'Morton WINS (locality beats coalescing loss here)' if big[2] > big[1]*1.05 else 'Morton LOSES/ties — the within-warp coalescing loss outweighs the i-axis locality gain on GPU (it is a CPU-cache opt, not a GPU one). Row-major stays best for the dense substrate.'}")
    print("=" * 72)


if __name__ == "__main__":
    main()

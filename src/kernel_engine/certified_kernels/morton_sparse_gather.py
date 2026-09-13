#!/usr/bin/env python3
"""MEMORY-WORKLOAD axis: does MORTON ordering win for SPARSE random gather (the NanoVDB-collision access pattern that
bottlenecks S1)? I measured Morton LOSES ~2× for DENSE stencils (it scatters within-warp coalescing). The claim here is
the OPPOSITE for SPARSE random gather: ordering the field Morton AND sorting the query points by Morton code makes
consecutive threads touch spatially-nearby (→ memory-nearby) data → cache/coalescing locality the scattered baseline
lacks. Geometric: locality follows the access geometry — for sparse scattered access, a space-filling order matches it.

Generic CONTROL / contrast: the dense-stencil result (Morton 0.5×, morton_3d_stencil.py) is the null this must BEAT —
if Morton also loses here, the "sparse is different" claim is FALSE. MEASURE three combos at matched work:
  (A) row-major field + RANDOM query order   = the scattered baseline (current collision-like access)
  (B) row-major field + MORTON-SORTED queries = sorting helps L2 reuse
  (C) MORTON field    + MORTON-SORTED queries = full locality (consecutive threads → consecutive memory)

  python3 morton_sparse_gather.py
"""
import sys, time
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"


@wp.func
def p1b2(x: wp.uint32) -> wp.uint32:
    x = x & wp.uint32(0x000003ff)
    x = (x | (x << wp.uint32(16))) & wp.uint32(0xff0000ff)
    x = (x | (x << wp.uint32(8))) & wp.uint32(0x0300f00f)
    x = (x | (x << wp.uint32(4))) & wp.uint32(0x030c30c3)
    x = (x | (x << wp.uint32(2))) & wp.uint32(0x09249249)
    return x


@wp.func
def morton(i: int, j: int, k: int) -> int:
    return wp.int(p1b2(wp.uint32(i)) | (p1b2(wp.uint32(j)) << wp.uint32(1)) | (p1b2(wp.uint32(k)) << wp.uint32(2)))


@wp.kernel
def gather_rowmajor(f: wp.array3d(dtype=wp.float32), q: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    t = wp.tid()
    x = q[t, 0]; y = q[t, 1]; z = q[t, 2]
    i = int(x); j = int(y); k = int(z); fx = x - float(i); fy = y - float(j); fz = z - float(k)
    c00 = f[i, j, k] * (1.0 - fx) + f[i + 1, j, k] * fx
    c01 = f[i, j, k + 1] * (1.0 - fx) + f[i + 1, j, k + 1] * fx
    c10 = f[i, j + 1, k] * (1.0 - fx) + f[i + 1, j + 1, k] * fx
    c11 = f[i, j + 1, k + 1] * (1.0 - fx) + f[i + 1, j + 1, k + 1] * fx
    out[t] = (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz


@wp.kernel
def gather_morton(f: wp.array(dtype=wp.float32), q: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    t = wp.tid()
    x = q[t, 0]; y = q[t, 1]; z = q[t, 2]
    i = int(x); j = int(y); k = int(z); fx = x - float(i); fy = y - float(j); fz = z - float(k)
    c00 = f[morton(i, j, k)] * (1.0 - fx) + f[morton(i + 1, j, k)] * fx
    c01 = f[morton(i, j, k + 1)] * (1.0 - fx) + f[morton(i + 1, j, k + 1)] * fx
    c10 = f[morton(i, j + 1, k)] * (1.0 - fx) + f[morton(i + 1, j + 1, k)] * fx
    c11 = f[morton(i, j + 1, k + 1)] * (1.0 - fx) + f[morton(i + 1, j + 1, k + 1)] * fx
    out[t] = (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz


def bench(kern, dim, args, iters=200):
    wp.launch(kern, dim=dim, inputs=args, device=DEV); wp.synchronize()
    t0 = time.time()
    for _ in range(iters):
        wp.launch(kern, dim=dim, inputs=args, device=DEV)
    wp.synchronize()
    return dim * iters / (time.time() - t0) / 1e6        # Mqueries/s


def main():
    N = 128; M = 1 << 20
    print("=" * 80)
    print(f"MORTON for SPARSE GATHER — does space-filling order win where it LOST for dense?  ({DEV})")
    print("=" * 80)
    rng = np.random.default_rng(0)
    f = rng.standard_normal((N, N, N)).astype(np.float32)
    # morton-stored copy
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    def mcode(i, j, k):
        def s(x):
            x = x.astype(np.uint64) & 0x3ff
            x = (x | (x << 16)) & 0xff0000ff
            x = (x | (x << 8)) & 0x0300f00f
            x = (x | (x << 4)) & 0x030c30c3
            x = (x | (x << 2)) & 0x09249249
            return x
        return (s(i) | (s(j) << 1) | (s(k) << 2)).astype(np.int64)
    fm = np.zeros(N ** 3, np.float32)
    fm[mcode(ii.ravel(), jj.ravel(), kk.ravel())] = f.ravel()

    qpos = rng.uniform(1.0, N - 2.0, (M, 3)).astype(np.float32)          # random scattered query points
    qcode = mcode(qpos[:, 0].astype(np.int64), qpos[:, 1].astype(np.int64), qpos[:, 2].astype(np.int64))
    qsorted = qpos[np.argsort(qcode)]                                   # MORTON-sorted query order

    f_d = wp.array(f, dtype=wp.float32, device=DEV); fm_d = wp.array(fm, dtype=wp.float32, device=DEV)
    out = wp.zeros(M, dtype=wp.float32, device=DEV)
    qr = wp.array(qpos, dtype=wp.float32, device=DEV); qs = wp.array(qsorted, dtype=wp.float32, device=DEV)

    A = bench(gather_rowmajor, M, [f_d, qr, out])
    B = bench(gather_rowmajor, M, [f_d, qs, out])
    C = bench(gather_morton, M, [fm_d, qs, out])
    print(f"\n  (A) row-major f + RANDOM queries  : {A:>8.0f} Mqueries/s   (scattered baseline)")
    print(f"  (B) row-major f + MORTON-sorted q : {B:>8.0f} Mqueries/s   ({B/A:.2f}× vs A)")
    print(f"  (C) MORTON f    + MORTON-sorted q : {C:>8.0f} Mqueries/s   ({C/A:.2f}× vs A)")

    win = C / A
    print("\n" + "=" * 80)
    if win > 1.2:
        print(f"VERDICT: Morton WINS for sparse gather — {win:.2f}× over the scattered baseline (sorting {B/A:.2f}× +")
        print(f"  Morton-store the rest). OPPOSITE of the dense stencil (Morton 0.5× there). ⇒ the memory-WORKLOAD lever")
        print(f"  for SPARSE access (NanoVDB collision / sparse-AMR) is space-filling ORDER — locality follows the access")
        print(f"  geometry. Apply to the S1 collision bottleneck: Morton-sort the (config,point) work-items + Morton SDF.")
    elif win > 1.05:
        print(f"VERDICT: modest Morton gain ({win:.2f}×) for sparse gather — real but small; the scattered baseline")
        print(f"  already gets some L2 reuse. Honest: a lever, not a transformation. Worth it on the S1 bottleneck.")
    else:
        print(f"VERDICT: Morton does NOT help sparse gather here ({win:.2f}×) — the GPU's L2 already absorbs the")
        print(f"  scattered access at this size; the principle did not transfer. Report honestly (scene_eyes).")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

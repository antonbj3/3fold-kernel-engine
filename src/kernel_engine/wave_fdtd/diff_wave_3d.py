#!/usr/bin/env python3
"""DIFFERENTIABLE 3D wave substrate — closes the 3D loop (the validated 3D Yee substrate + warp-adjoint).
If dJ/dc2[x,y,z] is recoverable by one backward pass and matches finite-difference, the 3D substrate is a
gradient-based inverse-design / hyperreal-twin-calibration engine in 3D (real devices/twins are 3D), the same
adjoint capability proven in 2D. Out-of-place leapfrog so reverse-mode is exact.

DECISIVE: dJ/dc2 (wp.Tape backward) matches central finite-difference to <5% at probed voxels.

  python3 diff_wave_3d.py
"""
import sys
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
N = 30; T = 46


@wp.kernel
def upd_v(p: wp.array3d(dtype=wp.float32), vx: wp.array3d(dtype=wp.float32), vy: wp.array3d(dtype=wp.float32),
         vz: wp.array3d(dtype=wp.float32), vxn: wp.array3d(dtype=wp.float32), vyn: wp.array3d(dtype=wp.float32),
         vzn: wp.array3d(dtype=wp.float32), cs: float):
    i, j, k = wp.tid()
    nx = p.shape[0]
    if i < nx - 1:
        vxn[i, j, k] = vx[i, j, k] - cs * (p[i + 1, j, k] - p[i, j, k])
    if j < nx - 1:
        vyn[i, j, k] = vy[i, j, k] - cs * (p[i, j + 1, k] - p[i, j, k])
    if k < nx - 1:
        vzn[i, j, k] = vz[i, j, k] - cs * (p[i, j, k + 1] - p[i, j, k])


@wp.kernel
def upd_p(p: wp.array3d(dtype=wp.float32), vxn: wp.array3d(dtype=wp.float32), vyn: wp.array3d(dtype=wp.float32),
         vzn: wp.array3d(dtype=wp.float32), pn: wp.array3d(dtype=wp.float32), csq: wp.array3d(dtype=wp.float32), cs: float):
    i, j, k = wp.tid()
    nx = p.shape[0]
    if i >= 1 and j >= 1 and k >= 1:
        div = (cs * (vxn[i, j, k] - vxn[i - 1, j, k]) + cs * (vyn[i, j, k] - vyn[i, j - 1, k])
               + cs * (vzn[i, j, k] - vzn[i, j, k - 1]))
        pn[i, j, k] = p[i, j, k] - csq[i, j, k] * div
    else:
        pn[i, j, k] = p[i, j, k]


@wp.kernel
def energy(p: wp.array3d(dtype=wp.float32), lo: int, hi: int, out: wp.array(dtype=wp.float32)):
    i, j, k = wp.tid()
    if i >= lo and i < hi and j >= lo and j < hi and k >= lo and k < hi:
        wp.atomic_add(out, 0, p[i, j, k] * p[i, j, k])


# Order-invariant accumulation: float atomics are non-associative, so the reported J depends on the
# (run-varying) order in which blocks reach the accumulator. Each contribution is rounded in float64 to an
# int64 fixed point and summed with integer atomics -> associative -> bit-identical run to run.
DETERMINISTIC_ACCUMULATION = True
ACC_SCALE = 2.0 ** 45                                 # |p| <= 1 (unit seed, non-amplifying leapfrog) -> |sum p^2| <= N^3
assert float(N ** 3) * ACC_SCALE < 2.0 ** 62, "fixed-point overflow"


@wp.kernel
def energy_i64(p: wp.array3d(dtype=wp.float32), lo: int, hi: int, scale: wp.float64,
               out: wp.array(dtype=wp.int64)):
    i, j, k = wp.tid()
    if i >= lo and i < hi and j >= lo and j < hi and k >= lo and k < hi:
        v = wp.float64(p[i, j, k]) * wp.float64(p[i, j, k])
        wp.atomic_add(out, 0, wp.int64(wp.round(v * scale)))


def energy_value(loss, p):
    """Value of the window energy J. With DETERMINISTIC_ACCUMULATION the sum is recomputed with int64
    fixed-point atomics (order-invariant); the float array stays for the adjoint, since a quantiser has
    no useful derivative."""
    if not DETERMINISTIC_ACCUMULATION:
        return float(loss.numpy()[0])
    acc = wp.zeros(1, dtype=wp.int64, device=DEV)
    wp.launch(energy_i64, dim=(N, N, N), inputs=[p, 2 * N // 5, 3 * N // 5, wp.float64(ACC_SCALE), acc], device=DEV)
    wp.synchronize()
    return float(int(acc.numpy()[0])) / ACC_SCALE


def forward(csq):
    cs = 0.4
    seed = np.zeros((N, N, N), np.float32)
    a = np.arange(N)
    seed += np.exp(-(((a[:, None, None] - 5) ** 2 + (a[None, :, None] - N // 2) ** 2
                      + (a[None, None, :] - N // 2) ** 2) / 7.0)).astype(np.float32)
    p = wp.array(seed, dtype=wp.float32, device=DEV, requires_grad=True)
    vx = wp.zeros((N, N, N), dtype=wp.float32, device=DEV, requires_grad=True)
    vy = wp.zeros((N, N, N), dtype=wp.float32, device=DEV, requires_grad=True)
    vz = wp.zeros((N, N, N), dtype=wp.float32, device=DEV, requires_grad=True)
    for _ in range(T):
        vxn = wp.zeros((N, N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        vyn = wp.zeros((N, N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        vzn = wp.zeros((N, N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        pn = wp.zeros((N, N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        wp.launch(upd_v, dim=(N, N, N), inputs=[p, vx, vy, vz, vxn, vyn, vzn, cs], device=DEV)
        wp.launch(upd_p, dim=(N, N, N), inputs=[p, vxn, vyn, vzn, pn, csq, cs], device=DEV)
        p, vx, vy, vz = pn, vxn, vyn, vzn
    loss = wp.zeros(1, dtype=wp.float32, device=DEV, requires_grad=True)
    wp.launch(energy, dim=(N, N, N), inputs=[p, 2 * N // 5, 3 * N // 5, loss], device=DEV)
    return loss, p


def main():
    print("=" * 72)
    print(f"DIFFERENTIABLE 3D wave substrate — dJ/dc2 via warp-adjoint, FD-verified  (device={DEV})")
    print("=" * 72)
    csq0 = np.full((N, N, N), 1.0, np.float32)
    csq = wp.array(csq0, dtype=wp.float32, device=DEV, requires_grad=True)
    tape = wp.Tape()
    with tape:
        loss, p0 = forward(csq)
    tape.backward(loss=loss)
    g = csq.grad.numpy(); J0 = energy_value(loss, p0)
    print(f"\n  J0={J0:.4e}; adjoint dJ/dc2 computed (one 3D backward pass).")
    eps = 1e-2; maxrel = 0.0
    cells = [(N // 2, N // 2, N // 2), (N // 3, N // 2, N // 2), (N // 2, 2 * N // 3, N // 2)]
    print(f"\n  {'voxel':>14} {'adjoint':>13} {'FD':>13} {'rel_err':>10}")
    for (ci, cj, ck) in cells:
        cp = csq0.copy(); cp[ci, cj, ck] += eps
        lp = energy_value(*forward(wp.array(cp, dtype=wp.float32, device=DEV)))
        cm = csq0.copy(); cm[ci, cj, ck] -= eps
        lm = energy_value(*forward(wp.array(cm, dtype=wp.float32, device=DEV)))
        fd = (lp - lm) / (2 * eps); ad = float(g[ci, cj, ck]); rel = abs(ad - fd) / (abs(fd) + 1e-12)
        maxrel = max(maxrel, rel)
        print(f"  {str((ci, cj, ck)):>14} {ad:>13.4e} {fd:>13.4e} {rel:>10.2e}")
    ok = maxrel < 0.05
    print("\n" + "=" * 72)
    print(f"VERDICT: differentiable 3D substrate = {'VALIDATED' if ok else 'PARTIAL'} (max FD rel_err {maxrel:.2e})")
    print(f"  One 3D warp-adjoint pass gives dJ/dc2 over the WHOLE volume → 3D inverse-design / hyperreal-twin")
    print(f"  calibration in 3D (real devices/twins), same adjoint capability as 2D. EM/elasto by swap.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

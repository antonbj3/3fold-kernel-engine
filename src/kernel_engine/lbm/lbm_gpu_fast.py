#!/usr/bin/env python3
"""Fused single-kernel D2Q9 lattice-Boltzmann step on the GPU (Warp), tuned for memory throughput.

Three properties: SoA layout (9, nx, ny) so threads read contiguous memory; collide and stream fused
into one kernel with ping-pong buffers and a pull scheme, which removes the intermediate round trip
(36 -> 18 float operations per voxel per step); half-way bounce-back integrated into the same kernel.
Traffic floor 18 * 4 = 72 B per voxel.

Validation: steady Poiseuille against the analytic profile, then a throughput sweep over 512^2 to
4096^2. A fast but wrong kernel is worthless, so the validation runs first.

Requires Warp; falls back to CPU if no CUDA device is present.

  python lbm_gpu_fast.py
"""
import sys
import time
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
vec9 = wp.types.vector(length=9, dtype=wp.float32)

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float32)
WT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
OP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


@wp.kernel
def collide_stream(fA: wp.array3d(dtype=wp.float32), fB: wp.array3d(dtype=wp.float32),
                   solid: wp.array2d(dtype=wp.int32),
                   cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
                   w: wp.array(dtype=wp.float32), opp: wp.array(dtype=wp.int32),
                   omega: float, gforce: float, u_in: float, periodic_x: int, nx: int, ny: int):
    """Fused collide+stream (pull, SoA). fA = post-collision state of the previous step, fB = of this step."""
    i, j = wp.tid()
    if solid[i, j] == 1:
        for k in range(9):
            fB[k, i, j] = fA[k, i, j]                     # solid passiv
        return
    g = vec9()
    for k in range(9):                                    # stream-PULL + half-way bounce-back
        si = i - int(cx[k]); sj = j - int(cy[k])
        if periodic_x == 1:
            if si < 0: si += nx
            if si >= nx: si -= nx
        else:
            if si < 0: si = 0
            if si >= nx: si = nx - 1
        if sj < 0: sj = 0
        if sj >= ny: sj = ny - 1
        if solid[si, sj] == 1:
            g[k] = fA[opp[k], i, j]                       # bounce-back (wall at half distance)
        else:
            g[k] = fA[k, si, sj]
    rho = float(0.0); mx = float(0.0); my = float(0.0)
    for k in range(9):
        rho += g[k]; mx += cx[k] * g[k]; my += cy[k] * g[k]
    ux = mx / rho + 0.5 * gforce
    uy = my / rho
    if u_in > 0.0 and i == 0:                             # inflow boundary
        ux = u_in; uy = 0.0; rho = 1.0
    usq = ux * ux + uy * uy
    for k in range(9):                                    # LOKAL BGK-kollision → fB
        cu = cx[k] * ux + cy[k] * uy
        feq = w[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        val = g[k] - omega * (g[k] - feq)
        if gforce != 0.0:
            val += (1.0 - 0.5 * omega) * 3.0 * w[k] * cx[k] * gforce
        fB[k, i, j] = val


@wp.kernel
def macro_ux_soa(f: wp.array3d(dtype=wp.float32), cx: wp.array(dtype=wp.float32),
                 solid: wp.array2d(dtype=wp.int32), ux: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    if solid[i, j] == 1:
        ux[i, j] = 0.0; return
    rho = float(0.0); mx = float(0.0)
    for k in range(9):
        fk = f[k, i, j]; rho += fk; mx += cx[k] * fk
    ux[i, j] = mx / rho


def _equil_soa(rho, ux, uy):
    nx, ny = rho.shape; f = np.empty((9, nx, ny), np.float32)
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = CX[k] * ux + CY[k] * uy
        f[k] = WT[k] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f


def run(nx, ny, tau, solid_np, max_steps, gforce=0.0, u_in=0.0, periodic_x=1, tol=0.0, check=2000, ret_field=False):
    """Fused-SoA LBM. Returnerar (ux[numpy] om ret_field, steps, res, mlups)."""
    omega = 1.0 / tau
    rho0 = np.ones((nx, ny), np.float32); ux0 = np.full((nx, ny), u_in, np.float32); uy0 = np.zeros((nx, ny), np.float32)
    fA = wp.array(_equil_soa(rho0, ux0, uy0), dtype=wp.float32, device=DEV)
    fB = wp.zeros((9, nx, ny), dtype=wp.float32, device=DEV)
    solid = wp.array(solid_np.astype(np.int32), dtype=wp.int32, device=DEV)
    cx = wp.array(CX, dtype=wp.float32, device=DEV); cy = wp.array(CY, dtype=wp.float32, device=DEV)
    w = wp.array(WT, dtype=wp.float32, device=DEV); opp = wp.array(OP, dtype=wp.int32, device=DEV)
    uxd = wp.zeros((nx, ny), dtype=wp.float32, device=DEV)
    ux_prev = np.zeros((nx, ny), np.float32); res = 1.0
    wp.synchronize(); t0 = time.time(); it = 0
    while it < max_steps:
        wp.launch(collide_stream, dim=(nx, ny), inputs=[fA, fB, solid, cx, cy, w, opp, omega, gforce, u_in, periodic_x, nx, ny], device=DEV)
        fA, fB = fB, fA
        it += 1
        if tol > 0.0 and it % check == 0:
            wp.launch(macro_ux_soa, dim=(nx, ny), inputs=[fA, cx, solid, uxd], device=DEV)
            wp.synchronize()
            uxn = uxd.numpy()
            res = float(np.max(np.abs(uxn - ux_prev)) / (np.max(np.abs(uxn)) + 1e-30))
            ux_prev = uxn.copy()
            if res < tol: break
    wp.synchronize(); dt = time.time() - t0
    mlups = nx * ny * it / dt / 1e6
    field = None
    if ret_field:
        wp.launch(macro_ux_soa, dim=(nx, ny), inputs=[fA, cx, solid, uxd], device=DEV)
        wp.synchronize(); field = uxd.numpy()
    return field, it, res, mlups


def main():
    print("=" * 80); print(f"LBM GPU FAST -- fused-SoA Warp/CUDA, device={DEV}"); print("=" * 80)

    nx, ny = 10, 42; tau = 0.8; nu = (tau - 0.5) / 3.0; G = 2e-5
    solid = np.zeros((nx, ny), bool); solid[:, 0] = True; solid[:, -1] = True
    ux, st, rs, _ = run(nx, ny, tau, solid, max_steps=200000, gforce=G, periodic_x=1, tol=1e-6, check=2000, ret_field=True)
    prof = ux[nx // 2, :]; H = ny - 2
    y = np.arange(ny) - 0.5
    u_ana = np.where((y > 0) & (y < H), G / (2 * nu) * y * (H - y), 0.0)
    inner = (np.arange(ny) >= 1) & (np.arange(ny) <= ny - 2)
    l2 = float(np.sqrt(np.mean((prof[inner] - u_ana[inner]) ** 2)) / (u_ana.max() + 1e-30)) * 100
    umax_lbm, umax_ana = float(prof.max()), float(u_ana.max())
    V_ok = l2 < 3.0 and abs(umax_lbm - umax_ana) / umax_ana < 0.03
    print(f"\nVALIDATION Poiseuille (fused-SoA): u_max {umax_lbm:.3e} vs analytic {umax_ana:.3e}; L2 {l2:.1f}% "
          f"{'fused-SoA kernel CORRECT (matches the analytic solution)' if V_ok else 'KERNEL BROKEN'}  [steady@{st}, res={rs:.1e}]")

    # ── THROUGHPUT-SVEP: fused-SoA vs baseline 1770 MLUPS ──
    print(f"\nTHROUGHPUT (fused-SoA, 400 steps fixed):")
    best = 0.0
    for n in [512, 1024, 2048, 4096]:
        bsolid = np.zeros((n, n), bool); bsolid[:, 0] = True; bsolid[:, -1] = True
        _, _, _, ml = run(n, n, 0.6, bsolid, max_steps=400, gforce=1e-6, periodic_x=1, tol=0.0)
        best = max(best, ml)
        print(f"   {n:>5}×{n:<5}: {ml:7.0f} MLUPS", flush=True)
    ceiling = 9333
    speedup_vs_base = best / 1770
    T_ok = best > 1770 * 1.5
    print(f"\n   best {best:.0f} MLUPS = {best/ceiling*100:.0f}% of the FP32 ceiling ({ceiling}); {speedup_vs_base:.1f}x baseline (1770); "
          f"{'throughput gain' if T_ok else 'no gain'}")
    print(f"   reference consumer-GPU implementations reach ~3-7k MLUPS {'(matched or passed)' if best > 3000 else '(remaining: FP16 storage + in-place EsoTwist, ceiling x2-4)'}")

    all_ok = V_ok and T_ok
    print("\n" + "=" * 80)
    print(f"VERDICT: fused-SoA kernel = {'PASS' if all_ok else 'PARTIAL'}  (correct {'yes' if V_ok else 'no'}; throughput {best:.0f} MLUPS {'pass' if T_ok else 'fail'})")
    print(f"  SoA coalescing + fused collide-stream (ping-pong, no fpost round trip) + half-way bounce-back -> {speedup_vs_base:.1f}x baseline,")
    print(f"  validated against the analytic Poiseuille profile. Next steps: FP16 storage (ceiling x2) + in-place EsoTwist (ceiling x2)")
    print(f"  -> ceiling ~18-37k on datacentre GPUs and multi-GPU. This is the fluid substrate of the engine.")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

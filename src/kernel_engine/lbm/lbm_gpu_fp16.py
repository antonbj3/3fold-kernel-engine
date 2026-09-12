#!/usr/bin/env python3
"""LBM GPU FP16 (★OUTCLASS steg 2) — FP16-lagring + FP32-compute, FluidX3D-tier bandbredd.

LBM is MEMORY-bound: halving the stored bytes (FP32 to FP16) roughly doubles effective bandwidth and MLUPS.
RISK: FP16 (10-bit mantissa, ~3 digits) loses precision in the SMALL non-equilibrium part that CARRIES the physics.
The storage trick: store the DEVIATION s_k = f_k - w_k (not f_k) in FP16. The deviation is small near equilibrium, so
FP16 precision is preserved where it matters. Compute stays FP32 (load, +w_k, FP32 collision, -w_k, FP16 store).

VALIDATION: Poiseuille against the ANALYTIC profile must hold (<1% L2); otherwise FP16 is too coarse and FP32 stays the default.
Throughput is compared against the FP32 fused baseline (lbm_gpu_fast.py, 7345 peak).

  python3 lbm_gpu_fp16.py
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
def collide_stream_fp16(sA: wp.array3d(dtype=wp.float16), sB: wp.array3d(dtype=wp.float16),
                        solid: wp.array2d(dtype=wp.int32),
                        cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
                        w: wp.array(dtype=wp.float32), opp: wp.array(dtype=wp.int32),
                        omega: float, gforce: float, u_in: float, periodic_x: int, nx: int, ny: int):
    """Fused collide+stream, FP16-lagring av AVVIKELSE s=f−w, FP32-compute."""
    i, j = wp.tid()
    if solid[i, j] == 1:
        for k in range(9):
            sB[k, i, j] = sA[k, i, j]
        return
    g = vec9()
    for k in range(9):
        si = i - int(cx[k]); sj = j - int(cy[k])
        if periodic_x == 1:
            if si < 0: si += nx
            if si >= nx: si -= nx
        else:
            if si < 0: si = 0
            if si >= nx: si = nx - 1
        if sj < 0: sj = 0
        if sj >= ny: sj = ny - 1
        if solid[si, sj] == 1:                            # bounce-back: w[opp[k]]==w[k] i D2Q9
            g[k] = wp.float32(sA[opp[k], i, j]) + w[k]
        else:
            g[k] = wp.float32(sA[k, si, sj]) + w[k]       # f = s + w (restore deviation to full f)
    rho = float(0.0); mx = float(0.0); my = float(0.0)
    for k in range(9):
        rho += g[k]; mx += cx[k] * g[k]; my += cy[k] * g[k]
    ux = mx / rho + 0.5 * gforce
    uy = my / rho
    if u_in > 0.0 and i == 0:
        ux = u_in; uy = 0.0; rho = 1.0
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = cx[k] * ux + cy[k] * uy
        feq = w[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        val = g[k] - omega * (g[k] - feq)
        if gforce != 0.0:
            val += (1.0 - 0.5 * omega) * 3.0 * w[k] * cx[k] * gforce
        sB[k, i, j] = wp.float16(val - w[k])              # lagra avvikelse s = f − w i FP16


@wp.kernel
def macro_ux_fp16(s: wp.array3d(dtype=wp.float16), cx: wp.array(dtype=wp.float32), w: wp.array(dtype=wp.float32),
                  solid: wp.array2d(dtype=wp.int32), ux: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    if solid[i, j] == 1:
        ux[i, j] = 0.0; return
    rho = float(0.0); mx = float(0.0)
    for k in range(9):
        fk = wp.float32(s[k, i, j]) + w[k]
        rho += fk; mx += cx[k] * fk
    ux[i, j] = mx / rho


def _equil_dev_soa(rho, ux, uy):                          # initial avvikelse s = f_eq − w
    nx, ny = rho.shape; s = np.empty((9, nx, ny), np.float16)
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = CX[k] * ux + CY[k] * uy
        feq = WT[k] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
        s[k] = (feq - WT[k]).astype(np.float16)
    return s


def run(nx, ny, tau, solid_np, max_steps, gforce=0.0, u_in=0.0, periodic_x=1, tol=0.0, check=2000, ret_field=False):
    omega = 1.0 / tau
    rho0 = np.ones((nx, ny), np.float32); ux0 = np.full((nx, ny), u_in, np.float32); uy0 = np.zeros((nx, ny), np.float32)
    sA = wp.array(_equil_dev_soa(rho0, ux0, uy0), dtype=wp.float16, device=DEV)
    sB = wp.zeros((9, nx, ny), dtype=wp.float16, device=DEV)
    solid = wp.array(solid_np.astype(np.int32), dtype=wp.int32, device=DEV)
    cx = wp.array(CX, dtype=wp.float32, device=DEV); cy = wp.array(CY, dtype=wp.float32, device=DEV)
    w = wp.array(WT, dtype=wp.float32, device=DEV); opp = wp.array(OP, dtype=wp.int32, device=DEV)
    uxd = wp.zeros((nx, ny), dtype=wp.float32, device=DEV)
    ux_prev = np.zeros((nx, ny), np.float32); res = 1.0
    wp.synchronize(); t0 = time.time(); it = 0
    while it < max_steps:
        wp.launch(collide_stream_fp16, dim=(nx, ny), inputs=[sA, sB, solid, cx, cy, w, opp, omega, gforce, u_in, periodic_x, nx, ny], device=DEV)
        sA, sB = sB, sA
        it += 1
        if tol > 0.0 and it % check == 0:
            wp.launch(macro_ux_fp16, dim=(nx, ny), inputs=[sA, cx, w, solid, uxd], device=DEV)
            wp.synchronize()
            uxn = uxd.numpy()
            res = float(np.max(np.abs(uxn - ux_prev)) / (np.max(np.abs(uxn)) + 1e-30))
            ux_prev = uxn.copy()
            if res < tol: break
    wp.synchronize(); dt = time.time() - t0
    mlups = nx * ny * it / dt / 1e6
    field = None
    if ret_field:
        wp.launch(macro_ux_fp16, dim=(nx, ny), inputs=[sA, cx, w, solid, uxd], device=DEV)
        wp.synchronize(); field = uxd.numpy()
    return field, it, res, mlups


def main():
    print("=" * 80); print(f"LBM GPU FP16 (★OUTCLASS steg 2) — FP16-lagring(avvikelse) + FP32-compute, device={DEV}"); print("=" * 80)

    # -- VALIDATION: Poiseuille vs ANALYTIC (FP16 must NOT degrade past ~1%) --
    nx, ny = 10, 42; tau = 0.8; nu = (tau - 0.5) / 3.0; G = 2e-5
    solid = np.zeros((nx, ny), bool); solid[:, 0] = True; solid[:, -1] = True
    ux, st, rs, _ = run(nx, ny, tau, solid, max_steps=200000, gforce=G, periodic_x=1, tol=1e-6, check=2000, ret_field=True)
    prof = ux[nx // 2, :]; H = ny - 2
    y = np.arange(ny) - 0.5
    u_ana = np.where((y > 0) & (y < H), G / (2 * nu) * y * (H - y), 0.0)
    inner = (np.arange(ny) >= 1) & (np.arange(ny) <= ny - 2)
    l2 = float(np.sqrt(np.mean((prof[inner] - u_ana[inner]) ** 2)) / (u_ana.max() + 1e-30)) * 100
    umax_lbm, umax_ana = float(prof.max()), float(u_ana.max())
    V_ok = l2 < 1.0 and abs(umax_lbm - umax_ana) / umax_ana < 0.02
    print(f"\nVALIDERING Poiseuille (FP16-avvikelse): u_max {umax_lbm:.3e} vs analytisk {umax_ana:.3e}; L2 {l2:.2f}% "
          f"{'FP16 holds precision (deviation trick works)' if V_ok else 'FP16 too coarse - keep FP32'}  [steady@{st}, res={rs:.1e}]")

    # ── THROUGHPUT vs FP32-fused-baseline (7345 peak) ──
    print(f"\nTHROUGHPUT (FP16, 400 steg):")
    best = 0.0
    for n in [512, 1024, 2048, 4096]:
        bsolid = np.zeros((n, n), bool); bsolid[:, 0] = True; bsolid[:, -1] = True
        _, _, _, ml = run(n, n, 0.6, bsolid, max_steps=400, gforce=1e-6, periodic_x=1, tol=0.0)
        best = max(best, ml)
        print(f"   {n:>5}×{n:<5}: {ml:7.0f} MLUPS", flush=True)
    fp32_peak = 7345.0; ceiling_fp16 = 18667
    gain = best / fp32_peak
    T_ok = best > fp32_peak
    print(f"\n   best {best:.0f} MLUPS = {best/ceiling_fp16*100:.0f}% of the FP16 ceiling ({ceiling_fp16}); {gain:.2f}x the FP32 fused peak (7345); "
          f"{'FP16 gives a throughput gain' if T_ok else 'no gain'}")

    all_ok = V_ok and T_ok
    print("\n" + "=" * 80)
    print(f"VERDICT: FP16 storage wins = {'yes' if all_ok else 'PARTIAL'}  (precision holds {'yes' if V_ok else 'no'} - throughput {best:.0f} MLUPS {'yes' if T_ok else 'no'})")
    print(f"  FP16 deviation storage (s=f-w) + FP32 compute -> {gain:.2f}x the FP32 peak, precision {'preserved' if V_ok else 'NOT preserved'} (Poiseuille L2 {l2:.2f}%).")
    print(f"  Progress: baseline 1770 -> FP32 fused {fp32_peak:.0f} -> FP16 {best:.0f} MLUPS. NEXT: in-place EsoTwist (one array,")
    print(f"  another x2 of ceiling), 3D D3Q19, multi-GPU. Honest trade-off: FP16 = faster but {'precision OK here' if V_ok else 'precision risk'} -> FP32 = default for the high-fidelity twin, FP16 = mass-batch mode.")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

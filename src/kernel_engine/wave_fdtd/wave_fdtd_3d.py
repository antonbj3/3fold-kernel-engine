#!/usr/bin/env python3
"""WAVE-FDTD 3D — the validated 2D Yee substrate extended to 3D (real twins/devices are 3D). Staggered
velocity-pressure leapfrog on a 3D voxel grid: p at cell centers, (vx,vy,vz) on the three face sets.
Same operator-agnostic kache substrate (acoustic now; EM/elasto by coefficient/stencil swap, as in 2D),
matrix-free, atomics-free → bit-reproducible.

VALIDATE: PEC/pressure-release cavity eigenmodes f_lmn=(c/2)·sqrt((l/Lx)²+(m/Ly)²+(n/Lz)²) vs FFT-measured;
CFL limit (c·dt·sqrt(1/dx²+1/dy²+1/dz²)≤1) real; bit-reproducibility; throughput (MLUPS) at scale.

  python3 wave_fdtd_3d.py
"""
import sys
import time
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"


@wp.kernel
def upd_v3(p: wp.array3d(dtype=wp.float32), vx: wp.array3d(dtype=wp.float32),
          vy: wp.array3d(dtype=wp.float32), vz: wp.array3d(dtype=wp.float32),
          csx: float, csy: float, csz: float, Nx: int, Ny: int, Nz: int):
    i, j, k = wp.tid()
    if i < Nx - 1:
        vx[i, j, k] = vx[i, j, k] - csx * (p[i + 1, j, k] - p[i, j, k])
    if j < Ny - 1:
        vy[i, j, k] = vy[i, j, k] - csy * (p[i, j + 1, k] - p[i, j, k])
    if k < Nz - 1:
        vz[i, j, k] = vz[i, j, k] - csz * (p[i, j, k + 1] - p[i, j, k])


@wp.kernel
def upd_p3(p: wp.array3d(dtype=wp.float32), vx: wp.array3d(dtype=wp.float32),
          vy: wp.array3d(dtype=wp.float32), vz: wp.array3d(dtype=wp.float32),
          ca: float, csx: float, csy: float, csz: float, Nx: int, Ny: int, Nz: int):
    i, j, k = wp.tid()
    if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1 or k == 0 or k == Nz - 1:
        p[i, j, k] = 0.0       # PEC / pressure-release perimeter
        return
    div = (csx * (vx[i, j, k] - vx[i - 1, j, k]) + csy * (vy[i, j, k] - vy[i, j - 1, k])
           + csz * (vz[i, j, k] - vz[i, j, k - 1]))
    p[i, j, k] = p[i, j, k] - ca * div


def fields(N):
    return [wp.zeros((N, N, N), dtype=wp.float32, device=DEV) for _ in range(4)]


def step(F, ca, cs, N):
    p, vx, vy, vz = F
    wp.launch(upd_v3, dim=(N, N, N), inputs=[p, vx, vy, vz, cs, cs, cs, N, N, N], device=DEV)
    wp.launch(upd_p3, dim=(N, N, N), inputs=[p, vx, vy, vz, ca, cs, cs, cs, N, N, N], device=DEV)


def main():
    print("=" * 76)
    print(f"WAVE-FDTD 3D — Yee acoustic cavity on a 3D voxel substrate, device={DEV}")
    print("=" * 76)
    c = 1.0; L = 1.0; N = 49
    dx = L / (N - 1); dt = 0.99 / (c * np.sqrt(3.0) / dx); cs = dt / dx; ca = c * c
    f111 = (c / 2.0) * np.sqrt(3.0) / L
    xi = np.arange(N)
    sh = (np.sin(np.pi * xi[:, None, None] / (N - 1)) * np.sin(np.pi * xi[None, :, None] / (N - 1))
          * np.sin(np.pi * xi[None, None, :] / (N - 1))).astype(np.float32)
    F = fields(N); F[0].assign(wp.array(sh, dtype=wp.float32, device=DEV))
    T = 1.0 / f111; n_steps = int(40 * T / dt)
    pi, pj, pk = N // 3, N // 3, 2 * N // 5
    rec = np.empty(n_steps)
    wp.synchronize()
    for it in range(n_steps):
        step(F, ca, cs, N)
        rec[it] = F[0].numpy()[pi, pj, pk]
    wp.synchronize()
    rr = rec - rec.mean(); spec = np.abs(np.fft.rfft(rr * np.hanning(len(rr))))
    fr = np.fft.rfftfreq(len(rr), d=dt); kpk = np.argmax(spec[1:]) + 1
    d = 0.5 * (spec[kpk - 1] - spec[kpk + 1]) / (spec[kpk - 1] - 2 * spec[kpk] + spec[kpk + 1] + 1e-30)
    f_meas = (kpk + np.clip(d, -0.5, 0.5)) / (len(rr) * dt)
    e111 = abs(f_meas - f111) / f111
    print(f"\n  cavity (1,1,1): analytic f={f111:.5f}  measured={f_meas:.5f}  rel_err={e111:.3e}  "
          f"{'✓' if e111 < 0.02 else '✗'}")

    # CFL self-check
    def blow(Sc):
        dts = Sc * dx / c; csb = dts / dx
        Fb = fields(33); seed = np.zeros((33, 33, 33), np.float32); seed[16, 16, 16] = 1.0
        Fb[0].assign(wp.array(seed, dtype=wp.float32, device=DEV))
        for _ in range(200):
            step(Fb, ca, csb, 33)
        wp.synchronize(); return float(np.max(np.abs(Fb[0].numpy())))
    stable = blow(0.55); unstable = blow(0.62)            # 3D limit 1/sqrt(3)=0.577
    cfl_ok = np.isfinite(stable) and (not np.isfinite(unstable) or unstable > 1e6 * stable)
    print(f"  CFL (3D limit 1/√3=0.577): Sc=0.55 max|p|={stable:.2e} (bounded), Sc=0.62 max|p|={unstable:.2e}  "
          f"{'✓' if cfl_ok else '✗'}")

    # reproducibility
    def once():
        Fr = fields(41); sd = np.zeros((41, 41, 41), np.float32); sd[20, 20, 20] = 1.0
        Fr[0].assign(wp.array(sd, dtype=wp.float32, device=DEV))
        for _ in range(120):
            step(Fr, ca, 0.99 / np.sqrt(3.0), 41)
        wp.synchronize(); return Fr[0].numpy().copy()
    repro = bool(np.array_equal(once(), once()))
    print(f"  reproducibility (no atomics → bit-identical): {repro}  {'✓' if repro else '✗'}")

    # throughput
    best = 0.0
    for Nt in [96, 160]:
        Ft = fields(Nt); sd = np.zeros((Nt, Nt, Nt), np.float32); sd[Nt // 2, Nt // 2, Nt // 2] = 1.0
        Ft[0].assign(wp.array(sd, dtype=wp.float32, device=DEV))
        wp.synchronize(); t0 = time.time()
        for _ in range(60):
            step(Ft, ca, 0.99 / np.sqrt(3.0), Nt)
        wp.synchronize(); ml = Nt ** 3 * 60 / (time.time() - t0) / 1e6
        best = max(best, ml); print(f"  throughput {Nt}³: {ml:7.0f} MLUPS", flush=True)

    ok = e111 < 0.02 and cfl_ok and repro
    print("\n" + "=" * 76)
    print(f"VERDICT: 3D Yee acoustic substrate = {'VALIDATED' if ok else 'PARTIAL'}")
    print(f"  cavity f111 {e111:.1e} · CFL {cfl_ok} · repro {repro} · {best:.0f} MLUPS. The operator-agnostic")
    print(f"  kache substrate scales to 3D (real twins/devices); EM/elasto by coeff/stencil swap as in 2D.")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

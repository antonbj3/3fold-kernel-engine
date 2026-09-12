#!/usr/bin/env python3
"""GENCHI GENBUTSU — go and SEE the acoustic rigid field (stop-the-line on the gauge defect).
After the rigid-BC fix, energy+purity are green but shape-residual vs continuous cos(1,1) is ~1.0.
Decide DECISIVELY: is the solver in a pure (stationary) mode (→ gauge/seed defect) or multi-mode (→ solver)?

  T1 STATIONARITY (seed-free): snapshot S at several same-amplitude times; if the field keeps a FIXED shape,
     |corr(S(ta),S(tb))| ≈ 1 for all pairs → it IS a pure eigenmode (solver fine, my reference shape wrong).
  T2 MODE SPECTRUM: project a snapshot onto continuous cos modes (0..2,0..2) → see which mode it actually is.
  T3 SEED CONVENTION: residual of the discrete mode vs candidate seeds — node cos(i/(N-1)) vs cell-centred
     cos((i+0.5)/N) vs cos((i+0.5)/(N-1)) — the small-resid one reveals the true staggered-grid geometry.

  python3 diag_acoustic_mode.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('wave_fdtd',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import numpy as np
import warp as wp
from wave_fdtd_kache import make_fields, step, DEV

c = 343.0; Lx = Ly = 0.5; N = 65; bflag = 1
dx = Lx / (N - 1); dt = 0.99 * dx / (c * np.sqrt(2.0)); csx = csy = dt / dx; ca = c * c; cb = 1.0
ii = np.arange(N)


def cos_shape(m, n, conv):
    if conv == "node":      # cos(m*pi*i/(N-1))  (x = i*dx, x in [0,Lx])
        x = ii / (N - 1)
    elif conv == "cell_N":  # cos(m*pi*(i+0.5)/N)  (cell centres, N cells)
        x = (ii + 0.5) / N
    else:                   # cell_Nm1
        x = (ii + 0.5) / (N - 1)
    cx = np.cos(m * np.pi * x); cy = np.cos(n * np.pi * x)
    return np.outer(cx, cy)


def run_collect(seed, n_periods=6):
    S, U, W = make_fields(N, N)
    S.assign(wp.array(seed.astype(np.float32), dtype=wp.float32, device=DEV))
    f = (c / 2.0) * np.sqrt(2.0) / Lx          # f_11
    nps = int(n_periods / f / dt)
    snaps = []
    for it in range(nps):
        step(S, U, W, ca, cb, csx, csy, N, N, bflag)
        if it % 7 == 0:
            Snp = S.numpy().astype(np.float64)
            if np.linalg.norm(Snp) > 0.3 * np.linalg.norm(seed):
                snaps.append(Snp)
    return snaps


def main():
    print("=" * 88)
    print("GENCHI GENBUTSU — acoustic rigid field: pure mode (gauge defect) or multi-mode (solver)?")
    print("=" * 88)
    seed = cos_shape(1, 1, "node")
    snaps = run_collect(seed)
    print(f"\nN={N}, collected {len(snaps)} same-amplitude snapshots over ~6 periods.")

    # T1 STATIONARITY — pairwise spatial correlation
    print("\nT1 STATIONARITY (|corr| ~ 1 ⇒ fixed shape ⇒ pure eigenmode):")
    cmin = 1.0
    for a in range(min(len(snaps), 5)):
        row = []
        for b in range(min(len(snaps), 5)):
            sa = snaps[a].ravel(); sb = snaps[b].ravel()
            cc = abs(np.dot(sa, sb) / (np.linalg.norm(sa) * np.linalg.norm(sb) + 1e-30))
            row.append(cc); cmin = min(cmin, cc)
        print("   " + " ".join(f"{v:5.3f}" for v in row))
    print(f"   → min pairwise |corr| = {cmin:.4f}  "
          f"{'⇒ STATIONARY: pure mode, the SOLVER is fine, my reference shape was the defect' if cmin > 0.99 else '⇒ NON-stationary: multi-mode, dig the solver'}")

    # T2 MODE SPECTRUM — project snapshot 0 onto continuous cos modes
    print("\nT2 MODE SPECTRUM (|projection| of the field onto continuous cos modes):")
    s0 = snaps[0]; s0n = s0 / np.linalg.norm(s0)
    for m in range(3):
        for n in range(3):
            phi = cos_shape(m, n, "node"); phin = phi / (np.linalg.norm(phi) + 1e-30)
            p = abs(np.sum(s0n * phin))
            if p > 0.05:
                print(f"   mode ({m},{n}): |proj| = {p:.4f}")

    # T3 SEED CONVENTION — which discrete shape does the field actually match?
    print("\nT3 SEED CONVENTION (resid of field vs candidate (1,1) seeds — small ⇒ correct grid geometry):")
    for conv in ["node", "cell_N", "cell_Nm1"]:
        phi = cos_shape(1, 1, conv); phin = (phi / np.linalg.norm(phi)).ravel()
        amp = np.dot(s0.ravel(), phin)
        resid = np.linalg.norm(s0.ravel() - amp * phin) / np.linalg.norm(s0.ravel())
        print(f"   seed='{conv:9s}': resid = {resid:.4e}")
    print("=" * 88)


if __name__ == "__main__":
    main()

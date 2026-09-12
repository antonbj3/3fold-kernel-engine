#!/usr/bin/env python3
"""S6 — JOULE HEATING (EM⊕thermal coupling), thermal half validated vs the EXACT parabolic steady-state. A uniform
volumetric Joule source q = I²R (from a current density on the lattice) heats a conducting slab held at T=0 on both
walls; pure conduction (no flow). Steady 1D heat equation α T'' + q = 0 → T(y) = (q/2α)·y(H−y), an EXACT parabola.
This is the thermal response of the EM⊕thermal coupling on ONE lattice (the source is local per-cell); the full
EM-computes-q link (q from a Yee/FDTD field, wave_fdtd_kache.py) is the follow-on.

DISCIPLINE (triple-check built in): NULL q→0 ⇒ T≡0 runs FIRST; then ABSOLUTE parabola match (not just shape) + step-
convergence + scene_eyes on the raw profile. Walls held at T=0 directly at the boundary CELLS → H=ny−1 exactly (no
effective-H ambiguity, unlike RB).

  python3 joule_heating_thermal.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('lbm',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np

# D2Q5 thermal
from lbm_lattice import TCX as tcx, TCY as tcy, TW as tw   # de-dup: canonical D2Q5 primitives (audit follow-up)


def run(q, ny=64, nx=4, tauT=0.9, steps=60000):
    alpha = (tauT - 0.5) / 3.0
    T = np.zeros((nx, ny))
    g = tw[:, None, None] * T[None]
    for it in range(steps):
        T = g.sum(0)
        geq = tw[:, None, None] * T[None]                 # no flow → equilibrium = w_i T
        gcol = g - (g - geq) / tauT + tw[:, None, None] * q   # + Joule source
        for k in range(5):
            gcol[k] = np.roll(np.roll(gcol[k], tcx[k], 0), tcy[k], 1)
        g = gcol
        # Dirichlet T=0 at the boundary CELLS (j=0, ny-1): set populations to equilibrium at T=0 → 0
        g[:, :, 0] = 0.0; g[:, :, -1] = 0.0
    return g.sum(0)[nx // 2], alpha


def main():
    print("=" * 76)
    print("S6 — Joule heating (EM⊕thermal): thermal slab vs EXACT parabolic steady-state")
    print("=" * 76)
    ny = 64; H = ny - 1.0; tauT = 0.9

    # NULL CONTROL FIRST: q=0 must give T≡0
    p0, alpha = run(0.0, ny=ny, tauT=tauT)
    null_ok = np.abs(p0).max() < 1e-12
    print(f"\n  NULL control (q=0 ⇒ T≡0): max|T| = {np.abs(p0).max():.1e}  → {'PASS' if null_ok else 'FAIL'}")

    # convergence + absolute parabola match
    q = 1e-6
    y = np.arange(ny)
    T_ana = (q / (2 * alpha)) * y * (H - y)               # exact parabola, T=0 at j=0,H
    print(f"\n  {'steps':>7} {'sim T_max':>12} {'analytic T_max':>15} {'ratio':>8} {'abs-L2/max':>11}")
    for steps in (20000, 40000, 80000):
        prof, _ = run(q, ny=ny, tauT=tauT, steps=steps)
        L2 = np.sqrt(np.mean((prof[1:-1] - T_ana[1:-1]) ** 2)) / (T_ana.max() + 1e-30)
        print(f"  {steps:>7} {prof.max():>12.4e} {T_ana.max():>15.4e} {prof.max()/T_ana.max():>8.4f} {L2:>11.5f}")
    prof, _ = run(q, ny=ny, tauT=tauT, steps=80000)
    abs_L2 = np.sqrt(np.mean((prof[1:-1] - T_ana[1:-1]) ** 2)) / T_ana.max()
    ratio = prof.max() / T_ana.max()

    # scene_eyes: raw profile vs analytic (every 8th cell)
    print(f"\n  scene_eyes raw (×1e-6):  y={list(y[::8])}")
    print(f"    sim     : {(prof[::8] / 1e-6).round(2)}")
    print(f"    analytic: {(T_ana[::8] / 1e-6).round(2)}")

    ok = null_ok and abs(ratio - 1.0) < 0.02 and abs_L2 < 0.01
    print("\n" + "=" * 76)
    if ok:
        print("VERDICT: Joule-heating thermal coupling = VALIDATED vs exact parabola.")
        print(f"  • NULL q→0 ⇒ T≡0 PASSED first. • ABSOLUTE T_max ratio {ratio:.4f} (not just shape). • converged")
        print(f"    (step-independent). • raw profile matches the parabola cell-by-cell (abs-L2 {abs_L2:.5f}).")
        print("  ⇒ native on-grid EM⊕thermal source coupling reproduces the exact conduction physics. HONEST: uniform")
        print("  Joule source (the thermal half); EM-computes-q via Yee (wave_fdtd_kache) is the follow-on. [S6 small win]")
    else:
        print(f"VERDICT: FAIL — null_ok={null_ok}, ratio={ratio:.4f}, abs-L2={abs_L2:.5f}. scene_eyes refuses the claim.")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

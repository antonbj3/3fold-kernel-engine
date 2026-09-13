#!/usr/bin/env python3
"""GENCHI GENBUTSU pt2 — what does the acoustic-rigid (1,1) seed DRIFT INTO over ~18 periods?
T1 showed the drift is REAL (direct snapshot corr 1.0→0.82 over 6 periods), energy-conserving, no (2,1)
leakage. Project the field onto a FULL cos-mode basis at antinode times → identify the growing contaminant
→ then fix the real defect (full diligence, unlimited resources; no skipping baselines).

  python3 diag_acoustic_spectrum.py
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
x = np.arange(N) / (N - 1)


def cm(m, n):
    return np.outer(np.cos(m * np.pi * x), np.cos(n * np.pi * x))


basis = {(m, n): (cm(m, n) / np.linalg.norm(cm(m, n))).ravel() for m in range(7) for n in range(7)}

S, U, W = make_fields(N, N)
S.assign(wp.array(cm(1, 1).astype(np.float32), dtype=wp.float32, device=DEV))
f11 = (c / 2.0) * np.sqrt(2.0) / Lx; T = 1.0 / f11
cps = {int(round(k * T / dt)): f"{k}T" for k in [0.5, 1, 3, 6, 10, 15, 18]}
maxstep = max(cps) + 2

print("=" * 88)
print(f"ACOUSTIC RIGID — spectrum over time (N={N}); what does the cos(1,1) seed drift INTO?")
print("=" * 88)
for it in range(maxstep + 1):
    if it in cps:
        s = S.numpy().astype(np.float64).ravel(); ns = np.linalg.norm(s)
        if ns > 1e-9:
            sn = s / ns
            pj = {k: abs(float(np.dot(sn, b))) for k, b in basis.items()}
            top = sorted(pj.items(), key=lambda kv: -kv[1])[:6]
            print(f" t={cps[it]:4s} ‖S‖={ns:8.3f}: " + "  ".join(f"{k}:{v:.3f}" for k, v in top))
    step(S, U, W, ca, cb, csx, csy, N, N, bflag)
print("=" * 88)
print("READ: if drift is into ADJACENT modes ((3,1)/(1,3)) → seed excites a superposition (continuous cos ≠")
print("discrete eigenvector). If into DC (0,0) or high-k/Nyquist → a real BC/stencil defect to fix.")
print("=" * 88)

#!/usr/bin/env python3
"""DECISIVE: which seed is the TRUE discrete rigid eigenmode (stays stationary ⇒ solver correct, gauge wrong)?
Geometric hypothesis: rigid walls at outer cell faces ⇒ discrete Neumann eigenmode = cos(mπ(i+½)/N) on a
domain of length N·dx, NOT the node-sampled cos(mπ·i/(N−1)). The stationary one is the true eigenmode."""
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
i = np.arange(N)


def seed_node():
    a = np.cos(np.pi * i / (N - 1)); return np.outer(a, a)


def seed_cell():
    a = np.cos(np.pi * (i + 0.5) / N); return np.outer(a, a)


f = (c / 2.0) * np.sqrt(2.0) / Lx; T = 1.0 / f


def stat(seedfn, lab):
    sd = seedfn()
    S, U, W = make_fields(N, N)
    S.assign(wp.array(sd.astype(np.float32), dtype=wp.float32, device=DEV))
    snaps = []
    nsd = np.linalg.norm(sd.ravel())
    for it in range(int(18 * T / dt)):
        step(S, U, W, ca, cb, csx, csy, N, N, bflag)
        if it % 29 == 0:
            s = S.numpy().ravel().astype(np.float64)
            if np.linalg.norm(s) > 0.5 * nsd:
                snaps.append(s)
    cmin = 1.0
    for a in range(len(snaps)):
        for b in range(a + 1, len(snaps)):
            cc = abs(np.dot(snaps[a], snaps[b]) / (np.linalg.norm(snaps[a]) * np.linalg.norm(snaps[b]) + 1e-30))
            cmin = min(cmin, cc)
    tag = "← STATIONARY = the true discrete eigenmode (solver correct)" if cmin > 0.99 else ""
    print(f"  {lab:16s}: {len(snaps):3d} antinode snaps, min|corr| = {cmin:.4f}  {tag}")


print("=" * 80)
print(f"ACOUSTIC RIGID (N={N}) — which seed stays stationary (= true discrete eigenmode)?")
print("=" * 80)
stat(seed_node, "node i/(N-1)")
stat(seed_cell, "cell (i+0.5)/N")
print("=" * 80)

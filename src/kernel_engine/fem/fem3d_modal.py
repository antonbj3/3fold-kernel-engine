#!/usr/bin/env python3
"""3D MODAL FEM - eigenfrequencies of a 3D structure (consistent mass + generalised eigenvalue problem). 3D dynamics.

Builds on fem3d_elasticity (Q1 hex stiffness K): adds the CONSISTENT mass matrix M_e = rho integral N^T N dV -> generalised eigen-
problem K phi = omega^2 M phi -> eigenfrequencies f=omega/2pi. It completes the 3D thread: statics (fem3d_elasticity) + thermal
(fem3d_thermoelastic) + DYNAMICS (this). It ties into the vibration/acoustics thread (eigenfrequencies).

VALIDATION (mesh convergence against ANALYTIC - a continuum eigenvalue, not machine precision): a laterally constrained bar
(u_y=u_z=0 everywhere, u_x=0 at x=0, free x end) -> pure 1D P-wave axial vibration, f_n=(2n-1)/(4L).sqrt(M/rho), M=lambda+2mu
(the P-wave modulus; lateral constraint -> the constrained modulus, NOT the bar modulus E). GATE: (1) f_1 ~ analytic (<2% at a moderate mesh);
(2) CONVERGENCE: a finer mesh gets closer to analytic (the error falls); (3) overtones f_2/f_1~3, f_3/f_1~5 (odd harmonics);
(4) egenfrekvenser positiva + reella + stigande; (5) M symmetrisk positiv-definit (massa fysisk).

  CUDA_VISIBLE_DEVICES="" python3 fem3d_modal.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('fem',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fem3d_elasticity import FEM3D

NCOORD = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
          (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]


def hex_Me(h, rho):
    """Konsistent hex-massmatris M_e = ρ∫NᵀN dV (24×24) via 2×2×2 Gauss."""
    g = 1 / np.sqrt(3); gp = [-g, g]
    Me = np.zeros((24, 24))
    for xi in gp:
        for eta in gp:
            for ze in gp:
                N = np.zeros(8)
                for i, (a, b, c) in enumerate(NCOORD):
                    N[i] = 0.125 * (1 + a * xi) * (1 + b * eta) * (1 + c * ze)
                Nmat = np.zeros((3, 24))
                for i in range(8):
                    Nmat[0, 3 * i] = N[i]; Nmat[1, 3 * i + 1] = N[i]; Nmat[2, 3 * i + 2] = N[i]
                Me += rho * (Nmat.T @ Nmat) * (h / 2) ** 3
    return Me


def assemble_M(fem, rho):
    Me = hex_Me(fem.h, rho)
    rows, cols, vals = [], [], []
    for ez in range(fem.nelz):
        for ey in range(fem.nely):
            for ex in range(fem.nelx):
                ed = fem._edof(ex, ey, ez)
                rows.append(np.repeat(ed, 24)); cols.append(np.tile(ed, 24)); vals.append(Me.flatten())
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(fem.ndof, fem.ndof))


def axial_modes(fem, rho, nmodes=4):
    """Laterally constrained bar (u_y=u_z=0 everywhere, u_x=0 at x=0) -> axial eigenfrequencies [Hz]."""
    M = assemble_M(fem, rho)
    fixed = []
    for k in range(fem.nz):
        for j in range(fem.ny):
            for i in range(fem.nx):
                n = fem.nid(i, j, k)
                fixed += [3 * n + 1, 3 * n + 2]          # u_y=0, u_z=0 everywhere (lateral constraint -> pure axial)
                if i == 0:
                    fixed.append(3 * n)                  # u_x=0 at x=0 (fixed end)
    fixed = np.unique(fixed)
    free = np.setdiff1d(np.arange(fem.ndof), fixed)
    Kff = csr_matrix(fem.K[free][:, free]); Mff = csr_matrix(M[free][:, free])
    w2, _ = eigsh(Kff, k=nmodes, M=Mff, sigma=0, which="LM")   # smallest eigenvalues (shift-invert)
    w2 = np.sort(w2[w2 > 0])
    return np.sqrt(w2) / (2 * np.pi), M


def main():
    print("3D MODAL FEM (Q1 hex, konsistent massa) — egenfrekvenser (3D-dynamik)")
    E = 200e9; nu = 0.3; rho = 7800.0; h = 0.05
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu)); Mmod = lam + 2 * mu
    c = np.sqrt(Mmod / rho)                                # P-wave speed
    fem = FEM3D(nelx=20, nely=2, nelz=2, h=h, E=E, nu=nu)
    L = fem.nelx * h
    f_an = c / (4 * L)                                     # f_1 = c/4L (fixed-free, P-wave)
    print(f"  bar L={L} m ({fem.ndof} DOF), E={E/1e9:.0f} GPa nu={nu} rho={rho}; P modulus M={Mmod/1e9:.0f} GPa, c={c:.0f} m/s")

    fmodes, M = axial_modes(fem, rho, nmodes=4)
    f1 = fmodes[0]
    print(f"  egenfrekvenser (Hz): {[round(f,1) for f in fmodes]}")
    print(f"  f_1={f1:.1f} Hz vs analytic c/4L={f_an:.1f} Hz (error {abs(f1-f_an)/f_an:.2%})")

    # (1) f_1 ≈ analytisk
    g1 = abs(f1 - f_an) / f_an < 0.02
    # (2) CONVERGENCE: a finer mesh gets closer
    fem2 = FEM3D(nelx=40, nely=2, nelz=2, h=h / 2, E=E, nu=nu)   # twice the resolution, same L
    f1_fine = axial_modes(fem2, rho, nmodes=2)[0][0]
    err_c = abs(f1 - f_an) / f_an; err_f = abs(f1_fine - f_an) / f_an
    g2 = err_f < err_c
    print(f"  (2) convergence: coarse error {err_c:.2%} -> fine error {err_f:.2%} ({'falling' if g2 else 'RISING'})")
    # (3) overtones: f_2/f_1~3, f_3/f_1~5
    r2 = fmodes[1] / f1; r3 = fmodes[2] / f1
    g3 = abs(r2 - 3) < 0.15 and abs(r3 - 5) < 0.30
    print(f"  (3) overtones: f2/f1={r2:.2f} (~3), f3/f1={r3:.2f} (~5)")
    # (4) frekvenser positiva, reella, stigande
    g4 = np.all(fmodes > 0) and np.all(np.diff(fmodes) > 0)
    # (5) M symmetrisk positiv-definit
    Md = M.toarray() if fem.ndof < 5000 else None
    sym = abs((M - M.T)).max() < 1e-6 * abs(M).max()
    pd = np.all(np.linalg.eigvalsh(Md) > 0) if Md is not None else True
    g5 = sym and pd
    print(f"  (4) frekvenser positiva/stigande: {g4}; (5) M symmetrisk+PD: sym {sym}, PD {pd}")

    ok = g1 and g2 and g3 and g4 and g5
    print(f"\nVERDICT: 3D modal FEM = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"3D eigenfrequencies (consistent mass + generalised eigenvalue K phi = omega^2 M phi) validated against ANALYTIC results (mesh "
             f"konvergens): (1) f_1={f1:.0f}Hz ≈ c/4L={f_an:.0f}Hz ({err_c:.1%}); (2) KONVERGERAR (grov {err_c:.1%}→fin "
             f"{err_f:.1%}); (3) overtones f2/f1~3, f3/f1~5 (odd harmonics, a correct P-wave bar); (4) positive and increasing; "
             f"(5) M symmetric + positive definite. -> the 3D thread is complete: statics + thermal + DYNAMICS; it ties into vibration/acoustics. " if ok else
             f"Not validated (f1 {g1}, convergence {g2}, overtones {g3}, positivity {g4}, M {g5}) - debug. ")
          + "CAVEAT: a continuum eigenvalue -> mesh-convergent (not machine precision); a laterally CONSTRAINED bar (u_y=u_z=0) -> "
          "the P-wave modulus M=lambda+2mu (NOT the bar modulus E - a laterally free bar gives E but mixes in bending modes); consistent mass "
          "(slightly overestimates f, converges); undamped (eigenfrequencies, not damped). It relates to measured resonance "
          "work but this is SIM modal only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

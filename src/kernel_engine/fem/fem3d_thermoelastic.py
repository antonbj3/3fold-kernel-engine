#!/usr/bin/env python3
"""3D THERMO-elastic FEM - thermal load (eigenstrain alpha deltaT) -> 3D thermal stress field. Multiphysics (3D structure + thermal).

Builds on fem3d_elasticity (Q1 hex): adds the thermal eigenstrain eps_th = alpha deltaT [1,1,1,0,0,0] -> thermal load vector
F_th = integral B^T D eps_th dV. Solves K u = F_th with given BC -> thermal displacement + stress sigma = D(B u - eps_th). The 3D analogue
of warpfem_thermoelastic (2D). Composes 3D structural physics + thermal expansion -> thermal stress in 3D.

GATE (rigorous, against EXACT analytics): (1) BAR CONSTRAINT (eps_xx=0, laterally free): sigma_xx = -E.alpha.deltaT (exact), sigma_yy=sigma_zz~0;
(2) FULL CONSTRAINT (u=0 on all faces): hydrostatic sigma = -E.alpha.deltaT/(1-2nu) (exact), shear ~0; (3) deltaT=0 -> sigma=0 (no spurious
load); (4) LINEARITY in deltaT (sigma proportional to deltaT); (5) sign: heating + constraint -> COMPRESSION (sigma<0).

  CUDA_VISIBLE_DEVICES="" python3 fem3d_thermoelastic.py
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
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fem3d_elasticity import FEM3D, D_iso

NCOORD = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
          (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]


def _B_at(xi, eta, ze, h):
    """B matrix (6x24) at natural coordinates (xi,eta,ze) for a cubic element of side h."""
    dN = np.zeros((3, 8))
    for i, (a, b, c) in enumerate(NCOORD):
        dN[0, i] = 0.125 * a * (1 + b * eta) * (1 + c * ze)
        dN[1, i] = 0.125 * b * (1 + a * xi) * (1 + c * ze)
        dN[2, i] = 0.125 * c * (1 + a * xi) * (1 + b * eta)
    dNxyz = dN * (2.0 / h)
    B = np.zeros((6, 24))
    for i in range(8):
        B[0, 3 * i] = dNxyz[0, i]; B[1, 3 * i + 1] = dNxyz[1, i]; B[2, 3 * i + 2] = dNxyz[2, i]
        B[3, 3 * i] = dNxyz[1, i]; B[3, 3 * i + 1] = dNxyz[0, i]
        B[4, 3 * i + 1] = dNxyz[2, i]; B[4, 3 * i + 2] = dNxyz[1, i]
        B[5, 3 * i] = dNxyz[2, i]; B[5, 3 * i + 2] = dNxyz[0, i]
    return B


def thermal_force(fem, alpha, dT):
    """Termisk lastvektor F_th = ∫ Bᵀ D ε_th dV (2×2×2 Gauss), ε_th = αΔT·[1,1,1,0,0,0]."""
    g = 1 / np.sqrt(3); gp = [-g, g]
    eps_th = alpha * dT * np.array([1, 1, 1, 0, 0, 0.0])
    Dq = fem.D @ eps_th
    F = np.zeros(fem.ndof)
    for ez in range(fem.nelz):
        for ey in range(fem.nely):
            for ex in range(fem.nelx):
                fe = np.zeros(24)
                for xi in gp:
                    for eta in gp:
                        for ze in gp:
                            B = _B_at(xi, eta, ze, fem.h)
                            fe += (B.T @ Dq) * (fem.h / 2) ** 3
                ed = fem._edof(ex, ey, ez)
                F[ed] += fe
    return F


def solve_thermal(fem, alpha, dT, mode):
    """Solve the thermal load with BC. mode='bar' (eps_xx=0, laterally free) or 'full' (u=0 on all faces). -> (u, fixed)."""
    fixed = []
    if mode == "bar":
        for k in range(fem.nz):
            for j in range(fem.ny):
                fixed += [3 * fem.nid(0, j, k), 3 * fem.nid(fem.nelx, j, k)]   # u_x=0 on +-x
        for k in range(fem.nz):
            for i in range(fem.nx):
                fixed.append(3 * fem.nid(i, 0, k) + 1)        # u_y=0 on -y (roller)
        for j in range(fem.ny):
            for i in range(fem.nx):
                fixed.append(3 * fem.nid(i, j, 0) + 2)        # u_z=0 on -z (roller)
    else:  # full: alla yt-noders alla DOF = 0
        for k in range(fem.nz):
            for j in range(fem.ny):
                for i in range(fem.nx):
                    if i in (0, fem.nelx) or j in (0, fem.nely) or k in (0, fem.nelz):
                        n = fem.nid(i, j, k); fixed += [3 * n, 3 * n + 1, 3 * n + 2]
    fixed = np.unique(fixed)
    free = np.setdiff1d(np.arange(fem.ndof), fixed)
    F = thermal_force(fem, alpha, dT)
    u = np.zeros(fem.ndof)
    u[free] = splu(csr_matrix(fem.K[free][:, free]).tocsc()).solve(F[free])
    return u, fixed


def stress(fem, u, alpha, dT):
    """Stress per element (centre): sigma = D(B u - eps_th)."""
    eps_th = alpha * dT * np.array([1, 1, 1, 0, 0, 0.0])
    B0 = _B_at(0, 0, 0, fem.h)
    out = []
    for ez in range(fem.nelz):
        for ey in range(fem.nely):
            for ex in range(fem.nelx):
                ed = fem._edof(ex, ey, ez)
                out.append(fem.D @ (B0 @ u[ed] - eps_th))
    return np.array(out)


def main():
    print("3D THERMO-elastic FEM (Q1 hex) - thermal eigenstrain -> 3D thermal stress field (multiphysics)")
    E = 200e9; nu = 0.3; alpha = 1.2e-5; dT = 50.0
    fem = FEM3D(nelx=6, nely=6, nelz=6, h=1.0, E=E, nu=nu)
    print(f"  grid 6×6×6 ({fem.ndof} DOF), E={E/1e9:.0f}GPa ν={nu} α={alpha} ΔT={dT}K")

    # (1) BAR CONSTRAINT: sigma_xx = -E alpha deltaT
    u1, _ = solve_thermal(fem, alpha, dT, "bar")
    s1 = stress(fem, u1, alpha, dT)
    sxx = s1[:, 0]; syy = s1[:, 1]
    sxx_an = -E * alpha * dT
    g1 = abs(sxx.mean() - sxx_an) / abs(sxx_an) < 1e-4 and sxx.std() / abs(sxx_an) < 1e-4 and abs(syy.mean()) / abs(sxx_an) < 1e-4
    print(f"  (1) bar constraint: sigma_xx={sxx.mean()/1e6:.3f} MPa vs analytic -E alpha deltaT={sxx_an/1e6:.3f} MPa "
          f"(error {abs(sxx.mean()-sxx_an)/abs(sxx_an):.1e}, std {sxx.std()/abs(sxx_an):.1e}); sigma_yy/|sigma_xx| {abs(syy.mean())/abs(sxx_an):.1e}")

    # (2) FULL CONSTRAINT: hydrostatic sigma = -E alpha deltaT/(1-2nu)
    u2, _ = solve_thermal(fem, alpha, dT, "full")
    s2 = stress(fem, u2, alpha, dT)
    hydro = s2[:, :3].mean(axis=1)
    hydro_an = -E * alpha * dT / (1 - 2 * nu)
    # audit fix: full constraint + uniform alpha deltaT -> EXACTLY u=0 everywhere (interior nodes have F_th=0 via integral B^T=0) -> hydrostatic
    # sigma exact on ALL elements. The earlier 5% / centre-only test was an unnecessary workaround with a false 'surface elements deviate' comment
    # (VERIFIED against raw data: u=0 exactly, max error 0e+00). Tightened to machine precision over the WHOLE field + shear ~0.
    g2 = np.max(np.abs(hydro - hydro_an)) / abs(hydro_an) < 1e-10 and np.max(np.abs(s2[:, 3:])) / abs(hydro_an) < 1e-10
    print(f"  (2) full constraint (ALL elements): sigma_hydro={hydro.mean()/1e6:.1f} MPa vs analytic -E alpha deltaT/(1-2nu)={hydro_an/1e6:.1f} MPa "
          f"(max error {np.max(np.abs(hydro-hydro_an))/abs(hydro_an):.0e}, shear/|sigma| {np.max(np.abs(s2[:,3:]))/abs(hydro_an):.0e})")

    # (3) ΔT=0 → σ=0
    u0, _ = solve_thermal(fem, alpha, 0.0, "bar")
    s0 = stress(fem, u0, alpha, 0.0)
    g3 = np.abs(s0).max() / abs(sxx_an) < 1e-10
    print(f"  (3) deltaT=0 -> sigma_max={np.abs(s0).max():.2e} (no spurious load)")

    # (4) LINEARITY: sigma_xx(2 deltaT) = 2 sigma_xx(deltaT)
    u4, _ = solve_thermal(fem, alpha, 2 * dT, "bar")
    sxx4 = stress(fem, u4, alpha, 2 * dT)[:, 0].mean()
    g4 = abs(sxx4 - 2 * sxx.mean()) / abs(2 * sxx.mean()) < 1e-10
    print(f"  (4) linearity: sigma_xx(2 deltaT)={sxx4/1e6:.2f} = 2 sigma_xx(deltaT)={2*sxx.mean()/1e6:.2f} (error {abs(sxx4-2*sxx.mean())/abs(2*sxx.mean()):.1e})")

    # (5) sign: heating + constraint -> compression (sigma<0). Audit fix: hydro[cen] -> hydro.mean() (the field is uniform)
    g5 = sxx.mean() < 0 and hydro.mean() < 0
    print(f"  (5) sign: heating + constraint -> COMPRESSION (sigma_xx<0: {sxx.mean()<0}, sigma_hydro<0: {hydro.mean()<0})")

    ok = g1 and g2 and g3 and g4 and g5
    print(f"\nVERDICT: 3D thermo-elastic FEM = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Thermal eigenstrain (alpha deltaT) -> 3D thermal stress field, validated against ANALYTIC results: (1) bar constraint sigma_xx=-E alpha deltaT "
             f"exact ({abs(sxx.mean()-sxx_an)/abs(sxx_an):.0e}) + laterally free (sigma_yy~0); (2) full constraint hydrostatic sigma=-E alpha deltaT/"
             f"(1-2nu) ({np.max(np.abs(hydro-hydro_an))/abs(hydro_an):.0e} on ALL elements, exact u=0); (3) deltaT=0 -> sigma=0; (4) linear in deltaT; (5) heating + "
             "constraint -> compression. -> MULTIPHYSICS: 3D structure + thermal expansion -> thermal stress (composes fem3d_elasticity + "
             "termik; 3D-analog av warpfem_thermoelastic). " if ok else
             f"Not validated (bar {g1}, full {g2}, deltaT0 {g3}, linear {g4}, sign {g5}) - debug. ")
          + "CAVEAT: linear elastic + linear thermo-elasticity (constant alpha, small strain); uniform deltaT (a spatial T gradient "
          "requires coupling a thermal solver, further work); the full-constraint case is EXACT on ALL elements (u=0, machine precision); "
          "structured hex grid. Composes 3D FEM + thermal eigenstrain.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

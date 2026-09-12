#!/usr/bin/env python3
"""B2 — NATIVE ON-GRID MHD coupling (LBM fluid ⊕ electromagnetic Lorentz force), validated against the EXACT analytic
HARTMANN flow. Geometric: a transverse magnetic field B_y on a channel flow induces J ∝ σ(u×B) and a Lorentz force
J×B = −σB²u_x — a LOCAL per-cell drag opposing the flow (low magnetic-Reynolds MHD). It flattens the parabolic profile
into the Hartmann profile, controlled by Ha = B·(H/2)·√(σ/(ρν)). This is native on-grid multiphysics: the EM drag is a
local body force from the local velocity, on the SAME lattice — no inter-solver handoff.

scene_eyes = the EXACT Hartmann profile (no calibration ambiguity, unlike RB). DISCIPLINE: the GENERIC/NULL control runs
FIRST — Ha→0 MUST recover plain parabolic Poiseuille; only then trust the flattening at higher Ha.

  python3 hartmann_mhd_lbm.py
"""
import sys
import numpy as np

cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]); cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]); opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def feq(rho, ux, uy):
    cu = cx[:, None, None] * ux[None] + cy[:, None, None] * uy[None]
    return w[:, None, None] * rho[None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * (ux ** 2 + uy ** 2)[None])


def hartmann_analytic(y, H, Ha):
    s = 2 * y / H - 1.0                                   # ∈[-1,1], 0 at centerline
    return (np.cosh(Ha) - np.cosh(Ha * s)) / (np.cosh(Ha) - 1.0 + 1e-30)   # 0 at walls, max at centre


def run(Ha, ny=64, nx=8, steps=40000):
    H = ny - 1.0
    nu = 0.05; tau = 3 * nu + 0.5
    G = 1e-6                                              # constant pressure-driving body force (+x)
    drag = (Ha ** 2) * nu / ((H / 2.0) ** 2)             # σB² s.t. Ha = B(H/2)√(σ/ρν)  → drag coeff
    rho = np.ones((nx, ny)); ux = np.zeros((nx, ny)); uy = np.zeros((nx, ny))
    f = feq(rho, ux, uy)
    for it in range(steps):
        rho = f.sum(0); ux = (cx[:, None, None] * f).sum(0) / rho; uy = (cy[:, None, None] * f).sum(0) / rho
        Fx = G - drag * ux                               # pressure drive + LOCAL Lorentz drag −σB²u_x
        feqv = feq(rho, ux, uy)
        feqF = feq(rho, ux + Fx / rho, uy)               # EDM forcing (shifted-equilibrium)
        fcol = f - (f - feqv) / tau + (feqF - feqv)
        for q in range(9):
            fcol[q] = np.roll(np.roll(fcol[q], cx[q], 0), cy[q], 1)
        f = fcol
        # no-slip half-way bounce-back at the two walls (only wall-normal incoming unknowns)
        f[2, :, 0] = f[4, :, 0]; f[5, :, 0] = f[7, :, 0]; f[6, :, 0] = f[8, :, 0]
        f[4, :, -1] = f[2, :, -1]; f[7, :, -1] = f[5, :, -1]; f[8, :, -1] = f[6, :, -1]
    rho = f.sum(0); ux = (cx[:, None, None] * f).sum(0) / rho
    prof = ux[nx // 2]                                    # velocity profile across the channel
    return prof, H


def main():
    print("=" * 80)
    print("B2 — MHD Hartmann flow on the LBM lattice (local Lorentz drag) vs EXACT analytic profile")
    print("=" * 80)
    print(f"\n  {'Ha':>5} {'profile-shape L2 err vs analytic':>33} {'flattening (u_wall_adj/u_mid)':>30}")
    errs = {}
    for Ha in (0.01, 1.0, 3.0, 8.0, 15.0):
        prof, H = run(Ha)
        y = np.arange(len(prof))
        ana = hartmann_analytic(y, H, max(Ha, 1e-6))
        # normalize both by their interior max (exclude walls), compare SHAPE
        pn = prof / (prof[1:-1].max() + 1e-30); an = ana / (ana[1:-1].max() + 1e-30)
        L2 = np.sqrt(np.mean((pn[1:-1] - an[1:-1]) ** 2))
        flat = prof[2] / (prof[len(prof) // 2] + 1e-30)  # near-wall/centre ratio → rises toward 1 with Ha (flatter)
        errs[Ha] = L2
        tag = '  ← NULL: must be PARABOLIC (Poiseuille)' if Ha < 0.05 else ''
        print(f"  {Ha:>5.2f} {L2:>33.4f} {flat:>30.3f}{tag}")

    null_ok = errs[0.01] < 0.03                           # Ha→0 recovers Poiseuille (the generic control)
    high_ok = errs[8.0] < 0.05 and errs[15.0] < 0.06     # Hartmann flattening matches analytic at high Ha
    print("\n" + "=" * 80)
    if null_ok and high_ok:
        print("VERDICT: MHD Hartmann coupling = VALIDATED against the EXACT analytic profile.")
        print(f"  • NULL control PASSED first: Ha→0 recovers parabolic Poiseuille (shape-L2 {errs[0.01]:.4f}).")
        print(f"  • Hartmann FLATTENING matches analytic across Ha (L2 {errs[8.0]:.4f}@Ha=8, {errs[15.0]:.4f}@Ha=15).")
        print("  ⇒ native on-grid MHD: a LOCAL per-cell Lorentz drag (EM ⊗ fluid on ONE lattice) reproduces the exact")
        print("  Hartmann boundary-layer physics. Clean analytic control — no calibration ambiguity (unlike RB Ra_c).")
    elif null_ok:
        print(f"VERDICT: NULL ok (Poiseuille recovered, {errs[0.01]:.4f}) but high-Ha flattening off (L2 {errs[8.0]:.4f},")
        print(f"  {errs[15.0]:.4f}) — the local-drag model or the wall BL resolution needs work. Honest partial.")
    else:
        print(f"VERDICT: NULL control FAILED (Ha→0 shape-L2 {errs[0.01]:.4f}, not parabolic) → the base channel solver is")
        print("  wrong BEFORE adding MHD. scene_eyes refuses the MHD claim until the null passes. Fix Poiseuille first.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

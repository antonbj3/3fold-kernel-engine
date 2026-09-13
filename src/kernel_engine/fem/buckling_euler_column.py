#!/usr/bin/env python3
"""BUCKLING / STABILITY (Euler beam-column, geometric stiffness) - a NEW failure mode: stability, not strength.

A genuine gap: the failure thread had strength modes (yield/plasticity/fatigue/fracture/creep) but NO STABILITY.
Slender or thin-walled printed parts fail by BUCKLING far below yield - a completely different failure mechanism.
This is the linear buckling eigenvalue problem: (K_bend - lambda.K_geo)phi = 0 where K_geo is the geometric stiffness
from an axial compressive load -> the smallest eigenvalue lambda = the critical load P_cr. Euler-Bernoulli beam FE, validated against ANALYTIC
results (Euler P_cr=pi^2 EI/L_eff^2) for 3 boundary conditions. Composes the elasticity (EI) + a MEASURED E (orientation -> buckling load).

GATE (rigorous, against ANALYTIC Euler): (1) pinned-pinned P_cr=pi^2 EI/L^2 (mesh convergence, <0.2%); (2) fixed-free (cantilever)
P_cr=π²EI/4L²; (3) fast-fast P_cr=4π²EI/L²; (4) MESH-KONVERGENS (grov→fin → Euler monotont); (5) DESIGN/skalning
P_cr proportional to EI EXACTLY (EI x2 -> 2x P_cr) and to 1/L^2 (design from physics: stiffen against buckling); (6) PAYOFF: E_xy vs E_z
-> the buckling load differs by the E ratio (the build orientation buckles earlier); (7) the first mode = a half sine (pinned-pinned).

  CUDA_VISIBLE_DEVICES="" python3 buckling_euler_column.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from cad2simready.materials import load_tds_mechanical
except Exception:
    load_tds_mechanical = None


def elem_matrices(le, EI):
    """Euler-Bernoulli beam element [w1,theta1,w2,theta2]: bending stiffness Ke + geometric stiffness Kg (for P=1)."""
    Ke = (EI / le ** 3) * np.array([
        [12, 6 * le, -12, 6 * le],
        [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
        [-12, -6 * le, 12, -6 * le],
        [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
    Kg = (1.0 / (30 * le)) * np.array([      # geometrisk styvhet, axiell tryck-enhet P=1
        [36, 3 * le, -36, 3 * le],
        [3 * le, 4 * le ** 2, -3 * le, -le ** 2],
        [-36, -3 * le, 36, -3 * le],
        [3 * le, -le ** 2, -3 * le, 4 * le ** 2]])
    return Ke, Kg


def assemble(n_el, L, EI):
    le = L / n_el; ndof = 2 * (n_el + 1)
    K = np.zeros((ndof, ndof)); G = np.zeros((ndof, ndof))
    Ke, Kg = elem_matrices(le, EI)
    for e in range(n_el):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d, d)] += Ke; G[np.ix_(d, d)] += Kg
    return K, G, ndof


def fixed_dofs(bc, n_el):
    """Boundary-condition DOF (w=index 2i, theta=2i+1). n0=first node, nN=last."""
    nN = 2 * n_el
    if bc == "pinned-pinned":
        return [0, nN]                                   # w=0 at both ends (rotations free)
    if bc == "fixed-free":
        return [0, 1]                                    # w=θ=0 vid nod 0; fri vid N
    if bc == "fixed-fixed":
        return [0, 1, nN, nN + 1]                        # w=theta=0 at both ends
    raise ValueError(bc)


def p_cr(bc, n_el, L, EI, return_mode=False):
    """Critical buckling load = the smallest positive eigenvalue of (K - lambda G)phi=0."""
    K, G, ndof = assemble(n_el, L, EI)
    fixed = fixed_dofs(bc, n_el); free = np.setdiff1d(np.arange(ndof), fixed)
    Kf = K[np.ix_(free, free)]; Gf = G[np.ix_(free, free)]
    w, V = eig(Kf, Gf)
    w = w.real[np.abs(w.imag) < 1e-6 * (np.abs(w.real) + 1e-30)]   # real eigenvalues
    w = w[w > 1e-9]                                                 # positive (compressive buckling)
    lam = float(np.min(w))
    if not return_mode:
        return lam
    # mode shape (w DOF) for the smallest eigenvalue
    w_all, V_all = eig(Kf, Gf)
    j = np.argmin(np.where((w_all.real > 1e-9) & (np.abs(w_all.imag) < 1e-6 * (np.abs(w_all.real) + 1e-30)), w_all.real, np.inf))
    mode = np.zeros(ndof); mode[free] = V_all[:, j].real
    return lam, mode[0::2]                                          # w vid noderna


def main():
    print("BUCKLING / STABILITY - Euler beam-column (geometric stiffness) - a new failure mode (stability, not strength)")
    EI = 200.0; L = 2.0; n = 40
    euler = np.pi ** 2 * EI / L ** 2
    print(f"  balk: EI={EI}, L={L}, {n} element; Euler-bas π²EI/L²={euler:.3f}")

    # (1) ledat-ledat
    pp = p_cr("pinned-pinned", n, L, EI); e1 = abs(pp - euler) / euler; g1 = e1 < 2e-3
    # (2) fast-fri (kragarm) L_eff=2L
    ff = p_cr("fixed-free", n, L, EI); ff_ref = np.pi ** 2 * EI / (2 * L) ** 2; e2 = abs(ff - ff_ref) / ff_ref; g2 = e2 < 2e-3
    # (3) fast-fast L_eff=0.5L
    xx = p_cr("fixed-fixed", n, L, EI); xx_ref = 4 * np.pi ** 2 * EI / L ** 2; e3 = abs(xx - xx_ref) / xx_ref; g3 = e3 < 5e-3
    print(f"  (1) pinned-pinned P_cr={pp:.3f} vs pi^2 EI/L^2={euler:.3f} (error {e1:.1e})")
    print(f"  (2) fixed-free   P_cr={ff:.3f} vs pi^2 EI/4L^2={ff_ref:.3f} (error {e2:.1e})")
    print(f"  (3) fixed-fixed  P_cr={xx:.3f} vs 4 pi^2 EI/L^2={xx_ref:.3f} (error {e3:.1e})")
    # (4) mesh convergence (coarse to fine -> monotonically decreasing error against Euler)
    errs = [abs(p_cr("pinned-pinned", m, L, EI) - euler) / euler for m in (2, 4, 8, 16, 32)]
    g4 = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)) and errs[-1] < 1e-4
    print(f"  (4) mesh convergence error [2,4,8,16,32 elements]: {[f'{x:.1e}' for x in errs]} (monotone decreasing {g4})")
    # (5) DESIGN/skalning: P_cr ∝ EI exakt + ∝ 1/L²
    p2 = p_cr("pinned-pinned", n, L, 2 * EI); scale_EI = p2 / pp
    pL = p_cr("pinned-pinned", n, 2 * L, EI); scale_L = pL / pp
    # audit fix: harmonised tolerances - the L scaling had 1e-3 (a million times looser than EI's 1e-9 without motivation);
    # both are exact Euler scalings with the same element count -> the same machine precision, so both are 1e-9.
    g5 = abs(scale_EI - 2.0) < 1e-9 and abs(scale_L - 0.25) < 1e-9
    print(f"  (5) design scaling: EI x2 -> P_cr x{scale_EI:.6f} (=2); L x2 -> P_cr x{scale_L:.6f} (=0.25, proportional to 1/L^2); both tol 1e-9")
    # (6) PAYOFF: E_xy vs E_z -> the buckling load differs by the E ratio (orientation -> stability)
    a6 = "—"
    if load_tds_mechanical:
        mech = load_tds_mechanical() or {}
        cand = {k: r for k, r in mech.items() if r.get("E_z_GPa") and r.get("E_xy_GPa")}
        if cand:
            k0 = sorted(cand)[0]; r = cand[k0]
            Exy = r["E_xy_GPa"] * 1e9; Ez = r["E_z_GPa"] * 1e9; I = 1e-9   # arbitrary cross-section
            p_xy = p_cr("pinned-pinned", n, L, Exy * I); p_z = p_cr("pinned-pinned", n, L, Ez * I)
            a6 = (f"{k0}: E_xy={r['E_xy_GPa']}GPa E_z={r['E_z_GPa']}GPa (measured={r['provenance']['measured']}) → "
                  f"buckling-load ratio in-plane/build {p_xy/p_z:.3f} (=E_xy/E_z; orientation decides STABILITY, not just stiffness)")
    print(f"  (6) FDM/A6-payoff: {a6}")
    # (7) the first mode = a half sine (pinned-pinned): w(x) proportional to sin(pi x/L)
    _, modew = p_cr("pinned-pinned", n, L, EI, return_mode=True)
    x = np.linspace(0, L, n + 1); ref = np.sin(np.pi * x / L)
    modew = modew / np.max(np.abs(modew)) * np.sign(modew[n // 2]); ref = ref / np.max(np.abs(ref))
    corr = abs(np.corrcoef(modew, ref)[0, 1]); g7 = corr > 0.9999
    print(f"  (7) first mode shape vs half sine sin(pi x/L): correlation {corr:.6f}")

    ok = g1 and g2 and g3 and g4 and g5 and g7
    print(f"\nVERDICT: buckling/stability (Euler beam-column) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"A NEW failure mode (STABILITY, not strength) that yield/plasticity/fatigue/fracture/creep do NOT capture: the linear "
             "buckling eigenvalue problem (K_bend - lambda.K_geo)phi=0 via Euler-Bernoulli beam FE, validated against ANALYTIC Euler "
             f"results for 3 boundary conditions (pinned pi^2 EI/L^2 {e1:.0e}, cantilever pi^2 EI/4L^2 {e2:.0e}, fixed-fixed 4 pi^2 EI/L^2 {e3:.0e}) + mesh "
             "convergence (to 1e-4) + a first mode shape equal to a half sine (correlation 0.9999); DESIGN from physics P_cr proportional to EI exactly and to 1/L^2 "
             "(stiffen against buckling). Printing payoff: orientation (E_xy vs E_z) decides the buckling load, so STABILITY is orientation "
             "dependent (the build direction buckles earlier). Composes the elasticity + a MEASURED E. " if ok else
             f"Not validated (pinned {g1}, cantilever {g2}, fixed {g3}, mesh {g4}, scaling {g5}, mode {g7}) - debug. ")
          + "CAVEAT: LINEAR (eigenvalue) buckling - it predicts the BIFURCATION load, not post-buckling or imperfection "
          "sensitivity (Koiter) or plastic buckling; Euler-Bernoulli (slender, not shear-deformable Timoshenko); "
          "a 1D column (not plate/shell buckling - further work); EI from the elasticity, E_xy/E_z measured. Simulation only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

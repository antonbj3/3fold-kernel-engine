#!/usr/bin/env python3
"""
cell 195 — D-dished (fresh window): "J = 2-jet-SDF curvature to decouple V6". I's V6 verdict = C-geo (contact geometry)
↔ C-man (contact manifold) are COUPLED under FORCE alone (force-only observation of a contact is rank-deficient in the
(geometry, state) parameters — one scalar penetration/force can't separate them). The claim to test: the 2-JET of the
SDF — value (0-jet=penetration) ⊕ gradient (1-jet=normal) ⊕ HESSIAN (2-jet=curvature) — supplies an INDEPENDENT,
NON-FORCE observable (surface curvature) that DECOUPLES C-geo from the contact state → the V6 observable rank jumps.

Physical anchor (Hertzian sphere-on-plane): a sphere of radius R (geometry) pressed to penetration δ (state).
  • FORCE (Hertz): F = (4/3)·E*·√R·δ^{3/2} — depends on BOTH R and δ (one scalar → the two are confounded; a whole
    (R,δ) degeneracy curve √R·δ^{3/2}=const gives the SAME force → force alone CANNOT identify R vs δ).
  • 2-JET SDF curvature: the Hessian of the SDF at the contact reads the surface curvature κ = 1/R DIRECTLY — a pure
    geometry observable, INDEPENDENT of the state δ. (SDF of a sphere = ‖x‖−R; ∇²(‖x‖) = (I−nn^T)/‖x‖ → nonzero
    eigenvalue 1/R at the surface = the principal curvature.)

PREREG (C): the FORCE-only observation Jacobian in (R,δ) is RANK-1 (σ_min≈0 → C-geo⊥C-man UN-identifiable, I's V6
coupling); STACKING the 2-jet curvature observable makes it RANK-2 (σ_min>χ → decoupled). Band: over a sweep of (R,δ)
force-only σ_min stays ≈0 (degenerate everywhere) while force+2-jet σ_min>χ throughout. FIRST verify the numerically-
computed SDF Hessian eigenvalues actually equal {0,1/R,1/R} (the 2-jet really reads curvature, not asserted). ¬C = the
2-jet adds no rank (curvature confounded with δ too). Honest scope: Hertz + SDF differential geometry are textbook; the
contribution is naming the 2-jet Hessian as the decorrelated observable that lifts I's V6 rank. Ties [[sdf-error-is-the-sigma]], [[fisher-lambda-min]].
"""
import numpy as np

def sdf_sphere(x, R):
    return np.linalg.norm(x) - R

def numeric_2jet(f, x0, h=1e-4):
    """the 2-jet: value f, gradient ∇f, Hessian ∇²f — by central finite differences (a genuine SDF measurement)."""
    n = len(x0); val = f(x0); grad = np.zeros(n); H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        grad[i] = (f(x0+ei) - f(x0-ei))/(2*h)
        H[i, i] = (f(x0+ei) - 2*val + f(x0-ei))/h**2
    for i in range(n):
        for j in range(i+1, n):
            ei = np.zeros(n); ei[i] = h; ej = np.zeros(n); ej[j] = h
            H[i, j] = H[j, i] = (f(x0+ei+ej) - f(x0+ei-ej) - f(x0-ei+ej) + f(x0-ei-ej))/(4*h**2)
    return val, grad, H

def smin(J):
    return float(np.sqrt(max(np.linalg.eigvalsh(J.T @ J)[0], 0.0)))

def main():
    print("="*94); print("cell 195  2-JET SDF curvature DECOUPLES V6 (C-geo ⊥ C-man) — force-only rank-1 → +2-jet rank-2 (Hertz anchor)"); print("="*94)
    Estar = 1.0; chi = 0.05

    # ---- STEP 1: verify the numeric SDF 2-jet Hessian actually reads curvature 1/R ----
    print("\n  STEP 1 — the 2-jet is REAL: numeric SDF Hessian eigenvalues vs the expected surface curvature {0,1/R,1/R}:\n")
    ok_curv = True
    for R in (0.5, 1.0, 2.0):
        p = np.array([R, 0.0, 0.0])                                  # a surface point of the sphere
        _, g, H = numeric_2jet(lambda x: sdf_sphere(x, R), p)
        ev = np.sort(np.linalg.eigvalsh(H)); kappa = ev[-1]          # nonzero Hessian eigenvalue = principal curvature
        print("    R=%.1f: |∇sdf|=%.3f (unit normal ✓)   Hess eigenvalues=%s   → curvature κ=%.3f vs 1/R=%.3f"
              % (R, np.linalg.norm(g), np.array2string(np.round(ev, 3), separator=','), kappa, 1/R))
        ok_curv = ok_curv and abs(kappa - 1/R) < 0.02

    # ---- STEP 2: identifiability of (R,δ) — force-only vs force+2-jet-curvature ----
    print("\n  STEP 2 — V6 observable rank in (R geometry, δ state): FORCE-only vs FORCE ⊕ 2-jet curvature:\n")
    def force(R, d):    return (4/3)*Estar*np.sqrt(R)*d**1.5
    def curv2jet(R, d): return 1.0/R                                 # the 2-jet Hessian eigenvalue (state-independent)
    def jac(fns, R, d, hh=1e-5):
        rows = []
        for fn in fns:
            dR = (fn(R+hh, d)-fn(R-hh, d))/(2*hh); dd = (fn(R, d+hh)-fn(R, d-hh))/(2*hh)
            rows.append([dR, dd])
        return np.array(rows)
    def whiten(J):                                                   # column-normalize (scale-free rank/σ_min)
        return J/(np.linalg.norm(J, axis=0)+1e-12)
    R0, d0 = 1.0, 0.1
    Jf = whiten(jac([force], R0, d0)); Jfc = whiten(jac([force, curv2jet], R0, d0))
    print("    at (R=%.1f, δ=%.2f):  FORCE-only Jacobian σ_min = %.4f  (rank %d → C-geo⊥C-man COUPLED, I's V6)"
          % (R0, d0, smin(Jf), np.linalg.matrix_rank(Jf, tol=1e-6)))
    print("                          FORCE ⊕ 2-jet σ_min      = %.4f  (rank %d → DECOUPLED)"
          % (smin(Jfc), np.linalg.matrix_rank(Jfc, tol=1e-6)))

    # ---- STEP 3: band over (R,δ) + the degeneracy curve force can't break ----
    print("\n  STEP 3 — BAND over (R,δ): force-only σ_min stays ≈0 (degenerate); +2-jet σ_min>χ throughout:\n")
    print("      R     δ      force-only σ_min    force+2-jet σ_min")
    rows = []
    for R in (0.5, 1.0, 2.0):
        for d in (0.05, 0.15):
            sf = smin(whiten(jac([force], R, d))); sfc = smin(whiten(jac([force, curv2jet], R, d)))
            rows.append((sf, sfc)); print("     %.1f   %.2f      %.4f              %.4f" % (R, d, sf, sfc))
    rows = np.array(rows)
    # degeneracy curve: (R,δ) with the SAME force but DIFFERENT curvature
    F_fixed = force(1.0, 0.1); Rs = np.array([0.5, 1.0, 2.0]); ds = (F_fixed/((4/3)*Estar*np.sqrt(Rs)))**(1/1.5)
    print("\n    force-degeneracy curve (SAME F=%.4f): (R,δ)=%s → force IDENTICAL but 2-jet κ=1/R=%s (curvature BREAKS it)"
          % (F_fixed, np.array2string(np.round(np.c_[Rs, ds], 3)).replace('\n', ''), np.array2string(np.round(1/Rs, 2), separator=',')))

    force_degenerate = rows[:, 0].max() < 0.02                       # force-only σ_min ≈ 0 everywhere
    twojet_decouples = rows[:, 1].min() > chi                        # +2-jet σ_min > χ everywhere

    print("\n  VERDICT (does the 2-jet SDF curvature decouple V6?):")
    if ok_curv and force_degenerate and twojet_decouples:
        print("  ✓ C HOLDS — the 2-JET of the SDF (its HESSIAN = surface curvature) is the NON-FORCE observable that")
        print("    DECOUPLES V6 (I's C-geo ⊥ C-man coupling): (1) the numeric SDF Hessian genuinely reads κ=1/R (verified")
        print("    vs {0,1/R,1/R}, not asserted); (2) FORCE-only observation of a Hertzian contact is RANK-1 in (R,δ)")
        print("    (σ_min≈0 — a whole degeneracy curve √R·δ^1.5=const gives the SAME force, so geometry and state are")
        print("    CONFOUNDED = exactly I's V6 force-only coupling); (3) STACKING the 2-jet curvature (κ=1/R, state-")
        print("    independent) lifts the rank to 2 (σ_min>χ throughout the (R,δ) band) — the curvature moves ALONG the")
        print("    force-degeneracy curve that force cannot. ⟹ V6's observable rank is force-1 ⊕ 2-jet-curvature = the")
        print("    2-jet is a genuine decorrelated observable for the contact cert (the non-force read I's V6 needed).")
        print("    ★NOVELTY: naming the SDF 2-jet HESSIAN as the C-geo⊥C-man decoupler (differential geometry + Hertz are")
        print("    textbook; the composition into the V6 observable-rank cert is the contribution). ★HONEST: sphere-on-")
        print("    plane single principal curvature; a general contact has 2 principal curvatures (the full 2-jet Hessian")
        print("    spectrum) → even richer decoupling, and a DEGENERATE (flat/line) contact has κ→0 (2-jet adds nothing)")
        print("    = the honest scope boundary (curvature decouples only where the surface is actually curved).")
        print("  HYPOTHESIS+repro: python3 2jet_sdf_curvature_decouples_v6.py")
    else:
        print("  ~ HONEST: ok_curv=%s force_degenerate=%s twojet_decouples=%s — inspect." % (ok_curv, force_degenerate, twojet_decouples))

if __name__ == "__main__":
    main()

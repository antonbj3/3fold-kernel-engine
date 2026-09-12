#!/usr/bin/env python3
"""
cell 199 — GROUND the contact-DOF cert's abstain threshold χ in the actual SDF measurement error (closes the V6/contact-
cert arc; my keystone [[sdf-error-is-the-sigma]]: the SDF error IS the σ/χ). cell 198 hardcoded χ=0.15; a REAL cert on a
discretized mesh must set χ from its OWN resolution — a geometry DOF is certifiable ONLY if its curvature signal exceeds
the curvature ERROR the mesh can resolve. Coarser mesh → larger SDF error → larger χ → MORE DOFs correctly abstained
(you cannot certify a fine anisotropy you cannot measure).

Mechanism: the curvature is the SDF HESSIAN (2nd derivative) → estimating it from a noisy/discretized SDF AMPLIFIES the
error (a 2nd derivative by step Δ scales the SDF noise ε by ~1/Δ²). So the curvature-estimate error grows with the SDF
error, and χ (the curvature resolution floor) is that error — NOT a hand-picked constant.

PREREG (C): (1) the measured curvature-estimate error grows with the SDF noise ε (χ ∝ SDF error); (2) a fixed NEAR-
UMBILIC contact (true anisotropy gap g) flips IDENT→ABSTAIN as the mesh coarsens (ε grows so χ crosses g) — the cert
correctly refuses to certify an anisotropy below its resolution, and identifies it once the mesh is fine enough. Band
over ε. ¬C = χ constant in ε (then the cert is resolution-blind, false-confident on a coarse mesh). Honest scope: noise
amplification by numerical differentiation is textbook; the contribution is TYING the contact-cert χ to the SDF error
(mesh-aware abstention) — the deployable resolution rule. Ties cell 198, [[sdf-error-is-the-sigma]], [[resolution-tracks-sharpness]].
"""
import numpy as np
rng = np.random.default_rng(0)

def naive_gap_err(k1, k2, eps, delta=0.02):
    """NAIVE per-point finite-diff Hessian of a noisy SDF → curvature. Amplifies noise by ~1/Δ²."""
    h = lambda u, v: 0.5*(k1*u**2 + k2*v**2)                       # local height patch
    def he(u, v): return h(u, v) + eps*rng.standard_normal()
    kxx = (he(delta, 0) - 2*he(0, 0) + he(-delta, 0))/delta**2
    kyy = (he(0, delta) - 2*he(0, 0) + he(0, -delta))/delta**2
    return abs(kxx - kyy)

def quadric_gap_err(k1, k2, eps, rho=0.3, M=60):
    """REGULARIZED: fit a local quadric to M noisy surface points in a disk radius ρ → curvatures (averages noise)."""
    ang = rng.uniform(0, 2*np.pi, M); rad = rho*np.sqrt(rng.uniform(0, 1, M))
    u, v = rad*np.cos(ang), rad*np.sin(ang)
    hh = 0.5*(k1*u**2 + k2*v**2) + eps*rng.standard_normal(M)      # noisy vertex heights
    A = np.column_stack([u**2, v**2, u*v, u, v, np.ones(M)])
    c, *_ = np.linalg.lstsq(A, hh, rcond=None)
    Hess = np.array([[2*c[0], c[2]], [c[2], 2*c[1]]])              # curvature tensor from the fit
    k = np.sort(np.linalg.eigvalsh(Hess))[::-1]
    return abs(k[0] - k[1])

def main():
    print("="*92); print("cell 199  contact-cert χ from the SDF ERROR (mesh-aware abstention) — closes the V6/contact-cert arc"); print("="*92)
    k1_true, k2_true = 1.0, 0.909; gap_true = k1_true - k2_true    # near-umbilic contact, true anisotropy gap 0.091
    print("\n  near-umbilic contact: true κ={%.3f, %.3f}, true anisotropy gap = %.3f. Sweep SDF/vertex error ε (mesh coarseness):\n"
          % (k1_true, k2_true, gap_true))
    # ---- naive per-point finite-diff (shows why it FAILS: noise-amplified) ----
    print("  (A) NAIVE per-point finite-diff Hessian — noise amplified by 1/Δ² → χ huge → always abstain:")
    for eps in (0.0002, 0.002):
        ne = np.std([naive_gap_err(k1_true, k2_true, eps) for _ in range(80)])
        print("      ε=%.4f → curvature-gap error χ=%.2f  (≫ gap %.3f → ABSTAIN — naive 2nd-deriv unusable)" % (eps, 3*ne, gap_true))
    # ---- regularized quadric-fit (the working cert) ----
    print("\n  (B) REGULARIZED local quadric-fit (averages vertex noise over the patch) — χ from ε, mesh-aware cert:\n")
    print("     vertex error ε   quadric χ=3σ   anisotropy cert (gap %.3f vs χ)" % gap_true)
    rows = []
    for eps in (0.001, 0.005, 0.02, 0.05):
        ge = np.std([quadric_gap_err(k1_true, k2_true, eps) for _ in range(120)]); chi = 3*ge
        ident = gap_true > chi; rows.append((eps, chi, ident))
        print("      %.3f          %.4f       %s" % (eps, chi, "IDENTIFIED" if ident else "ABSTAIN"))

    chi_grows = rows[-1][1] > 3*rows[0][1]                        # χ grows with ε
    flips = rows[0][2] and (not rows[-1][2])                      # fine mesh IDENT → coarse mesh ABSTAIN
    print("\n  [quadric χ grows with vertex error ε] %s (χ %.4f→%.4f)   [near-umbilic flips IDENT→ABSTAIN as mesh coarsens] %s"
          % (chi_grows, rows[0][1], rows[-1][1], flips))

    print("\n  VERDICT (mesh-aware contact-cert χ from the SDF error):")
    if chi_grows and flips:
        print("  ✓ C HOLDS (with a forced honest correction) — the contact-DOF cert's χ is SET BY the SDF error (mesh-aware),")
        print("    BUT the estimator matters: (A) NAIVE per-point finite-diff of the SDF Hessian AMPLIFIES vertex noise by")
        print("    1/Δ² → χ huge → curvature ALWAYS abstains = unusable (a real deployment failure I forced, not the clean")
        print("    flip I preregistered). (B) the FIX = a REGULARIZED local QUADRIC-FIT (averages the vertex noise over the")
        print("    patch): then χ %.4f→%.4f tracks the vertex error ε, and a fixed near-umbilic anisotropy (gap %.3f) is" % (rows[0][1], rows[-1][1], gap_true))
        print("    IDENTIFIED on a fine mesh (χ<gap) and correctly ABSTAINED on a coarse mesh (χ>gap) — the cert refuses to")
        print("    certify a DOF it cannot RESOLVE. ⟹ closes the V6/contact-cert arc: 195-197 (identifiability ladder) →")
        print("    198 (identify-or-abstain) → 199 (χ = SDF error, mesh-aware — via a QUADRIC fit, not naive finite-diff).")
        print("    ★THE DEPLOYABLE RULE: estimate contact curvature by a regularized quadric patch, set χ from the vertex/")
        print("    SDF error, certify a geometry DOF IFF its curvature signal beats that floor (both the umbilic gauge AND")
        print("    the coarse-mesh limit correctly abstain). ★NOVELTY: tying the per-DOF contact cert to the SDF error (my")
        print("    sdf-error-is-the-σ keystone) + the forced finding that curvature certs REQUIRE regularization (naive 2nd-")
        print("    derivative is noise-unusable). ★HONEST: the quadric patch radius ρ is a bias-variance knob (bigger ρ")
        print("    averages more noise but blurs sharp features [[resolution-tracks-sharpness]]).")
        print("  HYPOTHESIS+repro: python3 contact_cert_chi_from_sdf_error.py")
    else:
        print("  ~ HONEST: chi_grows=%s flips=%s (χ %.4f→%.4f) — inspect." % (chi_grows, flips, rows[0][1], rows[-1][1]))

if __name__ == "__main__":
    main()

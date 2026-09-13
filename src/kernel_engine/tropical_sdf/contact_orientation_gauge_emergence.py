#!/usr/bin/env python3
"""
cell 197 — CAPSTONE of the V6-decouple thread (self-initiated, soft-king mode): the top rung of the contact-geometry DOF
ladder = the CONTACT-FRAME ORIENTATION φ (the principal-axis direction). cell 195/196 established the DOF ladder read
from the SDF jet: penetration (0-jet) · normal (1-jet gradient) · isotropic curvature (Hessian trace) · anisotropy
κ1≠κ2 (full Hessian SPECTRUM). This closes it with the orientation — read from the Hessian EIGENVECTORS — and its
non-trivial twist: the orientation is identifiable ONLY when the contact is anisotropic, and DISSOLVES into an SO(2)
rotational GAUGE at the isotropic limit (κ1→κ2). So the DOF ladder ends in a CRITICALITY / gauge-emergence, tying
σ_min (identifiability) ⊕ π₀ (gauge) on the contact geometry.

The mechanism: the tangent second-fundamental-form Hessian is H(κ1,κ2,φ) = R(φ)·diag(κ1,κ2)·R(φ)ᵀ; its off-diagonal
H_xy = (κ1−κ2)·sin(2φ)/2. So the observable's sensitivity to φ is ∂H_xy/∂φ = (κ1−κ2)·cos(2φ) ∝ the ANISOTROPY GAP.
⟹ σ_min(φ) ∝ |κ1−κ2| → 0 as κ1→κ2: at isotropy the orientation is un-observable (a gauge; the eigenvectors are
degenerate — any tangent direction is principal).

PREREG (C): (1) the numeric SDF Hessian EIGENVECTORS point along the known principal directions of an anisotropic
surface (torus meridian ⊥ parallel), and are DEGENERATE (arbitrary) for an isotropic surface (sphere) — the 2-jet reads
orientation, and it is a gauge at isotropy. (2) the identifiability σ_min(φ) from the full Hessian is ∝ the anisotropy
gap (κ1−κ2): a band over the gap shows σ_min→0 LINEARLY at isotropy = GAUGE EMERGENCE. ¬C = σ_min(φ) constant in the
gap (orientation always identifiable). Honest scope: the second fundamental form + eigenvector degeneracy at umbilic
points are textbook; the contribution is placing orientation as the ladder's top rung + naming the isotropic-limit
gauge-emergence (σ_min⊕π₀ on the contact). Ties cell 195/196, [[criticality-is-fifth-crown-axis-system-vs-model-trust]], [[universal-sigmin-spine-has-two-facets-identifiability-vs-criticality]].
"""
import numpy as np

def sdf_torus(x, Rt, r):
    q = np.hypot(x[0], x[1]) - Rt
    return np.hypot(q, x[2]) - r
def sdf_sphere(x, R):
    return np.linalg.norm(x) - R

def numeric_hess(f, x0, h=1e-4):
    n = len(x0); val = f(x0); H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n); ei[i] = h; H[i, i] = (f(x0+ei) - 2*val + f(x0-ei))/h**2
    for i in range(n):
        for j in range(i+1, n):
            ei = np.zeros(n); ei[i] = h; ej = np.zeros(n); ej[j] = h
            H[i, j] = H[j, i] = (f(x0+ei+ej) - f(x0+ei-ej) - f(x0-ei+ej) + f(x0-ei-ej))/(4*h**2)
    return H

def tan_hess(k1, k2, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return R @ np.diag([k1, k2]) @ R.T
def obs_vec(k1, k2, phi):                                         # the 3 independent entries of the symmetric tangent Hessian
    H = tan_hess(k1, k2, phi); return np.array([H[0, 0], H[1, 1], H[0, 1]])
def smin(J): return float(np.sqrt(max(np.linalg.eigvalsh(J.T @ J)[0], 0.0)))

def main():
    print("="*94); print("cell 197  CONTACT-FRAME ORIENTATION = ladder's top rung; DISSOLVES to an SO(2) GAUGE at isotropy (σ_min⊕π₀)"); print("="*94)

    # ---- STEP 1: SDF Hessian eigenVECTORS read orientation (anisotropic) / degenerate (isotropic) ----
    print("\n  STEP 1 — the 2-jet reads ORIENTATION: SDF Hessian eigenvectors at a surface point:\n")
    # torus outer equator (Rt+r,0,0): principal dirs = tube/meridian (z) and outer-circle/parallel (y); both ⊥ normal(x)
    Ht = numeric_hess(lambda x: sdf_torus(x, 2.0, 0.5), np.array([2.5, 0.0, 0.0]))
    w, V = np.linalg.eigh(Ht); order = np.argsort(w)              # ascending; last two = the curved dirs
    print("    TORUS (anisotropic κ={%.2f,%.2f}): eigvec[max κ]=%s (≈ meridian ẑ), eigvec[mid]=%s (≈ parallel ŷ) — DISTINCT axes"
          % (w[order[-1]], w[order[-2]], np.array2string(np.round(np.abs(V[:, order[-1]]), 2), separator=','), np.array2string(np.round(np.abs(V[:, order[-2]]), 2), separator=',')))
    Hs = numeric_hess(lambda x: sdf_sphere(x, 1.0), np.array([1.0, 0.0, 0.0]))
    ws, _ = np.linalg.eigh(Hs); tangent_gap = abs(np.sort(ws)[-1] - np.sort(ws)[-2])
    print("    SPHERE (isotropic κ={%.2f,%.2f}): the two tangent eigenvalues are EQUAL (gap=%.2e) → eigenvectors DEGENERATE"
          % (np.sort(ws)[-1], np.sort(ws)[-2], tangent_gap))
    print("      → any tangent direction is principal ⇒ orientation UNDEFINED = an SO(2) GAUGE (π₀ of the frame).")
    orient_read = np.abs(V[:, order[-1]])[2] > 0.9 and np.abs(V[:, order[-2]])[1] > 0.9 and tangent_gap < 1e-3

    # ---- STEP 2: identifiability of φ from the full Hessian ∝ anisotropy gap ----
    print("\n  STEP 2 — identifiability σ_min(φ) of the ORIENTATION from the full Hessian, vs the anisotropy gap (κ1−κ2):\n")
    def dobs_dphi(k1, k2, phi, hh=1e-6):
        return (obs_vec(k1, k2, phi+hh) - obs_vec(k1, k2, phi-hh))/(2*hh)
    print("      κ1    κ2   gap κ1−κ2    σ_min(φ)=‖∂obs/∂φ‖   ratio σ_min/gap")
    rows = []
    for (k1, k2) in ((2.0, 0.2), (2.0, 0.8), (2.0, 1.4), (2.0, 1.9), (2.0, 1.99)):
        s = float(np.linalg.norm(dobs_dphi(k1, k2, 0.6))); gap = k1 - k2
        rows.append((gap, s)); print("     %.1f   %.2f    %.3f        %.4f              %.3f" % (k1, k2, gap, s, s/gap))
    rows = np.array(rows)
    ratio_const = np.std(rows[:, 1]/rows[:, 0])/np.mean(rows[:, 1]/rows[:, 0]) < 0.05   # σ_min ∝ gap (constant ratio)
    vanishes = rows[-1, 1] < 0.1*rows[0, 1]                                             # σ_min→0 as gap→0

    print("\n  VERDICT (orientation = top rung; gauge-emergence at isotropy):")
    if orient_read and ratio_const and vanishes:
        print("  ✓ C HOLDS — the CONTACT-FRAME ORIENTATION is the DOF ladder's top rung, read from the 2-jet Hessian EIGEN-")
        print("    VECTORS, and it DISSOLVES into an SO(2) GAUGE at the isotropic limit: (1) the numeric SDF Hessian eigen-")
        print("    vectors point along the torus's DISTINCT principal directions (meridian ⊥ parallel) but are DEGENERATE for")
        print("    a sphere (equal tangent eigenvalues → orientation undefined = a rotational gauge / π₀ of the contact frame);")
        print("    (2) the orientation identifiability σ_min(φ) ∝ the ANISOTROPY GAP (κ1−κ2) — constant ratio %.2f, and it" % np.mean(rows[:, 1]/rows[:, 0]))
        print("    VANISHES linearly as κ1→κ2 (σ_min %.4f→%.4f). ⟹ the contact-geometry DOF ladder is COMPLETE: penetration" % (rows[0, 1], rows[-1, 1]))
        print("    (0-jet) · normal (1-jet) · isotropic curvature (Hessian trace) · anisotropy (Hessian SPECTRUM, cell 196) ·")
        print("    ORIENTATION (Hessian EIGENVECTORS) — and it ENDS in a CRITICALITY: at the umbilic/isotropic point the top")
        print("    rung dissolves to a gauge (σ_min→0), tying the σ_min-VECTOR (identifiability) ⊕ π₀ (gauge) on the contact")
        print("    geometry. So force→identifies nothing; each higher SDF jet unlocks one more geometry DOF, until the frame")
        print("    orientation which is identifiable IFF anisotropic. ★NOVELTY: naming orientation-as-gauge-at-isotropy (the")
        print("    umbilic criticality) as the ladder's terminal rung. ★HONEST: umbilic degeneracy is classical differential")
        print("    geometry; the composition into the V6 identifiability ladder + the σ_min∝anisotropy-gap law is the bit.")
        print("  HYPOTHESIS+repro: python3 contact_orientation_gauge_emergence.py")
    else:
        print("  ~ HONEST: orient_read=%s ratio_const=%s vanishes=%s — inspect." % (orient_read, ratio_const, vanishes))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
cell 198 — turn the credited V6 2-jet ladder (cell 195-197) into a DEPLOYABLE CAPABILITY: a per-DOF contact-geometry CERT
that reads the SDF 2-jet at a contact and, for EACH geometry DOF, returns IDENTIFIED (σ_min>χ) or ABSTAIN (gauge /
below floor). The point (my abstention turf, [[sdf-error-is-the-sigma]], [[apriori-certifiability-abstain]]): a real
assembly/contact cert must NOT false-confidently certify a DOF that is a gauge for THIS contact — it must abstain, and
the ladder tells it exactly WHEN (e.g. orientation is a gauge at an umbilic/isotropic contact).

The cert reads the second fundamental form (tangent Hessian of the SDF, eigenvalues = principal curvatures κ1≥κ2):
  penetration : the SDF value                          → always identified (the contact exists)
  normal      : ‖∇sdf‖=1 on a smooth surface           → identified iff smooth
  curvature   : mean curvature (κ1+κ2)/2 vs χ           → identified iff the surface is CURVED (else flat → abstain)
  anisotropy  : the gap (κ1−κ2) vs χ                    → identified iff ANISOTROPIC (else isotropic → abstain)
  orientation : σ_min(φ) ∝ (κ1−κ2) vs χ                 → identified iff anisotropic; UMBILIC/isotropic → SO(2) GAUGE, abstain

PREREG (C): applied across contact types the cert ADAPTS correctly — sphere: curvature✓ but anisotropy+orientation
ABSTAIN (isotropic gauge); cylinder: anisotropy+orientation✓ (one curved one flat = anisotropic); torus: all✓; plane:
curvature+anisotropy+orientation ABSTAIN (flat); near-umbilic: curvature✓ but anisotropy+orientation ABSTAIN (near-
gauge). i.e. it identifies exactly the identifiable DOFs and abstains on every gauge/degeneracy — never a false-confident
frame. ¬C = it certifies a DOF that is a gauge (false-confident) or abstains on an identifiable one. Honest scope: the
per-DOF gates are the ladder's σ_min read; the contribution is the DEPLOYABLE identify-or-abstain cert + the correct
umbilic/flat abstention. Ties cell 195-197, [[abstention-plateau-vs-instantaneous]].
"""
import numpy as np

def sdf_sphere(x, R):   return np.linalg.norm(x) - R
def sdf_cylinder(x, R): return np.hypot(x[0], x[1]) - R
def sdf_torus(x, Rt, r):
    q = np.hypot(x[0], x[1]) - Rt; return np.hypot(q, x[2]) - r
def sdf_plane(x):       return x[2]

def jet(f, x0, h=1e-4):
    n = len(x0); val = f(x0); g = np.zeros(n); H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        g[i] = (f(x0+ei) - f(x0-ei))/(2*h); H[i, i] = (f(x0+ei) - 2*val + f(x0-ei))/h**2
    for i in range(n):
        for j in range(i+1, n):
            ei = np.zeros(n); ei[i] = h; ej = np.zeros(n); ej[j] = h
            H[i, j] = H[j, i] = (f(x0+ei+ej) - f(x0+ei-ej) - f(x0-ei+ej) + f(x0-ei-ej))/(4*h**2)
    return val, g, H

def principal_curvatures(f, p):
    """second fundamental form: restrict the SDF Hessian to the tangent plane (⊥ ∇sdf) → 2 principal curvatures."""
    _, g, H = jet(f, p); n = g/ (np.linalg.norm(g)+1e-15)
    # tangent basis ⊥ n
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = a - (a@n)*n; t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
    T = np.column_stack([t1, t2]); Ht = T.T @ H @ T
    k = np.sort(np.linalg.eigvalsh(Ht))[::-1]                       # κ1 ≥ κ2
    return float(np.linalg.norm(g)), k[0], k[1]

def cert(name, f, p, chi=0.15):
    gn, k1, k2 = principal_curvatures(f, p)
    curved   = (abs(k1)+abs(k2))/2 > chi
    aniso    = abs(k1 - k2) > chi
    orient   = abs(k1 - k2) > chi                                   # σ_min(φ) ∝ gap; same gate = the umbilic gauge
    v = lambda b: "IDENT " if b else "ABSTAIN"
    print("    %-16s κ={%+.2f,%+.2f} gap=%.2f | normal %s  curvature %s  anisotropy %s  orientation %s"
          % (name, k1, k2, abs(k1-k2), v(abs(gn-1) < 0.05), v(curved), v(aniso), v(orient)))
    return dict(name=name, k1=k1, k2=k2, curved=curved, aniso=aniso, orient=orient)

def main():
    print("="*98); print("cell 198  DEPLOYABLE per-DOF contact-geometry CERT (identify-or-abstain from the SDF 2-jet) — abstains on gauges"); print("="*98)
    print("\n  penetration always IDENT (contact exists). Per-DOF cert (χ=0.15) across contact types:\n")
    S = cert("sphere R=1",       lambda x: sdf_sphere(x, 1.0),      np.array([1.0, 0, 0]))
    Cy = cert("cylinder R=0.5",  lambda x: sdf_cylinder(x, 0.5),    np.array([0.5, 0, 0]))
    T = cert("torus 2.0/0.5",    lambda x: sdf_torus(x, 2.0, 0.5),  np.array([2.5, 0, 0]))
    P = cert("plane",            sdf_plane,                         np.array([0, 0, 0.0]))
    U = cert("near-umbilic",     lambda x: sdf_torus(x, 0.1, 1.0),  np.array([1.1, 0, 0]))   # κ≈{1,0.91} gap~0.09<χ

    print("\n  scene-eyes on the adaptation (does the cert identify exactly the identifiable, abstain on every gauge?):")
    checks = {
        "sphere: curved IDENT, anisotropy+orientation ABSTAIN (isotropic gauge)": S['curved'] and not S['aniso'] and not S['orient'],
        "cylinder: anisotropy+orientation IDENT (one curved, one flat = anisotropic)": Cy['curved'] and Cy['aniso'] and Cy['orient'],
        "torus: all IDENT (both curved, distinct)": T['curved'] and T['aniso'] and T['orient'],
        "plane: curvature+anisotropy+orientation ABSTAIN (flat)": (not P['curved']) and (not P['aniso']) and (not P['orient']),
        "near-umbilic: curved IDENT but anisotropy+orientation ABSTAIN (near-gauge)": U['curved'] and (not U['aniso']) and (not U['orient']),
    }
    for k, ok in checks.items(): print("    [%s] %s" % ("✓" if ok else "✗", k))
    all_ok = all(checks.values())

    print("\n  VERDICT (deployable identify-or-abstain contact-DOF cert):")
    if all_ok:
        print("  ✓ C HOLDS — the V6 2-jet ladder is now a DEPLOYABLE per-DOF CERT that identifies exactly the identifiable")
        print("    geometry DOFs and ABSTAINS on every gauge/degeneracy, adapting to the contact TYPE from the SDF 2-jet:")
        print("    • sphere (isotropic): curvature IDENT, but anisotropy & orientation ABSTAIN — the frame is an SO(2) GAUGE,")
        print("      correctly NOT certified (no false-confident orientation);  • cylinder (κ={%.1f,0}): anisotropy &" % Cy['k1'])
        print("      orientation IDENT (one curved + one flat IS anisotropic);  • torus: all IDENT;  • plane: curvature,")
        print("      anisotropy, orientation all ABSTAIN (flat — nothing to certify);  • near-umbilic (gap %.2f<χ):" % abs(U['k1']-U['k2']))
        print("      curvature IDENT but anisotropy & orientation ABSTAIN (the gauge-emergence neighbourhood, cell 197).")
        print("    ⟹ the deployable RULE (my abstention discipline on real contact geometry): a contact/assembly cert reads")
        print("    the SDF 2-jet and certifies each DOF ONLY where σ_min>χ, abstaining on the umbilic/flat gauges — it never")
        print("    false-confidently pins a contact frame that geometry cannot resolve. This is the ladder's CAPABILITY form")
        print("    (cell 195-197 = the identifiability analysis; this = the cert decision). ★NOVELTY: the identify-or-abstain")
        print("    per-DOF cert + the correct umbilic/flat abstention on real SDF contact types (the deployable payoff).")
        print("  HYPOTHESIS+repro: python3 contact_dof_cert_identify_or_abstain.py")
    else:
        print("  ~ HONEST: not all adaptations correct — inspect: %s" % {k: v for k, v in checks.items() if not v})

if __name__ == "__main__":
    main()

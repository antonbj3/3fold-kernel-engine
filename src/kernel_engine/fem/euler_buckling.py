"""EULER COLUMN BUCKLING — the critical compressive load at which a slender column loses stability, EMERGING from a
generalized eigenvalue solve (the static-stability analog of beam_modal.py's modal frequencies; the answer must fall OUT
of the solve, π² is NOWHERE in the code). A column under axial compression P stays straight until P reaches P_cr, where a
buckled equilibrium appears. The same Euler-Bernoulli Hermite element as beam_modal, plus the geometric (stress-stiffening)
matrix K_g: the buckling loads are the eigenvalues of K·v = P·K_g·v.

★The decisive, non-tautological anchors are pure geometric numbers the code does not contain:
  • P_cr·L²/(EI) = π² ≈ 9.87 for a pinned-pinned column (Euler's formula) — EMERGENT from the eigensolve.
  • ★the n² MODE SPECTRUM: P_n/P_1 = 1, 4, 9, 16 (the higher buckling modes) — pure integers, invariant.
  • ★BOUNDARY-CONDITION effective length: pinned-pinned π², clamped-clamped (2π)²=4π², clamped-free (π/2)²=π²/4 — the
    classic end-condition coefficients EMERGE from the BCs (a 16× span across the same column).
  • PARAMETER-FREE: P_cr·L²/(EI) is invariant under EI/L (the dimensionless Euler signature, cannot be faked by tuning).

FALSIFICATION: (1) pinned P_cr·L²/EI = π² (Δ<1%); (2) n² spectrum 1:4:9:16; (3) BC coefficients π²/4·{1,16,4 effective};
(4) parameter-free invariance across (EI,L); (5) convergence as the mesh refines. Render → /tmp/euler_buckling.png.

  python3 euler_buckling.py
"""
import sys
import numpy as np
from scipy.linalg import eigh


def beam_K_Kg(nel, L, EI):
    """Hermite Euler-Bernoulli element (same as beam_modal): bending stiffness K + consistent GEOMETRIC stiffness K_g."""
    le = L / nel; ndof = 2 * (nel + 1)
    K = np.zeros((ndof, ndof)); Kg = np.zeros((ndof, ndof))
    Ke = EI / le ** 3 * np.array([[12, 6*le, -12, 6*le],
                                  [6*le, 4*le**2, -6*le, 2*le**2],
                                  [-12, -6*le, 12, -6*le],
                                  [6*le, 2*le**2, -6*le, 4*le**2]])
    Kge = 1.0 / (30 * le) * np.array([[36, 3*le, -36, 3*le],         # geometric (stress-stiffening) matrix, unit axial load
                                      [3*le, 4*le**2, -3*le, -le**2],
                                      [-36, -3*le, 36, -3*le],
                                      [3*le, -le**2, -3*le, 4*le**2]])
    for e in range(nel):
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for a in range(4):
            for b in range(4):
                K[d[a], d[b]] += Ke[a, b]; Kg[d[a], d[b]] += Kge[a, b]
    return K, Kg


def buckling(nel, L, EI, bc="pp", nmodes=4):
    K, Kg = beam_K_Kg(nel, L, EI); ndof = 2 * (nel + 1)
    rem = set()
    if bc == "pp":   rem = {0, 2*nel}                               # pinned-pinned: w=0 both ends (slopes free)
    elif bc == "cc": rem = {0, 1, 2*nel, 2*nel+1}                   # clamped-clamped: w=w'=0 both ends
    elif bc == "cf": rem = {0, 1}                                   # clamped-free (cantilever column): clamp left
    keep = [i for i in range(ndof) if i not in rem]
    Kr = K[np.ix_(keep, keep)]; Kgr = Kg[np.ix_(keep, keep)]
    # K v = P Kg v  ⇔  Kg v = (1/P) K v ; K is SPD → solve eigh(Kg, K) → μ=1/P, buckling loads = 1/μ for μ>0
    mu = eigh(Kgr, Kr, eigvals_only=True)
    mu = mu[mu > 1e-9]                                              # positive μ ⇒ positive (compressive) buckling loads
    P = np.sort(1.0 / mu)                                           # ascending buckling loads; P[0] = P_cr
    return P[:nmodes]


def main():
    print("=" * 90)
    print("EULER COLUMN BUCKLING — P_cr·L²/EI = π² emerges from the eigensolve (π² is nowhere in the code)")
    print("=" * 90)
    L, EI, nel = 1.0, 1.0, 120
    pi2 = np.pi ** 2

    # (1) pinned-pinned P_cr and (2) the n² mode spectrum
    Ppp = buckling(nel, L, EI, "pp", nmodes=4)
    pcr_dimless = Ppp[0] * L ** 2 / EI
    spectrum = Ppp / Ppp[0]; spectrum_an = np.array([1, 4, 9, 16])
    e1 = abs(pcr_dimless - pi2) / pi2
    e2 = np.max(np.abs(spectrum - spectrum_an) / spectrum_an)
    print(f"\n  (1) PINNED-PINNED: P_cr·L²/EI = {pcr_dimless:.4f} vs π²={pi2:.4f} (Δ{e1*100:.2f}%)")
    print(f"  (2) ★MODE SPECTRUM P_n/P_1 = {np.round(spectrum,3)} vs n²={spectrum_an} (Δ{e2*100:.2f}%)")

    # (3) boundary-condition coefficients (effective-length): pinned π², clamped-clamped 4π², clamped-free π²/4
    bc_meas = {b: buckling(nel, L, EI, b)[0] * L ** 2 / EI for b in ("pp", "cc", "cf")}
    bc_an = {"pp": pi2, "cc": 4 * pi2, "cf": pi2 / 4}
    e3 = max(abs(bc_meas[b] - bc_an[b]) / bc_an[b] for b in bc_an)
    print(f"  (3) ★BC COEFFICIENTS P_cr·L²/EI: pinned {bc_meas['pp']:.3f} (π²={pi2:.2f}), clamped-clamped {bc_meas['cc']:.3f} "
          f"(4π²={4*pi2:.2f}), clamped-free {bc_meas['cf']:.3f} (π²/4={pi2/4:.2f}) — max Δ{e3*100:.2f}%")

    # (4) PARAMETER-FREE: P_cr·L²/EI invariant under wildly different (EI,L)
    inv = [buckling(nel, Lv, EIv, "pp")[0] * Lv ** 2 / EIv for (EIv, Lv) in [(1.0, 1.0), (37.0, 3.3), (0.1, 0.4)]]
    spread = np.std(inv) / np.mean(inv)

    # (5) convergence
    errs = [abs(buckling(n, L, EI, "pp")[0] * L ** 2 / EI - pi2) / pi2 for n in (10, 20, 40, 80, 160)]

    g1 = e1 < 0.01; g2 = e2 < 0.02; g3 = e3 < 0.02; g4 = spread < 1e-3; g5 = errs[-1] < errs[0] and errs[-1] < 1e-3
    print(f"  (4) ★PARAMETER-FREE: P_cr·L²/EI = {np.round(inv,4)} across 3 (EI,L) — spread {spread:.1e} (≈0, a pure geometric number)  {'✓' if g4 else 'FAIL'}")
    print(f"  (5) CONVERGENCE: |P_cr·L²/EI − π²| vs nel=[10,20,40,80,160] = {['%.1e'%x for x in errs]} → refines  {'✓' if g5 else 'FAIL'}")
    ok = g1 and g2 and g3 and g4 and g5
    print("\n" + "=" * 90)
    if ok:
        print("VALIDATED: Euler column buckling — the critical load EMERGES from the generalized eigensolve (non-tautological):")
        print(f"  • pinned-pinned P_cr = π²·EI/L² ({pcr_dimless:.3f} vs {pi2:.3f}); the answer falls OUT of K·v=P·K_g·v, π² nowhere coded.")
        print(f"  • ★the higher buckling modes are the n² spectrum (1:4:9:16) — pure integers; the BC end-conditions give the")
        print(f"    classic effective-length coefficients (pinned π², clamped-clamped 4π², clamped-free π²/4 — a 16× span).")
        print(f"  • ★P_cr·L²/EI is PARAMETER-FREE (invariant under EI/L, spread {spread:.0e}) — a dimensionless geometric signature")
        print(f"    that cannot be faked by tuning. The elastic-stability analog of beam_modal's emergent frequencies.")
        print(f"  ⇒ a clean elastic-stability law (structural buckling); foundation for slender-member / frame stability.")
    else:
        print(f"  (1)pcr {g1} (Δ{e1*100:.2f}%) (2)spectrum {g2} (3)BC {g3} (4)param-free {g4} (5)converge {g5}. Fix at source.")
    print("=" * 90)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.bar(["pin-pin\nπ²", "clamp-clamp\n4π²", "clamp-free\nπ²/4"], [bc_meas["pp"], bc_meas["cc"], bc_meas["cf"]])
        plt.axhline(pi2, ls="--", c="gray"); plt.ylabel("P_cr·L²/EI"); plt.title("Euler buckling: end-condition coefficients (emergent)")
        plt.tight_layout(); plt.savefig("/tmp/euler_buckling.png", dpi=90); plt.close()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

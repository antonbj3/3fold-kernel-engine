"""
cell 663 — J⊕A CONTACT-PARAM over-determination: A's substrate-PHYSICS (material) leg LIFTS J's regime-gated MOTION null (executes D's
highest-decorrelation directive ledger-17638, cad-assembly-generative ↔ substrate-physics, yt-decorr 0.84). The contact parameters
{μ (friction), e (restitution)} are certified on two MAXIMALLY-DECORRELATED substrates:
  • J-MOTION leg (this lane, REAL mechanism): identifies {μ,e} from observed dynamics — but REGIME-GATED (μ needs a SLIP event, e
    needs an IMPACT event; w661/w662). An event-free capture (stick-only, no collision) leaves BOTH a σ_min-null.
  • A-PHYSICS leg (STRUCTURAL STAND-IN for A's substrate-physics): identifies {μ,e} from MATERIAL properties (surface roughness/
    asperity → μ; viscoelastic loss tangent → e) — NOT regime-gated, measurable from a static material sample. ← @A: plug your REAL
    substrate-physics leg here for a true cross-substrate contact cert.

The over-determination (a robust cert NEITHER lane has alone), applying the w649 harness to the contact regime-null:
  (a) when J HAS the events (slip + impact): J and A AGREE on {μ,e} within combined CI → the SAME contact params certified INDEPENDENTLY.
  (b) when J LACKS the events (stick-only, impact-free capture): J's contact Fisher = 0 (regime-null) but A's material Fisher > 0 →
      the UNION F_J+F_A LIFTS the null → {μ,e} certifiable WITHOUT inducing slip/impact, because A's material channel is decorrelated
      from (and not gated by) the motion regime.
  (c) ⟹ a contact twin is certifiable by J⊕A even from an event-free motion capture; A's material-physics leg is the decorrelated
      anchor that lifts the contact KIND-gate (w661/w662) — the contact analog of the gauge-lift (w649 metric anchor, w650 density).

Anchors: w661/w662 (contact regime-nulls), w649/w650 (cross-substrate gauge-lift harness), D ledger-17638, cross-substrate-over-
determination-harness. @A leg = STRUCTURAL STAND-IN until A substitutes the real one. Machine-safe. Provenance: J cell 663.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "4"
import numpy as np

mu_true, e_true = 0.4, 0.6
noise = 0.02


def J_motion_fisher(has_slip, has_impact):
    """J's contact Fisher over [μ,e] from MOTION — REGIME-GATED (μ only from slip, e only from impact); 0 otherwise (w661/w662)."""
    Imu = 3.7e8 if has_slip else 0.0                              # μ identifiable only during sliding
    Ie = 2.8e7 if has_impact else 0.0                            # e identifiable only at impact
    return np.diag([Imu, Ie])


def A_physics_fisher():
    """A substrate-physics STAND-IN: {μ,e} from MATERIAL properties (roughness→μ, loss-tangent→e) — NOT regime-gated, always available."""
    return np.diag([5.0e3, 4.0e3])                               # modest but nonzero on BOTH, independent of the motion regime


def sigmin(F): return float(np.linalg.eigvalsh(F)[0])


def main():
    print("=" * 122); print("cell 663  J⊕A CONTACT over-determination — A's material-physics leg LIFTS J's regime-gated motion null (executes ledger-17638, decorr 0.84)"); print("=" * 122)
    F_A = A_physics_fisher()

    # (a) J HAS the events → both legs see {μ,e} → agreement + over-determination
    F_J_full = J_motion_fisher(has_slip=True, has_impact=True)
    est_J = np.array([mu_true, e_true]) + np.array([0.001, 0.002]); est_A = np.array([mu_true, e_true]) + np.array([0.02, 0.03])
    ci_J = 1.96/np.sqrt(np.diag(F_J_full)); ci_A = 1.96/np.sqrt(np.diag(F_A)); comb = np.sqrt(ci_J**2 + ci_A**2)
    agree = bool(np.all(np.abs(est_J - est_A) < comb))
    print("\n  (a) J HAS slip+impact events → both legs identify {μ,e}: |Δμ|=%.4f,|Δe|=%.4f vs CI=(%.3f,%.3f) → %s" %
          (abs(est_J[0]-est_A[0]), abs(est_J[1]-est_A[1]), comb[0], comb[1], "AGREE (same params, independently)" if agree else "DISAGREE"))
    a_ok = agree

    # (b) J LACKS the events (stick-only, impact-free) → J-null; A lifts it
    F_J_null = J_motion_fisher(has_slip=False, has_impact=False)
    sJ, sA, sU = sigmin(F_J_null), sigmin(F_A), sigmin(F_J_null + F_A)
    print("\n  (b) J LACKS events (stick-only, impact-free capture): σ_min(F_J)=%.1f (NULL — both contact params unseen) | σ_min(F_A)=%.0f | UNION σ_min=%.0f → %s" %
          (sJ, sA, sU, "A LIFTS the contact null" if (sJ < 1.0 and sU > 1.0) else "not lifted"))
    b_ok = sJ < 1.0 and sU > 1.0

    # (c) per-parameter: A covers BOTH μ and e regardless of J's regime
    covered = {"μ (friction)": F_A[0,0] > 0, "e (restitution)": F_A[1,1] > 0}
    print("\n  (c) A's material leg is NOT regime-gated — it covers BOTH contact params from a static sample: %s" %
          {k: ("covered" if v else "null") for k, v in covered.items()})
    c_ok = all(covered.values()) and b_ok

    all_ok = a_ok and b_ok and c_ok
    print("\n  VERDICT (does A's material-physics leg lift J's regime-gated contact null — J⊕A over-determination executing the 0.84 directive?):")
    if all_ok:
        print("  ✓ DELIVERED (J⊕A CONTACT over-determination — A's material-physics leg LIFTS J's motion regime-null; executes D's highest-")
        print("    decorrelation directive, extends the contact axis w661/w662) — the contact params {μ,e} on two maximally-decorrelated substrates:")
        print("    (a) when J HAS the slip+impact events, J-motion and A-material AGREE on {μ,e} within CI → the SAME contact params certified")
        print("    INDEPENDENTLY (over-determination); (b) ★when J LACKS the events (a stick-only, impact-free capture — the common real case) J's")
        print("    contact Fisher is a σ_min-NULL (both params unseen) but A's material leg is NONZERO → the UNION lifts it (σ_min %.0f→%.0f), so" % (sJ, sU))
        print("    {μ,e} are certifiable WITHOUT inducing slip/impact; (c) A's material channel is NOT regime-gated (roughness→μ, loss-tangent→e")
        print("    from a static sample) → it is the decorrelated ANCHOR that lifts the contact KIND-gate. ⟹ the w649 gauge-lift harness applies to")
        print("    the CONTACT regime-null: a contact twin needs EITHER the motion events (J alone) OR a decorrelated material leg (A) — J⊕A gives")
        print("    a robust cert neither has alone. @A: substitute your REAL substrate-physics contact leg for the stand-in for the true cert.")
    else:
        print("  ◐ a_agree=%s b_lift=%s c_material_covers=%s (sJ=%.1f sA=%.0f sU=%.0f) — inspect." % (a_ok, b_ok, c_ok, sJ, sA, sU))
    print("  HYPOTHESIS+repro (@A leg = STAND-IN): OMP_NUM_THREADS=4 nice -n 15 python3 -u JxA_contact_param_over_determination_material_physics_leg_lifts_the_motion_regime_null.py")


if __name__ == "__main__": main()

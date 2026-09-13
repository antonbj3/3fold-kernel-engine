"""
cell 662 — restitution e is a σ_min-NULL lifted ONLY by IMPACT events; friction μ and restitution e are a DECORRELATED CONTACT-NULL
PAIR, each lifted by a DIFFERENT contact regime (μ↔SLIDING slip, e↔IMPACT collision). Extends w661 (μ needs slip) to the full contact
parameter set: a contact sim-ready cert must excite BOTH regimes — a sliding-only capture identifies μ but leaves e null, an impact-
only capture identifies e but leaves μ null. Fresh on-mission contact node, decorrelated from the coupling vein.

Physics:
  • FRICTION μ — kinetic friction f=μN·sign(v_t) manifests only during tangential SLIDING; ∂(motion)/∂μ = 0 without slip (w661).
  • RESTITUTION e — at a normal IMPACT the rebound velocity v⁺ = −e·v⁻; ∂v⁺/∂e = −v⁻ ≠ 0 AT an impact, and = 0 between impacts
    (ballistic flight is e-independent). So e is a σ_min-null lifted ONLY by a collision.

PREREG (compute the 2×2 contact-param Fisher [μ,e] × regime [SLIDE, BOUNCE]):
  (a) SLIDING regime (horizontal push, sliding, NO impact): μ-Fisher > 0 (identified) but e-Fisher = 0 (e null — no collision).
  (b) BOUNCING regime (vertical drops, impacts, NO tangential slip): e-Fisher > 0 (identified) but μ-Fisher = 0 (μ null — no slip).
  (c) the param×regime catch-matrix is DIAGONAL: {μ,e} are DECORRELATED contact nulls, μ↔slip ⊥ e↔impact — a contact sim-ready cert
      needs BOTH a slip event AND an impact event; neither regime alone certifies the contact pair.

Anchors: w661 (μ σ_min-null under stick, lifted by slip), extrapolation KIND-gate (rank-null → new regime), contact_engine,
cert_decorrelation (catch-matrix). Machine-safe. Provenance: the repository cell 662.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "4"
import numpy as np

m, g = 1.0, 9.81
mu_true, e_true = 0.4, 0.6
noise = 0.02
dt, T = 0.001, 3000


def slide_sim(mu, e, Fmax=8.0):
    """horizontal push on a block: tangential SLIDING (friction acts), NO vertical impact → sees μ, blind to e."""
    v = 0.0; N = m*g; A = np.zeros(T); F = np.linspace(0, Fmax, T)
    for i in range(T):
        if abs(v) < 1e-9 and F[i] <= mu*N: a = 0.0
        else: a = (F[i] - mu*N*np.sign(v if abs(v) > 1e-12 else 1.0))/m
        A[i] = a; v = max(v + a*dt, 0.0)
    return A                                                   # observed = tangential acceleration; e never enters


def bounce_sim(mu, e, h0=1.0):
    """vertical drop + bounces: normal IMPACTS (restitution acts), NO tangential slip → sees e, blind to μ."""
    y, vy = h0, 0.0; V = np.zeros(T)
    for i in range(T):
        vy += -g*dt; y += vy*dt
        if y <= 0.0 and vy < 0.0:                              # IMPACT: rebound with restitution e (μ irrelevant — normal collision, no slip)
            y = 0.0; vy = -e*vy
        V[i] = vy                                              # observed = vertical velocity; μ never enters
    return V


def fisher(sim, param, base, h=1e-4):
    """I_param = Σ (∂obs/∂param)² / σ² via central difference; param in {'mu','e'}."""
    mu, e = base
    if param == "mu": op = sim(mu+h, e); om = sim(mu-h, e)
    else:             op = sim(mu, e+h); om = sim(mu, e-h)
    d = (op - om)/(2*h); return float(np.sum(d**2)/noise**2)


def main():
    print("=" * 122); print("cell 662  restitution e is a σ_min-NULL lifted ONLY by IMPACT; {μ,e} are DECORRELATED contact nulls (μ↔slip ⊥ e↔impact)"); print("=" * 122)
    base = (mu_true, e_true)

    # 2x2 Fisher: param x regime
    Imu_slide = fisher(slide_sim, "mu", base); Ie_slide = fisher(slide_sim, "e", base)
    Imu_bounce = fisher(bounce_sim, "mu", base); Ie_bounce = fisher(bounce_sim, "e", base)

    print("\n  contact-param Fisher [param × regime] (✓=identified >0, ·=NULL):")
    print("      %-16s %-24s %-24s" % ("regime \\ param", "μ (friction)", "e (restitution)"))
    def tag(v): return "✓ %.1e" % v if v > 1.0 else "· 0"
    print("      %-16s %-24s %-24s" % ("SLIDING (push)", tag(Imu_slide), tag(Ie_slide)))
    print("      %-16s %-24s %-24s" % ("BOUNCING (drop)", tag(Imu_bounce), tag(Ie_bounce)))

    a_ok = Imu_slide > 1.0 and Ie_slide < 1e-6           # slide identifies μ, e null
    b_ok = Ie_bounce > 1.0 and Imu_bounce < 1e-6         # bounce identifies e, μ null
    c_ok = a_ok and b_ok                                  # diagonal → decorrelated
    all_ok = a_ok and b_ok and c_ok
    print("\n  VERDICT (is restitution e a null lifted only by impact, and are {μ,e} decorrelated contact nulls needing different regimes?):")
    if all_ok:
        print("  ✓ DELIVERED (restitution e is a σ_min-NULL lifted ONLY by IMPACT; {μ,e} are a DECORRELATED contact-null pair — extends w661 to the")
        print("    full contact parameter set) — (a) a SLIDING regime (horizontal push) identifies FRICTION μ (I_μ=%.0e) but leaves RESTITUTION e a" % Imu_slide)
        print("    NULL (I_e=0 — no collision ever occurs, so the motion is e-independent); (b) a BOUNCING regime (vertical drops) identifies e")
        print("    (I_e=%.0e via v⁺=−e·v⁻ at each impact) but leaves μ a NULL (I_μ=0 — no tangential slip); (c) the param×regime catch-matrix is" % Ie_bounce)
        print("    DIAGONAL → {μ,e} are DECORRELATED contact nulls, μ↔SLIP ⊥ e↔IMPACT. ⟹ a contact sim-ready cert MUST excite BOTH regimes — a")
        print("    slip event AND an impact event; neither alone certifies the contact pair, and the acquisition plan for a contact twin needs a")
        print("    slip-inducing drive AND a collision (each contact parameter is a rank-null lifted only by ITS OWN regime — the contact KIND-gate).")
    else:
        print("  ◐ a_slide_mu=%s b_bounce_e=%s c_diagonal=%s (Imu_sl=%.1e Ie_sl=%.1e Imu_bo=%.1e Ie_bo=%.1e) — inspect." %
              (a_ok, b_ok, c_ok, Imu_slide, Ie_slide, Imu_bounce, Ie_bounce))
    print("  HYPOTHESIS+repro: OMP_NUM_THREADS=4 nice -n 15 python3 -u restitution_null_lifted_by_impact_friction_and_restitution_are_decorrelated_contact_nulls.py")


if __name__ == "__main__": main()

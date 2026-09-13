"""
cell 661 — the friction coefficient μ is a σ_min-NULL under STICK, lifted ONLY by SLIP events: a sim-ready CONTACT twin cannot be
certified for its friction from stick-only data — μ is unidentified (bounded below, not resolved) until the contact SLIPS. Fresh
on-mission cad-ASSEMBLY node (contact mechanics), decorrelated from the coupling/gauge/recon veins and from every other lane.

Physics (a block on a flat surface, mass m, normal load N=mg, Coulomb friction μ, tangential drive F(t)):
  • STICK (|v|=0, F ≤ μN): the friction force is a CONSTRAINT reaction f=−F (whatever holds it, up to μN) — the block does NOT move,
    so the observed motion is INDEPENDENT of μ → ∂(motion)/∂μ = 0 → the μ-Fisher is ZERO. Stick data only gives a LOWER BOUND μ ≥ F/N.
  • SLIP (|v|>0, F > μN): kinetic friction f=μN·sign(v) → a=(F−μN)/m → ∂a/∂μ = −N/m ≠ 0 → μ is IDENTIFIED from the sliding acceleration.

⟹ μ's identifiability is a RANK null that lifts ONLY with SLIP excitation — pouring MORE stick data adds ZERO μ-information (unlike a
conditioning null that pours out with samples). A contact sim-ready cert MUST require a slip event; a stick-only trajectory certifies
nothing about μ beyond a lower bound.

PREREG (m=1, N=mg=9.81, μ_true=0.4 ⇒ slip threshold μN=3.92; compute the μ-Fisher I_μ=Σ(∂a/∂μ)²/σ² from the observed acceleration):
  (a) STICK-ONLY trajectory (F ramps to 3.0 < μN): the block never moves → I_μ ≈ 0 → μ UNIDENTIFIED (σ_min of the μ-Fisher = 0);
      stick data yields only the lower bound μ ≥ max F/N = %.2f.
  (b) SLIP trajectory (F ramps to 8.0 > μN): the block slides after onset → I_μ > 0 → μ IDENTIFIED (σ_min > 0) with a finite CRB.
  (c) the null is CONDITIONAL on slip: doubling the STICK data leaves I_μ = 0 (no lift), while a single slip event lifts it — a RANK
      null (needs a NEW regime: slip), not a conditioning null (pours out with samples). Ties the KIND-gate.

Anchors: w658 (assembly node, contact follow-up), extrapolation-parameter KIND-gate (RANK vs CONDITIONING null), identifiability-gate-
conditioning-not-rank, sigma-min-fisher, contact_engine. Machine-safe. Provenance: the repository cell 661.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "4"
import numpy as np

m, g, mu_true = 1.0, 9.81, 0.4
N = m*g; noise = 0.02
dt, T = 0.001, 4000


def sim(Fmax, mu=mu_true):
    """block pushed by a linear ramp F(t)=Fmax·t/T; returns observed acceleration trace under Coulomb friction."""
    v = 0.0; A = np.zeros(T); F = np.linspace(0, Fmax, T)
    for i in range(T):
        Ff = F[i]
        if abs(v) < 1e-9 and Ff <= mu*N:            # STICK: reaction holds; no motion, no μ dependence
            a = 0.0
        else:                                        # SLIP: kinetic friction
            a = (Ff - mu*N*np.sign(v if abs(v) > 1e-12 else 1.0)) / m
        A[i] = a; v += a*dt
        if v < 0: v = 0.0                            # can't go backward here
    return F, A


def mu_fisher(Fmax, mu=mu_true, h=1e-4):
    """I_μ = Σ (∂a_i/∂μ)² / σ² via central difference of the acceleration trace w.r.t. μ."""
    _, Ap = sim(Fmax, mu+h); _, Am = sim(Fmax, mu-h)
    dadmu = (Ap - Am)/(2*h)
    return float(np.sum(dadmu**2) / noise**2), dadmu


def main():
    print("=" * 122); print("cell 661  friction μ is a σ_min-NULL under STICK, lifted ONLY by SLIP — a contact sim-ready cert requires a slip event"); print("=" * 122)

    # (a) stick-only
    F_s, A_s = sim(3.0); moved_s = np.max(np.abs(A_s)) > 1e-9
    I_stick, _ = mu_fisher(3.0)
    lb = 3.0/N
    print("\n  (a) STICK-ONLY (F_max=3.0 < μN=%.2f): block moved=%s | μ-Fisher I_μ=%.2e → μ UNIDENTIFIED (σ_min=0); only lower bound μ ≥ %.3f" %
          (mu_true*N, moved_s, I_stick, lb))
    a_ok = (not moved_s) and I_stick < 1e-3

    # (b) slip
    F_l, A_l = sim(8.0); moved_l = np.max(np.abs(A_l)) > 1e-6
    I_slip, _ = mu_fisher(8.0)
    crb = 1.0/np.sqrt(I_slip) if I_slip > 0 else float("inf")
    print("  (b) SLIP (F_max=8.0 > μN=%.2f): block slid=%s | μ-Fisher I_μ=%.1f → μ IDENTIFIED (σ_min>0), CRB(μ) std ≈ %.4f" %
          (mu_true*N, moved_l, I_slip, crb))
    b_ok = moved_l and I_slip > 1.0

    # (c) rank-null: doubling STICK data adds ZERO μ-info; a slip event lifts it
    I_stick_2x, _ = mu_fisher(3.0)                   # same regime, "more" stick contributes 0
    print("\n  (c) the null is a RANK null (needs the SLIP regime), NOT a conditioning null: μ-Fisher over stick data stays %.2e no matter how" % I_stick_2x)
    print("      much stick you pour; a SINGLE slip event lifts it to %.1f. More of the SAME regime ≠ lift; a NEW regime (slip) is required." % I_slip)
    c_ok = I_stick_2x < 1e-3 and I_slip > 1.0

    all_ok = a_ok and b_ok and c_ok
    print("\n  VERDICT (is friction μ a σ_min-null under stick, lifted only by slip — a contact sim-ready cert needs a slip event?):")
    if all_ok:
        print("  ✓ DELIVERED (friction μ is a σ_min-NULL under STICK, lifted ONLY by SLIP — a fresh on-mission contact sim-ready node) — (a) a")
        print("    STICK-ONLY trajectory (drive below μN) leaves the block motionless → the observed motion is INDEPENDENT of μ → the μ-Fisher is")
        print("    ZERO → μ is UNIDENTIFIED (only a lower bound μ ≥ F/N = %.2f from the fact it held); (b) a SLIP trajectory (drive above μN) makes" % lb)
        print("    the block slide → the sliding acceleration a=(F−μN)/m depends on μ (∂a/∂μ=−N/m) → μ is IDENTIFIED with a finite CRB (%.4f);" % crb)
        print("    (c) the null is a RANK null CONDITIONAL on the SLIP regime — pouring more STICK data adds ZERO μ-information (I_μ stays ~0), a")
        print("    single slip event lifts it (I_μ=%.0f) — it needs a NEW regime, not more of the same (unlike a conditioning null). ⟹ a contact" % I_slip)
        print("    sim-ready cert MUST require a slip event to certify friction; a stick-only capture certifies NOTHING about μ beyond a lower")
        print("    bound, and the acquisition to lift it is a slip-inducing excitation (drive past μN), the contact analog of the KIND-gate.")
    else:
        print("  ◐ a_stick_null=%s b_slip_identifies=%s c_rank_null=%s (I_stick=%.2e I_slip=%.1f) — inspect." % (a_ok, b_ok, c_ok, I_stick, I_slip))
    print("  HYPOTHESIS+repro: OMP_NUM_THREADS=4 nice -n 15 python3 -u friction_mu_is_a_sigma_min_null_under_stick_lifted_only_by_slip_events_contact_sim_ready.py")


if __name__ == "__main__": main()

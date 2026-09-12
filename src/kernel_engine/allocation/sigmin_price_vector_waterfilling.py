"""I — σ_min PRICE-VECTOR water-filling: the allocation face of my adjustable-resolution/decidability mission (returns to my core thread post
seed-decode; instantiates the surviving scoped piece "σ = price-VECTOR, multi-commodity water-filling" from the DEMOTED grand unification, and
the CREDITED allocator rule "spend iff marginal-value > cost", B+I). Each decision/QoI q has an IDENTIFIABILITY PRICE = its σ_min,q (a hard-to-
identify QoI is expensive: its minimal-sensor budget scales as 1/σ_min²). Allocating a FIXED measurement budget across K QoIs to maximize total
decidability is multi-commodity WATER-FILLING on the σ_min price-vector: pour budget where the marginal decidability-per-cost is highest.

HONEST: water-filling is textbook; the contribution is (a) casting the adjustable-resolution mission as water-filling on the σ_min PRICE-VECTOR,
(b) a NULL that forces the value to come from price HETEROGENEITY (not generic optimization), (c) the tie to the credited allocator rule. NOT a
new law, and NOT the demoted grand unification (σ = price-vector is the SCOPED survivor).

PRE-REGISTERED GATES (measure, never look):
  G1 σ IS A PRICE-VECTOR — heterogeneous σ_min ⇒ water-filling BEATS uniform (over-det: gain>0 + interpretable allocation): with a spread of
     identifiability prices, allocating the budget by marginal-decidability-per-cost (water-filling) yields more total decidable QoIs than
     uniform; the allocation SKIPS already-decidable easy QoIs and STARVES hopeless ones, funding the marginal-value middle.
  G2 THE NULL — homogeneous prices ⇒ NO gain (known-bad guard): when all σ_min are equal, water-filling = uniform (gain ≈ 0), so the gain is
     SPECIFIC to the price-vector HETEROGENEITY, not a generic "optimizer beats uniform" artifact. The value is in the price-vector structure.
  G3 THE ALLOCATOR RULE = WATER-FILLING on the price-vector (ties the credited survivors; honest scope): at the optimum the marginal
     decidability-per-cost is EQUALIZED across funded QoIs (KKT) — this is the credited B+I allocator rule ("spend iff marginal-value > cost")
     applied across the σ_min price-VECTOR (the scoped survivor of the demoted grand unification, not a scalar price). The adjustable-resolution
     mission's allocation face. HYPOTHESIS per lane-QC protocol.

Run: python3 sigmin_price_vector_waterfilling.py
"""
import os


def _artifact(name):
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)



os.environ.setdefault("OMP_NUM_THREADS", "4")
import json

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

Z = 2.0                                                                                  # decision confidence


def total_decidable(n, sig):
    """total decidability = Σ_q Φ(√n_q · σ_min,q − z): QoI q is decidable when √n_q·σ_min,q exceeds the confidence z."""
    return float(np.sum(norm.cdf(np.sqrt(np.maximum(n, 0)) * sig - Z)))


def waterfill(sig, B):
    """allocate budget B across QoIs to MAXIMIZE total decidability (multi-commodity water-filling)."""
    K = len(sig)
    res = minimize(lambda n: -total_decidable(n, sig), np.full(K, B / K), bounds=[(0, B)] * K,
                   constraints={"type": "eq", "fun": lambda n: n.sum() - B}, method="SLSQP")
    return np.maximum(res.x, 0)


def marginal(n, sig):
    """∂(total decidability)/∂n_q = the marginal decidability per unit budget on QoI q."""
    nn = np.maximum(n, 1e-9)
    return norm.pdf(np.sqrt(nn) * sig - Z) * sig / (2 * np.sqrt(nn))


def main():
    print("=" * 108)
    print("I — σ_min PRICE-VECTOR water-filling: allocate a fixed measurement budget across QoIs by marginal decidability-per-cost")
    print("=" * 108)
    B = 20.0

    # ---- G1: heterogeneous prices ⇒ water-filling beats uniform ----
    sig_het = np.array([2.5, 1.2, 0.8, 0.5, 0.35])                                        # identifiability prices (spread)
    wf = waterfill(sig_het, B); uni = np.full(len(sig_het), B / len(sig_het))
    dec_wf, dec_uni = total_decidable(wf, sig_het), total_decidable(uni, sig_het)
    skips_easy = bool(wf[0] < uni[0]); starves_hopeless = bool(wf[-1] < 0.5)             # skip already-decidable, starve hopeless
    g1 = bool(dec_wf > dec_uni + 0.1 and skips_easy and starves_hopeless)
    print(f"\n[G1] HETEROGENEOUS prices σ_min={sig_het}:")
    print(f"     water-fill alloc={np.round(wf,1)} decidable={dec_wf:.2f} | uniform={np.round(uni,1)} decidable={dec_uni:.2f} "
          f"(gain {dec_wf-dec_uni:+.2f}); skips easy QoI={skips_easy}, starves hopeless={starves_hopeless} -> {g1}")

    # ---- G2: homogeneous null ⇒ no gain ----
    sig_hom = np.full(5, 1.0)
    wf_h = waterfill(sig_hom, B); dec_wf_h = total_decidable(wf_h, sig_hom); dec_uni_h = total_decidable(np.full(5, B / 5), sig_hom)
    g2 = bool(abs(dec_wf_h - dec_uni_h) < 0.02)
    print(f"[G2] NULL — homogeneous σ_min=1.0: water-fill decidable={dec_wf_h:.2f} == uniform {dec_uni_h:.2f} (gain {dec_wf_h-dec_uni_h:+.3f}≈0) "
          f"⇒ the gain is SPECIFIC to price HETEROGENEITY, not generic optimization -> {g2}")

    # ---- G3: marginal decidability equalized at the optimum = the allocator rule ----
    m_wf = marginal(wf, sig_het); funded = wf > 0.3
    m_funded = m_wf[funded]
    equalized = bool(m_funded.std() / (m_funded.mean() + 1e-12) < 0.25)                  # KKT: equal marginal on funded QoIs
    m_uni = marginal(uni, sig_het); uni_unequal = bool(m_uni.std() / (m_uni.mean() + 1e-12) > 0.4)
    g3 = bool(g1 and g2 and equalized and uni_unequal)
    print(f"[G3] ALLOCATOR RULE = water-filling: at the optimum marginal-decidability EQUALIZED on funded QoIs (CoV {m_funded.std()/(m_funded.mean()+1e-12):.2f}<0.25, "
          f"={equalized}) vs uniform's UNEQUAL marginals (CoV {m_uni.std()/(m_uni.mean()+1e-12):.2f}) ⇒ 'spend where marginal-value highest until "
          f"equalized' = the credited allocator on the σ_min PRICE-VECTOR -> {g3}")

    ok = g1 and g2 and g3
    os.makedirs("reports", exist_ok=True)
    json.dump({
        "claim": ("σ_min price-vector water-filling: the allocation face of the adjustable-resolution/decidability mission. Each QoI has an "
                  "identifiability price σ_min,q; allocating a fixed measurement budget across K QoIs to maximize total decidability is multi-"
                  "commodity water-filling on the σ_min price-vector. G1: with heterogeneous prices, water-filling beats uniform (more QoIs "
                  "decidable) and the allocation skips already-decidable easy QoIs and starves hopeless ones, funding the marginal-value "
                  "middle. G2 (null): with homogeneous prices water-filling equals uniform (no gain), so the value comes from price "
                  "HETEROGENEITY, not generic optimization. G3: at the optimum the marginal decidability-per-cost is equalized across funded "
                  "QoIs (KKT) — the credited B+I allocator rule ('spend iff marginal-value > cost') applied across the σ_min price-VECTOR. This "
                  "is the scoped survivor of the demoted grand unification (σ = price-vector, not scalar), NOT a new law: water-filling is "
                  "textbook; the contribution is the σ_min-price-vector casting + the heterogeneity-null + the allocator tie."),
        "gates": {"G1_heterogeneous_beats_uniform": {"sigma_min": sig_het.tolist(), "waterfill": [round(x, 2) for x in wf],
                  "decidable_wf": round(dec_wf, 3), "decidable_uniform": round(dec_uni, 3), "skips_easy": skips_easy, "starves_hopeless": starves_hopeless}, "G1": g1,
                  "G2_homogeneous_null": {"decidable_wf": round(dec_wf_h, 3), "decidable_uniform": round(dec_uni_h, 3)}, "G2": g2,
                  "G3_allocator_equalized_marginal": {"marginal_funded_cov": round(float(m_funded.std() / (m_funded.mean() + 1e-12)), 3),
                  "marginal_uniform_cov": round(float(m_uni.std() / (m_uni.mean() + 1e-12)), 3), "equalized": equalized}, "G3": g3,
                  "verdict": "PASS" if ok else "FAIL"},
        "honest_scope": ("returns to my adjustable-resolution/decidability mission (its resource-allocation face) after the over-det-honesty "
                         "arc. σ_min,q is the per-QoI identifiability price (minimal-sensor budget ∝ 1/σ_min²); total decidability = Σ Φ(√n_q "
                         "σ_min,q − z); the allocation that maximizes it under a budget is water-filling on the price-vector. Water-filling / "
                         "KKT is textbook; the contribution is (a) the σ_min-PRICE-VECTOR casting of the mission, (b) the null (homogeneous "
                         "prices → no gain, so the value is heterogeneity-specific, not generic optimization — anti-'elegant-template', the "
                         "session's discipline), (c) the tie to the CREDITED allocator rule (spend iff marginal-value>cost, B+I) and the "
                         "SCOPED survivor σ=price-vector (NOT the demoted grand unification). Analytic K=5 QoIs; the CLAIM is the price-vector "
                         "water-filling + heterogeneity-specificity, not the numbers. Ties [[apriori-doe-budget-predictor-b0-gauge]], "
                         "[[sigma-kTeff-storage-dual-waterline-ledger]] (σ=price/waterline), and the decidability CLASS mission. HYPOTHESIS."),
        "provenance": ("σ_min price-vector water-filling (adjustable-resolution allocation face): per-QoI identifiability price σ_min,q; "
                       "allocate budget to maximize Σ decidability = multi-commodity water-filling; heterogeneous prices → water-fill beats "
                       "uniform (skip easy, starve hopeless, fund marginal), homogeneous NULL → no gain (heterogeneity-specific); optimum "
                       "equalizes marginal decidability = credited allocator rule on the price-VECTOR (scoped survivor, not demoted grand "
                       "unification); textbook water-filling, contribution is the casting+null+tie; OMP4"),
    }, open(_artifact("sigmin_price_vector_waterfilling.json"), "w"), indent=1)
    print("=" * 108)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — G1(heterogeneous → water-fill beats uniform)={g1} G2(homogeneous null → no gain)={g2} "
          f"G3(optimum equalizes marginal = allocator rule on price-vector)={g3}")
    print("EVIDENCE -> reports/sigmin_price_vector_waterfilling.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

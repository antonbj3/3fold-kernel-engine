"""A-V49-2 (repr-thread continuation — PRICE-VECTOR multi-commodity WATER-FILLING; D's seed-decode backlog).
The seed-decode's grand unification "enkoppling i allt" (one scalar allocator) was DEMOTED (anti-V9, U's cost-test) — it SURVIVES only as
a PRICE-VECTOR (multi-commodity water-filling), never a scalar price. This builds that: the allocation counterpart to the repr-router
(A-V49-1: which representation per QoI) — here HOW MUCH representation capacity (bits/resolution) to give each QoI, when the σ-PRICE is a
VECTOR (per-QoI uncertainty), not a scalar.

THE PHYSICS (textbook, 0-fit): N QoIs = parallel Gaussian sources with variances σ²_j (the price VECTOR — e.g. per-region twin uncertainty,
high in sparse-coverage regions). Given a total capacity budget R (bits), REVERSE WATER-FILLING minimizes total residual distortion:
   D_j = min(σ²_j, θ),   R_j = max(0, ½ log₂(σ²_j / θ)),   θ = the water-level set by Σ R_j = R.
High-price (high-σ²) QoIs get MORE bits (fill the deep wells first); QoIs whose σ² is BELOW the water-level θ get ZERO bits — the exact
"don't store below the noise floor χ" of the σ=kT_eff STORAGE waterline B*=log₂(range/χ) ([[sigma-kTeff-storage-dual-waterline]]). The
marginal distortion-reduction is EQUALIZED across active QoIs = B+I's allocator spend rule (spend iff Δpayoff>cost) in the multi-commodity setting
([[overdet-false-accept-floor-is-a-law-in-failure-correlation]] capstone). Reused, not re-derived; the NEW delta is the price-IS-A-VECTOR result.

GATES:
  G1 ★WATER-FILLING OPTIMAL: reverse water-filling achieves lower total distortion than EQUAL-rate allocation AND matches the KKT optimum
     (equalized marginal distortion-reduction / a common water-level θ over active QoIs; below-θ QoIs get 0 bits).
  G2 ★PRICE IS A VECTOR, NOT SCALAR (the anti-V9 result): vector water-filling (per-QoI σ²_j) beats a SCALAR-price allocation (all QoIs at the
     MEAN σ²) — and the advantage GROWS with the σ²-vector heterogeneity (spread) → collapsing the price to a scalar loses real value; the price is genuinely multi-commodity.
  G3 ★CONNECTS THE THREADS: below-water-level QoIs get 0 bits (= the storage-waterline "below-χ = don't store" floor), and the active QoIs share
     one water-level θ (= the marginal-value spend rule equalized across commodities) → grounds the repr-thread allocation on the storage floor + the B+I allocator.
  G4 ★NULL: random capacity allocation loses to water-filling (the allocation is real, not chance).

MATCH: multi-commodity reverse water-filling on the σ²-price VECTOR minimizes total twin distortion (beats equal + scalar-price allocation),
gives zero bits to below-noise-floor QoIs (the storage waterline) and one shared water-level to the rest (the equalized spend rule) — grounding
D's decoded "the allocator survives only as a price-VECTOR" (anti-V9: a scalar price loses, more so as the σ-vector spreads).
  python3 price_vector_waterfilling_multicommodity_capacity_allocation.py
"""
import os, sys, json


def _artifact(name):
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np


def reverse_waterfill(var, R):
    """reverse water-filling: allocate total rate R (bits) over parallel Gaussian sources with variances `var` to minimize Σ distortion.
    Returns (rates R_j, distortions D_j, water-level θ)."""
    var = np.asarray(var, float)
    # bisection on the water-level θ: Σ_j max(0, ½ log2(var_j/θ)) = R
    lo, hi = 1e-12, var.max()
    for _ in range(200):
        th = 0.5 * (lo + hi)
        R_tot = np.sum(np.maximum(0.0, 0.5 * np.log2(var / th)))
        if R_tot > R:
            lo = th          # need higher θ (fewer bits)
        else:
            hi = th
    th = 0.5 * (lo + hi)
    R_j = np.maximum(0.0, 0.5 * np.log2(var / th))
    D_j = np.minimum(var, th)
    return R_j, D_j, th


def dist_for_rates(var, R_j):
    """total distortion for a given rate allocation (Gaussian rate-distortion D_j = σ²_j·2^(−2R_j), capped at σ²_j)."""
    return float(np.sum(np.minimum(var, np.asarray(var) * 2.0 ** (-2.0 * np.asarray(R_j)))))


def main():
    print("=" * 122)
    print("A-V49-2 PRICE-VECTOR WATER-FILLING — multi-commodity capacity allocation; the price is a VECTOR not scalar (anti-V9); storage-floor + spend-rule grounded")
    print("=" * 122)
    rng = np.random.RandomState(492)
    N = 8
    # σ²-price VECTOR = per-QoI (per-region) twin uncertainty; heterogeneous — a few sparse/high-σ regions, several well-observed low-σ
    var = np.sort(np.concatenate([rng.uniform(0.02, 0.1, 5), rng.uniform(0.6, 2.0, 3)]))[::-1]
    R = 6.0     # total capacity budget (bits) to split across the 8 QoIs
    print(f"\n  σ²-price vector (per-QoI uncertainty): {np.round(var,3)}   total-rate budget R={R} bits")

    # ---- G1: water-filling optimal vs equal ----
    R_wf, D_wf, theta = reverse_waterfill(var, R)
    D_wf_tot = float(np.sum(D_wf))
    R_eq = np.full(N, R / N); D_eq_tot = dist_for_rates(var, R_eq)
    # KKT: active QoIs (R_j>0) all share the water-level θ (D_j=θ); below-θ QoIs get 0 bits
    active = R_wf > 1e-6
    kkt_ok = np.allclose(D_wf[active], theta, rtol=0.02) and np.all(R_wf[~active] < 1e-6)
    g1 = (D_wf_tot < D_eq_tot * 0.98) and kkt_ok
    print(f"\n  ★G1 WATER-FILLING OPTIMAL: total distortion water-fill {D_wf_tot:.4f} < equal-rate {D_eq_tot:.4f}; water-level θ={theta:.4f}, {active.sum()}/{N} QoIs active (below-θ get 0 bits), KKT equalized {'✓' if kkt_ok else '✗'}  {'✓' if g1 else 'FAIL'}")
    print(f"     rates R_j = {np.round(R_wf,2)}")

    # ---- G2: price is a VECTOR not scalar — vs scalar-price (mean σ²) allocation, and the gap grows with heterogeneity ----
    def scalar_price_dist(var, R):
        # scalar price = treat all QoIs at the MEAN variance → equal water-level → equal rates → distortion on the TRUE variances
        R_j, _, _ = reverse_waterfill(np.full(N, var.mean()), R)   # allocation planned on the scalar (mean) price
        return dist_for_rates(var, R_j)                             # but pay the TRUE per-QoI distortion
    D_scalar = scalar_price_dist(var, R)
    g2a = D_wf_tot < D_scalar * 0.98
    # heterogeneity sweep: gap grows with the σ²-spread
    spreads, gaps = [], []
    base = var.mean()
    for s in [0.2, 0.5, 1.0, 1.8]:
        v = np.clip(base + s * (var - base), 1e-3, None)
        dwf = np.sum(reverse_waterfill(v, R)[1])
        dsc = dist_for_rates(v, reverse_waterfill(np.full(N, v.mean()), R)[0])
        spreads.append(v.std() / v.mean()); gaps.append((dsc - dwf) / dsc)
    grows = all(gaps[i] <= gaps[i + 1] + 1e-6 for i in range(len(gaps) - 1))
    g2 = g2a and grows
    print(f"  ★G2 PRICE IS A VECTOR NOT SCALAR: a SCALAR price (all QoIs at mean-σ²) provably reduces to EQUAL allocation — a scalar CANNOT distinguish QoIs — so it loses ({D_scalar:.4f}) to the vector water-fill ({D_wf_tot:.4f}); "
          f"the advantage GROWS with σ²-heterogeneity (gap {np.round(gaps,3)} as spread {np.round(spreads,2)}) → the value IS the non-uniform allocation the vector enables; collapsing to a scalar is the V9 trap (anti-V9)  {'✓' if g2 else 'FAIL'}")

    # ---- G3: connects storage-floor (below-θ = 0 bits) + spend-rule (shared θ) ----
    n_zero = int((~active).sum())
    g3 = (n_zero >= 1) and kkt_ok
    print(f"  ★G3 CONNECTS THE THREADS: {n_zero} below-water-level QoI(s) get 0 bits (= σ=kT_eff storage 'below-χ don't-store' floor), the {active.sum()} active QoIs share one water-level θ={theta:.4f} "
          f"(= B+I marginal-value spend rule equalized across commodities) → repr-thread allocation grounded on storage-floor + allocator  {'✓' if g3 else 'FAIL'}")

    # ---- G4: null ----
    D_rand = np.mean([dist_for_rates(var, (lambda r: R * r / r.sum())(rng.dirichlet(np.ones(N)))) for _ in range(500)])
    g4 = D_wf_tot < D_rand * 0.9
    print(f"  ★G4 NULL: random capacity allocation mean-distortion {D_rand:.4f} > water-fill {D_wf_tot:.4f} → the allocation is real, not chance  {'✓' if g4 else 'FAIL'}")

    ok = g1 and g2 and g3 and g4
    print(f"\n  ★NO NAKED NUMBER: ships {{multi-commodity reverse water-filling on the σ²-price VECTOR minimizes total twin distortion ({D_wf_tot:.3f}) — beats equal-rate ({D_eq_tot:.3f}) and scalar-price ({D_scalar:.3f}) allocation, "
          f"the gap GROWING with σ²-heterogeneity; gives 0 bits to the {n_zero} below-water-level QoI(s) (storage floor) and one shared water-level θ={theta:.3f} to the active ones (spend rule) — grounding D's 'allocator survives only as a price-VECTOR'}}")
    print(f"  ★ANTI-V9: the demoted grand-unification survives ONLY multi-commodity — a SCALAR price provably loses (G2), more so as the σ-vector spreads, so the identity is the price-VECTOR water-filling, not a scalar allocator. Reuses storage-waterline + spend-rule (not re-derived). HYPOTHESIS.")

    os.makedirs("artifacts", exist_ok=True)
    with open(_artifact("price_vector_waterfilling_multicommodity_capacity_allocation.json"), "w") as fh:
        json.dump({"module": "price_vector_waterfilling_multicommodity_capacity_allocation",
                   "provenance": "A-V49-2 (repr-thread continuation, seed-decode backlog): multi-commodity reverse water-filling on the σ²-price vector — the allocation counterpart to the "
                   "repr-router (A-V49-1). Minimizes total distortion (beats equal + scalar-price allocation, gap growing with σ²-heterogeneity), gives 0 bits to below-water-level QoIs "
                   "(= σ=kT_eff storage floor) and a shared water-level to the active ones (= B+I marginal-value spend rule). Grounds D's decoded 'the allocator survives only as a price-VECTOR, "
                   "never a scalar' (anti-V9). Reuses the storage-waterline + spend-rule, not re-derived.",
                   "sigma2_price": var.tolist(), "R_budget": R, "water_level_theta": float(theta), "rates": R_wf.tolist(),
                   "dist_waterfill": D_wf_tot, "dist_equal": D_eq_tot, "dist_scalar_price": D_scalar, "dist_random": float(D_rand),
                   "heterogeneity_spread": spreads, "vector_vs_scalar_gap": gaps, "n_below_waterlevel": n_zero,
                   "gates": {"G1_waterfill_optimal": bool(g1), "G2_price_is_vector": bool(g2), "G3_connects_storage_and_spend": bool(g3), "G4_null": bool(g4)}, "ok": bool(ok)}, fh, indent=1)
    tail = (f"PRICE-VECTOR WATER-FILLING SHIPPED — multi-commodity reverse water-filling minimizes total twin distortion ({D_wf_tot:.3f}, vs equal {D_eq_tot:.3f}, scalar-price {D_scalar:.3f}); gap GROWS with σ²-heterogeneity → "
            f"the price is a VECTOR not scalar (anti-V9); {n_zero} below-water-level QoI(s) get 0 bits (storage floor), active QoIs share one water-level (spend rule). Grounds D's 'allocator survives only as a price-VECTOR'."
            if ok else "OPEN — see gates")
    print(f"\n{'='*122}\n{tail}   EXIT={0 if ok else 1}\n{'='*122}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

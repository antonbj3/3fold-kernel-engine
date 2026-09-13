#!/usr/bin/env python3
"""
cell 290 — COMPLETE the ENGINE manifold-reduction instrument (w288 rank rationale → w289 "3 points not 2" → w290 "WHICH 3?").
For an arbitrary clipped face polygon (N candidate contact vertices), the reduction keeps 3 of them (w289). WHICH 3? My σ_min /
optimal-experiment-design (OED) strength answers it. The wrench map for a face with normal n=(0,0,1) is G=[[1, r_iy, −r_ix]] per
point (3×N). A 3-point subset gives a 3×3 G; the manifold realizes a target wrench w by impulses λ=G⁻¹w, so the WORST-case impulse
to realize a unit wrench is 1/σ_min(G) — σ_min(G) is the manifold's wrench-conditioning / margin-against-rocking.

IMPORTANT — G (hence σ_min) is about a REFERENCE point (torque is reference-dependent): the physical choice is the body COM
projected onto the face. All points below are expressed relative to that COM (origin). σ_min(G) is then the worst-direction wrench
conditioning ABOUT THE COM.

FIRST PRINCIPLES / the OED distinction (the point of this cell): det(G) = 2·(signed triangle AREA) [checked below]. And — the
sharper closed form I derived after my FIRST construction failed to separate the two (force-own-negative) — for points centered at
the reference, σ_min(G)² = min(N, λ_min(scatter_about_reference)): the F_z ([1,…,1]) DOF is ALWAYS well-conditioned (its singular
value² = N), so the geometry only degrades the two TORQUE directions, and σ_min(G) = √(min(N, smallest principal 2nd-moment of the
points about the COM)). So:
  • D-OPTIMAL (max |det G| = max product of σ's = max triangle AREA ∝ √(λ₁λ₂)) — minimizes the confidence-ellipsoid VOLUME.
  • E-OPTIMAL (max σ_min(G) = √(min(N, λ_min scatter)) ∝ √λ_min) — maximizes the WORST-direction (thinnest-axis) wrench margin.
Area ∝ √(λ₁λ₂) (geometric mean) vs σ_min ∝ √λ_min (the MIN): these genuinely differ — a THIN large-area triangle (λ₁≫λ₂, big
√(λ₁λ₂) but tiny √λ₂ ⟹ near-collinear, near-w289-rank-deficient) LOSES to a ROUND straddle-the-COM triangle (λ₁≈λ₂, larger √λ_min)
on σ_min. For a ROBUST manifold the physically-right criterion is E-optimal (worst-direction margin = 1/σ_min the worst impulse to
hold a unit wrench = margin against rocking; and σ_min>0 needs the 3 points to STRADDLE the COM — the static support-polygon
condition), = my session's report-the-bound / worst-region theme — NOT max-area.

PREREG (C): (1) det(G) = 2·signed-area for 3 points (machine precision); (1b) σ_min(G)² = min(N, λ_min(scatter about reference))
to machine precision (the derived closed form / meaning); (2) on a constructed COM-centered candidate set the max-AREA (D-optimal)
triple ≠ the max-σ_min (E-optimal) triple — D-optimal is thin (small σ_min, near-rocking, larger area) while E-optimal is round
(larger σ_min, smaller area); (3) a near-collinear triple → σ_min(G)→0 (continuously → w289's rank(G)<3 rocks). ⟹ WHICH-3 =
E-optimal (max σ_min(G)), the worst-direction-robust round straddle-the-COM triangle, distinct from the max-area D-optimal pick.
¬C = det≠2·area / closed-form off / D-optimal==E-optimal always / σ_min not→0 at collinearity. Ties w288/w289 (the reduction
instrument), [[dataset-mean-fidelity-is-area-diluted-certify-per-frame-worst-region]] (worst not mean), [[flagship-headline-report-the-bound-not-the-naked-mean]], [[cert-budget-allocation-is-water-filling-on-the-price-vector]], [[fisher-lambda-min-not-jacobian-sigma-min]], [[watertight-verification]].
"""
import numpy as np
from itertools import combinations
np.set_printoptions(precision=4, suppress=True)

def G_of(points):
    """wrench map for a bottom face (normal +z): unit normal impulse at r=(x,y) makes wrench (F_z, τ_x, τ_y) = (1, y, −x)."""
    return np.array([[1.0, p[1], -p[0]] for p in points]).T                        # (3, npts)

def signed_area(tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    return 0.5*((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

def sigma_min(M): return float(np.linalg.svd(M, compute_uv=False)[-1])

def main():
    print("="*108); print("cell 290  ENGINE manifold reduction — WHICH 3 points? E-optimal (max σ_min(G), roundest) NOT D-optimal (max area)"); print("="*108)
    rng = np.random.default_rng(7)

    # (1) det(G) = 2·signed-area, over random triangles
    print("\n  (1) det(G) = 2·(signed triangle area)?  [G realizes the wrench; its determinant IS the geometric support]")
    print("      trial   signed_area   det(G)     2·area    match")
    det_ok = True
    for t in range(5):
        tri = rng.uniform(-2, 2, (3, 2)); G = G_of(tri); d = np.linalg.det(G); a = signed_area(tri)
        m = abs(d - 2*a) < 1e-9; det_ok = det_ok and m
        print("      %d       %+8.4f      %+8.4f   %+8.4f   %s" % (t, a, d, 2*a, m))

    # (1b) DERIVED closed form: for points centered at the reference, σ_min(G)² = min(N, λ_min(scatter about reference)).
    print("\n  (1b) σ_min(G)² = min(N, λ_min(scatter about reference))?  [F_z DOF gives σ²=N always; geometry degrades only torque]")
    print("      trial   σ_min(G)²   min(N, λ_min scatter)   match")
    cf_ok = True
    for t in range(5):
        pts = rng.uniform(-3, 3, (5, 2)); pts = pts - pts.mean(0)                   # center at reference (COM)
        G = G_of(pts); smin2 = sigma_min(G)**2
        S = pts.T @ pts                                                             # scatter (2×2) about the centered reference
        pred = min(len(pts), float(np.linalg.eigvalsh(S)[0]))
        m = abs(smin2 - pred) < 1e-8; cf_ok = cf_ok and m
        print("      %d       %.5f     %.5f                %s" % (t, smin2, pred, m))

    # (2) constructed COM-centered candidate face: a thin large-AREA triple AND a round straddle-the-COM triple both present.
    #     (reference = COM projection = origin; all points straddle it). Thin {0,1,·} = big area, small σ_min; round {3,4,5} large σ_min.
    r3 = 1.732
    cand = np.array([
        [-10.0, -0.30],  # 0  far left    }  {0,1,·} thin: base 20, tiny height => big AREA, small σ_min (near-collinear)
        [ 10.0, -0.30],  # 1  far right   }
        [  0.0,  0.60],  # 2  near-axis apex
        [  0.0, -2.00],  # 3  round triple: equilateral straddling the COM (circumradius 2) => λ_min scatter large => max σ_min
        [  r3,   1.00],  # 4  round triple
        [ -r3,   1.00],  # 5  round triple
    ])
    print("\n  (2) candidate clipped-face vertices (N=%d, about the COM=origin) — all 3-subsets, D-optimal (area) vs E-optimal (σ_min(G)):" % len(cand))
    print("      subset        area      σ_min(G)    shape")
    best_area = (-1, None); best_smin = (-1, None)
    for c in combinations(range(len(cand)), 3):
        tri = cand[list(c)]; a = abs(signed_area(tri)); s = sigma_min(G_of(tri))
        if a > best_area[0]: best_area = (a, c)
        if s > best_smin[0]: best_smin = (s, c)
    # print the two extremal picks + a couple references
    show = sorted({best_area[1], best_smin[1], (0, 1, 2), (3, 4, 5)}, key=lambda c: -abs(signed_area(cand[list(c)])))
    for c in show:
        tri = cand[list(c)]; a = abs(signed_area(tri)); s = sigma_min(G_of(tri))
        tag = []
        if c == best_area[1]: tag.append("★D-OPT max-AREA")
        if c == best_smin[1]: tag.append("★E-OPT max-σ_min")
        print("      %-12s  %6.3f    %7.4f     %s" % (str(c), a, s, " ".join(tag)))
    d_opt, e_opt = best_area[1], best_smin[1]
    d_vs_e_differ = d_opt != e_opt
    # the D-optimal (max-area) triple is the ELONGATED one (small σ_min) vs E-optimal rounder (larger σ_min)
    s_dopt = sigma_min(G_of(cand[list(d_opt)])); s_eopt = sigma_min(G_of(cand[list(e_opt)]))
    a_dopt = abs(signed_area(cand[list(d_opt)])); a_eopt = abs(signed_area(cand[list(e_opt)]))
    e_more_robust = s_eopt > s_dopt                                                # E-optimal has the larger worst-direction margin
    d_bigger_area = a_dopt > a_eopt                                                # D-optimal has the larger area

    # (3) near-collinear triple → σ_min(G)→0 continuously (approaches w289's rank(G)<3 rocks)
    print("\n  (3) collapse a triple toward collinear (apex height h→0): σ_min(G)→0 (continuous approach to w289's rank(G)<3 rocks):")
    print("      apex height h     area      σ_min(G)")
    smins = []
    for h in [1.0, 0.3, 0.1, 0.03, 0.003]:
        tri = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, h]]); s = sigma_min(G_of(tri)); smins.append(s)
        print("      %.3f             %6.3f    %.4e" % (h, abs(signed_area(tri)), s))
    smin_to_zero = smins[-1] < 0.05*smins[0] and smins[-1] < 0.02

    print("\n  [det(G)=2·area exactly] %s   [σ_min(G)²=min(N,λ_min scatter) closed form] %s   [D-optimal (max-area) ≠ E-optimal (max-σ_min): differ, σ_min %.3f→%.3f while area %.2f→%.2f] %s   [near-collinear ⟹ σ_min(G)→0 (w289 rocks)] %s"
          % (det_ok, cf_ok, s_dopt, s_eopt, a_dopt, a_eopt, d_vs_e_differ and e_more_robust and d_bigger_area, smin_to_zero))

    print("\n  VERDICT (WHICH 3 points — the σ_min/OED answer completing the ENGINE reduction instrument):")
    if det_ok and cf_ok and d_vs_e_differ and e_more_robust and d_bigger_area and smin_to_zero:
        print("  ✓ WHICH-3 = E-OPTIMAL (max σ_min(G)) — the round straddle-the-COM triangle, NOT the max-area one. First principles:")
        print("    det(G) = 2·(triangle AREA) exactly AND σ_min(G)² = min(N, λ_min(scatter about COM)) exactly (both verified). The F_z")
        print("    DOF is always well-conditioned (σ²=N), so σ_min(G) = √(smallest principal 2nd-moment of the 3 points about the COM) =")
        print("    the thinnest-axis spread. Area ∝ √(λ₁λ₂) (geometric mean) but σ_min ∝ √λ_min (the MIN) — so the max-AREA (D-optimal)")
        print("    triple can be a THIN large triangle (λ₁≫λ₂ = near-collinear = near w289's rank(G)<3) with TINY σ_min ⟹ ~1/σ_min→∞")
        print("    impulse to resist a torque about its long axis ⟹ it nearly ROCKS. On the constructed COM-centered face the max-area")
        print("    triple (%s, area %.2f) has σ_min=%.3f while the E-optimal round triple (%s, area %.2f) has σ_min=%.3f — %.1f× the" % (str(d_opt), a_dopt, s_dopt, str(e_opt), a_eopt, s_eopt, s_eopt/s_dopt))
        print("    worst-case wrench margin at only %.0f%% of the area. ⟹ for a ROBUST manifold the right criterion is E-OPTIMAL (max" % (100*a_eopt/a_dopt))
        print("    σ_min(G) = worst-direction robustness = margin against rocking, 1/σ_min the worst required impulse; and σ_min>0 needs the")
        print("    3 points to STRADDLE the COM — the static support-polygon condition), NOT D-optimal max-area (info-VOLUME). This is")
        print("    EXACTLY my session's worst-case-not-mean theme (report the bound / certify-per-worst-region) landed on the ENGINE")
        print("    geometry: the reduction minimizes the WORST wrench-conditioning, not the average support. And σ_min(G)→0 continuously")
        print("    as the triple → collinear, so w289's binary rank(G)=3 gate is the boundary of this continuous σ_min margin (rank = the")
        print("    0/nonzero limit). ★COMPLETED STEER for @ENGINE-driver: reduce the clipped face polygon to the 3 vertices MAXIMIZING")
        print("    σ_min(G) about the COM (E-optimal round straddle-the-COM triangle) — a cheap 3×3 σ_min over the O(N³) vertex triples")
        print("    (N≤~8 after clip) — NOT the largest-area 3. Full instrument: w288 (WHY reduce: rank-3 twist self-stress) → w289 (HOW")
        print("    MANY: 3, both ranks maximal) → w290 (WHICH 3: E-optimal max-σ_min about COM). HONEST: G/σ_min are reference-dependent —")
        print("    use the COM projection (done); if support-VOLUME must not shrink below a floor, constrain area ≥ area_min THEN maximize")
        print("    σ_min (lexicographic). Force-own-negative: my FIRST candidate set failed to separate D from E — that failure forced the")
        print("    √(min(N,λ_min)) closed form, the deeper result.")
        print("  HYPOTHESIS+repro: python3 engine_which_3_points_Eoptimal_sigmamin_manifold_not_max_area.py")
    else:
        print("  ~ RESULT: det_ok=%s d≠e=%s e_robust=%s d_bigger=%s smin→0=%s (D-opt %s area %.2f σ %.3f | E-opt %s area %.2f σ %.3f) — inspect."
              % (det_ok, d_vs_e_differ, e_more_robust, d_bigger_area, smin_to_zero, str(d_opt), a_dopt, s_dopt, str(e_opt), a_eopt, s_eopt))

if __name__ == "__main__":
    main()

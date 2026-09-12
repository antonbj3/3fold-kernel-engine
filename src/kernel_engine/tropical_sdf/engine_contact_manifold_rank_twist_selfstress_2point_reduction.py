#!/usr/bin/env python3
"""
cell 288 — ENGINE P2 (mission's neglected half) as an INSTRUMENT for the ENGINE driver (NOT committed to the ENGINE branch — my
worktree, handed off): verify via RANK/σ_min analysis (my strength) the CORE rationale of the box3d 2-point manifold reduction.
The ENGINE's engine_p2_deself.py found: the 4-coplanar-corner contact manifold's Delassus K is RANK-3 (one zero eigenvalue) with
a TWIST self-stress null [−½,½,½,−½] that produces zero net wrench — the un-removable-by-friction-fix redundancy that drives the
mu>0 friction-lean tumble (6 incremental fixes refuted). The claim the 2-point reduction rests on: keeping ≤2 non-redundant
contact points makes K FULL-RANK ⟹ NO twist self-stress ⟹ no friction-lean. This is a standalone rank fact (no dynamics needed).

This is my session's OVER-DETERMINATION / λ_min theme (w277/w286: a rank-deficient operator has a null mode = a σ_min=0 direction
that no incremental fix removes — only removing the redundancy does) applied to the CONTACT MANIFOLD. Delassus K = J·W·Jᵀ, J_i =
[n, r_i×n] the normal-contact Jacobian at corner i, W = blockdiag(I₃/m, I_body⁻¹).

PREREG (C): (1) the 4-coplanar-corner K is RANK-3 (exactly one zero eigenvalue) with null = the twist self-stress ∝[1,−1,1,−1]
(reproduces the ENGINE finding); (2) the twist mode produces ZERO net wrench (G·twist=0, G the corner→wrench map); (3) a 2-point
reduction (2 non-adjacent corners) is FULL-RANK (rank-2, no null); (4) so the 2-point reduction removes EXACTLY the σ_min=0
redundancy that friction-fixes could not. ⟹ the box3d 2-point rationale is confirmed by rank analysis. ¬C = 4-corner full-rank /
2-point rank-deficient. Ties the ENGINE engine_p2_deself.py finding, cell 277 (rank-deficiency under coupling), [[over-determination-requires-same-kind]],
[[determinism-necessary-for-certifiability-bitset-routing-is-cert-composition]], [[fisher-lambda-min-not-jacobian-sigma-min]], [[watertight-verification]].
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def delassus(corners, n, m, Iinv):
    """K[i,j] = J_i·W·J_jᵀ for normal contacts; J_i = [n, r_i×n], W = blockdiag(I₃/m, Iinv)."""
    J = np.array([np.concatenate([n, np.cross(r, n)]) for r in corners])          # (nc, 6)
    W = np.block([[np.eye(3)/m, np.zeros((3, 3))], [np.zeros((3, 3)), Iinv]])
    return J@W@J.T, J

def wrench_map(corners, n):
    """G: corner normal impulses → net wrench (F_z, τ_x, τ_y). twist self-stress ⟹ G·twist = 0."""
    return np.array([np.concatenate([n, np.cross(r, n)]) for r in corners]).T      # (6, nc); rows: [Fx,Fy,Fz,τx,τy,τz]

def main():
    print("="*104); print("cell 288  ENGINE contact-manifold RANK (instrument for @ENGINE-driver) — 4-corner rank-3 twist self-stress vs 2-point full-rank"); print("="*104)
    R = 0.5; m = 1.0; s = 2*R; Ic = m*s**2/6.0; Iinv = np.eye(3)/Ic                # unit cube, inertia I=m·s²/6 per axis
    n = np.array([0.0, 0.0, 1.0])
    corners = [np.array([sx*R, sy*R, -R]) for (sx, sy) in [(1, 1), (1, -1), (-1, -1), (-1, 1)]]  # 4 bottom corners (order around the face)

    # (1) 4-coplanar-corner K
    K4, J4 = delassus(corners, n, m, Iinv)
    ev4 = np.sort(np.linalg.eigvalsh(K4)); rank4 = int((ev4 > 1e-9*ev4.max()).sum())
    null_vec = np.linalg.svd(K4)[2][-1]                                            # null direction (smallest sing vector)
    print("\n  (1) 4-COPLANAR-CORNER Delassus K (4×4):\n%s" % np.array2string(K4, prefix="      "))
    print("      eigenvalues = %s  ⟹ rank %d/4 (%s)" % (ev4, rank4, "RANK-DEFICIENT: a self-stress null" if rank4 < 4 else "full"))
    print("      null mode (σ_min direction) = %s  (twist self-stress ∝ [1,−1,1,−1] pattern)" % np.round(null_vec/np.abs(null_vec).max(), 3))

    # (2) the twist mode produces zero net wrench
    G = wrench_map(corners, n); twist = np.array([1.0, -1.0, 1.0, -1.0])/2
    net_wrench = G@twist
    print("\n  (2) net wrench of the twist self-stress G·[½,−½,½,−½] = %s  (‖·‖=%.2e ⟹ ZERO net wrench)" % (np.round(net_wrench, 4), np.linalg.norm(net_wrench)))

    # (3) 2-point reduction: 2 non-adjacent (opposite) corners
    two = [corners[0], corners[2]]
    K2, _ = delassus(two, n, m, Iinv); ev2 = np.sort(np.linalg.eigvalsh(K2)); rank2 = int((ev2 > 1e-9*ev2.max()).sum())
    print("\n  (3) 2-POINT reduction (opposite corners) K (2×2):\n%s\n      eigenvalues = %s ⟹ rank %d/2 (%s)"
          % (np.array2string(K2, prefix="      "), ev2, rank2, "FULL-rank, NO twist null" if rank2 == 2 else "rank-deficient"))

    rank3_deficient = rank4 == 3 and ev4[0] < 1e-9*ev4.max()
    twist_null = abs(np.dot(null_vec/np.linalg.norm(null_vec), twist/np.linalg.norm(twist))) > 0.99
    zero_wrench = np.linalg.norm(net_wrench) < 1e-9
    twopoint_full = rank2 == 2
    print("\n  [4-corner K is RANK-3 (one zero eigenvalue)] %s   [null mode = the twist self-stress [1,−1,1,−1]] %s   [twist produces ZERO net wrench] %s   [2-point reduction FULL-rank (no twist null)] %s"
          % (rank3_deficient, twist_null, zero_wrench, twopoint_full))

    print("\n  VERDICT (ENGINE 2-point-reduction rationale, verified by rank — instrument for @ENGINE-driver):")
    if rank3_deficient and twist_null and zero_wrench and twopoint_full:
        print("  ✓ CONFIRMED — the box3d 2-point manifold reduction's core rationale holds by RANK analysis, standalone (no dynamics):")
        print("    the 4-coplanar-corner contact Delassus K is RANK-3 (eigenvalues %s, one exact zero) with a null mode = the TWIST" % np.round(ev4, 2))
        print("    self-stress ∝[1,−1,1,−1] that produces ZERO net wrench (‖G·twist‖=%.0e) — a σ_min=0 direction that carries force but" % np.linalg.norm(net_wrench))
        print("    no physical effect ⟹ the friction cone acts on a phantom DOF ⟹ the friction-lean tumble (the ENGINE's 6-refuted-fix")
        print("    mechanism). A 2-POINT reduction (2 non-adjacent corners) is FULL-rank (eigenvalues %s, no null) ⟹ it removes EXACTLY" % np.round(ev2, 2))
        print("    the σ_min=0 twist redundancy that NO incremental friction fix could remove (only removing the over-determination does).")
        print("    ⟹ the box3d 2-point reduction is the RIGHT structural fix, confirmed by rank (my σ_min/over-determination strength =")
        print("    the ENGINE's rank-3 finding). This is my session's λ_min/over-determination theme (w277/w286: a rank-deficient operator")
        print("    has a σ_min=0 null no incremental fix removes) applied to the CONTACT MANIFOLD. INSTRUMENT for the @ENGINE-driver: the")
        print("    reference-face-clip → ≤2-point manifold is validated in its rank rationale; the remaining build is the geometry (Sutherland-")
        print("    Hodgman clip, POC exists) + wiring the block-normal solve (verified). HANDED OFF, not committed to the ENGINE branch.")
        print("    HONEST: this verifies the RANK rationale (why 2-point works), not the full dynamic mu>0 K>10 run — that's the ENGINE-")
        print("    driver's integration (beyond-gate; M5 K=8 already solved by persep within K_crit=10).")
        print("  HYPOTHESIS+repro: python3 engine_contact_manifold_rank_twist_selfstress_2point_reduction.py")
    else:
        print("  ~ RESULT: rank3=%s twist_null=%s zero_wrench=%s 2pt_full=%s (ev4=%s ev2=%s) — inspect." % (rank3_deficient, twist_null, zero_wrench, twopoint_full, np.round(ev4, 3), np.round(ev2, 3)))

if __name__ == "__main__":
    main()

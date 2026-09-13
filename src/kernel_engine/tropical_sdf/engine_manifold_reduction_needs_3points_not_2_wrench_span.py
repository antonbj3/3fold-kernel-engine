#!/usr/bin/env python3
"""
cell 289 — SYMMETRIC-QC / force-own-negative on my OWN cell 288: I claimed the "2-point reduction" is the right fix because its
Delassus K is full-rank (no twist self-stress). That is INCOMPLETE. Full-rank K (no self-stress) is NECESSARY but NOT SUFFICIENT:
the reduced manifold must ALSO span the physical contact WRENCH. A 3-D face-face contact wrench is 3-DIMENSIONAL (F_z, τ_x, τ_y),
so the wrench map G must have rank 3 — which needs ≥3 points. A 2-POINT manifold has full-rank K (2×2) but rank-2 G ⟹ it UNDER-
CONSTRAINS one torque (the box rocks about the axis through the 2 points). So the correct non-redundant reduction is 3 POINTS,
not 2 — the minimal set that removes the self-stress WITHOUT losing a wrench DOF. This CORRECTS cell 288 (and the ENGINE handoff's
"≤2-point" wording) for the 3-D face contact.

TWO ranks govern a manifold: rank(K) (self-stress: rank<npts ⟹ a phantom-DOF self-stress, the friction-lean, w288) and rank(G)
(wrench span: rank<3 ⟹ under-constrained, rocks). The CORRECT reduction: minimal npts with rank(G)=3 AND rank(K)=npts (full).

PREREG (C): (1) 4-corner: rank(G)=3 (wrench OK) BUT rank(K)=3<4 (self-stress redundancy) — over-determined (w288); (2) 2-point:
rank(K)=2 (no self-stress) BUT rank(G)=2<3 — UNDER-CONSTRAINS a torque (my w288 over-claimed this as the fix); (3) 3-point:
rank(G)=3 (wrench OK) AND rank(K)=3 (no self-stress) — the CORRECT reduction (both conditions). ⟹ reduce to 3 non-redundant
points, not 2. ¬C = 2-point spans the wrench / 3-point rank-deficient. Ties cell 288 (corrects it), [[over-determination-requires-same-kind]],
[[force-own-honest-negative]], [[fisher-lambda-min-not-jacobian-sigma-min]], [[watertight-verification]].
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def K_and_G(corners, n, m, Iinv):
    J = np.array([np.concatenate([n, np.cross(r, n)]) for r in corners])          # (npts, 6) contact Jacobian
    W = np.block([[np.eye(3)/m, np.zeros((3, 3))], [np.zeros((3, 3)), Iinv]])
    K = J@W@J.T
    G = np.array([[1.0, r[1], -r[0]] for r in corners]).T                         # wrench map (F_z, τ_x, τ_y) per unit normal impulse; (3, npts)
    return K, G

def ranks(K, G):
    rk = int((np.linalg.svd(K, compute_uv=False) > 1e-9*np.linalg.svd(K, compute_uv=False).max()).sum())
    rg = int((np.linalg.svd(G, compute_uv=False) > 1e-9).sum())
    return rk, rg

def main():
    print("="*104); print("cell 289  ENGINE manifold reduction needs 3 POINTS not 2 (force-own-negative on w288): 2-point UNDER-constrains the 3-D wrench"); print("="*104)
    R = 0.5; m = 1.0; Ic = m*(2*R)**2/6.0; Iinv = np.eye(3)/Ic; n = np.array([0.0, 0.0, 1.0])
    corners = [np.array([sx*R, sy*R, -R]) for (sx, sy) in [(1, 1), (1, -1), (-1, -1), (-1, 1)]]

    cases = [("4-coplanar-corner", corners), ("2-point (opposite)", [corners[0], corners[2]]),
             ("2-point (adjacent)", [corners[0], corners[1]]), ("3-point", corners[:3])]
    print("\n     manifold             npts   rank(K) (self-stress)   rank(G) (wrench span, need 3)   verdict")
    rows = {}
    for name, cs in cases:
        K, G = K_and_G(cs, n, m, Iinv); rk, rg = ranks(K, G); npts = len(cs); rows[name] = (npts, rk, rg)
        selfstress = "OVER-det (self-stress null)" if rk < npts else "no self-stress"
        wrench = "UNDER-constrains (rocks)" if rg < 3 else "wrench OK"
        v = "★CORRECT (both)" if rk == npts and rg == 3 else ("over-determined" if rk < npts else "UNDER-constrained")
        print("     %-20s  %d      %d/%d  %-22s  %d/3  %-24s  %s" % (name, npts, rk, npts, "("+selfstress+")", rg, "("+wrench+")", v))

    n4, rk4, rg4 = rows["4-coplanar-corner"]; _, rk2, rg2 = rows["2-point (opposite)"]; _, rk3, rg3 = rows["3-point"]
    four_overdet = rk4 == 3 and rg4 == 3                                           # w288: wrench OK but self-stress
    two_underconstrains = rg2 < 3                                                  # THE CORRECTION: 2-point loses a wrench DOF
    three_correct = rk3 == 3 and rg3 == 3                                          # 3-point: both conditions
    print("\n  [4-corner: wrench-OK (rank G=%d) but self-stress (rank K=%d<4) = over-det (w288)] %s   [2-POINT UNDER-CONSTRAINS the 3-D wrench (rank G=%d<3) — w288 OVER-CLAIMED it as the fix] %s   [3-POINT is CORRECT: rank G=3 AND rank K=3] %s"
          % (rg4, rk4, four_overdet, rg2, two_underconstrains, three_correct))

    print("\n  VERDICT (force-own-negative on cell 288 — the reduction target is 3 points, not 2):")
    if four_overdet and two_underconstrains and three_correct:
        print("  ✗ FORCE-OWN-NEGATIVE — cell 288's '2-point reduction is the right fix' is INCOMPLETE and, as stated, WRONG for a 3-D face")
        print("    contact. w288 correctly showed the 2-point K is full-rank (no twist self-stress) — but that is NECESSARY, NOT")
        print("    SUFFICIENT: the manifold must ALSO span the 3-D contact WRENCH (F_z, τ_x, τ_y), i.e. rank(G)=3. A 2-POINT manifold")
        print("    has rank(G)=2 ⟹ it CANNOT produce a torque about the axis through the two points ⟹ it UNDER-CONSTRAINS ⟹ the box")
        print("    ROCKS about that axis (a NEW instability, trading the friction-lean for rocking). The CORRECT non-redundant reduction")
        print("    is 3 POINTS: rank(G)=3 (spans the full wrench, no rocking) AND rank(K)=3 (full, no self-stress, no friction-lean) —")
        print("    the minimal set satisfying BOTH. ⟹ TWO ranks govern a contact manifold and they pull opposite ways: rank(K)=npts")
        print("    (drop redundant points) vs rank(G)=3 (keep enough points to span the wrench); the sweet spot is exactly 3 for a 3-D")
        print("    face. ★CORRECTED STEER for @ENGINE-driver: reduce the clipped face polygon to 3 (not 2) non-redundant points — the")
        print("    box3d '≤2-point' wording is a 2-D (box2d, rank-2 wrench) carry-over; 3-D needs 3. This is my session's over-determination")
        print("    theme SHARPENED: removing an over-determination (rank K) must not create an UNDER-determination (rank G) — the")
        print("    identifiable/certifiable manifold is the one with BOTH ranks maximal-and-consistent. I over-claimed w288; corrected here.")
        print("  HYPOTHESIS+repro: python3 engine_manifold_reduction_needs_3points_not_2_wrench_span.py")
    else:
        print("  ~ RESULT: four_overdet=%s two_under=%s three_correct=%s (rows %s) — inspect." % (four_overdet, two_underconstrains, three_correct, rows))

if __name__ == "__main__":
    main()

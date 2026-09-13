#!/usr/bin/env python3
"""OUTPUT-SENSITIVE TWIN COMPUTE = "only render what you see on screen" (a set-aside rendering concept), realized
geometrically via the ADJOINT. In rendering you skip what the camera can't see; in a twin you skip COMPUTING what the
OBSERVABLE can't see. The "view frustum" of a quantity-of-interest (QoI) y=<g,u> is its ADJOINT field z (L^T z = g):
where z≈0 the observable is BLIND to that region → cull it to coarse resolution EVEN IF the solution varies wildly
there. This is the dual-weighted-residual principle and it unifies culling ⊗ value-equivalence ⊗ σ-governance: cost
scales with what you OBSERVE, not the full system — something a monolith (solve-everything) cannot do.

GEOMETRIC HYPOTHESIS: put the source (high solution variation = a bright "decoy") FAR from the QoI point. Solution-
based AMR refines the decoy (big |∇u|) and wastes DOF the observable can't use; GOAL-ORIENTED AMR refines by the
dual-weighted indicator |∇u|·|z|, CULLS the decoy (z≈0 there) and spends DOF where the observable actually sees →
reaches the QoI accuracy at far fewer DOF. We MEASURE QoI-error vs DOF for uniform / solution-based / goal-oriented.
Symmetric QC: if goal-oriented does NOT beat solution-based, report it; verify the adjoint culls the RIGHT region.

  python3 goal_oriented_culling.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('amr_poisson',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np
from amr_octree_fv import tpfa_solve, make_k

N = 96; BLK = 8; NB = N // BLK


def point_source(n, cx, cy, s=2.0):
    a = np.arange(n); xx, yy = np.meshgrid(a, a, indexing="ij")
    f = np.exp(-(((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * s ** 2)))
    return (f / (f.sum() * (1.0 / n) ** 2)).astype(np.float32)            # ∫f≈1


def regions_from_keep(keep_block):
    """keep_block: (NB,NB) bool. Kept blocks → fine (each cell its own region); others → one coarse region/block."""
    rid = np.zeros((N, N), int); nid = 0
    for bi in range(NB):
        for bj in range(NB):
            sl = (slice(bi * BLK, (bi + 1) * BLK), slice(bj * BLK, (bj + 1) * BLK))
            if keep_block[bi, bj]:
                rid[sl] = np.arange(nid, nid + BLK * BLK).reshape(BLK, BLK); nid += BLK * BLK
            else:
                rid[sl] = nid; nid += 1
    return rid


def qoi_at(field, n, px, py):
    return float(field[int(px * n), int(py * n)])


def main():
    print("=" * 88)
    print("OUTPUT-SENSITIVE TWIN COMPUTE ('render only what the observable sees') — goal-oriented vs solution AMR")
    print("=" * 88)
    k = make_k(N)                                       # uniform conductivity
    PX, PY = 0.80, 0.80                                 # QoI point (the "camera")
    # TWO sources: a DRIVER near the QoI (the observable depends on it) + a far DECOY (bright but off-screen).
    f = (0.5 * point_source(N, 0.72, 0.72) + point_source(N, 0.20, 0.20)).astype(np.float32)

    full = np.arange(N * N).reshape(N, N)
    u_ref = tpfa_solve(full, k, f, N); y_true = qoi_at(u_ref, N, PX, PY)
    g = point_source(N, PX, PY, s=1.5)                  # QoI functional g = (smoothed) point at P
    z = tpfa_solve(full, k, g, N)                       # ADJOINT field (L symmetric) = what the observable "sees"

    # block indicators
    gy, gx = np.gradient(u_ref); grad = np.sqrt(gx ** 2 + gy ** 2)
    dwr = grad * np.abs(z)                              # dual-weighted: solution-variation × observable-visibility
    def blockscore(field):
        return field.reshape(NB, BLK, NB, BLK).max(axis=(1, 3))
    s_sol = blockscore(grad); s_goal = blockscore(dwr)
    s_uni = np.zeros((NB, NB))                          # uniform = no preference (spread by index)

    # where does each strategy spend its fine blocks? (sanity: goal must CULL the decoy block, solution must KEEP it)
    decoy_bi, decoy_bj = int(0.20 * NB), int(0.20 * NB)
    qoi_bi, qoi_bj = int(PX * NB), int(PY * NB)

    print(f"\n  QoI u(P) at ({PX},{PY})={y_true:.4e}; source/decoy at (0.20,0.20).  adjoint z: |z| at decoy="
          f"{np.abs(z)[int(0.2*N), int(0.2*N)]:.2e}, at QoI={np.abs(z)[int(PX*N), int(PY*N)]:.2e}")
    print(f"\n  {'fine blks':>9} {'DOF':>7} | {'uniform QoIerr':>14} {'solution QoIerr':>15} {'GOAL QoIerr':>12}")
    rows = []
    for K in (4, 8, 16, 32):
        res = {}
        for name, score in (("uni", s_uni), ("sol", s_sol), ("goal", s_goal)):
            keep = np.zeros((NB, NB), bool)
            if name == "uni":
                idx = np.linspace(0, NB * NB - 1, K).astype(int)            # spread uniformly
            else:
                idx = np.argsort(score.ravel())[::-1][:K]
            keep.ravel()[idx] = True
            rid = regions_from_keep(keep)
            u = tpfa_solve(rid, k, f, N)
            res[name] = abs(qoi_at(u, N, PX, PY) - y_true) / abs(y_true)
            res[name + "_keepdecoy"] = keep[decoy_bi, decoy_bj]
        dof = K * BLK * BLK + (NB * NB - K)
        rows.append((K, dof, res))
        print(f"  {K:>9} {dof:>7} | {res['uni']:>14.3%} {res['sol']:>15.3%} {res['goal']:>12.3%}")

    # decisive: at matched fine-block budget, goal-oriented QoI error < solution-based (it culls the decoy)
    wins = sum(1 for _, _, r in rows if r["goal"] < r["sol"] * 0.9)
    goal_culls_decoy = not rows[0][2]["goal_keepdecoy"]
    sol_keeps_decoy = rows[0][2]["sol_keepdecoy"]
    ok = wins >= 3 and goal_culls_decoy
    print("\n" + "=" * 88)
    print(f"VERDICT: output-sensitive (goal-oriented) twin compute = {'VALIDATED' if ok else 'PARTIAL'}")
    print(f"  MEASURED: goal-oriented (|∇u|·|z|) beats solution-based (|∇u|) on QoI accuracy at matched DOF in")
    print(f"  {wins}/4 budgets. GEOMETRIC mechanism: the adjoint z is ~0 at the decoy source → goal-oriented CULLS it")
    print(f"  (keep-decoy: goal={rows[0][2]['goal_keepdecoy']}, solution={sol_keeps_decoy}) and spends DOF where the")
    print(f"  observable can SEE. = 'render only what's on screen' for a twin: compute cost ∝ the OBSERVABLE, not the")
    print(f"  whole field — a monolith cannot. Unifies culling ⊗ value-equivalence ⊗ σ-governance (the goal-aware")
    print(f"  criterion the σ-FWI re-exam flagged as the open frontier). HONEST: 2D elliptic, block-AMR, linear QoI.")
    print("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

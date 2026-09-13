#!/usr/bin/env python3
"""AUDIT: is the PDE ADJOINT worth its extra solve for POINT-QoI goal-oriented AMR vs CHEAP no-adjoint proxies?

QoI: y = u(P) for -div(k grad u) = f, Dirichlet 0. Block-AMR: pick K fine blocks by an indicator, rest coarse.
Reference = all-fine. Measure |y_amr - y_ref|/|y_ref| at MATCHED fine-block budget K.

INDICATORS COMPARED (block score = max over block cells):
  adjoint   : |grad u| * |z|,  z solves L^T z = g_P  (ONE extra solve)   <-- the textbook one
  grad      : |grad u|                                (no adjoint)
  euclid    : |grad u| * w(euclidean dist to P)        (no adjoint)
  geodesic  : |grad u| * w(geodesic dist to P through 1/k)  (no PDE, Dijkstra on grid)  <-- my main cheap candidate

GEOMETRIES (my own, varied so rerouting matters):
  - source-QoI separation, plus a LOW-K BARRIER wall placed BETWEEN source and QoI with a small GAP (gate).
    The true influence path must route THROUGH the gap. Euclidean distance ignores the wall; geodesic (through 1/k)
    should re-route around it; the adjoint knows it exactly. This is where a cheap geometric proxy should fail and the
    adjoint should win -- if it ever does.
  - homogeneous (no barrier) control.
  - contrast sweep on the barrier conductivity.

I design distance->weight as w = 1/(d+d0) and also test w=exp(-d/L); report best cheap proxy per config.
Report RAW errors, adjoint/best-cheap ratio distribution (median + win-rate), and WHEN the adjoint decisively wins.
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('amr_poisson',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys, heapq
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amr_octree_fv import tpfa_solve

N = 96; BLK = 8; NB = N // BLK


def kfield_homog(n):
    return np.ones((n, n), float)


def kfield_barrier(n, contrast, wall_col, wall_thick, gap_lo, gap_hi):
    """Low-k vertical wall at columns [wall_col, wall_col+wall_thick), with a conductive GAP in rows [gap_lo,gap_hi).
    contrast = k_wall / k_bg  (small = strong barrier)."""
    k = np.ones((n, n), float)
    cols = slice(wall_col, wall_col + wall_thick)
    k[:, cols] = contrast
    k[gap_lo:gap_hi, cols] = 1.0     # the gate
    return k


def point_source(n, ci, cj, s=2.0):
    a = np.arange(n); xx, yy = np.meshgrid(a, a, indexing="ij")
    f = np.exp(-(((xx - ci) ** 2 + (yy - cj) ** 2) / (2 * s ** 2)))
    return (f / (f.sum() * (1.0 / n) ** 2))


def regions_from_keep(keep_block):
    rid = np.zeros((N, N), int); nid = 0
    for bi in range(NB):
        for bj in range(NB):
            sl = (slice(bi * BLK, (bi + 1) * BLK), slice(bj * BLK, (bj + 1) * BLK))
            if keep_block[bi, bj]:
                rid[sl] = np.arange(nid, nid + BLK * BLK).reshape(BLK, BLK); nid += BLK * BLK
            else:
                rid[sl] = nid; nid += 1
    return rid


def geodesic_dist(k, src_ij):
    """Dijkstra shortest path from src cell, edge weight = 0.5*(1/k_a + 1/k_b) (resistive distance, no PDE).
    Low-k cells are EXPENSIVE to cross -> path routes around the barrier through the gap."""
    n = k.shape[0]
    res = 1.0 / np.maximum(k, 1e-12)
    INF = 1e30
    dist = np.full((n, n), INF)
    si, sj = src_ij
    dist[si, sj] = 0.0
    pq = [(0.0, si, sj)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:
                w = 0.5 * (res[i, j] + res[ni, nj])
                nd = d + w
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(pq, (nd, ni, nj))
    return dist


def blockmax(field):
    return field.reshape(NB, BLK, NB, BLK).max(axis=(1, 3))


def run_config(name, k, f, P_ij, src_ij):
    """Returns dict of results: per-indicator QoI error array over budgets K, plus best-cheap per K."""
    full = np.arange(N * N).reshape(N, N)
    u_ref = tpfa_solve(full, k, f, N)
    y_true = u_ref[P_ij[0], P_ij[1]]

    g = point_source(N, P_ij[0], P_ij[1], s=1.5)
    z = tpfa_solve(full, k, g, N)            # adjoint (L symmetric here -> L^T = L)

    gy, gx = np.gradient(u_ref); grad = np.sqrt(gx ** 2 + gy ** 2)

    # distance fields to P
    a = np.arange(N); xx, yy = np.meshgrid(a, a, indexing="ij")
    deuc = np.sqrt((xx - P_ij[0]) ** 2 + (yy - P_ij[1]) ** 2)
    dgeo = geodesic_dist(k, P_ij)            # geodesic through 1/k
    dgeo = dgeo / dgeo[dgeo < 1e29].max()    # normalize to ~[0,1]
    deuc_n = deuc / deuc.max()

    # weight forms
    def w_inv(d, d0): return 1.0 / (d + d0)
    def w_exp(d, L):  return np.exp(-d / L)

    indicators = {}
    indicators["adjoint"] = grad * np.abs(z)
    indicators["grad"] = grad
    # cheap euclidean: two weight forms, keep both (report best)
    indicators["euclid_inv"] = grad * w_inv(deuc_n, 0.05)
    indicators["euclid_exp"] = grad * w_exp(deuc_n, 0.25)
    # cheap geodesic
    indicators["geo_inv"] = grad * w_inv(dgeo, 0.05)
    indicators["geo_exp"] = grad * w_exp(dgeo, 0.25)

    budgets = (4, 8, 16, 24)
    errs = {nm: [] for nm in list(indicators) + ["uniform"]}
    dofs = []
    for K in budgets:
        # uniform spread
        keep = np.zeros((NB, NB), bool)
        idx = np.linspace(0, NB * NB - 1, K).astype(int)
        keep.ravel()[idx] = True
        rid = regions_from_keep(keep)
        u = tpfa_solve(rid, k, f, N)
        errs["uniform"].append(abs(u[P_ij[0], P_ij[1]] - y_true) / abs(y_true))
        for nm, score in indicators.items():
            bs = blockmax(score)
            # always keep the block containing P (any sane scheme would)
            keep = np.zeros((NB, NB), bool)
            order = np.argsort(bs.ravel())[::-1][:K]
            keep.ravel()[order] = True
            keep[P_ij[0] // BLK, P_ij[1] // BLK] = True
            rid = regions_from_keep(keep)
            u = tpfa_solve(rid, k, f, N)
            errs[nm].append(abs(u[P_ij[0], P_ij[1]] - y_true) / abs(y_true))
        dofs.append(K * BLK * BLK + (NB * NB - K))

    # collapse euclid/geo to best-of-weight-form per K
    cheap_families = {
        "grad": ["grad"],
        "euclid": ["euclid_inv", "euclid_exp"],
        "geodesic": ["geo_inv", "geo_exp"],
    }
    best_cheap = []
    best_cheap_name = []
    for ki in range(len(budgets)):
        vals = {}
        for fam, members in cheap_families.items():
            vals[fam] = min(errs[m][ki] for m in members)
        bn = min(vals, key=vals.get)
        best_cheap.append(vals[bn]); best_cheap_name.append(bn)
    return dict(name=name, budgets=budgets, dofs=dofs, errs=errs,
                best_cheap=best_cheap, best_cheap_name=best_cheap_name,
                y_true=float(y_true),
                z_at_src=float(np.abs(z)[src_ij[0], src_ij[1]]),
                z_at_P=float(np.abs(z)[P_ij[0], P_ij[1]]))


def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 100)
    print("ADJOINT vs CHEAP no-adjoint proxies for POINT-QoI goal-oriented block-AMR  (N=%d, BLK=%d)" % (N, BLK))
    print("=" * 100)

    P = (76, 76)                      # QoI point (upper-right), in cell indices
    src = (16, 16)                    # far source (lower-left)
    f = point_source(N, src[0], src[1], s=2.0)

    configs = []
    # 1. homogeneous control
    configs.append(("homog", kfield_homog(N), f, P, src))
    # 2-5. barrier between src and P, GAP at TOP, sweep contrast (strong->mild)
    #    wall at column ~46, gap rows [70,82] (near P side / top). Influence must route up through the gap.
    for contrast in (1e-3, 1e-2, 1e-1, 0.5):
        k = kfield_barrier(N, contrast, wall_col=44, wall_thick=4, gap_lo=68, gap_hi=82)
        configs.append((f"barrier_top_c{contrast:g}", k, f, P, src))
    # 6-7. barrier with gap at BOTTOM (away from straight line), strong contrast -> rerouting most extreme
    for contrast in (1e-3, 1e-2):
        k = kfield_barrier(N, contrast, wall_col=44, wall_thick=4, gap_lo=10, gap_hi=24)
        configs.append((f"barrier_bot_c{contrast:g}", k, f, P, src))
    # 8. barrier with gap in MIDDLE
    k = kfield_barrier(N, 1e-3, wall_col=44, wall_thick=4, gap_lo=42, gap_hi=54)
    configs.append(("barrier_mid_c0.001", k, f, P, src))
    # 9-10. EXTREME detour: tiny gap (4 cells) far from the straight line, strongest contrast. Two source seeds.
    for sj, tag in (((16, 16), "s1"), ((10, 30), "s2")):
        k = kfield_barrier(N, 1e-4, wall_col=44, wall_thick=4, gap_lo=8, gap_hi=12)
        ff2 = point_source(N, sj[0], sj[1], s=2.0)
        configs.append((f"barrier_tinygap_{tag}", k, ff2, P, sj))

    # DIAGNOSTIC: confirm the geodesic actually re-routes around a strong barrier (vs euclidean which goes straight).
    kdiag = kfield_barrier(N, 1e-3, wall_col=44, wall_thick=4, gap_lo=68, gap_hi=82)
    dgeo_diag = geodesic_dist(kdiag, P)
    deuc_diag = np.sqrt((np.arange(N)[:, None] - P[0]) ** 2 + (np.arange(N)[None, :] - P[1]) ** 2)
    # compare geodesic vs euclidean distance from P to the far source, across the wall:
    print(f"DIAGNOSTIC (strong barrier, gap@top): dist P->src  euclid={deuc_diag[src]:.1f}  "
          f"geodesic(thru 1/k)={dgeo_diag[src]:.1f}  ratio={dgeo_diag[src]/deuc_diag[src]:.1f}x "
          f"(>1 => geodesic detours around wall as intended)")
    # also correlation of each indicator's ranking with the adjoint's, on the strong-barrier case (does cheap
    # geodesic align with the adjoint better than euclidean?)
    print()

    all_ratios = []          # adjoint_err / best_cheap_err per (config, budget)
    rows_summary = []
    for name, k, fld, Pij, sij in configs:
        r = run_config(name, k, fld, Pij, sij)
        print(f"\n--- {name}   y(P)={r['y_true']:.4e}   |z| at src={r['z_at_src']:.2e}  at P={r['z_at_P']:.2e}")
        print(f"    {'K':>4} {'DOF':>6} | {'uniform':>9} {'grad':>9} {'adjoint':>9} | "
              f"{'bestcheap':>9} {'(which)':>9} | {'adj/cheap':>9}")
        for i, K in enumerate(r['budgets']):
            adj = r['errs']['adjoint'][i]; bc = r['best_cheap'][i]
            ratio = adj / bc if bc > 0 else float('inf')
            all_ratios.append(ratio)
            print(f"    {K:>4} {r['dofs'][i]:>6} | {r['errs']['uniform'][i]:>9.2%} {r['errs']['grad'][i]:>9.2%} "
                  f"{adj:>9.2%} | {bc:>9.2%} {r['best_cheap_name'][i]:>9} | {ratio:>8.3f}x")
            rows_summary.append((name, K, adj, bc, ratio, r['best_cheap_name'][i]))

    arr = np.array(all_ratios)
    arr = arr[np.isfinite(arr)]
    print("\n" + "=" * 100)
    print("ADJOINT / BEST-CHEAP-PROXY RATIO DISTRIBUTION  (ratio<1 => adjoint better; >1 => cheap better)")
    print("=" * 100)
    print(f"  n = {len(arr)} (config x budget) cases")
    print(f"  median ratio        = {np.median(arr):.3f}x   (adjoint/cheap)")
    print(f"  GEOMEAN ratio       = {np.exp(np.mean(np.log(arr))):.3f}x   (robust to accidental-zero outliers)")
    print(f"  mean   ratio        = {np.mean(arr):.3f}x   (dominated by accidental near-zero cheap hits)")
    print(f"  min / max           = {arr.min():.3f}x / {arr.max():.3f}x")
    print(f"  adjoint-wins (<1)   = {(arr < 1.0).mean():.0%}   "
          f"decisive-adjoint-wins (<0.5) = {(arr < 0.5).mean():.0%}")
    print(f"  cheap-wins  (>=1)   = {(arr >= 1.0).mean():.0%}   "
          f"cheap >2x better      = {(arr > 2.0).mean():.0%}")
    # ROBUST view: restrict to cases where BOTH already reached a usable error (<20%) -- the regime that matters
    # for matched-accuracy AMR. Accidental sign-cancellation zeros at ~100% baseline are not real wins for either.
    usable = [(ad, bc, rt) for (nm, K, ad, bc, rt, _) in rows_summary if ad < 0.20 and bc < 0.20]
    if usable:
        ru = np.array([x[2] for x in usable])
        print(f"\n  RESTRICTED to BOTH-usable (adj<20% AND cheap<20%): n={len(usable)}")
        print(f"    median ratio = {np.median(ru):.3f}x  geomean = {np.exp(np.mean(np.log(ru))):.3f}x  "
              f"adjoint-wins = {(ru < 1.0).mean():.0%}  decisive(<0.5) = {(ru < 0.5).mean():.0%}")
    # robustly-decisive adjoint regime?
    decisive = [(nm, K, ad, bc, rt) for (nm, K, ad, bc, rt, _) in rows_summary if rt < 0.5]
    print(f"\n  CASES where adjoint DECISIVELY beats best cheap (>2x, i.e. ratio<0.5):")
    if decisive:
        for nm, K, ad, bc, rt in decisive:
            print(f"    {nm:>22} K={K:>3}: adjoint={ad:.2%} vs cheap={bc:.2%}  -> {rt:.3f}x")
    else:
        print("    NONE.")
    # per-config best-cheap winner tally
    from collections import Counter
    tally = Counter(w for *_, w in rows_summary)
    print(f"\n  Which cheap proxy was best, tally: {dict(tally)}")

    # ROBUSTNESS: per config, does adjoint win >2x at EVERY budget? (a robust decisive win, not a lucky budget)
    from collections import defaultdict
    bycfg = defaultdict(list)
    for nm, K, ad, bc, rt, _ in rows_summary:
        bycfg[nm].append(rt)
    print(f"\n  ROBUST decisive adjoint win (ratio<0.5 at ALL budgets in a config)?")
    any_robust = False
    for nm, rts in bycfg.items():
        robust = all(r < 0.5 for r in rts)
        if robust:
            any_robust = True
            print(f"    {nm}: YES  ratios={['%.2f'%r for r in rts]}")
    if not any_robust:
        print("    NONE -- every config has at least one budget where cheap ties or beats the adjoint.")
        # show how close the best config got
        best_cfg = min(bycfg.items(), key=lambda kv: max(kv[1]))
        print(f"    closest config '{best_cfg[0]}' worst-budget ratio = {max(best_cfg[1]):.2f}x "
              f"(ratios {['%.2f'%r for r in best_cfg[1]]})")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())

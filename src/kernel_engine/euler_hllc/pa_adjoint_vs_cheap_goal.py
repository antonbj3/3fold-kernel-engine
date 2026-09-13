#!/usr/bin/env python3
"""AUDIT (replication #5): Goal-oriented AMR for a POINT QoI y=u(P) of -div(k grad u)=f.
Does a CHEAP indicator (NO adjoint solve) match/beat the textbook adjoint indicator |grad u|*|z|
at MATCHED DOF? Where, if ever, does the PDE adjoint DECISIVELY (>2x, robust) beat every cheap proxy?

MY OWN DESIGN (deliberately different from siblings):
  Block-AMR refinement on an N x N TPFA grid. Fix a fine-block BUDGET K; each strategy ranks blocks by an
  indicator and keeps the top-K fine (rest coarsened to one cell). Measure |u_AMR(P) - u_ref(P)|/|u_ref(P)|
  vs an all-fine reference. Lower = better.

  Indicators compared (all use ONE forward solve u; only ADJOINT pays a 2nd solve):
    adj   = max-block( |grad u| * |z| )          z solves L^T z = g_P   (TEXTBOOK, costs extra solve)
    grad  = max-block( |grad u| )                (bare solution-based)
    eucl  = max-block( |grad u| * w_eucl(P) )    w = 1/(1+(r/L)^2)  euclidean falloff to P
    geo   = max-block( |grad u| * w_geo(P) )     w from GEODESIC dist to P through resistivity 1/k (Dijkstra)
    geoE  = max-block( |grad u| * exp(-d_geo/Lg))  exponential geodesic falloff

  Geometries (heterogeneity GEOMETRY is the crux; vary it):
    homo     : k uniform.
    barrier  : a low-k WALL placed straddling the straight line between source and P (rerouting matters).
    channel  : a high-k CHANNEL connecting source region toward P (preferential path).
    blobs    : random low/high-k blobs (seeded).
  Contrast sweep (k_lo, k_hi ratios) and seeds. Also a config where source is a FAR DECOY (off the line to P).

  Numbers over narration. If cheap matches adjoint, say so. If adjoint decisively wins somewhere, say where.
  python3 pa_adjoint_vs_cheap_goal.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('amr_poisson',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys, os, heapq
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from amr_octree_fv import tpfa_solve

N = 96
BLK = 8
NB = N // BLK


def gauss_source(n, cx, cy, s=2.0):
    a = np.arange(n)
    xx, yy = np.meshgrid(a, a, indexing="ij")
    f = np.exp(-(((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * s ** 2)))
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


def qoi_at(field, n, px, py):
    return float(field[int(px * n), int(py * n)])


def make_k(geom, contrast, rng):
    """Return k field (N,N). contrast = k_hi/k_lo ; base k=1 except features."""
    k = np.ones((N, N))
    if geom == "homo":
        return k
    if geom == "barrier":
        # low-k vertical-ish wall straddling the line between source (~0.2,0.2) and P (~0.8,0.8).
        # Place a low-k diagonal-perpendicular wall in the middle with a small gap (forces rerouting).
        lo = 1.0 / contrast
        # wall: band around the anti-diagonal midline x+y in [0.95,1.05], with a gap near center
        a = (np.arange(N) + 0.5) / N
        xx, yy = np.meshgrid(a, a, indexing="ij")
        s = xx + yy
        wall = (np.abs(s - 1.0) < 0.06)
        gap = (np.abs(xx - yy) < 0.12)  # a passage along the main diagonal
        wall = wall & (~gap)
        k[wall] = lo
        return k
    if geom == "channel":
        hi = contrast
        a = (np.arange(N) + 0.5) / N
        xx, yy = np.meshgrid(a, a, indexing="ij")
        # high-k channel along the main diagonal connecting source->P
        chan = (np.abs(xx - yy) < 0.07)
        k[chan] = hi
        return k
    if geom == "blobs":
        lo = 1.0 / np.sqrt(contrast); hi = np.sqrt(contrast)
        a = (np.arange(N) + 0.5) / N
        xx, yy = np.meshgrid(a, a, indexing="ij")
        for _ in range(6):
            cx, cy = rng.uniform(0.1, 0.9, 2)
            r = rng.uniform(0.06, 0.14)
            val = hi if rng.random() < 0.5 else lo
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
            k[m] = val
        return k
    raise ValueError(geom)


def geodesic_dist(k, px, py):
    """Dijkstra shortest path from cell P to all cells, edge cost = resistivity 1/k * h (4-neighbour).
    High-k => low cost => 'close' geodesically. Returns (N,N) distance in physical-ish units."""
    h = 1.0 / N
    res = (1.0 / k) * h  # per-cell resistivity weight
    src = (int(px * N), int(py * N))
    INF = np.inf
    dist = np.full((N, N), INF)
    dist[src] = 0.0
    pq = [(0.0, src[0], src[1])]
    visited = np.zeros((N, N), bool)
    while pq:
        d, i, j = heapq.heappop(pq)
        if visited[i, j]:
            continue
        visited[i, j] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N and not visited[ni, nj]:
                # cost to traverse into neighbour = average resistivity of the two cells (face)
                w = 0.5 * (res[i, j] + res[ni, nj])
                nd = d + w
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(pq, (nd, ni, nj))
    return dist


def blockmax(field):
    return field.reshape(NB, BLK, NB, BLK).max(axis=(1, 3))


def run_config(geom, contrast, fx, fy, px, py, seed):
    rng = np.random.default_rng(seed)
    k = make_k(geom, contrast, rng)
    f = gauss_source(N, fx, fy, s=2.0)
    full = np.arange(N * N).reshape(N, N)
    u_ref = tpfa_solve(full, k, f, N)
    y_true = qoi_at(u_ref, N, px, py)
    if abs(y_true) < 1e-12:
        return None

    gy, gx = np.gradient(u_ref)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # adjoint: L^T z = g_P. L is symmetric here (TPFA), so z solves same operator with rhs g.
    g = gauss_source(N, px, py, s=1.5)
    z = tpfa_solve(full, k, g, N)
    adj = grad * np.abs(z)

    # euclidean weight to P
    a = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(a, a, indexing="ij")
    r = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    L = 0.25
    eucl = grad * (1.0 / (1.0 + (r / L) ** 2))

    # geodesic weight to P through resistivity
    dgeo = geodesic_dist(k, px, py)
    dgeo_n = dgeo / (np.median(dgeo[np.isfinite(dgeo)]) + 1e-12)
    geo = grad * (1.0 / (1.0 + dgeo_n ** 2))
    Lg = 1.0
    geoE = grad * np.exp(-dgeo_n / Lg)

    indicators = {
        "adj": blockmax(adj),
        "grad": blockmax(grad),
        "eucl": blockmax(eucl),
        "geo": blockmax(geo),
        "geoE": blockmax(geoE),
    }

    out = {}
    for K in (4, 8, 16):
        errs = {}
        for name, score in indicators.items():
            keep = np.zeros((NB, NB), bool)
            idx = np.argsort(score.ravel())[::-1][:K]
            keep.ravel()[idx] = True
            rid = regions_from_keep(keep)
            u = tpfa_solve(rid, k, f, N)
            errs[name] = abs(qoi_at(u, N, px, py) - y_true) / abs(y_true)
        # also uniform-spread baseline
        keep = np.zeros((NB, NB), bool)
        idx = np.linspace(0, NB * NB - 1, K).astype(int)
        keep.ravel()[idx] = True
        rid = regions_from_keep(keep)
        u = tpfa_solve(rid, k, f, N)
        errs["uni"] = abs(qoi_at(u, N, px, py) - y_true) / abs(y_true)
        out[K] = errs
    return out


def main():
    print("=" * 100)
    print("ADJOINT vs CHEAP goal-oriented indicators for POINT QoI y=u(P).  err = |u_AMR(P)-u_ref(P)|/|u_ref(P)|")
    print("Lower is better. CHEAP = grad/eucl/geo/geoE (one solve). ADJOINT = adj (extra solve).")
    print("=" * 100)

    # configs: (label, geom, contrast, src(fx,fy), P(px,py), seeds)
    configs = []
    # Homogeneous, source far from P (decoy-ish far source) and source near P
    configs.append(("homo_far", "homo", 1.0, (0.20, 0.20), (0.80, 0.80), [0]))
    configs.append(("homo_near", "homo", 1.0, (0.65, 0.65), (0.80, 0.80), [0]))
    # Barrier between source and P, contrast sweep (rerouting should matter most here)
    for c in (10.0, 100.0, 1000.0):
        configs.append((f"barrier_c{int(c)}", "barrier", c, (0.18, 0.18), (0.82, 0.82), [0]))
    # Channel (high-k preferential path)
    for c in (10.0, 100.0):
        configs.append((f"channel_c{int(c)}", "channel", c, (0.18, 0.18), (0.82, 0.82), [0]))
    # Random blobs, several seeds, two contrasts
    for c in (20.0, 200.0):
        for sd in (1, 2, 3, 4):
            configs.append((f"blobs_c{int(c)}_s{sd}", "blobs", c, (0.20, 0.20), (0.80, 0.80), [sd]))
    # An off-line far decoy with barrier: source at (0.2,0.8), P at (0.8,0.2), barrier between
    for c in (100.0, 1000.0):
        configs.append((f"barrierXdecoy_c{int(c)}", "barrier", c, (0.20, 0.80), (0.82, 0.18), [0]))

    proxies = ["grad", "eucl", "geo", "geoE"]
    all_ratios = []  # best-cheap / adj  (>1 => adjoint better; <1 => cheap better) -- careful with zeros
    win_adj = 0; win_cheap = 0; tie = 0; total = 0
    decisive_cases = []

    hdr = f"  {'config':>22} {'K':>3} | {'adj':>10} {'grad':>10} {'eucl':>10} {'geo':>10} {'geoE':>10} {'uni':>10} | {'best_cheap':>10} {'ratio(bc/adj)':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr)))
    for label, geom, contrast, (fx, fy), (px, py), seeds in configs:
        for sd in seeds:
            res = run_config(geom, contrast, fx, fy, px, py, sd)
            if res is None:
                print(f"  {label:>22}  -- QoI ~0, skipped")
                continue
            for K, errs in res.items():
                adj = errs["adj"]
                best_cheap_name = min(proxies, key=lambda p: errs[p])
                best_cheap = errs[best_cheap_name]
                # ratio: how much worse is best-cheap than adjoint. guard tiny denom.
                denom = max(adj, 1e-9)
                ratio = best_cheap / denom
                all_ratios.append(ratio)
                total += 1
                if best_cheap < adj * 0.9:
                    win_cheap += 1
                elif adj < best_cheap * 0.9:
                    win_adj += 1
                    if adj < best_cheap * 0.5:
                        decisive_cases.append((label, K, adj, best_cheap, best_cheap_name, ratio))
                else:
                    tie += 1
                print(f"  {label:>22} {K:>3} | {adj:>10.3%} {errs['grad']:>10.3%} {errs['eucl']:>10.3%} "
                      f"{errs['geo']:>10.3%} {errs['geoE']:>10.3%} {errs['uni']:>10.3%} | "
                      f"{best_cheap:>9.3%}({best_cheap_name[:4]}) {ratio:>11.2f}x")

    arr = np.array(all_ratios)
    print("\n" + "=" * 100)
    print(f"  total comparisons = {total}")
    print(f"  ratio = best_cheap_err / adjoint_err   (>1 adjoint better, <1 cheap better)")
    print(f"    median ratio = {np.median(arr):.3f}   mean = {np.mean(arr):.3f}")
    print(f"    p25 = {np.percentile(arr,25):.3f}  p75 = {np.percentile(arr,75):.3f}  "
          f"min = {arr.min():.3f}  max = {arr.max():.3f}")
    print(f"  adjoint WINS (>10% better): {win_adj}/{total} ({win_adj/total:.0%})   "
          f"cheap WINS (>10% better): {win_cheap}/{total} ({win_cheap/total:.0%})   "
          f"tie: {tie}/{total} ({tie/total:.0%})")
    # decisive: adjoint >2x better than best cheap, robustly
    n_decisive_2x = int((arr > 2.0).sum())
    print(f"  cases where adjoint >2x better than BEST cheap: {n_decisive_2x}/{total}")
    if decisive_cases:
        print("  decisive (>2x) cases:")
        for label, K, adj, bc, bcn, ratio in decisive_cases:
            print(f"    {label} K={K}: adj={adj:.3%} best_cheap={bc:.3%}({bcn}) ratio={ratio:.2f}x")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())

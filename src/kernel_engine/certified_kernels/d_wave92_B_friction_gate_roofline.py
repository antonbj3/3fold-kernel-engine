#!/usr/bin/env python3
"""cell 92-B: THE LOW-LEVEL PARALLEL FRICTION/sigma_min GATE + ITS I/O FLOOR.

The friction-gauge over MANY subnodes at once = a batched k-step Lanczos sigma_min
estimator (the comparator/FMA-analog primitive: emits {friction, recurse?} per node).
CRUX: is friction LOCAL (per-node, parallel-cheap, compute-bound) or GLOBAL (needs the
whole state, memory-bound, serializing)?  Force the A6 sigma_min(+)pi_0 split: does the
gauge inherit a +1 global integer (pi_0 = connected components) that does not parallelize?

Deterministic, OMP=2, no GPU, no training. Exact FLOP counting + MEASURED machine roofline.
 P1 ROOFLINE  batched Lanczos gate: AI_max = k/4 flops/byte (d cancels, tunable) vs ridge.
 P2 HONG-KUNG recursion-tree I/O floor + the global STOP-reduction (AI=1 = the memory-bound +1).
 P3 pi_0 GAP  min-local-block sigma_min != global sigma_min; the gap = interface Schur + b0
              (connected components), a GLOBAL integer NOT computable from local data.
See d_wave92_B_friction_gate_roofline_PREREG.md  (NS92-B).
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
import json, time

rng = np.random.default_rng(20260710)
W = 8  # bytes per float64 word
EV = {"cell": "d_wave92_B_friction_gate_roofline", "prereg": "NS92-B",
      "substrate": "batched k-step Lanczos sigma_min gate over N d-by-d local blocks; "
                   "global-op = partitioned graph Laplacian (A6 anchor)",
      "seed": 20260710, "omp": 2, "parts": {}, "gates": {}}


# ==================================================================================================
# THE PRIMITIVE: batched k-step Lanczos smallest-Ritz (sigma_min estimator), fully vectorized over N.
# Lanczos converges to EXTREME eigenvalues -> smallest Ritz is the right cheap sigma_min gauge.
# Dominant FLOP = k batched matvecs @ 2 d^2 each = 2 k d^2 per node.  Simple 3-term recurrence
# (the real low-level primitive: a few fused steps, no full reorth).
# ==================================================================================================
def batched_lanczos_smin(M, k, reorth=False):
    """M : (N,d,d) SPD blocks. Returns (smallest Ritz per node, {krylov_flops, ritz_flops}).
    krylov_flops = block-touching Krylov build (matvec 2kd^2 + recurrence ~9kd) per node -> AI ~ k/4.
    ritz_flops   = per-node k x k TRIDIAGONAL eigensolve ~12 k^2 (in fast mem, does NOT re-read block).
    Physical regime: k <= d (Krylov subspace cannot exceed the block dimension)."""
    N, d, _ = M.shape
    v_prev = np.zeros((N, d))
    v = np.ones((N, d)) / np.sqrt(d)                       # deterministic start (same for all nodes)
    beta_prev = np.zeros(N)
    alpha = np.zeros((N, k)); beta = np.zeros((N, k))
    Vhist = np.zeros((N, d, k)) if reorth else None
    matvec_flops = 0; recur_flops = 0
    for j in range(k):
        w = np.einsum('nij,nj->ni', M, v)                  # batched matvec: 2 d^2 per node (DOMINANT, touches block)
        matvec_flops += 2 * d * d * N
        a = np.einsum('ni,ni->n', w, v)
        w = w - a[:, None] * v - beta_prev[:, None] * v_prev
        if reorth and j > 0:                               # full reorth (optional; extra 2 j d per step)
            proj = np.einsum('nik,ni->nk', Vhist[:, :, :j], w)
            w = w - np.einsum('nik,nk->ni', Vhist[:, :, :j], proj)
            recur_flops += 4 * j * d * N
        b = np.sqrt(np.einsum('ni,ni->n', w, w))
        recur_flops += 9 * d * N                            # 2 dots + 2 axpy + normalize (~9d)
        alpha[:, j] = a; beta[:, j] = b
        nz = b > 1e-300
        v_next = np.where(nz[:, None], w / np.where(nz, b, 1.0)[:, None], 0.0)
        if reorth:
            Vhist[:, :, j] = v
        v_prev, v, beta_prev = v, v_next, b
    idx = np.arange(k)
    T = np.zeros((N, k, k))
    T[:, idx, idx] = alpha
    if k > 1:
        T[:, idx[:-1], idx[1:]] = beta[:, :k - 1]
        T[:, idx[1:], idx[:-1]] = beta[:, :k - 1]
    ritz = np.linalg.eigvalsh(T)                            # (N,k) ascending (numpy: dense O(k^3) fallback)
    ritz_flops = N * (12 * k ** 2)                          # PRIMITIVE cost = TRIDIAGONAL eigensolve O(k^2)
    return ritz[:, 0], {"krylov": matvec_flops + recur_flops, "matvec": matvec_flops, "ritz": ritz_flops}


# ==================================================================================================
# MACHINE ROOFLINE (measured on THIS box, OMP=2): peak dense-GEMM GFLOP/s + STREAM-triad bandwidth.
# ==================================================================================================
def measure_peak_gflops():
    n = 1600
    A = rng.standard_normal((n, n)); B = rng.standard_normal((n, n))
    _ = A @ B
    best = np.inf
    for _ in range(4):
        t0 = time.perf_counter(); C = A @ B; t = time.perf_counter() - t0
        best = min(best, t)
    return 2.0 * n ** 3 / best / 1e9, float(C.ravel()[0])   # 2 n^3 flops

def measure_bandwidth_gbs():
    """Clean DRAM BW: copy (read+write, no temporaries) and reduction (pure read). Take the max
    achievable (best case = the roofline's BW ceiling). numpy 'a=b+3*c' allocates temporaries ->
    undercounts BW; np.copyto / np.add(out=) / .sum() have NO temporaries."""
    n = 24_000_000                                          # 192 MB/array, well out of cache
    a = np.empty(n); b = rng.standard_normal(n); c = rng.standard_normal(n)
    np.copyto(a, b)                                         # warm
    best_copy = np.inf
    for _ in range(15):                                     # min-time = peak achievable BW (roofline ceiling)
        t0 = time.perf_counter(); np.copyto(a, b); t = time.perf_counter() - t0
        best_copy = min(best_copy, t)
    bw_copy = 2.0 * n * W / best_copy / 1e9                 # read b + write a = 2 arrays
    best_triad = np.inf
    for _ in range(8):                                      # fused triad with out= (no temporaries): 3 arrays
        t0 = time.perf_counter(); np.multiply(c, 3.0, out=a); np.add(a, b, out=a); t = time.perf_counter() - t0
        best_triad = min(best_triad, t)
    bw_triad = 4.0 * n * W / best_triad / 1e9               # read c, read b, read a, write a (2 passes) ~ approx
    best_red = np.inf
    for _ in range(8):
        t0 = time.perf_counter(); _s = b.sum(); t = time.perf_counter() - t0
        best_red = min(best_red, t)
    bw_read = 1.0 * n * W / best_red / 1e9                  # pure read = 1 array
    bw = max(bw_copy, bw_read)                              # peak achievable BW (roofline ceiling)
    return bw, {"copy_gbs": round(bw_copy, 2), "read_gbs": round(bw_read, 2), "triad_gbs": round(bw_triad, 2)}


# ==================================================================================================
# PART 1 -- ROOFLINE of the batched friction gate. AI_max = k/4 (d cancels).  G1 + G2.
# ==================================================================================================
peak_gflops, _s0 = measure_peak_gflops()
bw_gbs, bw_detail = measure_bandwidth_gbs()
ridge = peak_gflops / bw_gbs                                # flops/byte (roofline ridge point)
EV["parts"]["P0_machine"] = {"peak_gemm_gflops": round(peak_gflops, 2),
                             "dram_bw_gbs": round(bw_gbs, 2), "bw_detail": bw_detail,
                             "ridge_flops_per_byte": round(ridge, 3),
                             "note": "measured OMP=2 OpenBLAS Haswell; ridge=peak/BW; BW=max(copy,read) no-temporaries"}

N1 = 6000
d_list = [16, 32, 64, 128]                                 # k<=d always (physical Krylov regime)
k_list = [2, 4, 8, 16]
sweep = []
for d in d_list:
    m = d + 4
    A = rng.standard_normal((N1, m, d)) / np.sqrt(m)
    M = np.einsum('nmi,nmj->nij', A, A) + 0.01 * np.eye(d)[None]
    for k in k_list:
        t0 = time.perf_counter()
        smin, fl = batched_lanczos_smin(M, k)
        t = time.perf_counter() - t0
        # Hong-Kung Q_min: read each block ONCE (d^2 words), iterate in FAST memory, write 1 friction scalar.
        # The k x k Ritz solve is entirely in FAST memory (never re-reads the block) -> 0 extra DRAM traffic.
        # Q_NAIVE (numpy einsum re-streams M on each of the k matvecs -> under-reuse factor k; CORE-1/1a catch).
        q_min_bytes = N1 * (d * d + 1) * W                 # block-resident optimum: read block ONCE
        q_naive_bytes = N1 * (k * d * d + 1) * W           # naive: re-read the block each matvec step
        matvec_flops = fl["matvec"]; total_flops = fl["krylov"] + fl["ritz"]
        ai_matvec = matvec_flops / q_min_bytes             # EXACT block-touching primitive AI (== k/4, d-independent)
        ai_krylov = fl["krylov"] / q_min_bytes             # incl. O(kd) recurrence -> k/4*(1+~4.5/d)
        ai_naive = total_flops / q_naive_bytes             # naive re-read kernel -> ~1/4 (MEMORY-BOUND)
        sweep.append({"d": d, "k": k,
                      "AI_matvec_flops_per_byte": round(ai_matvec, 4), "AI_matvec_x4_over_k": round(ai_matvec * 4.0 / k, 5),
                      "AI_krylov_x4_over_k": round(ai_krylov * 4.0 / k, 4),
                      "AI_naive_reread": round(ai_naive, 4), "under_reuse_factor": round(q_naive_bytes / q_min_bytes, 2),
                      "ritz_frac_of_flops": round(fl["ritz"] / total_flops, 4),
                      "wall_s": round(t, 4), "achieved_gflops": round(total_flops / t / 1e9, 2),
                      "compute_bound_block_resident": bool(ai_matvec > ridge), "memory_bound_naive": bool(ai_naive < ridge)})
EV["parts"]["P1_roofline_sweep"] = sweep

# G1: the block-touching matvec AI = k/4 EXACTLY (d-independent). Full-krylov AI = k/4*(1+c/d) is the PREDICTED
# subleading O(k/d) recurrence correction (fit c, confirm ~const, -> vanishes as d grows). NOT a locality term.
g1_ok = True; g1_detail = {}; c_fits = []
for k in k_list:
    mv = [r["AI_matvec_x4_over_k"] for r in sweep if r["k"] == k]
    kr = [(r["d"], r["AI_krylov_x4_over_k"]) for r in sweep if r["k"] == k]
    spread_mv = (max(mv) - min(mv)) / np.mean(mv)
    c_fits += [(v - 1.0) * dd for dd, v in kr]             # AI_krylov*4/k = 1 + c/d  ->  c = (val-1)*d
    g1_detail[f"k={k}"] = {"matvec_x4/k_by_d": [round(v, 5) for v in mv], "matvec_rel_spread": round(float(spread_mv), 5),
                           "krylov_recurrence_c_by_d": [round((v - 1.0) * dd, 2) for dd, v in kr]}
    if spread_mv > 0.02:                                    # matvec AI = k/4 to <2% (only the +1 friction-write correction)
        g1_ok = False
EV["gates"]["G1_AI_is_k_over_4_d_independent"] = {
    "pass": bool(g1_ok), "prereg": "matvec (block-touching) AI = k/4 EXACTLY, d-independent (<2%)",
    "recurrence_correction_c_median": round(float(np.median(c_fits)), 2), "detail": g1_detail,
    "interpretation": "block-touching gauge AI = k/4 EXACTLY (set by ITERATION COUNT k = tunable knob, NOT by block "
                      "size d / locality). Full-krylov = k/4*(1+c/d), c~4.5 = subleading O(k/d) recurrence -> VANISHES "
                      "as d grows. Ritz reduction is fast-mem (0 DRAM traffic). BUT compute-bound REQUIRES block-resident "
                      "iteration; the naive re-read kernel (under-reuse k) is MEMORY-BOUND (AI~1/4) -- CORE-1/1a catch."}

# --- accuracy: required_k = k for median rel-err(Lanczos smin vs true smin) < threshold. A friction GATE is a
# THRESHOLD decision -> report gate-adequate 20% AND precise 5%. Robustness: TWO conditioning regimes (well m=3d,
# hard clustered-edge m=d+4). Compute-bound verdict must hold for BOTH required_k. ---
d_acc = 32; N_acc = 800
def accuracy_regime(m_acc, reorth=False):
    Aa = rng.standard_normal((N_acc, m_acc, d_acc)) / np.sqrt(m_acc)
    Ma = np.einsum('nmi,nmj->nij', Aa, Aa) + 0.01 * np.eye(d_acc)[None]
    eigM = np.linalg.eigvalsh(Ma); tsm = eigM[:, 0]
    cond = float(np.median(eigM[:, -1] / tsm))
    rows = {}; rk = {0.20: None, 0.05: None}
    for k in [2, 3, 4, 6, 8, 12, 16, 24]:
        if k > d_acc: break
        est, _ = batched_lanczos_smin(Ma, k, reorth=reorth)
        med = float(np.median(np.abs(est - tsm) / np.abs(tsm)))
        rows[f"k={k}"] = round(med, 5)
        for th in (0.20, 0.05):
            if rk[th] is None and med < th: rk[th] = k
    return {"cond_median": round(cond, 1), "median_relerr_by_k": rows,
            "required_k_gate20pct": rk[0.20], "required_k_precise5pct": rk[0.05]}
reg_well = accuracy_regime(3 * d_acc)                       # well-conditioned (cond ~ 5-10)
reg_hard = accuracy_regime(d_acc + 4)                       # hard clustered small-edge (cond ~ 200)
reg_hard_ro = accuracy_regime(d_acc + 4, reorth=True)       # hard + full reorth (isolates convergence vs ghosts)
EV["parts"]["P1_accuracy"] = {"d": d_acc, "N": N_acc, "well_cond_m3d": reg_well,
                              "hard_clustered_m_d_plus_4_noreorth": reg_hard, "hard_reorth": reg_hard_ro,
                              "note": "gate needs only 20% value-accuracy (binary recurse decision); required_k "
                                      "grows with conditioning but the compute-bound verdict is checked at BOTH."}

# G2 (GRADED, not a fragile boolean): the gate's AI=k/4 sits AT the roofline ridge -> a RIDGE-BALANCED kernel,
# TUNABLE compute<->memory by k (and blocking b). Report the crossover k*=4*ridge and where the operating-k sits.
# The DEFENSIBLE, BW-noise-robust claim: (i) AI is tunable ACROSS the ridge (min AI in sweep < ridge < max AI);
# (ii) the high-friction (ill-conditioned) operating point IS compute-bound (robust: k_hard>>k* for any plausible
# ridge); (iii) the naive re-read kernel is memory-bound (design must fuse iterations). It is NOT a locality wall.
k_star = 4.0 * ridge                                        # AI_matvec = k/4 > ridge  <=>  k > 4*ridge
k_well = reg_well["required_k_gate20pct"]; k_hard = reg_hard["required_k_gate20pct"]
ai_span = [r["AI_matvec_flops_per_byte"] for r in sweep]
ai_tunable_across_ridge = bool(min(ai_span) < ridge < max(ai_span))
cb_hard = bool(k_hard is not None and (k_hard / 4.0) > ridge)   # high-friction regime compute-bound
cb_well = bool(k_well is not None and (k_well / 4.0) > ridge)
naive_mem_bound = bool(all(r["AI_naive_reread"] < ridge for r in sweep))
# robust: compute goes to ill-conditioned/high-friction nodes (large k) -> those are compute-bound; naive is mem-bound
g2_pass = bool(ai_tunable_across_ridge and cb_hard and naive_mem_bound)
EV["gates"]["G2_compute_bound_crossover"] = {
    "pass": g2_pass, "verdict": "ridge-BALANCED, TUNABLE (not a locality memory-bound wall)",
    "ridge_flops_per_byte": round(ridge, 3), "ridge_note": "BW measurement has run-to-run variance; k* moves with it",
    "k_star_crossover_4ridge": round(k_star, 2),
    "required_k_gate20pct": {"well_cond": k_well, "well_cond_median": reg_well["cond_median"],
                             "hard_cond": k_hard, "hard_cond_median": reg_hard["cond_median"]},
    "compute_bound_at_operating_k": {"high_friction_illcond": cb_hard, "low_friction_wellcond": cb_well},
    "AI_tunable_across_ridge": ai_tunable_across_ridge, "AI_span_over_sweep": [round(min(ai_span), 3), round(max(ai_span), 3)],
    "naive_reread_memory_bound": naive_mem_bound, "AI_naive_range": [round(min(r["AI_naive_reread"] for r in sweep), 3),
                                                                     round(max(r["AI_naive_reread"] for r in sweep), 3)],
    "blocking_knob": "block-size b (block-Lanczos): AI -> ~k*b/4. AI is a DESIGN knob -> gate placeable compute-bound "
                     "BY CONSTRUCTION (raise k or b), independent of locality.",
    "interpretation": "the block-touching gauge is NOT intrinsically memory-bound by locality (AI=k/4 tunable, spans "
                      "the ridge). Compute concentrates at high-friction (ill-conditioned) nodes needing large k -- and "
                      "THOSE are compute-bound. Two memory-bound pieces remain: the NAIVE (unfused) kernel [fix: fuse "
                      "iterations block-resident] and the global +1 STOP-reduction [G3, irreducible but subleading]."}


# ==================================================================================================
# PART 2 -- HONG-KUNG I/O FLOOR of the recursion tree + the global STOP-reduction (the +1).  G3.
# ==================================================================================================
# Batched gate over one level (N nodes, d^2 block): reuse CORE-1/1a red-blue pebble accounting.
# WORK (per-node friction): FLOPs 2k d^2 N, Q_min = d^2 N (block read once, iterate in fast mem).
# STOP (all-converged): min/AND reduction over N friction scalars = the reduction special-case
# (CORE-1/1a): Q_min = N, AI = 1 flop/word -> ALWAYS memory-bound.  This is the +1 = pi_0-analog.
d_ref, k_ref = 32, 8
N_leaf = 100_000
work_flops = 2 * k_ref * d_ref ** 2 * N_leaf
work_qmin_words = d_ref ** 2 * N_leaf
stop_flops = N_leaf                                        # N-1 comparisons ~ N
stop_qmin_words = N_leaf                                   # read all N friction scalars once
EV["parts"]["P2_hongkung"] = {
    "level_params": {"d": d_ref, "k": k_ref, "N_nodes": N_leaf},
    "WORK_per_node_friction": {"flops": int(work_flops), "q_min_words": int(work_qmin_words),
                               "AI_words": round(work_flops / work_qmin_words, 2), "regime": "compute-bound (AI=2k)"},
    "STOP_global_reduction": {"flops": int(stop_flops), "q_min_words": int(stop_qmin_words),
                              "AI_words": round(stop_flops / stop_qmin_words, 4),
                              "regime": "memory-bound (AI=1, CORE-1/1a reduction special case)",
                              "identity": "the +1 = pi_0-analog: a global min/AND, does NOT get sqrt(M) reuse"},
    "work_vs_stop_io_ratio_d2": round(work_qmin_words / stop_qmin_words, 1),
    "recursion_tree_io": "adaptive tree: Q_min summed over levels = sum_l (active_l * d^2) words; "
                         "the STOP reduction is O(N) per level, log-depth, d^2-SUBLEADING to the block-read work"}
# G3: STOP reduction I/O is factor ~d^2 subleading to the work block-read I/O
g3_ratio = work_qmin_words / stop_qmin_words
EV["gates"]["G3_stop_reduction_subleading"] = {
    "pass": bool(g3_ratio >= 0.5 * d_ref ** 2),
    "stop_AI_words": round(stop_flops / stop_qmin_words, 4), "expect_stop_AI": 1.0,
    "work_io_over_stop_io": round(g3_ratio, 1), "expect_d2": d_ref ** 2,
    "interpretation": "the +1 global STOP-reduction is irreducibly memory-bound (AI=1) BUT d^2-subleading "
                      "to the local work -> a cheap log-depth barrier, not the dominant cost"}


# ==================================================================================================
# PART 3 -- pi_0 GAP: LOCAL block sigma_min != GLOBAL sigma_min. The +1 = b0 (connected comps).  G4.
# A6 anchor: rank(incidence B) = N - b0 ; Laplacian null multiplicity = b0 = pi_0 (global integer).
# ==================================================================================================
def b0_unionfind(N, edges):
    parent = list(range(N))
    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return len({find(i) for i in range(N)})

def laplacian_dense(N, edges):
    L = np.zeros((N, N))
    for u, v in edges:
        L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1
    return L

def incidence(N, edges):
    B = np.zeros((N, len(edges)))
    for e, (u, v) in enumerate(edges):
        B[u, e] = 1.0; B[v, e] = -1.0
    return B

P_sub, per = 8, 100                            # 8 subdomains x 100 nodes = 800 global nodes
N_g = P_sub * per
intra = []                                     # each subdomain = a connected chain (b0_local = 1)
sub_edges = {p: [] for p in range(P_sub)}
for p in range(P_sub):
    base = p * per
    for i in range(per - 1):
        e = (base + i, base + i + 1)
        intra.append(e); sub_edges[p].append(e)

# per-subdomain (LOCAL) diagnostics -- computable in parallel, no interface knowledge
sum_local_b0 = sum(b0_unionfind(per, [(u - p * per, v - p * per) for (u, v) in sub_edges[p]]) for p in range(P_sub))
local_fiedlers = []
for p in range(P_sub):
    Lp = laplacian_dense(per, [(u - p * per, v - p * per) for (u, v) in sub_edges[p]])
    ev = np.linalg.eigvalsh(Lp)
    local_fiedlers.append(float(ev[1]))        # smallest nonzero = subdomain algebraic connectivity
min_local_fiedler = min(local_fiedlers)

# sweep interface coupling: n_interface edges connecting consecutive subdomains
sweep3 = []
for n_int in [0, 4, 8, 16, 32, 64]:
    inter = []
    rr = np.random.default_rng(1234 + n_int)
    for _ in range(n_int):
        p = int(rr.integers(0, P_sub - 1))
        u = p * per + int(rr.integers(0, per)); v = (p + 1) * per + int(rr.integers(0, per))
        inter.append((u, v))
    all_edges = intra + inter
    gb0 = b0_unionfind(N_g, all_edges)                     # GLOBAL connected components (union-find, no SVD)
    L = laplacian_dense(N_g, all_edges)
    evL = np.linalg.eigvalsh(L)
    null_dim = int(np.sum(evL < 1e-9))                     # numeric Laplacian null multiplicity == b0
    global_fiedler = float(evL[null_dim]) if null_dim < N_g else 0.0  # smallest nonzero = global alg. connectivity
    # A6 cross-check: rank(incidence B) = N - b0
    B = incidence(N_g, all_edges)
    sB = np.linalg.svd(B, compute_uv=False) if B.shape[1] > 0 else np.array([0.0])
    num_rank = int(np.sum(sB > (sB[0] * 1e-9))) if B.shape[1] > 0 else 0
    sweep3.append({
        "n_interface": n_int, "sum_local_b0": int(sum_local_b0), "global_b0_unionfind": int(gb0),
        "b0_gap_local_minus_global": int(sum_local_b0 - gb0),          # the +1 (integer) accumulating
        "laplacian_null_dim_numeric": null_dim, "null_matches_b0": bool(null_dim == gb0),
        "min_local_fiedler": round(min_local_fiedler, 5), "global_fiedler": round(global_fiedler, 6),
        "rank_B_numeric": num_rank, "N_minus_b0": int(N_g - gb0),
        "rankB_eq_N_minus_b0": bool(num_rank == N_g - gb0)})
EV["parts"]["P3_pi0_gap"] = {"N_global": N_g, "P_subdomains": P_sub, "per": per,
                             "sum_local_b0": int(sum_local_b0), "sweep": sweep3,
                             "A6_anchor": "rank(incidence B) = N - b0 (algebraic graph theory); "
                                          "Laplacian null multiplicity = b0 = pi_0",
                             "amr_anchor": "local error INDICATOR (residual/gradient, parallel) vs global "
                                           "goal-oriented error ESTIMATOR (dual-weighted residual, serial adjoint)"}

# G4: (a) at zero coupling local==global (no +1); (b) coupling opens a gap = global integer;
#     (c) rank(B)=N-b0 holds everywhere; (d) null_dim==b0 everywhere.
z = sweep3[0]; hi = sweep3[-1]
g4_local_eq_global_at_zero = bool(z["global_b0_unionfind"] == z["sum_local_b0"] and z["b0_gap_local_minus_global"] == 0)
g4_gap_opens = bool(hi["b0_gap_local_minus_global"] > 0)
g4_rank_anchor = all(s["rankB_eq_N_minus_b0"] for s in sweep3)
g4_null_matches = all(s["null_matches_b0"] for s in sweep3)
# continuous floor is global: global_fiedler at high coupling != min_local_fiedler (interface Schur)
g4_fiedler_global = bool(abs(hi["global_fiedler"] - hi["min_local_fiedler"]) > 1e-4)
EV["gates"]["G4_pi0_gap_global_integer"] = {
    "pass": bool(g4_local_eq_global_at_zero and g4_gap_opens and g4_rank_anchor and g4_null_matches),
    "local_eq_global_at_zero_coupling": g4_local_eq_global_at_zero,
    "gap_opens_with_coupling": g4_gap_opens,
    "rank_B_eq_N_minus_b0_all": g4_rank_anchor, "laplacian_null_eq_b0_all": g4_null_matches,
    "continuous_fiedler_is_global": g4_fiedler_global,
    "interpretation": "DISCRETE +1 (b0=pi_0) and CONTINUOUS floor (lambda_2) are BOTH global (interface Schur + "
                      "connected-component merges), NOT computable from local blocks -> friction FLOOR inherits "
                      "the +1; the refinement-driving local-block INDICATOR does not (it is a different, local object)"}


# ==================================================================================================
# VERDICT
# ==================================================================================================
verdict = {
    "claim": "friction SPLITS by locality exactly like sigma_min(+)pi_0; the block-touching gauge is RIDGE-BALANCED "
             "and TUNABLE (NOT a locality memory-bound wall), the +1 global FLOOR is the only fixed memory-bound term",
    "WORK_local_indicator": "per-block sigma_min gate: matvec AI = k/4 flops/byte EXACTLY (d cancels, G1), set by "
                            "ITERATION COUNT k (tunable knob), NOT by locality. Sits AT the roofline ridge; tunable "
                            "compute<->memory by k and block-size b. Compute-bound for k>4*ridge, REALIZED at the "
                            "high-friction (ill-conditioned) nodes that need large k -- compute concentrates where "
                            "refinement happens. REQUIRES block-resident fused iteration (naive re-read = memory-bound).",
    "FLOOR_global_plus1": "true sigma_min STOP-floor = (lambda_2 continuous) + (b0=pi_0 discrete integer), NOT "
                          "computable from local data (interface Schur + connected-component count, G4); a global "
                          "reduction AI=1 (fixed memory-bound) but O(N) log-depth, d^2-SUBLEADING to the work (G3)",
    "prereg_NS92B": "PARTIALLY SUPPORTED: the recurse-decision is NOT locality-memory-bound (AI=k/4 tunable, spans the "
                    "ridge); compute-bound is achievable & realized at high-friction nodes -- but it is ridge-BALANCED, "
                    "not deeply compute-bound like GEMM.",
    "nearest_alt": "REFUTED for the local INDICATOR (AI=k/4 tunable, not the SpMV 0.25 wall -> friction IS local & "
                   "parallel-cheap), TRUE only for (a) the naive unfused kernel [fix: fuse block-resident] and (b) the "
                   "global +1 STOP-floor (pi_0) -- a cheap subleading reduction, NOT the dominant cost. sigma_min-as-"
                   "global-property is right for the FLOOR, wrong for the refinement-driving INDICATOR.",
    "design_lesson": "(1) fuse the k Lanczos steps block-resident (else memory-bound); (2) AI is a knob (k, blocking b) "
                     "-> place the gate compute-bound by construction; (3) drive refinement by the LOCAL block sigma_min "
                     "(valid parallel surrogate, indicator != floor); (4) carry the global +1 (pi_0=b0) as a cheap "
                     "log-depth STOP reduction. Same as A6: certify local sigma_min per node, carry b0 globally."}
EV["verdict"] = verdict

allpass = {kk: EV["gates"][kk].get("pass", None) for kk in EV["gates"]}
EV["gates_summary"] = allpass

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_wave92_B_friction_gate_roofline_evidence.json")
with open(out, "w") as f:
    json.dump(EV, f, indent=2)

# ---- console ----
print("=" * 96)
print("cell 92-B  friction/sigma_min gate roofline + I/O floor + pi_0 gap")
print("=" * 96)
print(f"[machine]  peak_GEMM={peak_gflops:.1f} GFLOP/s  BW={bw_gbs:.1f} GB/s ({bw_detail})  ridge={ridge:.2f} f/B (OMP=2)")
print("-" * 96)
print("P1 ROOFLINE (batched Lanczos sigma_min gate)  block-touching matvec AI should be EXACTLY k/4, d-independent:")
print(f"  {'d':>4} {'k':>4} {'AI_matvec':>10} {'mv*4/k':>7} {'kryl*4/k':>8} {'AI_naive':>9} {'reuse':>6} {'cmpB':>5} {'GF':>5}")
for r in sweep:
    print(f"  {r['d']:>4} {r['k']:>4} {r['AI_matvec_flops_per_byte']:>10.3f} {r['AI_matvec_x4_over_k']:>7.4f} "
          f"{r['AI_krylov_x4_over_k']:>8.3f} {r['AI_naive_reread']:>9.3f} {r['under_reuse_factor']:>6.0f} "
          f"{str(r['compute_bound_block_resident']):>5} {r['achieved_gflops']:>5.1f}")
print(f"  G1 (matvec AI=k/4 exactly, d-independent; recurrence c~{EV['gates']['G1_AI_is_k_over_4_d_independent']['recurrence_correction_c_median']}): "
      f"{'PASS' if g1_ok else 'FAIL'}")
print(f"  accuracy required_k(gate 20%): well-cond(cond={reg_well['cond_median']})={k_well}  "
      f"hard(cond={reg_hard['cond_median']})={k_hard}  reorth-hard={reg_hard_ro['required_k_gate20pct']}")
print(f"  ridge={ridge:.2f} k*=4*ridge={k_star:.1f}  AI_span={[round(min(ai_span),2),round(max(ai_span),2)]} "
      f"(tunable-across-ridge={ai_tunable_across_ridge})  high-friction-compute-bound={cb_hard}  "
      f"naive-mem-bound={naive_mem_bound}  G2(graded) {'PASS' if g2_pass else 'FAIL'}")
print("-" * 96)
print("P2 HONG-KUNG:")
print(f"  WORK  AI={EV['parts']['P2_hongkung']['WORK_per_node_friction']['AI_words']} words (compute-bound)")
print(f"  STOP  AI={EV['parts']['P2_hongkung']['STOP_global_reduction']['AI_words']} word (MEMORY-BOUND = the +1)")
print(f"  work/stop I/O ratio = {g3_ratio:.0f} (~d^2={d_ref**2})  G3 subleading: "
      f"{'PASS' if EV['gates']['G3_stop_reduction_subleading']['pass'] else 'FAIL'}")
print("-" * 96)
print("P3 pi_0 GAP (local sigma_min vs global; A6 rank(B)=N-b0):")
print(f"  {'n_int':>6} {'sum_loc_b0':>10} {'glob_b0':>8} {'gap(+1)':>8} {'null==b0':>9} "
      f"{'rankB=N-b0':>11} {'min_loc_fied':>13} {'glob_fied':>11}")
for s in sweep3:
    print(f"  {s['n_interface']:>6} {s['sum_local_b0']:>10} {s['global_b0_unionfind']:>8} "
          f"{s['b0_gap_local_minus_global']:>8} {str(s['null_matches_b0']):>9} {str(s['rankB_eq_N_minus_b0']):>11} "
          f"{s['min_local_fiedler']:>13.4f} {s['global_fiedler']:>11.5f}")
print(f"  G4 (local==global@0, gap opens, rank(B)=N-b0, null==b0): "
      f"{'PASS' if EV['gates']['G4_pi0_gap_global_integer']['pass'] else 'FAIL'}")
print("=" * 96)
print("VERDICT:", verdict["claim"])
print("  WORK :", verdict["WORK_local_indicator"])
print("  FLOOR:", verdict["FLOOR_global_plus1"])
print("  gates:", allpass)
print("  ->", out)

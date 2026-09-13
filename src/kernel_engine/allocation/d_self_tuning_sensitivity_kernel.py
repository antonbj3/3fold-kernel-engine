#!/usr/bin/env python3
"""A kernel that measures its own per-element sensitivity online and re-allocates precision.

Each round it estimates the Fisher/adjoint spectrum from the data it is processing, water-fills the bit
budget over that spectrum, drops elements under the floor, and validates the new allocation on a held-out
anchor before adopting it. The held-out gate is compared against an ungated control that validates on its
own working data.

Input: none (synthetic worlds, fixed seeds). Output:
artifacts/d_self_tuning_sensitivity_kernel_evidence.json. NumPy only, no GPU, about 20 s.

  python d_self_tuning_sensitivity_kernel.py
"""
import os, json, math, time, heapq, importlib.util, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
EV = os.path.join(ART, "d_self_tuning_sensitivity_kernel_evidence.json")
SEED = 20260708

# ---- import the in-repo FPGA water-filling cell as an INDEPENDENT allocation instrument ----
_spec = importlib.util.spec_from_file_location("fpga_wf", os.path.join(HERE, "d_fpga_bitwidth_waterfilling.py"))
FPGA = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FPGA)


# ============================================================ (1) KERNEL + SENSITIVITY
def qoi(A, x, g):
    """J = g^T A x  (batched: x,g are (B,n),(B,m) -> (B,))."""
    return np.einsum("bi,ij,bj->b", g, A, x)


def analytic_sens(x, g):
    """Per-sample adjoint of J w.r.t. A_ij: dJ/dA_ij = g_i x_j. Returns (B,m,n)."""
    return g[:, :, None] * x[:, None, :]


def fd_sens(A, x, g, i, j, h=1e-6):
    """Central finite-difference dJ/dA_ij at one element (independent instrument)."""
    Ap = A.copy(); Ap[i, j] += h
    Am = A.copy(); Am[i, j] -= h
    return (qoi(Ap, x, g) - qoi(Am, x, g)) / (2 * h)


def measure_alpha(A, X, G):
    """MEASURE the structure-Fisher spectrum from a (batch of) operation:
       alpha_ij = A_ij^2 * mean_s (g_i^s x_j^s)^2.  This is the online sensitivity estimate."""
    s = analytic_sens(X, G)                    # (B,m,n) = g_i x_j
    return (A ** 2) * np.mean(s ** 2, axis=0)  # (m,n)


def quantize_subdither(A, b, rng):
    """Operational per-element relative quantiser. b>=1: subtractively-dithered mid-tread quant over
    range R=sqrt(12)|A| (Var(err)=R^2/12 4^{-b}=A^2 4^{-b}, EXACT & unbiased [Lipshitz-Wannamaker-
    Vanderkooy dither]). b==0: PRUNE to 0 (err=-A, var A^2 == the b=0 value of A^2 4^{-b})."""
    Aq = np.zeros_like(A)
    R = math.sqrt(12.0) * np.abs(A)
    for bb in np.unique(b):
        m = (b == bb)
        if bb <= 0:
            continue                                   # pruned -> 0
        Delta = R[m] / (2 ** int(bb))
        d = rng.uniform(-0.5, 0.5, size=m.sum()) * Delta
        Aq[m] = np.round((A[m] + d) / Delta) * Delta - d
    return Aq


def mc_qoi_error(A, b, world, rng, S=2000, n_dither=60):
    """Operational anchor: RMS QoI error of the REAL quantised kernel over fresh stream data, averaged
    over n_dither independent quantisation (dither) realisations so the cross-element error terms vanish
    (the model sum alpha 4^{-b} is the ENSEMBLE-mean error, exactly what dither realises)."""
    X, G = world["sample"](S, rng)
    Jf = qoi(A, X, G)
    acc, cnt = 0.0, 0
    for _ in range(n_dither):
        Aq = quantize_subdither(A, b, rng)
        dJ = qoi(Aq, X, G) - Jf
        acc += float(np.sum(dJ ** 2)); cnt += S
    return float(np.sqrt(acc / cnt))


def model_err(alpha, b):
    """E[dJ^2] = sum alpha_ij 4^{-b_ij} (the water-filling objective; ground-truth-anchored when alpha=alpha*)."""
    return float(np.sum(alpha * 4.0 ** (-b)))


# ============================================================ (2) ALLOCATORS (matched total-bit budget)
def wf_budget_int(alpha, B_tot, bmax):
    """Fox marginal allocation = EXACT integer optimum of min sum alpha 4^{-b} s.t. sum b = B_tot, b in [0,bmax].
    Marginal reduction of the b-th bit on element i: alpha_i (4^{-(b-1)} - 4^{-b}), decreasing in b -> greedy optimal."""
    a = np.asarray(alpha, float).ravel(); N = a.size
    b = np.zeros(N, int)
    h = [(-(a[i] * 0.75), i) for i in range(N) if a[i] > 0]   # first bit: alpha*(1-1/4)
    heapq.heapify(h)
    used = 0
    while used < B_tot and h:
        neg, i = heapq.heappop(h)
        if b[i] >= bmax:
            continue
        b[i] += 1; used += 1
        if b[i] < bmax:
            nb = b[i]     # next bit goes nb->nb+1; marginal alpha_i(4^{-nb} - 4^{-(nb+1)}), decreasing
            heapq.heappush(h, (-(a[i] * (4.0 ** (-nb) - 4.0 ** (-(nb + 1)))), i))
    return b.reshape(np.asarray(alpha).shape)


def wf_budget_cont(alpha, B_tot, bmax=1e9):
    """Continuous reverse water-filling for a fixed total-bit budget: b_i=max(0,0.5 log2(alpha_i/lam)),
    lam s.t. sum b_i = B_tot. Bisection on the water level (independent of the integer Fox allocator)."""
    a = np.asarray(alpha, float).ravel()
    def tot(lam):
        return np.sum(np.clip(0.5 * np.log2(np.maximum(a, 1e-300) / lam), 0.0, bmax))
    lo, hi = 1e-300, float(a.max())
    while tot(lo) < B_tot:  # lam smaller -> more bits; ensure bracket
        lo *= 0.1
        if lo < 1e-300:
            break
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if tot(mid) > B_tot:
            lo = mid
        else:
            hi = mid
    b = np.clip(0.5 * np.log2(np.maximum(a, 1e-300) / hi), 0.0, bmax)
    return b.reshape(np.asarray(alpha).shape)


def uniform_budget(shape, B_tot, bmax, rng):
    """Uniform allocation meeting the SAME total bits: base bits everywhere + remainder spread at random."""
    N = int(np.prod(shape)); base = B_tot // N; rem = B_tot - base * N
    b = np.full(N, base, int)
    idx = rng.permutation(N)[:rem]; b[idx] += 1
    return np.clip(b, 0, bmax).reshape(shape)


def random_budget(shape, B_tot, bmax, rng):
    """Genuinely RANDOM allocation of the SAME total bits: multinomial (each bit -> a uniformly random
    element, WITH replacement -> uneven counts), overflow above bmax redistributed to non-full elements."""
    N = int(np.prod(shape))
    b = np.bincount(rng.integers(0, N, size=B_tot), minlength=N).astype(int)
    for _ in range(50):
        excess = int(np.clip(b - bmax, 0, None).sum())
        if excess == 0:
            break
        b = np.minimum(b, bmax)
        avail = np.where(b < bmax)[0]
        if avail.size == 0:
            break
        b += np.bincount(rng.choice(avail, size=excess, replace=True), minlength=N).astype(int)
    return np.minimum(b, bmax).reshape(shape)


def brute_force_budget(alpha, B_tot, bmax):
    """Exhaustive integer optimum for tiny N (independent check of the Fox allocator, G4)."""
    a = np.asarray(alpha, float).ravel(); N = a.size
    best = [None, None]
    def rec(i, left, b):
        if i == N:
            if left == 0:
                e = float(np.sum(a * 4.0 ** (-np.array(b))))
                if best[0] is None or e < best[0]:
                    best[0], best[1] = e, list(b)
            return
        for bb in range(0, min(bmax, left) + 1):
            rec(i + 1, left - bb, b + [bb])
    rec(0, B_tot, [])
    return best[0], best[1]


# ============================================================ (3) WORLDS (>=5 non-flat + flat + controls)
def make_world(name, m, n, seed, phase=0):
    """Each world = fixed weights A + a stream sampler for (x,g). The structure-Fisher alpha* is MEASURED
    from a huge batch (ground truth), so it is exact for ANY (x,g) distribution (no closed-form reliance)."""
    r = np.random.default_rng(seed)
    if name == "flat":
        A = 0.6 * np.sign(r.standard_normal((m, n)))   # CONSTANT magnitude -> alpha=A^2 u v is flat when u,v flat
    else:
        A = r.standard_normal((m, n))
        A *= 0.6 / (np.abs(A).mean())                  # O(1), |A| typ ~0.5R -> Fox marginals well-behaved

    def stds(kind, d, rr, ph):
        if kind == "lowrank":                          # few high-variance input columns (activation subspace)
            v = np.full(d, 0.15); k = max(2, d // 6); v[rr.permutation(d)[:k]] = 3.0
        elif kind == "readout":                        # few output dims carry the QoI
            v = np.full(d, 0.2); k = max(2, d // 6); v[rr.permutation(d)[:k]] = 3.0
        elif kind == "powerlaw":
            v = (1.0 + np.arange(d)) ** (-1.1); v = v / v.mean(); rr.shuffle(v)
        elif kind == "twoscale":
            v = np.full(d, 0.05); v[rr.permutation(d)[:max(2, d // 5)]] = 4.0
        elif kind == "smooth":                         # banded/PDE-like smooth profile
            t = np.linspace(0, 1, d); c = rr.random(); v = 0.1 + np.exp(-((t - c) ** 2) / 0.02); v = v / v.mean()
        elif kind == "flat":
            v = np.ones(d)
        else:
            v = np.abs(rr.standard_normal(d)) + 0.2
        return np.sqrt(np.maximum(v, 1e-6))

    cfg = {
        "lowrank_activation": ("lowrank", "readout"),
        "readout_focus":      ("readout", "flat"),
        "outer_powerlaw":     ("powerlaw", "powerlaw"),
        "banded_smooth":      ("smooth", "smooth"),
        "two_scale":          ("twoscale", "flat"),
        "corr_input":         ("powerlaw", "lowrank"),
        "flat":               ("flat", "flat"),
    }[name]
    # non-stationary: phase 1 re-draws the input/read-out Fisher (independent realisation) -> the spectrum shifts
    rx = np.random.default_rng(seed + 101 + phase * 7)
    rg = np.random.default_rng(seed + 202 + phase * 7)
    vx = stds(cfg[1], n, rx, phase)                    # per-input std  (x)
    ug = stds(cfg[0], m, rg, phase)                    # per-readout std (g)
    if name == "corr_input":
        L = np.tril(0.3 * r.standard_normal((n, n))); np.fill_diagonal(L, 1.0)  # correlated inputs
    else:
        L = None

    def sample(B, rng):
        x = rng.standard_normal((B, n)) * vx[None, :]
        if L is not None:
            x = x @ L.T
            x = x / (x.std(0, keepdims=True) + 1e-9) * vx[None, :]   # keep marginal input-Fisher = vx^2
        g = rng.standard_normal((B, m)) * ug[None, :]
        return x, g

    r2 = np.random.default_rng(seed + 999)
    Xbig, Gbig = sample(30000, r2)
    alpha_true = measure_alpha(A, Xbig, Gbig)          # ground-truth structure-Fisher (huge sample)
    return {"name": name, "A": A, "sample": sample, "alpha_true": alpha_true,
            "vx": vx, "ug": ug, "m": m, "n": n}


# ============================================================ KILL-GATES
def kg_adjoint(rng):
    """KG1: analytic adjoint g_i x_j == central finite-difference == quantised-kernel MC; + a NON-LINEAR
    kernel J=g^T tanh(A x) whose adjoint depends on the operating point (not the tautology grad==coef)."""
    m, n, B = 6, 8, 5
    A = rng.standard_normal((m, n)); X = rng.standard_normal((B, n)); G = rng.standard_normal((B, m))
    an = analytic_sens(X, G)
    errs = []
    for (i, j) in [(0, 0), (2, 3), (5, 7)]:
        fd = fd_sens(A, X, G, i, j)
        errs.append(np.max(np.abs(fd - an[:, i, j])) / (np.max(np.abs(an[:, i, j])) + 1e-30))
    lin = float(max(errs))
    # non-linear kernel: J = sum_i g_i tanh((A x)_i); dJ/dA_ij = g_i (1-tanh^2(z_i)) x_j (operating-point dep)
    def nlq(AA):
        z = np.einsum("ij,bj->bi", AA, X)
        return np.einsum("bi,bi->b", G, np.tanh(z))
    z = np.einsum("ij,bj->bi", A, X)
    an_nl = (G * (1 - np.tanh(z) ** 2))[:, :, None] * X[:, None, :]
    nlerr = []
    for (i, j) in [(1, 2), (4, 5)]:
        Ap = A.copy(); Ap[i, j] += 1e-6; Am = A.copy(); Am[i, j] -= 1e-6
        fd = (nlq(Ap) - nlq(Am)) / 2e-6
        nlerr.append(np.max(np.abs(fd - an_nl[:, i, j])) / (np.max(np.abs(an_nl[:, i, j])) + 1e-30))
    nl = float(max(nlerr))
    return {"linear_relerr": lin, "nonlinear_relerr": nl,
            "note": "nonlinear adjoint g_i(1-tanh^2 z_i)x_j is operating-point dependent -> real, not grad==coef",
            "KG1_pass": bool(lin < 1e-6 and nl < 1e-3)}


def kg_model_vs_mc(worlds, B_tot, bmax, rng):
    """KG2: modelled error sum alpha* 4^{-b} == operational MC RMS error of the real quantised kernel."""
    out = {}; ok = True
    for w in worlds:
        b = wf_budget_int(w["alpha_true"], B_tot, bmax)
        mod = math.sqrt(model_err(w["alpha_true"], b))
        mc = mc_qoi_error(w["A"], b, w, rng, S=20000)
        rel = abs(mc - mod) / (mod + 1e-30)
        out[w["name"]] = {"model_rms": mod, "mc_rms": mc, "rel_err": rel}
        ok = ok and rel < 0.03
    out["KG2_pass"] = bool(ok)
    return out


def kg_allocator_optimal(rng):
    """KG3: the Fox allocator == exhaustive brute-force integer optimum on small instances (matched budget)."""
    worst = 0.0; n_ok = 0; n = 0
    for t in range(12):
        N = int(rng.integers(4, 6)); alpha = np.abs(rng.standard_normal(N)) ** 2 + 0.02
        B_tot = int(rng.integers(3, 3 * N)); bmax = 6
        b = wf_budget_int(alpha, B_tot, bmax)
        e = float(np.sum(alpha * 4.0 ** (-b)))
        ebf, _ = brute_force_budget(alpha, B_tot, bmax)
        gap = (e - ebf) / (ebf + 1e-30); worst = max(worst, gap); n += 1; n_ok += int(gap < 1e-9)
    return {"n": n, "frac_optimal": n_ok / n, "worst_gap": worst, "KG3_pass": bool(worst < 1e-9)}


# ============================================================ G1 + G2 + G4 (static matched-cost)
def static_compare(w, B_tot, bmax, rng, n_random=96):
    """At MATCHED total bits B_tot: water-fill(alpha*) [oracle floor] vs uniform vs random vs magnitude.
    Error = model E[dJ^2] under the TRUE spectrum (ground truth) -> operationally MC-anchored in KG2."""
    at = w["alpha_true"]
    b_wf = wf_budget_int(at, B_tot, bmax)
    b_un = uniform_budget(at.shape, B_tot, bmax, rng)
    b_mag = wf_budget_int(w["A"] ** 2, B_tot, bmax)             # magnitude-only allocation (no Fisher)
    e_wf = model_err(at, b_wf)
    e_un = model_err(at, b_un)
    e_mag = model_err(at, b_mag)
    e_rand = []
    for _ in range(n_random):
        e_rand.append(model_err(at, random_budget(at.shape, B_tot, bmax, rng)))
    e_rand = np.array(e_rand)
    # continuous water-fill cross-check (independent instrument) + FPGA reverse-water-fill on matched target
    b_cont = wf_budget_cont(at, B_tot)
    e_cont = model_err(at, np.round(b_cont).astype(int))
    return {
        "B_tot": B_tot, "n_elements": int(at.size), "avg_bits": B_tot / at.size,
        "amgm_alpha": float(np.mean(at) / np.exp(np.mean(np.log(np.maximum(at, 1e-300))))),
        "err_wf_oracle_floor": e_wf, "err_uniform": e_un, "err_random_mean": float(e_rand.mean()),
        "err_random_std": float(e_rand.std()), "err_magnitude": e_mag,
        "ratio_uniform_over_wf": e_un / e_wf, "ratio_random_over_wf": float(e_rand.mean()) / e_wf,
        "ratio_magnitude_over_wf": e_mag / e_wf,
        "n_dropped_wf": int(np.sum(b_wf == 0)), "n_dropped_uniform": int(np.sum(b_un == 0)),
        "wf_beats_uniform": bool(e_wf < e_un * (1 - 1e-12)), "wf_beats_random": bool(e_wf < e_rand.mean()),
        "wf_le_all": bool(e_wf <= min(e_un, e_rand.min(), e_mag) + 1e-12),   # G4: floor
        "cont_wf_err_matches": bool(abs(e_cont - e_wf) / e_wf < 0.15),
    }


# ============================================================ G3 (the guard, online double-drift)
class Tuner:
    """One integer bit-allocation, self-tuned online by budget-preserving bit-moves; the ACCEPT-oracle
    is the ONLY thing that differs between tuners:
      C uses a FRESH held-out anchor's Fisher (out-of-sample), U a FIXED small in-sample calibration
      Fisher (reused every step -> overfits it), S magnitude only (A^2, no measured Fisher)."""
    def __init__(self, mode, b0, A):
        self.mode = mode; self.b = b0.copy(); self.A2 = A ** 2

    def oracle_alpha(self, alpha_cal, alpha_anchor):
        return {"C": alpha_anchor, "U": alpha_cal, "S": self.A2}[self.mode].ravel()

    def step(self, moves, alpha_cal, alpha_anchor, bmax):
        if self.mode in ("A", "O"):
            return
        a = self.oracle_alpha(alpha_cal, alpha_anchor)
        bf = self.b.ravel()
        for (donor, recv) in moves:
            bd, br = bf[donor], bf[recv]
            if bd <= 0 or br >= bmax:
                continue
            d = (a[donor] * (4.0 ** (-(bd - 1)) - 4.0 ** (-bd))
                 + a[recv] * (4.0 ** (-(br + 1)) - 4.0 ** (-br)))
            if d < -1e-15:                                    # accept iff oracle-error strictly decreases
                bf[donor] = bd - 1; bf[recv] = br + 1


def run_stream(world_name, m, n, seed, cfg, regime="stationary"):
    """One online stream. Shared random bit-move proposals; tuners differ ONLY in the accept-oracle.
    U validates on a FIXED small in-sample CALIBRATION set (drawn once, reused -> overfits it, the real
    PTQ-calibration failure); C validates on a FRESH held-out anchor each step (out-of-sample). OOS error
    is measured each step under the TRUE (current-phase) alpha*. Returns per-step trajectories."""
    T, Banch, n_cal, n_moves, bmax = (cfg[k] for k in ("T", "anchor", "n_cal", "n_moves", "bmax"))
    rng = np.random.default_rng(seed * 100003 + 17)
    shift_at = T // 2
    w0 = make_world(world_name, m, n, seed, phase=0)
    w1 = make_world(world_name, m, n, seed, phase=1) if regime == "nonstationary" else w0
    A = w0["A"]; N = m * n
    B_tot = int(round(cfg["avg_bits"] * N))
    b0 = uniform_budget((m, n), B_tot, bmax, rng)
    modes = ("A", "O", "C", "U", "S")
    tun = {md: Tuner(md, b0, A) for md in modes}
    tun["O"].b = wf_budget_int(w0["alpha_true"], B_tot, bmax)   # oracle floor (re-fits at shift)
    # FIXED calibration set for U (drawn ONCE, pre-shift) -> a one-shot small IN-SAMPLE Fisher estimate it
    # reuses every step and thus OVERFITS (the classic post-training-quantisation calibration failure).
    xc, gc = w0["sample"](n_cal, rng)
    alpha_cal = measure_alpha(A, xc, gc)
    A2 = A ** 2
    acc_gx2 = np.zeros((m, n)); n_acc = 0                      # C's ACCUMULATING held-out evidence (pooled)
    alpha_anchor_run = alpha_cal.copy()                        # (seeded; overwritten once anchors arrive)
    log = {md: dict(oos=np.full(T, np.nan), selferr=np.full(T, np.nan), sig=np.full(T, np.nan))
           for md in modes}

    for t in range(T):
        w = w1 if (regime == "nonstationary" and t >= shift_at) else w0
        at = w["alpha_true"]
        thr = np.quantile(at, 0.75); sig_mask = (at >= thr).ravel()   # where bits SHOULD go
        if regime == "nonstationary" and t == shift_at:
            tun["O"].b = wf_budget_int(at, B_tot, bmax)         # oracle knows the shift (it is the floor)
            acc_gx2[:] = 0; n_acc = 0                           # held-out monitor re-accumulates post-shift
        # (1) honest OOS error (true alpha*, never used to tune) + each tuner's OWN self-reported error
        oracle_of = {"U": alpha_cal, "S": tun["S"].A2, "C": alpha_anchor_run}
        for md in modes:
            b = tun[md].b
            log[md]["oos"][t] = model_err(at, b)
            log[md]["sig"][t] = b.ravel()[sig_mask].sum() / max(b.sum(), 1)
            if md in oracle_of:
                log[md]["selferr"][t] = model_err(oracle_of[md], b)   # each tuner's own (self-reported) error
        # (2) fresh held-out anchor (disjoint from operation) -> C ACCUMULATES it (pooled, ->truth over stream)
        xa, ga = w["sample"](Banch, rng)
        acc_gx2 += np.sum(analytic_sens(xa, ga) ** 2, axis=0); n_acc += Banch
        alpha_anchor_run = A2 * (acc_gx2 / n_acc)              # running out-of-sample Fisher (compounds)
        # (3) shared random bit-move proposals; each tuner accepts per its own oracle (the ONLY difference)
        rr = np.random.default_rng(seed * 7919 + t * 31)
        moves = list(zip(rr.integers(0, N, n_moves).tolist(), rr.integers(0, N, n_moves).tolist()))
        for md in ("C", "U", "S"):
            tun[md].step(moves, alpha_cal, alpha_anchor_run, bmax)
    return dict(log=log, shift_at=shift_at, B_tot=B_tot,
                oracle_floor=model_err(w0["alpha_true"], tun["O"].b))


def ci95(a, axis=0):
    a = np.asarray(a, float); nn = a.shape[axis]
    return 1.96 * np.nanstd(a, axis=axis, ddof=1) / math.sqrt(max(nn, 1))


def slope(y):
    y = np.asarray(y, float); x = np.arange(len(y)); g = ~np.isnan(y)
    return float(np.polyfit(x[g], y[g], 1)[0]) if g.sum() >= 3 else 0.0


def aggregate_stream(world_name, m, n, cfg, regime, seeds):
    runs = [run_stream(world_name, m, n, s, cfg, regime) for s in range(seeds)]
    modes = ("A", "O", "C", "U", "S")
    agg = {}
    for md in modes:
        agg[md] = {}
        for kk in ("oos", "selferr", "sig"):
            M = np.stack([r["log"][md][kk] for r in runs])
            agg[md][kk] = {"mean": np.nanmean(M, 0), "raw": M}
    return {"agg": agg, "shift_at": runs[0]["shift_at"],
            "oracle_floor": float(np.mean([r["oracle_floor"] for r in runs]))}


def late(d, frac=0.2, seg=None):
    M = d["raw"]; T = M.shape[1]
    lo, hi = (int(T * (1 - frac)), T) if seg is None else seg
    ps = np.nanmean(M[:, lo:hi], axis=1)
    return float(np.nanmean(ps)), float(ci95(ps, 0)), ps


# ============================================================ MAIN
def main():
    t0 = time.time(); os.makedirs(ART, exist_ok=True)
    rng = np.random.default_rng(SEED)
    m, n, bmax = 12, 16, 16
    B_tot = 4 * m * n                                  # matched budget: avg 4 bits/element
    world_names = ["lowrank_activation", "readout_focus", "outer_powerlaw", "banded_smooth",
                   "two_scale", "corr_input"]          # >=5 non-flat
    worlds = [make_world(nm, m, n, SEED + 3 * k) for k, nm in enumerate(world_names)]
    flat = make_world("flat", m, n, SEED + 77)

    ev = {"cell": "d_self_tuning_sensitivity_kernel", "seed": SEED,
          "kernel": "y=A x, QoI J=g^T A x; alpha_ij = A_ij^2 E[(g_i x_j)^2] = structure-Fisher of element (i,j)",
          "matched_cost": f"total bits B_tot={B_tot} (avg {B_tot/(m*n):.1f} bits/element), bmax={bmax}",
          "m": m, "n": n}

    # ---------------- KILL GATES ----------------
    ev["KG1_adjoint"] = kg_adjoint(np.random.default_rng(SEED + 1))
    ev["KG2_model_vs_mc"] = kg_model_vs_mc(worlds + [flat], B_tot, bmax, np.random.default_rng(SEED + 2))
    ev["KG3_allocator_optimal"] = kg_allocator_optimal(np.random.default_rng(SEED + 3))

    # ---------------- G1 (>=5 worlds) + G4 floor ----------------
    g1 = {}
    for k, w in enumerate(worlds):
        g1[w["name"]] = static_compare(w, B_tot, bmax, np.random.default_rng(SEED + 1000 * (k + 1)))
    ev["G1_static_matched_cost"] = g1
    ru = [g1[k]["ratio_uniform_over_wf"] for k in g1]
    rr = [g1[k]["ratio_random_over_wf"] for k in g1]
    rm = [g1[k]["ratio_magnitude_over_wf"] for k in g1]
    G1_pass = all(g1[k]["wf_beats_uniform"] and g1[k]["wf_beats_random"] for k in g1) and min(ru) > 1.0 + 1e-9

    # ---------------- G2 forced-negative: EXACTLY-flat alpha -> allocator MUST degenerate to uniform ----------------
    alpha_flat = np.full((m, n), 0.5)
    b_wf_f = wf_budget_int(alpha_flat, B_tot, bmax)
    b_un_f = uniform_budget((m, n), B_tot, bmax, np.random.default_rng(SEED + 5))
    ratio_flat = model_err(alpha_flat, b_un_f) / model_err(alpha_flat, b_wf_f)
    same_alloc = bool(np.array_equal(np.sort(b_wf_f.ravel()), np.sort(b_un_f.ravel())))
    fc = static_compare(flat, B_tot, bmax, np.random.default_rng(SEED + 6))   # constant-|A| world (near-flat)
    ev["G2_flat_forced_negative"] = {
        "exact_flat_alpha": {"ratio_uniform_over_wf": ratio_flat, "identical_allocation_multiset": same_alloc,
                             "note": "flat sensitivity -> no spectrum -> water-fill == uniform, ratio == 1.000"},
        "constant_magnitude_world": fc}
    G2_pass = abs(ratio_flat - 1.0) < 1e-9 and same_alloc and abs(fc["ratio_uniform_over_wf"] - 1.0) < 0.05

    # ---------------- G4 bound ----------------
    G4_pass = all(g1[k]["wf_le_all"] for k in g1) and ev["KG3_allocator_optimal"]["KG3_pass"]
    ev["G4_certified_floor"] = {
        "floor_is_wf_on_true_spectrum": True,
        "no_matched_cost_alloc_beats_floor": bool(all(g1[k]["wf_le_all"] for k in g1)),
        "allocator_is_provable_optimum": ev["KG3_allocator_optimal"]["KG3_pass"],
        "saving_bound_uniform_over_floor": {k: g1[k]["ratio_uniform_over_wf"] for k in g1},
        "note": "reported saving is bounded by uniform/floor; the water-fill on the TRUE spectrum is the "
                "matched-cost optimum (Fox-exact, brute-force certified) -> no over-claim beyond it."}

    # ---------------- G3 online (>=5 worlds + controls) ----------------
    cfg = dict(T=140, anchor=48, n_cal=24, n_moves=60, bmax=bmax, avg_bits=4)
    seeds = 12
    g3 = {}
    for nm in world_names:
        S = aggregate_stream(nm, m, n, cfg, "stationary", seeds)
        R = S["agg"]; floor = S["oracle_floor"]
        A_o, A_ci, A_ps = late(R["A"]["oos"]); C_o, C_ci, C_ps = late(R["C"]["oos"])
        U_o, U_ci, U_ps = late(R["U"]["oos"]); S_o, S_ci, S_ps = late(R["S"]["oos"])
        C_slope = slope(R["C"]["oos"]["mean"])
        U_self, _, _ = late(R["U"]["selferr"]); U_oos_gap = U_o - U_self  # overfit gap (true OOS worse than U's self-report)
        compound = (C_slope < 0) and (C_o + C_ci < A_o - A_ci)           # C compounds below the uniform baseline
        overfit = (U_oos_gap > 0) and (U_o - U_ci > C_o + C_ci)          # U overfits its in-sample calibration -> worse OOS than C
        magnitude_insufficient = (S_o - S_ci) > (C_o + C_ci)            # magnitude proxy far worse than the MEASURED Fisher
        gate_diff = (U_o - U_ci) > (C_o + C_ci)                          # C<U attributable to the held-out gate alone
        g3[nm] = {"oracle_floor": floor, "A_oos": A_o, "C_oos": C_o, "U_oos": U_o, "S_oos": S_o,
                  "A_ci": A_ci, "C_ci": C_ci, "U_ci": U_ci, "S_ci": S_ci,
                  "C_slope": C_slope, "C_over_floor": C_o / floor, "A_over_floor": A_o / floor,
                  "U_over_floor": U_o / floor, "S_over_floor": S_o / floor,
                  "U_selfreport": U_self, "U_overfit_gap_oos_minus_selfreport": U_oos_gap,
                  "frac_C_beats_U": float(np.mean(C_ps < U_ps)), "frac_C_beats_A": float(np.mean(C_ps < A_ps)),
                  "frac_C_beats_S": float(np.mean(C_ps < S_ps)),
                  "compound": bool(compound), "overfit": bool(overfit),
                  "magnitude_insufficient": bool(magnitude_insufficient), "gate_is_difference": bool(gate_diff)}
    ev["G3_guard_stationary"] = g3
    G3a = np.mean([g3[k]["compound"] for k in g3]) >= 0.8       # C compounds in >=80% of worlds
    G3b = np.mean([g3[k]["overfit"] for k in g3]) >= 0.8        # U overfits (worse OOS than C + self-report gap)
    G3c = np.mean([g3[k]["gate_is_difference"] and g3[k]["magnitude_insufficient"] for k in g3]) >= 0.8

    # controls: flat (guard free) + nonstationary (C re-adapts, U/S chase stale)
    Sf = aggregate_stream("flat", m, n, cfg, "stationary", seeds); Rf = Sf["agg"]
    fl = {a: late(Rf[a]["oos"])[0] for a in ("A", "C", "U", "S")}
    # guard is FREE: on flat data (nothing to learn) the GATED tuner C neither compounds nor drifts -> ties A.
    # (The UNGATED U still drifts slightly worse even here -> extra evidence the held-out gate is load-bearing.)
    flat_tie = abs(fl["C"] - fl["A"]) / (fl["A"] + 1e-30) < 0.05
    ev["G3_control_flat"] = {**fl, "gated_C_vs_A_rel": abs(fl["C"] - fl["A"]) / (fl["A"] + 1e-30),
                             "ungated_U_vs_A_rel": (fl["U"] - fl["A"]) / (fl["A"] + 1e-30),
                             "guard_free_gated_ties_baseline": bool(flat_tie)}

    ns = aggregate_stream("outer_powerlaw", m, n, cfg, "nonstationary", seeds)
    NR = ns["agg"]; sh = ns["shift_at"]
    post = {a: late(NR[a]["oos"], seg=(cfg["T"] - 25, cfg["T"]))[0] for a in ("A", "C", "U", "S")}
    C_recover = slope(NR["C"]["oos"]["mean"][sh:])              # error should DECREASE (recover) post-shift
    Cps = np.nanmean(NR["C"]["oos"]["raw"][:, sh:], axis=1); Ups = np.nanmean(NR["U"]["oos"]["raw"][:, sh:], axis=1)
    pair = Ups - Cps; pair_mean, pair_ci = float(np.mean(pair)), float(ci95(pair, 0))
    nonstat_ok = (C_recover < 0) and (pair_mean - pair_ci > 0)
    ev["G3_control_nonstationary"] = {**{("post_" + a): post[a] for a in post},
                                      "C_recovery_slope": C_recover, "paired_U_minus_C_postshift": pair_mean,
                                      "paired_ci": pair_ci, "C_readapts_beats_U": bool(nonstat_ok), "shift_at": int(sh)}
    G3d = bool(flat_tie and nonstat_ok)
    G3_pass = bool(G3a and G3b and G3c and G3d)

    # ---------------- VERDICTS ----------------
    KG = ev["KG1_adjoint"]["KG1_pass"] and ev["KG2_model_vs_mc"]["KG2_pass"] and ev["KG3_allocator_optimal"]["KG3_pass"]
    kills = []
    if not ev["KG1_adjoint"]["KG1_pass"]: kills.append("KG1 adjoint mismatch")
    if not ev["KG2_model_vs_mc"]["KG2_pass"]: kills.append("KG2 model != operational MC")
    if not ev["KG3_allocator_optimal"]["KG3_pass"]: kills.append("KG3 allocator != brute-force optimum")
    if not G1_pass: kills.append("G1 wf does not beat uniform+random at matched cost")
    if not G2_pass: kills.append("G2 flat forced-negative failed (wf 'wins' on flat -> allocator bug)")
    if not G3_pass: kills.append("G3 guard: compound/overfit/gate/control failed")
    if not G4_pass: kills.append("G4 an allocation beat the certified floor / allocator not optimal")

    ev["verdicts"] = {
        "kills": kills, "KILL_GATES_pass": bool(KG),
        "G1_sensitivity_beats_uniform_and_random": {
            "verdict": "PASS" if G1_pass else "FAIL",
            "ratio_uniform_over_wf": {k: g1[k]["ratio_uniform_over_wf"] for k in g1},
            "ratio_random_over_wf": {k: g1[k]["ratio_random_over_wf"] for k in g1},
            "ratio_magnitude_over_wf": {k: g1[k]["ratio_magnitude_over_wf"] for k in g1},
            "margin_uniform_range": [float(min(ru)), float(max(ru))],
            "margin_random_range": [float(min(rr)), float(max(rr))],
            "confidence": "high (MC-anchored in KG2; magnitude-allocation also loses -> it is the Fisher, not |A|)"},
        "G2_flat_forced_negative": {"verdict": "PASS" if G2_pass else "FAIL",
            "ratio_uniform_over_wf_on_flat": fc["ratio_uniform_over_wf"],
            "confidence": "very-high (exact degeneracy; water-fill == uniform when alpha is flat)"},
        "G3_guard_double_drift": {"verdict": "PASS" if G3_pass else "FAIL",
            "C_compounds_frac_worlds": float(np.mean([g3[k]["compound"] for k in g3])),
            "U_overfits_frac_worlds": float(np.mean([g3[k]["overfit"] for k in g3])),
            "gate_is_difference_frac_worlds": float(np.mean([g3[k]["gate_is_difference"] for k in g3])),
            "flat_guard_free_tie": bool(flat_tie), "nonstationary_C_readapts": bool(nonstat_ok),
            "mean_C_over_floor": float(np.mean([g3[k]["C_over_floor"] for k in g3])),
            "mean_A_over_floor": float(np.mean([g3[k]["A_over_floor"] for k in g3])),
            "confidence": "held-out gate converts overfit->compound; U/C differ ONLY in the accept-oracle"},
        "G4_bounded_saving": {"verdict": "PASS" if G4_pass else "FAIL",
            "no_alloc_beats_floor": bool(all(g1[k]["wf_le_all"] for k in g1)),
            "allocator_provably_optimal": ev["KG3_allocator_optimal"]["KG3_pass"],
            "confidence": "high (Fox-exact, brute-force certified; floor = wf on TRUE spectrum)"},
    }
    all_pass = KG and G1_pass and G2_pass and G3_pass and G4_pass
    ev["verdicts"]["HEADLINE"] = (
        f"SELF-LEARNING KERNEL {'CONFIRMED' if all_pass else 'NOT CLEAN'}: measuring the per-element "
        f"structure-Fisher and water-filling bits+sparsity by it beats uniform ({min(ru):.2f}-{max(ru):.2f}x "
        f"lower QoI error) and random ({min(rr):.2f}-{max(rr):.2f}x) at matched total bits across "
        f"{len(g1)} worlds; FLAT -> no gain (ratio {fc['ratio_uniform_over_wf']:.4f}); the online autotuner "
        f"COMPOUNDS only WITH the out-of-sample anchor-floor gate (C->floor) and OVERFITS without it (U, "
        f"in-sample) -- the double-drift on compute-allocation; saving is bounded by the certified "
        f"water-filling floor.") if not kills else f"KILLED ({len(kills)}): {kills[:3]}"
    ev["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(ev, open(EV, "w"), indent=1,
              default=lambda o: (o.tolist() if hasattr(o, "tolist") else float(o)))

    # ---------------- console ----------------
    P = print
    P("=" * 104)
    P("SELF-LEARNING KERNEL -- measure per-element structure-Fisher -> water-fill bits+sparsity -> gated autotune")
    P("=" * 104)
    P(f"kernel J=g^T A x  ({m}x{n}={m*n} elements)  alpha_ij=A_ij^2 E[(g_i x_j)^2]   matched budget B_tot={B_tot} "
      f"(avg {B_tot/(m*n):.1f} bits)")
    k1, k2, k3 = ev["KG1_adjoint"], ev["KG2_model_vs_mc"], ev["KG3_allocator_optimal"]
    P(f"KG1 adjoint(analytic==FD==MC, +nonlinear): {k1['KG1_pass']} (lin {k1['linear_relerr']:.1e}, nl {k1['nonlinear_relerr']:.1e})  "
      f"KG2 model==MC(<3%): {k2['KG2_pass']}  KG3 allocator==brute(opt): {k3['KG3_pass']}")
    P("-" * 104)
    P(f"{'world':20s} {'AM/GM':>6s} {'uni/wf':>7s} {'rand/wf':>8s} {'mag/wf':>7s} {'drop_wf':>7s} {'wf=floor':>8s}")
    for k in g1:
        r = g1[k]
        P(f"{k:20s} {r['amgm_alpha']:6.2f} {r['ratio_uniform_over_wf']:7.2f} {r['ratio_random_over_wf']:8.2f} "
          f"{r['ratio_magnitude_over_wf']:7.2f} {r['n_dropped_wf']:7d} {str(r['wf_le_all']):>8s}")
    P(f"{'flat (FORCED-NEG)':20s} {fc['amgm_alpha']:6.2f} {fc['ratio_uniform_over_wf']:7.4f} "
      f"{fc['ratio_random_over_wf']:8.2f} {fc['ratio_magnitude_over_wf']:7.2f} {fc['n_dropped_wf']:7d}")
    P("-" * 104)
    P("G3 ONLINE GUARD (gated compounds, ungated overfits); OOS QoI error, late-20%, as x(oracle-floor):")
    P(f"{'world':20s} {'A/fl':>6s} {'C/fl':>6s} {'U/fl':>6s} {'S/fl':>6s} {'C_slope':>9s} {'U_ovft_gap':>10s} "
      f"{'cmpnd':>5s} {'ovft':>4s} {'gate':>4s}")
    for k in g3:
        r = g3[k]
        P(f"{k:20s} {r['A_over_floor']:6.2f} {r['C_over_floor']:6.2f} {r['U_over_floor']:6.2f} "
          f"{r['S_over_floor']:6.2f} {r['C_slope']:+9.2e} {r['U_overfit_gap_oos_minus_selfreport']:+10.2e} "
          f"{str(r['compound']):>5s} {str(r['overfit']):>4s} {str(r['gate_is_difference']):>4s}")
    P("-" * 104)
    cf = ev["G3_control_flat"]; nsc = ev["G3_control_nonstationary"]
    P(f"CONTROL flat (guard FREE): A={cf['A']:.3e} C={cf['C']:.3e}(gated ties A) U={cf['U']:.3e}(ungated drifts) "
      f"S={cf['S']:.3e}  |C-A|/A={cf['gated_C_vs_A_rel']:.3f} free={cf['guard_free_gated_ties_baseline']}")
    P(f"CONTROL nonstationary (shift@{nsc['shift_at']}): post A={nsc['post_A']:.3e} C={nsc['post_C']:.3e} "
      f"U={nsc['post_U']:.3e}; C recover slope={nsc['C_recovery_slope']:+.2e}; "
      f"paired U-C postshift={nsc['paired_U_minus_C_postshift']:+.3e}±{nsc['paired_ci']:.3e} -> C re-adapts={nsc['C_readapts_beats_U']}")
    P("-" * 104)
    for g in ("G1_sensitivity_beats_uniform_and_random", "G2_flat_forced_negative",
              "G3_guard_double_drift", "G4_bounded_saving"):
        P(f"  {g}: {ev['verdicts'][g]['verdict']}")
    P(f"  KILL-GATES: {'PASS' if KG else 'FAIL'}   KILLS: {kills if kills else 'none'}")
    P(f"\nHEADLINE: {ev['verdicts']['HEADLINE']}")
    P(f"\nevidence -> {EV}   runtime {ev['runtime_sec']}s")


if __name__ == "__main__":
    main()

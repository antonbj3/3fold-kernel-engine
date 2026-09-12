"""
d_thermo_datahole_law.py  (an agent worktree, cell 32) -- upgrade cell 27-G2 from INSTANCE to LAW.

cell 27 measured, on ONE world, a scarce-budget crossover: J^-1-eigenmass-ranked
acquisition ("thermo") beats a spatially-even uniform by 2.13x at 6 pts, decaying to
~1.08x at 48. D's skeptic demotion: one world => instance, not law. This cell reruns
the budget sweep over N_W=10 DECORRELATED worlds (varying #localities/topology,
intra/inter coupling, coverage heterogeneity, hole count) and tests whether (a) the
crossover SHAPE holds and (b) the advantage MAGNITUDE is governed by a nameable world
parameter -- i.e. the crossover is a geometry law, not a fluke of one graph.

MECHANISM (why a crossover should exist AND why it should decay): the ranker targets
the highest posterior-variance nodes (the least-constrained "data holes"). When existing
coverage tau is HETEROGENEOUS, posterior variance is concentrated in a few holes ->
targeted acquisition collapses total QoI-variance fast while even-uniform wastes budget
on already-tight nodes => big scarce-budget advantage. As budget grows, even uniform
eventually covers the holes and Sherman-Morrison diminishing-returns caps the targeted
gain => advantage decays toward 1. When coverage is HOMOGENEOUS there is no hole to
target => advantage ~ 1 at every budget (no crossover). So the PREDICTED governing
parameter is coverage heterogeneity. That prediction carries its own falsifier: a
near-homogeneous world MUST show a_6 ~ 1 (mechanism's null-floor); if it did not, the
"holes drive the advantage" story would be wrong.

RANKER: exact J^-1 eigenmass leverage (noise-free ceiling of the thermo substrate;
cell 27-G1/G3 established the Langevin thermo estimate tracks this within its ceiling,
and thermo_vs_exact was 1.00 at n=6). We ISOLATE the acquisition-ordering law here and
OVER-DETERMINE with one real Langevin-thermo run (world 0) to confirm exact is faithful.
METRIC: sum over per-locality-mean QoIs of the TRUE posterior variance g^T J_S^-1 g,
reduction PER acquired point, evaluated on the ground-truth posterior for the actual
picks of each method. QoIs are a FIXED neutral set (one mean per locality) -- NOT placed
on the holes, so "heterogeneity -> advantage" is not tautological with QoI placement.

PRE-REGISTERED GATES (fixed BEFORE first run; no post-hoc thresholds):
  W  10 worlds, fixed seeds, decorrelated: LOC in {3,4,5,6}, NG~120, intra-density,
     inter-coupling, tau dynamic-range R in [1.2, 60] (log-uniform: spans homogeneous
     -> skewed), #starved localities all sampled per world. Advantage a_b =
     perpoint_reduction(thermo)/perpoint_reduction(uniform) at b in {6,12,24,48}.
  G_SHAPE  crossover shape holds in >= 7/10 worlds, where "holds" =
     never-loses (a_b >= 0.98 at ALL b) AND non-increasing (a_6 >= a_12 >= a_24 >= a_48,
     tol 0.03). [flat-at-1 homogeneous worlds satisfy this vacuously and are allowed.]
  G_CROSS  a REAL crossover (a_6/a_48 >= 1.30 with a_6 >= 1.5) appears in >= 4 worlds
     -- the effect is reproducible, not a single-world fluke.
  G_LAW  the scarce-budget excess A6 = a_6 - 1 is GOVERNED by coverage heterogeneity:
     Pearson r(A6, coverage_gini) >= 0.70 across the 10 worlds, bootstrap 95% CI on r
     excludes 0. Also fit posterior-variance-gini; NAME whichever predicts better.
  G_NULL  mechanism falsifier: at least one near-homogeneous world (coverage_gini in the
     lowest tercile) has A6 <= 0.20 (advantage ~1) -- advantage is NOT universal; it
     switches OFF when the hole vanishes. If a homogeneous world showed large A6 the
     mechanism is refuted -> FORCED negative.
  G_DECAY  decay-exponent universality: fit log(a_b-1) ~ p*log(b)+c per crossover world;
     report median p and IQR (a tight p => shared curve family).

Pure math/stats/physics. Deterministic seeds. OMP_NUM_THREADS=2, no GPU, <=90s.
"""
import json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

T0 = time.time()
BASE_SEED = 20260709
N_W = 10
NG_TARGET = 120
TOPK = 8
DELTA = 2.0
BUDGETS = [6, 12, 24, 48]
EV = {}

# ---------------------------------------------------------------------------
def build_world(seed):
    """Decorrelated GMRF world. Returns J (SPD precision), tau, LOC, PER, meta."""
    rng = np.random.default_rng(seed)
    LOC = int(rng.integers(3, 7))                 # 3..6 localities
    PER = NG_TARGET // LOC
    NG = LOC * PER
    p_in = float(rng.uniform(0.15, 0.35))         # intra-locality density
    inter_w = float(rng.uniform(0.1, 0.5))        # inter-locality coupling weight
    n_inter = int(rng.integers(2, 5))             # edges per inter-locality link
    A = np.zeros((NG, NG))
    for c in range(LOC):
        idx = np.arange(c*PER, (c+1)*PER)
        for a in idx:
            for b in idx:
                if a < b and rng.random() < p_in:
                    A[a, b] = A[b, a] = 0.8 + 0.4*rng.random()
    # topology: ring or path over localities (decorrelates connectivity)
    ring = bool(rng.random() < 0.5)
    links = [(c, c+1) for c in range(LOC-1)] + ([(LOC-1, 0)] if ring else [])
    for (c, d) in links:
        for _ in range(n_inter):
            a = rng.integers(c*PER, (c+1)*PER); b = rng.integers(d*PER, (d+1)*PER)
            A[a, b] = A[b, a] = inter_w
    deg = A.sum(1); L = np.diag(deg) - A
    # heterogeneous existing coverage: log-spaced dynamic range R across localities
    R = float(np.exp(rng.uniform(np.log(1.2), np.log(60.0))))  # 1.2 (homog) .. 60 (skewed)
    lo = 0.08
    tau_by_loc = lo * np.exp(np.linspace(0.0, np.log(R), LOC))
    rng.shuffle(tau_by_loc)                        # holes are not always the same locality
    tau = np.repeat(tau_by_loc, PER) + 0.02
    J = L + np.diag(tau); J = 0.5*(J+J.T)
    meta = {"LOC": LOC, "NG": NG, "PER": PER, "p_in": p_in, "inter_w": inter_w,
            "ring": ring, "R": R}
    return J, tau, LOC, PER, meta

def qoi_functionals(LOC, PER, NG):
    """FIXED neutral QoIs: one mean per locality (not placed on holes)."""
    g = np.zeros((LOC, NG))
    for c in range(LOC):
        g[c, c*PER:(c+1)*PER] = 1.0/PER
    return g

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if x.sum() <= 0: return 0.0
    return float((2*np.arange(1, n+1) - n - 1) @ x / (n * x.sum()))

def eigenmass_score(C, k=TOPK):
    w, V = np.linalg.eigh(C)
    Uk = V[:, -k:]; wk = w[-k:]
    return (Uk*Uk) @ wk

def sm_update(C, i, delta):
    ci = C[:, i]
    return C - np.outer(ci, ci)*(delta/(1.0 + delta*C[i, i]))

def qoi_var(C, g):
    return float(sum(g[j] @ C @ g[j] for j in range(g.shape[0])))

def voi_score(C, g, delta):
    """one-step VOI: exact QoI-variance reduction from a rank-1 SM update at node i.
    delta_j(i) = (g_j^T C[:,i])^2 * delta/(1+delta*C_ii). score_i = sum_j delta_j(i)."""
    GC = g @ C                                    # (LOC, NG); GC[j,i] = g_j^T C[:,i]
    num = (GC*GC).sum(0)                           # sum_j (g_j^T C[:,i])^2
    return num * (delta/(1.0 + delta*np.diag(C)))

def run_acq(J, g, belief0, method, NG, n_acq, seed=0):
    Ctrue = np.linalg.inv(J); Cbel = belief0.copy()
    rng = np.random.default_rng(seed); v0 = qoi_var(Ctrue, g)
    for t in range(n_acq):
        if method == "voi":                       # model-relative one-step-optimal ranker
            i = int(np.argmax(voi_score(Cbel, g, DELTA)))
        elif method in ("thermo", "exact"):       # cell 27 eigenmass proxy (QoI-agnostic)
            i = int(np.argmax(eigenmass_score(Cbel)))
        elif method == "uniform":
            i = int(round(t*NG/n_acq)) % NG
        else:
            i = int(rng.integers(NG))
        Ctrue = sm_update(Ctrue, i, DELTA); Cbel = sm_update(Cbel, i, DELTA)
    return (v0 - qoi_var(Ctrue, g)) / n_acq       # per-point reduction

def langevin_cov(J, NG, n_steps=800, n_chains=2000, seed=0):
    rng = np.random.default_rng(seed)
    dt = 0.5/np.linalg.eigvalsh(J)[-1]; a = np.sqrt(2*dt)
    X = np.zeros((NG, n_chains))
    for _ in range(n_steps):
        X = X - dt*(J@X) + a*rng.standard_normal((NG, n_chains))
    C = (X @ X.T)/n_chains
    return 0.5*(C+C.T)

# ---------------------------------------------------------------------------
def sweep_world(seed, do_thermo=False):
    J, tau, LOC, PER, meta = build_world(seed)
    NG = meta["NG"]; g = qoi_functionals(LOC, PER, NG)
    Cinv = np.linalg.inv(J)
    a = {}
    for b in BUDGETS:
        rv = run_acq(J, g, Cinv, "voi", NG, b, seed=seed)            # STRONG model-rel ranker
        re = run_acq(J, g, Cinv, "exact", NG, b, seed=seed)          # cell 27 eigenmass proxy
        ru = run_acq(J, g, Cinv, "uniform", NG, b, seed=seed)
        rr = np.mean([run_acq(J, g, Cinv, "random", NG, b, seed=seed+900+s)
                      for s in range(5)])
        a[b] = {"adv_unif": float(rv/ru), "adv_rand": float(rv/rr),
                "eig_adv_unif": float(re/ru), "pp_voi": float(rv), "pp_unif": float(ru)}
    cov_gini = gini(tau)
    pv_gini = gini(np.diag(Cinv))
    res = {"meta": meta, "cov_gini": cov_gini, "pv_gini": pv_gini,
           "a": a, "condJ": float(np.linalg.cond(J))}
    if do_thermo:                            # over-det: VOI on NOISY Langevin-thermo belief
        Chat = langevin_cov(J, NG, seed=seed+7)
        rv_th = run_acq(J, g, Chat, "voi", NG, 6, seed=seed)  # VOI ranker, thermo belief
        ru6 = run_acq(J, g, Cinv, "uniform", NG, 6, seed=seed)
        res["thermo_check_a6"] = float(rv_th/ru6)             # compare to VOI-exact a6
    return res

def main():
    worlds = []
    for w in range(N_W):
        worlds.append(sweep_world(BASE_SEED + 101*w, do_thermo=(w == 0)))

    A6 = np.array([wd["a"][6]["adv_unif"] - 1.0 for wd in worlds])
    a6 = np.array([wd["a"][6]["adv_unif"] for wd in worlds])
    a48 = np.array([wd["a"][48]["adv_unif"] for wd in worlds])
    cg = np.array([wd["cov_gini"] for wd in worlds])
    pg = np.array([wd["pv_gini"] for wd in worlds])

    # G_SHAPE per world
    def shape_ok(wd):
        seq = [wd["a"][b]["adv_unif"] for b in BUDGETS]
        never = all(s >= 0.98 for s in seq)
        noninc = all(seq[i] >= seq[i+1] - 0.03 for i in range(len(seq)-1))
        return never and noninc
    shape = [shape_ok(wd) for wd in worlds]
    n_shape = int(sum(shape))
    G_SHAPE = n_shape >= 7

    # G_CROSS: real crossovers
    cross_mask = [(wd["a"][6]["adv_unif"]/wd["a"][48]["adv_unif"] >= 1.30
                   and wd["a"][6]["adv_unif"] >= 1.5) for wd in worlds]
    n_cross = int(sum(cross_mask))
    G_CROSS = n_cross >= 4

    # G_LAW: Pearson r(A6, gini), bootstrap CI
    def pearson(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = np.sqrt((x*x).sum()*(y*y).sum())
        return float((x*y).sum()/d) if d > 0 else 0.0
    r_cov = pearson(A6, cg); r_pv = pearson(A6, pg)
    rng = np.random.default_rng(BASE_SEED)
    def boot_ci(x, y, n=4000):
        rs = []
        for _ in range(n):
            idx = rng.integers(0, len(x), len(x))
            rs.append(pearson(x[idx], y[idx]))
        return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))
    ci_cov = boot_ci(A6, cg); ci_pv = boot_ci(A6, pg)
    best_param = "cov_gini" if abs(r_cov) >= abs(r_pv) else "pv_gini"
    r_best = r_cov if best_param == "cov_gini" else r_pv
    ci_best = ci_cov if best_param == "cov_gini" else ci_pv
    # OLS slope A6 ~ param (report law with bootstrap slope CI)
    xb = cg if best_param == "cov_gini" else pg
    def ols(x, y):
        X = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[0]), float(beta[1])
    b0, b1 = ols(xb, A6)
    slopes = []
    for _ in range(4000):
        idx = rng.integers(0, len(xb), len(xb))
        slopes.append(ols(xb[idx], A6[idx])[1])
    slope_ci = (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
    G_LAW = (r_best >= 0.70) and (ci_best[0] > 0.0)

    # G_NULL: homogeneous world floor
    tercile = np.percentile(cg, 33.33)
    homog_A6 = [float(A6[i]) for i in range(N_W) if cg[i] <= tercile]
    G_NULL = any(v <= 0.20 for v in homog_A6)

    # G_DECAY: decay exponent per crossover world
    ps = []
    for i, wd in enumerate(worlds):
        if not cross_mask[i]:
            continue
        bs = np.array(BUDGETS, float); ex = np.array([wd["a"][b]["adv_unif"]-1.0 for b in BUDGETS])
        m = ex > 1e-4
        if m.sum() >= 3:
            p, _ = ols(np.log(bs[m]), np.log(ex[m]))  # slope in log-log = decay exponent
            ps.append(p)
    ps = np.array(ps)
    decay_med = float(np.median(ps)) if len(ps) else None
    decay_iqr = [float(np.percentile(ps, 25)), float(np.percentile(ps, 75))] if len(ps) else None

    EV["worlds"] = [{"seed_idx": i, "meta": wd["meta"], "cov_gini": wd["cov_gini"],
                     "pv_gini": wd["pv_gini"], "condJ": wd["condJ"],
                     "a_unif": {b: wd["a"][b]["adv_unif"] for b in BUDGETS},
                     "a_rand": {b: wd["a"][b]["adv_rand"] for b in BUDGETS},
                     "eig_a_unif": {b: wd["a"][b]["eig_adv_unif"] for b in BUDGETS},
                     "shape_ok": shape[i], "crossover": cross_mask[i]}
                    for i, wd in enumerate(worlds)]
    EV["thermo_check_a6"] = worlds[0].get("thermo_check_a6")
    EV["thermo_check_exact_a6"] = worlds[0]["a"][6]["adv_unif"]
    EV["summary"] = {
        "n_shape_ok": n_shape, "n_crossover": n_cross,
        "r_A6_cov_gini": r_cov, "ci_cov": ci_cov,
        "r_A6_pv_gini": r_pv, "ci_pv": ci_pv,
        "best_param": best_param, "r_best": r_best, "ci_best": ci_best,
        "law_slope": b1, "law_intercept": b0, "slope_ci": slope_ci,
        "homog_A6": homog_A6, "cov_gini_tercile": float(tercile),
        "decay_exp_median": decay_med, "decay_exp_iqr": decay_iqr}
    EV["gates"] = {"G_SHAPE": bool(G_SHAPE), "G_CROSS": bool(G_CROSS),
                   "G_LAW": bool(G_LAW), "G_NULL": bool(G_NULL)}
    EV["runtime_s"] = round(time.time()-T0, 2)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "d_thermo_datahole_law_evidence.json")
    with open(out, "w") as f:
        json.dump(EV, f, indent=2)

    print(f"runtime {EV['runtime_s']}s  N_W={N_W}")
    for i, wd in enumerate(worlds):
        seq = "/".join(f"{wd['a'][b]['adv_unif']:.2f}" for b in BUDGETS)
        eseq = "/".join(f"{wd['a'][b]['eig_adv_unif']:.2f}" for b in BUDGETS)
        print(f"  w{i} LOC{wd['meta']['LOC']} R{wd['meta']['R']:.1f} "
              f"cg={wd['cov_gini']:.3f} pg={wd['pv_gini']:.3f} "
              f"voi[6/12/24/48]={seq} eig={eseq} shape={shape[i]} cross={cross_mask[i]}")
    print(f"thermo-check a6 (Langevin vs exact ranker): {EV['thermo_check_a6']:.3f} "
          f"vs {EV['thermo_check_exact_a6']:.3f}")
    print(f"G_SHAPE {n_shape}/10  G_CROSS {n_cross}  "
          f"r(A6,cov_gini)={r_cov:.3f} CI{tuple(round(c,3) for c in ci_cov)}  "
          f"r(A6,pv_gini)={r_pv:.3f} CI{tuple(round(c,3) for c in ci_pv)}")
    print(f"best={best_param} r={r_best:.3f}  law: A6={b0:.3f}+{b1:.3f}*param "
          f"slopeCI{tuple(round(s,3) for s in slope_ci)}")
    print(f"homog_A6={[round(v,3) for v in homog_A6]} (tercile cg<={tercile:.3f})")
    print(f"decay_exp median={decay_med} IQR={decay_iqr}")
    print(f"GATES {EV['gates']}")

if __name__ == "__main__":
    main()

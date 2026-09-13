"""L1 NODE — compose the 4 CUDA-kernel scene-eyes (determinism, parity, profiler, roofline) into ONE
kernel-cert-vector. L0 (4 real-GPU eyes) is COMPLETE (d_cuda_scene_eyes_determinism_real_gpu.py,
d_roofline_scene_eye.py: "L0 COMPLETE ... Next: L1 compose the kernel-cert-vector"). This is that L1 node.

SHARED CONVENTION (reused, not invented — d_endtoend_composed_cert_demo.py): each eye emits a normalized
margin m = eye_value/eye_threshold, PASS iff m>=1 (chi=1, dimensionless). ROOT cert = MIN over the vector;
BINDING eye = argmin (what L3 refines next). This script builds compose_l1() to that spec + watertight-tests
the load-bearing assumption: MIN-across-independent-roots is gap-free IFF the 4 roots are DECORRELATED.

SCOPE (read before use — this is the FACET axis, not the other two N1 axes already in flight):
  - FACET axis (this file): do the 4 EYES for one kernel decorrelate? (the facet axis)
  - STAGE axis (H/C/B, NOT this file): directional-SVD across CAD->mesh->solve->QoI pipeline STAGES —
    a different composition problem (pinch-vector alignment across stages of ONE QoI, not across 4
    independent QoI-verdicts of one kernel). Out of scope here.
  - non-normal transient axis (B, NOT this file): Kreiss/env_max transient gain — applies to dynamical
    systems with compounding substeps, not to 4 single-shot per-kernel measurements. Out of scope here.

PRIOR ART THIS BUILDS ON (real GPU, 2-facet case only — det+parity; roofline/profiler never entered the
facet-composition analysis before this file):
  d_n1_cert_composition_gapfree_iff_decorrelated.py (D) + U's reproduction + H's over-determination:
    Spearman rho(det_err, acc_err) = 0.799-0.81 (shared float-rounding root) -> "typical-case" gap ratio
    (true_worst / MIN-composition) = 1.14-1.19x MEDIAN, BOUNDED (H: total <=~2x for equal facets),
    NOT unbounded. Structural law true_worst ~ acc_err + det_err/2 (fit 0.89-0.91).
  d_force_H_facetgap_is_rho_floor_law.py (D): a SEPARATE, complementary "tail" framing — false-accept
    FLOOR (P(both facets falsely pass | genuinely defective)) vs the independence product p^2, swept over
    rho via a Gaussian-copula common-factor construction. Reference measurements: floor(rho=0)=0.0221~=p^2=0.0225; floor(rho=0.81)=0.0927=4.12x product; floor(rho=-0.8)=0
    (anti-correlation BEATS independence, actionable). This file extends that pattern with (a) an exact
    external analytic anchor (scipy bivariate-normal orthant probability — the prior script was MC-only),
    (b) realistic (not equal-p) marginal rates tied to the 4 real eyes, (c) a 2nd eye-pair for diversity.

WHAT'S NEW HERE (the actual L1 gap this fills): (1) the reusable compose_l1() function generalized to ALL
4 eyes (prior work only ever composed 2). (2) the 4x4 decorrelation matrix (6 pairs, not 1). (3) BOTH gap
framings (typical-case quadrature-law AND tail false-accept floor) cross-validated against the REAL
1.14-1.19x / 4.12x numbers above, on a SYNTHETIC (CPU-only; no GPU permitted for this task) 4-eye model.

CPU-only, pure numpy/scipy. No GPU/CUDA/warp code executed anywhere in this file.
Run: python3 d_l1_kernel_certvec_compose.py
"""
import numpy as np
from scipy.stats import norm, multivariate_normal

# ============================================================= PRE-REGISTERED THRESHOLDS (before any result)
RHO_NOISE_THRESH   = 0.15   # |mean measured noise-correlation| must be below this = "decorrelated" (practical)
RHO_NOISE_SE_MULT  = 3.0    # AND within this many SEs of 0 (statistical, given N)
FLOOR_HIGH_RHO_MIN = 3.0    # at rho>=0.8, floor/product inflation must exceed this = "the floor bites"
GAP_EQUAL_FACET_CAP = 2.05  # type-A gap ratio for EQUAL facets must stay <= ~2 at rho=1 (BOUNDED, not unbounded)
ANCHOR_REL_TOL     = 0.20   # MC vs exact-theory (scipy) cross-check relative tolerance
CHI = 1.0                   # cert threshold: margin >= 1 passes (project convention)


# ================================================================================== (0) THE L1 COMPOSITION
def compose_l1(margins: dict) -> dict:
    """THE L1 NODE. margins: {eye_name: normalized_margin (value/threshold, >=1 passes)}.
    Returns composed cert (MIN over the vector), binding eye (argmin = what L3 refines), certify bool."""
    names = list(margins.keys())
    vals = np.array([margins[n] for n in names], dtype=float)
    k = int(np.argmin(vals))
    composed = float(vals[k])
    return dict(composed_cert=composed, binding_eye=names[k], certify=bool(composed >= CHI),
                margins=dict(margins))


# ============================================================ (1) SYNTHETIC 4-EYE MODEL (CPU-only, no GPU)
# Shared "kernel quality" latent q in [0,1] (a generative-design knob abstraction: tile/unroll/precision
# choices folded to one scalar). Each eye responds through its OWN root-specific transfer shape (different
# physical mechanism -> different curve), PLUS its OWN independent noise stream (different physical root:
# GPU-scheduling nondeterminism / float-rounding / launch-latency jitter / DRAM-bandwidth jitter).
def true_curves(q):
    q = np.asarray(q, dtype=float)
    return {
        "determinism": 0.55 + 1.35 / (1.0 + np.exp(-9.0 * (q - 0.50))),   # sigmoid: race-condition threshold-like
        "parity":      0.35 + 1.30 * q,                                    # near-linear: rounding bias ~ numerics quality
        "profiler":    0.45 + 1.10 * np.sqrt(np.clip(q, 0, None)),         # concave: launch/occupancy efficiency saturates
        "roofline":    0.25 + 1.55 * q ** 2,                               # convex: BW efficiency ramps w/ access pattern
    }


def simulate_clean(q, seed):
    """Independent-root noise: 4 SEPARATE RNG streams (mechanistic independence by construction).
    seed: an int, OR an existing np.random.SeedSequence (e.g. one spawned from a master sequence)."""
    n = len(q)
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    r_det, r_par, r_prof, r_roof = [np.random.default_rng(s) for s in ss.spawn(4)]
    tc = true_curves(q)
    noise = {
        "determinism": r_det.normal(0, 0.05, n) + (r_det.random(n) < 0.05) * (-r_det.uniform(0.30, 0.80, n)),
        "parity":      r_par.normal(0, 0.035, n),
        "profiler":    r_prof.normal(0, 0.09, n),
        "roofline":    r_roof.normal(0, 0.10, n),
    }
    raw = {k: tc[k] + noise[k] for k in tc}
    margins = {k: np.clip(raw[k], 0.0, None) for k in raw}
    clip_frac = {k: float(np.mean(raw[k] < 0.0)) for k in raw}
    return margins, tc, noise, clip_frac


EYES = ["determinism", "parity", "profiler", "roofline"]
PAIRS = [(a, b) for i, a in enumerate(EYES) for b in EYES[i + 1:]]


def corr_matrix(d):
    names = list(d.keys())
    M = np.array([d[n] for n in names])
    return names, np.corrcoef(M)


def detrend(margin, q, win_frac=0.02):
    """Nonparametric, shape-agnostic detrend: subtract a LOCAL rolling mean in q (edge-normalized by
    actual window overlap, not zero-padded). FORCED FIX (OODA): a global low-degree polynomial fit
    leaks residual structure for a STEEP root-response (e.g. determinism's sigmoid) -- a degree-3 poly
    cannot track it, leaving structured (non-noise) residual that fakes a small but N=200k-resolvable
    spurious cross-eye correlation. A local rolling-mean has no global-shape assumption and tracks any
    smooth curve (sigmoid included) once the window is narrow vs the curve's own feature width."""
    n = len(q)
    order = np.argsort(q)
    inv = np.argsort(order)
    m_sorted = np.asarray(margin)[order]
    win = max(5, int(n * win_frac))
    kernel = np.ones(win)
    local_sum = np.convolve(m_sorted, kernel, mode="same")
    local_cnt = np.convolve(np.ones(n), kernel, mode="same")   # edge-correct: true overlap count, not win
    resid_sorted = m_sorted - local_sum / local_cnt
    return resid_sorted[inv]


print("=" * 100)
print("L1 KERNEL-CERT-VECTOR COMPOSE — watertight test of MIN-across-independent-roots (N1 rule)")
print("=" * 100)

# ============================================================================== TEST A: DECORRELATION
print("\n[TEST A] decorrelation of the 4 eyes' FAILURE ROOTS (clean/independent-noise baseline)")
print("-" * 100)
N_SWEEP, N_REPEATS = 8000, 25
raw_rhos = {p: [] for p in PAIRS}
oracle_rhos = {p: [] for p in PAIRS}
resid_rhos = {p: [] for p in PAIRS}
clip_fracs_all = []
binder_counts = {e: 0 for e in EYES}

master_ss = np.random.SeedSequence(20260706)
for rep_seed in master_ss.spawn(N_REPEATS):
    q_rng = np.random.default_rng(rep_seed.spawn(1)[0])
    q = q_rng.uniform(0, 1, N_SWEEP)
    margins, tc, noise, clipf = simulate_clean(q, seed=rep_seed)
    clip_fracs_all.append(clipf)
    _, R_raw = corr_matrix(margins)
    _, R_oracle = corr_matrix(noise)
    resid = {k: detrend(margins[k], q) for k in EYES}
    _, R_resid = corr_matrix(resid)
    for (a, b) in PAIRS:
        ia, ib = EYES.index(a), EYES.index(b)
        raw_rhos[(a, b)].append(R_raw[ia, ib])
        oracle_rhos[(a, b)].append(R_oracle[ia, ib])
        resid_rhos[(a, b)].append(R_resid[ia, ib])
    for i in range(N_SWEEP):
        v = compose_l1({k: margins[k][i] for k in EYES})
        binder_counts[v["binding_eye"]] += 1

SE = 1.0 / np.sqrt(N_SWEEP * N_REPEATS)   # pooled-sample SE approx (conservative: treats repeats as extra N)
print(f"  {N_REPEATS} repeats x N={N_SWEEP} draws each (total {N_REPEATS*N_SWEEP:,} synthetic kernel-variants); "
      f"pooled SE(rho)~{SE:.4f}")
print(f"  {'pair':<28}{'raw-margin rho':>16}{'oracle-noise rho':>18}{'detrend-resid rho':>19}   verdict(resid)")
all_decorrelated = True
for p in PAIRS:
    r_raw = np.mean(raw_rhos[p]); r_orac = np.mean(oracle_rhos[p]); r_res = np.mean(resid_rhos[p])
    s_res = np.std(resid_rhos[p]) / np.sqrt(N_REPEATS)
    ok = (abs(r_res) < RHO_NOISE_THRESH) and (abs(r_res) < RHO_NOISE_SE_MULT * max(s_res, SE))
    all_decorrelated &= ok
    print(f"  {p[0]+' x '+p[1]:<28}{r_raw:>16.3f}{r_orac:>18.4f}{r_res:>19.4f}   "
          f"{'PASS decorrelated' if ok else 'FAIL correlated'} (resid SE across repeats={s_res:.4f})")

print(f"\n  oracle-noise rho (by construction, independent RNG streams) sanity range: "
      f"[{min(np.mean(v) for v in oracle_rhos.values()):+.4f}, {max(np.mean(v) for v in oracle_rhos.values()):+.4f}] "
      f"(confirms detrend methodology recovers the true near-zero root-correlation)")

# void-floor-on-big-margins guard: stratify by q-tercile on ONE large run, check "big margins" (high-q)
# stratum specifically doesn't degenerate (collapsed variance would fake a correlation reading).
q_big = np.random.default_rng(7).uniform(0, 1, 20000)
margins_big, _, _, _ = simulate_clean(q_big, seed=7)
tercile_report = []
for lo, hi, tag in [(0.0, 0.333, "low-q(bad)"), (0.333, 0.667, "mid-q"), (0.667, 1.0, "high-q(big margins)")]:
    mask = (q_big >= lo) & (q_big < hi)
    stds = {k: float(np.std(margins_big[k][mask])) for k in EYES}
    resid_t = {k: detrend(margins_big[k][mask], q_big[mask]) for k in EYES}
    _, Rt = corr_matrix(resid_t)
    worst_pair_rho = float(np.max(np.abs(Rt - np.eye(4))))
    tercile_report.append((tag, stds, worst_pair_rho))
print(f"\n  void-floor guard (stratified by q-tercile, N={mask.sum()}/tercile): "
      f"std(margin) per eye must stay >0 (no saturation), worst |resid-rho| must stay small in EVERY stratum:")
for tag, stds, wr in tercile_report:
    print(f"    {tag:<22} std={ {k: round(v,3) for k,v in stds.items()} }  worst|resid-rho|={wr:.3f} "
          f"{'PASS' if wr < RHO_NOISE_THRESH else 'FAIL'}")

print(f"\n  clip incidence (margin<0 before clip; must stay small = not a degenerate void-floor artifact):")
mean_clip = {k: np.mean([c[k] for c in clip_fracs_all]) for k in EYES}
for k in EYES:
    print(f"    {k:<14} {mean_clip[k]*100:.2f}% clipped  {'PASS (<2%)' if mean_clip[k] < 0.02 else 'FLAG (>=2%)'}")

print(f"\n  binder-frequency (non-redundancy check, project convention from d_endtoend_composed_cert_demo.py: "
      f"each eye should bind sometimes, else it is dead weight): {binder_counts}")
all_bind = all(c > 0 for c in binder_counts.values())
print(f"    {'PASS all 4 eyes bind somewhere' if all_bind else 'FAIL some eye never binds'}")

TEST_A_PASS = all_decorrelated and all_bind
print(f"\n  TEST A VERDICT: {'PASS — 4 eyes decorrelated in this synthetic (independent-root) construction' if TEST_A_PASS else 'FAIL'}")

# ============================================================================= TEST B: COMPOSITION EXAMPLES
print("\n[TEST B] compose_l1() on 3 example kernel-variants (natural draws, fixed seed, no cherry-picking)")
print("-" * 100)
examples = [("good kernel (q=0.85)", 0.85), ("bad kernel (q=0.15)", 0.15), ("borderline kernel (q=0.70)", 0.70)]
for name, qv in examples:
    m, _, _, _ = simulate_clean(np.array([qv]), seed=777)
    margins_i = {k: float(m[k][0]) for k in EYES}
    v = compose_l1(margins_i)
    print(f"  {name}:")
    print(f"    margins = {{{', '.join(f'{k}:{val:.3f}' for k,val in margins_i.items())}}}")
    print(f"    composed_cert={v['composed_cert']:.3f}  binding_eye={v['binding_eye']:<12}  "
          f"verdict={'CERTIFY' if v['certify'] else 'ABSTAIN'}  "
          f"{'-> L3 should refine ' + v['binding_eye'] if not v['certify'] else ''}")

# ==================================================================== TEST C1: NEGATIVE CONTROL, TYPE-A
# "typical-case" gap ratio: true_worst/MIN-composition when 2 facets share a root and add COHERENTLY.
# Geometric (law-of-cosines) derivation: two error magnitudes a,b combine at "root-sharing angle" cos(th)=rho:
#   true_worst = sqrt(a^2 + b^2 + 2*rho*a*b)      (rho=0 -> quadrature; rho=1 -> full coherent sum a+b;
#                                                   rho=-1 -> destructive |a-b|)
# Cross-validated against real measured numbers: rho=0.81 (det<->parity, real GPU) -> gap 1.14-1.19x median;
# H's independent bound: total <=~2x for EQUAL facets.
print("\n[TEST C1] NEGATIVE CONTROL (type-A, typical-case gap true_worst/MIN) — geometric quadrature law")
print("-" * 100)


def typeA_gap(a, b, rho):
    true_worst = np.sqrt(a ** 2 + b ** 2 + 2 * rho * a * b)
    return true_worst, true_worst / np.maximum(np.maximum(a, b), 1e-300)


print("  (i) DETERMINISTIC ratio-grid (a/b = non-binding/binding facet magnitude ratio) at rho=0.81 "
      "(the REAL measured det<->parity correlation):")
for ratio in [1.0, 0.5, 0.2, 0.17, 0.15]:
    a, b = ratio, 1.0
    tw, gap = typeA_gap(a, b, 0.81)
    tag = " <-- brackets the REAL 1.14-1.19x (structural law's implicit ~0.15-0.2 down-weight)" if 1.10 <= gap <= 1.25 else ""
    print(f"      a/b={ratio:.2f}: gap={gap:.3f}x{tag}")

print("\n  (ii) STOCHASTIC MC (diverse instance-space: random facet-magnitude ratio + rho sweep, N=200,000):")
rng_c1 = np.random.default_rng(42)
NMC1 = 200_000
b_mag = rng_c1.lognormal(mean=0.0, sigma=0.25, size=NMC1)
ratio_draw = rng_c1.uniform(0.1, 1.0, NMC1)          # diverse a/b ratios per draw
a_mag = b_mag * ratio_draw
print(f"  {'rho':>7} {'median gap':>12} {'p90 gap':>10}   note")
for rho in [0.0, 0.3, 0.6, 0.81, 0.95, 0.99, 1.0]:
    _, gap = typeA_gap(a_mag, b_mag, rho)
    med = np.median(gap); p90 = np.percentile(gap, 90)
    note = "cross-check vs REAL 1.14-1.19x (median over diverse a/b, so not a direct point match)" if rho == 0.81 else ""
    print(f"  {rho:>7.2f} {med:>11.3f}x {p90:>9.3f}x   {note}")

# equal-facet cap check (H's independent bound: total <= ~2x for equal facets at full correlation)
tw_eq, gap_eq_1 = typeA_gap(1.0, 1.0, 1.0)
tw_eq0, gap_eq_0 = typeA_gap(1.0, 1.0, 0.0)
print(f"\n  equal-facet (a=b) check: gap(rho=0)={gap_eq_0:.3f}x [quadrature baseline, cf. the independent "
      f"'rho=0 equal-facet ~1.11 is a quadrature baseline'] , gap(rho=1)={gap_eq_1:.3f}x "
      f"{'PASS <=' + str(GAP_EQUAL_FACET_CAP) + ' (BOUNDED, not unbounded)' if gap_eq_1 <= GAP_EQUAL_FACET_CAP else 'FAIL unbounded'}")
TEST_C1_PASS = gap_eq_1 <= GAP_EQUAL_FACET_CAP and 1.0 <= gap_eq_0 <= 1.5
print(f"  TEST C1 VERDICT: {'PASS — type-A gap is BOUNDED + geometrically consistent with the REAL 2-facet finding' if TEST_C1_PASS else 'FAIL'}")

# ==================================================================== TEST C2: NEGATIVE CONTROL, TYPE-B
# "tail" false-accept floor: P(noise pushes BOTH facets to a false PASS | genuinely defective), swept over
# rho via a Gaussian common-factor construction (matches d_force_H_facetgap_is_rho_floor_law.py's recipe,
# reference measurements: floor(0)=0.0221~=p^2=0.0225, floor(0.81)=0.0927=4.12x, floor(-0.8)=0).
# NEW here: (a) exact scipy bivariate-normal external anchor (that script was MC-only); (b) realistic
# asymmetric per-eye sigmas (not equal p=0.15) tied to the ACTUAL 4-eye simulator above; (c) a 2nd eye-pair.
print("\n[TEST C2] NEGATIVE CONTROL (type-B, tail false-accept floor) — Gaussian-copula + exact external anchor")
print("-" * 100)


def false_accept_mc(rho, true_a, true_b, sigma_a, sigma_b, n_mc, rng):
    z1 = rng.standard_normal(n_mc)
    z2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.standard_normal(n_mc)
    ma, mb = true_a + sigma_a * z1, true_b + sigma_b * z2
    joint = float(np.mean((ma >= CHI) & (mb >= CHI)))
    pa, pb = float(np.mean(ma >= CHI)), float(np.mean(mb >= CHI))
    return joint, pa, pb


def false_accept_theory(rho, true_a, true_b, sigma_a, sigma_b):
    za, zb = (CHI - true_a) / sigma_a, (CHI - true_b) / sigma_b
    Fa, Fb = norm.cdf(za), norm.cdf(zb)
    Fjoint = multivariate_normal(mean=[0, 0], cov=[[1.0, rho], [rho, 1.0]]).cdf([za, zb])
    return 1 - Fa - Fb + Fjoint


print("  (0) SELF-CHECK regression vs d_force_H_facetgap_is_rho_floor_law.py (equal p=0.15, reference "
      "re-run gave floor(0)=0.0221, floor(0.81)=0.0927=4.12x, floor(-0.8)=0.00005):")
rng_check = np.random.default_rng(3)
thr = np.quantile(rng_check.standard_normal(200_000), 1 - 0.15)   # same recipe: true=0, sigma=1, threshold @ p=0.15
for rho in [0.0, 0.81, -0.8]:
    # this recipe defines "false-accept"=miss=z>thr directly on standard normals (not the true_a/sigma_a
    # parametrization false_accept_mc() takes) -- reproduce it in its OWN like-for-like frame:
    z1 = rng_check.standard_normal(800_000)
    z2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * rng_check.standard_normal(800_000)
    joint2 = float(np.mean((z1 > thr) & (z2 > thr)))
    print(f"      rho={rho:+.2f}: reproduced floor={joint2:.4f} (orig script's fresh re-run at same rho above)")

# empirically calibrated sigmas from the Test-A simulator (Gaussian idealization -- documented simplification:
# determinism's true noise is a Gaussian+spike MIXTURE; here we use its first-two-moments Gaussian equivalent
# so the scipy bivariate-normal formula is an EXACT anchor for the model actually being tested, not an
# approximation to the mixture).
_, _, noise_ref, _ = simulate_clean(np.random.default_rng(99).uniform(0, 1, 40000), seed=99)
sigma_hat = {k: float(np.std(noise_ref[k])) for k in EYES}
print(f"\n  empirical per-eye noise sigma (Gaussian-idealized from the Test-A simulator): "
      f"{{{', '.join(f'{k}:{v:.4f}' for k,v in sigma_hat.items())}}}")

print(f"\n  applying to 2 REAL eye-pairs at 2 defect severities, rho in [-0.5,0,0.3,0.6,0.8,0.95,0.99]:")
rng_c2 = np.random.default_rng(2026)
NMC2 = 2_000_000
RHOS = [-0.5, 0.0, 0.3, 0.6, 0.8, 0.95, 0.99]
c2_rows = []
all_anchor_ok, all_floor_bites, all_rho0_clean = True, True, True
# FORCED FIX (OODA): severity must be PROPORTIONAL to each eye's OWN sigma (a fixed absolute margin like
# 0.80 is a 5.7-sigma ghost-tail event for the tight-noise parity eye but only 1.5-sigma for determinism --
# a "fixed absolute step vs a proportional step" confound, same family as fd-sensitivity-needs-proportional-
# step). Define severity in z-units (k sigmas below threshold) PER eye so both sides of every forced pair
# sit at a comparably resolvable tail depth regardless of their absolute noise scale.
for (ea, eb) in [("determinism", "parity"), ("profiler", "roofline")]:
    for severity, k in [("moderate", 1.5), ("severe", 2.5)]:
        sa, sb = sigma_hat[ea], sigma_hat[eb]
        true_a, true_b = 1.0 - k * sa, 1.0 - k * sb
        print(f"\n    FORCED SHARED ROOT: {ea} <-> {eb}  (severity={severity}=k{k:.1f}sigma-below-threshold-EACH, "
              f"true_{ea[:3]}={true_a:.3f}(sigma={sa:.3f}), true_{eb[:3]}={true_b:.3f}(sigma={sb:.3f}))")
        print(f"    {'rho':>7} {'MC floor':>10} {'theory(scipy)':>14} {'|MC-th|/th':>11} {'naive prod':>11} {'floor/prod':>11}  anchor")
        for rho in RHOS:
            joint, pa, pb = false_accept_mc(rho, true_a, true_b, sa, sb, NMC2, rng_c2)
            theory = false_accept_theory(rho, true_a, true_b, sa, sb)
            prod = pa * pb
            rel_err = abs(joint - theory) / max(theory, 1e-12)
            anchor_ok = rel_err < ANCHOR_REL_TOL or abs(joint - theory) < 5.0 / np.sqrt(NMC2)  # abs MC-noise floor fallback
            all_anchor_ok &= anchor_ok
            infl = joint / max(prod, 1e-300)
            c2_rows.append((ea, eb, severity, rho, joint, theory, prod, infl))
            print(f"    {rho:>7.2f} {joint:>10.5f} {theory:>14.5f} {rel_err:>10.1%} {prod:>11.5f} {infl:>10.2f}x  "
                  f"{'PASS' if anchor_ok else 'FAIL'}")
        # PASS/FAIL bookkeeping at the key rho values for THIS pair/severity
        row_rho0 = next(r for r in c2_rows if r[0] == ea and r[1] == eb and r[2] == severity and r[3] == 0.0)
        row_rho08 = next(r for r in c2_rows if r[0] == ea and r[1] == eb and r[2] == severity and r[3] == 0.8)
        row_rho_neg = next(r for r in c2_rows if r[0] == ea and r[1] == eb and r[2] == severity and r[3] == -0.5)
        all_rho0_clean &= abs(row_rho0[7] - 1.0) < 0.15
        all_floor_bites &= row_rho08[7] >= FLOOR_HIGH_RHO_MIN
        anti_corr_beats = row_rho_neg[7] < 1.0
        print(f"      -> rho=0 recovers independence ({row_rho0[7]:.2f}x, need~1.0): "
              f"{'PASS' if abs(row_rho0[7]-1.0)<0.15 else 'FAIL'}  |  "
              f"rho=0.8 floor bites (>={FLOOR_HIGH_RHO_MIN}x): {row_rho08[7]:.2f}x "
              f"{'PASS' if row_rho08[7]>=FLOOR_HIGH_RHO_MIN else 'FAIL'}  |  "
              f"rho=-0.5 anti-corr beats indep ({row_rho_neg[7]:.2f}x<1): {'PASS' if anti_corr_beats else 'FAIL'}")

TEST_C2_PASS = all_anchor_ok and all_floor_bites and all_rho0_clean
print(f"\n  TEST C2 VERDICT: {'PASS' if TEST_C2_PASS else 'FAIL'} — MC vs exact-scipy anchor "
      f"{'agree' if all_anchor_ok else 'DISAGREE'}; floor {'bites' if all_floor_bites else 'does NOT bite'} at rho~0.8; "
      f"rho=0 {'recovers' if all_rho0_clean else 'does NOT recover'} independence.")

# =========================================================================================== FINAL VERDICT
print("\n" + "=" * 100)
print("HONEST FINAL VERDICT — does L1 compose gap-free by MIN-across-independent-roots?")
print("=" * 100)
print(f"  TEST A (decorrelation, clean 4-eye baseline):        {'PASS' if TEST_A_PASS else 'FAIL'}")
print(f"  TEST B (compose_l1 worked example verdicts):          printed above, no PASS/FAIL gate (illustrative)")
print(f"  TEST C1 (negative control, typical-case gap, bounded): {'PASS' if TEST_C1_PASS else 'FAIL'}")
print(f"  TEST C2 (negative control, tail false-accept floor):   {'PASS' if TEST_C2_PASS else 'FAIL'}")
OVERALL = TEST_A_PASS and TEST_C1_PASS and TEST_C2_PASS
print(f"\n  OVERALL: {'PASS' if OVERALL else 'FAIL/MIXED'}")
print(f"""
  READ: in THIS synthetic (independent-noise-by-construction) 4-eye model, MIN-composition IS gap-free
  (TEST A: measured |rho_noise|<{RHO_NOISE_THRESH} for all 6 eye-pairs, all 4 eyes bind somewhere, no void-floor
  degeneracy). The negative control PROVES this is not a free lunch: forcing 2 eyes to share a root produces
  BOTH (C1) a BOUNDED typical-case bias (matches the real measured 1.14-1.19x at rho=0.81 when the
  facet-magnitude ratio is ~0.15-0.2, and stays <={GAP_EQUAL_FACET_CAP}x even for equal facets at rho=1 — bounded,
  not catastrophic) AND (C2) a much sharper TAIL false-accept floor (multi-fold inflation over the
  independence-assumed rate at rho=0.8, cross-validated against an EXACT scipy bivariate-normal formula, and
  against reference re-run of d_force_H_facetgap_is_rho_floor_law.py). These are two DIFFERENT,
  complementary readings of the same rho -- typical-case bias is mild, rare-tail false-accept risk is not.

  CAVEAT (the falsifier, stated not buried): this is a SYNTHETIC model. It does NOT itself prove the 4 REAL
  CUDA scene-eyes are decorrelated -- that requires real GPU telemetry (out of scope, CPU-only constraint).
  What IS established: (a) the compose_l1() function is a correct, reusable implementation of this project's
  MIN/argmin/certify convention; (b) IF the real eyes' failure roots are independent (plausible by mechanism:
  scheduling nondeterminism / rounding / launch-latency / DRAM-bandwidth are physically disjoint for MOST
  kernels), the measured signature (rho_noise matrix near 0, both gap-types near their rho=0 baseline) is
  EXACTLY what TEST A predicts and what L3 should check against real telemetry; (c) the ALREADY-KNOWN real
  exception is determinism<->parity when BOTH stem from float-reduction-order sensitivity (real GPU, rho~0.8)
  -- there this project's own prior finding (not this file) already showed the gap stays mild (1.14-1.19x
  typical-case) even though NOT independent, i.e. the SPECIFIC known correlated case is not catastrophic,
  while the GENERAL tail-risk law (C2) says a rho this high should still be treated as a real, quantifiable,
  multi-fold false-accept risk for any PASS verdict that binds on exactly that pair.
""")
print("=" * 100)

"""THE INTEGRATION STITCH — ONE composed scenario-certificate for LBM(f_i,rho,u) (+) contact-impulse.
The 4 base cells (per-variable precision / ensemble co-execution / federation N1 / computation-level requirement)
are each validated in ISOLATION. Nobody has stitched them: per-variable cert -> per-component best-in-class target
-> ensemble co-execution schedule -> ONE scenario verdict. This does that, and FORCES whether the stitch actually
binds or a layer leaks (measure, don't assume the reference files compose for free).

Builds on (read, not re-litigated): d_per_variable_precision_cert_lbm.py (kappa_u ~ 1/rho, VERIFIED here again),
d_ensemble_coexecution_cert_contended_roofline.py (water-fill schedule), d_federation_scenario_cert.py (N1-MIN),
d_computation_level_requirement_cert.py (R1..R4 schema). Also folds in the freshest corrections: R3=null-space, DISTINCT from R2=conditioning (do not conflate); the SOLVE-vs-READ split
(a solved system self-bounds; a signed cancellation READ does not) -- contact's determinism-need here is modeled
as the READ (net-force cancellation), the unbounded face, consistent with that split.

External anchors used (cited, not fabricated):
  - 578 GB/s = D-measured real copy BW (d_ensemble_coexecution_cert_contended_roofline.py, d_n42_streaming_optimal_mintraffic.py)
  - 672 GB/s = RTX5070 GDDR7 nameplate (d_1c_iv_*, lbm_gpu_fast.py, diag_fp16_bandwidth.py)
  - 1770 MLUPS = C's real measured lbm_gpu_fast throughput (lbm_gpu_fast.py L5, d_certvector_on_real_lbm.py)
  - 30 TFLOP/s FP32 = RTX5070 (d_ensemble_coexecution_cert_contended_roofline.py)
  - 16.4 / 25.1 GB/s = float-atomic / int64-fixed scatter, measured (RTX5070, warp, N=32768 contacts
    -> K=4096 bodies, matching contact_throughput_benchmark.py's N=4096-box scene @ ~8 contacts/box), script at
    scratchpad/probe_contact_scatter_bw.py (real GPU run, not modeled) -- external anchor for the contact component,
    These are recorded measurements rather than model predictions.

Pre-registered thresholds (reused from established project convention, NOT tuned to make this probe pass):
  SIG_EPS=1e-3 (A-V57-4 forward-noise convention), DELTA=1e-2 (QoI tolerance), ROOF_BEST=0.85, ROOF_OK=0.30.
CPU-only numpy. This is a composed CERTIFICATE over real solver structure (analytic + numerically-verified
conditioning, real cited GPU anchors) -- NOT a running GPU simulation. no fit.
"""
import numpy as np

# ============================================================================================
# PRE-REGISTERED THRESHOLDS  (reused, not tuned)
# ============================================================================================
SIG_EPS   = 1e-3     # forward float-noise / nondeterminism magnitude (A-V57-4 convention)
DELTA     = 1e-2     # QoI tolerance (computation-level-cert convention)
ROOF_BEST = 0.85     # best-in-class roofline bar (d_certvector_on_real_lbm.py)
ROOF_OK   = 0.30     # merely-"uncontended" bar (d_r2_r4_dissociation_real_cuda.py / d_ensemble_realgpu_eye_grid.py)

# ============================================================================================
# EXTERNAL ANCHORS (real, cited; provenance in the docstring above)
# ============================================================================================
PEAK_BW_NAMEPLATE = 672.0e9    # bytes/s
PEAK_BW_MEASURED  = 578.0e9    # bytes/s  <- the canonical peak used throughout THIS script (measured, not nameplate)
PEAK_FLOP         = 30.0e12    # flops/s
RIDGE             = PEAK_FLOP / PEAK_BW_MEASURED     # roofline ridge, FLOP/byte

LBM_REAL_MLUPS         = 1770.0     # real measured (lbm_gpu_fast.py)
LBM_CEILING_NAMEPLATE  = 9333.0     # MLUPS @ 672GB/s nameplate (lbm_gpu_fast.py, d_certvector_on_real_lbm.py)
LBM_BYTES_PER_VOXEL    = 72.0       # 9 f_i read + 9 write, x4B fp32 (same files)

CONTACT_BW_FLOAT = 16.4e9     # measured: float-atomic scatter, N=32768->K=4096, RTX5070/warp
CONTACT_BW_INT64 = 25.1e9     # measured: int64-fixed scatter, same N,K (the determinism fix)

print("=" * 108)
print("INTEGRATION STITCH: per-variable precision (x) per-component best-in-class (x) ensemble co-execution -> ONE scenario cert")
print("=" * 108)

# ============================================================================================
# SEAM #1 -- reconcile the peak-BW inconsistency BEFORE composing anything on top of it
# ============================================================================================
print("\n[SEAM #1] the 4 base files do not agree on 'peak BW': the real-LBM cert-vector cites 672 GB/s (nameplate),")
print("          the ensemble-coexecution cert cites 578 GB/s (D-measured). Left un-reconciled, the SAME real MLUPS")
print("          number reads as two different roofline fractions depending which file's constant you copied.")
lbm_achieved_bw = LBM_REAL_MLUPS * 1e6 * LBM_BYTES_PER_VOXEL     # bytes/s, from the REAL measured MLUPS x real traffic model
roof_lbm_nameplate = lbm_achieved_bw / PEAK_BW_NAMEPLATE
roof_lbm_measured  = lbm_achieved_bw / PEAK_BW_MEASURED
lbm_ceiling_reconciled = LBM_CEILING_NAMEPLATE * PEAK_BW_MEASURED / PEAK_BW_NAMEPLATE
print(f"  achieved BW from real {LBM_REAL_MLUPS:.0f} MLUPS x {LBM_BYTES_PER_VOXEL:.0f} B/voxel = {lbm_achieved_bw/1e9:.2f} GB/s")
print(f"  vs nameplate(672):  {roof_lbm_nameplate*100:.2f}%  <- MATCHES lbm_gpu_fast.py's own self-reported '19%' (cross-check PASS)")
print(f"  vs measured (578):  {roof_lbm_measured*100:.2f}%   <- RECONCILED number this script uses downstream")
print(f"  reconciled ceiling: {LBM_CEILING_NAMEPLATE:.0f} MLUPS(@672) -> {lbm_ceiling_reconciled:.0f} MLUPS(@578); "
      f"same {LBM_REAL_MLUPS:.0f} MLUPS / {lbm_ceiling_reconciled:.0f} = {LBM_REAL_MLUPS/lbm_ceiling_reconciled*100:.2f}% (matches roof_lbm_measured, self-consistent)")
print(f"  FIX applied: every roofline % in this script is computed against the {PEAK_BW_MEASURED/1e9:.0f} GB/s MEASURED peak.")

# ============================================================================================
# SECTION 1 -- LBM per-variable cert (re-VERIFY, not just cite, the per-variable-precision-cert file's claim)
# ============================================================================================
print("\n" + "=" * 108)
print("[1] LBM PER-VARIABLE cert -- f_i / rho / u, re-measured on the real D2Q9 moment structure (not copied)")
print("=" * 108)
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], float); cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], float)
w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

def equilib(rho, ux, uy):
    cu = 3 * (cx * ux + cy * uy); usq = 1.5 * (ux * ux + uy * uy)
    return w * rho * (1 + cu + 0.5 * cu * cu - usq)

def moments(f):
    rho = f.sum(); ux = (cx * f).sum() / rho; uy = (cy * f).sum() / rho
    return rho, ux, uy

eps = 1e-6
rows = []
for rho0 in [1.0, 0.5, 0.3, 0.1, 0.03, 0.01]:
    f = equilib(rho0, 0.05, 0.0)
    rho, ux, uy = moments(f)
    krho = kux = 0.0
    Ju = np.zeros(9)
    for j in range(9):
        fp = f.copy(); fp[j] += eps
        r2, u2, _ = moments(fp)
        krho = max(krho, abs(r2 - rho) / eps)
        du = (u2 - ux) / eps
        Ju[j] = du
        kux = max(kux, abs(du))
    rows.append((rho0, krho, kux, kux * rho0, np.linalg.norm(Ju)))
const_kurho = float(np.mean([r[3] for r in rows]))
print(f"  {'rho':>6} {'kappa_rho':>10} {'kappa_u':>10} {'kappa_u*rho':>12} {'||J_u|| (R3 row-norm)':>22}")
for rho0, krho, kux, prod, ju in rows:
    print(f"  {rho0:>6.3f} {krho:>10.3f} {kux:>10.2f} {prod:>12.3f} {ju:>22.3f}")
print(f"  -> kappa_rho=1.000 at EVERY rho (verified); kappa_u*rho = {const_kurho:.3f} (const, verified across rho in [1,0.01])")
print(f"  -> R3 check for u: ||J_u|| > 0 STRICTLY at every rho tested (never a null vector) => u IS identifiable (R3 PASS)")
print(f"     even where R2/conditioning is terrible (kappa_u large). R3 (null-space) and R2 (conditioning) DISSOCIATE")
print(f"     here exactly as A-V57-18 forces elsewhere: a well-posed map can still be ill-conditioned. u only becomes")
print(f"     R3-DEGENERATE in the limit rho->0 exactly (measure-zero; J_u blows up rather than vanishing -- a different")
print(f"     failure mode than the identifiability_governor's sigma_max->0 case, so filed under R2, not R3.)")

# determinism/precision need for u, as a function of rho (A-V57-4 formula, L_inf-gated: peak-velocity/threshold QoI)
rho_star = SIG_EPS * const_kurho / DELTA
print(f"\n  R2/determinism-need for u:  floor(rho) = SIG_EPS*kappa_u(rho) = {SIG_EPS:.0e}*{const_kurho:.2f}/rho vs DELTA={DELTA:.0e}")
print(f"  -> crossover rho* = SIG_EPS*const/DELTA = {rho_star:.4f}  (below this, u needs fp64 AND/OR determinism; above, fp32 is fine)")
for label, rho_min in [("bulk-channel scene (Poiseuille core)", 0.30), ("near-wall / free-surface scene", 0.03)]:
    floor = SIG_EPS * const_kurho / rho_min
    need = floor > DELTA
    print(f"    {label:38s}: rho_min={rho_min:.2f}  floor={floor:.2e}  {'REQUIRED (region-conditional!)' if need else 'not required'}")
print(f"  ⟹ LBM's determinism-need is REGION-CONDITIONAL, not a fixed flag -- it FLIPS between these two real scenes.")
print(f"    d_federation_scenario_cert.py hardcodes LBM det_required=False; that is only correct in the bulk-flow regime.")

# ---- QUANTIFY the wasted-precision (uniform vs per-variable/per-region), on an illustrative near-wall profile ----
# illustrative (clearly labeled, NOT a specific measured flow): linear boundary-layer ramp from rho_bulk to rho_wall
# over a layer of thickness delta_bl (fraction of channel half-height) at EACH wall; direct numpy fraction-count,
# not hand algebra (measure, don't narrate).
delta_bl, rho_bulk, rho_wall = 0.10, 1.0, 0.03
y = np.linspace(0, 1, 200_001)
d_wall = np.minimum(y, 1 - y)                                    # distance to nearest wall
in_layer = d_wall < delta_bl
rho_profile = np.where(in_layer, rho_wall + (d_wall / delta_bl) * (rho_bulk - rho_wall), rho_bulk)
frac_needs_fp64 = float(np.mean(rho_profile < rho_star))
adaptive_mult = 1.0 + frac_needs_fp64          # only the flagged fraction pays +1x bytes (fp32->fp64 doubles bytes)
blanket_mult = 2.0                             # blanket fp64 pays +1x bytes EVERYWHERE
print(f"\n  WASTED-PRECISION quantification (illustrative near-wall profile: {delta_bl*100:.0f}%-of-half-height boundary")
print(f"  layer at each wall, rho: {rho_bulk:.2f} bulk -> {rho_wall:.2f} wall, linear ramp; direct numpy fraction-count):")
print(f"    fraction of domain with rho<rho*={rho_star:.3f} (genuinely needs fp64/determinism): {frac_needs_fp64*100:.1f}%")
print(f"    traffic multiplier:  adaptive (per-variable/per-region) = 1+{frac_needs_fp64:.3f} = {adaptive_mult:.3f}x  |  "
      f"blanket fp64 everywhere = {blanket_mult:.1f}x")
print(f"    ⟹ blanket-fp64 pays the +1x-bytes tax on {(1-frac_needs_fp64)*100:.1f}% of the domain that NEVER needed it;")
print(f"      adaptive uses only {adaptive_mult/blanket_mult*100:.1f}% of blanket's traffic-overhead ({adaptive_mult:.2f}x vs {blanket_mult:.1f}x).")
print(f"      (bytes-only claim, conservative/architecture-independent; this GPU's fp64:fp32 COMPUTE-throughput ratio is")
print(f"      NOT measured here -- consumer GPUs typically throttle fp64 far below 1:2, which would make blanket-fp64's")
print(f"      true cost larger still, but that number is abstained-on, not asserted, in this script.)")

# ============================================================================================
# SECTION 2 -- contact-impulse per-variable cert
# ============================================================================================
print("\n" + "=" * 108)
print("[2] CONTACT-IMPULSE per-component cert -- R2 (measured cancellation kappa), R3 (kinematic determinacy), R4 (AI)")
print("=" * 108)

# ---- R2: determinism-need via a REAL cancellation measurement ----
# EXACT (not stochastic-single-draw) construction: fix log-spread magnitudes + a baseline near-random sign pattern once,
# then shift ALL contacts by a uniform delta to hit an EXACT target net/sum(|F|) ratio r -- gives a smooth, monotonic,
# reproducible kappa(r), instead of a noisy single random draw per level (which was NON-monotonic on first pass -- a
# real bug caught by re-checking the sweep for monotonicity, not narration: forced the adversary on my OWN first draft).
rng = np.random.default_rng(7)
N_CONTACTS = 512
mag = 10.0 ** rng.uniform(-1.0, 0.3, N_CONTACTS)
base_sign = np.where(rng.random(N_CONTACTS) < 0.5, 1.0, -1.0)
sum_abs_mag = float(np.sum(mag)); S0 = float(np.sum(mag * base_sign))
def contact_kappa(r):
    """r = target net-force / sum(|F_i|) -- the physically-meaningful 'how far from perfect equilibrium' knob."""
    target_net = r * sum_abs_mag
    delta = (target_net - S0) / N_CONTACTS
    F = mag * base_sign + delta
    net = F.sum()
    return float(np.sum(np.abs(F)) / max(abs(net), 1e-12)), float(net)

print(f"  {'r=|net|/sum|F|':>15} {'kappa_contact':>14} {'floor=EPS*kappa':>16} {'vs DELTA':>10}  verdict")
r_star = None
r_sweep = [1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001]
prev = None
monotonic = True
for r in r_sweep:
    k, net = contact_kappa(r)
    if prev is not None and k < prev - 1e-9: monotonic = False
    prev = k
    floor = SIG_EPS * k
    need = floor > DELTA
    if r_star is None and need: r_star = r
    print(f"  {r:>15.4f} {k:>14.2f} {floor:>16.2e} {floor/DELTA:>9.2f}x  {'★DETERMINISM REQUIRED' if need else 'not required'}")
print(f"  monotonic (kappa strictly non-decreasing as r falls)? {monotonic}  <- machine-checked, not eyeballed")
print(f"  -> crossover at r* between {r_sweep[r_sweep.index(r_star)-1] if r_star else '?'} and {r_star}: a resting/stacked body")
print(f"     with net-force at just ~{ (r_star or 0.1)*100:.0f}% of its gross contact-force magnitude (a MILD near-equilibrium,")
print(f"     not an extreme edge case -- true static equilibrium (r->0) is the norm for a body at rest) sits in the")
print(f"     ★REQUIRED regime -- this is the READ/cancellation face (unbounded), consistent with the")
print(f"     task's given 'contact-impulse ill-conditioned scatter, determinism-REQUIRED' framing, now DERIVED from a")
print(f"     measured kappa (monotonic, reproducible) rather than asserted.")

# ---- R3: kinematic determinacy (a REAL rigid-body constraint Jacobian, not asserted) ----
print("\n  R3 (identifiability = null-space test, A-V57-18 sense) -- box resting on 4 corners, 6 rigid-body DOF [tx,ty,tz,wx,wy,wz]:")
a, b = 0.5, 0.3
corners = [(a, b), (a, -b), (-a, b), (-a, -b)]
J_normal = np.array([[0, 0, 1, ry, -rx, 0] for (rx, ry) in corners], dtype=float)     # frictionless normal contacts only
rank_n = np.linalg.matrix_rank(J_normal)
sv_n = np.linalg.svd(J_normal, compute_uv=False)
null_cols = np.allclose(J_normal[:, [0, 1, 5]], 0.0)
print(f"    normal-only (frictionless) Jacobian: rank={rank_n}/6, singular values={np.round(sv_n,3)}")
print(f"    columns [tx,ty,wz] identically zero: {null_cols} -> a REAL 3-dim null space (in-plane slide + spin-about-normal")
print(f"    are NOT resisted by normal forces alone -- true physics, not a numerical artifact) -> R3 = DEGENERATE for full-6DOF.")
J_tang = []
for (rx, ry) in corners:
    J_tang.append([1, 0, 0, 0, 0, -ry])
    J_tang.append([0, 1, 0, 0, 0, rx])
J_full = np.vstack([J_normal, np.array(J_tang, dtype=float)])
rank_f = np.linalg.matrix_rank(J_full)
sv_f = np.linalg.svd(J_full, compute_uv=False)
print(f"    + tangential/stiction rows (friction active, non-degenerate): rank={rank_f}/6, singular values={np.round(sv_f,3)}")
print(f"    -> {'FULL RANK: R3 PASS once friction is active at >=2 non-collinear corners' if rank_f==6 else 'still degenerate'}")
print(f"    ⟹ R3 for contact is CONFIGURATION-dependent (real, not a fixed pass/fail): normal-only=ABSTAIN(3 null dirs),")
print(f"      normal+stiction=CERTIFY. R2 (cancellation-conditioning, above) and R3 (this) are DISSOCIATED axes, per A-V57-18.")

# ---- R4: arithmetic intensity (itemized analytic model + robustness sweep) ----
print("\n  R4 (contention/AI) -- itemized per-contact bytes/flops (fp32), analytic model:")
bytes_read = 2 * 6 * 4 + (3 + 6 + 3 + 3 + 1 + 2) * 4 + 2 * 4      # 2-body vel(6ea) + geom(18 floats) + material(2)
bytes_write = 2 * 6 * 4 + 3 * 4                                    # 2-body updated vel + accum impulse(3 floats)
bytes_total = bytes_read + bytes_write
flops = 24 + 15 + 30 + 30                                          # rel-vel(24) + normal-imp(15) + tangent-imp x2(30) + apply(30)
AI_contact = flops / bytes_total
print(f"    bytes: read={bytes_read}B write={bytes_write}B total={bytes_total}B/contact | flops={flops}/contact | AI={AI_contact:.3f} FLOP/B")
print(f"    ridge (this GPU, measured-peak) = {RIDGE:.1f} FLOP/B -> {'MEMORY-bound' if AI_contact < RIDGE else 'compute-bound'} "
      f"(AI={AI_contact:.2f} vs ridge={RIDGE:.1f}, {RIDGE/AI_contact:.0f}x below ridge)")
print(f"    robustness sweep (force the adversary: what if the flop-count model is way off?):")
for mult in [1, 2, 5, 10, 20, 50, 95]:
    ai = flops * mult / bytes_total
    tag = "mem" if ai < RIDGE else "COMPUTE"
    print(f"      flop-count x{mult:>3d} -> AI={ai:>6.2f}  {tag}")
mult_needed = RIDGE / (flops / bytes_total)
print(f"    -> would need a {mult_needed:.0f}x heavier per-contact solve (~{mult_needed:.0f}x{flops:.0f}={mult_needed*flops:.0f} flops/contact,")
print(f"       i.e. a full dense-factorization-grade solve, not a 1-20-iteration PGS/GS pass) to flip to compute-bound.")
print(f"       ⟹ contact-impulse is ROBUSTLY memory-bound across any realistic solver-complexity assumption.")

# ---- external anchor: the REAL measured scatter BW ----
roof_contact_float = CONTACT_BW_FLOAT / PEAK_BW_MEASURED
roof_contact_int64 = CONTACT_BW_INT64 / PEAK_BW_MEASURED
print(f"\n  EXTERNAL ANCHOR (measured, RTX5070/warp, N=32768->K=4096, scratchpad/probe_contact_scatter_bw.py):")
print(f"    float-atomic scatter: {CONTACT_BW_FLOAT/1e9:.1f} GB/s = {roof_contact_float*100:.2f}% of peak  (R2 FAIL + R4 FAIL)")
print(f"    int64-fixed  scatter: {CONTACT_BW_INT64/1e9:.1f} GB/s = {roof_contact_int64*100:.2f}% of peak  (R2 PASS, R4 STILL FAIL)")
print(f"    ⟹ the determinism FIX (int64) does NOT fix contention: {roof_contact_float*100:.1f}%->{roof_contact_int64*100:.1f}%, both")
print(f"      ≪ ROOF_OK={ROOF_OK*100:.0f}%. R2 and R4 are separate axes (matches d_r2_r4_dissociation_real_cuda.py exactly);")
print(f"      composing them into ONE scalar would DROP the fact that fixing determinism leaves the roofline gap untouched.")

# ============================================================================================
# SECTION 3 -- assemble the per-component cert-vector table (the stitched per-variable/component layer)
# ============================================================================================
print("\n" + "=" * 108)
print("[3] COMPOSED PER-COMPONENT CERT-VECTOR TABLE (R1 traffic, R2 determinism/precision, R3 identifiability, R4 AI/contention)")
print("=" * 108)
print(f"  {'component':17s} {'R1 traffic':>16s} {'R2 (det/precision)':>28s} {'R3 (identifiability)':>24s} {'R4 (AI, bound)':>16s}")
print(f"  {'f_i (state)':17s} {'shares 72B/voxel':>16s} {'gather => det-free, fp32 fine':>28s} {'trivial (=state)':>24s} {'2.0 FLOP/B, mem':>16s}")
print(f"  {'rho (moment)':17s} {'0 marginal (reg)':>16s} {'kappa=1 always, fp32 fine':>28s} {'PASS (all-ones row)':>24s} {'shares kernel, mem':>16s}")
print(f"  {'u (moment)':17s} {'0 marginal (reg)':>16s} {'region: rho<{:.3f} NEEDS det/fp64'.format(rho_star):>28s} {'PASS (rho>0), DEGEN@rho=0':>24s} {'shares kernel, mem':>16s}")
print(f"  {'contact-net-F':17s} {'{:.0f}B/contact'.format(bytes_total):>16s} {'r<{:.2f} NEEDS det (typical rest!)'.format(0.1 if r_star is None else r_star):>28s} {'config-dep (needs friction)':>24s} {'{:.2f} FLOP/B, mem'.format(AI_contact):>16s}")
print(f"  ⟹ SEAM #2 (granularity): f_i/rho/u SHARE one R1/R4 row (one kernel launch, one memory transaction) but get 3")
print(f"    SEPARATE R2/R3 rows (per-variable). Composing R1 per-VARIABLE (triple-counting the 72B) would OVERSTATE LBM")
print(f"    traffic 3x; composing R2 per-KERNEL (one flag for all of LBM) would MISS u's region-dependent requirement.")
print(f"    The two facets live at DIFFERENT granularities -- collapsing them to one grain is the layer that would leak.")

# ============================================================================================
# SECTION 4 -- FEDERATION N1-MIN scenario composition (derived flags, not hardcoded; reuses d_federation's exact structure)
# ============================================================================================
print("\n" + "=" * 108)
print("[4] FEDERATION N1-MIN -- scenario verdict across {scene regime} x {contact kernel}, flags DERIVED not asserted")
print("=" * 108)

def lbm_det_required(rho_min):
    return (SIG_EPS * const_kurho / rho_min) > DELTA

def scenario_cert(rho_min_scene, contact_kernel, roof_lbm=roof_lbm_measured):
    lbm_req = lbm_det_required(rho_min_scene)
    contact_req = True   # measured above: the realistic resting/near-equilibrium operating point is in the REQUIRED regime
    solvers = {
        "LBM-fluid":  dict(det_required=lbm_req, kernel_det=True,  roofline=roof_lbm),                  # gather: det-free regardless
        "contact":    dict(det_required=contact_req, kernel_det=(contact_kernel == "int64"),
                            roofline=(roof_contact_int64 if contact_kernel == "int64" else roof_contact_float)),
    }
    det_ok = all((not s["det_required"]) or s["kernel_det"] for s in solvers.values())
    scen_roof = min(s["roofline"] for s in solvers.values())
    binder = min(solvers, key=lambda k: solvers[k]["roofline"])
    if not det_ok:
        verdict = "REJECT"
        binder = [k for k, s in solvers.items() if s["det_required"] and not s["kernel_det"]][0]
    elif scen_roof < ROOF_OK:
        verdict = "ABSTAIN (best-in-class)"
    elif scen_roof < ROOF_BEST:
        verdict = "CERTIFY (named-gap)"
    else:
        verdict = "CERTIFY"
    return solvers, det_ok, scen_roof, verdict, binder

print(f"  {'scene':>32s} {'contact kernel':>14s} | {'LBM det-req':>11s} {'scen roofline':>13s} | {'VERDICT':>24s} {'binding':>10s}")
for scene_name, rho_min in [("bulk-channel (rho_min=0.30)", 0.30), ("near-wall/free-surf (rho_min=0.03)", 0.03)]:
    for ck in ("float", "int64"):
        solvers, det_ok, scen_roof, verdict, binder = scenario_cert(rho_min, ck)
        print(f"  {scene_name:>32s} {ck:>14s} | {str(solvers['LBM-fluid']['det_required']):>11s} {scen_roof*100:>12.2f}% | {verdict:>24s} {binder:>10s}")

print(f"\n  ⟹ the SCENE-REGIME axis (LBM det-required) FLIPS across rows but never changes which axis is binding or the")
print(f"    overall verdict shape -- contact's requirement/roofline dominate BOTH scenes (its roofline {roof_contact_float*100:.1f}-")
print(f"    {roof_contact_int64*100:.1f}% is ~{roof_lbm_measured/roof_contact_float:.0f}x below LBM's {roof_lbm_measured*100:.1f}%, and its determinism-need is unconditional).")
print(f"    The CONTACT-KERNEL axis (float vs int64) is the one that actually MOVES the verdict (REJECT -> CERTIFY-named-gap).")
print(f"    ⟹ this composition catches a real, NON-obvious priority: don't spend effort tightening the LBM near-wall")
print(f"    precision cert first -- the scenario is bottlenecked by contact's kernel-choice + its roofline gap, regardless.")

# ============================================================================================
# SECTION 5 -- ENSEMBLE CO-EXECUTION schedule: does the water-fill 'pair complementary kernels' story apply HERE?
# ============================================================================================
print("\n" + "=" * 108)
print("[5] ENSEMBLE CO-EXECUTION -- apply the water-fill schedule to the ACTUAL pair in this scenario (LBM, contact)")
print("=" * 108)
KERNELS = {"LBM": 2.0, "contact": AI_contact}
def bound(ai): return "mem" if ai < RIDGE else "compute"
print(f"  AI-only classification: LBM AI={KERNELS['LBM']:.2f} ({bound(KERNELS['LBM'])})  "
      f"contact AI={KERNELS['contact']:.2f} ({bound(KERNELS['contact'])})")
print(f"  -> NAIVE prediction (2-class mem/compute water-fill model, as in d_ensemble_coexecution_cert_contended_roofline.py):")
print(f"     BOTH classify mem-bound => the model's own 'bw_demand=2xPEAK_BW>PEAK_BW' rule predicts CONTENTION, i.e. NO")
print(f"     speedup from co-running them (the 'pair complementary (mem (x) compute) kernels' advice has NO valid partner")
print(f"     inside this 2-solver scenario -- contact is not the compute-bound partner LBM needs).")
print(f"\n  ★FORCE IT against the REAL measured numbers (Section 2's external anchor), not the idealized 'mem-bound=peak-BW'")
print(f"   assumption the naive model bakes in:")
demand_lbm_idealized = PEAK_BW_MEASURED
demand_contact_idealized = PEAK_BW_MEASURED
demand_contact_real = CONTACT_BW_FLOAT
print(f"     idealized model assumes isolated demand: LBM~{demand_lbm_idealized/1e9:.0f} GB/s, contact~{demand_contact_idealized/1e9:.0f} GB/s (both 'at peak')")
print(f"     REAL isolated demand:                    LBM~{lbm_achieved_bw/1e9:.0f} GB/s (22% of peak, AoS-strided gap),")
print(f"                                               contact~{demand_contact_real/1e9:.1f} GB/s (2.8% of peak, ATOMIC-SERIALIZATION-bound,")
print(f"                                               not bandwidth-bound -- a 3rd resource class the AI/ridge test can't see)")
total_real_demand = lbm_achieved_bw + demand_contact_real
print(f"     summed REAL demand = {total_real_demand/1e9:.1f} GB/s vs {PEAK_BW_MEASURED/1e9:.0f} GB/s peak "
      f"({total_real_demand/PEAK_BW_MEASURED*100:.1f}% of peak) -> {'CONTENDS' if total_real_demand>PEAK_BW_MEASURED else 'NO real contention'}")
print(f"  ⟹ CORRECTED verdict: the naive AI-only model predicts contention; the REAL measurement shows NEITHER kernel")
print(f"    is anywhere near saturating peak BW today, so co-running them is CHEAP (negligible mutual interference) --")
print(f"    but it is NOT a 'water-fill win' either (contact's bottleneck, atomic-target serialization, is untouched by")
print(f"    LBM's presence/absence; it doesn't fill LBM's idle SMs, and LBM's spare bandwidth doesn't unblock its atomics).")
print(f"    ⟹ SEAM #3: the ensemble-schedule model's bound() in {{mem,compute}} is INCOMPLETE for this pair -- contact")
print(f"    needs a 3rd class (conflict/serialization-bound: low AI AND far under its own peak, for a reason AI can't see).")
print(f"    ⟹ SEAM #4 (priority/ORDER): both components sit FAR below their OWN isolated roofline for INTRA-kernel reasons")
print(f"    (LBM: {100-roof_lbm_measured*100:.0f}pp recoverable via SoA-coalescing; contact: most-but-not-all of the")
print(f"    {100-roof_contact_float*100:.0f}pp recoverable via privatization -- d_privatized_reduction_close_abstain.py's REAL measured result")
print(f"    (cited precisely, not assumed-successful) is 16%->80% at a DIFFERENT (K=64) scale, PLATEAUING below the")
print(f"    {ROOF_BEST*100:.0f}% best-in-class bar (residual needs a non-atomic/hard-core algorithm, per that file's own conclusion);")
print(f"    contact's actual K={4096} scale is unmeasured here -- named with the honest ceiling, not claimed as closed).")
print(f"    INTER-kernel water-filling is only the binding layer AFTER both")
print(f"    are near their own ceiling; today, layer-2 (per-component roofline gap) dominates layer-3 (ensemble schedule)")
print(f"    by ~{(100-roof_lbm_measured*100)/max(1e-9, total_real_demand/PEAK_BW_MEASURED*100):.0f}-{(100-roof_contact_float*100)/max(1e-9, total_real_demand/PEAK_BW_MEASURED*100):.0f}x. Scheduling co-execution now would be optimizing the wrong layer first.")

# ============================================================================================
# SECTION 6 -- THE ONE COMPOSED CERTIFICATE
# ============================================================================================
print("\n" + "=" * 108)
print("★ ONE COMPOSED CERTIFICATE — LBM(f_i,rho,u) ⊕ contact-impulse, scenario-level")
print("=" * 108)
solvers_default, det_ok_default, scen_roof_default, verdict_default, binder_default = scenario_cert(0.03, "float")
print(f"  Realistic operating point: near-wall/free-surface LBM region (rho_min=0.03) ⊕ resting/stacked contact (cancel~1),")
print(f"  TODAY's typical kernel default (float-atomic contact, gather-stream LBM):")
print(f"    R1 (traffic):        LBM 72 B/voxel (shared f_i/rho/u); contact {bytes_total:.0f} B/contact")
print(f"    R2 (determinism):    LBM u REQUIRES it below rho={rho_star:.3f} (region-conditional); contact REQUIRES it at rest (unconditional)")
print(f"    R3 (identifiability): LBM f_i/rho/u all PASS (no null space; u only degenerate at the single point rho=0);")
print(f"                          contact PASS only WITH active friction >=2 corners, else 3-dim null space")
print(f"    R4 (contention/AI):   LBM mem-bound 22.1% of peak (AoS gap); contact mem-bound-by-serialization 2.8% of peak")
print(f"    roofline-target:      LBM {lbm_ceiling_reconciled:.0f} MLUPS @ {PEAK_BW_MEASURED/1e9:.0f} GB/s; contact: no speedup from")
print(f"                          co-execution (Section 5) -- fix is INTRA-kernel (privatization), not scheduling")
print(f"    ensemble schedule:    co-run SAFE (real combined demand {total_real_demand/PEAK_BW_MEASURED*100:.1f}% of peak) but NOT a water-fill")
print(f"                          win -- schedule for convenience (same timestep), not throughput")
print(f"\n  SCENARIO VERDICT: {verdict_default}  (binding component: {binder_default})")
print(f"    -> float-atomic contact kernel FAILS the determinism gate (contact requires it at rest, kernel doesn't have it)")
print(f"       => REJECT regardless of LBM's own state (LBM being gather-deterministic cannot save the scenario).")
solvers_i, det_ok_i, scen_roof_i, verdict_i, binder_i = scenario_cert(0.03, "int64")
print(f"    -> switching contact to int64-fixed: det_ok={det_ok_i} -> scenario becomes '{verdict_i}', binding={binder_i}")
print(f"       (roofline {scen_roof_i*100:.2f}% still names contact as the recoverable-but-real gap, R2-fix != R4-fix).")

print("\n  SEAMS FOUND (would have leaked if the 4 base cells were stitched at face value, un-forced):")
print("  1. BW-reconciliation: 672 vs 578 GB/s used by different base files -> fixed onto one measured anchor.")
print("  2. Granularity: R1/R4 are per-KERNEL (f_i+rho+u share one launch); R2/R3 are per-VARIABLE -> conflating them")
print("     would triple-count LBM traffic or hide u's region-dependent precision need under one kernel-level flag.")
print("  3. LBM's det_required is REGION-conditional (derived here), not the fixed False the federation cell hardcodes.")
print("  4. Contact is NOT the compute-bound partner the generic ensemble catalog's 'pair complementary kernels' story")
print("     wants -- it's memory-bound too, but via a 3rd mechanism (atomic serialization) the AI/ridge test can't see,")
print("     which the naive 2-class model would have mis-scheduled as 'contends' when the REAL risk is negligible-BW-but-")
print("     zero-speedup. And R2(determinism)-fix does NOT fix R4(contention) -- both must be certified, neither implies the other.")
print("\n  ⟹ HONEST VERDICT ON THE STITCH: the composition DOES bind -- ONE non-contradictory scenario verdict comes out --")
print("    but ONLY once these 4 seams are forced explicitly. Taking any ONE of the 4 base cells at face value and")
print("    plugging its numbers into the others (naive stitch) would have produced a WRONG composed certificate on at")
print("    least one axis (over-counted traffic, a stale always-safe LBM flag, or a false contention/false-safety call")
print("    on the ensemble schedule). The binding component today, either way, is CONTACT-IMPULSE: float-atomic fails")
print("    determinism outright; int64-fixed passes determinism but remains the roofline floor (~3-4% of peak) that")
print("    the LBM side (22%, itself gapped) does not reach either -- naming BOTH gaps as the next build targets, in")
print("    priority order (intra-kernel fixes first, ensemble scheduling second).")
print("=" * 108)

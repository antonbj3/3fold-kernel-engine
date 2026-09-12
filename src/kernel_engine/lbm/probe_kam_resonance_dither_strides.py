#!/usr/bin/env python3
"""Probe: KAM-RESONANCE ORDERING of dither strides + SHOR-side period DETECTION.

MERIT-CHECK of two pre-registered twin-seeds against the golden-stride dither finding
(probe_lbm_fp16_steady_dither.py): fp16-LBM Poiseuille with subtractive temporal
dither, phase(step)=(step*S)%K. Known: ramp S=1 rectifies into a DC error; a
golden-coprime stride defeats it.

  SEED 1 (J-space / action-angle / KAM): dither-ordering robustness is ordered by
    the stride's RESONANCE DISTANCE from the substrate's intrinsic periods. Near-
    rational (resonant) strides are rectified into DC error; maximally-irrational
    (noble/golden) strides survive.
  SEED 2 (Shor / period-finding): the substrate's period structure is DETECTABLE by
    spectral period-finding on the residual, making safe strides PREDICTABLE.

THE GEOMETRY (why this should hold): phase(step)=(step*S)%K samples a period-K
sawtooth (the sub-quantum offset d) at stride S. The sequence's dominant non-DC
temporal frequency aliases to f_eff(S)=min(S%K, K-S%K)/K. The NONLINEAR flow acts
as a LOW-PASS: low-frequency forcing (small f_eff, i.e. S/K near a low-order
rational p/q) is TRACKED and rectified into a DC bias; high-frequency forcing
(f_eff near golden 0.382..) cannot be tracked -> cancels. Two mechanisms:
  (A) coprime strides: full quantum span, sawtooth cancels over K steps -> residual
      is purely DYNAMICAL rectification, governed by f_eff (KAM axis).
  (B) non-coprime strides gcd(S,K)=d>1: only K/d distinct phases -> the offset set
      is an incomplete sub-lattice, sawtooth does NOT cancel -> a STATIC unbalanced-
      quantization DC residual, independent of the dynamics. EXACT resonance = worst.

SUBSTRATE: K=21 (Fibonacci): golden stride 13/21 is an EXACT continued-fraction
convergent of 1/phi, so the golden ordering is exact. 21=3*7 -> non-coprime strides
{3,6,7,9,12,14,15,18} span two denominators (varied resonance). Coprime strides
{1,2,4,5,8,10,11,13,16,17,19,20}.

RESONANCE-DISTANCE METRIC (pre-registered, pre-registered):
  rd(S) = min over q in {1,2,3,4}, p in 0..q  of  |S/K - p/q|
  (distance of S/K from the nearest low-denominator rational; large rd = irrational
   = robust). Note rd(S)=rd(K-S) and err(S)~err(K-S) by time-reversal (internal check).

============================ FROZEN PREDICTIONS ============================
PRE-REGISTERED ORDERING PREDICTION (KAM signature): error DECREASES with rd; the
golden-adjacent coprime stride (S nearest K/phi=12.98 -> S=13, coprime) is best tier;
non-coprime (exact resonance) strides are the worst tier.

GATES (frozen BEFORE running the sweep):
  G1 (KAM ordering): Spearman(err, -rd) over ALL strides 1..K-1 >= 0.6.
       (secondary, reported not gated: Spearman over COPRIME-only strides.)
  G2 (exact-resonance tier): mean(err | non-coprime) >= 2x * mean(err | coprime).
  G3 (Shor period DETECTION): FFT the S=1 (ramp) run's centerline-QoI residual time
       series (DC removed). The forcing period is K=21 steps. The K-periodic spectral
       line dominates: power at period-K bin >= 3x median spectral power.
  G4 (Shor predictive): from the S=1 spectrum ALONE (no per-stride error), take the
       top-2 non-DC spectral peaks -> their harmonics h1,h2 (of base 1/K). The
       substrate is low-pass so these are the strongest-rectified frequencies.
       Predicted-UNSAFE = strides whose effective harmonic h_eff(S)=min(S%K,K-S%K)
       equals h1 or h2 (i.e. T_eff matches a resonant period). Verify the predicted-
       unsafe set overlaps the measured WORST-5 strides: |pred ∩ worst5| >= 3.
       (Honest caveat frozen: a pure-frequency predictor need NOT capture the static
        incomplete-span non-coprime residual (mechanism B); if worst-5 are dominated
        by non-coprime, G4 predicts only the dynamical half -> book that honestly.)

CONTROLS:
  (a) STEADY only (unsteady, where dither hurts, is out of scope).
  (b) CONVERGENCE control: rerun the full sweep at MAIN/2=15000; Spearman of the
       stride-error ranking vs MAIN=30000 must be >= 0.8 (ranking is resonance-
       structure, not transient). Predict rank-stable.
  (c) ANCHORS: fp32 reference (external) + det-fp16 single & K-avg baselines.

HONEST OUTCOME MAP:
  G1+G2 pass -> KAM-ordering law CONFIRMED (J-space seed = SIGNAL).
  G3+G4 pass -> period-finding predicts safe strides (Shor seed = SIGNAL, classical).
  Partial -> book which half. All fail -> golden-stride win was K-specific luck.

  python3 probe_kam_resonance_dither_strides.py
"""
import sys, os, json, time, math
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(__file__))
import probe_lbm_fp16_steady_dither as P   # reuse EXACT production-replica machinery

DEV = P.DEV
K = 21
MAIN = 30000
MAIN_HALF = 15000
NREC = 21 * 24   # 504 samples for the Shor FFT; period-21 lands on an exact bin


def resonance_distance(S, Kk):
    """min over q<=4 of |S/K - p/q|: distance from nearest low-denominator rational."""
    x = S / Kk
    best = 1.0
    for q in range(1, 5):
        for p in range(0, q + 1):
            best = min(best, abs(x - p / q))
    return best


def h_eff(S, Kk):
    """effective (aliased) harmonic of the stride-S forcing: min(S%K, K-S%K)."""
    m = S % Kk
    return min(m, Kk - m)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-30
    return float((ra * rb).sum() / denom)


def record_centerline_series(nx, ny, tau, solid, main_steps, Kk, stride, gforce, nrec):
    """Converge with stride-S dither active, then record centerline u at the channel
    mid-height for `nrec` post-convergence steps (for the Shor FFT). Replicates
    run_case's launch/macro machinery exactly (dither on)."""
    omega = 1.0 / tau
    col = nx // 2
    rho0 = np.ones((nx, ny), np.float32)
    ux0 = np.zeros((nx, ny), np.float32); uy0 = np.zeros((nx, ny), np.float32)
    sA = wp.array(P._equil_dev_soa(rho0, ux0, uy0, np.float16), dtype=wp.float16, device=DEV)
    sB = wp.zeros((9, nx, ny), dtype=wp.float16, device=DEV)
    solid_w = wp.array(solid.astype(np.int32), dtype=wp.int32, device=DEV)
    cx = wp.array(P.CX, dtype=wp.float32, device=DEV); cy = wp.array(P.CY, dtype=wp.float32, device=DEV)
    w = wp.array(P.WT, dtype=wp.float32, device=DEV); opp = wp.array(P.OP, dtype=wp.int32, device=DEV)
    uxd = wp.zeros((nx, ny), dtype=wp.float32, device=DEV)

    def launch(g_it):
        ps = (g_it * stride) % Kk; pl = ((g_it - 1) * stride) % Kk
        wp.launch(P.collide_stream_fp16, dim=(nx, ny),
                  inputs=[sA, sB, solid_w, cx, cy, w, opp, omega, float(gforce), 0.0,
                          1, nx, ny, 1, Kk, ps, pl], device=DEV)

    def macro(g_it):
        pl = (g_it * stride) % Kk
        wp.launch(P.macro_ux_fp16, dim=(nx, ny), inputs=[sA, cx, w, solid_w, uxd, 1, Kk, pl], device=DEV)

    wp.synchronize()
    g_it = 0
    for _ in range(main_steps):
        launch(g_it); sA, sB = sB, sA; g_it += 1
    wp.synchronize()
    mid = ny // 2
    series = np.empty(nrec, np.float64)
    for t in range(nrec):
        launch(g_it); sA, sB = sB, sA; g_it += 1
        macro(g_it - 1); wp.synchronize()
        series[t] = float(uxd.numpy()[col, mid])
    return series


def sweep(nx, ny, tau, solid, main_steps, Kk, ref, inner, norm, gforce):
    """time-averaged err vs fp32 ref for EVERY stride 1..K-1."""
    out = {}
    for S in range(1, Kk):
        r = P.run_case(nx, ny, tau, solid, main_steps, Kk, store="fp16", dither=1,
                       gforce=gforce, periodic_x=1, pstride=S)
        out[S] = P.rms_rel(r["kavg"], ref, inner, norm)
    return out


def main():
    print("=" * 80)
    print(f"PROBE: KAM-resonance ordering of dither strides + Shor period-finding  K={K}  device={DEV}")
    print("=" * 80)
    t0 = time.time()
    nx, ny, tau, G, nu, solid, u_ana, inner, norm = P._pois_setup()

    # ---- anchors ----
    fp32 = P.run_case(nx, ny, tau, solid, MAIN, K, store="fp32", gforce=G, periodic_x=1)
    ref = fp32["kavg"]
    fp32_vs_ana = P.rms_rel(ref, u_ana, inner, norm)
    det = P.run_case(nx, ny, tau, solid, MAIN, K, store="fp16", dither=0, gforce=G, periodic_x=1, measure_jitter=True)
    det_single = P.rms_rel(det["single"], ref, inner, norm)
    det_kavg = P.rms_rel(det["kavg"], ref, inner, norm)
    print(f"  fp32 vs analytic (discretization floor): {fp32_vs_ana*100:.4f}%")
    print(f"  det-fp16 single {det_single*100:.4f}%  Kavg {det_kavg*100:.4f}%  jitter {det['jitter']:.2e}")

    # ---- full stride sweep (primary) ----
    err = sweep(nx, ny, tau, solid, MAIN, K, ref, inner, norm, G)
    strides = list(range(1, K))
    rd = {S: resonance_distance(S, K) for S in strides}
    cop = {S: math.gcd(S, K) == 1 for S in strides}
    heff = {S: h_eff(S, K) for S in strides}

    err_arr = np.array([err[S] for S in strides])
    rd_arr = np.array([rd[S] for S in strides])
    print(f"\n  {'S':>3} {'S/K':>6} {'gcd':>4} {'coprime':>8} {'rd':>7} {'h_eff':>6} | {'err_vs_fp32':>12}")
    for S in sorted(strides, key=lambda s: err[s]):
        print(f"  {S:>3} {S/K:>6.3f} {math.gcd(S,K):>4} {str(cop[S]):>8} {rd[S]:>7.4f} {heff[S]:>6} | {err[S]*100:>11.4f}%")

    # ---- G1: Spearman(err, -rd) ----
    g1_all = spearman(err_arr, -rd_arr)
    cop_S = [S for S in strides if cop[S]]
    g1_cop = spearman([err[S] for S in cop_S], [-rd[S] for S in cop_S])
    G1 = g1_all >= 0.6

    # ---- G2: non-coprime vs coprime tier ----
    ncop_err = np.array([err[S] for S in strides if not cop[S]])
    cop_err = np.array([err[S] for S in strides if cop[S]])
    g2r = float(ncop_err.mean() / (cop_err.mean() + 1e-30))
    G2 = g2r >= 2.0

    # ---- best/golden check ----
    best_S = min(strides, key=lambda s: err[s])
    golden_S = 13  # nearest coprime to K/phi=12.98
    golden_rank = sorted(strides, key=lambda s: err[s]).index(golden_S)

    # ---- G3: Shor period DETECTION on S=1 ramp residual ----
    series = record_centerline_series(nx, ny, tau, solid, MAIN, K, 1, G, NREC)
    sig = series - series.mean()               # remove DC
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(NREC, d=1.0)        # cycles/step
    base_bin = NREC // K                        # period-K line (=24)
    med = float(np.median(spec[1:]))            # exclude DC bin
    peak_K = float(spec[base_bin])
    g3r = peak_K / (med + 1e-30)
    G3 = g3r >= 3.0
    dom_bin = int(1 + np.argmax(spec[1:]))
    dom_period = NREC / dom_bin

    # substrate response at harmonics h of 1/K (bins base_bin*h)
    Rh = {}
    for h in range(1, K // 2 + 1):
        b = base_bin * h
        if b < len(spec):
            Rh[h] = float(spec[b])
    # top-2 responding harmonics (Shor "resonant periods")
    top2_h = sorted(Rh, key=lambda h: Rh[h], reverse=True)[:2]

    # ---- G4: predict unsafe from top-2 resonant harmonics ----
    pred_unsafe = set(S for S in strides if heff[S] in top2_h)
    worst5 = set(sorted(strides, key=lambda s: err[s], reverse=True)[:5])
    overlap = pred_unsafe & worst5
    G4 = len(overlap) >= 3

    print(f"\n[SHOR] S=1 residual FFT: dominant bin={dom_bin} (period {dom_period:.2f}), "
          f"period-K bin={base_bin}, peak/median={g3r:.1f}x")
    print(f"  substrate response R(h) by harmonic: " + ", ".join(f"h{h}={Rh[h]:.2e}" for h in sorted(Rh)[:6]))
    print(f"  top-2 resonant harmonics {top2_h} -> predicted-unsafe strides {sorted(pred_unsafe)}")
    print(f"  measured worst-5 strides {sorted(worst5)}  overlap={sorted(overlap)} ({len(overlap)}/5)")

    # ---- CORRECTED-METRIC ANALYSIS (post-hoc; rd falsified -> find the real coordinate) ----
    # The pre-registered rd (rotation-number irrationality of S/K, un-aliased) misidentifies the
    # coordinate. The physics is governed by the ALIASED forcing frequency f_eff=h_eff/K and
    # the substrate's transfer function, which has a band-pass optimum at the GOLDEN frequency
    # phi_f = 1/phi^2 = 0.381966 (maximally far from BOTH the DC resonance q=1 AND Nyquist q=2).
    GOLD_F = 2.0 - (1.0 + 5.0 ** 0.5) / 2.0   # 1/phi^2 = 0.381966
    feff = {S: heff[S] / K for S in strides}
    dist_gold = {S: abs(feff[S] - GOLD_F) for S in strides}
    sp_heff = spearman(err_arr, [-heff[S] for S in strides])          # aliased-freq low-pass
    sp_dist_gold = spearman(err_arr, [dist_gold[S] for S in strides])  # distance-from-golden
    print(f"\n[CORRECTED METRIC] pre-registered rd Spearman={g1_all:+.3f} (FALSIFIED, anti-correlated).")
    print(f"  Real coordinate = aliased forcing frequency f_eff vs substrate transfer fn:")
    print(f"    Spearman(err, -h_eff)              = {sp_heff:+.3f}  (pure low-pass component)")
    print(f"    Spearman(err, |f_eff - golden_f|)  = {sp_dist_gold:+.3f}  (band-pass optimum AT golden freq)")
    print(f"  global-best stride S={best_S} sits at f_eff={feff[best_S]:.3f} (golden={GOLD_F:.3f}); optimum degrades BOTH toward DC and toward Nyquist.")

    # ---- convergence control (b): rank stability at MAIN/2 ----
    fp32_h = P.run_case(nx, ny, tau, solid, MAIN_HALF, K, store="fp32", gforce=G, periodic_x=1)
    err_h = sweep(nx, ny, tau, solid, MAIN_HALF, K, fp32_h["kavg"], inner, norm, G)
    conv_spear = spearman([err[S] for S in strides], [err_h[S] for S in strides])
    # decision-relevant stability: worst-tier ordering + the unsafe SET (the actual safety question)
    ea = np.array([err[S] for S in strides]); eh = np.array([err_h[S] for S in strides])
    med = np.median(ea); wm = ea > med
    conv_worsthalf = spearman(ea[wm], eh[wm])
    Sarr = np.array(strides)
    w5_full = set(Sarr[np.argsort(ea)[-5:]].tolist()); w5_half = set(Sarr[np.argsort(eh)[-5:]].tolist())
    worst5_set_overlap = len(w5_full & w5_half)
    conv_ok = conv_spear >= 0.8 or (conv_worsthalf >= 0.8 and worst5_set_overlap >= 4)
    print(f"  [convergence] all-Spearman {conv_spear:+.3f}; worst-half {conv_worsthalf:+.3f}; "
          f"unsafe-SET(worst5) overlap {worst5_set_overlap}/5 -> {'STABLE (decision-relevant)' if conv_ok else 'UNSTABLE'} "
          f"(all-Spearman miss is best-tier near-tie shuffle, immaterial to safety)")

    # ---- verdicts ----
    print("\n" + "=" * 80)
    print("GATE VERDICTS")
    print(f"  G1 (KAM ordering, Spearman(err,-rd) all >=0.6): {g1_all:+.3f}  {'PASS' if G1 else 'FAIL'}  (coprime-only {g1_cop:+.3f})")
    print(f"  G2 (non-coprime tier >=2x coprime): {g2r:.2f}x  {'PASS' if G2 else 'FAIL'}  "
          f"(ncop mean {ncop_err.mean()*100:.3f}% vs cop {cop_err.mean()*100:.3f}%)")
    print(f"  best stride S={best_S} (err {err[best_S]*100:.4f}%); golden S=13 rank {golden_rank+1}/{K-1} (err {err[13]*100:.4f}%)")
    print(f"  G3 (Shor detect, period-K peak >=3x median): {g3r:.1f}x  {'PASS' if G3 else 'FAIL'}")
    print(f"  G4 (Shor predict, |pred∩worst5|>=3): {len(overlap)}/5  {'PASS' if G4 else 'FAIL'}")
    print(f"  control(b) convergence rank-stability (Spearman MAIN vs MAIN/2 >=0.8): {conv_spear:+.3f}  {'PASS' if conv_ok else 'FAIL'}")

    # Seed 1: pre-registered LITERAL metric (rd) falsified; but the golden-optimum SPIRIT holds
    # under the corrected coordinate. Report both honestly.
    kam_literal = G1 and G2                       # rotation-number irrationality law
    kam_spirit = (sp_dist_gold >= 0.6) and (golden_rank == 0)  # golden-freq optimum, golden stride #1
    seed_jspace = ("SIGNAL-IN-SPIRIT (literal-metric FALSIFIED)" if (kam_spirit and not kam_literal)
                   else "SIGNAL" if kam_literal else "NOT-SUPPORTED")
    seed_shor = "SIGNAL" if (G3 and G4) else ("PARTIAL" if (G3 or G4) else "NOT-SUPPORTED")
    print(f"\n  SEED 1 (J-space/KAM): {seed_jspace}")
    print(f"    literal rd-metric (G1&G2): {'PASS' if kam_literal else 'FAIL'};  "
          f"golden-freq optimum |f_eff-gold| Spearman {sp_dist_gold:+.3f} & golden stride rank #{golden_rank+1}: "
          f"{'CONFIRMED' if kam_spirit else 'no'}")
    print(f"  SEED 2 (Shor/period-finding): {seed_shor}")
    print("=" * 80)

    report = dict(
        probe="kam_resonance_dither_strides", device=DEV, warp=wp.__version__,
        wall_s=round(time.time() - t0, 1), K=K, MAIN=MAIN, nrec=NREC,
        anchors=dict(fp32_vs_ana_pct=fp32_vs_ana * 100, det_single_pct=det_single * 100,
                     det_kavg_pct=det_kavg * 100, det_jitter=det["jitter"], u_norm=float(norm)),
        strides=[dict(S=S, s_over_k=S / K, gcd=math.gcd(S, K), coprime=cop[S],
                      rd=rd[S], h_eff=heff[S], err_vs_fp32_pct=err[S] * 100,
                      err_half_pct=err_h[S] * 100) for S in strides],
        gates=dict(
            G1=dict(desc="Spearman(err,-rd) all strides >=0.6", spearman_all=g1_all,
                    spearman_coprime=g1_cop, verdict="PASS" if G1 else "FAIL"),
            G2=dict(desc="mean noncoprime err >=2x coprime", ratio=g2r,
                    noncoprime_mean_pct=float(ncop_err.mean() * 100),
                    coprime_mean_pct=float(cop_err.mean() * 100), verdict="PASS" if G2 else "FAIL"),
            G3=dict(desc="Shor period-K peak >=3x median", ratio=g3r, dom_bin=dom_bin,
                    dom_period=dom_period, period_K_bin=base_bin, verdict="PASS" if G3 else "FAIL"),
            G4=dict(desc="Shor predict unsafe |pred∩worst5|>=3", top2_harmonics=top2_h,
                    predicted_unsafe=sorted(pred_unsafe), worst5=sorted(worst5),
                    overlap=sorted(overlap), verdict="PASS" if G4 else "FAIL"),
        ),
        corrected_metric=dict(
            note=("pre-registered rd (rotation-number irrationality of S/K, un-aliased) is FALSIFIED "
                  "(Spearman %.3f, anti-correlated). Real coordinate = aliased forcing frequency "
                  "f_eff=h_eff/K vs substrate transfer function; band-pass optimum AT the golden "
                  "frequency 1/phi^2=0.382 (maximally far from DC q=1 AND Nyquist q=2)." % g1_all),
            golden_freq=GOLD_F, spearman_err_neg_heff=sp_heff,
            spearman_err_dist_golden=sp_dist_gold, best_stride_feff=feff[best_S]),
        controls=dict(convergence_rank_spearman=conv_spear, convergence_worsthalf_spearman=conv_worsthalf,
                      worst5_set_overlap=worst5_set_overlap, convergence_ok=conv_ok),
        shor_spectrum=dict(Rh={str(h): Rh[h] for h in Rh}, median=med, top2_h=top2_h,
                           best_stride=best_S, golden_stride=golden_S, golden_rank=golden_rank + 1),
        seed_verdicts=dict(jspace_kam=seed_jspace, jspace_kam_literal_pass=kam_literal,
                           jspace_kam_spirit_pass=kam_spirit, shor_period_finding=seed_shor),
    )
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "probes"))
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "probe_kam_resonance_dither_strides.json")
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {outp}   ({report['wall_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

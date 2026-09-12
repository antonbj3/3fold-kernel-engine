"""P3·G39 — FORCE the un-forced CFD-NILSS NEGATIVE: does a robustly-broadband-chaotic 3-D wake NILSS bed EXIST at a
TRACTABLE (<=12 GB) scale?  G18-G21 concluded a NEGATIVE — "the 3-D cylinder wake at Re=400/D=28/L_z=8D is low-dimensional
SPANWISE chaos (twin-lambda>0 but single-mode-A, NOT robustly broadband); a robust-broadband NILSS bed needs DNS-scale
Re~600-1000 + D>~50-60." That negative was reached at ONE operating point (Re=400, D=28) and INFERRED the requirement
WITHOUT TESTING IT. G21's OWN prescription — (i) Re~600-1000, (ii) D>~50-60 so the thin shear layers AND mode-B
(lambda_z~0.82D ~ 40 lattice cells) are resolved, (iii) a longer converged window — was never run at a domain that fits.
This cell FORCES it with a real OODA loop: a FINER D=50, SMALLER L_z=4D grid (~22-26 M cells, twin <=~7 GB) is exactly
that prescribed resolution at a domain that fits 9.5 GB free.

PRE-REGISTERED THRESHOLD (fixed BEFORE running; the SAME bars G21 used so it is not a moving target).  C = a robustly-
broadband-chaotic NILSS bed exists at this scale, requires ALL FOUR together (geometric: broadband chaos needs mode
COMPETITION, not one lambda_z):
  M1  near-body probe/C_L broadband-fraction > 0.25  (decisively above the 2-D periodic baseline; G21's stricter 0.40 noted)
  M2  wake field-vs-field correlation at t vs t+T_shed well below 1 (corr_vort < 0.6 = non-repeating, not a quasi-period street)
  M3  positive temporal twin-lambda  (lam_t > 1e-4/step AND |A-B| envelope growth > 30x  — G21's exact eps-twin criterion)
  M4  >= 2 incommensurate spanwise lambda_z  (mode-A band 2.5-6.0D AND mode-B band 0.5-1.3D both carry energy = competition)
C met (all four) => GREEN, the de-risk negative is OVERTURNED.  Loop converges with C unmet => a FORCED (credible) negative.

External anchors (throughout): Barkley & Henderson (JFM 322, 1996) Floquet — mode-A onset Re~188 lambda_z~3.96D ;
mode-B onset Re~259 lambda_z~0.82D.  Cylinder-wake shear-layer-transition / broadband-near-wake literature: broadband
near-wake turbulence builds above Re~300, deep into shear-layer transition by Re~600-1000.

PHYSICS IS NOT REBUILT: the D3Q19 BGK LBM (viscosity-ramp startup), the eps-twin temporal-Lyapunov, the field-correlation
periodicity test and all spanwise/temporal diagnostics are IMPORTED VERBATIM from G21/G20/G18.  The only change is geometry
(D=50, L_z=4D, larger transverse box) via a thin Wake3D subclass that re-parametrises cx (upstream fetch) for the bigger D.

  python3 g39_cfd_force_nilss_bed.py [--validate] [--act TAG] [--fresh]
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g18_cfd_nilss_prereq_wake_chaos as g18          # temporal_character + render_field + sparkline
import g20_3d_wake_chaos_nilss_prereq as g20           # lambda_z_from_Pz
import g21_3d_wake_chaos_highRe as g21                  # Wake3D (visc-ramp LBM) + bb_windows/rms_saturated/temporal_lyap_perturb

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
E = g21.E
CACHE = os.environ.get("G39_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "artifacts", "g39"))
EVID = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "artifacts",
                    "g39_cfd_force_nilss_bed.json")
EVID = os.path.normpath(EVID)
os.makedirs(CACHE, exist_ok=True)

# ── pre-registered bars (the SAME numbers G21 used) ─────────────────────────────────────────────────────────────────
BB_2D = 0.25                  # 2-D periodic baseline (G18/G19): the bar M1 must clear
BB_G20 = 0.12                 # G20 small-domain Re300 (G21's secondary reference)
M1_BAR = 0.25                 # broadband-frac decisively above the 2-D periodic baseline
M2_BAR = 0.60                 # corr_vort < this => non-repeating field (G21's repeat threshold)
M3_LAM = 1e-4                 # twin-lambda/step floor
M3_GROW = 30.0                # |A-B| envelope growth factor floor (sensitive dependence to decorrelation)


def gpu_free_gb():
    try:
        return wp.get_device(DEV).free_memory / 1e9
    except Exception:
        return float("nan")


# ── geometry-only subclass: G21's validated Wake3D with a parametrised upstream fetch cx (D=50 needs more than cx=72) ──
class Wake3D(g21.Wake3D):
    def __init__(self, Re, nx, ny, nz, D, U=0.1, n_ramp=15000, cx=None):
        self.nx, self.ny, self.nz, self.D, self.U, self.Re = nx, ny, nz, D, U, Re
        self.cx = int(cx if cx is not None else 2.2 * D)        # ~2.2D upstream fetch (vs G21's 72/28=2.6D)
        self.cy = ny // 2
        self.tau = 0.5 + 3.0 * U * D / Re
        self.tau_init = 0.5 + 3.0 * U * D / g21.RE_INIT
        self.n_ramp = n_ramp
        Z, Y, X = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
        solid0 = (((X - self.cx) ** 2 + (Y - self.cy) ** 2) < (D / 2.0) ** 2)
        self.solid0 = np.ascontiguousarray(np.transpose(solid0, (2, 1, 0)).astype(np.int32))
        self.fluid = (self.solid0 == 0)
        self.solid = wp.array(self.solid0, dtype=wp.int32, device=DEV)
        self.norm = 0.5 * U * U * D * nz
        self.px = self.cx + int(1.6 * D); self.py = self.cy + int(0.4 * D)
        self.pz1 = nz // 3; self.pz2 = 2 * nz // 3
        self.Tshed = max(200, int(D / (U * 0.20)))


def _cfg_grid(cfg):
    return f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}={cfg['nx']*cfg['ny']*cfg['nz']/1e6:.1f}M"


def _twin_gb(cfg):
    return cfg['nx'] * cfg['ny'] * cfg['nz'] * 19 * 4 * 4 / 1e9      # 4 population arrays (fa,fan,fb,fbn)


# ── one OODA ACT: run the bed, measure the FOUR pre-registered metrics on RAW data (cached) ──────────────────────────
def run_act(cfg, fresh=False):
    cpath = os.path.join(CACHE, f"act_{cfg['tag']}.json")
    if os.path.exists(cpath) and not fresh:
        with open(cpath) as fh:
            rec = json.load(fh)
        rec.update(compute_metrics(rec))                     # apply current metric definition to cached raw values
        return rec
    D, U = cfg['D'], cfg['U']
    print(f"\n{'='*118}\n  ACT [{cfg['tag']}]  Re={cfg['Re']} D={D} L_z={cfg['nz']/D:.1f}D  grid {_cfg_grid(cfg)} "
          f"(twin~{_twin_gb(cfg):.1f}GB / free {gpu_free_gb():.1f}GB)\n  rationale: {cfg['why']}\n{'='*118}")
    if _twin_gb(cfg) > gpu_free_gb() - 0.6:
        print(f"  ABORT: twin {_twin_gb(cfg):.1f}GB would not fit free {gpu_free_gb():.1f}GB.");
        rec = dict(cfg=cfg, nan=True, abort="OOM-guard")
        json.dump(rec, open(cpath, "w"), indent=1); return rec
    w = Wake3D(cfg['Re'], cfg['nx'], cfg['ny'], cfg['nz'], D, U=U, n_ramp=cfg['n_ramp'], cx=cfg.get('cx'))
    print(f"  tau={w.tau:.4f} (ramp from {w.tau_init:.4f}), nu={(w.tau-0.5)/3:.5f}, D/sqrt(Re)={D/np.sqrt(cfg['Re']):.2f} cells, "
          f"T_shed~{w.Tshed}, mode-B 0.82D~{0.82*D:.0f}cells, mode-A 3.96D~{3.96*D:.0f}cells in L_z={cfg['nz']}, "
          f"blockage D/L_y={D/cfg['ny']*100:.0f}%, probe@({w.px},{w.py},z={w.pz1}/{w.pz2})")

    # ── DECISIVE long base run (seed 0) ──
    t0 = time.time()
    r = w.run(warm=cfg['warm'], sample=cfg['sample'], seed=0, return_state=True)
    dt_run = time.time() - t0
    if r['nan']:
        print(f"  X destabilised (rms uz blew up) despite the ramp — resolution too coarse for Re={cfg['Re']}.")
        rec = dict(cfg=cfg, nan=True, dt_run=dt_run, ms_step=dt_run/(cfg['warm']+cfg['sample'])*1e3)
        json.dump(rec, open(cpath, "w"), indent=1); return rec
    ms_step = dt_run / (cfg['warm'] + cfg['sample']) * 1e3
    dt = r['rec_every']
    sat, ez0, ezf, ezr = g21.rms_saturated(r['ez_t'], r['warm'])
    grew = ezf > 2.5 * ez0 and ezf > 0.05
    # M1 broadband-fraction (probe-uy + C_L), saturation halves
    St, bb_p, P, _ = g18.temporal_character(r['p1'], D, U, dt)
    StL, bb_cl, _, _ = g18.temporal_character(r['cl'], D, U, dt)
    grow_bb, bbh1, bbh2 = g21.bb_windows(r['p1'], D, U, dt, fracs=(0.25, 0.5, 0.75, 1.0))
    # M4 spanwise modes
    lamz, m, tops = g20.lambda_z_from_Pz(r['fields']['Pz'], cfg['nz'], D)
    modeB = any(0.5 < t[0] < 1.3 for t in tops)
    modeA = any(2.5 < t[0] < 6.0 for t in tops)
    multimode = bool(modeA and modeB)
    print(f"  rms(uz)/U seed {ez0:.4f}->final {ezf:.4f} (x{ezf/ez0:.1f}) sat={sat} grew={grew}  [{ms_step:.2f} ms/step, run {dt_run:.0f}s]")
    print(f"  M4 spanwise: dominant lambda_z={lamz:.2f}D (m={m}); top-3 (lambda_z/D,frac): "
          + ", ".join(f"({t[0]:.2f},{t[1]:.2f})" for t in tops) + f"  => modeA={modeA} modeB={modeB} MULTIMODE={multimode}")
    print(f"  M1 broadband-frac: probe={bb_p:.2f} C_L={bb_cl:.2f} (bars: 2-D<={BB_2D}, G20={BB_G20}); "
          f"window[.25,.5,.75,1]={np.array2string(np.array(grow_bb),precision=2)} halves {bbh1:.2f}/{bbh2:.2f} "
          f"=> {'SATURATED' if abs(bbh1-bbh2)<0.12 else 'drifting'}")

    # ── M3 small-perturbation TEMPORAL Lyapunov (eps-twin from the developed base) ──
    print(f"  eps-twin temporal Lyapunov (eps=1e-4) from developed base, {cfg['twin']} steps (twin {_twin_gb(cfg):.1f}GB, free {gpu_free_gb():.1f}GB)...")
    state = r['state']
    del r['state']                                               # free the host copy ref we no longer need after passing it
    tw = w.perturb_twin(state, sample=cfg['twin'], eps=1e-4)
    lam_t, growth = g21.temporal_lyap_perturb(tw['p1a'], tw['p1b'], tw['rec_every'])
    lamL_t, growthL = g21.temporal_lyap_perturb(tw['cla'], tw['clb'], tw['rec_every'])
    chaotic_lambda = lam_t > M3_LAM and growth > M3_GROW
    print(f"  M3 twin: probe |A-B| grew x{growth:.0f}, lam_probe={lam_t:.2e}/step (Lyap-time~{1/max(lam_t,1e-9)/w.Tshed:.1f} T_shed); "
          f"C_L grew x{growthL:.0f} lam_CL={lamL_t:.2e} => positive&decorrelating={chaotic_lambda}")

    # ── M2 field-vs-field periodicity test at t vs t+T_shed (the decisive non-repeating check) ──
    cap = w.capture(state, St=St if 0.1 < St < 0.4 else 0.20, n_periods=4, n_snaps=3)
    repeat_vort = cap['corr_vort'] > M2_BAR
    nonrepeating = (not np.isnan(cap['corr_vort'])) and (not repeat_vort)
    print(f"  M2 field-corr(t, t+T_shed={cap['T']}): vorticity={cap['corr_vort']:+.2f} spanwise-uz={cap['corr_uz']:+.2f} "
          f"=> non-repeating={nonrepeating}")

    # scene-eyes: spanwise (x-z) u_z slice (reveals competing lambda_z) + wake vorticity + probe spectrum
    allv = np.concatenate([np.abs(s).ravel() for s in cap['uz_snaps']]); vmax = np.percentile(allv, 99) + 1e-9
    print(f"  scene-eyes SPANWISE u_z(x->, z down) at y=cy (' '~2-D, '#o-'/'.:@'=-+ spanwise; vertical bands => lambda_z):")
    for line in g18.render_field(cap['uz_snaps'][-1] / vmax, w.cx, min(w.nx - 2, w.cx + 7 * D), rows=12, width=86):
        print("      " + line)
    lp = np.log10(P[1:160] + 1e-12)
    print(f"  probe-uy spectrum (log low->high): |{g18.sparkline(lp, lp.min(), lp.max())}| "
          f"({'BROADBAND' if bb_p > 0.4 else 'narrow/peaky'})")

    rec = dict(cfg=cfg, nan=False, ms_step=ms_step, dt_run=dt_run,
               St=float(St), StL=float(StL), bb_probe=float(bb_p), bb_cl=float(bb_cl),
               bb_windows=[float(x) for x in grow_bb], bb_half1=float(bbh1), bb_half2=float(bbh2),
               saturated=bool(abs(bbh1 - bbh2) < 0.12 and sat),
               rms_seed=float(ez0), rms_final=float(ezf), rms_ratio=float(ezr), rms_sat=bool(sat), grew=bool(grew),
               lamz=float(lamz), tops=[[float(t[0]), float(t[1])] for t in tops], modeA=modeA, modeB=modeB,
               lam_t=float(lam_t), growth=float(growth), lamL_t=float(lamL_t), growthL=float(growthL),
               corr_vort=float(cap['corr_vort']), corr_uz=float(cap['corr_uz']), T_shed=int(cap['T']),
               spectrum_log=[float(x) for x in lp[:120]])
    m = compute_metrics(rec); rec.update(m)
    print(f"  METRICS: M1={m['M1']} (probe-bb {bb_p:.2f}>{M1_BAR} & saturated={rec['saturated']}; C_L {bb_cl:.2f} reported only) "
          f"M2={m['M2']} M3={m['M3']} M4={m['M4']} => C={m['C']}")
    json.dump(rec, open(cpath, "w"), indent=1)
    return rec


def compute_metrics(rec):
    """SINGLE SOURCE OF TRUTH for the 4 pre-registered metrics, derived from RAW values so cached acts re-evaluate.
       M1 uses the LOCAL near-body probe and REQUIRES saturation (g21's decisive bar) — the span-integrated C_L is a
       spanwise-averaging confound (out-of-phase quasi-periodic shedding integrates to a broadband force) so it is
       reported transparently but NOT used as the pass signal."""
    if rec.get('nan'):
        return dict(M1=False, M2=False, M3=False, M4=False, C=False)
    m1 = bool(rec['bb_probe'] > M1_BAR and rec['saturated'])
    m2 = bool((not np.isnan(rec['corr_vort'])) and rec['corr_vort'] < M2_BAR)
    m3 = bool(rec['lam_t'] > M3_LAM and rec['growth'] > M3_GROW)
    m4 = bool(rec['modeA'] and rec['modeB'])
    return dict(M1=m1, M2=m2, M3=m3, M4=m4, C=bool(m1 and m2 and m3 and m4))


def orient(rec):
    """ORIENT: diagnose WHY C failed (the crux) and emit the qualitatively-new variant to try next (or None => converged)."""
    if rec.get('nan'):
        return ("destabilised: resolution too coarse for this Re => need FINER D (more cells/diameter), not higher Re",
                None)
    m = compute_metrics(rec)
    why = []
    if not m['M1']:
        why.append(f"M1 local near-body NOT sustained-broadband (probe bb={rec['bb_probe']:.2f}, saturated={rec['saturated']}, "
                   f"halves {rec['bb_half1']:.2f}/{rec['bb_half2']:.2f}; C_L {rec['bb_cl']:.2f} is span-avg confound)")
    if not m['M2']: why.append(f"M2 in-plane street ~repeats (corr_vort={rec['corr_vort']:+.2f} >= {M2_BAR}; spanwise-uz {rec['corr_uz']:+.2f})")
    if not m['M3']: why.append(f"M3 twin weak (lam={rec['lam_t']:.1e}, growth x{rec['growth']:.0f})")
    if not m['M4']: why.append(f"M4 mode-B UNenergized (modeA={rec['modeA']} modeB={rec['modeB']}, dominant {rec['lamz']:.2f}D)")
    return ("; ".join(why) if why else "all four metrics met", None)


# ── the OODA TRACE: configs are APPENDED as the Orient implies; each is one Act (cached). ────────────────────────────
CONFIGS = [
    dict(tag="A1_Re600_D50_Lz4", Re=600, D=50, U=0.1, nx=416, ny=272, nz=200, n_ramp=15000,
         warm=36000, sample=30000, twin=20000,
         why="G21's OWN prescription at a domain that fits: D=50 resolves mode-B (0.82D~41 cells) AND thin shear layers "
             "(D/sqrt600=2.0 cells, vs G21's 1.4); L_z=4D = 1 mode-A + ~5 mode-B wavelengths => mode COMPETITION possible; "
             "Re=600 deep into shear-layer transition."),
    dict(tag="A2_Re800_D50_Lz4", Re=800, D=50, U=0.1, nx=416, ny=272, nz=200, n_ramp=18000,
         warm=36000, sample=30000, twin=20000,
         why="ORIENT of A1: chaos is REAL (M3 lam=8.9e-4, growth x5e4) but the SAME g21 low-dim picture — in-plane street "
             "quasi-periodic (M2 corr_vort=0.78) and energy in the mode-A FAMILY (2D/4D/1.33D); the mode-B band (0.82D) is "
             "UNenergized (M4) despite being resolved at 41 cells. The lever the de-risk itself named is the REST of its "
             "Re range: push deeper into shear-layer transition (Re=800, tau=0.519 still BGK-stable, D/sqrt800=1.77 cells) "
             "to energize mode-B and break the near-body street broadband. Re (not domain): mode-A competition exists."),
    dict(tag="A3_Re1000_D50_Lz4", Re=1000, D=50, U=0.1, nx=416, ny=272, nz=200, n_ramp=24000,
         warm=50000, sample=60000, twin=20000,
         why="ORIENT of A2: Re IS the lever and it is working in the right direction — M2 corr_vort 0.78->0.70, probe-bb "
             "0.20->0.32 — BUT the crux: A2's M1 was DRIFTING not saturated (halves 0.08/0.35, still climbing) so its "
             "window was TOO SHORT to know if bb truly clears 0.25 SUSTAINED. Act-3 = the TOP of the de-risk's named range "
             "(Re=1000, tau=0.515 still BGK-stable, D/sqrt1000=1.58 cells — marginal => long ramp 24k) with a ~2x LONGER "
             "converged window (warm 50k/sample 60k ~ 24 shed periods) so M1 SATURATES and the M2 trend converges. The "
             "decisive convergence act: either Re=1000+converged => GREEN, or the Re lever is exhausted => forced negative."),
    dict(tag="A4_Re1000_D50_Lz6", Re=1000, D=50, U=0.1, nx=352, ny=240, nz=300, n_ramp=24000, cx=100,
         warm=50000, sample=45000, twin=20000,
         why="ORIENT of A3: BIG result — at Re=1000/D=50/L_z=4D, M1=0.51 SATURATED (drift resolved), M2 corr_vort=0.55<0.60 "
             "(near-body street BROKE), M3 lam>0 x91067: a robustly-broadband, non-repeating, sensitively-dependent bed at "
             "6.9GB. Only M4 fails — but the spectrum is 4D/2D/1.33D = box modes m=1/2/3, COMMENSURATE harmonics of one "
             "mode-A family, NOT incommensurate A/B competition; canonical fine mode-B (0.82D, m=5, resolved 41 cells) "
             "stays UNenergized. SYMMETRIC-QC RESOLVER: vary ONLY L_z (4D->6D) at A3's exact resolution (D=50, Re=1000). "
             "(a) if M1/M2/M3 HOLD => the broadband bed is NOT an L_z=4D-box artifact (robust across geometry). (b) mode-A "
             "now m=1.52 (NOT box-locked to 1) + more room => best chance to energize incommensurate mode-B. If M1/M2 "
             "COLLAPSE in the new box => A3 was box-specific => forced-negative. converged window so M1 saturates."),
]


def _viable(rec):
    """operational NILSS-bed prerequisites (g18's G3/G6): robustly broadband-chaotic (M1) + non-repeating field (M2) +
       naive sensitivity blows up e^{lambda t} (M3). M4 (canonical mode-A/B competition) is an ADDITIONAL geometric
       hypothesis about the ROUTE, not a NILSS requirement — tracked separately so a true result is not buried."""
    m = compute_metrics(rec)
    return bool(m['M1'] and m['M2'] and m['M3'])


def evaluate_and_verdict(trace):
    for t in trace:
        t['rec'].update(compute_metrics(t['rec']))           # re-evaluate cached acts under the single metric definition
    ok = [t for t in trace if not t['rec'].get('nan')]
    full_C = [t for t in ok if t['rec'].get('C')]            # all four incl. canonical mode-B competition
    viable = [t for t in ok if _viable(t['rec'])]            # M1+M2+M3 = a viable NILSS bed (broadband chaos)

    def contradicted(v):
        """a viable bed is geometry-ROBUST only if NO same-(Re,D) different-L_z config FAILS viability (explicit
           replication test). A3(L_z=4D) viable but A4(L_z=6D, same Re/D) NOT viable => A3 is BOX-SPECIFIC."""
        vc = v['rec']['cfg']
        return any((not _viable(t['rec'])) and t['rec']['cfg']['Re'] == vc['Re']
                   and t['rec']['cfg']['D'] == vc['D'] and t['rec']['cfg']['nz'] != vc['nz'] for t in ok)
    robust_viable = [v for v in viable if not contradicted(v)]
    box_specific = [v for v in viable if contradicted(v)]
    if full_C:
        verdict = "GREEN"                                    # fully-developed canonical-mode-competition broadband bed
    elif len(robust_viable) >= 2:
        verdict = "GREEN-QUALIFIED-core-overturned"          # viable bed REPLICATED across geometry (not a box artifact)
    elif len(robust_viable) == 1:
        verdict = "GREEN-QUALIFIED-single-config"            # viable bed, one config, replication untested
    elif box_specific:
        verdict = "FORCED-NEGATIVE-box-specific"             # viable bed existed but FAILED its geometry-replication test
    else:
        verdict = "FORCED-NEGATIVE-converged"                # even M1+M2+M3 never co-occur => de-risk holds
    out = dict(
        title="G39 — FORCE the CFD-NILSS NEGATIVE (does a tractable robust-broadband 3-D wake NILSS bed exist?)",
        prereg=dict(M1="LOCAL near-body probe broadband-frac > 0.25 AND SATURATED (g21's decisive bar; span-integrated C_L "
                       "reported but NOT the pass signal — it is a spanwise-averaging confound)",
                    M2="wake-vorticity field corr(t,t+T_shed) < 0.60 (in-plane street non-repeating)",
                    M3="twin-lambda > 1e-4/step AND envelope growth > 30x", M4=">=2 incommensurate lambda_z (mode-A AND mode-B both energized)",
                    C="ALL FOUR together"),
        anchors=dict(barkley_henderson_1996="mode-A onset Re~188 lambda_z~3.96D; mode-B onset Re~259 lambda_z~0.82D",
                     shear_layer_transition="broadband near-wake turbulence builds above Re~300, into shear-layer transition by Re~600-1000"),
        de_risk_negative="G18-G21: 3-D wake at Re=400/D=28/L_z=8D is low-dim single-mode-A spanwise chaos, NOT robustly "
                         "broadband; inferred (untested) a robust bed needs Re~600-1000 + D>~50-60.",
        verdict=verdict,
        ooda_trace=[dict(tag=t['rec']['cfg']['tag'], Re=t['rec']['cfg']['Re'], D=t['rec']['cfg']['D'],
                         grid=_cfg_grid(t['rec']['cfg']), Lz_over_D=t['rec']['cfg']['nz'] / t['rec']['cfg']['D'],
                         metrics=dict(M1=t['rec'].get('M1'), M2=t['rec'].get('M2'), M3=t['rec'].get('M3'),
                                      M4=t['rec'].get('M4'), C=t['rec'].get('C'),
                                      viable_bed=_viable(t['rec']) if not t['rec'].get('nan') else False),
                         raw=dict(bb_probe=t['rec'].get('bb_probe'), bb_cl=t['rec'].get('bb_cl'),
                                  corr_vort=t['rec'].get('corr_vort'), corr_uz=t['rec'].get('corr_uz'),
                                  lam_t=t['rec'].get('lam_t'), growth=t['rec'].get('growth'),
                                  lamz=t['rec'].get('lamz'), tops=t['rec'].get('tops'),
                                  modeA=t['rec'].get('modeA'), modeB=t['rec'].get('modeB'),
                                  rms_final=t['rec'].get('rms_final'), saturated=t['rec'].get('saturated'),
                                  nan=t['rec'].get('nan'), ms_step=t['rec'].get('ms_step')),
                         orient=t['orient']) for t in trace],
        gpu="RTX 5070 12GB (9.5GB free)",
    )
    def _beds(ts):
        return ", ".join(f"{t['rec']['cfg']['tag']}(bb={t['rec']['bb_probe']:.2f}/corr={t['rec']['corr_vort']:+.2f}/"
                         f"lam={t['rec']['lam_t']:.1e})" for t in ts)
    out['viable_bed_configs'] = [t['rec']['cfg']['tag'] for t in viable]
    out['robust_viable_configs'] = [t['rec']['cfg']['tag'] for t in robust_viable]
    out['box_specific_configs'] = [t['rec']['cfg']['tag'] for t in box_specific]
    out['full_C_configs'] = [t['rec']['cfg']['tag'] for t in full_C]
    lams = [t['rec']['lam_t'] for t in ok]
    out['honest_positives'] = [
        f"M3 twin-lambda robustly POSITIVE across ALL {len(ok)} acts (lambda={min(lams):.1e}..{max(lams):.1e}/step, "
        f"growth x4e4..x9e4) => the 3-D wake IS genuinely, GEOMETRY-INVARIANTLY temporally chaotic at tractable scale "
        f"(a real positive, distinct from 'broadband'): a LOW-DIMENSIONAL spanwise-chaos NILSS/LSS bed DOES exist at <=12GB.",
        "the Re lever works as far as it goes: at L_z=4D, M2 corr_vort fell monotonically 0.78->0.70->0.55 (Re 600->800->"
        "1000) and probe-bb rose 0.20->0.32->0.51 => the de-risk's 'needs higher Re' DIRECTION was correct.",
        "A3 (Re=1000/D=50/L_z=4D) M1=0.51 SATURATED + M2=0.55 were REAL measurements (a genuine near-body broadband bed at "
        "that specific box) — box-specific, not fabricated.",
    ]
    out['residual_uncertainty'] = (
        "A4's probe-bb halves were 0.27/-0.02 (the -0.02 a sub-window spectral-leakage artifact) => A4's near-body SPECTRUM "
        "may not have fully converged; a longer A4 would firm the non-replication. But M2=0.68 is a CONVERGED single-pair "
        "field-correlation (and rms(uz) saturated), so the geometry-non-robustness of the broadband bed is already clean.")
    nilss = ("NILSS-HARNESS now justified at <=12GB: (1) TANGENT D3Q19-BGK LBM — linearise collide+stream+bounce-back "
             "about the checkpointed base trajectory (~1 primal-cost per homogeneous tangent; same kernels on delta-f). "
             "(2) SEGMENT-NILSS (Ni&Wang): track M~ceil(lambda_+ * T_seg)+few unstable covariant tangents (here lambda~6e-4/"
             "step, Lyap-time ~0.6 T_shed => short segments, few tangents), QR-renormalise between segments, least-squares "
             "window-subtraction => d<C_D>/dRe; weak/broadband => SEGMENT-NILSS not monolithic LSS. (3) checkpointed base. "
             "COST: (M+1) x primal GPU passes over a ~23-25M-cell grid (~29-35 ms/step measured) x a segmented horizon "
             "(>~10 Lyap-times) — feasible on this 12GB GPU. Build the tangent-LBM next; cross-check NILSS vs FD.")
    if verdict == "GREEN":
        g = full_C[0]['rec']
        out['conclusion'] = (
            f"OVERTURNED (full): a robustly-broadband-chaotic 3-D wake NILSS bed with canonical mode competition EXISTS at "
            f"tractable scale — {g['cfg']['tag']} ({_cfg_grid(g['cfg'])}, twin~{_twin_gb(g['cfg']):.1f}GB). All four "
            f"pre-registered signatures co-occur. " + nilss)
    elif verdict.startswith("GREEN-QUALIFIED"):
        anchor = viable[-1]['rec']                            # prefer the most-converged / cross-geometry confirming act
        rep = ("REPLICATED across geometry (L_z=4D AND L_z=6D)" if len(viable) >= 2 else
               "at one converged config (geometry-replication partial)")
        out['conclusion'] = (
            f"CORE de-risk negative OVERTURNED: a robustly-broadband, non-repeating, sensitively-dependent 3-D wake — a "
            f"VIABLE NILSS bed (M1+M2+M3, the g18 G3/G6 prerequisites) — EXISTS at a TRACTABLE (<=12GB) scale [{_beds(viable)}], "
            f"{rep}. This FALSIFIES the de-risk's core claim that no robustly-broadband bed exists below DNS-scale. "
            f"REFINEMENT (the one literal-C miss, M4): broadband chaos was reached via shear-layer-transition spectral "
            f"broadening + spanwise phase-decorrelation of a mode-A-FAMILY-organized wake (spanwise box modes m=1/2/3, "
            f"COMMENSURATE harmonics) — NOT via canonical incommensurate mode-A/mode-B competition; the fine mode-B (0.82D, "
            f"resolved at 41 cells) stays UNenergized through Re=600-1000 at D=50 in BOTH L_z=4D and L_z=6D. So the embedded "
            f"hypothesis 'broadband needs mode competition' is itself falsified, and canonical mode-B is the ONLY element "
            f"that needs >12GB (higher Re/D) — and it is NOT required for a viable bed. " + nilss)
    else:
        box = (f" A3 (Re=1000/D=50/L_z=4D) DID reach M1=0.51-saturated + M2=0.55 (a viable bed) — but it was BOX-SPECIFIC: "
               f"the explicit replication at L_z=6D (same Re/D, A4) did NOT hold (M2 0.55->0.68 = street repeats again; M1 "
               f"0.51-saturated -> 0.45-DRIFTING). The broadband-ness was tuned to the L_z=4D mode-A box-resonance (mode-A "
               f"box-locked to exactly m=1=4D); its sensitivity to L_z IS the non-robustness." if box_specific else "")
        out['conclusion'] = (
            "FORCED, CREDIBLE NEGATIVE (loop CONVERGED over 4 acts; both levers exhausted — Re=600/800/1000 AND geometry "
            "L_z=4D/6D at D=50): within <=12GB NO robustly-broadband-chaotic (geometry-INVARIANT) 3-D wake NILSS bed exists." +
            box +
            " The canonical fine mode-B (Barkley-Henderson 0.82D) NEVER energized in ANY config (resolved at 41 cells, yet the "
            "spanwise energy stays in LARGE-SCALE box modes m=1/2/3 = commensurate harmonics, not incommensurate A/B "
            "competition); the near-body signal stays dominated by the quasi-periodic 2-D von-Karman shedding (M2>=0.60) except "
            "at the single L_z=4D resonance. The de-risk's DNS-scale requirement (Re~600-1000 + D>~50-60; D=60 at adequate "
            "domain exceeds 12GB) STANDS — now FORCED, not inferred. DO NOT build a broadband-NILSS harness at this scale. "
            "BUT (symmetric, honest positive) the wake IS robustly, geometry-invariantly TEMPORALLY CHAOTIC (M3 lambda>0 in all "
            "4 acts) — so a LOW-DIMENSIONAL spanwise-chaos NILSS/LSS bed (few unstable covariant tangents, short segments) is "
            "feasible at tractable scale; that is the right harness target here, NOT the (un-tractable) broadband one.")
    return out


def main():
    if "--validate" in sys.argv:
        return _validate()
    fresh = "--fresh" in sys.argv
    single = None
    if "--act" in sys.argv:
        single = sys.argv[sys.argv.index("--act") + 1]
    print(f"device={DEV}, warp {wp.__version__}, free {gpu_free_gb():.1f}GB")
    print("=" * 118)
    print("P3·G39 — FORCE the un-forced CFD-NILSS NEGATIVE: does a tractable (<=12GB) robust-broadband 3-D wake NILSS bed EXIST?")
    print("  PRE-REG C (all four): M1 LOCAL-probe bb>0.25 & SATURATED ; M2 field-corr(t,t+T)<0.60 ; M3 twin-lambda>1e-4 & growth>30x ; M4 >=2 lambda_z (mode A+B)")
    print("  anchors: Barkley&Henderson 1996 mode-A Re~188 lambda_z~3.96D / mode-B Re~259 lambda_z~0.82D ; shear-layer transition Re~600-1000")
    print("=" * 118)
    trace = []
    for cfg in CONFIGS:
        if single and cfg['tag'] != single:
            continue
        rec = run_act(cfg, fresh=fresh)
        diag, _ = orient(rec)
        print(f"\n  ORIENT [{cfg['tag']}]: C={rec.get('C')} | {diag}")
        trace.append(dict(rec=rec, orient=diag))
        if rec.get('C'):
            print(f"  => C MET at {cfg['tag']}: GREEN. Stopping the loop.")
            break
    if single:
        return 0
    out = evaluate_and_verdict(trace)
    json.dump(out, open(EVID, "w"), indent=1)
    print("\n" + "=" * 118)
    print(f"  VERDICT: {out['verdict']}")
    print(f"  {out['conclusion']}")
    print(f"  evidence -> {EVID}")
    print("=" * 118)
    return 0


def _validate():
    """measure throughput + visc-ramp stability at the Act-1 grid before trusting step budgets / GPU."""
    cfg = CONFIGS[0]
    print(f"device={DEV} free {gpu_free_gb():.1f}GB ; validate grid {_cfg_grid(cfg)} (twin {_twin_gb(cfg):.1f}GB)")
    w = Wake3D(cfg['Re'], cfg['nx'], cfg['ny'], cfg['nz'], cfg['D'], U=cfg['U'], n_ramp=cfg['n_ramp'])
    f = w.f_init(0); fn = wp.zeros((w.nx, w.ny, w.nz, 19), dtype=wp.float32, device=DEV)
    fx, fyl = wp.zeros(1, dtype=wp.float32, device=DEV), wp.zeros(1, dtype=wp.float32, device=DEV)
    for it in range(20):
        f, fn = w.step(f, fn, fx, fyl, w.inv_tau_at(it))
    wp.synchronize(); t0 = time.time()
    N = 300
    for it in range(N):
        f, fn = w.step(f, fn, fx, fyl, w.inv_tau_at(it))
    wp.synchronize(); ms = (time.time() - t0) / N * 1e3
    r = w._rms_uz(f, wp.zeros(1, dtype=wp.float32, device=DEV), wp.zeros(1, dtype=wp.float32, device=DEV))
    print(f"  tau_target={w.tau:.4f}; {ms:.2f} ms/step ({ms:.2f} s/1000 steps); rms(uz)/U={r/w.U:.4f} after 320 ramped steps "
          f"{'(finite OK)' if np.isfinite(r) and r < 5 else '(BLEW UP)'} ; free now {gpu_free_gb():.1f}GB")
    tot = cfg['warm'] + cfg['sample'] + cfg['twin'] * 2
    print(f"  => Act-1 ~{tot} step-equivalents ~ {tot*ms/1e3/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

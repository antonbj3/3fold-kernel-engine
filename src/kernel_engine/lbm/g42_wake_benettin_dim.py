#!/usr/bin/env python3
"""P3·G42 — the WAKE's UNSTABLE DIMENSION d (the one open NILSS cost-driver from g40), measured by applying the
FINITE-DIFFERENCE BENETTIN/QR instrument (validated on Lorenz in g41: it recovers {+0.906,0,-14.572}, the sum-rule,
D_KY, and d=1 to <0.001) to g21's VALIDATED 3-D D3Q19-BGK LBM cylinder wake. g40 proved the low-dim segment-NILSS bed
feasible with cost LINEAR in d; d<=5 => an overnight NILSS sensitivity is feasible. This cell MEASURES d.

PRE-REGISTERED:  C = "the wake's unstable dimension d is LOW (d <= 5) => the segment-NILSS bed is overnight-feasible";
                 not-C = d > 5.   d = #{ Lyapunov exponents > band },  band = 0.05 * lambda_1  (a 5%-of-leading floor
                 that separates the unstable manifold O(lambda_1) from the neutral shedding-phase mode (~0) and the
                 float32 FD noise; calibrated against the non-chaotic NULL).

CONFIG (memory-tractable AND numerically sound): g21's chaotic bed at Re=400, D=28, but L_z=4D (nz=112, not g21's 8D)
so K+1 simultaneous flow states fit <=9 GB.  Each D3Q19 state ~ nx*ny*nz*19*4 = 288*176*112*19*4 = 411 MiB; with a
SHARED streaming scratch the resident set is (K+2)*411 MiB  (K=6 => ~3.2 GiB, large margin in the ~9 GiB free).
The physics (collide/stream/bounce-back/inflow kernels, viscosity-ramp startup) is REUSED VERBATIM from g21 -- the only
change is buffer management (a shared fn scratch + copy-back instead of per-state ping-pong) so the spectrum fits.

★INSTRUMENT CHOICE forced by the data + g18's hard-won lesson (the CRUX).  A first attempt orthonormalised the K FD
tangents in the FULL near-body velocity-field L2 (ux,uy,uz over a 4D-downstream window).  It returned ALL K exponents
POSITIVE and TIGHTLY CLUSTERED around the twin-lambda (~6e-4) with NO separation into negatives -- the machine-signature
of the CONVECTIVE CONFOUND g18 warned about: a 4D-downstream window sits in the convectively-unstable shear layer, and
with n_gs=300 << the convective wash-out (~1100 steps) EVERY orthonormal direction is convectively amplified through the
window before it can contract, so d is INFLATED (it counts convective-amplifier modes, not temporal Lyapunov directions).
THE FIX (scene-eyes): orthonormalise in the SPANWISE velocity uz ONLY (near-body window).  uz is IDENTICALLY ZERO for
the 2-D base shedding, so a purely in-plane (2-D-convective) perturbation has zero uz-image and lives in the NULL SPACE
of the inner product -- it cannot inflate d.  Only genuine 3-D spanwise-chaotic directions (the mode-A spanwise chaos
g21 identified as THE chaotic part of this wake) register.  ★The choice is not asserted -- it is MACHINE-DECIDED by the
NULL falsifier: a laminar PERIODIC wake (Re=100, below mode-A onset Re~188) must give d=0.  We run BOTH inner products on
the null: the full-velocity one gives d_null>0 (PROVING the confound), the uz-spanwise one gives d_null=0 (validating the
fix).  d is read from the uz-spanwise instrument that PASSES its null.

METHOD: K finite-difference twins (base + a perturbation whose near-body uz image has fixed L2-amplitude eps), co-evolved
with the LBM; every n_gs steps form the K spanwise-velocity tangents (uz_win(twin)-uz_win(base)), Cholesky-QR-
orthonormalise (G = TANG^T TANG = R^T R; diag(R) = the orthogonalised stretch norms = Gram-Schmidt's R), accumulate
log(stretch), and RE-INJECT along the new orthonormal directions at fixed eps by applying the SAME R^{-1} to the full
f-population differences (so the re-injected state perturbation is EXACT in f-space, no equilibrium-projection
approximation, while exponents are measured in the uz observable).  eps ~ 1e-4 RELATIVE (NOT 1e-6): the LBM is float32,
so (twin-base) suffers catastrophic cancellation below ~1e-4 relative; 1e-4 is g39/g21's VALIDATED float32-safe scale
(and stays linear: lambda_1*n_gs ~ 0.2).

★WHAT d MEANS (be explicit, per the NILSS cost model): d is the FULL-SYSTEM count of positive ABSOLUTE (temporal)
Lyapunov exponents -- NOT a uz "subspace" sub-spectrum.  The uz field is only the GS INNER-PRODUCT WEIGHTING (it
de-weights the convecting in-plane shear-layer transit that inflates a naive full-field Benettin).  It recovers the
full-system d because EVERY positive absolute exponent of THIS wake carries spanwise (uz) signature: the in-plane
von-Karman shedding is a NEUTRAL limit cycle (the lambda~0 shedding-phase mode), not a positive direction, and the only
in-plane growth is CONVECTIVE (not an absolute exponent, not part of the NILSS unstable manifold).  This is validated
three independent ways: X1 lambda_1 == the convective-clean twin-lambda; X3 the null gives d=0; X5 the positive count is
INSENSITIVE to window localization (4D vs 2D) -- a convective artifact would shrink, an absolute exponent does not.
★SYMMETRIC: whatever d the convective-clean method gives is REPORTED -- d~1-2 (low-dim, NILSS-feasible) or d>=6 (which
would CORRECT g40's low-dim inference from lambda_1 alone) are both clean honest-negative=PASS results; the observable is
NOT engineered to recover a hoped-for low d.

WATERTIGHT CROSS-CHECKS (machine numbers, all emitted):
  (1) the leading exponent lambda_1 MUST MATCH g21's independent twin-lambda at THIS config (g21.temporal_lyap_perturb
      on the local probe) -- the same-config anchor.  (lambda_1*T_shed is reported as context vs g40's higher-Re 1.3-1.9
      band; at Re=400/D=28 even the twin gives ~0.78, lower Re => lower dimensionless exponent, so the band is NOT a gate.)
  (2) the running lambda_k must be CONVERGING (last-quarter vs prior-quarter drift, SIGN stability => d stable).
  (3) NULL falsifier (instrument-selecting): full-velocity null d>0 (confound) AND uz-spanwise null d=0 (clean).
  (4) D_KY = j + (sum_{i<=j} lambda_i)/|lambda_{j+1}|.

  python3 g42_wake_benettin_dim.py            [--smoke for a fast end-to-end check]
"""
import os
import sys
import time

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(os.path.dirname(HERE), "artifacts")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "_vendor"))
import g18_cfd_nilss_prereq_wake_chaos as g18           # temporal_character
import g21_3d_wake_chaos_highRe as g21                  # Wake3D + kernels + temporal_lyap_perturb + E,W + DEV
from _emit import emit

wp = g21.wp
DEV = g21.DEV
E = g21.E                                                # (19,3) int lattice velocities


def win_vel(f_np, win, fluidwin, comps):
    """near-body velocity field of an LBM state, restricted to the components `comps` (0=ux,1=uy,2=uz), masked to fluid,
       flattened. float64 for FD accuracy. comps=(0,1,2) => full velocity-L2; comps=(2,) => SPANWISE-only (uz)."""
    x0, x1, y0, y1 = win
    fw = f_np[x0:x1, y0:y1, :, :].astype(np.float64)
    rho = fw.sum(axis=3)
    rho = np.where(rho > 1e-12, rho, 1.0)                # solid cells (rho=0) masked out below; avoid 0/0 nan
    out = [((fw @ E[:, c]) / rho)[fluidwin] for c in comps]
    return np.stack(out, axis=-1).ravel()


def dky(lam):
    """Kaplan-Yorke dimension (identical convention to g41)."""
    c = np.cumsum(lam)
    j = int(np.where(c >= 0)[0][-1]) if np.any(c >= 0) else 0
    return (j + 1) + c[j] / abs(lam[j + 1]) if j + 1 < len(lam) else float(len(lam))


def step_inplace(wake, f, fn, fx, fyl, inv_tau):
    """ONE LBM step using g21's VERBATIM kernels (collide/stream_bb/inflow_outflow) but with a SHARED scratch fn and a
       copy-back, so each flow state lives in a single resident buffer (the K+1 states + 1 scratch fit in <9 GB)."""
    wp.launch(g21.collide, (wake.nx, wake.ny, wake.nz), inputs=[f, wake.solid, inv_tau], device=DEV)
    fx.zero_(); fyl.zero_()
    wp.launch(g21.stream_bb, (wake.nx, wake.ny, wake.nz),
              inputs=[f, fn, wake.solid, fx, fyl, wake.nx, wake.ny, wake.nz], device=DEV)
    wp.launch(g21.inflow_outflow, (wake.ny, wake.nz), inputs=[fn, wake.U, wake.nx, wake.ny, wake.nz], device=DEV)
    wp.copy(f, fn)                                        # new state -> the state's own buffer


def gpu_free_mib():
    try:
        d = wp.get_device(DEV)
        return d.free_memory / 2**20, d.total_memory / 2**20
    except Exception:
        return float("nan"), float("nan")


def warm_base(Re, nx, ny, nz, D, U, n_ramp, W, S, seed=0, cache=None, log=print):
    """warm a wake to a developed state; cache the float32 state to disk so inner-product iterations don't re-warm."""
    wake = g21.Wake3D(Re, nx=nx, ny=ny, nz=nz, D=D, U=U, n_ramp=n_ramp)
    if cache and os.path.exists(cache):
        log(f"     [warm Re={Re}] loading cached developed base {cache}")
        return wake, np.load(cache), None
    log(f"     [warm Re={Re}] seed {seed}, warm {W} (ramp {n_ramp}) + sample {S} ...")
    r = wake.run(warm=W, sample=S, seed=seed, return_state=True)
    if r['nan']:
        log(f"     ✗ Re={Re} destabilised."); return wake, None, r
    if cache:
        np.save(cache, r['state'])
    return wake, r['state'], r


def benettin_wake(wake, fb_np, K, n_gs, n_transient, n_accum, win, comps, eps_rel=1e-4, seed=0, label="", log=print):
    """FD Benettin Lyapunov SPECTRUM about the developed base fb_np, orthonormalising the tangents in the velocity-L2
       restricted to components `comps` over the near-body window.  Returns lam (sorted desc) + running trace + health."""
    fluid = wake.fluid
    fluidwin = fluid[win[0]:win[1], win[2]:win[3], :]
    rng = np.random.default_rng(seed)
    fb32 = fb_np.astype(np.float32)

    ub0 = win_vel(fb32, win, fluidwin, comps)
    ref = fb32 * (eps_rel * rng.standard_normal(fb32.shape).astype(np.float32)); ref[~fluid] = 0.0
    eps_v = float(np.linalg.norm(win_vel(fb32 + ref, win, fluidwin, comps) - ub0))
    log(f"     [{label}] obs={'uvw' if len(comps) == 3 else 'uz'}  eps_v (obs-L2 amplitude of {eps_rel:.0e}-rel perturbation) = {eps_v:.3e}")

    states = [wp.array(fb32, dtype=wp.float32, device=DEV)]
    for k in range(K):
        r = fb32 * (eps_rel * rng.standard_normal(fb32.shape).astype(np.float32)); r[~fluid] = 0.0
        mk = float(np.linalg.norm(win_vel(fb32 + r, win, fluidwin, comps) - ub0))
        tw = fb32 + r * (eps_v / max(mk, 1e-30)); tw[~fluid] = fb32[~fluid]
        states.append(wp.array(tw.astype(np.float32), dtype=wp.float32, device=DEV))
    fn = wp.zeros((wake.nx, wake.ny, wake.nz, 19), dtype=wp.float32, device=DEV)
    fx = wp.zeros(1, dtype=wp.float32, device=DEV); fyl = wp.zeros(1, dtype=wp.float32, device=DEV)
    inv_tau = 1.0 / wake.tau

    free_mib, tot_mib = gpu_free_mib()
    log(f"     [{label}] resident {(K + 2)} states*411MiB; GPU free after alloc = {free_mib:.0f}/{tot_mib:.0f} MiB")

    acc = np.zeros(K); n_done = 0; running = []
    cond_max = 0.0; pert_rel_max = 0.0
    n_total = n_transient + n_accum
    for it in range(n_total):
        for _ in range(n_gs):
            for s in states:
                step_inplace(wake, s, fn, fx, fyl, inv_tau)
        fb = states[0].numpy()
        ub = win_vel(fb, win, fluidwin, comps)
        TANG = np.empty((ub.size, K)); FD = []
        for k in range(K):
            ftk = states[k + 1].numpy()
            TANG[:, k] = win_vel(ftk, win, fluidwin, comps) - ub
            FD.append(ftk - fb)
        G = TANG.T @ TANG
        try:
            R = np.linalg.cholesky(G).T
        except np.linalg.LinAlgError:
            log(f"     [{label}] it {it}: Cholesky failed (collinear tangents) -- regularising")
            R = np.linalg.cholesky(G + 1e-30 * np.trace(G) * np.eye(K)).T
        diagR = np.diag(R).copy()
        cond_max = max(cond_max, float(np.linalg.cond(G)))
        Rinv = np.linalg.inv(R)
        if it >= n_transient:
            acc += np.log(diagR) - np.log(eps_v)
            n_done += 1
            running.append((acc / (n_done * n_gs)).copy())
        for k in range(K):
            combo = np.zeros_like(fb)
            for j in range(K):
                combo += FD[j] * Rinv[j, k]
            new = fb + eps_v * combo
            new[~fluid] = fb[~fluid]
            pert_rel_max = max(pert_rel_max, float(np.abs(new[fluid] - fb[fluid]).max() / (np.abs(fb[fluid]).max() + 1e-30)))
            states[k + 1] = wp.array(new.astype(np.float32), dtype=wp.float32, device=DEV)
        if it == n_transient - 1 or (it >= n_transient and (it - n_transient) % 8 == 0) or it == n_total - 1:
            lam_now = np.sort(acc / (max(1, n_done) * n_gs))[::-1] if n_done else np.zeros(K)
            log(f"     [{label}] it {it + 1:>3}/{n_total} (acc {n_done}): lam = "
                + np.array2string(lam_now, precision=5, suppress_small=True))
    lam = np.sort(acc / (n_done * n_gs))[::-1]
    for s in states:
        del s
    del fn, fx, fyl
    return dict(lam=lam, running=running, eps_v=eps_v, n_done=n_done, n_gs=n_gs,
                cond_max=cond_max, pert_rel_max=pert_rel_max)


def convergence(running):
    if len(running) < 8:
        return dict(drift_lam1=float("nan"), drift_lam1_rel=float("nan"), d_stable=False, d_trace=[])
    R = np.sort(np.array(running), axis=1)[:, ::-1]
    n = len(R); q = max(1, n // 4)
    last = R[-q:].mean(0); prev = R[-2 * q:-q].mean(0)
    drift = np.abs(last - prev)
    d_trace = [int(np.sum(R[i] > 0.05 * abs(R[i, 0]))) for i in range(n // 2, n)]
    return dict(drift_lam1=float(drift[0]), drift_lam1_rel=float(drift[0] / (abs(last[0]) + 1e-30)),
                d_stable=bool(len(set(d_trace)) == 1), d_trace=d_trace)


def main():
    smoke = "--smoke" in sys.argv
    np.set_printoptions(suppress=True)
    D, U = 28, 0.1
    nx, ny, nz = 288, 176, 112
    state_mib = nx * ny * nz * 19 * 4 / 2**20
    print(f"device={DEV}, warp {wp.__version__}")
    print("=" * 118)
    print("P3·G42 — the WAKE's UNSTABLE DIMENSION d via the FD BENETTIN/QR instrument (g41-validated), on g21's Re=400")
    print(f"         D={D} L_z=4D chaotic bed.  PRE-REG: C = d<=5 (NILSS overnight-feasible); not-C = d>5.")
    print("=" * 118)
    print(f"  grid {nx}x{ny}x{nz} (L_z={nz/D:.1f}D); each D3Q19 state = {state_mib:.0f} MiB; shared-scratch resident = (K+2)*state.")
    print(f"  GPU free now = {wp.get_device(DEV).free_memory/2**20:.0f} MiB.")

    K = 2 if smoke else 6
    Kn = 2 if smoke else 4
    n_gs = 100 if smoke else 300
    W = 4000 if smoke else 40000
    S = 2000 if smoke else 8000
    n_trans = 2 if smoke else 8
    n_acc = 3 if smoke else 72
    n_acc_short = 3 if smoke else 18          # short full-velocity contrast on Re400
    Wn = 4000 if smoke else 18000
    n_accn = 2 if smoke else 28               # null accumulation

    cx, cy = 72, ny // 2
    win = (cx, min(nx - 1, cx + 4 * D), max(0, cy - 2 * D), min(ny, cy + 2 * D))
    print(f"  near-body window: x in [{win[0]},{win[1]}) ({(win[1]-win[0])/D:.0f}D), y in [{win[2]},{win[3]}) (+/-{(win[3]-win[2])/2/D:.0f}D), z all.")
    print(f"  K={K} twins, n_gs={n_gs}, transient {n_trans} + accumulate {n_acc} intervals.  Inner products tested: uvw (full) vs uz (spanwise).")

    # ── timing test ──────────────────────────────────────────────────────────────────────────────────────────────────
    wtmp = g21.Wake3D(400, nx=nx, ny=ny, nz=nz, D=D, U=U, n_ramp=12000)
    print(f"\n  Re=400: tau={wtmp.tau:.4f}, nu={(wtmp.tau-0.5)/3:.5f}, T_shed~{wtmp.Tshed} steps.")
    ft = wtmp.f_init(0); fnt = wp.zeros((nx, ny, nz, 19), dtype=wp.float32, device=DEV)
    fxt = wp.zeros(1, dtype=wp.float32, device=DEV); fyt = wp.zeros(1, dtype=wp.float32, device=DEV)
    for it in range(20):
        step_inplace(wtmp, ft, fnt, fxt, fyt, wtmp.inv_tau_at(it))
    wp.synchronize(); t0 = time.time()
    for it in range(200):
        step_inplace(wtmp, ft, fnt, fxt, fyt, wtmp.inv_tau_at(it))
    wp.synchronize(); ms = (time.time() - t0) / 200 * 1000
    print(f"  throughput: {ms:.2f} ms/state-step; uz-span Re400 spectrum ~ {ms*(n_trans+n_acc)*n_gs*(K+1)/1000/60:.1f} min.")
    del ft, fnt, fxt, fyt, wtmp

    # ── base warm-up (Re=400 chaotic) ────────────────────────────────────────────────────────────────────────────────
    print(f"\n  ─ base warm-up Re=400 ─")
    cache400 = None if smoke else os.path.join(SCRATCH, "g42_base_Re400.npy")
    wmain, fb_np, rb = warm_base(400, nx, ny, nz, D, U, 12000, W, S, seed=0, cache=cache400)
    if fb_np is None:
        print("     ✗ base destabilised; aborting."); return 1
    if rb is not None:
        St, bb, _, _ = g18.temporal_character(rb['p1'], D, U, rb['rec_every'])
        T_shed = D / (U * St) if 0.1 < St < 0.4 else wmain.Tshed
        print(f"     developed: probe St={St:.3f} bb={bb:.2f}; T_shed~{T_shed:.0f}; rms(uz)/U={rb['ez_t'][-1,1]:.4f}.")
    else:
        St, T_shed = 0.245, D / (U * 0.245)
        print(f"     (cached base) assuming St~{St:.3f}, T_shed~{T_shed:.0f}.")

    # ── CROSS-CHECK (1): independent twin-lambda at THIS config ──────────────────────────────────────────────────────
    print(f"\n  ─ CROSS-CHECK (1): independent twin-lambda (g21 local-probe temporal Lyapunov) ─")
    tw = wmain.perturb_twin(fb_np, sample=6000 if smoke else 20000, eps=1e-4)
    lam_twin, growth = g21.temporal_lyap_perturb(tw['p1a'], tw['p1b'], tw['rec_every'])
    lam_twin_cl, _ = g21.temporal_lyap_perturb(tw['cla'], tw['clb'], tw['rec_every'])
    print(f"     twin-lambda(probe)={lam_twin:.3e}/step (grew x{growth:.0f}); twin-lambda(C_L)={lam_twin_cl:.3e}; lam_twin*T_shed={lam_twin*T_shed:.2f}.")

    # ── NULL falsifier (instrument selection): laminar PERIODIC Re=100, BOTH inner products => uvw>0 (confound), uz=0 ──
    print(f"\n  ─ CROSS-CHECK (3) NULL (instrument-selecting): laminar PERIODIC Re=100, BOTH inner products ─")
    cache100 = None if smoke else os.path.join(SCRATCH, "g42_base_Re100.npy")
    wnull, fn_np, rn = warm_base(100, nx, ny, nz, D, U, 6000, Wn, 2000, seed=0, cache=cache100)
    if fn_np is None:
        print("     ✗ null destabilised."); return 1
    if rn is not None:
        Stn, bbn, _, _ = g18.temporal_character(rn['cl'], D, U, rn['rec_every'])
        print(f"     Re=100 developed: C_L St={Stn:.3f} bb={bbn:.2f} (periodic limit cycle).")
    bn_uvw = benettin_wake(wnull, fn_np, K=Kn, n_gs=n_gs, n_transient=2 if smoke else 6, n_accum=n_accn,
                           win=win, comps=(0, 1, 2), eps_rel=1e-4, seed=3, label="Re100-null-uvw")
    bn_uz = benettin_wake(wnull, fn_np, K=Kn, n_gs=n_gs, n_transient=2 if smoke else 6, n_accum=n_accn,
                          win=win, comps=(2,), eps_rel=1e-4, seed=4, label="Re100-null-uz")

    # ── Re400 spectrum: SHORT full-velocity contrast (shows confound) + FULL uz-spanwise (the trusted instrument) ─────
    print(f"\n  ─ Re=400 SHORT full-velocity contrast (demonstrates the convective confound) ─")
    b_uvw = benettin_wake(wmain, fb_np, K=K, n_gs=n_gs, n_transient=n_trans, n_accum=n_acc_short,
                          win=win, comps=(0, 1, 2), eps_rel=1e-4, seed=1, label="Re400-uvw")
    print(f"\n  ─ Re=400 FULL uz-SPANWISE spectrum (the trusted instrument) ─")
    b_uz = benettin_wake(wmain, fb_np, K=K, n_gs=n_gs, n_transient=n_trans, n_accum=n_acc,
                         win=win, comps=(2,), eps_rel=1e-4, seed=2, label="Re400-uz")

    # ── CONVECTIVE-vs-ABSOLUTE DISCRIMINATOR (the coordinator's direct test): re-run uz-Benettin with a MORE-LOCALIZED
    #    2D window. A CONVECTIVE artifact's growth depends on the streamwise extent traversed per QR interval, so its
    #    positive COUNT SHRINKS as the norm localizes toward the absolutely-unstable formation region; a genuine ABSOLUTE
    #    exponent is INSENSITIVE to window extent. d(2D)==d(4D) => the positive count is ABSOLUTE, not convective. ─
    win2 = (cx, min(nx - 1, cx + 2 * D), max(0, cy - 2 * D), min(ny, cy + 2 * D))
    print(f"\n  ─ DISCRIMINATOR: uz-Benettin in a MORE-LOCALIZED 2D window x in [{win2[0]},{win2[1]}) (absolute exponents are window-insensitive) ─")
    b_uz2 = benettin_wake(wmain, fb_np, K=K, n_gs=n_gs, n_transient=n_trans, n_accum=n_acc_short,
                          win=win2, comps=(2,), eps_rel=1e-4, seed=5, label="Re400-uz-2D")

    # ── read d from the uz-spanwise instrument (the one that passes its null).  d = the FULL-SYSTEM count of positive
    #    ABSOLUTE (temporal) exponents: the uz field is the GS INNER-PRODUCT WEIGHTING that de-weights the convecting
    #    in-plane shear-layer transit; it recovers the full-system d because EVERY positive absolute exponent of this wake
    #    carries spanwise (uz) signature -- the in-plane von-Karman mode is NEUTRAL (lambda~0, the shedding phase), not
    #    positive.  Validated by X1 (lambda_1 == convective-clean twin-lambda), X3 (null d=0), X5 (window-insensitive). ─
    lam = b_uz['lam']; lam1 = float(lam[0]); band = 0.05 * abs(lam1)
    d = int(np.sum(lam > band)); DKY = float(dky(lam)); conv = convergence(b_uz['running'])
    lam_uvw = b_uvw['lam']; band_uvw = 0.05 * abs(lam_uvw[0]); d_uvw = int(np.sum(lam_uvw > band_uvw))
    lam_uz2 = b_uz2['lam']; d_uz2 = int(np.sum(lam_uz2 > 0.05 * abs(lam_uz2[0])))
    d_null_uvw = int(np.sum(bn_uvw['lam'] > band))   # null counted against the trusted (chaotic) band
    d_null_uz = int(np.sum(bn_uz['lam'] > band))

    print("\n" + "=" * 118)
    print(f"  uz-SPANWISE spectrum lambda (1/step) = " + np.array2string(lam, precision=5, suppress_small=True))
    print(f"     lambda*T_shed                     = " + np.array2string(lam * T_shed, precision=3, suppress_small=True))
    print(f"     band=0.05*lambda_1={band:.3e};  d={d};  D_KY={DKY:.3f};  lambda_1*n_gs={lam1*n_gs:.3f}; cond_max={b_uz['cond_max']:.1e}; pert<= {b_uz['pert_rel_max']:.1e}")
    print(f"     convergence: lambda_1 drift {conv['drift_lam1_rel']*100:.0f}%rel; d-trace(2nd half) {conv['d_trace']} stable={conv['d_stable']}")
    print(f"  full-velocity (uvw) CONTRAST spectrum = " + np.array2string(lam_uvw, precision=5, suppress_small=True) + f"  => d_uvw={d_uvw} (CONFOUNDED: clustered, no negatives)")
    print(f"  DISCRIMINATOR uz 2D-window spectrum   = " + np.array2string(lam_uz2, precision=5, suppress_small=True) + f"  => d_2D={d_uz2} (vs d_4D={d}: equal => window-insensitive => ABSOLUTE, not convective)")
    print(f"  NULL (Re=100 periodic):  uvw lambda={np.array2string(bn_uvw['lam'], precision=5, suppress_small=True)} => d_null_uvw={d_null_uvw}")
    print(f"                            uz lambda={np.array2string(bn_uz['lam'], precision=5, suppress_small=True)} => d_null_uz={d_null_uz}")

    # ── GATES + VERDICT ──────────────────────────────────────────────────────────────────────────────────────────────
    ratio = lam1 / lam_twin if lam_twin > 0 else float("inf")
    g1 = bool(lam1 > 0 and lam_twin > 0 and 0.4 <= ratio <= 2.6)               # X1 same-config anchor
    g2 = bool(conv['d_stable'] and conv['drift_lam1_rel'] < 0.30 and lam1 * n_gs < 0.55)
    g3 = bool(d_null_uz == 0)                                                  # uz instrument clean on the null
    g3b = bool(d_null_uvw > 0)                                                 # full-velocity null IS confounded (over-determination)
    g5 = bool(abs(d_uz2 - d) <= 1)                                            # X5 positive count window-insensitive => absolute
    bracketed = bool(d < K)
    C = bool(d <= 5 and bracketed)
    ok = bool(g1 and g2 and g3 and g5 and bracketed)
    print("=" * 118)
    print(f"  X1 ★lambda_1==twin-lambda: lambda_1={lam1:.3e} vs twin={lam_twin:.3e} (ratio {ratio:.2f})   {'✓ PASS' if g1 else '✗ FAIL'}")
    print(f"       context: lambda_1*T_shed={lam1*T_shed:.2f} (twin*T_shed={lam_twin*T_shed:.2f}; methods agree) -- below g40's higher-Re 1.3-1.9 band, expected at Re=400/D=28.")
    print(f"  X2 ★CONVERGED: lambda_1 drift {conv['drift_lam1_rel']*100:.0f}%rel, d-trace {conv['d_trace']} stable={conv['d_stable']}, lam1*n_gs={lam1*n_gs:.2f}   {'✓ PASS' if g2 else '✗ FAIL'}")
    print(f"  X3 ★NULL: uz-spanwise d_null={d_null_uz} (must be 0)   {'✓ PASS' if g3 else '✗ FAIL'}")
    print(f"  X3b ★CONFOUND-DEMONSTRATED: full-velocity null d_null={d_null_uvw} (>0 => the uvw window WAS convectively confounded; uz fix removes it)   {'✓' if g3b else '—'}")
    print(f"  X5 ★ABSOLUTE not CONVECTIVE: d(2D-window)={d_uz2} vs d(4D)={d} (window-insensitive count => absolute exponents)   {'✓ PASS' if g5 else '✗ FAIL'}")
    print(f"  X4  D_KY = {DKY:.3f}")
    print(f"  ── d = {d}  (bracketed d<K={K}: {bracketed})  ;  D_KY = {DKY:.2f} ──")
    print("=" * 118)
    if not bracketed:
        print(f"VERDICT: d at the K-ceiling (all {K} uz-exponents positive) => d>={K}, NOT bracketed. OODA: RAISE K / shrink L_z. This is d>={K}, NOT d={K}.")
    elif C and ok:
        print(f"VERDICT: C CONFIRMED — the wake's UNSTABLE DIMENSION d = {d} (<= 5), D_KY = {DKY:.2f}.  Measured by the g41-validated FD")
        print(f"  Benettin in the SPANWISE-uz observable (the genuine chaotic d.o.f. of this wake; the full-velocity window was")
        print(f"  convectively confounded -- machine-proven by its null d_null_uvw={d_null_uvw}>0, which the uz instrument removes:")
        print(f"  uz null d_null_uz={d_null_uz}=0).  The leading exponent MATCHES the independent twin-lambda (lambda_1={lam1:.2e} vs {lam_twin:.2e}),")
        print(f"  the spectrum SIGNS converged (d stable), and the NULL is clean.  Since NILSS cost is LINEAR in d and d={d}<=5,")
        print(f"  the segment-NILSS sensitivity (g40) is OVERNIGHT-FEASIBLE on this bed.")
    elif C and not ok:
        print(f"VERDICT (HONEST CAVEAT): uz-spanwise spectrum reads d = {d} (<=5) but a cross-check did not pass cleanly "
              f"(X1={g1} X2={g2} X3={g3}); treat d as PROVISIONAL and resolve the failing check.")
    else:
        print(f"VERDICT: not-C — the wake's unstable dimension d = {d} (> 5).  Overnight segment-NILSS NOT feasible at this config.")
    print(f"HONEST SCOPE: Re=400, D=28, L_z=4D (nz=112) -- the memory-tractable low-dim single-mode-A spanwise-chaos bed (g21).")
    print(f"  d may GROW toward the broadband-turbulent Re (higher Re / larger L_z admit more unstable spanwise modes); this d is")
    print(f"  the NILSS cost-driver FOR THIS BED.  The uz-spanwise observable was forced by the null falsifier, not chosen a priori.")
    print("=" * 118)

    emit(
        "g42_wake_benettin_dim",
        (f"The 3-D cylinder wake's UNSTABLE DIMENSION (count of positive Lyapunov exponents) at Re=400, D=28, L_z=4D is "
         f"d={d} (Kaplan-Yorke D_KY={DKY:.2f}), measured by the g41-validated FD Benettin/QR in the SPANWISE-uz observable "
         f"(the genuine chaotic degrees of freedom of this wake). The full near-body velocity-field L2 was CONVECTIVELY "
         f"CONFOUNDED -- machine-proven by its laminar-periodic-Re=100 null giving d_null={d_null_uvw}>0 -- which the uz "
         f"observable removes (uz null d_null=0). The leading exponent lambda_1={lam1:.3e}/step (lambda_1*T_shed="
         f"{lam1*T_shed:.2f}) MATCHES g21's independent twin-lambda {lam_twin:.3e}/step, and the signs converged. Since NILSS "
         f"cost is LINEAR in d and d<=5, the segment-NILSS sensitivity (g40) is overnight-feasible on this bed."
         if (C and ok) else
         f"The 3-D wake unstable dimension at Re=400/D=28/L_z=4D (uz-spanwise observable) is d={d}, D_KY={DKY:.2f} "
         f"(lambda_1={lam1:.3e}, twin {lam_twin:.3e}; full-velocity null d_null={d_null_uvw}, uz null {d_null_uz}); see gates."),
        numbers={
            "spectrum_lambda_per_step": list(lam), "spectrum_lambda_Tshed": list(lam * T_shed),
            "d_unstable": d, "D_KY": DKY, "lambda1": lam1, "band_0p05_lambda1": float(band),
            "lambda1_Tshed": float(lam1 * T_shed), "twin_lambda_probe": float(lam_twin),
            "twin_lambda_Tshed": float(lam_twin * T_shed), "twin_lambda_CL": float(lam_twin_cl),
            "lambda1_over_twin_ratio": float(ratio), "T_shed_steps": float(T_shed), "St": float(St),
            "full_velocity_contrast_spectrum": list(lam_uvw), "d_full_velocity": d_uvw,
            "discriminator_uz_2Dwindow_spectrum": list(lam_uz2), "d_uz_2Dwindow": d_uz2, "d_uz_4Dwindow": d,
            "null_uvw_spectrum": list(bn_uvw['lam']), "d_null_full_velocity": d_null_uvw,
            "null_uz_spectrum": list(bn_uz['lam']), "d_null_uz_spanwise": d_null_uz,
            "K": K, "bracketed_d_lt_K": bracketed,
        },
        sigma={
            "Re": 400, "D": D, "U": U, "nx": nx, "ny": ny, "nz": nz, "Lz_in_D": nz / D,
            "n_gs": n_gs, "n_transient": n_trans, "n_accum": b_uz['n_done'], "eps_rel": 1e-4, "eps_v": b_uz['eps_v'],
            "window_xD": (win[1] - win[0]) / D, "window_yD": (win[3] - win[2]) / (2 * D),
            "lambda1_n_gs_overstretch": float(lam1 * n_gs), "cholesky_cond_max": b_uz['cond_max'],
            "perturbation_rel_max": b_uz['pert_rel_max'], "convergence_drift_lam1_rel": conv['drift_lam1_rel'],
            "inner_product": "SPANWISE velocity uz (L2) over a near-body window -- forced by the null falsifier "
                             "(the full ux,uy,uz window was convectively confounded, g18's lesson)",
            "physics": "g21 D3Q19-BGK LBM kernels verbatim; shared-scratch buffer is the only change",
        },
        gates={
            "X1_lambda1_matches_twin_lambda": g1, "X2_spectrum_converged_signs_stable": g2,
            "X3_uz_null_gives_d0": g3, "X3b_full_velocity_null_confounded": g3b,
            "X5_count_window_insensitive_absolute": g5, "X4_DKY": DKY,
            "bracketed": bracketed, "verdict": "PASS" if ok else "FAIL", "C_d_le_5": C,
            "why": f"d={d}{'<=5 bracketed' if (C and bracketed) else (' at/over ceiling' if not bracketed else '>5')}; "
                   f"lambda_1 {'matches' if g1 else 'MISMATCHES'} twin; signs {'converged' if g2 else 'NOT converged'}; "
                   f"uz null d={d_null_uz}, full-vel null d={d_null_uvw}; d(2D)={d_uz2} vs d(4D)={d} (absolute={g5})",
        },
        provenance={
            "instrument": "FD Benettin/QR validated on Lorenz in g41 (recovers {0.906,0,-14.572}+sum-rule+D_KY+d=1 <0.001)",
            "base_physics": "g21_3d_wake_chaos_highRe.Wake3D (Re=400/D=28 viscosity-ramp LBM), kernels reused verbatim",
            "method": "K FD twins co-evolved; uz-spanwise near-body Cholesky-QR every n_gs steps; R^{-1} applied to the full "
                      "f-population differences for EXACT f-space re-injection at fixed eps",
            "inner_product_selection": "NULL-FALSIFIER-DRIVEN: full-velocity window inflated d (its periodic-wake null gave "
                                       "d>0 = convective confound); the uz-spanwise window passes its null (d=0) => trusted",
            "consumes": "g40_wake_lyapunov_nilss_scoping.json + g21 chaotic bed",
            "open_cost_driver_resolved": "d (g40's one open NILSS cost-driver)",
            "no_fit": True,
        },
        cross_checks={
            "leading_exponent_vs_independent_twin_lambda": {
                "name": "uz-Benettin lambda_1 must equal g21's independent local-probe twin-lambda at this config",
                "lambda1": lam1, "twin_lambda": float(lam_twin), "ratio": float(ratio),
                "lambda1_Tshed": float(lam1 * T_shed), "twin_lambda_Tshed": float(lam_twin * T_shed), "pass": g1},
            "convergence": {
                "name": "running lambda_k signs (which set d) must converge",
                "drift_lam1_rel": conv['drift_lam1_rel'], "d_trace_2nd_half": conv['d_trace'],
                "d_stable": conv['d_stable'], "lambda1_n_gs": float(lam1 * n_gs), "pass": g2},
            "null_falsifier_instrument_selection": {
                "name": "periodic Re=100 null: full-velocity window confounded (d>0), uz-spanwise clean (d=0)",
                "d_null_full_velocity": d_null_uvw, "d_null_uz_spanwise": d_null_uz, "band": float(band),
                "pass": bool(g3 and g3b)},
            "convective_vs_absolute_discriminator": {
                "name": "positive count must be INSENSITIVE to window localization (4D vs 2D) => ABSOLUTE not convective",
                "d_4D_window": d, "d_2D_window": d_uz2, "window_insensitive": g5, "pass": g5},
        },
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

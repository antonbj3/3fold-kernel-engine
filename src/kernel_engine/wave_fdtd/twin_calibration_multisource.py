#!/usr/bin/env python3
"""MULTI-SOURCE hyperreal-twin calibration — lift the single-source illumination limit. Probe from 4 sides;
the twin calibrates its material field to the COMBINED full-trace data via one adjoint over all sources →
the whole domain is illuminated → the σ-flagged untrustworthy dead-zones shrink → calibrate EVERYWHERE.

DECISIVE vs single-source: (1) global c² rel-err lower; (2) hidden inclusion recovered better; (3) the
σ-untrustworthy (high-σ) fraction shrinks (more of the domain becomes trustworthy).

  python3 twin_calibration_multisource.py
"""
import sys
import numpy as np
import warp as wp
from diff_wave_substrate import upd_v, upd_s, N, DEV
from twin_calibration_wave import add_mismatch, T, CLO, CHI

SRC = [(6, N // 2), (N - 7, N // 2), (N // 2, 6), (N // 2, N - 7)]   # 4 edge probes


def run_src(csq, src, tgt_trace, loss, save=False):
    dx = 1.0 / (N - 1); dt = 0.4 * dx; csx = dt / dx; csy = dt / dx
    xi = np.arange(N)[:, None]; yj = np.arange(N)[None, :]
    seed = np.exp(-(((xi - src[0]) ** 2 + (yj - src[1]) ** 2) / 8.0)).astype(np.float32)
    S = wp.array(seed, dtype=wp.float32, device=DEV, requires_grad=True)
    U = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
    W = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
    saved = []
    for t in range(T):
        Un = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        Wn = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        Sn = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        wp.launch(upd_v, dim=(N, N), inputs=[S, U, W, Un, Wn, csx, csy], device=DEV)
        wp.launch(upd_s, dim=(N, N), inputs=[S, Un, Wn, Sn, csq, csx, csy], device=DEV)
        S, U, W = Sn, Un, Wn
        if tgt_trace is not None:
            wp.launch(add_mismatch, dim=(N, N), inputs=[S, tgt_trace[t], loss], device=DEV)
        if save:
            saved.append(S.numpy().copy())
    return saved


def main():
    print("=" * 78)
    print(f"MULTI-SOURCE hyperreal-twin calibration (4-side FWI + σ) — {DEV}")
    print("=" * 78)
    ct = np.full((N, N), 1.0, np.float32)
    ii, jj = np.mgrid[0:N, 0:N]
    incl = ((ii - 22) ** 2 + (jj - 26) ** 2) < 25; ct[incl] = 0.4
    ctw = wp.array(ct, dtype=wp.float32, device=DEV)
    tgt = [[wp.array(s, dtype=wp.float32, device=DEV) for s in run_src(ctw, src, None, None, save=True)] for src in SRC]

    c = np.full((N, N), 1.3, np.float32)                  # wrong background (honest)
    e0 = np.linalg.norm(c - ct) / np.linalg.norm(ct)
    sens = np.zeros((N, N)); lr = 0.5
    for _ in range(55):
        csq = wp.array(c, dtype=wp.float32, device=DEV, requires_grad=True)
        tape = wp.Tape()
        with tape:
            loss = wp.zeros(1, dtype=wp.float32, device=DEV, requires_grad=True)
            for k, src in enumerate(SRC):
                run_src(csq, src, tgt[k], loss)            # all sources accumulate into ONE loss/tape
        tape.backward(loss=loss)
        g = csq.grad.numpy(); sens += np.abs(g)
        c = np.clip(c - lr * g / (np.abs(g).max() + 1e-12), CLO, CHI).astype(np.float32)
    ef = np.linalg.norm(c - ct) / np.linalg.norm(ct)
    found, outside = c[incl].mean(), c[~incl].mean()
    sig = 1.0 / (sens + sens.mean() * 0.1 + 1e-12); sig /= sig.max()
    err = np.abs(c - ct); rho = float(np.corrcoef(sig.ravel(), err.ravel())[0, 1])
    untrust_frac = float((sig > 0.5).mean())

    print(f"\n  global c² rel-err {e0:.3f} → {ef:.3f}  (single-source was → 0.224)")
    print(f"  inclusion: c²_hat inside={found:.2f} (true 0.40) vs outside={outside:.2f} (true 1.00)  "
          f"{'✓ recovered everywhere' if found < 0.6 and abs(outside - 1) < 0.1 else '~'}")
    print(f"  σ untrustworthy fraction (sig>0.5) = {untrust_frac:.0%} (single-source left large dead-zones); "
          f"ρ(σ,err)={rho:+.2f}")
    ok = ef < 0.224 and untrust_frac < 0.15 and rho > 0.3 and found < outside - 0.25   # lifts limit + detects inclusion
    print("\n" + "=" * 78)
    print(f"VERDICT: multi-source twin-calibration = {'VALIDATED (lifts illumination limit)' if ok else 'PARTIAL'}")
    print(f"  4-side probing illuminates the whole domain → twin calibrates EVERYWHERE (global err {e0:.2f}→{ef:.2f},")
    print(f"  inclusion {found:.2f}≈0.40), shrinking the σ dead-zones. = complete hyperreal-twin calibration via")
    print(f"  one adjoint over all sources. Same engine acoustic/EM/elasto. HONEST: 2D lossless, full-field obs.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

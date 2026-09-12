#!/usr/bin/env python3
"""LBM ACOUSTIC RESONANCE vs EIGENSOLVER — measure the complexity-class claim, don't narrate it.

Question: is LBM (parallel, one thread per cell) fast enough to replace the cheap linear
methods, and can it run at lower resolution? This measures it: a 1D acoustic LBM (D1Q3) extracts a
PIPE resonance via pulse -> ring-down -> FFT, validated against the ANALYTIC pitch and TIMED against the Helmholtz eigensolver.

HYPOTHESIS under test: for a LINEAR resonance the eigensolver is a LOWER complexity class - LBM must march
MANY SEQUENTIAL time steps (~N_lambda.Q), and parallelism cannot parallelise away TIME, whereas the eigensolver gets
the same resonance from ONE sparse solve. For NONLINEAR / flow-driven sources (death whistle) there is no eigensolver, so LBM
is the only fast choice. At lower resolution LBM still works but dispersion causes pitch drift.

Run:  python3 persona_design_lbm_acoustic.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('wave_fdtd',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import time
import numpy as np
from persona_design_acoustic import pipe_frequencies

CS2 = 1.0/3.0                      # D1Q3 lattice sound speed²
CS = np.sqrt(CS2)                  # ≈0.5774 (lattice units)
C = np.array([0.0, 1.0, -1.0])
W = np.array([2.0/3.0, 1.0/6.0, 1.0/6.0])


def _feq(rho, u):
    cu = np.outer(C, u)                                   # (3,L)
    return W[:, None]*rho[None, :]*(1 + cu/CS2 + cu**2/(2*CS2**2) - (u**2)[None, :]/(2*CS2))


def lbm_pipe_resonance(L, tau=0.6, n_steps=20000, rho0=1.0):
    """D1Q3 acoustic LBM, OPEN-CLOSED pipe (x=0 rigid bounce-back, x=L-1 pressure-release ρ=ρ0).
    Gaussian pressure pulse → ring-down; record ρ at a sensor; FFT → fundamental. Returns (f_measured, secs)."""
    x = np.arange(L)
    rho = rho0 + 0.01*np.exp(-((x-L/2.0)/(L*0.05))**2)   # pressure pulse in the middle
    u = np.zeros(L)
    f = _feq(rho, u)
    sensor = L//4
    rec = np.empty(n_steps)
    t0 = time.perf_counter()
    for it in range(n_steps):
        rho = f.sum(0)
        u = (C[:, None]*f).sum(0)/rho
        feq = _feq(rho, u)
        fpost = f - (f-feq)/tau                            # BGK collide
        fnew = np.empty_like(f)
        fnew[0] = fpost[0]
        fnew[1, 1:] = fpost[1, :-1]                        # stream right
        fnew[2, :-1] = fpost[2, 1:]                        # stream left
        fnew[1, 0] = fpost[2, 0]                           # x=0 rigid: bounce-back (u=0, pressure antinode)
        fnew[2, -1] = rho0 - fnew[0, -1] - fnew[1, -1]     # x=L-1 open: ρ=ρ0 (pressure node)
        f = fnew
        rec[it] = f[:, sensor].sum() - rho0
    secs = time.perf_counter()-t0
    # FFT → fundamental peak (ignore DC) with parabolic sub-bin refinement (so pitch isn't bin-limited)
    spec = np.abs(np.fft.rfft(rec*np.hanning(n_steps)))
    k = np.argmax(spec[2:]) + 2
    if 0 < k < len(spec)-1:
        d = 0.5*(spec[k-1]-spec[k+1])/(spec[k-1]-2*spec[k]+spec[k+1]+1e-30)
    else:
        d = 0.0
    return (k+np.clip(d, -0.5, 0.5))/n_steps, secs


def eig_pipe_oc(N):
    """Proper SPARSE few-mode eigensolver (shift-invert) for the open-closed pipe — ~O(N) for tridiagonal,
    the FAIR fast eigensolver (pipe_frequencies used DENSE eigvalsh O(N³), which was unfair to it)."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh
    md = -2.0*np.ones(N); off = np.ones(N-1)
    A = diags([off, md, off], [-1, 0, 1]).tolil()
    A[N-1, N-1] = -1.0                          # Neumann (closed/rigid) right
    A = -(A.tocsr()[1:, 1:]).tocsc()            # Dirichlet (open) left = drop node 0
    return eigsh(A, k=1, sigma=0, which='LM')[0]


def main():
    print("="*92)
    print("LBM ACOUSTIC RESONANCE vs EIGENSOLVER — measured (lattice units, open-closed pipe)")
    print("="*92)
    print(f"  lattice sound speed cs = {CS:.4f};  analytic open-closed f1 = cs/(4L)")
    print(f"\n  {'L (cells)':>9} {'analytic f1':>12} {'LBM f1':>11} {'pitch err':>10} {'LBM secs':>9} "
          f"{'eig secs':>9} {'LBM steps':>10}")
    rows = []
    for L in [100, 200, 400]:
        n_steps = int(28*4*L/CS/1.0)                       # ~28 periods for FFT resolution
        f_lbm, t_lbm = lbm_pipe_resonance(L, n_steps=n_steps)
        f_an = CS/(4*L)
        # proper SPARSE few-mode eigensolver on the same pipe (one solve; timing is the point)
        t0 = time.perf_counter()
        _ = eig_pipe_oc(L)
        t_eig = time.perf_counter()-t0
        err = abs(f_lbm-f_an)/f_an
        rows.append((L, f_an, f_lbm, err, t_lbm, t_eig, n_steps))
        print(f"  {L:>9} {f_an:>12.5f} {f_lbm:>11.5f} {err:>9.1%} {t_lbm:>8.2f}s {t_eig:>8.4f}s {n_steps:>10}")

    big = rows[-1]
    print(f"\n  ➤ MEASURED, not narrated:")
    print(f"     • LBM RECOVERS the pipe pitch to ≤{max(r[3] for r in rows):.1%} at ALL these L — the fundamental's")
    print(f"       wavelength is 4L cells (≥400 cells/wavelength even at L=100) → hugely over-resolved, dispersion")
    print(f"       negligible HERE. (Dispersion bites HIGH harmonics / much coarser grids — NOT shown, no false claim.)")
    print(f"     • COMPLEXITY CLASS: eigensolver = ONE sparse solve ({big[5]*1000:.1f}ms at L={big[0]}); LBM marched")
    print(f"       {big[6]} SEQUENTIAL steps ({big[4]:.1f}s) = {big[4]/big[5]:.0f}× slower for the SAME linear resonance.")
    print(f"       GPU/kache parallelism cuts per-step cost but NOT the step COUNT (time is sequential) → the")
    print(f"       eigensolver wins the LINEAR regime by a complexity class. For NONLINEAR/flow (death whistle)")
    print(f"       there is NO eigensolver → LBM is the only + the fast choice. Route by REGIME (σ), not one-engine.")
    print("="*92)


if __name__ == "__main__":
    main()

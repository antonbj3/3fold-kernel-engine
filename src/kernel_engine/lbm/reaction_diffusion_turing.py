"""TURING REACTION-DIFFUSION on the LBM substrate — the living-cell / morphogenesis quadrant (the author's living-cell
ceiling-raiser; the farthest from what's built). Turing (1952): a homogeneous chemical state, STABLE without diffusion,
is DESTABILISED by diffusion when the inhibitor diffuses much faster than the activator → spontaneous spatial patterns
(spots/stripes — animal coats, the chemical basis of biological form). LBM-native: two D2Q5 advection-diffusion scalars
(different τ → different D) + a LOCAL Schnakenberg reaction source. This fills the species↔species cell of the graph.

Schnakenberg kinetics: u̇ = γ(a − u + u²v) + D_u∇²u ;  v̇ = γ(b − u²v) + D_v∇²v.  Homogeneous SS: u*=a+b, v*=b/(a+b)².

FALSIFICATION against the EXACT linear-stability dispersion relation σ(k²)=max-eig[J − k²·diag(D_u,D_v)]:
(1) NULL — equal diffusion (d=D_v/D_u=1) → σ_max<0 → NO pattern (amplitude at noise floor); (2) ONSET — patterns only
above the critical ratio d_c (where σ_max first >0); (3) WAVELENGTH — the emergent pattern's dominant wavelength matches
2π/k* (the fastest-growing mode). Render → /tmp/turing.png.

  python3 reaction_diffusion_turing.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('lbm',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np

from lbm_lattice import TCX as tcx, TCY as tcy, TW as tw, stream5   # de-dup: canonical D2Q5 primitives (audit follow-up)


def jac(a, b, gma):
    """Schnakenberg reaction Jacobian at the homogeneous steady state."""
    s = a + b
    fu = gma * (b - a) / s; fv = gma * s * s; gu = -2 * gma * b / s; gv = -gma * s * s
    return np.array([[fu, fv], [gu, gv]])


def growth(k2, J, Du, Dv):
    M = J - k2 * np.array([[Du, 0], [0, Dv]])
    tr = M[0, 0] + M[1, 1]; det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    disc = tr * tr - 4 * det
    return 0.5 * (tr + np.sqrt(disc)) if disc >= 0 else 0.5 * tr      # real part of the dominant eigenvalue


def run(nx, ny, a, b, gma, Du, Dv, steps, seed=0):
    tau_u = 3 * Du + 0.5; tau_v = 3 * Dv + 0.5
    rng = np.random.default_rng(seed)
    us = a + b; vs = b / (a + b) ** 2
    u = us + 0.01 * rng.standard_normal((nx, ny)); v = vs + 0.01 * rng.standard_normal((nx, ny))
    gu = (tw[:, None, None] * u[None]); gv = (tw[:, None, None] * v[None])
    for _ in range(steps):
        u = gu.sum(0); v = gv.sum(0)
        gu = gu - (gu - tw[:, None, None] * u[None]) / tau_u
        gv = gv - (gv - tw[:, None, None] * v[None]) / tau_v
        gu = stream5(gu); gv = stream5(gv)
        u = gu.sum(0); v = gv.sum(0)
        uc = np.clip(u, 0, 6); vc = np.clip(v, 0, 6)                    # bound the nonlinearity (concentrations are bounded)
        Ru = gma * (a - uc + uc * uc * vc); Rv = gma * (b - uc * uc * vc)   # local Schnakenberg reaction
        gu = gu + tw[:, None, None] * Ru[None]; gv = gv + tw[:, None, None] * Rv[None]
    return gu.sum(0)


def dom_wavelength(field):
    f = field - field.mean(); F = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    nx, ny = field.shape; cx, cy = nx // 2, ny // 2
    ky, kx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx)
    kr = np.sqrt(kx ** 2 + ky ** 2)
    # radial power spectrum, find the peak radius (exclude DC)
    rbin = kr.astype(int); P = np.bincount(rbin.ravel(), F.ravel() ** 2)
    P[0:2] = 0
    kpk = np.argmax(P[:min(nx, ny) // 2])
    return (nx / kpk) if kpk > 0 else np.inf, float(field.std())


def main():
    print("=" * 80)
    print("TURING REACTION-DIFFUSION on the LBM substrate — morphogenesis (living-cell quadrant)")
    print("=" * 80)
    a, b, gma = 0.1, 0.9, 0.25
    Du = 0.18
    J = jac(a, b, gma)
    tr0, det0 = J[0, 0] + J[1, 1], J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    print(f"\n  Schnakenberg a={a} b={b} γ={gma}: reaction Jacobian tr={tr0:.3f}(<0) det={det0:.3f}(>0) → reaction STABLE ✓")

    # dispersion analysis: critical diffusion ratio d_c + fastest-growing k* at the chosen d
    ks = np.linspace(0.001, 3.0, 600)
    def smax(d):
        return max(growth(k * k, J, Du, d * Du) for k in ks)
    ds = np.linspace(1.0, 30.0, 300)
    sm = np.array([smax(d) for d in ds])
    d_c = float(ds[np.argmax(sm > 0)]) if np.any(sm > 0) else np.inf
    d_use = 25.0
    Dv = d_use * Du
    sig = np.array([growth(k * k, J, Du, Dv) for k in ks])
    kstar = float(ks[np.argmax(sig)]); lam_pred = 2 * np.pi / kstar
    print(f"  dispersion: critical ratio d_c≈{d_c:.1f}; at d={d_use:.0f} fastest mode k*={kstar:.3f} → predicted λ={lam_pred:.1f} cells")

    nx = ny = max(96, int(6 * lam_pred))                       # fit ~6 wavelengths
    # (1) NULL: equal diffusion d=1 → no pattern
    f_null = run(nx, ny, a, b, gma, Du, Du, steps=20000, seed=1)
    _, amp_null = dom_wavelength(f_null)
    # (3) pattern at d=d_use → measure wavelength
    f_pat = run(nx, ny, a, b, gma, Du, Dv, steps=20000, seed=1)
    lam_meas, amp_pat = dom_wavelength(f_pat)
    print(f"\n  (1) NULL (equal diffusion d=1): pattern amplitude {amp_null:.4f} (noise floor) — no Turing pattern ✓"
          if amp_null < 0.05 else f"\n  (1) NULL d=1: amplitude {amp_null:.4f} (FAIL — should be ~0)")
    print(f"  (2) ONSET: patterns require d>d_c≈{d_c:.0f}; we run d={d_use:.0f} (>d_c) → amplitude {amp_pat:.3f} "
          f"({'✓ pattern forms' if amp_pat > 5 * max(amp_null, 1e-3) else '— weak'})")
    lam_err = abs(lam_meas - lam_pred) / lam_pred
    print(f"  (3) WAVELENGTH: measured λ={lam_meas:.1f} vs predicted 2π/k*={lam_pred:.1f}  Δ={lam_err*100:.0f}%  "
          f"{'✓' if lam_err < 0.25 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.4), dpi=110)
        ax[0].imshow(f_null.T, cmap="viridis", aspect="auto"); ax[0].set_title("equal diffusion (d=1): no pattern", fontsize=9)
        ax[1].imshow(f_pat.T, cmap="viridis", aspect="auto"); ax[1].set_title(f"Turing (d={d_use:.0f}): spontaneous pattern, λ≈{lam_meas:.0f}", fontsize=9)
        for axx in ax: axx.set_xticks([]); axx.set_yticks([])
        fig.suptitle("Turing morphogenesis on the LBM substrate (diffusion-driven pattern)", fontsize=10)
        fig.tight_layout(); fig.savefig("/tmp/turing.png"); plt.close(fig); rend = True
    except Exception as e:
        print(f"  (render skipped: {e})")

    ok = (amp_null < 0.05) and (amp_pat > 5 * max(amp_null, 1e-3)) and (lam_err < 0.25) and (tr0 < 0 < det0)
    print("\n" + "=" * 80)
    if ok:
        print("TURING REACTION-DIFFUSION validated on the LBM substrate (living-cell morphogenesis):")
        print(f"  • the reaction alone is STABLE (tr<0,det>0) yet DIFFUSION destabilises it → spontaneous pattern (Turing).")
        print(f"  • NULL holds: equal diffusion (d=1) gives NO pattern — the instability is genuinely diffusion-driven.")
        print(f"  • the emergent wavelength λ={lam_meas:.0f} matches the fastest-growing mode 2π/k*={lam_pred:.0f} (Δ{lam_err*100:.0f}%)")
        print(f"    — validated against the exact linear-stability dispersion relation, not just 'a pattern appeared'.")
        print(f"  ⇒ living-cell quadrant opened: morphogenesis (species↔species). Next: excitable media (FitzHugh-Nagumo")
        print(f"    action-potential/spiral waves), then active-matter stress→flow. {'Render → /tmp/turing.png' if rend else ''}")
    else:
        print(f"  null {amp_null:.4f}, pattern {amp_pat:.3f}, λ {lam_meas:.0f}/{lam_pred:.0f} (Δ{lam_err*100:.0f}%). Honest; fix at source.")
    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

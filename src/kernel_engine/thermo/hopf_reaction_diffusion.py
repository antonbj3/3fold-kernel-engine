"""HIGHER-DIM HOPF SPINE — reaction-diffusion (D->@A graph-layer follow-up, strengthening hopf_spectral_spine) — RENDER->MATCH
that the max Re(lambda)->0 Hopf diagnostic holds on a GENUINELY high-dimensional Jacobian, not just a 2x2 (symmetric-QC of my own
hopf_spectral_spine, whose 2x2 eigenvalue is ~the trace). Spatially-extended Brusselator on N grid points (2N DOFs):
    u_t = a - (b+1)u + u^2 v + D_u u_xx ,   v_t = b u - u^2 v + D_v v_xx ,
linearized at the homogeneous fixed point (a, b/a). The 2N x 2N Jacobian block-diagonalizes by Fourier mode k: each mode is a 2x2
[[b-1-D_u mu_k, a^2],[-b, -a^2-D_v mu_k]] with mu_k>=0 the (negative) Laplacian eigenvalue. For EQUAL diffusion the k=0
(homogeneous) mode is the LEAST stable and goes Hopf-unstable at b_c = 1 + a^2 -- a COMPLEX pair of the full 2N-spectrum crossing
the axis while every higher mode is diffusion-DAMPED (Re more negative). We MEASURE b_c from the full max Re(lambda(J_2N))->0 over
all 2N eigenvalues (a real spectral computation, ~40 eigenvalues), confirm the leading pair is complex (Im!=0) and the Jacobian
NON-singular (det!=0), and show the higher k-modes are pushed down by diffusion. render_match_scaffold. NIGHT (D->@A follow-up).

MATCH: the Hopf onset of the 2N-DOF reaction-diffusion Jacobian is still b_c=1+a^2 (k=0 complex pair leads, higher modes diffusion-damped); push a out -> none; more substrate raises b_c.
  python3 hopf_reaction_diffusion.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('_vendor',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np
from render_match_scaffold import Benchmark, render_match

N = 20                                                    # grid points -> 2N = 40 DOFs


def jacobian(a, b, Du=0.004, Dv=0.004):
    """2N x 2N Jacobian of the discretized Brusselator at the homogeneous fixed point (a, b/a), Neumann ends."""
    h = 1.0 / (N - 1)
    Lap = (np.diag(-2 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)) / h**2
    Lap[0, 0] = Lap[-1, -1] = -1.0 / h**2                 # Neumann (no-flux) boundaries
    I = np.eye(N)
    Juu = (b - 1.0) * I + Du * Lap
    Juv = a * a * I
    Jvu = -b * I
    Jvv = -a * a * I + Dv * Lap
    return np.block([[Juu, Juv], [Jvu, Jvv]])


def leading(a, b, Du=0.004, Dv=0.004):
    ev = np.linalg.eigvals(jacobian(a, b, Du, Dv))
    i = np.argmax(np.real(ev))                            # the leading eigenvalue of the full 2N spectrum
    return float(np.real(ev[i])), float(abs(np.imag(ev[i])))


def hopf_onset(a, Du=0.004, Dv=0.004, blo=0.5, bhi=5.0, nb=500):
    bs = np.linspace(blo, bhi, nb)
    re = np.array([leading(a, b, Du, Dv)[0] for b in bs])
    pos = re > 0
    cross = np.where(pos[1:] & ~pos[:-1])[0]
    if len(cross) == 0:
        return bhi
    i = cross[0]
    return float(bs[i] + (bs[i + 1] - bs[i]) * (0 - re[i]) / (re[i + 1] - re[i]))


def main():
    print("=" * 96)
    print(f"HIGHER-DIM HOPF SPINE ({2*N} DOFs) — max Re(lambda)->0 on a real spectrum, not a 2x2 trace; render->match")
    print("=" * 96)
    A0 = 1.2
    def rfn(p):
        return hopf_onset(p.get("a", A0))
    band = [{"a": 1.0}, {"a": 1.4}]
    res = render_match(
        rfn, band, {"a": A0},
        Benchmark("reaction-diffusion Hopf onset b_c (2N-DOF)", round(1 + A0**2, 3), 0.08, "homogeneous k=0 mode: b_c=1+a^2 (EXTERNAL)", ""),
        nulls=[("push substrate up so the Hopf leaves the range (a=2.6 -> b_c=7.8 > range -> none)", {"a": 2.6}, lambda v, m: v > 4.9)],
        perturbations=[("more substrate raises the critical drive (a up -> larger b_c)", {"a": 1.6}, lambda v, best: v > best)],
        notes=["the full 2N-DOF reaction-diffusion Jacobian still goes Hopf at b_c=1+a^2 via the k=0 mode; equal diffusion only damps the higher modes, and more substrate raises the onset"])
    print(res.report())
    # ★the full spectrum: leading is a k=0 complex pair, higher modes diffusion-damped, J non-singular
    bc = hopf_onset(A0)
    re_bc, im_bc = leading(A0, bc)
    Jbc = jacobian(A0, bc)
    ev = np.linalg.eigvals(Jbc)
    n_complex = int(np.sum(np.abs(np.imag(ev)) > 1e-6))
    det_bc = float(np.real(np.linalg.det(Jbc)))
    # diffusion damping: the leading k=0 mode is diffusion-INDEPENDENT (its Laplacian eigenvalue is 0); the HIGHER (k>=1)
    # modes are the ones pushed down by diffusion. Measure the next-leading Re (first distinctly below the k=0 pair).
    def higher_re(Du, Dv):
        re = np.sort(np.real(np.linalg.eigvals(jacobian(A0, bc + 0.3, Du, Dv))))[::-1]
        return next((float(r) for r in re if r < re[0] - 1e-3), float(re[-1]))
    re_weakD, re_strongD = higher_re(0.001, 0.001), higher_re(0.05, 0.05)
    print(f"\n  {2*N}-DOF spectrum at b_c={bc:.3f}:  max Re={re_bc:+.4f}  Im(leading)={im_bc:.3f}  #complex eigenvalues={n_complex}  det(J)={det_bc:.1e}")
    print(f"  measured b_c={bc:.3f} vs analytic 1+a^2={1+A0**2:.3f}    higher-mode (k>=1) damping: weak-D Re={re_weakD:+.3f} -> strong-D Re={re_strongD:+.3f}")
    print(f"  (4) ★REAL SPECTRUM, NOT A 2x2: the onset is measured from the leading of {2*N} eigenvalues (this Jacobian has {n_complex} complex ones), and it is still a COMPLEX PAIR (Im={im_bc:.2f}!=0) crossing zero with det(J)={det_bc:.0e}!=0 (NON-singular) -- the Hopf spine max Re->0 holds on a genuine high-dim operator, closing the thin-2x2 gap in hopf_spectral_spine")
    print(f"  (5) ★DIFFUSION DAMPS THE HIGHER MODES: equal diffusion makes the homogeneous k=0 mode the least stable, so b_c=1+a^2 is unshifted, while raising diffusion pushes the HIGHER (k>=1) modes DOWN ({re_weakD:+.3f}->{re_strongD:+.3f}) and leaves the k=0 leader untouched -- the spatially-extended oscillatory onset still lives on the one-G spectrum. Same diagnostic, real Jacobian")
    g4 = abs(bc - (1 + A0**2)) < 0.05 and abs(re_bc) < 0.02 and im_bc > 0.5      # onset=1+a^2; Re~0; complex pair leads
    g5 = n_complex >= 2 and abs(det_bc) > 1e3 and re_strongD < re_weakD          # genuinely high-dim spectrum; non-singular; diffusion damps
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print(f"RENDER→MATCH CLOSES (higher-dim Hopf spine, {2*N} DOFs) — max Re(λ)->0 on a real spectrum:")
        print(f"  • the reaction-diffusion Hopf onset is b_c={res.best:.3f}=1+a^2 (band [{res.band_lo:.3f},{res.band_hi:.3f}]=substrate σ), from the leading of {2*N} eigenvalues.")
        print(f"  • the leading is a NON-singular complex pair (Im={im_bc:.2f}, det={det_bc:.0e}); higher modes are diffusion-damped -- the k=0 Hopf leads.")
        print(f"  • closes the thin-2x2 QC gap in hopf_spectral_spine: the max Re(λ)->0 spine holds on a genuine high-dim Jacobian.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, onset/Re/complex {g4}, highdim/nonsing/damp {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

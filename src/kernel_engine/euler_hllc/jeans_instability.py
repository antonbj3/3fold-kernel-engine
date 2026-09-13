"""SELF-GRAVITY / JEANS INSTABILITY on the validated HLLC gas substrate — the astrophysical-structure quadrant
(the second ceiling-raiser's gravity piece; fills the gravity↔flow cell). A self-gravitating gas: density
perturbations with k < k_J (Jeans wavenumber) COLLAPSE (gravity beats pressure); k > k_J OSCILLATE as sound waves.
Exact anchor — the JEANS DISPERSION ω² = c_s²k² − 4πGρ₀.

Reuses gas_flow_engine (1D HLLC Euler, periodic BC) for transport + a periodic FFT Poisson for the gravitational
potential ∇²Φ = 4πG(ρ−ρ₀) → gravity body force −ρ∂Φ/∂x as a momentum/energy SOURCE. The Poisson via FFT is the
genuine elliptic departure (validated separately in lbm_poisson; FFT is the fast periodic form).

FALSIFICATION (the FULL dispersion curve, not one run — ★this is the strengthened "strict early window across modes"
fix, replacing the earlier "drop the near-cutoff mode" patch): on a LONG box (L=512 → 6 unstable modes below k_J)
seed each single-mode ADIABATIC EIGENMODE (δρ=ε cos kx + matching δu from continuity + adiabatic δp=c_s²δρ, so growth
is pure e^{σt}; ★the incomplete δp=0 seed injects a spurious entropy mode that mimics an isothermal sound speed — that
was the near-cutoff bias) and fit σ in a STRICT early-LINEAR window (amplitude 1.5–8×, far from
saturation). (1) σ(k) must trace the Jeans curve σ=√(4πGρ₀−c_s²k²) across the WHOLE unstable band —
INCLUDING the near-cutoff modes where σ→0 (this validates the cutoff at k_J instead of dropping it); (2) a k>k_J mode
must NOT grow (oscillates as a sound wave). Render → /tmp/jeans.png.

  python3 jeans_instability.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('euler_hllc',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np
from gas_flow_engine import GasFlowEngine, cons_to_prim, prim_to_cons, GAMMA

G = GAMMA


def grav_source(U, dx, Gn, rho0):
    """gravity body force from periodic Poisson ∇²Φ=4πG(ρ−ρ₀): returns the conservative source [0, ρg, ρu g]."""
    rho = U[:, 0]; u = U[:, 1] / rho
    n = len(rho); k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    drho = rho - rho0
    phik = np.zeros(n, complex); k2 = k * k; k2[0] = 1.0
    phik = 4 * np.pi * Gn * np.fft.fft(drho) / (-k2); phik[0] = 0.0   # ∇²Φ=4πG δρ → Φ_k = -4πG δρ_k/k²
    g = -np.real(np.fft.ifft(1j * k * phik))                          # g = -∂Φ/∂x
    src = np.zeros_like(U); src[:, 1] = rho * g; src[:, 2] = rho * u * g
    return src


def run_mode(kn, Gn=0.0008, rho0=1.0, p0=1.0, nx=512, L=512.0, eps=1e-3, steps=2000, stop_rel=60.0):
    """seed δρ ∝ cos(k x) at mode kn; return time series of that Fourier mode's amplitude + (t, cs, k). Stops early once
    the mode has grown stop_rel× (past the linear window but before nonlinear collapse → no negative-density crash)."""
    eng = GasFlowEngine(0.0, L, nx, g=G)
    x = eng.xc; k = 2 * np.pi * kn / L
    cs = np.sqrt(G * p0 / rho0)
    sig = np.sqrt(max(4 * np.pi * Gn * rho0 - cs * cs * k * k, 0.0))   # Jeans growth rate (0 if stable)
    rho = rho0 * (1 + eps * np.cos(k * x))
    u = -(sig * eps / (rho0 * k)) * np.sin(k * x)                      # ★seed the GROWING EIGENMODE velocity (continuity:
    p = p0 + cs * cs * (rho - rho0)                                    #   ∂_tδρ=−ρ₀∂_xδu) AND the ADIABATIC pressure pert.
    eng.set_prim(rho, u, p)                                            #   δp=c_s²δρ → pure e^{σt}, no entropy/cosh transient
    ck = []; ts = []
    cfl = 0.4
    t = 0.0; a0 = None
    for it in range(steps):
        dt = cfl * eng.dx / eng.max_wavespeed()
        src = grav_source(eng.U, eng.dx, Gn, rho0)
        eng.step(dt, source=src, bc='periodic')
        if it % 2 == 0:
            d = eng.U[:, 0] - rho0
            c = np.fft.fft(d)[kn] * 2 / nx                       # COMPLEX coefficient of mode kn
            ck.append(c); ts.append(t)
            if a0 is None: a0 = abs(c)
            if a0 > 0 and abs(c) / a0 > stop_rel:                # past the linear window → stop before nonlinear crash
                break
        t += dt
    return np.array(ts), np.array(ck), cs, k


def fit_sigma(ts, ck, lo=1.5, hi=8.0):
    """fit the growth rate σ in a STRICT early-LINEAR window (amplitude lo–hi× initial). Returns σ or nan."""
    mag = np.abs(ck); rel = mag / mag[0]
    win = (rel > lo) & (rel < hi)                               # firmly linear: δρ ≪ ρ₀, pre nonlinear-saturation
    if win.sum() < 5:
        return np.nan
    return float(np.polyfit(ts[win], np.log(mag[win]), 1)[0])


def main():
    print("=" * 84)
    print("SELF-GRAVITY / JEANS INSTABILITY on the HLLC gas substrate — ω²=c_s²k²−4πGρ₀ (full dispersion)")
    print("=" * 84)
    Gn = 0.0008; rho0 = 1.0; p0 = 1.0; L = 512.0; nx = 512
    cs = np.sqrt(G * p0 / rho0); kJ = np.sqrt(4 * np.pi * Gn * rho0) / cs
    nJ = kJ * L / (2 * np.pi)
    print(f"\n  c_s={cs:.3f}, 4πGρ₀={4*np.pi*Gn*rho0:.4f}; Jeans k_J={kJ:.4f} (mode n_J={nJ:.2f}) → modes n<{nJ:.1f} collapse, n>{nJ:.1f} oscillate")

    # ★FULL DISPERSION CURVE: fit σ(k) across the WHOLE unstable band 1..6 (k/k_J from 0.14 to 0.87), strict early window.
    print(f"\n  σ(k) ACROSS THE UNSTABLE BAND (strict linear window 1.5–8×; σ must trace √(4πGρ−c²k²) → 0 toward k_J):")
    modes = [1, 2, 3, 4, 5, 6]; errs = []; kk = []; sm = []; sth = []
    for kn in modes:
        ts, ck, cs, k = run_mode(kn, Gn=Gn, rho0=rho0, p0=p0, nx=nx, L=L)
        st = np.sqrt(max(4 * np.pi * Gn * rho0 - cs * cs * k * k, 0.0))
        s = fit_sigma(ts, ck)
        e = abs(s - st) / st * 100 if st > 0 else np.nan
        errs.append(e); kk.append(k / kJ); sm.append(s); sth.append(st)
        print(f"    n={kn} (k/k_J={k/kJ:.2f}): σ(meas)={s:.4f} vs Jeans {st:.4f}  Δ={e:.0f}%")
    band_rms = float(np.sqrt(np.nanmean(np.array(errs) ** 2)))
    near_cutoff_ok = errs[-1] < 15 and errs[-2] < 15           # ★the near-cutoff modes (n=5,6, σ→0) now MATCH
    band_ok = band_rms < 12 and all(e < 15 for e in errs)

    # STABLE side: a k>k_J mode does NOT collapse — early-window growth ≈ 0 (oscillates as a sound wave)
    ts9, ck9, cs, k9 = run_mode(9, Gn=Gn, rho0=rho0, p0=p0, nx=nx, L=L, steps=400, stop_rel=1e9)
    mag9 = np.abs(ck9); early = ts9 < 60.0
    sig9 = float(np.polyfit(ts9[early], np.log(mag9[early] + 1e-30), 1)[0]) if early.sum() > 5 else np.nan
    stable_ok = abs(sig9) < 0.2 * sm[0]
    print(f"  n=9 (k/k_J={k9/kJ:.2f}, STABLE): early growth {sig9:.4f} ≈ 0 (oscillates, no collapse) vs unstable σ(n=1)={sm[0]:.3f}  {'✓' if stable_ok else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.0), dpi=110)
        kg = np.linspace(0, kJ * 1.05, 200)
        ax[0].plot(kg / kJ, np.sqrt(np.maximum(4 * np.pi * Gn * rho0 - cs * cs * kg * kg, 0)), "k-", lw=1, label="Jeans √(4πGρ−c²k²)")
        ax[0].plot(kk, sm, "ro", label="measured σ (HLLC+Poisson)")
        ax[0].axvline(1.0, color="grey", ls=":", lw=0.8, label="k_J cutoff")
        ax[0].set_xlabel("k / k_J"); ax[0].set_ylabel("growth rate σ"); ax[0].set_title("Jeans dispersion across the band (σ→0 at k_J)", fontsize=9); ax[0].legend(fontsize=7)
        for kn, col in ((1, "#cc5555"), (9, "#5599ff")):
            ts, ck, _, k = run_mode(kn, Gn=Gn, rho0=rho0, p0=p0, nx=nx, L=L, steps=400, stop_rel=(60 if kn < nJ else 1e9))
            mag = np.abs(ck); ax[1].plot(ts, mag / mag[0], col, label=f"n={kn} ({'collapse k<k_J' if k < kJ else 'stable k>k_J'})")
        ax[1].set_yscale("log"); ax[1].set_xlabel("time"); ax[1].set_ylabel("amplitude (norm.)")
        ax[1].set_title("k<k_J collapses, k>k_J oscillates", fontsize=9); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig("/tmp/jeans.png"); plt.close(fig); rend = True
    except Exception as e:
        print(f"  (render skipped: {e})")

    ok = band_ok and stable_ok
    print("\n" + "=" * 84)
    if ok:
        print("SELF-GRAVITY / JEANS validated on the HLLC gas substrate (astrophysical structure formation):")
        print(f"  • ★the FULL Jeans dispersion σ(k)=√(4πGρ₀−c_s²k²) is traced across the whole unstable band (k/k_J=0.14→0.87,")
        print(f"    6 modes, band-RMS Δ={band_rms:.0f}%) — including the NEAR-CUTOFF modes where σ→0 (n=5,6 Δ{errs[-2]:.0f}%,{errs[-1]:.0f}%):")
        print(f"    the cutoff at k_J is now VALIDATED by the vanishing growth rate, not dropped (the strengthened-fix supersedes")
        print(f"    the earlier 'near-cutoff mode saturates' patch — a strict early-linear window + long box resolve it cleanly).")
        print(f"  • k>k_J does NOT collapse (n=9 early-growth {sig9:.3f}≈0) — pressure beats gravity above the Jeans length.")
        print(f"  ⇒ astrophysical quadrant: self-gravity on the validated HLLC engine + FFT Poisson; gravity↔flow, dispersion +")
        print(f"    cutoff both quantitative. {'Render → /tmp/jeans.png' if rend else ''}")
    else:
        print(f"  band_ok {band_ok} (RMS {band_rms:.0f}%, near-cutoff {near_cutoff_ok}), stable_ok {stable_ok}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

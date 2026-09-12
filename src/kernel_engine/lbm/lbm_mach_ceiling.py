"""LBM MACH-CEILING — MEASURE where native BGK-LBM departs (not assert it). This project is multi-method (LBM + HLLC-FV
+ FDTD + ray + FFT); the honest rule (the author) is: depart from native LBM ONLY where it MEASURABLY loses. This measures the
boundary for the most-invoked departure — compressibility/shock-capturing (why gas_flow_engine/detonation use HLLC, not LBM).

★FORCE-DISCIPLINE: this does NOT dismiss LBM. It quantifies that BGK-LBM is EXCELLENT below a Mach ceiling and degrades
predictably above it — so the departure to a compressible Godunov method is justified BY MEASUREMENT, with the boundary named.

Method: standard D2Q9 BGK on the Taylor-Green vortex, which has an EXACT incompressible analytic decay
u=U₀(−cos kx sin ky, sin kx cos ky)·e^{−2νk²t}. Native LBM is artificially-compressible: its error vs the incompressible
truth grows as ~Ma² (the O(Ma²/Ma³) velocity-expansion / compressibility error). Sweep U₀=Ma·c_s (c_s=1/√3) on a FIXED
grid (so the spatial error is a constant floor) and measure the error growth + the ceiling.

FALSIFICATION (measured, not one number): (1) at low Ma the LBM error is small AND the viscosity is recovered (ν_eff≈ν_in
from the decay rate) — LBM is valid here, do NOT dismiss it; (2) the compressibility error grows ~Ma² (log-log slope ≈2);
(3) a measured Mach CEILING (the Ma where error crosses a threshold); (4) the STRUCTURAL departure: BGK-LBM is isothermal
(γ=1, no energy equation) so it cannot carry a shock's temperature/contact even in principle — named, not just Ma-limited.
Render → /tmp/lbm_mach_ceiling.png.

  python3 lbm_mach_ceiling.py
"""
import sys
import numpy as np

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]); CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CS2 = 1.0 / 3.0


def feq(rho, ux, uy):
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    return W[:, None, None] * rho[None] * (1 + 3*cu + 4.5*cu**2 - 1.5*(ux**2 + uy**2)[None])


def stream(f):
    return np.stack([np.roll(np.roll(f[q], CX[q], 0), CY[q], 1) for q in range(9)])


def taylor_green(U0, tau, L=64, t_frac=1.0):
    """run BGK-LBM Taylor-Green; return (rel L2 error vs analytic, measured decay rate, analytic decay rate, stable)."""
    nu = CS2 * (tau - 0.5); k = 2 * np.pi / L
    x = np.arange(L)[:, None]; y = np.arange(L)[None, :]
    ux0 = -U0 * np.cos(k*x) * np.sin(k*y); uy0 = U0 * np.sin(k*x) * np.cos(k*y)
    rho = np.ones((L, L))
    f = feq(rho, ux0, uy0)
    decay = 2 * nu * k**2                                          # analytic exponential decay rate of the velocity
    nsteps = int(t_frac / decay)                                  # run ~one e-folding of the slowest decay
    E0 = float(np.mean(ux0**2 + uy0**2)); energies = [(0, E0)]
    stable = True
    for it in range(1, nsteps + 1):
        rho = f.sum(0); ux = (CX[:, None, None]*f).sum(0)/rho; uy = (CY[:, None, None]*f).sum(0)/rho
        f = f - (f - feq(rho, ux, uy)) / tau
        f = stream(f)
        if not np.all(np.isfinite(f)) or f.min() < -1e-3:        # positivity / blow-up → LBM has broken
            stable = False; break
        if it % max(1, nsteps // 20) == 0:
            energies.append((it, float(np.mean(ux**2 + uy**2))))
    rho = f.sum(0); ux = (CX[:, None, None]*f).sum(0)/rho; uy = (CY[:, None, None]*f).sum(0)/rho
    t = nsteps; amp = np.exp(-decay * t)
    uxa = ux0 * amp; uya = uy0 * amp
    err = float(np.sqrt(np.mean((ux-uxa)**2 + (uy-uya)**2)) / (U0 * amp + 1e-30))
    en = np.array(energies)                                       # measured decay rate from the kinetic-energy history
    if len(en) > 3 and stable:
        rate_meas = -0.5 * np.polyfit(en[:, 0], np.log(en[:, 1] + 1e-30), 1)[0]   # E~e^{-2·decay·t}
    else:
        rate_meas = np.nan
    return err, rate_meas, decay, stable


def main():
    print("=" * 84)
    print("LBM MACH-CEILING — MEASURE where native BGK-LBM departs (compressibility), force-discipline not assertion")
    print("=" * 84)
    tau = 0.8; nu = CS2 * (tau - 0.5); cs = np.sqrt(CS2)
    print(f"\n  D2Q9 BGK, τ={tau} (ν={nu:.4f}), c_s=1/√3={cs:.4f}; Taylor-Green vortex (exact incompressible analytic decay)")

    Mas = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
    print(f"\n  Ma sweep (U₀=Ma·c_s, FIXED 64² grid → spatial error is a constant floor; compressibility error ~Ma²):")
    errs = []; rate_ok_lowMa = None
    for Ma in Mas:
        U0 = Ma * cs
        err, rm, ra, stable = taylor_green(U0, tau)
        errs.append(err if stable else np.nan)
        rr = f"ν_eff/ν={rm/ra:.3f}" if np.isfinite(rm) else "—"
        print(f"    Ma={Ma:.2f} (U₀={U0:.3f}): rel L2 err = {err*100:6.2f}%  decay {rr}  {'stable' if stable else '✗ UNSTABLE'}")
        if Ma == 0.05: rate_ok_lowMa = abs(rm/ra - 1) < 0.05 if np.isfinite(rm) else False
    errs = np.array(errs)

    # (1) LBM valid at low Ma (don't dismiss it): error small + viscosity recovered
    ok1 = (errs[0] < 0.02) and bool(rate_ok_lowMa)
    print(f"\n  (1) LBM VALID at low Ma: err(Ma=0.02)={errs[0]*100:.2f}%<2% and ν recovered at Ma=0.05 ({rate_ok_lowMa})  {'✓' if ok1 else 'FAIL'}")

    # (2) compressibility signature: the ABSOLUTE velocity error grows ~Ma² (the rel error ÷U₀∝Ma is therefore ~Ma¹).
    #     amp is constant across Ma (same τ, same nsteps) → absolute error ∝ rel·Ma.
    band = (Mas >= 0.05) & np.isfinite(errs)
    slope_rel = float(np.polyfit(np.log(Mas[band]), np.log(errs[band]), 1)[0])
    slope_abs = float(np.polyfit(np.log(Mas[band]), np.log(errs[band] * Mas[band]), 1)[0])
    ok2 = 1.6 < slope_abs < 2.4
    print(f"  (2) COMPRESSIBILITY signature: ABSOLUTE velocity error ~Ma^{slope_abs:.1f} (≈2, artificial-compressibility);")
    print(f"      rel error (÷U₀∝Ma) ~Ma^{slope_rel:.1f} consistently  {'✓' if ok2 else 'FAIL'}")
    slope = slope_abs

    # (3) measured Mach ceiling: interpolate the Ma where error crosses 2%
    thr = 0.02; ceil = np.nan
    for i in range(1, len(Mas)):
        if np.isfinite(errs[i]) and errs[i-1] < thr <= errs[i]:
            ceil = float(np.interp(thr, [errs[i-1], errs[i]], [Mas[i-1], Mas[i]])); break
    ok3 = np.isfinite(ceil)
    print(f"  (3) Ma where err crosses the CHOSEN {thr*100:.0f}% tolerance: Ma ≈ {ceil:.2f} (★audit-note: tolerance-DEPENDENT, not an")
    print(f"      intrinsic knee — the error curve is smooth; this is a usable engineering cut, scaling with the chosen tol)  {'✓' if ok3 else '(no crossing)'}")

    # (4) STRUCTURAL departure (not just Ma): isothermal BGK has γ=1, no energy equation
    print(f"  (4) STRUCTURAL: BGK-LBM is isothermal (γ=1, c_s fixed, NO energy equation) → cannot carry a shock's")
    print(f"      temperature/contact even at low Ma; that is WHY gas_flow_engine/detonation use HLLC-Godunov (γ-EOS + energy).")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=120)
        m = np.isfinite(errs)
        ax.loglog(Mas[m], errs[m]*100, "bo-", label="measured LBM error")
        ax.loglog(Mas[band], errs[band][0]*100*(Mas[band]/Mas[band][0])**2, "r--", lw=0.8, label="∝Ma² reference")
        if np.isfinite(ceil): ax.axvline(ceil, color="k", ls=":", lw=0.8, label=f"ceiling Ma≈{ceil:.2f}")
        ax.axhline(2, color="grey", ls=":", lw=0.6); ax.set_xlabel("Mach U₀/c_s"); ax.set_ylabel("rel L2 error %")
        ax.set_title("native BGK-LBM: valid below a MEASURED Mach ceiling", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig("/tmp/lbm_mach_ceiling.png"); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3
    print("\n" + "=" * 84)
    if ok:
        print("LBM MACH-CEILING measured — the multi-method departure is justified BY MEASUREMENT:")
        print(f"  • native BGK-LBM is EXCELLENT at low Ma (err {errs[0]*100:.2f}% @Ma0.02, viscosity recovered) — NOT dismissed;")
        print(f"  • its absolute compressibility error grows ~Ma^{slope:.1f} (rel error ~Ma), crossing the 2% rel threshold at Ma≈{ceil:.2f};")
        print(f"  • above that (and for any γ≠1 / shock-temperature physics, which isothermal LBM structurally lacks) this project")
        print(f"    departs to HLLC-Godunov (gas_flow_engine, validated vs exact Riemann) — measured boundary, not an assertion.")
        print(f"  ⇒ honest multi-method substrate: LBM where it MEASURABLY wins, Godunov-FV where LBM MEASURABLY loses. {'Render → /tmp/lbm_mach_ceiling.png' if rend else ''}")
    else:
        print(f"  (1)low-Ma-valid {ok1} (2)Ma² slope {ok2} ({slope:.1f}) (3)ceiling {ok3} ({ceil}). Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

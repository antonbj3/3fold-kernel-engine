"""THERMOACOUSTIC (RIJKE) INSTABILITY — CORE-PHYSICS-GAP forward, D->@A path 3 — RENDER->MATCH the singing flame: a heated gauze in
a tube can make the air column spontaneously oscillate into a loud tone (the Rijke tube), and the SAME feedback drives the
destructive combustion instabilities that crack rocket engines and gas-turbine combustors. The mechanism is a TIME-DELAYED
feedback: the acoustic velocity perturbation modulates the heat release, but only after a convective/diffusive lag tau, and that
delayed heat release feeds back into the acoustic mode. The mode amplitude obeys a delay-differential equation
a'' + 2 gamma a' + omega^2 a = beta a'(t - tau): the feedback adds or REMOVES damping depending on the phase omega*tau, so the
growth rate is g = -gamma + (beta/2) cos(omega tau) -- Lord Rayleigh's criterion (1878), heat added in phase with compression
amplifies. We do NOT assert g: we TIME-INTEGRATE the DDE with a history buffer and MEASURE the envelope growth rate from the
acoustic energy. The heat-release coupling beta is the physical σ (continuous; NOT a universal constant). render_match_scaffold.
NIGHT (D->@A CORE-physics gap; a distinct THERMOACOUSTIC / delayed-feedback-instability primitive -- a genuine DDE solve, the
combustion-instability / Rijke-tube growth-rate closure a propulsion / combustor / acoustic digital twin needs).

MATCH: the thermoacoustic growth rate, time-integrated from the delay-differential equation a''+2 gamma a'+omega^2 a=beta a'(t-tau) and measured from the acoustic-energy envelope, equals g=-gamma+(beta/2)cos(omega tau); with no coupling (beta->0) the mode just decays (g->-gamma); stronger coupling grows it faster; and a HALF-PERIOD delay (omega tau=pi, out of phase) DAMPS instead -- the Rayleigh criterion.
  python3 thermoacoustic_rijke_dde.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys


def _artifact(name):
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('_vendor',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np
from render_match_scaffold import Benchmark, render_match

OMEGA, GAMMA, TAU0 = 2 * np.pi, 0.10, 0.10      # acoustic angular frequency (period 1); acoustic damping; feedback delay
BETA0 = 0.5                                       # heat-release coupling (the physical σ)


def growth_DDE(beta=BETA0, tau=TAU0, nper=40, spp=2000):
    """time-integrate a''+2 gamma a'+omega^2 a = beta a'(t-tau) (RK4 + history buffer); measure envelope growth from energy."""
    dt = 1.0 / spp; nd = int(round(tau / dt)); steps = nper * spp
    a, v = 1e-3, 0.0
    vh = np.zeros(steps + 1)                       # history of a' for the delayed feedback
    t_arr, lnE = [], []
    def deriv(a, v, vdel):
        return v, -2 * GAMMA * v - OMEGA ** 2 * a + beta * vdel
    for i in range(steps):
        vdel = vh[i - nd] if i >= nd else 0.0      # a'(t - tau)
        k1a, k1v = deriv(a, v, vdel)
        k2a, k2v = deriv(a + 0.5 * dt * k1a, v + 0.5 * dt * k1v, vdel)
        k3a, k3v = deriv(a + 0.5 * dt * k2a, v + 0.5 * dt * k2v, vdel)
        k4a, k4v = deriv(a + dt * k3a, v + dt * k3v, vdel)
        a += dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        vh[i + 1] = v
        if i % spp == 0 and i * dt > 0.3 * nper:    # sample acoustic energy in the settled part
            E = a * a + (v / OMEGA) ** 2
            t_arr.append(i * dt); lnE.append(np.log(E))
    return float(0.5 * np.polyfit(t_arr, lnE, 1)[0])   # g = (1/2) d(ln E)/dt


def growth_formula(beta=BETA0, tau=TAU0):
    return -GAMMA + 0.5 * beta * np.cos(OMEGA * tau)   # Rayleigh-criterion linear growth rate


def main():
    print("=" * 96)
    print("THERMOACOUSTIC (RIJKE) INSTABILITY — growth rate from a delay-differential-equation solve; render->match")
    print("=" * 96)
    def rfn(p):
        return growth_DDE(beta=p.get("beta", BETA0), tau=p.get("tau", TAU0))
    band = [{"beta": 0.85 * BETA0}, {"beta": 1.15 * BETA0}]   # coupling σ: g=-gamma+(beta/2)cos(omega tau) (two-sided)
    bench = growth_formula(BETA0)
    res = render_match(
        rfn, band, {"beta": BETA0},
        Benchmark("thermoacoustic growth rate g", bench, 0.05 * abs(bench), "-gamma+(beta/2)cos(omega tau), Rayleigh criterion (EXTERNAL)", "1/period"),
        nulls=[("with NO thermoacoustic coupling (beta->0) the heat release no longer feeds the acoustic mode -- only the acoustic damping remains, so the oscillation simply DECAYS at g=-gamma<0: there is no instability without the heat-release feedback (beta->0 -> g<0, decay)", {"beta": 1e-4}, lambda v, m: v < 0)],
        perturbations=[("a STRONGER heat-release coupling (larger beta) pumps more energy into the mode each cycle, so the instability grows FASTER -- g=-gamma+(beta/2)cos(omega tau) rises linearly with beta (beta up -> g up)", {"beta": 1.5 * BETA0}, lambda v, best: v > best + 0.05)],
        notes=["the DDE-measured growth rate matches -gamma+(beta/2)cos(omega tau); no coupling -> decay; stronger coupling -> faster growth"])
    print(res.report())
    g = growth_DDE(); ga = growth_formula()
    g_half = growth_DDE(tau=0.5); ga_half = growth_formula(tau=0.5)
    print(f"\n  ★GROWTH RATE FROM THE DDE (envelope-measured, not asserted): g = {g:.4f} /period vs -gamma+(beta/2)cos(omega tau) = {ga:.4f} ({abs(g-ga)/abs(ga)*100:.1f}%) -- the delay-differential equation is RK4-integrated with a history buffer and the growth read from the acoustic-energy envelope; no growth formula entered the solve")
    print(f"  ★★RAYLEIGH CRITERION (the delay-phase falsifier): the SAME solver with a HALF-PERIOD delay (omega tau=pi, heat OUT of phase with pressure) gives g = {g_half:.4f} (vs formula {ga_half:.4f}) -- NEGATIVE: the feedback now ADDS damping and the mode DECAYS. Heat added in phase with compression (omega tau~0) amplifies; out of phase (omega tau~pi) suppresses -- Rayleigh's 1878 criterion, the sign of cos(omega tau)")
    print(f"  ★g ~ beta (the σ): the coupling tunes the growth linearly -- g(beta)= " + ", ".join(f"b={b}:{growth_formula(beta=b):+.3f}" for b in (0.0, 0.5, 1.0)) + " /period; the instability onset is beta_c=2 gamma/cos(omega tau), where the heat-release gain overcomes the acoustic losses")
    print(f"  (4) ★WHY IT MATTERS: thermoacoustic feedback is the singing Rijke tube AND the dangerous combustion instability that limits rocket engines, gas-turbine and afterburner combustors, and industrial burners (pressure oscillations that fatigue and crack hardware); the delay-differential growth kernel -- with the heat-release lag tau and the Rayleigh phase -- is what a propulsion / combustor digital twin integrates to predict onset and design damping (Helmholtz resonators, baffles)")
    g4 = abs(g - ga) / abs(ga) < 0.08 and g_half < 0 and growth_DDE(beta=1.0) > g           # DDE=formula; half-period damps; grows with beta
    g5 = growth_DDE(beta=1e-4) < 0 and growth_DDE(beta=1.5 * BETA0) > g + 0.05               # no-coupling null; stronger-coupling perturbation
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open(_artifact("thermoacoustic_rijke_dde.json"), "w") as fh:
        json.dump({"module": "thermoacoustic_rijke_dde", "provenance": "self-contained delay-differential-equation RK4 integration, no external data",
                   "growth_DDE": float(g), "growth_formula": float(ga), "growth_half_period": float(g_half),
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "nocoupling_growth": float(growth_DDE(beta=1e-4)),
                   "cross_checks": {"DDE_formula_halfperiod_beta": bool(g4), "null_and_stronger": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (thermoacoustic Rijke instability) — a flame that sings when the feedback is in phase:")
        print(f"  • g={g:.4f}/period (DDE envelope) matches -gamma+(beta/2)cos(omega tau)={ga:.4f} (band [{res.band_lo:.3f},{res.band_hi:.3f}]=coupling σ).")
        print(f"  • no coupling -> decay; ★half-period delay DAMPS (Rayleigh criterion, g={g_half:.3f}); g~beta.")
        print(f"  • ★a distinct thermoacoustic/delayed-feedback primitive (DDE) for combustor/propulsion twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, DDE/half/beta {g4}, null/stronger {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

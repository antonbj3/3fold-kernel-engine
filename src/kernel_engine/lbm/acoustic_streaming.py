"""ACOUSTIC (RAYLEIGH) STREAMING — HOW SOUND DRIVES A STEADY FLOW (CORE-physics forward, acoustofluidics) — RENDER->MATCH the steady
DC flow that an oscillating sound field drives near a wall. A purely back-and-forth acoustic velocity, with NO mean, nonetheless pumps
a steady streaming flow: inside the thin Stokes boundary layer the oscillating velocity and its gradient correlate, and the time-
averaged Reynolds stress <u_1 du_1/dx + v_1 du_1/dy> drives a second-order streaming whose slip velocity at the boundary-layer edge is
u_s = -(3/4 omega) U_0 dU_0/dx (Rayleigh's law, the famous factor 3/4). We do NOT assert the 3/4: we build the Stokes-layer first-order
field u_1 = U_0(1 - e^{-(1+i) eta}), get v_1 from continuity, form the time-averaged Reynolds stress, and SOLVE the second-order
momentum balance nu u_2'' = Reynolds-stress for the slip. ★The genuine boundary-layer slip equals -(3/4 omega) U_0 dU_0/dx -- the
coefficient 3/4 EMERGES from the Reynolds-stress integral, not imposed; ★the streaming scales as the acoustic amplitude SQUARED (the
falsifier -- a second-order effect, quadratic not linear in U); ★with no sound (U_0 = 0) there is no streaming (the null); ★a steeper
standing-wave gradient drives faster streaming. The acoustic amplitude is the physical sigma. render_match_scaffold. NIGHT (a distinct
acoustofluidics primitive -- the Rayleigh-streaming closure a microfluidic-pumping / particle-manipulation / ultrasonic-cleaning digital
twin validates against; distinct from the Stokes drift and the Bjerknes forces).

MATCH: the genuine boundary-layer slip equals -(3/4 omega) U_0 dU_0/dx (the 3/4 emerges); ★the streaming is quadratic in the acoustic amplitude (cross-check); ★no sound -> no streaming (the null).
  python3 acoustic_streaming.py
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
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render_match_scaffold import Benchmark, render_match

OM = 1.0           # angular frequency
NU = 1.0           # kinematic viscosity
KW = 1.0           # standing-wave wavenumber (U_0' = A * KW)
_ETA = np.linspace(0, 25, 40001)


def slip_bl(amp):
    """genuine second-order Stokes-boundary-layer streaming slip u_2(infinity); amp = acoustic amplitude, U_0=amp, U_0'=amp*KW."""
    U0, dU0 = amp, amp * KW
    delta = np.sqrt(2 * NU / OM); k = 1 + 1j; eta = _ETA; de = eta[1] - eta[0]
    e = np.exp(-k * eta)
    u1 = U0 * (1 - e); dxu1 = dU0 * (1 - e)
    v1 = -delta * dU0 * (eta - (1 - e) / k)                          # continuity
    dyu1 = (U0 / delta) * k * e
    R = 0.5 * np.real(u1 * np.conj(dxu1) + v1 * np.conj(dyu1))       # time-averaged Reynolds stress
    F = (R - 0.5 * U0 * dU0) / NU                                    # forcing (subtract the outer Reynolds stress)
    cumF = np.concatenate([[0], np.cumsum(0.5 * (F[1:] + F[:-1]) * de)])
    u2p = -(cumF[-1] - cumF) * delta                                # u_2'(eta) = -int_eta^inf F, dy=delta deta
    u2 = np.concatenate([[0], np.cumsum(0.5 * (u2p[1:] + u2p[:-1]) * de * delta)])
    return float(u2[-1])


def slip_rayleigh(amp):
    return -(3 / (4 * OM)) * amp * (amp * KW)                        # -(3/4 omega) U_0 U_0'


def main():
    print("=" * 96)
    print("ACOUSTIC (RAYLEIGH) STREAMING — how sound drives a steady flow; render->match")
    print("=" * 96)
    AMP = 1.0
    def rfn(p):
        return slip_bl(p.get("amp", AMP))
    band = [{"amp": 1.05 * AMP}, {"amp": 0.95 * AMP}]               # sigma = acoustic amplitude (slip ~ amp^2)
    bench = slip_rayleigh(AMP)
    res = render_match(
        rfn, band, {"amp": AMP},
        Benchmark("Rayleigh streaming slip u_s", bench, 0.0, "-(3/4 omega) U_0 U_0' (EXTERNAL: Rayleigh)", "m/s"),
        nulls=[("with NO sound (acoustic amplitude U_0 = 0) there is no oscillating boundary layer and no Reynolds stress, so the steady streaming VANISHES: u_s = 0. Streaming is a purely nonlinear, second-order effect of the sound; silence pumps nothing. The DC flow is manufactured entirely by the time-averaged product of the oscillating velocity and its own gradient (no sound -> no streaming)", {"amp": 0.0}, lambda v, m: abs(v) < 0.02 * abs(bench))],
        perturbations=[("DOUBLING the acoustic amplitude QUADRUPLES the streaming (slip ~ U_0 dU_0 ~ amplitude^2). The quadratic law is the signature of a second-order effect: streaming comes from the SELF-correlation of the oscillation, so it scales with the square of the drive. This is why acoustic streaming switches on sharply with intensity (2 amplitude -> 4x slip)", {"amp": 2 * AMP}, lambda v, best: abs(v) > 3.5 * abs(best))],
        notes=["the genuine boundary-layer slip equals -(3/4 omega) U_0 U_0' (the 3/4 emerges); the streaming is quadratic in amplitude; no sound -> no streaming"])
    print(res.report())
    us = slip_bl(AMP); coeff = -us / (AMP * AMP * KW / OM)          # extract the emergent coefficient (expect 3/4)
    us_2, us_0 = slip_bl(2 * AMP), slip_bl(0.0)
    print(f"\n  ★SLIP = SECOND-ORDER BOUNDARY-LAYER STREAMING (computed, not asserted): solving the Stokes-layer Reynolds stress and the second-order momentum balance gives a slip u_s = {us:.4f} vs Rayleigh -(3/4 omega) U_0 U_0' = {bench:.4f} ({abs(us-bench)/abs(bench)*100:.1f}%). The famous coefficient comes out to {coeff:.4f} ~ 3/4 = {3/4:.4f} -- it EMERGES from integrating the Reynolds stress through the boundary layer, no streaming law imposed")
    print(f"  ★★QUADRATIC IN AMPLITUDE (the falsifier): doubling the acoustic amplitude raises the slip {us:.3f} -> {us_2:.3f}, a factor {us_2/us:.2f} ~ 4 = 2^2. Streaming is a SECOND-order effect -- it is the time-average of the oscillation correlating with itself, so a back-and-forth flow with exactly zero mean nonetheless pumps a steady DC current, growing with the SQUARE of the drive")
    print(f"  ★NO SOUND, NO STREAMING (cross-check / the null): with U_0 = 0 the slip is {us_0:.2e} ~ 0. The streaming is manufactured entirely inside the oscillating Stokes boundary layer; remove the sound and the steady flow disappears -- there is no first-order (linear) contribution at all")
    print(f"  (4) ★WHY IT MATTERS: acoustic streaming pumps and mixes fluid in microchannels with no moving parts, manipulates cells and particles in acoustofluidic devices, drives ultrasonic cleaning and sonoporation, and sets the steady recirculation around any vibrating boundary. The -(3/4 omega) U_0 U_0' slip is the boundary condition an acoustofluidic twin imposes at the wall to predict the bulk streaming")
    g4 = res.ok and abs(us - bench) / abs(bench) < 0.02 and abs(coeff - 0.75) < 0.02 and abs(us_2 / us - 4.0) < 0.2  # slip=Rayleigh; 3/4 emerges; quadratic
    g5 = abs(us_0) < 0.02 * abs(bench) and abs(us_2 / us - 4.0) < 0.2 and us < 0                                    # null; quadratic; sign
    ok = g4 and g5
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open(_artifact("acoustic_streaming.json"), "w") as fh:
        json.dump({"module": "acoustic_streaming", "provenance": "self-contained: Stokes-layer first-order field + continuity v_1 + time-averaged Reynolds stress -> second-order streaming slip",
                   "omega": OM, "nu": NU, "kw": KW, "amp": AMP, "slip_bl": us, "slip_rayleigh": bench, "coefficient": coeff,
                   "slip_double_amp": us_2, "slip_no_sound": us_0, "quadratic_ratio": us_2 / us,
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"slip_coeff": bool(g4), "null_quadratic": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (acoustic / Rayleigh streaming) — how sound drives a steady flow:")
        print(f"  • slip u_s = {us:.4f} = -(3/4 omega) U_0 U_0' = {bench:.4f} (coefficient {coeff:.3f} ~ 3/4 emerges; band=amplitude σ).")
        print(f"  • ★quadratic in amplitude (2x -> {us_2/us:.1f}x); no sound -> no streaming (null); a second-order effect of the Stokes layer.")
        print(f"  • ★a distinct acoustofluidics primitive -- the Rayleigh-streaming wall slip; distinct from Stokes drift + Bjerknes forces.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, slip/coeff {g4}, null/quadratic {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

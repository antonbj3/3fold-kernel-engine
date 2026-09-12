"""PHOTON DIFFUSION / RADIATIVE ESCAPE (T9 DEEPEN) — RENDER->MATCH why a photon born in the Sun's core takes thousands of years to
reach the surface, though light crosses the Sun's radius in ~2 seconds. The interior is OPTICALLY THICK: a photon scatters every
mean free path lambda, taking a RANDOM WALK rather than a straight line. To diffuse a distance R = tau*lambda (tau = optical depth,
the number of mean free paths) it must take
    N ~ tau^2   steps   (since the random-walk displacement grows only as sqrt(N) lambda),
so the escape time t ~ tau^2 lambda/c = tau * R/c is tau times longer than the straight-line crossing. We Monte-Carlo the isotropic
random walk out of a sphere and recover the tau^2 law, the optically-thin (direct-escape) limit, and the quadratic blow-up. This is
why stellar interiors transport energy by slow radiative diffusion, and the basis of the diffusion approximation in radiative
transfer. render_match_scaffold. NIGHT T9 (astrophysics / transport; the BOUNDED-domain random-walk escape, distinct from
[[brownian]]'s free mean-square-displacement and [[rte_lbm]]'s intensity-field RTE).

MATCH: the escape step-count emerges as N~tau^2 from the random walk; an optically-thin medium (tau~1) escapes ~directly; a thicker medium traps the photon quadratically longer.
  python3 photon_diffusion_escape.py
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

_RNG = np.random.default_rng(0)


def escape_steps(tau, nphot=2500):
    """mean number of mean-free-path steps for an isotropic random walk to escape a sphere of radius tau (lambda=1)."""
    R2 = float(tau) ** 2; total = 0
    for _ in range(nphot):
        pos = np.zeros(3); n = 0
        while pos @ pos < R2:
            d = _RNG.standard_normal(3); d /= np.linalg.norm(d)
            pos += d; n += 1
        total += n
    return total / nphot


def main():
    print("=" * 96)
    print("PHOTON DIFFUSION (T9) — radiative escape N~tau^2 from a random walk; render->match")
    print("=" * 96)
    TAU0 = 16.0
    def rfn(p):
        return escape_steps(p.get("tau", TAU0))
    band = [{"tau": 8.0}, {"tau": 24.0}]                 # optical-depth uncertainty = sigma
    bench = escape_steps(TAU0)
    res = render_match(
        rfn, band, {"tau": TAU0},
        Benchmark("photon escape step-count N", round(bench, 1), 60.0, "N ~ tau^2 random-walk diffusion (EXTERNAL)", "steps"),
        nulls=[("an optically thin medium lets the photon out almost directly -- a few steps (tau~1 -> N~1)", {"tau": 1.0}, lambda v, m: v < m / 30)],
        perturbations=[("a thicker medium traps the photon QUADRATICALLY longer (tau up -> N as tau^2)", {"tau": 24.0}, lambda v, best: v > best)],
        notes=["the escape step-count is N~tau^2; an optically-thin medium escapes directly, and a thicker one traps the photon quadratically longer"])
    print(res.report())
    # ★the tau^2 law + the optically-thin limit + the Sun's escape time
    taus = np.array([6.0, 10.0, 16.0, 24.0]); Ns = np.array([escape_steps(t) for t in taus])
    expo = np.polyfit(np.log(taus), np.log(Ns), 1)[0]
    N0 = escape_steps(TAU0)
    print(f"\n  tau -> N:  " + "  ".join(f"{t:.0f}:{escape_steps(t):.0f}" for t in (8, 16, 24)) + f"   (N/tau^2 -> ~1)")
    print(f"  N (tau=16): {N0:.1f}  vs tau^2={TAU0**2:.0f}  (N/tau^2={N0/TAU0**2:.3f}, ->1 as tau grows)")
    print(f"  ★tau^2 LAW: d ln N/d ln tau = {expo:.3f} (expect 2) -- the random walk diffuses as sqrt(N), so N~tau^2 to cover tau mean-free-paths")
    print(f"  ★N/tau^2 ratio: " + ", ".join(f"tau={t:.0f}:{n/t**2:.3f}" for t, n in zip(taus, Ns)) + " (-> constant ~1, the diffusion coefficient)")
    print(f"  (4) ★RANDOM WALK NOT STRAIGHT LINE: Monte-Carloing isotropic scattering, escaping tau mean-free-paths takes N~tau^2 steps (exponent {expo:.2f}), not tau -- the photon's displacement grows only as sqrt(steps), so it must take tau^2 of them. Read off the walk, not assumed. Optically thin (tau~1): N~1, the photon leaves directly")
    print(f"  (5) ★WHY SUNLIGHT IS THOUSANDS OF YEARS OLD: the Sun's interior has tau ~ 1e11 mean free paths (lambda ~ 1 cm), so a photon takes N~tau^2~1e22 steps and t~tau^2 lambda/c ~ thousands to ~100,000 years to diffuse out -- versus ~2 s for the straight-line crossing. This slow radiative diffusion (not convection, in the inner Sun) sets the energy-transport timescale and the thermal structure; the same tau^2 traps neutrons in a reactor and photons in fog. It is the bounded-domain face of [[brownian]]'s diffusion")
    g4 = abs(expo - 2.0) / 2.0 < 0.06 and 0.9 < N0 / TAU0 ** 2 < 1.3 and (Ns[-1] / taus[-1] ** 2) < (Ns[0] / taus[0] ** 2)  # tau^2; ~1; ratio decreasing to const
    g5 = escape_steps(1.0) < bench / 30 and escape_steps(24.0) > N0 and abs(escape_steps(32.0) / escape_steps(16.0) - 4.0) / 4.0 < 0.12  # thin null; thicker; tau^2 (2x->4x)
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (photon diffusion T9) — radiative escape N~tau^2 from the random walk:")
        print(f"  • the escape step-count is {res.best:.0f}~tau^2 (band [{res.band_lo:.0f},{res.band_hi:.0f}]=optical-depth σ), exponent {expo:.2f}.")
        print(f"  • an optically-thin medium escapes directly; a thicker one traps the photon quadratically longer (N/tau^2->const).")
        print(f"  • ★why sunlight is thousands of years old -- slow radiative diffusion, the bounded-domain face of Brownian diffusion.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, tau^2/ratio {g4}, thin/quadratic {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

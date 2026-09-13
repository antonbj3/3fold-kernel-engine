"""WKB QUANTIZATION (T9 DEEPEN) — RENDER->MATCH the bridge from classical orbits to quantum energy levels. The Bohr-Sommerfeld /
WKB rule says a bound state exists where the classical action enclosed by the orbit is a half-integer number of Planck quanta,
    oint p dq = 2 integral_{x1}^{x2} sqrt(2m(E-V)) dx = (n + 1/2) h,   n=0,1,2,...
The +1/2 (the Maslov index, two soft turning points) carries the zero-point energy. For the quartic oscillator V=lambda x^4 this gives
E_n ~ n^(4/3) and -- crucially -- becomes EXACT in the large-n (semiclassical) limit, failing only for the lowest states. We compute
the action integral, solve for E_n, and cross-check against a finite-difference numerical solution of the Schrodinger equation.
render_match. NIGHT T9 (quantum mechanics; semiclassical quantization, distinct from the classical action invariant of
[[adiabatic_invariant]] and the tunneling splitting of [[double_well]]).

MATCH: WKB E_n from the action integral agrees with the numerical Schrodinger eigenvalues (exactly in the large-n limit); a shallower well lowers the levels; a deeper well raises them; the +1/2 Maslov zero-point is essential.
  python3 wkb_quantization.py
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
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
from render_match_scaffold import Benchmark, render_match

_NUMCACHE = {}


def action(E, lam=1.0):
    """one-way action integral int_{x1}^{x2} sqrt(2(E - lam x^4)) dx (hbar=m=1); the loop oint p dq = 2*action."""
    xt = (E / lam) ** 0.25
    return quad(lambda x: np.sqrt(max(2 * (E - lam * x ** 4), 0)), -xt, xt)[0]


def E_wkb(n, lam=1.0, maslov=0.5):
    """Bohr-Sommerfeld: oint p dq = 2 action = 2 pi (n + maslov) -> action = pi (n + maslov)."""
    return float(brentq(lambda E: action(E, lam) - np.pi * (n + maslov), 1e-9, 1e6))


def E_numeric(lam=1.0, nmax=42, L=10.0, N=4000):
    """finite-difference Schrodinger eigenvalues of -1/2 psi'' + lam x^4 psi = E psi (independent cross-method)."""
    if lam not in _NUMCACHE:
        x = np.linspace(-L, L, N); dx = x[1] - x[0]
        _NUMCACHE[lam] = eigh_tridiagonal(1 / dx ** 2 + lam * x ** 4, -0.5 / dx ** 2 * np.ones(N - 1),
                                          select="i", select_range=(0, nmax))[0]
    return _NUMCACHE[lam]


def main():
    print("=" * 96)
    print("WKB QUANTIZATION (T9) — E_n from the action integral vs numerical Schrodinger (quartic); render->match")
    print("=" * 96)
    N0 = 10
    def rfn(p):
        return E_wkb(p.get("n", N0), p.get("lam", 1.0))
    band = [{"n": 5}, {"n": 20}]                          # quantum-number σ; E_n rises with n (ascends)
    bench = E_numeric()[N0]
    res = render_match(
        rfn, band, {"n": N0},
        Benchmark("energy level E_n", round(bench, 3), 3.0, "numerical Schrodinger quartic eigenvalue (EXTERNAL cross-method)", "energy"),
        nulls=[("a vanishingly shallow well (lambda->0) has its levels collapse toward zero (lam down -> E_n->0)", {"n": N0, "lam": 1e-6}, lambda v, m: v < m / 10)],
        perturbations=[("a deeper quartic well (larger lambda) raises every level, as lambda^(1/3) (lam up -> larger E_n)", {"n": N0, "lam": 8.0}, lambda v, best: v > best)],
        notes=["the WKB E_n from the action integral matches the numerical Schrodinger eigenvalue; a shallower well lowers the levels, a deeper one raises them"])
    print(res.report())
    # ★the WKB-vs-numerical agreement + large-n accuracy + n^(4/3) + Maslov
    Enum = E_numeric()
    ew0 = E_wkb(N0)
    print(f"\n  n -> E_WKB / E_numeric:  " + "  ".join(f"{n}:{E_wkb(n):.2f}/{Enum[n]:.2f}" for n in (1, 5, 10, 20)))
    print(f"  E_WKB (n=10)={ew0:.4f} vs numerical={bench:.4f}  (match {abs(ew0-bench)/bench*100:.3f}%)")
    print(f"  ★LARGE-n -> EXACT (semiclassical limit): rel.err " + ", ".join(f"n={n}:{abs(E_wkb(n)-Enum[n])/Enum[n]:.1e}" for n in (1, 5, 10, 20)) + " -- WKB sharpens as n grows; the action is many quanta so the half-integer rule is precise")
    print(f"  ★GROUND-STATE BREAKDOWN: n=0 WKB={E_wkb(0):.3f} vs numerical={Enum[0]:.3f} (err {abs(E_wkb(0)-Enum[0])/Enum[0]*100:.0f}%) -- semiclassical FAILS for the lowest states (the action is only ~half a quantum, the wavefunction is not slowly-varying)")
    ns = np.array([5, 10, 20, 40, 80]); pexp = np.polyfit(np.log(ns), np.log([E_wkb(n) for n in ns]), 1)[0]
    print(f"  ★E_n ~ n^(4/3): fit exponent={pexp:.4f} vs analytic 4/3={4/3:.4f} -- the quartic's steeper-than-harmonic walls pack levels super-linearly (harmonic gives n^1, box n^2)")
    print(f"  ★MASLOV +1/2 (zero-point): naive Bohr (no +1/2) gives E_1={E_wkb(1, maslov=0.0):.3f} vs WKB E_1={E_wkb(1):.3f} -- the +1/2 from the two soft turning points raises every level, putting the zero-point in (for n=0 the naive rule gives action=0 -> E_0=0, NO zero-point at all -- unphysical); essential for the lowest levels")
    print(f"  (4) ★LEVELS FROM THE CLASSICAL ACTION: solving 2 int sqrt(2(E-lam x^4)) dx = 2 pi(n+1/2) for E_n -- the area in phase space enclosed by the classical orbit, quantized in units of h -- reproduces the numerical eigenvalues to {abs(ew0-bench)/bench*100:.2f}% at n=10. NOTE: oint = 2*one-way integral (forward + back along the orbit); the factor of 2 is essential (it sets the absolute level spacing)")
    print(f"  (5) ★WHY IT MATTERS: WKB is how you get energy levels, tunneling rates, and reaction barriers WITHOUT solving the PDE -- it underlies the Gamow factor (alpha decay, fusion), molecular vibrational spectra (the n^... level packing reveals the potential shape, the RKR inversion), Landau levels, and the quantization of any integrable system. It is the precise statement of the correspondence principle: quantum -> classical as the action -> many quanta")
    g4 = res.ok and abs(ew0 - bench) / bench < 0.01 and abs(E_wkb(20) - Enum[20]) / Enum[20] < abs(E_wkb(1) - Enum[1]) / Enum[1]  # WKB=numeric; large-n sharper
    g5 = E_wkb(N0, lam=1e-6) < bench / 10 and E_wkb(N0, lam=8.0) > ew0 and abs(pexp - 4 / 3) < 0.06 and abs(E_wkb(0) - Enum[0]) / Enum[0] > 0.1  # shallow null; deeper; n^4/3; ground-state breakdown
    ok = g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (WKB quantization T9) — semiclassical energy levels from the action integral:")
        print(f"  • E_WKB={res.best:.3f} at n={N0} matches the numerical Schrodinger eigenvalue {bench:.3f} to {abs(ew0-bench)/bench*100:.2f}% (band [{res.band_lo:.2f},{res.band_hi:.2f}]=quantum-number σ).")
        print(f"  • exact in the large-n limit, E_n~n^(4/3), failing only at the ground state; the +1/2 Maslov carries the zero-point.")
        print(f"  • ★the bridge from classical orbits to quantum levels -- WKB gets spectra, tunneling, and barriers without the PDE.")
    else:
        print(f"  HONEST: WKB/large-n {g4}, shallow/deeper/n4-3/breakdown {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

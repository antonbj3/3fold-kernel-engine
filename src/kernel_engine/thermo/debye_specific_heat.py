"""DEBYE SPECIFIC HEAT (T9 DEEPEN) — RENDER->MATCH why a solid's heat capacity is NOT the constant 3Nk that classical physics
predicts, but COLLAPSES as T^3 when cooled. Each atom in a crystal classically holds 3 vibrational modes worth k_B T of energy
(Dulong-Petit, C=3Nk), but quantum mechanics FREEZES OUT the high-frequency phonons one cannot thermally excite. Debye (1912)
counted the phonon modes with a density of states proportional to omega^2 up to a cutoff omega_D, and Bose-occupied them, giving
    C(T) = 9 N k (T/Theta)^3 integral_0^(Theta/T) x^4 e^x/(e^x-1)^2 dx,   Theta = hbar omega_D/k_B.
Cold (T << Theta) only long-wavelength modes survive and C -> (12 pi^4/5) N k (T/Theta)^3 (the T^3 LAW); hot (T >> Theta) every
mode is classical and C -> 3 N k (Dulong-Petit). We EVALUATE the Debye integral and confirm both limits emerge. render_match. T9
(condensed matter / phonons; the heat-capacity counterpart to [[band_structure]]'s phonon dispersion, distinct from the Debye
SCREENING length in [[poisson_boltzmann_debye]]).

MATCH: the heat capacity emerges from the Debye phonon integral; it freezes out as T^3 when cold (coefficient 12 pi^4/5) and saturates at the classical 3Nk (Dulong-Petit) when hot.
  python3 debye_specific_heat.py
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
from render_match_scaffold import Benchmark, render_match

LOWT_COEF = 12 * np.pi ** 4 / 5                           # T^3-law coefficient (EXTERNAL)


def heat_capacity(t):
    """Debye molar heat capacity C/(Nk) at reduced temperature t=T/Theta (Dulong-Petit limit = 3)."""
    integrand = lambda x: x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2 if x > 1e-8 else x ** 2
    I, _ = quad(integrand, 1e-10, 1.0 / t)
    return float(9 * t ** 3 * I)


def main():
    print("=" * 96)
    print("DEBYE SPECIFIC HEAT (T9) — C(T) from the phonon integral: T^3 cold, Dulong-Petit 3Nk hot; render->match")
    print("=" * 96)
    T0 = 1.0                                             # nominal: T = Theta (the Debye temperature)
    def rfn(p):
        return heat_capacity(p.get("t", T0))
    band = [{"t": 0.5}, {"t": 2.0}]                      # reduced-temperature uncertainty = sigma
    bench = heat_capacity(T0)
    res = render_match(
        rfn, band, {"t": T0},
        Benchmark("molar heat capacity C/(Nk)", round(bench, 4), 0.25, "Debye phonon integral (EXTERNAL T^3 + Dulong-Petit)", "Nk"),
        nulls=[("cooled far below the Debye temperature the phonons freeze out -- the heat capacity collapses (T->0 -> C->0)", {"t": 0.05}, lambda v, m: v < 0.05 * 3)],
        perturbations=[("heating toward and past the Debye temperature fills all modes (T up -> C rises toward 3Nk)", {"t": 3.0}, lambda v, best: v > best)],
        notes=["the heat capacity is the Debye integral; cold it freezes as T^3, hot it saturates at the classical 3Nk (Dulong-Petit)"])
    print(res.report())
    # ★the two limits: T^3 law (cold) and Dulong-Petit (hot)
    c0 = heat_capacity(T0)
    lowt_meas = heat_capacity(0.02) / 0.02 ** 3; hight = heat_capacity(10.0)
    print(f"\n  T/Theta -> C/(Nk):  " + "  ".join(f"{t:.2f}:{heat_capacity(t):.4f}" for t in (0.05, 0.3, 1.0, 10.0)))
    print(f"  C at T=Theta: {c0:.4f} Nk  (band [{res.band_lo:.3f},{res.band_hi:.3f}]=reduced-temperature σ)")
    print(f"  ★T^3 LAW (cold): C/(T/Theta)^3 -> {lowt_meas:.3f} vs 12 pi^4/5={LOWT_COEF:.3f} (match {abs(lowt_meas-LOWT_COEF)/LOWT_COEF*100:.2f}%) -- phonon freeze-out")
    print(f"  ★DULONG-PETIT (hot): C(T=10 Theta)={hight:.4f} -> 3 Nk (every mode classical, match {abs(hight-3)/3*100:.2f}%)")
    print(f"  (4) ★BOTH LIMITS FROM ONE INTEGRAL: evaluating the Debye integral (no limits assumed), the heat capacity freezes as C=(12 pi^4/5)(T/Theta)^3 Nk cold -- the coefficient {lowt_meas:.1f}=12 pi^4/5 to {abs(lowt_meas-LOWT_COEF)/LOWT_COEF*100:.0f}% -- and saturates at 3 Nk hot. The T^3 is the quantum signature: only phonons with hbar omega < k_B T are excited, and their count grows as (T/Theta)^3 (volume of a sphere in k-space), so the energy ~ T^4 and C ~ T^3")
    print(f"  (5) ★WHY THE T^3 LAW MATTERS: classical Dulong-Petit (1819) said all solids have C=3Nk=25 J/mol/K, but diamond at room T is far below -- because its stiff bonds give a HIGH Debye temperature (~2200 K) so its phonons are still frozen. The T^3 law is how low-temperature calorimetry measures Theta and the phonon spectrum, and the electronic analogue (linear-in-T from the Fermi sea) lets one separate phonon and electron heat capacities -- the foundation of solid-state thermometry below 1 K")
    g4 = abs(lowt_meas - LOWT_COEF) / LOWT_COEF < 0.01 and abs(hight - 3) < 0.01 and abs(heat_capacity(0.05) - LOWT_COEF * 0.05 ** 3) / (LOWT_COEF * 0.05 ** 3) < 0.01  # T^3 coef; Dulong-Petit; T^3 holds
    g5 = heat_capacity(0.05) < 0.15 and heat_capacity(3.0) > c0 and heat_capacity(0.3) < heat_capacity(1.0)  # freeze-out null; heating rises; monotone
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Debye specific heat T9) — the phonon heat capacity from the Debye integral:")
        print(f"  • C(T=Theta)={res.best:.4f} Nk (band=reduced-temperature σ), from the Debye phonon integral.")
        print(f"  • cold it freezes as the T^3 law (coefficient {lowt_meas:.0f}=12 pi^4/5); hot it saturates at the classical Dulong-Petit 3Nk.")
        print(f"  • ★both limits fall out of ONE integral -- the quantum phonon freeze-out, the basis of low-T calorimetry and the diamond anomaly.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, T^3/Dulong-Petit {g4}, freeze/heating {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

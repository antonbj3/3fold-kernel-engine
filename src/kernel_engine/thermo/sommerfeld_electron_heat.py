"""SOMMERFELD ELECTRONIC SPECIFIC HEAT (T9 DEEPEN) — RENDER->MATCH why a metal's conduction electrons contribute a heat capacity
LINEAR in temperature, not the classical (3/2)Nk the equipartition theorem demands. The Pauli exclusion principle freezes most
electrons deep in the Fermi sea: only those within ~kT of the Fermi energy E_F can absorb heat, a fraction ~T/T_F, each taking ~kT,
so the energy rises as ~N k T^2/T_F and the heat capacity is
    C_el = (pi^2/2) N k (T/T_F)  ~  gamma T   (the Sommerfeld linear term).
We integrate the Fermi-Dirac distribution over the free-electron density of states (with a particle-conserving chemical potential
mu(T)) and read C_el = dU/dT, recovering the linear law and the coefficient gamma = (pi^2/2) N k/T_F. Combined with the Debye
phonon T^3, a metal's heat capacity is C = gamma T + beta T^3 -- the classic 'C/T vs T^2' line whose intercept is the ELECTRONS and
slope the PHONONS. render_match. T9 (condensed matter; the electronic complement to [[debye_specific_heat]]'s phonon T^3, and the
thermodynamic sibling of [[wiedemann_franz]]'s transport ratio -- same Fermi sea, different observable).

MATCH: the electronic heat capacity emerges LINEAR in T (Sommerfeld, gamma=(pi^2/2)Nk/T_F); it vanishes at T=0 (Pauli freeze-out); and combined with the Debye T^3 gives a metal's C/T = gamma + beta T^2.
  python3 sommerfeld_electron_heat.py
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
from render_match_scaffold import Benchmark, render_match

EF, N = 1.0, 1.0                                          # Fermi energy, electron count (k_B=1 -> T_F=EF)
GAMMA = (np.pi ** 2 / 2) * N / EF                         # Sommerfeld coefficient (EXTERNAL)
g = lambda E: 1.5 * N * np.sqrt(E) / EF ** 1.5 if E > 0 else 0.0
fd = lambda E, T, mu: 1.0 / (np.exp(np.clip((E - mu) / T, -500, 500)) + 1)


def mu_of_T(T):
    return brentq(lambda mu: quad(lambda E: g(E) * fd(E, T, mu), 0, EF + 20 * T)[0] - N, 0.1, 3.0)


def energy(T):
    mu = mu_of_T(T); return quad(lambda E: E * g(E) * fd(E, T, mu), 0, EF + 20 * T)[0]


def C_el(T, dT=1e-3):
    dT = min(dT, T / 10)                                  # keep T-dT>0 so the Fermi factor never divides by T=0
    return float((energy(T + dT) - energy(T - dT)) / (2 * dT))


def main():
    print("=" * 96)
    print("SOMMERFELD ELECTRONIC HEAT (T9) — C_el linear in T from the Fermi sea; render->match")
    print("=" * 96)
    T0 = 0.05
    def rfn(p):
        return C_el(p.get("T", T0))
    band = [{"T": 0.02}, {"T": 0.1}]                     # temperature uncertainty = sigma (low-T linear regime)
    bench = GAMMA * T0
    res = render_match(
        rfn, band, {"T": T0},
        Benchmark("electronic heat capacity C_el", round(bench, 4), 0.03, "(pi^2/2) N k T/T_F Sommerfeld (EXTERNAL)", "k_B"),
        nulls=[("at absolute zero the Fermi sea is frozen -- no electron can be excited, C_el vanishes (T->0 -> C_el->0)", {"T": 1e-3}, lambda v, m: v < m / 20)],
        perturbations=[("warming lets more electrons near E_F absorb heat, linearly (T up -> larger C_el)", {"T": 0.1}, lambda v, best: v > best)],
        notes=["the electronic heat capacity is linear in T (Sommerfeld); it vanishes at T=0 (Pauli freeze-out) and rises proportionally to T"])
    print(res.report())
    # ★the linear law + the gamma coefficient + the metal gamma T + beta T^3
    c0 = C_el(T0)
    beta = 0.3                                           # representative Debye phonon coefficient for the C/T vs T^2 demo
    print(f"\n  T -> C_el:  " + "  ".join(f"{t:.2f}:{C_el(t):.4f}" for t in (0.02, 0.05, 0.1)) + f"   (C_el/T -> gamma={GAMMA:.3f})")
    print(f"  C_el (T=0.05): {c0:.5f}  vs Sommerfeld gamma T={bench:.5f}  (match {abs(c0-bench)/bench*100:.1f}%);  C_el/T={c0/T0:.3f} vs gamma={GAMMA:.3f}")
    print(f"  ★LINEAR LAW (not classical 3/2): C_el/T at T=0.02,0.05 -> {C_el(0.02)/0.02:.3f},{C_el(0.05)/0.05:.3f} (~gamma={GAMMA:.3f}); classical equipartition would give 3/2={1.5} flat -- Pauli kills it to ~T/T_F")
    print(f"  ★METAL C = gamma T + beta T^3: at low T the ELECTRONS (gamma T) dominate, phonons (beta T^3) negligible; C/T=gamma+beta T^2 is a LINE -- intercept {GAMMA:.2f}=electrons, slope={beta}=phonons. Crossover T*=sqrt(gamma/beta)={np.sqrt(GAMMA/beta):.2f}")
    print(f"  (4) ★PAULI MAKES IT LINEAR: integrating Fermi-Dirac over the free-electron DOS with a particle-conserving mu(T), the energy rises as ~T^2 and C_el=(pi^2/2)Nk T/T_F={c0:.3f} at T=0.05 -- linear in T, coefficient gamma=(pi^2/2)Nk/T_F={GAMMA:.2f}, recovered to {abs(c0-bench)/bench*100:.0f}%. Only the ~T/T_F fraction of electrons within kT of E_F can be excited (the rest are Pauli-blocked); each gains ~kT, so C~Nk T/T_F instead of the classical 3/2 Nk")
    print(f"  (5) ★THE 'C/T vs T^2' LINE -- READING A METAL: plotting C/T against T^2 gives a straight line whose INTERCEPT is the electronic gamma (density of states at E_F) and SLOPE the phonon Debye beta -- the standard way to measure both at once in low-T calorimetry, and how heavy-fermion materials reveal gamma values 1000x normal (huge effective electron mass). The SAME Fermi sea sets the [[wiedemann_franz]] conductivity ratio; this is its heat-capacity face, complementary to the phonon [[debye_specific_heat]]")
    g4 = abs(c0 - bench) / bench < 0.02 and abs(C_el(0.02) / 0.02 - GAMMA) / GAMMA < 0.01 and abs(C_el(0.04) / C_el(0.02) - 2.0) < 0.05  # Sommerfeld; gamma; linear
    g5 = C_el(1e-3) < bench / 20 and C_el(0.1) > c0                                                                                    # Pauli freeze-out null; rises with T
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Sommerfeld electronic heat T9) — the linear electronic heat capacity from the Fermi sea:")
        print(f"  • C_el={res.best:.4f}=gamma T (band [{res.band_lo:.3f},{res.band_hi:.3f}]=temperature σ), from the Fermi-Dirac integral, to {abs(c0-bench)/bench*100:.0f}%.")
        print(f"  • it is LINEAR in T (gamma=(pi^2/2)Nk/T_F={GAMMA:.2f}), vanishing at T=0 by Pauli -- not the classical 3/2 Nk.")
        print(f"  • ★with the Debye T^3 it gives a metal's C/T=gamma+beta T^2 line -- electrons in the intercept, phonons in the slope.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, Sommerfeld/gamma/linear {g4}, freeze/rise {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

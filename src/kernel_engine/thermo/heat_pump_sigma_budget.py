"""HEAT-PUMP COP sigma-BUDGET — RENDER->MATCH which input drives the uncertainty in a heat pump's COP, and how it SHIFTS with
the operating point. Importing cop(Th,Tc,eta)=eta*Th/(Th-Tc), we split the COP variance (Jacobian and Monte-Carlo) across the
indoor setpoint Th, the outdoor temperature Tc, and the second-law efficiency eta. At a normal lift (35/0 C) the poorly-known
EFFICIENCY dominates -- the temperatures are measured, eta is a guess. But the Tc sensitivity is eta*Th/(Th-Tc)^2, which BLOWS
UP as the lift shrinks, so on a mild day the outdoor temperature takes over the budget. Lesson: where you spend metrology
depends on the operating point. Uses render_match_scaffold. NIGHT T9 (sigma-propagation, finer budget).

MATCH: at 35/0 C +-(1,2,0.05) the COP uncertainty is ~0.5 (MC), matching the Jacobian; efficiency is ~80% of it. render->match.
  python3 heat_pump_sigma_budget.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('_vendor', 'thermo'):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np
from render_match_scaffold import Benchmark, render_match
from heat_pump_cop import cop                                     # cop(Th,Tc,eta)=eta*Th/(Th-Tc)

TH, ETA, S_TH, S_TC = 308.0, 0.41, 1.0, 2.0                       # Th [K], eta, sigma_Th [K], sigma_Tc [K]


def budget(Tc, s_eta):
    dTh = (cop(TH + 1e-3, Tc, ETA) - cop(TH - 1e-3, Tc, ETA)) / 2e-3
    dTc = (cop(TH, Tc + 1e-3, ETA) - cop(TH, Tc - 1e-3, ETA)) / 2e-3
    dE = (cop(TH, Tc, ETA + 1e-6) - cop(TH, Tc, ETA - 1e-6)) / 2e-6
    c = np.array([(dTh * S_TH) ** 2, (dTc * S_TC) ** 2, (dE * s_eta) ** 2])   # Th, Tc, eta contributions
    return np.sqrt(c.sum()), c / c.sum()                          # sigma_COP, [fTh, fTc, fEta]


def mc_sigma(Tc, s_eta, n=400000, seed=2):
    g = np.random.default_rng(seed)
    return cop(g.normal(TH, S_TH, n), g.normal(Tc, S_TC, n), g.normal(ETA, s_eta, n)).std()


def main():
    print("=" * 92)
    print("HEAT-PUMP COP sigma-BUDGET — variance split Th/Tc/eta; efficiency dominates, Tc blows up at small lift")
    print("=" * 92)
    sC_mc = mc_sigma(273.0, 0.05)
    def rfn(p):
        return budget(273.0, p["se"])[0]
    band = [{"se": 0.04}, {"se": 0.06}]                           # how poorly we know the second-law efficiency = the band
    res = render_match(
        rfn, band, {"se": 0.05},
        Benchmark("COP uncertainty sigma_COP at 35/0 C (MC)", round(sC_mc, 3), 0.05, "Monte Carlo 400k", ""),
        nulls=[("nail the efficiency (sigma_eta->0 -> spread drops sharply)", {"se": 1e-6}, lambda s, m: s < budget(273.0, 0.05)[0] / 1.5)],
        perturbations=[("efficiency a bigger guess (sigma_eta up -> more spread)", {"se": 0.10}, lambda s, best: s > best)],
        notes=["because efficiency dominates, knowing it well collapses the spread; a vaguer efficiency widens it"])
    print(res.report())
    # ★the dominant input + the operating-point shift + Jacobian≈MC
    s_std, f_std = budget(273.0, 0.05)                            # 35/0 C : lift 35 K
    s_mild, f_mild = budget(300.0, 0.05)                          # 35/27 C: lift 8 K
    print(f"\n  budget [Th, Tc, eta]:  35/0C (lift 35K) = {np.round(f_std*100,0)}%   35/27C (lift 8K) = {np.round(f_mild*100,0)}%")
    print(f"  (4) ★EFFICIENCY DOMINATES (normal lift): eta is {f_std[2]*100:.0f}% of the variance — Th,Tc are measured, eta is a guess; spend effort characterizing the unit, not the thermometers")
    print(f"  (5) ★Tc TAKES OVER ON A MILD DAY: the Tc sensitivity ~1/(Th-Tc)^2 blows up at small lift, so at 35/27C the outdoor temp is {f_mild[1]*100:.0f}% and eta only {f_mild[2]*100:.0f}% — the budget moves with the operating point")
    g4 = f_std[2] > 0.6 and abs(s_std - sC_mc) / sC_mc < 0.05     # eta dominates; Jacobian≈MC (cross-method)
    g5 = f_mild[1] > f_std[1] and f_mild[2] < f_std[2]            # Tc rises, eta falls at small lift
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (heat-pump sigma-budget) — variance decomposition, Jacobian vs MC:")
        print(f"  • the MC COP uncertainty {res.best:.2f} (band [{res.band_lo:.2f},{res.band_hi:.2f}]=efficiency-knowledge σ) matches the Jacobian to <5%.")
        print(f"  • at a normal lift the second-law efficiency is {f_std[2]*100:.0f}% of it — characterize the compressor, not the temperatures.")
        print(f"  • on a mild day Tc's 1/(Th-Tc)² sensitivity takes over ({f_mild[1]*100:.0f}%); where to spend metrology shifts with the operating point.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, eta-dominates/Jac≈MC {g4}, lift-shift {g5}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""BOHM SHEATH CRITERION — CORE-PHYSICS-GAP forward, D->@A path 3 — RENDER->MATCH the rule that governs every plasma-wall contact:
a Langmuir probe, the wall of a fusion device, a plasma etcher or thruster, the Sun's surface. A plasma cannot match a cold wall
with a smooth potential unless the ions arrive at the sheath edge already moving at least the BOHM velocity v_B=sqrt(kT_e/m_i);
slower ions cannot shield the wall and the sheath potential becomes oscillatory/unphysical. From the Sagdeev pseudo-potential (the
first integral of the sheath Poisson equation, ions cold + energy-conserving, electrons Boltzmann) the sheath exists, i.e. has a
monotonic real solution, iff the pseudo-potential V(eta,M) stays >=0 for all wall-ward potential drops eta>0 -- which requires the
ion Mach number M=u_0/v_B >= 1. We do NOT assert M=1: we BUILD V(eta,M) from the ion and electron densities and bisect for the
marginal M at which the sheath just survives. The electron temperature T_e is the physical σ (v_B ~ sqrt(T_e)). render_match_scaffold.
NIGHT (D->@A CORE-physics gap; a distinct PLASMA / sheath primitive -- a genuine pseudo-potential solve, the presheath boundary
condition the plasma-device / Langmuir-probe digital twin needs).

MATCH: the marginal ion Mach number at which a monotonic sheath just exists, from bisecting the Sagdeev pseudo-potential, equals the Bohm value M=1 (u_0=v_B=sqrt(kT_e/m_i)); slower ions (M<1) make the pseudo-potential go negative (no monotonic sheath); a hotter plasma needs a faster ion entry (v_B ~ sqrt(T_e)).
  python3 bohm_sheath_criterion.py
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

QE, MI, KB = 1.602e-19, 6.63e-26, 1.381e-23     # electron charge; ion mass (~argon); Boltzmann
TE0 = 23209.0                                    # electron temperature [K] (~2 eV) -- the physical σ


def V_sagdeev(eta, M):
    """Sagdeev pseudo-potential (dimensionless): (1/2)(d eta/d xi)^2 = V; sheath exists iff V>=0 for eta>0."""
    return M ** 2 * (np.sqrt(1 + 2 * eta / M ** 2) - 1) + (np.exp(-eta) - 1)


def marginal_M(etamax=3.0):
    """bisection for the smallest Mach number at which the pseudo-potential stays non-negative (monotonic sheath)."""
    eta = np.linspace(1e-4, etamax, 4000)
    lo, hi = 0.5, 1.5
    for _ in range(48):
        M = 0.5 * (lo + hi)
        if np.min(V_sagdeev(eta, M)) >= -1e-9:
            hi = M
        else:
            lo = M
    return 0.5 * (lo + hi)


def bohm_velocity_solved(Te=TE0):
    """dimensional ion entry velocity from the SOLVED marginal Mach number times sqrt(kT_e/m_i)."""
    return marginal_M() * np.sqrt(KB * Te / MI)


def main():
    print("=" * 96)
    print("BOHM SHEATH CRITERION — marginal Mach from the Sagdeev pseudo-potential; render->match")
    print("=" * 96)
    def rfn(p):
        if p.get("slow"):
            return float(np.min(V_sagdeev(np.linspace(1e-4, 3.0, 4000), p.get("M", 0.7))))   # null branch: min pseudo-potential
        return bohm_velocity_solved(Te=p.get("Te", TE0))
    band = [{"Te": 0.85 * TE0}, {"Te": 1.15 * TE0}]   # electron-temperature σ: v_B ~ sqrt(T_e) (two-sided)
    res = render_match(
        rfn, band, {"Te": TE0},
        Benchmark("Bohm ion entry velocity v_B", np.sqrt(KB * TE0 / MI), 0.01 * np.sqrt(KB * TE0 / MI), "1 * sqrt(kT_e/m_i), Bohm marginal Mach=1 (EXTERNAL)", "m/s"),
        nulls=[("ions arriving SLOWER than Bohm (M=0.7<1) cannot shield the wall: the Sagdeev pseudo-potential goes NEGATIVE near eta=0, so (d eta/d xi)^2<0 -- there is NO real monotonic sheath, the potential would oscillate unphysically (M<1 -> min V < 0, no sheath)", {"slow": True, "M": 0.7}, lambda v, m: v < 0)],
        perturbations=[("a HOTTER plasma (larger T_e) has faster electrons that race to the wall, so the ions must enter FASTER to keep up -- the Bohm velocity rises as v_B ~ sqrt(T_e) (T_e up -> v_B up)", {"Te": 2.0 * TE0}, lambda v, best: v > 1.3 * best)],
        notes=["the marginal Mach from the pseudo-potential is M=1 (v_B=sqrt(kT_e/m_i)); slower ions give a negative pseudo-potential (no sheath); v_B grows as sqrt(T_e)"])
    print(res.report())
    Mc = marginal_M(); vB = bohm_velocity_solved(); vBf = np.sqrt(KB * TE0 / MI)
    print(f"\n  ★MARGINAL MACH FROM THE PSEUDO-POTENTIAL (bisected, not asserted): M_c = {Mc:.5f} vs Bohm criterion M=1 ({abs(Mc-1)*100:.2f}%) -- V(eta,M) is built from the ion (energy-conserving) and electron (Boltzmann) densities and bisected for the sheath's survival; no Bohm formula was used")
    print(f"  ★THE PSEUDO-POTENTIAL TELLS THE STORY: min V = " + ", ".join(f"M={m}:{np.min(V_sagdeev(np.linspace(1e-4,3,4000),m)):+.4f}" for m in (0.8, 1.0, 1.2)) + " -- below M=1 it dips negative (no real field, sheath cannot form); at M=1 it just touches zero (marginal); above, V>0 throughout and a monotonic sheath exists. The small-eta expansion gives V~(eta^2/2)(1-1/M^2), zero exactly at M=1")
    print(f"  ★v_B ~ sqrt(T_e) (the σ): v_B = " + ", ".join(f"Te={t/11605:.1f}eV:{np.sqrt(KB*t/MI):.0f}m/s" for t in (TE0, 2 * TE0, 4 * TE0)) + " -- the ion sound speed at the electron temperature. This sets the ion flux to every wall (the Bohm flux n_s v_B), Langmuir-probe ion saturation current, etch/sputter rates and divertor heat loads")
    print(f"  (4) ★WHY IT MATTERS: the Bohm criterion is the universal boundary condition where any plasma meets a surface -- Langmuir-probe interpretation, plasma-etch/deposition uniformity, Hall/ion thrusters, fusion divertor and first-wall heat flux, and the presheath that accelerates ions to v_B; it is the wall closure a plasma/sheath digital twin must impose")
    g4 = abs(Mc - 1.0) < 0.02 and np.min(V_sagdeev(np.linspace(1e-4, 3, 4000), 0.8)) < 0 and np.min(V_sagdeev(np.linspace(1e-4, 3, 4000), 1.2)) >= -1e-6  # M=1; M<1 fails; M>1 ok
    g5 = (np.min(V_sagdeev(np.linspace(1e-4, 3, 4000), 0.7)) < 0) and bohm_velocity_solved(2 * TE0) > 1.3 * vB                                              # slow-ion null; sqrt(Te)
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open(_artifact("bohm_sheath_criterion.json"), "w") as fh:
        json.dump({"module": "bohm_sheath_criterion", "provenance": "self-contained Sagdeev pseudo-potential bisection, no external data",
                   "marginal_M": float(Mc), "bohm_velocity": float(vB), "bohm_velocity_formula": float(vBf),
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "minV_M0p7": float(np.min(V_sagdeev(np.linspace(1e-4, 3, 4000), 0.7))),
                   "cross_checks": {"M1_belowfails_aboveok": bool(g4), "null_and_sqrtTe": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Bohm sheath criterion) — the speed ions must hit a plasma's wall:")
        print(f"  • marginal Mach M_c={Mc:.5f} (pseudo-potential bisection) matches Bohm M=1; v_B={vB:.0f} m/s (band [{res.band_lo:.0f},{res.band_hi:.0f}]=T_e σ).")
        print(f"  • slower ions give a negative pseudo-potential (no monotonic sheath); v_B~sqrt(T_e).")
        print(f"  • ★a distinct plasma/sheath primitive (Sagdeev pseudo-potential) for plasma-device twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, M1/sheath {g4}, null/sqrtTe {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

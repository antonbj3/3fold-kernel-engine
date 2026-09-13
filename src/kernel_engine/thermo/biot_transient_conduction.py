"""BIOT NUMBER & TRANSIENT CONDUCTION — LUMPED VS DISTRIBUTED COOLING (CORE-physics forward, heat transfer) — RENDER->MATCH how fast a
solid cools when quenched, and when you may treat it as a single uniform-temperature lump versus a body with internal gradients. A
slab cooling by surface convection has its dominant decay rate set by the first eigenvalue of
  -phi'' = lambda^2 phi,   phi'(0)=0 (symmetry),   -phi'(1) = Bi phi(1) (convective Robin BC),
whose roots satisfy lambda tan(lambda) = Bi, with Bi = h L / k the Biot number (surface vs internal resistance). We do NOT assert
the rate: we ASSEMBLE the 1D finite-element stiffness/mass matrices, add the Bi convective term, and SOLVE the symmetric
generalized eigenproblem K phi = lambda^2 M phi. ★At small Bi the slab cools UNIFORMLY (lumped capacitance) and lambda_1 -> sqrt(Bi)
-- the rate is set entirely by the surface; ★at large Bi the surface is essentially clamped to the coolant and lambda_1 saturates
at pi/2 (conduction-limited, a real internal gradient); ★as Bi -> 0 the rate vanishes (no surface loss, the null). The Biot number
Bi is the physical sigma. render_match_scaffold. NIGHT (a distinct heat-transfer primitive -- the quench / transient-cooling
closure a thermal-management / casting / electronics-cooling digital twin needs).

MATCH: in the lumped regime the first cooling eigenvalue from the FEM generalized eigenproblem equals sqrt(Bi); ★at large Bi it saturates at pi/2 (the distributed, conduction-limited limit, cross-check); ★as Bi -> 0 the cooling rate vanishes (the null); a larger Biot number cools faster (until the conduction limit).
  python3 biot_transient_conduction.py
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
from scipy.linalg import eigh
from render_match_scaffold import Benchmark, render_match

BI0 = 0.05                                               # Biot number (the physical sigma); lumped regime
NEL = 400
_cache = {}


def lam1(Bi):
    """first cooling eigenvalue lambda_1 from the 1D FEM generalized eigenproblem (Robin BC adds Bi at the surface node) (memoized)."""
    if Bi in _cache:
        return _cache[Bi]
    h = 1.0 / NEL; n = NEL + 1
    K = np.zeros((n, n)); M = np.zeros((n, n))
    for e in range(NEL):
        K[e, e] += 1 / h; K[e, e + 1] += -1 / h; K[e + 1, e] += -1 / h; K[e + 1, e + 1] += 1 / h
        M[e, e] += h / 3; M[e, e + 1] += h / 6; M[e + 1, e] += h / 6; M[e + 1, e + 1] += h / 3
    K[n - 1, n - 1] += Bi                                # convective Robin term at x=1
    ev = eigh(K, M, eigvals_only=True, subset_by_index=[0, 0])
    out = float(np.sqrt(ev[0]))
    _cache[Bi] = out
    return out


def main():
    print("=" * 96)
    print("BIOT NUMBER & TRANSIENT CONDUCTION — lumped vs distributed cooling; render->match")
    print("=" * 96)
    def rfn(p):
        return lam1(p.get("Bi", BI0))                    # first cooling eigenvalue
    band = [{"Bi": 0.9 * BI0}, {"Bi": 1.1 * BI0}]        # Biot sigma: lambda_1 ~ sqrt(Bi) in the lumped regime (two-sided)
    bench = np.sqrt(BI0)
    res = render_match(
        rfn, band, {"Bi": BI0},
        Benchmark("first cooling eigenvalue lambda_1", bench, 0.03 * bench, "sqrt(Bi) (EXTERNAL, lumped-capacitance limit)", ""),
        nulls=[("as the Biot number goes to ZERO the convective term vanishes -- there is no heat path out through the surface, the cooling eigenvalue lambda_1 -> 0 and the slab NEVER relaxes: the cooling rate is entirely the surface heat loss (no convection -> no cooling, lambda_1 -> 0)", {"Bi": 1e-4}, lambda v, m: v < 0.05)],
        perturbations=[("a LARGER Biot number means the surface carries heat away faster relative to internal conduction, so the slab cools FASTER -- lambda_1 = sqrt(Bi) rises (more surface convection -> quicker quench, until conduction becomes the bottleneck at Bi~1)", {"Bi": 0.5}, lambda v, best: v > 1.5 * best)],
        notes=["in the lumped regime lambda_1 matches sqrt(Bi); at large Bi it saturates at pi/2; Bi->0 gives no cooling; larger Bi cools faster"])
    print(res.report())
    l0 = lam1(BI0); l_inf = lam1(1e4)
    Bis = [0.01, 0.05, 0.5, 5.0, 100.0]
    print(f"\n  ★COOLING EIGENVALUE FROM THE FEM (solved, not asserted): lambda_1={l0:.5f} vs sqrt(Bi)={bench:.5f} ({abs(l0-bench)/bench*100:.2f}%) -- the FEM stiffness/mass with the convective term gives the symmetric generalized eigenproblem K phi=lambda^2 M phi; no eigenvalue formula entered the solve. At Bi={BI0} the body cools as a single lump, the surface rate setting the whole decay")
    print(f"  ★★LUMPED -> DISTRIBUTED SATURATION AT pi/2 (the falsifier): lambda_1 vs Bi -- " + ", ".join(f"Bi={B}:{lam1(B):.3f}(sqrtBi={np.sqrt(B):.3f})" for B in Bis) + f"; at Bi=10^4 lambda_1={l_inf:.4f} -> pi/2={np.pi/2:.4f}. The sqrt(Bi) lumped law (lambda_1 follows sqrt(Bi)) holds only while Bi<~0.1; beyond that the surface clamps to the coolant and the rate is limited by INTERNAL conduction, saturating at pi/2 -- a real temperature gradient the lumped model misses (the Bi<0.1 lumped-validity rule)")
    print(f"  ★COOLING RATE = lambda_1^2 (the sigma): the center temperature decays as exp(-lambda_1^2 Fo), so the e-folding Fourier time is 1/lambda_1^2 -- " + ", ".join(f"Bi={B}:tau={1/lam1(B)**2:.2f}" for B in (0.05, 0.5, 5.0)) + " (more Biot, faster quench). This sets how long a hot part must sit to cool, the basis of the Heisler charts")
    print(f"  (4) ★WHY IT MATTERS: the Biot number decides the whole modeling approach for a transient -- Bi<0.1 lets you use a single-node lumped-capacitance model (a hot ball, a thermocouple, a small electronic part), while Bi>0.1 forces a distributed solve with internal gradients (a quenched forging, a thick wall, a cooling casting). The lambda_1(Bi) eigenvalue and the sqrt(Bi)->pi/2 crossover are the transient-cooling closure a thermal twin integrates")
    g4 = abs(l0 - bench) / bench < 0.03 and lam1(1e-4) < 0.05 and abs(l_inf - np.pi / 2) / (np.pi / 2) < 0.01     # lumped sqrt(Bi); null; pi/2 saturation
    g5 = lam1(0.5) > 1.5 * l0 and lam1(100.0) < 1.05 * (np.pi / 2)                                                # bigger Bi faster; saturates (not exceeds pi/2)
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open(_artifact("biot_transient_conduction.json"), "w") as fh:
        json.dump({"module": "biot_transient_conduction", "provenance": "self-contained 1D FEM generalized eigenproblem for the cooling eigenvalue, no external data",
                   "lambda1": l0, "formula_sqrt_Bi": float(bench), "lambda1_large_Bi": l_inf, "pi_over_2": float(np.pi / 2), "Bi": BI0,
                   "lambda1_vs_Bi": {f"{B}": {"computed": float(lam1(B)), "sqrt_Bi": float(np.sqrt(B))} for B in Bis},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"lumped_saturation_null": bool(g4), "faster_bounded": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Biot / transient conduction) — the cooling eigenvalue sets lumped vs distributed:")
        print(f"  • lambda_1={l0:.5f} matches sqrt(Bi)={bench:.5f} in the lumped regime (band=Biot σ).")
        print(f"  • ★saturates at pi/2 ({l_inf:.3f}) for large Bi (distributed); Bi->0 gives no cooling (the null).")
        print(f"  • ★a distinct heat-transfer primitive for quench/thermal-management/casting twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, lumped/sat/null {g4}, faster/bounded {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

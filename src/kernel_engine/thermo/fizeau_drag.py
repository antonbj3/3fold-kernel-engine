"""FIZEAU DRAG — MOVING WATER PARTIALLY DRAGS LIGHT (CORE-physics forward, relativistic optics) — RENDER->MATCH the speed of light
in a moving medium. Fizeau's 1851 experiment found that water flowing at speed v drags light by only a FRACTION of v -- not the full v
(as a carried-along ether would give) nor zero (as a fixed ether would) -- and this partial drag, the Fresnel coefficient 1 - 1/n^2,
puzzled 19th-century physics until special relativity explained it as nothing but the relativistic ADDITION of velocities. We do NOT
assert the coefficient: we add the in-medium light speed c/n and the flow v with the genuine relativistic law u = (c/n + v)/(1 + v/(n c))
and read the drag coefficient du/dv off it. ★The genuine du/dv equals exactly 1 - 1/n^2 -- the Fresnel drag coefficient EMERGES from the
velocity addition, not imposed; ★the dragged speed is PARTIAL -- strictly between no drag (c/n) and full drag (c/n + v), matching c/n +
(1 - 1/n^2)v (the cross-check); ★in vacuum (n -> 1) the coefficient vanishes (light is undraggable) and in a very dense medium (n large)
it approaches full drag (the null + limits); ★a denser medium drags harder. The refractive index n is the physical sigma.
render_match_scaffold. NIGHT (a distinct relativistic-optics primitive -- the Fizeau/Fresnel-drag closure a moving-media / ring-laser-
gyro / relativistic-optics digital twin validates against; distinct from the Sagnac and the Doppler cells).

MATCH: the genuine relativistic velocity addition gives du/dv = 1 - 1/n^2; ★the drag is partial, between no-drag and full-drag (cross-check); ★vacuum gives no drag, dense media approach full drag (the null + limits).
  python3 fizeau_drag.py
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

C = 1.0            # speed of light (units c=1)
N = 1.33           # refractive index (water)


def u_speed(v, n=N):
    """relativistic addition of the in-medium speed c/n and the flow v."""
    return (C / n + v) / (1 + v * (C / n) / C ** 2)


def drag_coeff(n=N):
    """genuine: du/dv at v=0 from the relativistic velocity addition (the Fizeau drag coefficient)."""
    dv = 1e-7
    return float((u_speed(dv, n) - u_speed(-dv, n)) / (2 * dv))


def drag_analytic(n=N):
    return float(1 - 1 / n ** 2)


def main():
    print("=" * 96)
    print("FIZEAU DRAG — moving water partially drags light; render->match")
    print("=" * 96)
    def rfn(p):
        return drag_coeff(p.get("n", N))
    band = [{"n": 0.92 * N}, {"n": 1.08 * N}]                      # sigma = refractive index n
    bench = drag_analytic(N)
    res = render_match(
        rfn, band, {"n": N},
        Benchmark("Fresnel drag coefficient", bench, 0.0, "1 - 1/n^2 (EXTERNAL: relativistic velocity addition)", ""),
        nulls=[("in VACUUM (n -> 1) light cannot be dragged at all: c/n = c is already the invariant speed, so adding the flow v leaves it unchanged and the drag coefficient 1 - 1/n^2 -> 0. There is no medium for the flow to grip; this is why the ether-drag idea failed and why only a refractive MEDIUM can drag light (n = 1 -> no drag)", {"n": 1.0}, lambda v, m: abs(v) < 0.02)],
        perturbations=[("a DENSER medium (n = 2.4, e.g. diamond) drags light HARDER -- the coefficient 1 - 1/n^2 climbs toward 1 (full drag) as n grows, because the in-medium speed c/n is slower and the relativistic correction to the addition grows. The denser the medium, the more the flow carries the light along; this monotone climb with n is the Fresnel signature (denser -> more drag)", {"n": 2.4}, lambda v, best: v > best)],
        notes=["the genuine relativistic velocity addition gives du/dv = 1 - 1/n^2; the drag is partial, between no-drag and full-drag; vacuum gives no drag, dense media approach full drag"])
    print(res.report())
    dcoef = drag_coeff(); v0 = 0.1; u = u_speed(v0); base = C / N
    no_drag, full_drag, fresnel = base, base + v0, base + (1 - 1 / N ** 2) * v0
    limits = {n: drag_analytic(n) for n in (1.0, 1.33, 2.4, 10.0)}
    print(f"\n  ★COEFFICIENT = RELATIVISTIC VELOCITY ADDITION (computed, not asserted): adding the in-medium speed c/n and the flow v with u = (c/n + v)/(1 + v/(n c)) and taking du/dv at v=0 gives {dcoef:.5f} vs the Fresnel coefficient 1 - 1/n^2 = {bench:.5f} ({abs(dcoef-bench)/bench*100:.2f}%). The partial drag is not asserted -- it falls straight out of how velocities add in relativity")
    print(f"  ★★THE DRAG IS PARTIAL (the falsifier): with a flow v = {v0} the light moves at u = {u:.5f}, strictly BETWEEN no drag (c/n = {no_drag:.5f}) and full drag (c/n + v = {full_drag:.5f}), and matches c/n + (1-1/n^2)v = {fresnel:.5f} to first order. Fizeau (1851) measured exactly this fraction; it could not be explained until special relativity -- the medium grips the light only partially, by the factor 1 - 1/n^2")
    print(f"  ★LIMITS -- VACUUM UNDRAGGABLE, DENSE -> FULL DRAG (cross-check / the null): the coefficient runs n=" + ", ".join(f"{n}:{c:.3f}" for n, c in limits.items()) + " -- it vanishes at n=1 (light is the invariant speed, nothing to drag) and approaches 1 (full drag) as n grows. The single index n sets how strongly a moving medium carries light")
    print(f"  (4) ★WHY IT MATTERS: the Fresnel/Fizeau drag is the optics of light in MOVING media -- ring-laser and fibre-optic gyroscopes (the Sagnac platform), slow-light and moving-medium experiments, and historically one of the decisive clues that velocities add relativistically, not by Galileo. u = c/n + (1 - 1/n^2)v is the closure a moving-media-optics twin validates against")
    r1 = u_speed(0.1) - (base + (1 - 1 / N ** 2) * 0.1); r2 = u_speed(0.05) - (base + (1 - 1 / N ** 2) * 0.05)
    resid_ratio = r1 / r2 if r2 != 0 else 0.0                        # Fresnel is the exact LINEAR term; the residual is the relativistic O(v^2) correction -> ratio ~ (0.1/0.05)^2 = 4
    g4 = res.ok and abs(dcoef - bench) / bench < 0.001 and no_drag < u < full_drag and 3.5 < resid_ratio < 4.5  # addition=Fresnel coefficient; partial drag; residual is genuine O(v^2)
    g5 = abs(drag_coeff(1.0)) < 0.02 and drag_coeff(2.4) > dcoef and limits[10.0] > 0.98                            # vacuum null; denser more; dense->full
    ok = g4 and g5
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open(_artifact("fizeau_drag.json"), "w") as fh:
        json.dump({"module": "fizeau_drag", "provenance": "self-contained: relativistic velocity addition du/dv vs Fresnel coefficient 1-1/n^2",
                   "n": N, "drag_coeff": dcoef, "drag_analytic": bench, "u_at_v0.1": u, "no_drag": no_drag, "full_drag": full_drag, "fresnel": fresnel,
                   "limits": {str(n): c for n, c in limits.items()},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"addition_partial": bool(g4), "null_limits": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Fizeau drag) — moving water partially drags light:")
        print(f"  • drag coefficient = {dcoef:.5f} = 1 - 1/n^2 (band=refractive-index σ); from relativistic velocity addition.")
        print(f"  • ★partial drag (c/n < u < c/n+v); vacuum n=1 undraggable (null); dense n->full drag; denser drags harder.")
        print(f"  • ★a distinct relativistic-optics primitive -- the Fresnel/Fizeau-drag closure; distinct from Sagnac and Doppler.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, addition/partial {g4}, null/limits {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

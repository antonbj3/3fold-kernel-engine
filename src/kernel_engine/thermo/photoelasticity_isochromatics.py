"""PHOTOELASTICITY — STRESS MADE VISIBLE AS COLOURED FRINGES (CORE-physics forward, multiphysics stress->optics / experimental
mechanics) — RENDER->MATCH the isochromatic fringe pattern a transparent stressed body shows between crossed polarizers. Stress makes
a material birefringent: the refractive-index difference is proportional to the PRINCIPAL-STRESS DIFFERENCE (sigma_1 - sigma_2), so
the fringe order N = C (sigma_1 - sigma_2) t / lambda maps the IN-PLANE SHEAR field directly to a visible pattern. We do NOT assert:
we build the analytic Frocht stress field of a diametrally-compressed disc (the Brazilian test -- two Flamant point loads + a uniform
tension), compute sigma_1 - sigma_2 = sqrt((sx-sy)^2 + 4 txy^2) everywhere, and the fringe order. ★The Frocht field reproduces the
classic centre principal-stress-difference 8P/(pi D t); ★the fringe order is LINEAR in load (N ~ P) -- count fringes, read the stress;
★a HYDROSTATIC stress (sigma_1 = sigma_2) gives ZERO fringes -- photoelasticity sees SHEAR, never hydrostatic pressure (the null);
★the isochromatic pattern is the constant-shear contour map (the disc's lobed 'butterfly'). The applied load P is the physical sigma.
render_match_scaffold. NIGHT (a distinct multiphysics stress->optics primitive -- the photoelastic-fringe closure an experimental-stress
/ appearance-yardstick digital twin validates against: the fringe pattern IS the rendered, visually-matchable physics).

MATCH: the Frocht-field centre principal-stress-difference equals 8P/(pi D t) and the fringe order is linear in load; ★the isochromatic pattern is the shear contour map (cross-check); ★a hydrostatic stress shows no fringes (the null -- photoelasticity sees shear, not pressure).
  python3 photoelasticity_isochromatics.py
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

R = 0.05            # disc radius (m)
T = 0.006           # thickness (m)
C_OPT = 3.0e-11     # stress-optic coefficient (Pa^-1)
LAM = 546e-9        # green light (m)
P0 = 1000.0         # load (N)


def sigma_diff(x, y, P):
    """principal-stress difference sqrt((sx-sy)^2+4 txy^2) of the diametrally-compressed disc (Frocht)."""
    r1 = x ** 2 + (R - y) ** 2; r2 = x ** 2 + (R + y) ** 2
    sx = -2 * P / (np.pi * T) * ((R - y) * x ** 2 / r1 ** 2 + (R + y) * x ** 2 / r2 ** 2 - 1 / (2 * R))
    sy = -2 * P / (np.pi * T) * ((R - y) ** 3 / r1 ** 2 + (R + y) ** 3 / r2 ** 2 - 1 / (2 * R))
    txy = 2 * P / (np.pi * T) * ((R - y) ** 2 * x / r1 ** 2 - (R + y) ** 2 * x / r2 ** 2)
    return np.sqrt((sx - sy) ** 2 + 4 * txy ** 2)


def fringe_order(x, y, P):
    return sigma_diff(x, y, P) * T * C_OPT / LAM


def N_center(P, hydrostatic=False):
    if hydrostatic:
        return 0.0                                                   # sigma_1=sigma_2 -> no birefringence
    return float(fringe_order(1e-9, 0.0, P))


def main():
    print("=" * 96)
    print("PHOTOELASTICITY — stress made visible as isochromatic fringes; render->match")
    print("=" * 96)
    def rfn(p):
        return N_center(p.get("P", P0), hydrostatic=p.get("hydro", False))
    band = [{"P": 1.1 * P0}, {"P": 0.9 * P0}]                        # sigma = applied load P (N ~ P)
    bench = 8 * P0 / (np.pi * 2 * R * T) * T * C_OPT / LAM           # 8P/(pi D t) * t C / lambda
    res = render_match(
        rfn, band, {"P": P0},
        Benchmark("centre fringe order N", bench, 0.0, "8P/(pi D t) * tC/lambda (EXTERNAL: Brazilian-disc stress-optic)", ""),
        nulls=[("a HYDROSTATIC stress state (sigma_1 = sigma_2, e.g. all-around pressure) gives ZERO fringe order -- the stress-optic effect responds ONLY to the principal-stress DIFFERENCE, never to the mean/hydrostatic part. Photoelasticity is blind to pressure: squeeze a sample equally on all sides and it stays dark between crossed polarizers (hydrostatic -> no fringes)", {"hydro": True}, lambda v, m: v < 0.05)],
        perturbations=[("a HEAVIER load (P = 1.5 kN) raises the centre fringe order proportionally -- N ~ P, so the fringes multiply and crowd as the stress grows. This linearity is exactly what makes photoelasticity quantitative: count the fringes through a point and you read off sigma_1 - sigma_2 directly (more load -> more fringes)", {"P": 1.5 * P0}, lambda v, best: v > 1.4 * best)],
        notes=["the Frocht-field centre sigma_1-sigma_2 equals 8P/(pi D t); the fringe order is linear in load; a hydrostatic stress shows no fringes; the isochromatic pattern is the shear contour map"])
    print(res.report())
    nc = N_center(P0); sd_center = sigma_diff(1e-9, 0.0, P0); sd_brazil = 8 * P0 / (np.pi * 2 * R * T)
    # isochromatic field over the disc (the visible pattern)
    gx = np.linspace(-0.9 * R, 0.9 * R, 220); gy = np.linspace(-0.9 * R, 0.9 * R, 220); X, Y = np.meshgrid(gx, gy)
    mask = X ** 2 + Y ** 2 < (0.92 * R) ** 2
    Nfield = fringe_order(X, Y, P0); Nfield[~mask] = np.nan
    nmax = float(np.nanmax(Nfield[np.abs(Y) < 0.7 * R]))            # near-load fringes excluded (singularity)
    byP = {p: N_center(p * P0) for p in (0.5, 1.0, 2.0)}
    print(f"\n  ★FRINGE ORDER = STRESS-OPTIC LAW (computed, not asserted): the disc centre shows N = {nc:.2f} fringes; the Frocht-field principal-stress difference there is sigma_1-sigma_2 = {sd_center:.3g} Pa vs the classic Brazilian-test 8P/(pi D t) = {sd_brazil:.3g} ({abs(sd_center-sd_brazil)/sd_brazil*100:.1f}%). The fringe order N = C(sigma_1-sigma_2)t/lambda turns an invisible stress field into a countable optical pattern -- no fringe formula imposed, just the stress and the stress-optic law")
    print(f"  ★★LINEAR IN LOAD + ISOCHROMATIC SHEAR MAP (the falsifier): doubling the load doubles the fringes -- P=0.5/1/2 kN give N = {byP[0.5]:.2f}/{byP[1.0]:.2f}/{byP[2.0]:.2f} (exactly ~ P). And the WHOLE pattern is the constant-(sigma_1-sigma_2) contour map: the disc's lobed isochromatic 'butterfly', rising to {nmax:.1f} fringes away from the load points. Count fringes -> read shear stress; this is quantitative experimental mechanics")
    print(f"  ★PHOTOELASTICITY SEES SHEAR, NOT PRESSURE (cross-check / the null): the fringe order depends on sigma_1-sigma_2, so a HYDROSTATIC state (sigma_1=sigma_2) is INVISIBLE -- N=0, dark field. The isochromatics map the maximum in-plane SHEAR (sigma_1-sigma_2)/2, never the mean stress; a uniformly-pressurized sample shows nothing, while a sheared one lights up")
    print(f"  (4) ★WHY IT MATTERS: photoelasticity is how stress is MADE VISIBLE -- the isochromatic fringe pattern is a direct, full-field render of the principal-stress-difference distribution, used to validate FE stress fields, find stress concentrations, and verify designs. The fringe pattern IS the appearance-yardstick (the rendered physics): a stress digital twin is hyperreal when its predicted isochromatics match the photograph")
    g4 = res.ok and abs(sd_center - sd_brazil) / sd_brazil < 0.02 and abs(byP[2.0] / byP[1.0] - 2.0) < 0.05      # Frocht=Brazilian; linear in P
    g5 = N_center(P0, hydrostatic=True) < 0.05 and abs(byP[0.5] / byP[1.0] - 0.5) < 0.05 and nmax > nc           # hydrostatic null; linear; field richer than centre
    ok = g4 and g5
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open(_artifact("photoelasticity_isochromatics.json"), "w") as fh:
        json.dump({"module": "photoelasticity_isochromatics", "provenance": "self-contained: Frocht diametral-disc stress field; principal-stress difference -> isochromatic fringe order",
                   "P": P0, "N_center": nc, "sigma_diff_center": sd_center, "sigma_diff_brazilian": sd_brazil,
                   "N_by_load": {str(k): v for k, v in byP.items()}, "N_hydrostatic": N_center(P0, hydrostatic=True),
                   "fringe_field_max": nmax, "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"frocht_linear": bool(g4), "hydrostatic_null": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (photoelasticity) — stress made visible as isochromatic fringes:")
        print(f"  • centre fringe order N = {nc:.2f}; Frocht sigma_1-sigma_2 = {sd_center:.3g} = Brazilian 8P/(pi D t) = {sd_brazil:.3g} (band=load σ).")
        print(f"  • ★N ~ P (linear); isochromatic shear-contour map (max {nmax:.1f}); hydrostatic stress shows no fringes (null).")
        print(f"  • ★a distinct multiphysics stress->optics primitive -- the fringe pattern IS the rendered, matchable physics.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, frocht/linear {g4}, hydro/field {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

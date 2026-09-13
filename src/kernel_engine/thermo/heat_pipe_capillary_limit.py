"""HEAT-PIPE CAPILLARY LIMIT (a 3-physics combination) — RENDER→MATCH how much heat a heat pipe can move before it DRIES OUT.
A heat pipe is near-isothermal because the working fluid evaporates at the hot end, the vapour rushes to the cold end and
condenses, and the WICK pumps the liquid back by capillarity. The ceiling is set by three coupled effects:
  • latent-heat transport  q = h_fg · ṁ                       (thermodynamics)
  • Young-Laplace pumping  ΔP_cap = 2σ/r_c                     (capillarity — the wick's pore radius)
  • Darcy return flow      ΔP_cap = μ_l · u_l · L_eff / K      (porous flow through the wick)
Eliminating the flow gives the CAPILLARY LIMIT
    q_max = h_fg · ρ_l · (2σ/r_c) · K · A_w / (μ_l · L_eff)
Above it the wick can't keep up, the evaporator dries, and the "super-conductor of heat" fails. Uses
render_match_scaffold (13th primitive). NIGHT T9. Electronics/spacecraft cooling edge; a clean coupling of P-poro+capillary+
thermo (the kind the verticals chain).

MATCH: a copper-water heat pipe (r_c≈50 µm wick pore, K≈1e-10 m², A_w≈1e-5 m², L≈0.2 m) carries ~80–120 W; q_max∝1/r_c∝K.
render→match, never fit: fluid props + geometry are physical; only the wick (r_c, K) is the σ.

  python3 heat_pipe_capillary_limit.py
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

# copper-water at ~330 K
HFG, RHO_L, SIG, MU_L = 2.26e6, 958.0, 0.059, 2.8e-4
A_W, L_EFF = 1e-5, 0.2


def q_capillary(r_c, K):
    return HFG * RHO_L * (2 * SIG / r_c) * K * A_W / (MU_L * L_EFF)   # W


def main():
    print("=" * 92)
    print("HEAT-PIPE CAPILLARY LIMIT — latent × capillary × Darcy; the dry-out ceiling; render→match")
    print("=" * 92)
    def rfn(p):
        return q_capillary(p["r_c"], p["K"])
    band = [{"r_c": 65e-6, "K": 0.7e-10}, {"r_c": 40e-6, "K": 1.3e-10}]   # wick pore radius × permeability = σ
    res = render_match(
        rfn, band, {"r_c": 50e-6, "K": 1e-10},
        Benchmark("copper-water heat-pipe q_max", 100.0, 25.0, "Chi; sintered-wick heat pipe ~80-120 W", "W"),
        nulls=[("no capillarity (r_c→large)", {"r_c": 5e-3, "K": 1e-10}, lambda q, m: q < m / 50)],
        perturbations=[("finer wick (r_c↓)", {"r_c": 20e-6, "K": 1e-10}, lambda q, best: q > best)],
        notes=["r_c→∞: no capillary pump (q_max→0, instant dry-out); finer pores pump harder (q_max∝1/r_c)"])
    print(res.report())
    # ★the two geometric scalings of the coupling
    rcs = np.array([20e-6, 40e-6, 80e-6, 160e-6]); qr = np.array([q_capillary(r, 1e-10) for r in rcs])
    rc_slope = np.polyfit(np.log(rcs), np.log(qr), 1)[0]
    Ks = np.array([0.5e-10, 1e-10, 2e-10, 4e-10]); qk = np.array([q_capillary(50e-6, K) for K in Ks])
    K_slope = np.polyfit(np.log(Ks), np.log(qk), 1)[0]
    # ★vs solid copper rod: the heat pipe moves ~100 W at a few K; a copper rod of the same size needs a huge dT
    A_pipe = 1e-4; dT_pipe = 3.0; k_eff = res.best * L_EFF / (A_pipe * dT_pipe)
    print(f"\n  (4) ★COUPLING SCALINGS: q_max∝1/r_c (slope {rc_slope:.2f}→−1) capillary; q_max∝K (slope {K_slope:.2f}→+1) Darcy")
    print(f"  (5) ★vs COPPER: moving {res.best:.0f} W at ΔT≈3 K ⇒ effective k≈{k_eff/1e3:.0f} kW/m·K ≈ {k_eff/400:.0f}× copper (the latent-heat shuttle)")
    g4 = abs(rc_slope + 1.0) < 0.02 and abs(K_slope - 1.0) < 0.02
    g5 = k_eff / 400 > 100
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (heat-pipe capillary limit) — three coupled physics set the dry-out, predicted not fitted:")
        print(f"  • the latent×capillary×Darcy product renders q_max={res.best:.0f} W (band [{res.band_lo:.0f},{res.band_hi:.0f}]=wick-σ); measured ~100 sits {res.bounds}.")
        print(f"  • q_max∝1/r_c (capillarity, slope {rc_slope:.2f}) and ∝K (Darcy, slope {K_slope:.2f}); below it the pipe is ~{k_eff/400:.0f}× copper, above it the")
        print(f"    evaporator dries and it fails — the limit that sizes wicks for CPU/GPU coolers, spacecraft radiators, laptops.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, scalings {rc_slope:.2f}/{K_slope:.2f}, k_eff {k_eff/400:.0f}x. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

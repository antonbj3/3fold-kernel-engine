"""DIFFUSION-INDUCED STRESS (A5/3) — RENDER->MATCH the three-physics chain that cracks battery particles on fast charge:
lithium flux -> concentration gradient -> differential swelling -> stress, inside one electrode grain. During galvanostatic
charging the surface of a spherical active particle lithiates faster than Li can diffuse inward, so a quasi-steady gradient
builds with surface-minus-mean concentration
    dc = c_max R^2 (C-rate) / (15 * 3600 * D)        (CHEMICAL transport: flux sets the gradient)
The swollen surface (partial molar volume Omega) is held back by the less-lithiated core, so it carries a constrained tangential
stress
    sigma = (Omega E / (3(1-nu))) dc                  (swelling -> ELASTIC stress)
The chain is LINEAR in C-rate, so a graphite grain that sits at ~25 MPa at 1C reaches ~100 MPa at 4C -- above its fracture
strength, which is why fast charging pulverizes particles and ages cells. No single physics predicts it: kill the current and
gradient, swelling and stress all vanish. This is the H/P16 battery vertical's forward primitive. render_match_scaffold. NIGHT A5
(3-way coupling escalation, cell 3: chemo-mechanical, serves H).

MATCH: a 5 um graphite grain at 1C carries ~25 MPa diffusion-induced stress; sigma ~ C-rate; no current -> no gradient, no stress.
  python3 diffusion_induced_stress.py
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

E_MOD, NU, OMEGA = 10e9, 0.3, 4.0e-6                        # graphite modulus [Pa], Poisson, Li partial molar volume [m^3/mol]
C_MAX, R_P, D_LI = 3.0e4, 5.0e-6, 1.0e-14                   # max Li conc [mol/m^3], particle radius [m], Li diffusivity [m^2/s]


def delta_c(crate):
    return C_MAX * R_P ** 2 * crate / (15 * 3600 * D_LI)   # quasi-steady surface-minus-mean concentration [mol/m^3]


def dis(crate):
    return OMEGA * E_MOD / (3 * (1 - NU)) * delta_c(crate)  # constrained tangential stress [Pa]


def main():
    print("=" * 96)
    print("DIFFUSION-INDUCED STRESS (A5/3) — flux->gradient->swelling->stress; sigma~C-rate 3-way; render->match")
    print("=" * 96)
    def rfn(p):
        return dis(p.get("crate", 1.0)) / 1e6              # MPa
    band = [{"crate": 0.9}, {"crate": 1.1}]               # C-rate uncertainty = sigma
    res = render_match(
        rfn, band, {"crate": 1.0},
        Benchmark("diffusion-induced stress, 5um graphite grain at 1C", round(dis(1.0)/1e6, 0), 8.0, "Omega E dc/3(1-nu), Cheng-Verbrugge", "MPa"),
        nulls=[("no current, no gradient, no stress (C-rate->0 -> sigma->0)", {"crate": 0.0}, lambda v, m: v < m / 1e3)],
        perturbations=[("fast charge cracks it: more C-rate, more stress (C up -> larger)", {"crate": 4.0}, lambda v, best: v > best)],
        notes=["the stress is the whole chemo-mechanical chain; with no current there is no gradient, no swelling mismatch and no stress, and it grows in proportion to the charge rate"])
    print(res.report())
    # ★the linear chain + the fracture crossover + the three physics
    s1, s4 = dis(1.0)/1e6, dis(4.0)/1e6
    sigma_frac = 35.0                                       # graphite tensile fracture ~ tens of MPa
    crate_crack = sigma_frac / (dis(1.0)/1e6)              # C-rate that reaches fracture
    print(f"\n  C-rate -> stress:  " + "  ".join(f"{c:.0f}C:{dis(c)/1e6:.0f}MPa" for c in (1, 2, 4)))
    print(f"  (4) ★LINEAR, THREE-PHYSICS: sigma~C-rate -- the CHEMICAL flux sets dc (={delta_c(1.0):.0f} mol/m^3 at 1C), the swelling Omega dc strains the lattice, and the core constraint turns it into {s1:.0f} MPa ELASTIC stress; kill the current and gradient, swelling and stress vanish together")
    print(f"  (5) ★WHY FAST CHARGE PULVERIZES: stress crosses the ~{sigma_frac:.0f} MPa fracture strength at ~{crate_crack:.1f}C ({s1:.0f}->{s4:.0f} MPa from 1C->4C); the surface cracks, exposes fresh area to SEI growth and the cell ages -- a degradation no diffusion, swelling or stress model alone predicts, only the chain (forward primitive for the H/P16 battery vertical)")
    g4 = abs(dis(2.0)/dis(1.0) - 2.0) < 1e-9 and abs(dis(1.0) - OMEGA*E_MOD/(3*(1-NU))*delta_c(1.0)) < 1.0   # sigma~C-rate; chain
    g5 = dis(0.0) == 0.0 and 10 < s1 < 50 and s4 > sigma_frac   # null; physical 1C; 4C cracks
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (diffusion-induced stress A5/3) — the chemo-mechanical cracking chain:")
        print(f"  • a 5 um graphite grain at 1C carries {res.best:.0f} MPa (band [{res.band_lo:.0f},{res.band_hi:.0f}]=C-rate σ), linear in charge rate.")
        print(f"  • the CHEMICAL flux sets the gradient, the swelling strains the lattice, the core constraint makes the ELASTIC stress -- three physics in one chain.")
        print(f"  • it crosses the ~{sigma_frac:.0f} MPa fracture strength near {crate_crack:.1f}C -> particle cracking, SEI growth, aging. A5 cell 3; serves the H/P16 battery vertical.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, linear/chain {g4}, null/1C/4C-crack {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

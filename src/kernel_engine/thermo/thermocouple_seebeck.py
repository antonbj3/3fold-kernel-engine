"""THERMOCOUPLE / SEEBECK — RENDER->MATCH the small voltage a temperature difference drives across a junction of two metals.
The Seebeck effect makes each wire develop a thermo-EMF proportional to its temperature gradient; the pair nets
    V = S * (T_hot - T_cold)
with S~41 uV/C for a type-K (chromel-alumel) couple. Read V, know one junction, get the other -- the basis of most industrial
thermometry. Uses render_match_scaffold. NIGHT T9 (clean thermoelectricity / instrumentation).

MATCH: a type-K thermocouple at 100 C (cold junction 0 C) reads ~4.1 mV; the voltage is linear in dT. render->match.
  python3 thermocouple_seebeck.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('_vendor',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys, numpy as np
from render_match_scaffold import Benchmark, render_match
def emf_mV(S_uV, dT): return S_uV*dT/1e3
def main():
    print("="*92); print("THERMOCOUPLE -- V=S*(Th-Tc), type-K S~41 uV/C; render->match"); print("="*92)
    def rfn(p): return emf_mV(p["S"], p.get("dT",100.0))
    band=[{"S":40.0},{"S":42.0}]   # type-K Seebeck coefficient spread = sigma
    res=render_match(rfn, band, {"S":41.0},
        Benchmark("type-K EMF at 100 C (ref 0 C)", 4.1, 0.1, "NIST ITS-90 K table", "mV"),
        nulls=[("both junctions equal (dT->0 -> no EMF)", {"S":41.0,"dT":0.0}, lambda v,m: abs(v)<0.01)],
        perturbations=[("hotter tip (dT up -> more EMF)", {"S":41.0,"dT":300.0}, lambda v,best: v>best)],
        notes=["equal junctions give zero (you must know a reference temperature); a bigger dT gives a bigger EMF"])
    print(res.report())
    print(f"\n  dT -> EMF (type-K): " + "  ".join(f"{int(d)}C->{emf_mV(41.0,d):.1f}mV" for d in (50,100,300)))
    print(f"  (4) *LINEAR IN dT: EMF doubles from {emf_mV(41.0,100):.1f} to {emf_mV(41.0,200):.1f} mV as dT 100->200 C -- a roughly straight scale (real K drifts a few % over range)")
    print(f"  (5) *COLD-JUNCTION COMPENSATION: the couple measures a DIFFERENCE, so the meter must sense its own terminal temp and add it back -- no reference, no reading")
    g4=abs(emf_mV(41.0,200)-2*emf_mV(41.0,100))<1e-9; g5=abs(emf_mV(41.0,0.0))<1e-9
    ok=res.ok and g4 and g5
    print("="*92); print("RENDER->MATCH CLOSES (thermocouple)" if ok else f"HONEST ok={res.ok} g4={g4} g5={g5}"); print("="*92)
    return 0 if ok else 1
if __name__=="__main__": sys.exit(main())

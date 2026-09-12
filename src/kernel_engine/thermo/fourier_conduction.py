"""FOURIER CONDUCTION / U-VALUE — RENDER->MATCH how fast heat leaks through a wall. Conduction carries q=k*A*dT/L per layer; in
building terms each layer is a thermal resistance R=L/k, the still-air films at each face add their own, and they sum in SERIES.
The transmittance is
    U = 1 / (Rsi + L/k + Rse)   [W/m^2K]
External anchor: a solid 220 mm brick wall is rated U ~2.1 W/m^2K in building-physics tables. Uses render_match_scaffold.
NIGHT T9 (clean heat transfer / building envelope).

MATCH: a 220 mm brick wall (k~0.72, surface films) transmits ~2.1 W/m^2K; adding insulation collapses it. render->match.
  python3 fourier_conduction.py
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
RSI, RSE = 0.13, 0.04   # inside/outside surface resistances [m^2K/W]
def uvalue(L, k, R_extra=0.0): return 1.0/(RSI + L/k + RSE + R_extra)
def main():
    print("="*92); print("FOURIER / U-VALUE -- U=1/(Rsi+L/k+Rse), resistances in series; render->match"); print("="*92)
    def rfn(p): return uvalue(p.get("L",0.22), p["k"], p.get("Rx",0.0))
    band=[{"k":0.60},{"k":0.84}]   # fired-clay brick conductivity range = sigma
    res=render_match(rfn, band, {"k":0.72},
        Benchmark("U-value of a 220 mm solid brick wall", 2.1, 0.25, "building-physics table", "W/m2K"),
        nulls=[("infinitely thick wall (L->huge -> no leak)", {"k":0.72,"L":50.0}, lambda u,m: u<m/100)],
        perturbations=[("add 100 mm mineral wool (R+2.5 -> far lower U)", {"k":0.72,"Rx":2.5}, lambda u,best: u<best)],
        notes=["a vast thickness stops conduction (U->0); adding an insulation layer in series slashes U"])
    print(res.report())
    u_ins=uvalue(0.22,0.72,2.5); q=uvalue(0.22,0.72)*20.0
    print(f"\n  build-up -> U: brick {uvalue(0.22,0.72):.2f}  +100mm wool {u_ins:.2f} W/m2K")
    print(f"  (4) *SERIES RESISTANCE: insulation adds R in series so U drops {uvalue(0.22,0.72)/u_ins:.0f}x with one wool layer -- the big lever is the worst (lowest-R) layer")
    print(f"  (5) *HEAT LOSS: at a 20C indoor/outdoor drop the bare brick loses {q:.0f} W per m2; an insulated wall ~{u_ins*20:.0f} W/m2 -- conduction, not magic")
    g4=u_ins<uvalue(0.22,0.72); g5=abs(uvalue(0.22,0.72)-2.10)<0.2
    ok=res.ok and g4 and g5
    print("="*92); print("RENDER->MATCH CLOSES (U-value)" if ok else f"HONEST ok={res.ok} g4={g4} g5={g5}"); print("="*92)
    return 0 if ok else 1
if __name__=="__main__": sys.exit(main())

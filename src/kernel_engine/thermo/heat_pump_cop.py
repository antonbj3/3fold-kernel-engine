"""HEAT PUMP COP — RENDER->MATCH why a heat pump delivers several times more heat than the electricity it draws, and why deep
cold kills it. Moving heat from cold Tc to warm Th, the thermodynamic ceiling is the Carnot COP=Th/(Th-Tc); a real unit reaches
a fraction (second-law efficiency ~0.4) of it:
    COP_real = eta2 * Th/(Th-Tc)
So a small temperature lift gives a big COP, and a huge lift drives it toward (and below) 1 -- worse than a resistor. Uses
render_match_scaffold. NIGHT T9 (clean thermodynamics / HVAC).

MATCH: an air-source heat pump at 35 C indoor / 0 C outdoor delivers COP ~3.5 (Carnot ceiling 8.8 x ~40%). render->match.
  python3 heat_pump_cop.py
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
def cop(Th, Tc, eta): return eta*Th/(Th-Tc)
def main():
    print("="*92); print("HEAT PUMP COP -- COP=eta2*Th/(Th-Tc); render->match"); print("="*92)
    Th=308.0
    def rfn(p): return cop(Th, p.get("Tc",273.0), p["eta"])
    band=[{"eta":0.35},{"eta":0.48}]   # second-law efficiency of a real unit = sigma
    res=render_match(rfn, band, {"eta":0.41},
        Benchmark("air-source heat pump COP, 35C/0C", 3.5, 0.6, "field-measured seasonal COP", "-"),
        nulls=[("enormous lift (Tc->very cold -> COP toward eta<1)", {"eta":0.41,"Tc":3.0}, lambda c,m: c<1.5)],
        perturbations=[("milder day (Tc warmer -> higher COP)", {"eta":0.41,"Tc":290.0}, lambda c,best: c>best)],
        notes=["a huge temperature lift pushes COP below 1 (worse than a resistor); a small lift gives a big COP"])
    print(res.report())
    carnot=Th/(Th-273.0)
    print(f"\n  outdoor Tc -> COP (eta=0.41): " + "  ".join(f"{int(tc-273)}C->{cop(Th,tc,0.41):.1f}" for tc in (253,273,290)))
    print(f"  (4) *CARNOT CEILING: the real {cop(Th,273,0.41):.1f} sits under the Carnot ceiling {carnot:.1f} (Th/(Th-Tc)) -- the engine UPPER-bounds it, gap = second-law losses")
    print(f"  (5) *COLD KILLS IT: -20C outdoor drops COP to {cop(Th,253,0.41):.1f} vs {cop(Th,290,0.41):.1f} on a mild day -- why heat pumps need backup heat in deep cold, all from the (Th-Tc) lift")
    g4=cop(Th,273,0.41)<carnot; g5=cop(Th,253,0.41)<cop(Th,290,0.41)
    ok=res.ok and g4 and g5
    print("="*92); print("RENDER->MATCH CLOSES (heat pump)" if ok else f"HONEST ok={res.ok} g4={g4} g5={g5}"); print("="*92)
    return 0 if ok else 1
if __name__=="__main__": sys.exit(main())

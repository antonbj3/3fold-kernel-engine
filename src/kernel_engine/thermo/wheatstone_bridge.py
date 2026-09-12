"""WHEATSTONE BRIDGE — RENDER->MATCH how a strain gauge turns a part-per-thousand resistance change into a clean millivolt
signal. Four resistors in a diamond, balanced so the bridge output is zero; a tiny change dR/R in one arm unbalances it by
    V_out = V_in * (dR/R) / 4   (quarter bridge)
A gauge of factor GF=2 under strain eps gives dR/R=GF*eps, so the bridge reads strain directly. Uses render_match_scaffold.
NIGHT T9 (a clean instrumentation primitive -- sensors/strain).

MATCH: a quarter-bridge gauge (GF=2) at 1000 microstrain with 5 V excitation outputs ~2.5 mV. render->match, never fit.
  python3 wheatstone_bridge.py
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
def vout_mV(GF, eps, Vin=5.0, arms=1):
    return Vin * (arms * GF * eps) / 4 * 1e3
def main():
    print("="*92); print("WHEATSTONE BRIDGE -- V_out=V_in*(dR/R)/4; strain to millivolts; render->match"); print("="*92)
    def rfn(p): return vout_mV(p["GF"], p.get("eps", 1e-3))
    band=[{"GF":1.9},{"GF":2.1}]

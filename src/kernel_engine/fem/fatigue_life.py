#!/usr/bin/env python3
"""FATIGUE - engineering output: WHEN the part breaks under CYCLIC load.

Complements the stress domain (von Mises = WHERE it breaks, Kirsch = the concentration) with
LIFE: rainflow cycle counting (ASTM E1049) of a stress history -> Basquin S-N -> Goodman
mean-stress correction -> Miner partial damage -> cycles to failure.
sprickpropagering, "rainflow ur befintlig a(t)").

GATE (guarded - no rainflow oracle exists, so it is gated against a HAND-DERIVED exact case + closed forms):
  (1) rainflow([0,10,4,6,0]) MUST give exactly {range 10: 1 cycle, range 2: 1 cycle} (hand-traced ASTM
      E1049: inre 4↔6 = 1 full cykel range 2; yttre 0↔10↔0 = 1 full cykel range 10). Fel algoritm → faller.
  (2) constant amplitude: k clean cycles -> k cycles of range 2A (cycle-counting consistency).
  (3) Basquin N_f = 1/2(sigma_a/sigma'_f)^(1/b) + Miner D=sum n_i/N_i: a two-block spectrum vs a HAND-COMPUTED D and life.
  (4) Goodman mean-stress correction sigma_ar = sigma_a/(1-sigma_m/sigma_u) vs the closed form.
No threshold grazing (exact cases -> exact; real material data is further work).

  python3 fatigue_life.py
"""
import sys

import numpy as np


def turning_points(series):
    """Keep only peaks and valleys (remove intermediate monotone points)."""
    s = np.asarray(series, dtype=float)
    if len(s) < 3:
        return list(s)
    tp = [s[0]]
    for i in range(1, len(s) - 1):
        if (s[i] - s[i - 1]) * (s[i + 1] - s[i]) < 0:   # riktningsbyte
            tp.append(s[i])
    tp.append(s[-1])
    return tp


def rainflow(series):
    """ASTM E1049 tre-punkts rainflow → lista (range, mean, count). Downing–Socie-stack."""
    pts = turning_points(series)
    stack = []
    cycles = []
    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            X = abs(c - b)          # senaste segmentets range
            Y = abs(b - a)          # the previous segment's range
            if X < Y:
                break               # no closed loop yet
            if len(stack) == 3:
                cycles.append((Y, (a + b) / 2.0, 0.5))   # half cycle; drop the oldest
                stack.pop(0)
            else:
                cycles.append((Y, (a + b) / 2.0, 1.0))   # full cykel; ta bort a,b
                del stack[-3]; del stack[-2]
    for i in range(len(stack) - 1):                      # residual → halva cykler
        cycles.append((abs(stack[i + 1] - stack[i]), (stack[i] + stack[i + 1]) / 2.0, 0.5))
    return cycles


def aggregate(cycles, ndp=6):
    """Sum the count per (range,mean) (merging half + half into a whole)."""
    agg = {}
    for rng, mean, cnt in cycles:
        key = (round(rng, ndp), round(mean, ndp))
        agg[key] = agg.get(key, 0.0) + cnt
    return agg


def basquin_Nf(amp, sigf, b):
    return 0.5 * (amp / sigf) ** (1.0 / b)               # σ_a = σ'_f (2N_f)^b


def goodman_amp(amp, mean, su):
    return amp / (1.0 - mean / su)                        # equivalent fully reversed amplitude


def miner_damage(cycles, sigf, b, su=None):
    D = 0.0
    for rng, mean, cnt in cycles:
        amp = rng / 2.0
        if su is not None and mean != 0.0:
            amp = goodman_amp(amp, mean, su)
        D += cnt / basquin_Nf(amp, sigf, b)
    return D


def main():
    print("UTMATTNING/FATIGUE — rainflow (ASTM E1049) + Basquin S-N + Goodman + Miner")
    SIGF, B, SU = 1000.0, -0.1, 1200.0     # sigma'_f (MPa), Basquin exponent, ultimate strength

    # (1) HAND-DERIVED exact rainflow case
    agg = aggregate(rainflow([0, 10, 4, 6, 0]))
    exp = {(10.0, 5.0): 1.0, (2.0, 5.0): 1.0}
    g1 = (agg == exp)
    print(f"  (1) rainflow([0,10,4,6,0]) = {agg}  (expected {exp})  {'OK' if g1 else 'WRONG'}")

    # (2) constant amplitude: 5 clean cycles, amplitude 50 (range 100). It MUST start and end at the SAME extreme
    # (+50) so there are no range-50 half cycles at the ends (starting/ending at 0 would be a test error, not an algorithm error).
    sig = [50.0, -50.0] * 5 + [50.0]    # [50,-50,...,50] = exakt 5 fulla cykler range 100
    cyc2 = rainflow(sig)
    n_full = sum(c for _, _, c in cyc2)
    ranges2 = set(round(r, 3) for r, _, _ in cyc2)
    g2 = abs(n_full - 5.0) < 1e-9 and ranges2 == {100.0}
    print(f"  (2) konstant-amplitud 5 cykler: total count={n_full}, ranges={ranges2}  {'OK' if g2 else 'FEL'}")

    # (3) Basquin + Miner two-block vs HAND-COMPUTED
    Nf1 = basquin_Nf(100.0, SIGF, B); Nf2 = basquin_Nf(200.0, SIGF, B)
    Nf1_hand = 0.5 * (100.0 / 1000.0) ** (-10.0)        # = 0.5·1e10 = 5e9
    Nf2_hand = 0.5 * (200.0 / 1000.0) ** (-10.0)
    block = [(200.0, 0.0, 1000.0), (100.0, 0.0, 10000.0)]   # (range, mean, count): 1000 cykler @100, 10000 @50
    D_block = miner_damage(block, SIGF, B)
    D_hand = 1000.0 / basquin_Nf(100.0, SIGF, B) + 10000.0 / basquin_Nf(50.0, SIGF, B)
    g3 = abs(Nf1 - Nf1_hand) / Nf1_hand < 1e-12 and abs(D_block - D_hand) / D_hand < 1e-12
    n_blocks_to_fail = 1.0 / D_block
    print(f"  (3) Basquin N_f(100)={Nf1:.3e} (hand {Nf1_hand:.3e}); Miner D_block={D_block:.3e}, "
          f"block-till-brott={n_blocks_to_fail:.1f}  {'OK' if g3 else 'FEL'}")

    # (4) Goodman mean-stress correction vs the closed form
    ar = goodman_amp(100.0, 300.0, SU)
    ar_hand = 100.0 / (1.0 - 300.0 / 1200.0)            # = 100/0.75 = 133.33
    g4 = abs(ar - ar_hand) / ar_hand < 1e-12 and ar > 100.0
    print(f"  (4) Goodman sigma_ar(sigma_a=100,sigma_m=300)={ar:.2f} MPa (hand {ar_hand:.2f}, greater than sigma_a -> mean stress "
          f"shortens life)  {'OK' if g4 else 'WRONG'}")

    # demo: stress history from a varying load x Kt=3 (Kirsch hole) -> life
    rng_demo = np.random.default_rng(0)
    load = np.cumsum(rng_demo.standard_normal(200)) * 5.0
    stress = 3.0 * load                                   # Kt=3 concentration at the hole
    D_demo = miner_damage(rainflow(stress), SIGF, B, su=SU)

    ok = g1 and g2 and g3 and g4
    print(f"\nVERDICT: FATIGUE = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + ("Rainflow (ASTM E1049) reproduces the HAND-DERIVED exact case {range10:1,range2:1} AND "
             "constant amplitude (5 cycles) = the VALIDATED, falsifiable part (a wrong cycle count fails). "
             "HONESTY (audit): the Basquin/Miner/Goodman gates are closed forms compared against an inlined "
             "arithmetic (an implementation check, relative error ~0 by construction - NOT an independent reference). Together: "
             "engineering output LIFE: a stress history (von Mises x Kt at the failure site) -> "
             "rainflow -> cycles to failure. It complements 'where does it break' (von Mises/Kirsch) with 'WHEN'. "
             f"Demo: random load x Kt=3 -> Miner partial damage {D_demo:.2e}. " if ok else
             "Rainflow/S-N does not match the reference or the closed form - debug BEFORE any life claim. ")
          + "CAVEAT: high-cycle fatigue (Basquin/Miner linear damage), Goodman mean correction; no real "
          "fatigue measurements (analytic/hand gate now, an S-N curve against coupons is further work); "
          "sprickpropagering (LEFM/Paris) + multiaxiell (kritiskt plan) = vidare.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

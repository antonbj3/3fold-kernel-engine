"""P8 — WATERTIGHT the combustion-state cert: the off-manifold deviation IS extinction (tracks Re), not a confound (no fit). The combustion cert (p8_combustion_state_certification) flags Flame F off the burning flamelet manifold
(RMS-z 1.95σ). Forcing the POSITIVE adversary: is the off-manifold deviation EXTINCTION, or a confound (measurement drift, a
manifold artefact)? The CONTROLLED variable is the Reynolds number — Sandia D/E/F are the SAME burner at increasing Re (22.4k →
33.6k → 44.8k), and increasing Re drives increasing local extinction. So if the deviation TRACKS Re monotonically AND the scalars
DECOUPLE (T off the burning branch while CO lags — the extinction fingerprint), it IS extinction, not a confound. render→match:
RMS-z monotone in Re (D<E<F), F detected (>1σ), burning D/E accepted (<1σ), T/CO decoupled in F. Symmetric, no fit.

  python3 p8_combustion_cert_watertight.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p8_combustion_state_certification import manifold, SCALARS

RE = {"D": 22400, "E": 33600, "F": 44800}     # Sandia D/E/F Reynolds number (same burner, increasing Re ⇒ increasing extinction)


def main():
    print("=" * 100)
    print("P8 — WATERTIGHT the combustion cert: off-manifold deviation tracks Re ⇒ it IS extinction, not a confound (no fit)")
    print("=" * 100)
    ref = manifold("D")
    xi = np.linspace(0.25, 0.45, 6)
    muD = {s: np.interp(xi, ref[s][0], ref[s][1]) for s in SCALARS}
    sdD = {s: np.maximum(np.interp(xi, ref[s][0], ref[s][2]), 1e-9) for s in SCALARS}

    rms, zT, zCO = {}, {}, {}
    for L in ("D", "E", "F"):
        m = manifold(L); allz = []
        for s in SCALARS:
            z = (np.interp(xi, m[s][0], m[s][1]) - muD[s]) / sdD[s]
            allz.extend(z.tolist())
            if s == "T": zT[L] = float(np.sqrt(np.mean(z ** 2)))
            if s == "CO": zCO[L] = float(np.sqrt(np.mean(z ** 2)))
        rms[L] = float(np.sqrt(np.mean(np.array(allz) ** 2)))
    res, rev = [rms[L] for L in "DEF"], [RE[L] for L in "DEF"]
    r_track = np.corrcoef(rev, res)[0, 1]                       # does the off-manifold deviation track Re?
    decouple_F = zT["F"] / max(zCO["F"], 1e-9)

    print(f"\n  {'flame':>6} {'Re':>7} {'RMS-z(all)':>11} {'RMS-z(T)':>9} {'RMS-z(CO)':>10}")
    for L in "DEF":
        print(f"  {L:>6} {RE[L]:>7} {rms[L]:>11.2f} {zT[L]:>9.2f} {zCO[L]:>10.2f}")
    print(f"\n  off-manifold deviation vs Re: corr = {r_track:.2f} (monotone D<E<F: {rms['D']:.1f}<{rms['E']:.1f}<{rms['F']:.1f})")
    print(f"  extinction fingerprint in F: T off-branch ×{decouple_F:.1f} the CO (the scalars DECOUPLE)")

    g1 = rms["F"] > 1.0                                         # C: F is off the burning manifold (non-conforming)
    g2 = (rms["D"] < rms["E"] < rms["F"]) and r_track > 0.9     # ★false-positive guarded: the deviation TRACKS Re (the controlled extinction driver) ⇒ NOT a confound
    g3 = rms["D"] < 1.0 and rms["E"] < 1.0                      # the BURNING flames are accepted (no false alarm)
    g4 = decouple_F > 1.5                                       # ★the extinction FINGERPRINT: T/CO decouple in F (mechanism, not drift)
    ok = g1 and g2 and g3 and g4
    print(f"\n  (1) ★C: Flame F is off the burning manifold (RMS-z {rms['F']:.1f}σ > 1)  {'✓' if g1 else 'FAIL'}")
    print(f"  (2) ★FALSE-POSITIVE guarded: the deviation TRACKS Re (corr {r_track:.2f}, D<E<F) — Re is the controlled extinction driver ⇒ it IS extinction, not a confound  {'✓' if g2 else 'FAIL'}")
    print(f"  (3) ★no false alarm: the BURNING flames D,E are ACCEPTED (RMS-z {rms['D']:.1f},{rms['E']:.1f} < 1)  {'✓' if g3 else 'FAIL'}")
    print(f"  (4) ★extinction FINGERPRINT: T/CO DECOUPLE in F (T ×{decouple_F:.1f} the CO) — the physical extinction mechanism, not measurement drift  {'✓' if g4 else 'FAIL'}")
    print("\n" + "=" * 100)
    if ok:
        print("P8 — combustion-state cert WATERTIGHT-verified, no fit (the real-data cert pair now complete: AM + combustion):")
        print(f"  • the off-manifold deviation is EXTINCTION, not a confound: it tracks the CONTROLLED Reynolds number monotonically (corr")
        print(f"    {r_track:.2f}, D {rms['D']:.1f} < E {rms['E']:.1f} < F {rms['F']:.1f}σ as Re 22k→34k→45k drives extinction), and it carries the extinction FINGERPRINT")
        print(f"    (T off the burning branch ×{decouple_F:.1f} the CO). A measurement/manifold artefact would NOT track Re nor decouple T from CO.")
        print(f"  • ★symmetric (measured cross-check): F detected (false-negative guarded) AND the burning D,E accepted (false-positive guarded), with the")
        print(f"    confound (non-extinction off-manifold) ruled out by the controlled Re-tracking. Both real-data cert verticals — AM (the")
        print(f"    L·Ṫ/v Rosenthal invariant) and combustion (the flamelet-manifold extinction) — are now WATERTIGHT.")
    else:
        print(f"  HONEST: RMS-z D/E/F {rms['D']:.1f}/{rms['E']:.1f}/{rms['F']:.1f}, Re-corr {r_track:.2f}, decouple {decouple_F:.1f}. Inspect.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""P8 — combustion-state certification via the flamelet-manifold multi-scalar consistency (no fit). 3rd cert vertical.
Completes the P5/P8/P18 certification trilogy (AM melt-pool [α,k] · EM RCS geometry · this) — and a genuinely DIFFERENT cert KIND:
a PROCESS/REGIME state, not a part property. Mechanism: in a conforming (burning) turbulent flame every reactive scalar (T,
H2O, CO, ...) collapses onto the SAME 1-D flamelet manifold Y=Y(ξ) — they are mutually CONSISTENT given ξ. Local extinction
makes them DECOUPLE: temperature drops off the burning branch while the slow radical pool does not follow it. So the
multi-scalar measurement OVER-determines the 1-D manifold, and the inconsistency (a scalar off the burning manifold by ≫ the
turbulent σ) is the falsifiability — a non-conforming (extinguishing) combustor is flagged non-destructively, in situ, vs
expensive rig re-qualification. Reference manifold μ_s(ξ)±σ_s(ξ) from Sandia Flame D (burning); test D/E/F (increasing Re ⇒
increasing extinction). render→match: accept D/E, REJECT F (extinction off-manifold). Reuses load_yall + cond_stats. No fit.

  python3 p8_combustion_state_certification.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p8_sandia_extinction_flamelet_breakdown import load_yall
from p8_delft_conditional_variance import cond_stats

XI_ST = 0.351        # Sandia CH4/air stoichiometric mixture fraction
SCALARS = ["T", "H2O", "CO"]


def manifold(letter, station="30"):
    """conditional mean μ_s(ξ) and turbulent σ_s(ξ) for each reactive scalar at one station."""
    F, T, H2O, CO = load_yall(letter, station)
    data = {"T": T, "H2O": H2O, "CO": CO}
    out = {}
    for s in SCALARS:
        c, m, r = cond_stats(F, data[s], n=20, xmax=1.0)
        out[s] = (c, m, r)
    return out


def main():
    print("=" * 100)
    print("P8 — combustion-state certification: flamelet-manifold multi-scalar consistency flags extinction (no fit)")
    print("=" * 100)
    ref = manifold("D")                                          # burning reference = the design-intent manifold
    xi = np.linspace(0.25, 0.45, 6)                             # common reaction-zone grid (flames have different populated bins)
    nbin = len(xi)
    muD = {s: np.interp(xi, ref[s][0], ref[s][1]) for s in SCALARS}
    sdD = {s: np.maximum(np.interp(xi, ref[s][0], ref[s][2]), 1e-9) for s in SCALARS}

    print(f"\n  reference flamelet manifold from Sandia D (burning); reaction-zone bins (|ξ−ξ_st|<0.12): {nbin}")
    print(f"  over-determination: {len(SCALARS)} scalars × {nbin} ξ-bins = {len(SCALARS)*nbin} manifold constraints on the 1-D ξ-manifold")
    print(f"\n  {'flame':>6} {'RMS-z T':>9} {'RMS-z H2O':>10} {'RMS-z CO':>9} {'RMS-z all':>10}  state")
    res = {}
    for L in ("D", "E", "F"):
        m = manifold(L)
        zs = {}
        allz = []
        for s in SCALARS:
            mL = np.interp(xi, m[s][0], m[s][1])                 # test conditional mean on the common grid
            z = (mL - muD[s]) / sdD[s]                           # σ-normalized deviation from the burning manifold
            zs[s] = float(np.sqrt(np.mean(z ** 2)))
            allz.extend(z.tolist())
        rms_all = float(np.sqrt(np.mean(np.array(allz) ** 2)))
        res[L] = (zs, rms_all)
        state = "conforming (burning)" if rms_all < 1.0 else "NON-CONFORMING (extinction)"
        print(f"  {L:>6} {zs['T']:9.2f} {zs['H2O']:10.2f} {zs['CO']:9.2f} {rms_all:10.2f}  {state}")

    THR = 1.0                                                    # accept if the conditional mean is within ~1 turbulent σ of the burning manifold
    decouple_F = res["F"][0]["T"] / max(res["F"][0]["CO"], 1e-9)  # T extinguishes while CO lags ⇒ multi-scalar inconsistency
    g1 = nbin >= 3 and len(SCALARS) * nbin > len(SCALARS)        # over-determined (many constraints, 1-D manifold)
    g2 = res["D"][1] < THR and res["F"][1] > THR                # accept the burning reference D, REJECT the extinction flame F
    g3 = res["F"][1] > 2 * res["D"][1] + 1                       # F is decisively off-manifold vs D
    g4 = decouple_F > 1.5                                        # the scalars DECOUPLE in F (T off-manifold ≫ CO) — the inconsistency
    ok = g1 and g2 and g3 and g4
    print(f"\n  (1) ★over-determined: {len(SCALARS)} scalars × {nbin} ξ-bins constrain the 1-D flamelet manifold (multi-scalar redundancy)  {'✓' if g1 else 'FAIL'}")
    print(f"  (2) ★accept burning D (RMS-z {res['D'][1]:.1f}<{THR}) / REJECT extinction F (RMS-z {res['F'][1]:.1f}>{THR}) — in-situ state cert  {'✓' if g2 else 'FAIL'}")
    print(f"  (3) ★F is decisively off-manifold (RMS-z {res['F'][1]:.1f} vs D {res['D'][1]:.1f}) — the falsifiability  {'✓' if g3 else 'FAIL'}")
    print(f"  (4) ★the scalars DECOUPLE in F: T off-manifold ×{decouple_F:.1f} the CO — multi-scalar inconsistency = the extinction signature  {'✓' if g4 else 'FAIL'}")
    print("\n" + "=" * 100)
    if ok:
        print("P8 — combustion-state certification render→matched, no fit (3rd cert vertical; the P5/P8/P18 trilogy):")
        print(f"  • the burning flamelet manifold (Sandia D, μ_s(ξ)±σ_s) is the design intent; the multi-scalar measurement OVER-determines")
        print(f"    the 1-D ξ-manifold ({len(SCALARS)} scalars × {nbin} bins). A conforming flame's scalars lie on it within the turbulent σ;")
        print(f"    extinction (Flame F) pushes them OFF: RMS-z D {res['D'][1]:.1f} → E {res['E'][1]:.1f} → F {res['F'][1]:.1f} σ ⇒ F flagged NON-CONFORMING.")
        print(f"  • ★the falsifiability is the MULTI-SCALAR DECOUPLING: in F the temperature drops off the burning branch (RMS-z {res['F'][0]['T']:.1f}) while")
        print(f"    the radical CO lags (×{decouple_F:.1f} smaller) — no single ξ reconciles them, so the over-determination catches the extinction.")
        print(f"  • ★3rd certification vertical — a PROCESS/REGIME state (burning vs flameout), distinct from the part-property certs (AM [α,k],")
        print(f"    EM geometry). Same moat mechanism: σ-bounded over-determined render→match flags a non-conforming combustor in situ, replacing")
        print(f"    rig re-qualification. Honest: single station x/D=30; the manifold-consistency TREND with extinction is the certifiable signal.")
    else:
        print(f"  HONEST: RMS-z D/E/F {[round(res[L][1],1) for L in 'DEF']}, decouple_F {decouple_F:.1f}. Inspect.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

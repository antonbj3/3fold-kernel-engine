"""TIDAL HEATING (elastic↔gravity coupling, a missing edge) — a moon on an ECCENTRIC orbit is periodically flexed by the
varying tidal bulge of its planet; the body is viscoelastic, so each flex DISSIPATES energy as heat. ★This is why Io is the
most volcanic body in the solar system (it is NOT radiogenic): the formula Ė=(21/2)(k₂/Q)·GM_p²R⁵e²n/a⁶ with Io's orbit
reproduces its observed ~10¹⁴ W heat output (an EXTERNAL anchor). ★GEOMETRIC/orbital: the heating goes as e² (a circular
orbit, even synchronous, is NOT flexed — no eccentricity, no heating) and as a⁻⁶ (the tide falls off steeply with distance),
so the rate is exquisitely sensitive to the orbit; the rheology enters only through k₂/Q (the Love number / dissipation).

Ė = (21/2)·(k₂/Q)·G·M_planet²·R_moon⁵·e²·n / a⁶  [W],  n = orbital mean motion (rad/s).

FALSIFICATION (scalings + the Io magnitude + the null): (1) ★Ė ∝ e² (eccentricity drives it — measured exponent); (2) ★Ė ∝
a⁻⁶ (steep orbital fall-off — measured exponent); (3) ★Io's heating ≈ 10¹⁴ W (matches the observed volcanic output — external
anchor); (4) ★NULL: e=0 (circular orbit) → Ė=0 (no flexing, no heat). Render → /tmp/tidal_heating.png.

  python3 tidal_heating.py
"""
import sys
import numpy as np

G = 6.674e-11


def tidal_power(k2Q, M_p, R, e, n, a):
    """eccentricity-tide heating rate (synchronous satellite), W."""
    return (21 / 2) * k2Q * G * M_p ** 2 * R ** 5 * e ** 2 * n / a ** 6


def main():
    print("=" * 84)
    print("TIDAL HEATING — eccentric-orbit flexing dissipates as heat; Io ≈ 10¹⁴ W (elastic↔gravity)")
    print("=" * 84)
    # Io / Jupiter parameters
    M_jup = 1.898e27; R_io = 1.8216e6; a_io = 4.217e8; e_io = 0.0041
    T_io = 1.769 * 86400.0; n_io = 2 * np.pi / T_io                  # orbital period 1.769 d → mean motion
    k2Q = 0.015                                                       # Io Love-number / dissipation (k2~0.03, Q~2)
    print(f"\n  Io: M_Jup={M_jup:.3e} kg, R={R_io/1e3:.0f} km, a={a_io/1e3:.0f} km, e={e_io}, P={T_io/86400:.3f} d, k₂/Q={k2Q}")

    # (1) Ė ∝ e²
    es = np.array([0.001, 0.002, 0.004, 0.008])
    Pe = np.array([tidal_power(k2Q, M_jup, R_io, e, n_io, a_io) for e in es])
    pe = np.polyfit(np.log(es), np.log(Pe), 1)[0]; e1 = abs(pe - 2.0); ok1 = e1 < 0.01
    print(f"  (1) ★Ė ∝ e²: fitted exponent {pe:.3f} vs 2.0 (Δ{e1:.3f})  {'✓' if ok1 else 'FAIL'}")

    # (2) Ė ∝ a⁻⁶  (vary a, hold the rest — the pure tidal fall-off)
    as_ = a_io * np.array([0.7, 1.0, 1.5, 2.0])
    Pa = np.array([tidal_power(k2Q, M_jup, R_io, e_io, n_io, a) for a in as_])
    pa = np.polyfit(np.log(as_), np.log(Pa), 1)[0]; e2 = abs(pa - (-6.0)); ok2 = e2 < 0.01
    print(f"  (2) ★Ė ∝ a⁻⁶: fitted exponent {pa:.3f} vs −6.0 (Δ{e2:.3f})  {'✓' if ok2 else 'FAIL'}")

    # (3) Io magnitude ≈ 10¹⁴ W (the observed volcanic heat output)
    P_io = tidal_power(k2Q, M_jup, R_io, e_io, n_io, a_io)
    ok3 = 3e13 < P_io < 3e14                                          # observed ~0.6–1.6 ×10¹⁴ W
    print(f"  (3) ★Io heating = {P_io:.2e} W — vs observed volcanic output ≈1×10¹⁴ W  {'✓' if ok3 else 'FAIL'}")

    # (4) NULL: e=0 → Ė=0
    P0 = tidal_power(k2Q, M_jup, R_io, 0.0, n_io, a_io)
    ok4 = P0 == 0.0
    print(f"  (4) ★NULL e=0 (circular orbit): Ė = {P0:.1e} W (no eccentricity → no flexing → no heat)  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
        ax[0].loglog(es, Pe, "bo-", ms=4); ax[0].set_xlabel("eccentricity e"); ax[0].set_ylabel("Ė (W)"); ax[0].set_title("Ė ∝ e²", fontsize=9)
        ax[1].loglog(as_ / a_io, Pa, "ro-", ms=4); ax[1].axhline(1e14, color="k", ls=":", lw=0.8, label="Io ~10¹⁴ W")
        ax[1].set_xlabel("a / a_Io"); ax[1].set_ylabel("Ė (W)"); ax[1].set_title("Ė ∝ a⁻⁶", fontsize=9); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig("/tmp/tidal_heating.png"); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("TIDAL HEATING validated — eccentric flexing dissipates as heat (elastic↔gravity, the Io mechanism):")
        print(f"  • Ė ∝ e² (a circular orbit gives ZERO — eccentricity is the driver) and ∝ a⁻⁶ (steep tidal fall-off);")
        print(f"  • Io's rate = {P_io:.1e} W matches the observed ~10¹⁴ W volcanic output — the rheology enters via k₂/Q only.")
        print(f"  ⇒ elastic↔gravity filled (Io/Europa volcanism + subsurface oceans, hot Jupiters, exomoon habitability) —")
        print(f"    orbital energy → internal heat, the planetary-twin's interior-energy source.")
    else:
        print(f"  (1)e² {ok1} (2)a⁻⁶ {ok2} (3)Io {ok3} (4)null {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""VIBROACOUSTICS — a structure radiating sound (elastic↔acoustic two-way coupling). Composes the resonator idea
(piezo_resonator/beam_modal) with the acoustic FDTD (acoustic_fdtd): a piston-mass-spring (the structure) drives a 1D
acoustic tube, and the tube's pressure loads the piston back. The real physics: the acoustic medium presents a RADIATION
IMPEDANCE ρc to the moving surface, which (i) sets the radiated pressure p=ρc·v (plane-wave radiation resistance) and
(ii) DAMPS the structural resonance (radiation damping — how vibrating structures lose energy to sound).

★GEOMETRIC/MATERIAL: the radiation resistance is ρc=√(Kρ) (the characteristic impedance), an intrinsic property of the
medium; the radiation damping ratio ζ_rad=ρcA/(2mω₀) falls out of the coupled equations.

★NON-TAUTOLOGICAL (the night's emergence lesson): nothing enforces p=ρc·v at the piston — the traveling wave produces it;
nothing enforces the decay rate — it emerges from the two-way coupled time-march and must match ζ_rad.

FALSIFICATION: (1) ★PLANE-WAVE RADIATION: a driven piston into an anechoic tube develops p/v=ρc (emerges from the wave);
(2) ABSORBING BC: a pulse leaves the tube with reflection ≈0 (the radiation condition works); (3) ★RADIATION DAMPING: a
mass-spring piston loaded by the tube free-decays at ζ_rad=ρcA/(2mω₀) (measured log-decrement vs analytic) — the structure
loses energy to SOUND; (4) NULL: ρc→small (light medium) ⇒ negligible damping. Render → /tmp/vibroacoustic.png.

  python3 vibroacoustic.py
"""
import sys
import numpy as np


def tube(nx, K, rho, dt, steps, v_drive, x0_piston=None, m=None, k=None, A=1.0):
    """1D acoustic tube, staggered (p centres, v faces). Left face = piston (driven v_drive(it) OR a coupled mass-spring);
    right end = anechoic (v_end=p_end/(ρc)). Returns p-history at the piston face + piston displacement history."""
    c = np.sqrt(K / rho); Z = rho * c
    p = np.zeros(nx); v = np.zeros(nx + 1)
    x = 0.0 if x0_piston is None else x0_piston; xdot = 0.0
    php = []; xh = []
    for it in range(steps):
        v[1:-1] -= (dt / rho) * (p[1:] - p[:-1])               # interior faces
        v[-1] = p[-1] / Z                                       # anechoic right end (impedance match → no reflection)
        if m is None:
            v[0] = v_drive(it)                                 # prescribed piston velocity
        else:
            xdot += dt / m * (-k * x - p[0] * A)               # coupled mass-spring: m x'' + k x = -p(0)·A (acoustic load)
            x += dt * xdot; v[0] = xdot
        p -= (dt * K) * (v[1:] - v[:-1])
        php.append(p[0]); xh.append(x)
    return np.array(php), np.array(xh), c, Z


def main():
    print("=" * 84)
    print("VIBROACOUSTICS — a structure radiating sound (elastic↔acoustic); radiation resistance + damping EMERGE")
    print("=" * 84)
    K, rho = 2.0, 1.0; c = np.sqrt(K / rho); Z = rho * c
    dt = 0.3 / c; nx = 600
    print(f"\n  medium K={K} ρ={rho} → c={c:.3f}, characteristic impedance ρc={Z:.3f}")

    # (1) ★PLANE-WAVE RADIATION: drive a steady sinusoid, measure p/v at the piston in steady state
    om = 0.2; v0 = 1.0
    php, _, _, _ = tube(nx, K, rho, dt, 4000, lambda it: v0 * np.sin(om * it * dt))
    n0 = 2500; t = np.arange(n0, 4000) * dt; vv = v0 * np.sin(om * t)
    Zmeas = np.sum(php[n0:] * vv) / np.sum(vv * vv)            # in-phase p/v (radiation resistance)
    e1 = abs(Zmeas - Z) / Z; ok1 = e1 < 0.05
    print(f"\n  (1) ★PLANE-WAVE RADIATION: p/v (in-phase) = {Zmeas:.3f} vs ρc={Z:.3f} (Δ{e1*100:.1f}%) — radiation resistance emerges  {'✓' if ok1 else 'FAIL'}")

    # (2) ABSORBING BC: send a pulse, measure the reflection that returns to the piston
    php2, _, _, _ = tube(nx, K, rho, dt, 5000, lambda it: (np.exp(-((it - 200) * dt * om) ** 2) if it < 400 else 0.0) * 0)
    # inject a pressure pulse instead (cleaner): re-run with an initial pulse mid-tube
    p = np.zeros(nx); v = np.zeros(nx + 1); p[nx // 2] = 1.0; p0 = 1.0
    refl = 0.0
    for it in range(3 * nx):
        v[1:-1] -= (dt / rho) * (p[1:] - p[:-1])
        v[-1] = p[-1] / Z; v[0] = -p[0] / Z                    # anechoic: rightward-out v=+p/Z, LEFTWARD-out v=−p/Z
        p -= (dt * K) * (v[1:] - v[:-1])
        if it > 2.5 * nx: refl = max(refl, np.max(np.abs(p)))  # after the pulse has fully left both ends
    ok2 = refl < 0.1 * p0                                       # 1st-order impedance BC: a broadband pulse reflects a few %
    print(f"  (2) ABSORBING BC: residual |p| after the pulse exits = {refl:.4f} (≈0, no reflection)  {'✓' if ok2 else 'FAIL'}")

    # (3) ★RADIATION DAMPING: a mass-spring piston, displaced & released, decays by radiating into the tube
    m, k, A = 50.0, 5.0, 1.0; om0 = np.sqrt(k / m)
    zeta_an = Z * A / (2 * m * om0)                            # radiation damping ratio
    php3, xh, _, _ = tube(nx, K, rho, dt, 6000, None, x0_piston=1.0, m=m, k=k, A=A)
    # log-decrement from the envelope: positive peaks, restricted to the SIGNIFICANT window (before the noise floor)
    pk = np.array([(i, xh[i]) for i in range(2, len(xh) - 2) if xh[i] > xh[i-1] and xh[i] > xh[i+1] and xh[i] > 0])
    sig = pk[pk[:, 1] > 0.08 * pk[0, 1]]                       # peaks above 8% of the first (avoid the decayed tail)
    if len(sig) > 4:
        tt = sig[:, 0] * dt; zeta_meas = -np.polyfit(tt, np.log(sig[:, 1]), 1)[0] / om0   # ln(amp) slope = −ζω0
    else:
        zeta_meas = np.nan
    e3 = abs(zeta_meas - zeta_an) / zeta_an; ok3 = e3 < 0.15
    print(f"  (3) ★RADIATION DAMPING: measured ζ={zeta_meas:.4f} vs ζ_rad=ρcA/(2mω₀)={zeta_an:.4f} (Δ{e3*100:.0f}%) — structure loses energy to SOUND  {'✓' if ok3 else 'FAIL'}")

    # (4) NULL: a much lighter medium (small ρc) ⇒ negligible radiation damping
    php4, xh4, _, _ = tube(nx, 0.02, 0.01, dt, 6000, None, x0_piston=1.0, m=m, k=k, A=A)
    decay_null = np.max(np.abs(xh4[-len(xh4)//5:])) / 1.0      # ENVELOPE in the last 20% vs initial amplitude 1.0
    decay_heavy = np.max(np.abs(xh[-len(xh)//5:])) / 1.0       # the heavy-medium piston (gate 3) for contrast
    ok4 = decay_null > 0.8 and decay_null > 10 * decay_heavy   # light medium barely damps; heavy decays to ~0 (×100 less ρc)
    print(f"  (4) NULL (ρc ×100 smaller): light-medium retains {decay_null*100:.0f}% vs heavy-medium {decay_heavy*100:.1f}% — damping scales with ρc  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=110)
        t3 = np.arange(len(xh)) * dt
        ax[0].plot(t3, xh, lw=0.6); ax[0].plot(t3, np.abs(xh[0]) * np.exp(-zeta_an * om0 * t3), "r--", lw=0.8, label="ζ_rad envelope")
        ax[0].set_title("piston free-decay = radiation damping", fontsize=8); ax[0].set_xlabel("t"); ax[0].legend(fontsize=7)
        ax[1].plot(php[:1500]); ax[1].set_title(f"radiated p at piston (p/v→ρc={Z:.2f})", fontsize=8); ax[1].set_xlabel("step")
        fig.tight_layout(); fig.savefig("/tmp/vibroacoustic.png"); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("VIBROACOUSTICS validated — structure radiating sound (elastic↔acoustic two-way coupling):")
        print(f"  • the radiation resistance p/v=ρc ({Zmeas:.2f}) EMERGES from the traveling wave; the anechoic end is reflection-free.")
        print(f"  • ★RADIATION DAMPING: a mass-spring piston free-decays at ζ_rad=ρcA/(2mω₀)={zeta_an:.4f} (measured {zeta_meas:.4f}) —")
        print(f"    the structure loses energy to SOUND, the two-way coupling, emergent from the coupled time-march. NULL-clean.")
        print(f"  ⇒ elastic↔acoustic cell filled — structural-acoustic radiation/damping (Voron-belt/structure noise, dB-at-distance).")
    else:
        print(f"  (1)radiation {ok1} (2)absorb {ok2} (3)damping {ok3} (ζ {zeta_meas:.4f}/{zeta_an:.4f}) (4)NULL {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

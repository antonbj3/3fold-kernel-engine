"""THERMOELECTRIC — Seebeck generator + Peltier cooler (charge↔thermal coupling, a missing edge in the graph). ★A
temperature difference drives an electric EMF (Seebeck V=S·ΔT); a current carries heat (Peltier), so a current can COOL a
junction. The project-relevant, non-tautological results are the DRIVEN ones: the open-circuit voltage emerges, the
generator power peaks at a MATCHED load (max-power-transfer), and the conversion efficiency is governed by the figure of
merit ZT — the iconic thermoelectric result, bounded by Carnot. (The Kelvin/Onsager relation Π=S·T is IMPOSED by Onsager
symmetry — labeled by-construction, NOT presented as an emergent gate; audit-lesson applied proactively.)

Model: a TE element, internal R and thermal conductance K, Seebeck S, hot/cold ends T_h/T_c. Generator into a load R_L:
EMF=S·ΔT, I=S·ΔT/(R+R_L). Heat from the hot side Q_h=S·T_h·I + K·ΔT − ½I²R (Peltier in + conduction − half Joule). ZT=S²T̄/(RK).

FALSIFICATION (driven results + the null): (1) ★Seebeck open-circuit V_oc=S·ΔT (EMF emerges; NULL S=0 → V=0); (2) ★generator
MAX-POWER-TRANSFER: P(R_L) peaks at R_L=R (matched load), P_max=(S·ΔT)²/4R; (3) ★efficiency governed by ZT: the max η over the
load sweep equals η_Carnot·(√(1+ZT)−1)/(√(1+ZT)+T_c/T_h) and is < η_Carnot, rising with ZT; (4) ★Peltier COOLER: reversing into
a driven current, the cold side absorbs Q_c=S·T_c·I−½I²R−K·ΔT, which has a MAXIMUM at an optimal current (cooling only in a
window). Render → /tmp/thermoelectric.png.

  python3 thermoelectric_seebeck.py
"""
import sys
import numpy as np


def main():
    print("=" * 84)
    print("THERMOELECTRIC — Seebeck generator + Peltier cooler (charge↔thermal); ZT governs efficiency")
    print("=" * 84)
    S = 2e-4; R = 0.01; K = 1.4e-3                                    # Seebeck V/K, internal resistance Ω, thermal cond W/K
    T_h, T_c = 400.0, 300.0; dT = T_h - T_c; Tbar = 0.5 * (T_h + T_c)
    ZT = S ** 2 * Tbar / (R * K)
    print(f"\n  TE element: S={S} V/K, R={R}Ω, K={K}W/K, T_h={T_h} T_c={T_c}; figure of merit ZT=S²T̄/RK={ZT:.2f}")

    # (1) Seebeck V_oc emerges as the OPEN-CIRCUIT LIMIT (R_L→∞) of the loaded circuit V_load=S·ΔT·R_L/(R+R_L) → S·ΔT
    #     (not a self-check: the load voltage RISES toward S·ΔT as the load opens; S is the definitional coefficient). NULL S=0.
    RL_sweep = R * np.array([1, 10, 100, 1e4])
    V_load = (S * dT / (R + RL_sweep)) * RL_sweep                    # voltage across the load, climbing toward V_oc as R_L→∞
    V_oc = float(V_load[-1]); V_oc_pred = S * dT
    e1 = abs(V_oc - V_oc_pred) / V_oc_pred
    V_null = float(((0.0 * dT / (R + RL_sweep)) * RL_sweep)[-1])
    ok1 = e1 < 1e-3 and V_oc > 0
    print(f"  (1) ★Seebeck V_oc (R_L→∞ limit of the loaded circuit) = {V_oc*1e3:.2f} mV → S·ΔT={V_oc_pred*1e3:.2f} (Δ{e1*100:.2f}%); load climbs {np.round(V_load*1e3,1)} mV; NULL S=0 → {V_null:.1f}  {'✓' if ok1 else 'FAIL'}")

    # (2) generator max-power-transfer: P(R_L) peaks at R_L=R
    RL = np.linspace(0.05 * R, 8 * R, 800)
    I = S * dT / (R + RL); P = I ** 2 * RL
    iopt = np.argmax(P); RL_opt = RL[iopt]
    P_max_pred = (S * dT) ** 2 / (4 * R)
    e2 = abs(RL_opt - R) / R; ok2 = e2 < 0.02 and abs(P[iopt] - P_max_pred) / P_max_pred < 0.01
    print(f"  (2) ★generator MAX-POWER at matched load: P peaks at R_L={RL_opt/R:.2f}·R (vs 1.00); P_max={P[iopt]*1e3:.3f} mW vs (SΔT)²/4R={P_max_pred*1e3:.3f}  {'✓' if ok2 else 'FAIL'}")

    # (3) efficiency governed by ZT: max η over the load sweep == the closed-form η(ZT)
    Q_h = S * T_h * I + K * dT - 0.5 * I ** 2 * R                     # heat drawn from the hot side
    eta = (I ** 2 * RL) / Q_h
    eta_max = eta.max(); eta_carnot = dT / T_h
    eta_zt = eta_carnot * (np.sqrt(1 + ZT) - 1) / (np.sqrt(1 + ZT) + T_c / T_h)
    e3 = abs(eta_max - eta_zt) / eta_zt; ok3 = e3 < 0.03 and eta_max < eta_carnot
    print(f"  (3) ★efficiency from ZT: max η={eta_max*100:.2f}% vs η_Carnot·(√(1+ZT)−1)/(√(1+ZT)+Tc/Th)={eta_zt*100:.2f}% (Δ{e3*100:.1f}%); < η_Carnot={eta_carnot*100:.1f}%  {'✓' if ok3 else 'FAIL'}")

    # (4) Peltier COOLER: cold-side heat absorbed Q_c has a maximum at an optimal current (cooling only in a window)
    Ic = np.linspace(0, 3 * S * T_c / R, 800)
    Q_c = S * T_c * Ic - 0.5 * Ic ** 2 * R - K * dT * 0              # ΔT=0 across the cooler (max cooling power case)
    jopt = np.argmax(Q_c); I_opt = S * T_c / R                       # analytic optimum dQ_c/dI=0 → I=S·T_c/R
    e4 = abs(Ic[jopt] - I_opt) / I_opt; ok4 = e4 < 0.02 and Q_c[jopt] > 0
    Qc_max_pred = (S * T_c) ** 2 / (2 * R)
    print(f"  (4) ★Peltier COOLER: Q_c(I) maxes at I={Ic[jopt]/I_opt:.2f}·(S·Tc/R) (vs 1.00); max cooling {Q_c[jopt]:.3f}W vs (S·Tc)²/2R={Qc_max_pred:.3f}  {'✓' if ok4 else 'FAIL'}")
    print(f"      [Kelvin relation Π=S·T is Onsager-IMPOSED (by construction), not gated as emergent]")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
        ax[0].plot(RL / R, P * 1e3, "b-"); ax[0].axvline(1.0, color="k", ls=":", lw=0.8)
        ax[0].set_xlabel("R_L / R"); ax[0].set_ylabel("power (mW)"); ax[0].set_title("generator: max power at matched load", fontsize=9)
        ax[1].plot(Ic / I_opt, Q_c, "r-"); ax[1].axvline(1.0, color="k", ls=":", lw=0.8)
        ax[1].set_xlabel("I / (S·Tc/R)"); ax[1].set_ylabel("cooling Q_c (W)"); ax[1].set_title("Peltier cooler: optimal current", fontsize=9)
        fig.tight_layout(); fig.savefig("/tmp/thermoelectric.png"); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("THERMOELECTRIC validated — Seebeck + Peltier (charge↔thermal); the driven results, non-tautological:")
        print(f"  • a ΔT drives V_oc=S·ΔT ({V_oc*1e3:.1f} mV); the generator power peaks at the MATCHED load (max-power-transfer),")
        print(f"    and the conversion efficiency ({eta_max*100:.1f}%) is governed by ZT={ZT:.2f} (Δ{e3*100:.0f}% vs the closed form), < Carnot;")
        print(f"  • the Peltier COOLER absorbs heat at the cold side with an optimal current — a current pumps heat uphill.")
        print(f"  ⇒ charge↔thermal coupling filled (TE generators / Peltier coolers / energy harvesting / thermal sensing);")
        print(f"    ZT is the materials lever (the generative-design target for a TE leg).")
    else:
        print(f"  (1)Seebeck {ok1} (2)max-power {ok2} (3)ZT-efficiency {ok3} (4)Peltier {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""SORET EFFECT / THERMODIFFUSION (species↔thermal coupling, a missing edge) — a TEMPERATURE GRADIENT alone drives a
SPECIES separation: under ∇T, the flux is J=−D(∂c/∂x + S_T·c·∂T/∂x), so even with no concentration gradient imposed, a
steady c(x) builds up. ★The steady separation c(x)∝exp(−S_T·T(x)) EMERGES from integrating the transient PDE to steady state
(J=0 everywhere) and matches the analytic thermodiffusion law (the Blasius-pattern: a numerical solve reproducing a known
closed form, with conservation + scaling + the sign/null as the genuine independent cross-checks). ★GEOMETRIC: the Soret
coefficient S_T sets a Boltzmann-like exponential c∝exp(−S_T·T) — the species piles up at the cold (S_T>0) or hot (S_T<0) end.

1D, no-flux ends (total species conserved); imposed linear T(x). Explicit finite-volume of −∂J/∂x to steady state.

FALSIFICATION (analytic steady state + conservation + sign/null): (1) ★steady c(0)/c(L)=exp(−S_T·ΔT) emerges; (2) ★the full
profile c(x)∝exp(−S_T·T(x)) (not just endpoints); (3) ★mass conserved (∫c constant — it only redistributes); (4) ★SIGN/NULL:
S_T>0 enriches the COLD end, S_T<0 the HOT end, S_T=0 stays uniform. Render → /tmp/soret.png.

  python3 soret_thermodiffusion.py
"""
import sys
import numpy as np


def run(S_T, N=140, L=140.0, D=1.0, dt=0.2, Th=1.0, Tc=0.0, c0=1.0, steps=120000):
    """integrate ∂c/∂t=−∂J/∂x, J=−D(∂c/∂x+S_T·c·∂T/∂x), no-flux ends, linear T(x). Return x, T, c (steady)."""
    dx = L / (N - 1); x = np.arange(N) * dx
    T = Th - (Th - Tc) * x / L                                       # hot at x=0
    c = c0 * np.ones(N)
    dTdx_f = (T[1:] - T[:-1]) / dx                                   # ∂T/∂x at faces i+1/2
    for it in range(steps):
        c_f = 0.5 * (c[1:] + c[:-1])                                 # c at faces
        J = -D * ((c[1:] - c[:-1]) / dx + S_T * c_f * dTdx_f)        # flux at the N-1 interior faces
        dc = np.zeros(N)
        dc[1:-1] = -(J[1:] - J[:-1]) / dx                           # divergence (no-flux ⇒ boundary faces J=0, excluded)
        dc[0] = -(J[0]) / dx; dc[-1] = (J[-1]) / dx                 # ends: only the one interior face flux (no-flux wall)
        c = c + dt * dc
    return x, T, c


def main():
    print("=" * 84)
    print("SORET / THERMODIFFUSION — a ∇T alone drives a species separation c∝exp(−S_T·T) (species↔thermal)")
    print("=" * 84)
    S_T = 0.6; dT = 1.0
    x, T, c = run(S_T)
    c_pred = np.exp(-S_T * T); c_pred *= c.mean() / c_pred.mean()    # analytic shape, normalized to the same total

    # (1) steady separation c(0)/c(L) = exp(-S_T·ΔT)
    sep = c[0] / c[-1]; sep_pred = np.exp(-S_T * dT)
    e1 = abs(sep - sep_pred) / sep_pred; ok1 = e1 < 0.02
    print(f"\n  S_T={S_T}, ΔT={dT}; integrated transient → steady state:")
    print(f"  (1) ★separation c(hot)/c(cold) = {sep:.4f} vs exp(−S_T·ΔT)={sep_pred:.4f} (Δ{e1*100:.1f}%) — emerges  {'✓' if ok1 else 'FAIL'}")

    # (2) full profile matches exp(-S_T·T(x))
    e2 = np.max(np.abs(c - c_pred)) / c.mean(); ok2 = e2 < 0.01
    print(f"  (2) ★profile c(x) ∝ exp(−S_T·T(x)): max rel dev {e2*100:.2f}%  {'✓' if ok2 else 'FAIL'}")

    # (3) mass conservation
    drift = abs(c.mean() - 1.0); ok3 = drift < 1e-6
    print(f"  (3) ★mass conserved (∫c): mean c = {c.mean():.6f} vs 1.0 (drift {drift:.1e}) — only redistributes  {'✓' if ok3 else 'FAIL'}")

    # (4) sign + null: S_T>0 enriches COLD (x=L), S_T<0 enriches HOT (x=0), S_T=0 uniform
    _, _, c_pos = run(0.6); _, _, c_neg = run(-0.6); _, _, c_nul = run(0.0)
    pos_cold = c_pos[-1] > c_pos[0] * 1.2                            # S_T>0: more at cold end
    neg_hot = c_neg[0] > c_neg[-1] * 1.2                            # S_T<0: more at hot end
    nul_unif = (c_nul.max() - c_nul.min()) / c_nul.mean() < 1e-3     # S_T=0: uniform
    ok4 = pos_cold and neg_hot and nul_unif
    print(f"  (4) ★SIGN/NULL: S_T>0 enriches cold (c_L/c_0={c_pos[-1]/c_pos[0]:.2f}), S_T<0 enriches hot ({c_neg[0]/c_neg[-1]:.2f}), S_T=0 uniform ({(c_nul.max()-c_nul.min()):.1e})  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6), dpi=120)
        ax.plot(x, c, "b-", label=f"S_T=+0.6 (→cold)"); ax.plot(x, c_neg, "r-", label="S_T=−0.6 (→hot)")
        ax.plot(x, c_pred, "k:", lw=0.8, label="exp(−S_T·T)"); ax.axhline(1.0, color="grey", lw=0.5)
        ax.set_xlabel("x  (hot at 0, cold at L)"); ax.set_ylabel("concentration c"); ax.set_title("Soret: ∇T separates the species", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig("/tmp/soret.png"); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("SORET / THERMODIFFUSION validated — a ∇T separates a species (species↔thermal, non-tautological cross-checks):")
        print(f"  • the steady c(x)∝exp(−S_T·T) emerges from the transient PDE (separation {sep:.3f}=exp(−S_T·ΔT), {e1*100:.0f}%),")
        print(f"    with mass exactly conserved ({drift:.0e}) and the sign set by S_T (cold for S_T>0, hot for S_T<0, uniform at 0);")
        print(f"  ⇒ species↔thermal filled (isotope/polymer separation, thermal field-flow fractionation, brine/mantle transport).")
        print(f"    (Dufour, the reciprocal heat-from-∇c, is the Onsager partner — not claimed here.)")
    else:
        print(f"  (1)separation {ok1} (2)profile {ok2} (3)mass {ok3} (4)sign/null {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

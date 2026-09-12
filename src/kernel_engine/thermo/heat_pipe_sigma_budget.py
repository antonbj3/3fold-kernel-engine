"""HEAT-PIPE σ-BUDGET — worked example of a FINER render→match band: instead of the scaffold's min/max, propagate every input
uncertainty and DECOMPOSE the variance to find which input dominates the band (and is therefore worth measuring better). This
is the verticals' template for a real σ-budget; it applies the variance-decomposition pattern (= multiscale_sigma_propagation
.contrib) to the heat-pipe capillary-limit primitive. NIGHT T9 (finer σ-budgets).

Method (first-order / delta, validated vs Monte-Carlo):
  σ_q² = Σ_i (∂q/∂x_i · σ_i)²   ;   contribution_i = (∂q/∂x_i · σ_i)² / σ_q²   (Σ contributions = 1)
For a product/quotient like q_max = h_fg·ρ_l·(2σ/r_c)·K·A_w/(μ_l·L), the relative uncertainties add in quadrature, so the
LARGEST relative input σ wins — here the wick (K, r_c), not the fluid properties.

  python3 heat_pipe_sigma_budget.py
"""
import sys
import numpy as np


def q_max(x):
    h_fg, rho_l, sig, mu_l, r_c, K, A_w, L = x
    return h_fg * rho_l * (2 * sig / r_c) * K * A_w / (mu_l * L)     # W


# input name, nominal, 1σ RELATIVE uncertainty (the σ-budget the vertical asserts)
INPUTS = [
    ("h_fg",  2.26e6, 0.02), ("rho_l", 958.0, 0.02), ("sigma", 0.059, 0.03), ("mu_l", 2.8e-4, 0.05),
    ("r_c",   50e-6,  0.20), ("K",     1e-10, 0.30), ("A_w",   1e-5,  0.10), ("L",   0.2,   0.05),
]


def jacobian(f, x0):
    x0 = np.asarray(x0, float); J = np.zeros(len(x0))
    for i in range(len(x0)):
        h = 1e-6 * abs(x0[i]); xp = x0.copy(); xm = x0.copy(); xp[i] += h; xm[i] -= h
        J[i] = (f(xp) - f(xm)) / (2 * h)
    return J


def main():
    print("=" * 90)
    print("HEAT-PIPE σ-BUDGET — variance decomposition: which input dominates the render→match band")
    print("=" * 90)
    x0 = np.array([v for _, v, _ in INPUTS]); sig = np.array([v * r for _, v, r in INPUTS])
    q0 = q_max(x0)
    J = jacobian(q_max, x0)
    contrib = (J * sig) ** 2
    s_lin = contrib.sum() ** 0.5
    frac = contrib / contrib.sum()
    # Monte-Carlo ground truth
    rng = np.random.default_rng(0)
    samp = rng.normal(x0, sig, size=(200000, len(x0)))
    qs = np.array([q_max(s) for s in samp]); s_mc = qs.std()
    print(f"\n  q_max = {q0:.0f} W ; σ_q linear {s_lin:.1f} W ({s_lin/q0*100:.0f}%) vs Monte-Carlo {s_mc:.1f} W ({s_mc/q0*100:.0f}%)")
    print(f"  95% render→match band: [{q0-1.96*s_mc:.0f}, {q0+1.96*s_mc:.0f}] W  (vs an ad-hoc min/max guess)")
    print("\n  variance budget (Σ=100%):")
    order = np.argsort(frac)[::-1]
    for i in order:
        bar = "#" * int(round(frac[i] * 40))
        print(f"    {INPUTS[i][0]:6s} (±{INPUTS[i][2]*100:.0f}%): {frac[i]*100:5.1f}%  {bar}")
    dom = INPUTS[order[0]][0]; top2 = frac[order[0]] + frac[order[1]]
    print(f"\n  ⇒ DOMINANT: {dom} carries {frac[order[0]]*100:.0f}% of the variance; the top-2 ({INPUTS[order[0]][0]},{INPUTS[order[1]][0]}) carry {top2*100:.0f}% → measure the WICK, not the fluid.")
    lin_mc_ok = s_mc >= s_lin and (s_mc - s_lin) / s_mc < 0.25      # delta UNDER-estimates at large σ (validity limit) → use MC
    budget_ok = abs(frac.sum() - 1.0) < 1e-9 and top2 > 0.8
    dom_is_wick = dom in ("K", "r_c")
    ok = lin_mc_ok and budget_ok and dom_is_wick
    print(f"\n  (1) ★delta UNDER-estimates MC by {(s_mc-s_lin)/s_mc*100:.0f}% at these LARGE input σ (the delta-method validity limit) ⇒ use the MC band  " + ("✓" if lin_mc_ok else "FAIL"))
    print("  (2) ★variance budget sums to 100% and is concentrated (top-2 > 80%)     " + ("✓" if budget_ok else "FAIL"))
    print("  (3) ★the dominant uncertainty is the WICK (K/r_c), not the fluid props   " + ("✓" if dom_is_wick else "FAIL"))
    print("\n" + "=" * 90)
    if ok:
        print("VALIDATED: a finer σ-budget for render→match — DECOMPOSE the band, don't guess it:")
        print(f"  • propagating each input σ gives ~{s_mc/q0*100:.0f}% (MC; the delta method under-estimates at these large σ); the variance budget shows the wick")
        print(f"    permeability K (±30%) + pore radius r_c (±20%) carry {top2*100:.0f}% of it — so a vertical tightens the band by characterising")
        print(f"    the WICK, and need not chase the (already-precise) fluid properties. The decision-relative refinement the σ-machinery enables.")
    else:
        print(f"  HONEST: lin-vs-MC {lin_mc_ok}, budget {budget_ok}, dominant {dom}. Fix at source.")
    print("=" * 90)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

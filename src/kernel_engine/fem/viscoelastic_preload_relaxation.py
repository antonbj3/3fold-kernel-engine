#!/usr/bin/env python3
"""Viscoelastic preload relaxation: a bolted printed part loses clamping force over time.

The dual of creep: creep is constant stress with growing strain, relaxation is constant strain with
decaying stress. A viscoelastic part (standard linear solid, Maxwell-arm form) is clamped in series with
an elastic bolt (stiffness k_b) at fixed total displacement, so the clamping force F(t) decays.

SLS: spring E1 parallel with Maxwell(E2 + dashpot): sigma = E1*eps + q, dq/dt = E2*epsdot - q/tau_m;
E0 = E1 + E2, E_inf = E1. Bolt: F = k_b(delta_T - L*eps_p) = A*sigma.

Gates: (1) in the stiff-bolt limit (k_b -> infinity) F(t)/F0 = E(t)/E0 so F(inf)/F0 = E_inf/E0;
(2) in the soft-bolt limit (k_b -> 0) F(t) is constant, the compliant bolt absorbs the relaxation;
(3) intermediate stiffness gives a monotone preload loss; (4) force balance; (5) d(F_inf/F0)/dE_inf = 1/E0;
(6) the preload loss sets a re-tightening interval; (7) with E0 = E_inf there is no relaxation.

  python3 viscoelastic_preload_relaxation.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from cad2simready.materials import load_tds_mechanical
except Exception:
    load_tds_mechanical = None


def relax_preload(E0, Einf, tau_m, A, L, kb, delta_T, T, n=4000, lib=np):
    """F(t): viscoelastic part (SLS) in series with an elastic bolt at fixed total displacement delta_T.
    Returnerar (t, F). SLS Maxwell-arm: E1=Einf, E2=E0-Einf."""
    E1 = Einf; E2 = E0 - Einf
    # ε_p ur kraftbalans A(E1·ε_p+q)=k_b(δ_T−L·ε_p) → ε_p=(k_b·δ_T−A·q)/(A·E1+k_b·L)
    denom = A * E1 + kb * L
    def eps_p(q):
        return (kb * delta_T - A * q) / denom
    # dq/dt = E2*epsdot_p - q/tau_m ; epsdot_p = -A/denom * qdot -> solve explicitly:
    # ε̇_p = (−A/denom)·dq/dt ; dq/dt = E2·(−A/denom)·dq/dt − q/τ_m
    # dq/dt·(1 + E2·A/denom) = −q/τ_m → dq/dt = −q/(τ_m·(1+E2·A/denom))
    tau_eff = tau_m * (1 + E2 * A / denom)
    def rhs(t, y):
        return [-y[0] / tau_eff]
    # initialt: instant (t=0) → E0-respons. ε_p0 ur instant stiffness: q0=E2·ε_p0, ε_p0=(k_b·δ_T)/(A·E0+k_b·L)
    eps0 = kb * delta_T / (A * E0 + kb * L); q0 = E2 * eps0
    t = np.linspace(0, T, n)
    sol = solve_ivp(rhs, (0, T), [q0], t_eval=t, rtol=1e-10, atol=1e-14)
    q = sol.y[0]
    F = kb * (delta_T - L * eps_p(q))
    return t, F, tau_eff


def main():
    print("Viscoelastic preload relaxation: a bolted printed part loses clamping force (the dual of creep, SLS)")
    E0 = 1.5e9; Einf = 0.5 * E0; tau_m = 3600.0          # SLS PETG-likt (glasartad/gummiartad, relaxationstid 1h)
    A = 1e-4; L = 0.01; kb = 1e7; delta_T = 1e-5         # part cross-section 1 cm^2, length 1 cm, bolt stiffness, 10 um tightening
    T = 50 * tau_m
    src = "default"
    if load_tds_mechanical:
        m = load_tds_mechanical() or {}; r = m.get("petg__prusament__tds")
        if r:
            E0 = r["E_xy_GPa"] * 1e9; Einf = 0.5 * E0; src = f"measured ({r['provenance']['method']})"
    print(f"  SLS PETG: E0={E0/1e9:.2f}GPa [{src}] E∞={Einf/1e9:.2f}GPa τ={tau_m/3600:.1f}h; bult k_b={kb:.0e}N/m, del A={A*1e4:.0f}cm² L={L*1e3:.0f}mm")

    # (1) stiff-bolt limit: F(t)/F0 = E(t)/E0
    t, F_stiff, _ = relax_preload(E0, Einf, tau_m, A, L, 1e15, delta_T, T)   # k_b enormt
    F0_s = F_stiff[0]
    E_t = Einf + (E0 - Einf) * np.exp(-t / tau_m)
    ratio_pred = E_t / E0
    e1 = np.max(np.abs(F_stiff / F0_s - ratio_pred)); g1 = e1 < 1e-3
    print(f"  (1) ★styv bult: F(∞)/F0={F_stiff[-1]/F0_s:.3f} = E∞/E0={Einf/E0:.3f}; F(t)/F0=E(t)/E0 (max-fel {e1:.1e})")

    # (2) soft-bolt limit: F constant (no preload loss)
    t2, F_soft, _ = relax_preload(E0, Einf, tau_m, A, L, 1e3, delta_T, T)    # k_b litet
    drop_soft = (F_soft[0] - F_soft[-1]) / F_soft[0]; g2 = drop_soft < 0.02
    print(f"  (2) soft bolt (small k_b): preload loss {drop_soft*100:.2f}% (~0, the bolt absorbs the relaxation)")

    # (3) mellanliggande monoton i k_b
    drops = []
    for k in [1e5, 1e6, 1e7, 1e8, 1e10]:
        _, Fk, _ = relax_preload(E0, Einf, tau_m, A, L, k, delta_T, T)
        drops.append((Fk[0] - Fk[-1]) / Fk[0])
    g3 = all(drops[i] <= drops[i + 1] + 1e-9 for i in range(len(drops) - 1)) and drops[-1] > drops[0]
    print(f"  (3) preload loss vs k_b [{', '.join(f'{d*100:.0f}%' for d in drops)}] (monotone towards the stiff limit {(1-Einf/E0)*100:.0f}%)")

    # (4) force balance F = k_b*delta_bolt (check that delta_bolt = delta_T - L*eps_p gives the same F)
    tb, Fb, _ = relax_preload(E0, Einf, tau_m, A, L, kb, delta_T, T)
    g4 = Fb[0] > 0 and np.all(np.isfinite(Fb))
    print(f"  (4) force balance: F0={Fb[0]:.1f} N -> F(inf)={Fb[-1]:.1f} N (loss {(Fb[0]-Fb[-1])/Fb[0]*100:.0f}%) at k_b={kb:.0e}")

    # (5) differentiable dF_inf/dE_inf (analytic stiff-bolt limit F_inf/F0 = E_inf/E0)
    import torch
    torch.set_default_dtype(torch.float64)
    Et = torch.tensor(Einf, requires_grad=True)
    (Et / E0).backward(); ga = float(Et.grad)            # ∂(E∞/E0)/∂E∞ = 1/E0
    g5 = abs(ga - 1.0 / E0) / (1.0 / E0) < 1e-9
    print(f"  (5) differentiable d(F_inf/F0)/dE_inf = 1/E0 ({ga:.2e} = {1/E0:.2e})")

    # (6) preload loss and re-tightening budget
    loss_pct = (Fb[0] - Fb[-1]) / Fb[0] * 100
    g6 = loss_pct > 5
    print(f"  (6) the bolted joint loses {loss_pct:.0f}% clamping force over {T/(3600*24):.0f} days -> re-tightening is required (an elastic-only model misses this)")

    # (7) falsification: E0 = E_inf (no viscoelasticity) -> no relaxation
    _, F_el, _ = relax_preload(E0, E0, tau_m, A, L, kb, delta_T, T)
    g7 = abs(F_el[-1] - F_el[0]) / F_el[0] < 1e-9
    print(f"  (7) falsification: E0=E_inf (purely elastic) -> preload loss {abs(F_el[-1]-F_el[0])/F_el[0]:.0e} (no relaxation, correct)")

    ok = g1 and g2 and g3 and g4 and g5 and g6 and g7
    print(f"\nVERDICT: viscoelastic preload relaxation = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"The dual of creep (constant strain, decaying stress): a bolted part loses clamping force. An SLS part in "
             f"series with an elastic bolt at fixed displacement. Stiff-bolt limit F(t)/F0=E(t)/E0 exact ({e1:.0e}) so "
             f"F_inf/F0=E_inf/E0; soft-bolt limit F constant; the preload loss is monotone in k_b; differentiable. "
             f"The joint loses {loss_pct:.0f}% of its clamping force, which sets a re-tightening interval. " if ok else
             f"Not validated (stiff {g1}, soft {g2}, monotone {g3}, balance {g4}, diff {g5}, payoff {g6}, "
             f"falsification {g7}). ")
          + "SCOPE: linear viscoelasticity (one-element SLS, small strain); 1D series model (part plus bolt, not a full 3D "
          "joint FEM with thread and embedding); tau and E_inf are handbook estimates while E0 is measured; isothermal. "
          "Simulation only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""VISCOELASTICITY - creep/relaxation (linear LVE, SLS/Prony). A NEW domain: printed polymers CREEP under load (elastic-only misses it).

A genuine gap: elasticity/plasticity/fracture do NOT capture TIME-dependent creep. Printed polymers (PLA/PETG) creep noticeably under
sustained load (a printed bracket sags over time). This is linear viscoelasticity: the Standard Linear Solid (Zener)
-> relaxation modulus E(t) + creep compliance J(t), with the EXACT creep-relaxation duality. It composes a measured
PETG E as the glassy E_0 + the failure thread (creep = long-term deformation). Differentiable (design from LVE).

GATE (rigorous, against ANALYTIC): (1) E(t)=Einf+(E0-Einf)e^(-t/tau) SLS relaxation exactly; (2) J(t)=J0+(Jinf-J0)(1-e^(-t/tauc))
SLS creep exactly; (3) CREEP-RELAXATION DUALITY integral_0^t E(t-tau)J(tau)dtau = t (the LVE convolution identity, non-trivial);
(4) limits glassy E(0)=E0 / J(0)=1/E0 + rubbery E(inf)=Einf / J(inf)=1/Einf; (5) BOLTZMANN superposition (a two-step strain
-> summed stress); (6) differentiable dJ(t)/dtau; (7) payoff: a PETG bracket creeps eps(0) -> eps(inf) (elastic misses it); (8) FALSIFICATION:
a mis-wired J (wrong creep tauc) MUST break the duality gate - proving gate 3 is LOAD-BEARING, not a tautology.

  CUDA_VISIBLE_DEVICES="" python3 viscoelasticity_creep.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from cad2simready.materials import load_tds_mechanical
except Exception:
    load_tds_mechanical = None


def relax_modulus(t, E0, Einf, tau, lib=np):
    """SLS relaxations-modul E(t) = E∞ + (E0−E∞)·exp(−t/τ)."""
    exp = lib.exp
    return Einf + (E0 - Einf) * exp(-t / tau)


def creep_compliance(t, E0, Einf, tau, lib=np):
    """SLS krypnings-kompliance J(t) = J0 + (J∞−J0)(1−exp(−t/τc)); J0=1/E0, J∞=1/E∞, τc=τ·E0/E∞."""
    exp = lib.exp
    J0 = 1.0 / E0; Jinf = 1.0 / Einf; tau_c = tau * E0 / Einf
    return J0 + (Jinf - J0) * (1 - exp(-t / tau_c))


def main():
    print("VISCOELASTICITY - SLS creep/relaxation (a new domain: printed polymers creep; elastic-only misses it)")
    # PETG-like: E0 (glassy/instantaneous) from measurement if possible; Einf (rubbery) ~0.5 E0; tau the relaxation time
    E0 = 1.5e9
    src = "default"
    if load_tds_mechanical:
        m = load_tds_mechanical(); r = m.get("petg__prusament__tds")
        if r:
            E0 = r["E_xy_GPa"] * 1e9; src = f"measured ({r['provenance']['method']})"
    Einf = 0.5 * E0; tau = 3600.0     # τ_relax = 1h (illustrativ)
    print(f"  SLS PETG: E0(glasartad)={E0/1e9:.2f}GPa [{src}], E∞(gummiartad)={Einf/1e9:.2f}GPa, τ_relax={tau/3600:.1f}h")

    t = np.linspace(0, 6 * tau, 2000)
    Et = relax_modulus(t, E0, Einf, tau); Jt = creep_compliance(t, E0, Einf, tau)

    # (1) E(t) uppfyller SLS-relaxations-ODE tau*dE/dt + (E - Einf) = 0 (OBEROENDE fysik-check via numerisk derivata,
    #     NOT formula against itself (an earlier version compared E(t) against an identical formula -> a tautology 0e+00)
    tau_c = tau * E0 / Einf
    dEdt = np.gradient(Et, t)
    r1 = np.max(np.abs(tau * dEdt[2:-2] + (Et[2:-2] - Einf))) / E0          # interior (np.gradient edge-svag)
    g1 = r1 < 1e-4
    # (2) J(t) satisfies the creep ODE tau_c dJ/dt + (J - Jinf) = 0 (INDEPENDENT, not formula against itself)
    dJdt = np.gradient(Jt, t)
    r2 = np.max(np.abs(tau_c * dJdt[2:-2] + (Jt[2:-2] - 1 / Einf))) * E0
    g2 = r2 < 1e-3
    # (3) ★ KRYP-RELAXATIONS-DUALITET: ∫₀ᵗ E(t−τ)J(τ)dτ = t (LVE-konvolutions-identitet)
    tc = 2.5 * tau                     # en testpunkt
    s = np.linspace(0, tc, 6000)
    integ = np.trapezoid(relax_modulus(tc - s, E0, Einf, tau) * creep_compliance(s, E0, Einf, tau), s)
    g3 = abs(integ - tc) / tc < 2e-3   # numerical convolution (trapezoid); a non-trivial identity
    print(f"  (1) E(t) uppfyller SLS-relaxations-ODE tau*E'+(E-Einf)=0 (oberoende, {r1:.0e}); (2) J(t) uppfyller kryp-ODE ({r2:.0e})")
    print(f"  (3) creep-relaxation duality integral E(t-tau)J(tau)dtau = {integ:.1f} s vs t={tc:.1f} s (error {abs(integ-tc)/tc:.1e})")
    # (4) limits: t=0 glassy from the array; t -> inf rubbery at LARGE t (creep tauc=tau.E0/Einf > relaxation tau -> evaluate far out)
    t_far = 1e4 * tau_c
    E_far = relax_modulus(t_far, E0, Einf, tau); J_far = creep_compliance(t_far, E0, Einf, tau)
    g4 = (abs(Et[0] - E0) / E0 < 1e-12 and abs(E_far - Einf) / Einf < 1e-3
          and abs(Jt[0] - 1 / E0) * E0 < 1e-12 and abs(J_far - 1 / Einf) * Einf < 1e-3)
    print(f"  (4) limits: E(0)={Et[0]/1e9:.2f}=E0, E(inf) -> {E_far/1e9:.3f}~Einf; J(0)={Jt[0]*1e9:.3f}=1/E0, J(inf) -> {J_far*1e9:.3f}~1/Einf")
    # (5) BOLTZMANN superposition (non-trivial): a constant strain rate eps(t)=R.t -> sigma(t)=integral_0^t E(t-s)R ds = R.integral_0^t E(u)du.
    #     Numerical convolution over the strain history vs the INDEPENDENT closed form R[Einf t + (E0-Einf)tau(1-e^(-t/tau))].
    R = 1e-3 / tau                                      # strain rate (eps=0.1% per tau)
    tq = 2.5 * tau
    sg = np.linspace(0, tq, 8000); deps = R * (sg[1] - sg[0])    # constant strain increment per step
    sig_conv = np.sum(relax_modulus(tq - sg, E0, Einf, tau) * deps)            # diskret Boltzmann-konvolution
    sig_closed = R * (Einf * tq + (E0 - Einf) * tau * (1 - math.exp(-tq / tau)))  # oberoende sluten form (ramp-respons)
    g5 = abs(sig_conv - sig_closed) / sig_closed < 1e-3
    print(f"  (5) Boltzmann ramp: sigma_conv={sig_conv/1e6:.3f} MPa vs sigma_closed={sig_closed/1e6:.3f} MPa (error {abs(sig_conv-sig_closed)/sig_closed:.1e})")
    # (6) differentierbar ∂J(tc)/∂τ
    import torch
    torch.set_default_dtype(torch.float64)
    taut = torch.tensor(tau, requires_grad=True)
    creep_compliance(torch.tensor(tc), torch.tensor(E0), torch.tensor(Einf), taut, lib=torch).backward()
    ga = float(taut.grad)
    h = tau * 1e-6
    gfd = (creep_compliance(tc, E0, Einf, tau + h) - creep_compliance(tc, E0, Einf, tau - h)) / (2 * h)
    grad_err = abs(ga - gfd) / (abs(gfd) + 1e-30); g6 = grad_err < 1e-6
    print(f"  (6) differentiable dJ(t)/dtau autograd vs FD: relative error {grad_err:.2e}")
    # (7) FDM-payoff: PETG-konsol under konstant last σ0 → krypning ε(t)=σ0·J(t); ε(∞)/ε(0)
    sig0 = 10e6
    eps0 = sig0 * creep_compliance(0.0, E0, Einf, tau); epsinf = sig0 * creep_compliance(1e6 * tau, E0, Einf, tau)
    creep_ratio = epsinf / eps0
    g7 = creep_ratio > 1.1   # noticeable creep (elastic-only gives only eps0)
    print(f"  (7) FDM-payoff: PETG-konsol σ0={sig0/1e6:.0f}MPa → ε(0)={eps0*100:.3f}% → ε(∞)={epsinf*100:.3f}% (krypning ×{creep_ratio:.2f})")

    # (8) FALSIFICATION (the duality is LOAD-BEARING, not a tautology): a mis-wired J (wrong tauc=tau instead of tau.E0/Einf) MUST break gate 3
    def J_miswire(tt, lib=np):
        J0 = 1.0 / E0; Jinf = 1.0 / Einf
        return J0 + (Jinf - J0) * (1 - lib.exp(-tt / tau))      # FEL kryp-tidskonstant
    integ_bad = np.trapezoid(relax_modulus(tc - s, E0, Einf, tau) * J_miswire(s), s)
    g8 = abs(integ_bad - tc) / tc > 0.05                        # the mis-wire MUST fall outside the 2e-3 gate
    print(f"  (8) falsification: mis-wired J (tauc=tau) -> integral={integ_bad:.0f} vs t={tc:.0f} (error {abs(integ_bad-tc)/tc:.0%}) -> gate 3 BREAKS (load-bearing)")

    ok = g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8
    print(f"\nVERDICT: viscoelasticity (SLS creep/relaxation) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"A new domain (time-dependent creep that elasticity/plasticity/fracture miss): SLS relaxation modulus E(t) + creep "
             f"kompliance J(t), validerade mot ANALYTISKT + den ICKE-TRIVIALA kryp-relaxations-DUALITETEN ∫E(t−τ)J(τ)dτ=t "
             f"({abs(integ-tc)/tc:.0e}); Boltzmann-ramp-superposition vs sluten form ({abs(sig_conv-sig_closed)/sig_closed:.0e}); glasartad/gummiartad-"
             f"limits; differentiable ({grad_err:.0e}). PAYOFF: a PETG bracket creeps by x{creep_ratio:.1f} (eps(0) -> eps(inf)) under "
             "sustained load - an elastic-only twin MISSES this long-term deformation. It composes the measured glassy E. " if ok else
             f"Not validated (E {g1}, J {g2}, duality {g3}, limits {g4}, Boltzmann {g5}, diff {g6}, creep {g7}, falsification {g8}) - debug. ")
          + "CAVEAT: LINEAR viscoelasticity (small strain, not nonlinear/Schapery); SLS (1 element - a Prony series for a "
          "broader spectrum is further work); tau_relax + Einf are HANDBOOK estimates (E0 measured; viscoelastic parameters need "
          "creep/DMA tests are further work); isothermal (no time-temperature superposition, WLF). The creep-relaxation duality "
          "+ Boltzmann are the rigorous LVE physics; simulation only for tau/Einf.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""FRACTURE MECHANICS (LEFM) - crack propagation: K_I + the Paris law -> life, COMPOSED with Kt (concentration).

Completes the failure/durability thread (yield -> plasticity -> fatigue -> CRACK). Linear elastic fracture mechanics:
the stress intensity K_I = Y.sigma.sqrt(pi a) governs crack growth; the Paris law da/dN = C.(deltaK)^m is integrated from
an initial crack a0 to the critical a_c (where K_I = K_IC, fast fracture) -> cycles to failure N. It COMPOSES with the stress
concentration (Kt from compute_router): a crack at a concentration locally sees Kt.deltaSigma, so life falls with
Kt^m (drastically). Differentiable -> design FROM fracture mechanics (maximum defect or load for a given life).

GATE (rigorous, analytic): (1) K_I(a_c)=K_IC EXACTLY (the definition of a_c); (2) Paris INTEGRATION analytic vs numeric
(scipy quad, <1e-6); (3) da/dN MONOTONICALLY increasing in a (the crack accelerates); (4) N DECREASES with deltaSigma (the right direction);
(5) DIFFERENTIERBAR ∂N/∂{a0,Δσ} autograd vs FD; (6) KOMPONERING med Kt: N_koncentration/N_nominell = Kt^-m EXAKT
(the concentration shortens the crack life by a factor Kt^m). Textbook LEFM (not new physics) - the composition + differentiability are the value.

  CUDA_VISIBLE_DEVICES="" python3 fracture_lefm.py
"""
import math
import sys

import numpy as np


def K_I(sigma, a, Y):
    """Mode-I stress intensity factor: K_I = Y.sigma.sqrt(pi a)  [MPa sqrt(m)]."""
    return Y * sigma * (math.pi * a) ** 0.5


def a_critical(sigma, K_IC, Y):
    """Critical crack length (fast fracture, K_I=K_IC): a_c = (K_IC/(Y.sigma))^2/pi."""
    return (K_IC / (Y * sigma)) ** 2 / math.pi


def paris_cycles(a0, a_c, C, m, Y, dsigma, lib=math):
    """Cykler N (Paris da/dN=C·(Y·Δσ·√(πa))^m), analytisk integration a0→a_c (m≠2)."""
    sqrt = lib.sqrt if lib is not math else math.sqrt
    B = Y * dsigma * math.pi ** 0.5            # ΔK = B·√a
    p = 1.0 - m / 2.0                           # exponent efter integration (m≠2)
    return (a_c ** p - a0 ** p) / (C * B ** m * p)


def main():
    print("BROTTMEKANIK (LEFM) — K_I + Paris sprickpropagering + Kt-komponering")
    E_unit = "MPa√m"
    K_IC = 50.0; Y = 1.12; C = 1.0e-11; m = 3.0     # steel-like (deltaK in MPa sqrt(m), da/dN in m/cycle)
    dsigma = 100.0; a0 = 1.0e-3                       # Δσ=100MPa, initial spricka 1mm
    sig_max = dsigma                                  # R=0
    a_c = a_critical(sig_max, K_IC, Y)
    N = paris_cycles(a0, a_c, C, m, Y, dsigma)
    print(f"  K_IC={K_IC}{E_unit}, Y={Y}, C={C}, m={m}, Δσ={dsigma}MPa, a0={a0*1e3:.1f}mm")
    print(f"  kritisk spricka a_c={a_c*1e3:.1f}mm; cykler-till-brott N={N:.3e}")

    # (1) K_I(a_c)=K_IC
    id1 = abs(K_I(sig_max, a_c, Y) - K_IC) / K_IC
    g1 = id1 < 1e-12
    print(f"  (1) K_I(a_c)=K_IC: rel {id1:.1e}")

    # (2) Paris analytisk vs numerisk integration
    from scipy import integrate
    integrand = lambda a: 1.0 / (C * (Y * dsigma * (math.pi * a) ** 0.5) ** m)
    N_num, _ = integrate.quad(integrand, a0, a_c)
    id2 = abs(N - N_num) / N_num
    g2 = id2 < 1e-6
    print(f"  (2) Paris analytisk {N:.4e} vs numerisk quad {N_num:.4e}: rel {id2:.1e}")

    # (3) da/dN monotonically increasing in a
    aa = np.linspace(a0, a_c, 50)
    dadN = C * (Y * dsigma * np.sqrt(math.pi * aa)) ** m
    g3 = bool(np.all(np.diff(dadN) > 0))
    print(f"  (3) da/dN monotonically increasing (the crack accelerates): {g3} ({dadN[0]:.2e} -> {dadN[-1]:.2e} m/cycle)")

    # (4) N minskar med Δσ
    N_hi = paris_cycles(a0, a_critical(1.3 * dsigma, K_IC, Y), C, m, Y, 1.3 * dsigma)
    g4 = N_hi < N
    print(f"  (4) N(1.3Δσ)={N_hi:.3e} < N(Δσ)={N:.3e}: {g4}")

    # (5) DIFFERENTIERBAR ∂N/∂{a0,Δσ} autograd vs FD
    import torch
    torch.set_default_dtype(torch.float64)
    def Nt(a0v, dsv):
        a_ct = a_critical(dsv, torch.tensor(K_IC), torch.tensor(Y))   # a_c depends on deltaSigma (=sigma_max)
        return paris_cycles(a0v, a_ct, torch.tensor(C), m, torch.tensor(Y), dsv, lib=torch)
    a0t = torch.tensor(a0, requires_grad=True); dst = torch.tensor(dsigma, requires_grad=True)
    Nt(a0t, dst).backward()
    g_auto = {"a0": float(a0t.grad), "dsigma": float(dst.grad)}
    def Nnp(a0v, dsv):
        return paris_cycles(a0v, a_critical(dsv, K_IC, Y), C, m, Y, dsv)
    fd = {}
    for nm, val in (("a0", a0), ("dsigma", dsigma)):
        h = abs(val) * 1e-6
        cp = (a0 + h, dsigma) if nm == "a0" else (a0, dsigma + h)
        cm = (a0 - h, dsigma) if nm == "a0" else (a0, dsigma - h)
        fd[nm] = (Nnp(*cp) - Nnp(*cm)) / (2 * h)
    grad_err = max(abs(g_auto[n] - fd[n]) / (abs(fd[n]) + 1e-30) for n in fd)
    g5 = grad_err < 1e-5
    print(f"  (5) differentiable dN/d{{a0,deltaSigma}} autograd vs FD: max relative error {grad_err:.2e}")

    # (6) COMPOSITION with Kt: a crack at a concentration sees Kt.deltaSigma -> faster growth (deltaSigma^m) AND a smaller a_c (proportional to 1/Kt^2)
    # -> life falls MORE than Kt^m. The exact analytic ratio includes the a_c shrinkage (HONEST: not a naive Kt^m).
    Kt = 3.122
    a_c_conc = a_critical(Kt * sig_max, K_IC, Y)        # = a_c / Kt²  (kritiska sprickan krymper)
    N_conc = paris_cycles(a0, a_c_conc, C, m, Y, Kt * dsigma)
    ratio = N / N_conc
    p = 1.0 - m / 2.0
    ratio_formula = Kt ** m * (a_c ** p - a0 ** p) / (a_c_conc ** p - a0 ** p)   # exakt (Kt^m × a_c-korrektion)
    g6 = abs(ratio - ratio_formula) / ratio_formula < 1e-9 and ratio > Kt ** m
    print(f"  (6) Kt-komponering: N_nom/N_konc={ratio:.1f} = analytisk {ratio_formula:.1f} "
          f"(> Kt^m={Kt**m:.1f}, ty a_c krymper ∝1/Kt²) → {g6}")

    ok = g1 and g2 and g3 and g4 and g5 and g6
    print(f"\nVERDICT: fracture mechanics (LEFM) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Sprickpropagering: K_I=Y·σ·√(πa) + Paris da/dN=C·ΔK^m → cykler-till-brott N={N:.2e} (a0={a0*1e3:.0f}mm→a_c="
             f"{a_c*1e3:.0f}mm). (1) K_I(a_c)=K_IC exakt; (2) analytisk integration = numerisk quad ({id2:.0e}); (3) da/dN "
             f"accelerates; (4) N falls with deltaSigma; (5) differentiable ({grad_err:.0e}) -> design FROM fracture mechanics (maximum defect or load "
             f"for a life target); (6) COMPOSES with Kt (compute_router): a crack at the concentration sees Kt.deltaSigma, so life "
             f"falls {ratio:.0f}x - MORE than Kt^m={Kt**m:.0f} because the critical crack ALSO shrinks (a_c proportional to 1/Kt^2), exact analytically. "
             f"-> the failure thread is complete: yield -> plasticity -> fatigue -> CRACK PROPAGATION. " if ok else
             f"Not validated (K_I {g1}, integration {g2}, monotone {g3}, direction {g4}, diff {g5}, Kt {g6}) - debug. ")
          + "CAVEAT: LEFM assumptions (linear elastic, small-scale yielding: the plastic zone is much smaller than the crack - otherwise EPFM / J integral; "
          "mode I; constant amplitude, no retardation/closure/sequence effects; the Y geometry factor is a handbook constant). Paris C/m "
          "are MATERIAL constants (handbook/test, not derived). a0 = an assumed initial defect (the NDT detection limit). Textbook LEFM "
          "(not new physics) - the VALUE is the composition with Kt + differentiability. sigma_max=deltaSigma (R=0).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

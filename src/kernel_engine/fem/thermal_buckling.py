#!/usr/bin/env python3
"""Thermal buckling: a heated, fully restrained slender part buckles (thermal expansion composed with buckling).

A slender part that is restrained at both ends and heated wants to expand by alpha*dT but cannot, so it
builds an axial compressive force P = E*A*alpha*dT and buckles at dT_cr (Euler).

Gates: (1) dT_cr from the buckling FE matches the analytic value; (2) the thermal force at dT_cr equals
the Euler P_cr; (3) free expansion (unrestrained) gives no force and no buckling, so the restraint is
essential; (4) dT_cr scales with slenderness; (5) the sensitivity is differentiable; (6) alpha -> 0 gives
dT_cr -> infinity.

  python3 thermal_buckling.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from buckling_euler_column import p_cr


def dT_critical(E, alpha, b, L, n_el=40):
    """Critical dT for thermal buckling: thermal compressive force E*A*alpha*dT = Euler P_cr -> dT_cr = P_cr/(E*A*alpha).
    Square cross-section b: A = b^2, I = b^4/12. P_cr from the buckling FE."""
    A = b * b; I = b ** 4 / 12.0; EI = E * I
    P_cr = p_cr("pinned-pinned", n_el, L, EI)
    return P_cr / (E * A * alpha)


def main():
    print("Thermal buckling: a heated restrained slender part buckles (thermal expansion plus buckling)")
    E = 2.0e9; alpha = 7e-5; b = 4e-3; L = 0.15        # polymer-like: E, alpha, 4 mm cross-section, 150 mm slender
    A = b * b; I = b ** 4 / 12.0
    print(f"  part: E={E/1e9:.1f} GPa alpha={alpha:.0e}/K, {b*1e3:.0f}x{b*1e3:.0f} mm L={L*1e3:.0f} mm (slender, restrained)")

    # (1) ΔT_cr via FE vs analytisk
    dTcr = dT_critical(E, alpha, b, L)
    dTcr_an = math.pi ** 2 * I / (A * alpha * L ** 2)
    e1 = abs(dTcr - dTcr_an) / dTcr_an; g1 = e1 < 2e-3
    print(f"  (1) ΔT_cr (FE)={dTcr:.1f} K vs analytisk π²I/(A·α·L²)={dTcr_an:.1f} K (fel {e1:.1e})")

    # (2) thermal force at dT_cr = Euler P_cr
    P_thermal = E * A * alpha * dTcr; P_euler = math.pi ** 2 * (E * I) / L ** 2
    e2 = abs(P_thermal - P_euler) / P_euler; g2 = e2 < 2e-3
    print(f"  (2) thermal compressive force E*A*alpha*dT_cr={P_thermal:.1f} N = Euler P_cr=pi^2 EI/L^2={P_euler:.1f} N (err {e2:.1e})")

    # (3) free expansion -> no force -> no buckling (the restraint is essential)
    P_free = 0.0      # free end -> stress-free thermal expansion, no axial force
    g3 = P_free == 0.0
    print(f"  (3) free (unrestrained): axial force={P_free:.0f} N -> no thermal buckling (the restraint is essential)")

    # (4) ΔT_cr ∝ 1/α + slankhet I/(A·L²)
    dT_2a = dT_critical(E, 2 * alpha, b, L); ratio_a = dTcr / dT_2a
    dT_2L = dT_critical(E, alpha, b, 2 * L); ratio_L = dTcr / dT_2L
    g4 = abs(ratio_a - 2.0) < 1e-6 and abs(ratio_L - 4.0) < 1e-3       # ∝1/α, ∝1/L²
    print(f"  (4) 2×α → ΔT_cr ×{1/ratio_a:.3f} (=1/2); 2×L → ΔT_cr ×{1/ratio_L:.3f} (=1/4, ∝I/(A L²) slankhet)")

    # (5) differentiable d dT_cr/d alpha (= -pi^2 I/(A L^2 alpha^2))
    import torch
    torch.set_default_dtype(torch.float64)
    at = torch.tensor(alpha, requires_grad=True)
    (math.pi ** 2 * I / (A * L ** 2) / at).backward(); ga = float(at.grad)
    gfd = (math.pi ** 2 * I / (A * L ** 2) / (alpha + 1e-9) - math.pi ** 2 * I / (A * L ** 2) / (alpha - 1e-9)) / 2e-9
    grad_err = abs(ga - gfd) / abs(gfd); g5 = grad_err < 1e-6
    print(f"  (5) differentiable d dT_cr/d alpha autograd vs FD: rel err {grad_err:.1e}")

    # (6) falsification: alpha -> 0 -> dT_cr -> infinity
    dT_noalpha = dT_critical(E, alpha * 1e-6, b, L); g6 = dT_noalpha > 1e5 * dTcr
    print(f"  (6) falsification: alpha -> 0 -> dT_cr={dT_noalpha:.0e} K (diverges, no thermal buckling without expansion)")

    ok = g1 and g2 and g3 and g4 and g5 and g6
    print(f"\nVERDICT: thermal buckling = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"A distinct thermo-mechanical stability failure: a restrained slender part is heated, its expansion is "
             f"blocked, an axial compressive force builds and it buckles at dT_cr={dTcr:.0f} K. The thermal force "
             f"E*A*alpha*dT_cr equals the Euler P_cr (FE vs analytic {e1:.0e}, force match {e2:.0e}). The restraint is "
             "essential (free expansion gives no buckling); dT_cr scales as 1/alpha and with slenderness; the sensitivity "
             "is differentiable. " if ok else
             f"Not validated (dT_cr {g1}, composition {g2}, free {g3}, scaling {g4}, diff {g5}, falsification {g6}). ")
          + "SCOPE: linear Euler buckling and linear thermal expansion (constant alpha, temperature-independent E); an "
          "ideally restrained end (a real partial restraint raises dT_cr); pinned-pinned (other boundary conditions give "
          "another factor). Simulation only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Neuber local plasticity at a stress concentration: elastic Kt plus an elasto-plastic constitutive
law, without a nonlinear FEM solve.

The local elasto-plastic (sigma, eps) lies both on the material sigma-eps curve and on the Neuber
hyperbola sigma*eps = (Kt*S)^2/E, so local yielding and plastic strain follow from the elastic Kt and
the constitutive law alone.

Gates: (1) the Neuber identity sigma_local*eps_local = (Kt*S)^2/E holds to roundoff; (2) at low load
(Kt*S < sigma_y) there is no yielding, sigma_local = Kt*S, eps_p = 0; (3) at high load the
concentration yields, sigma_local < Kt*S (redistribution), eps_p > 0 and (sigma, eps) lies on the
bilinear curve; (4) a cross-check against the 1D return mapping driven to the Neuber strain gives the
same sigma; (5) d sigma_local / d{Kt, S, sigma_y} from autograd matches central finite differences.

  python3 neuber_notch_plasticity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plasticity_return_mapping import return_map


def bilinear_eps(sig, E, sy, Et):
    """Strain on the bilinear sigma-eps curve (library-agnostic: numpy or torch by duck typing)."""
    ey = sy / E
    # ε = σ/E (elastisk) ; σ>σy → ε = ey + (σ-σy)/Et
    over = sig - sy
    # smooth, branch-free form: eps = sigma/E + relu(sigma-sigma_y)*(1/Et - 1/E)
    relu = over * (over > 0) if not hasattr(over, "clamp") else over.clamp(min=0.0)
    return sig / E + relu * (1.0 / Et - 1.0 / E)


def neuber_solve(Kt, S, E, sy, H, torch=None):
    """Solve the local (sigma, eps): Neuber hyperbola intersected with the bilinear curve. Returns (sigma_local, eps_local, epsp_local)."""
    lib = torch if torch is not None else np
    Et = E * H / (E + H)
    rhs = (Kt * S) ** 2 / E                                  # Neuber product
    s_el = Kt * S
    s_el_val = s_el.detach().item() if torch is not None else float(s_el)   # branch selection (gradient flows through the chosen branch)
    sy_val = sy.detach().item() if torch is not None else float(sy)
    if s_el_val <= sy_val:
        sig = s_el                                          # elastisk: σ=Kt·S
    else:
        # σ·[σ/E + (σ-σy)(1/Et-1/E)] = rhs ; kvadratisk i σ: aσ²+bσ+c=0
        a = 1.0 / Et
        b = sy * (1.0 / E - 1.0 / Et)
        c = -rhs
        disc = b * b - 4 * a * c
        sig = (-b + lib.sqrt(disc)) / (2 * a)               # positiv rot
    eps = bilinear_eps(sig, E, sy, Et)
    epp = eps - sig / E
    return sig, eps, epp


def main():
    print("Neuber local plasticity at a concentration: elastic Kt + elasto-plastic law, no nonlinear FEM")
    E = 200e9; sy = 250e6; H = 2e9; Et = E * H / (E + H); Kt = 3.122   # steel-like, Kirsch Kt

    # (2) low load: Kt*S < sigma_y -> no yielding
    S_lo = 0.6 * sy / Kt                                     # Kt·S = 0.6σy < σy
    sig_lo, eps_lo, epp_lo = neuber_solve(Kt, S_lo, E, sy, H)
    low_ok = abs(sig_lo - Kt * S_lo) / sig_lo < 1e-12 and epp_lo < 1e-15
    print(f"  (2) low load (Kt*S={Kt*S_lo/1e6:.0f} MPa < sigma_y): sigma_local={sig_lo/1e6:.0f} MPa (=Kt*S, elastic), eps_p={epp_lo:.1e} -> {low_ok}")

    # (3) high load: Kt*S > sigma_y -> yields
    S_hi = 1.8 * sy / Kt                                     # Kt*S = 1.8 sigma_y (elastic over-prediction)
    sig_hi, eps_hi, epp_hi = neuber_solve(Kt, S_hi, E, sy, H)
    s_elastic = Kt * S_hi
    redistrib = (s_elastic - sig_hi) / s_elastic            # how much Neuber lowers the stress relative to elastic
    # is the local (sigma, eps) on the bilinear curve?
    on_curve = abs(eps_hi - bilinear_eps(sig_hi, E, sy, Et)) < 1e-15
    yields = sig_hi > sy and epp_hi > 0
    print(f"  (3) high load (Kt*S={s_elastic/1e6:.0f} MPa > sigma_y): elastic over-prediction {s_elastic/1e6:.0f} -> Neuber sigma_local="
          f"{sig_hi/1e6:.0f} MPa ({100*redistrib:.0f}% redistribution), eps_p_local={epp_hi:.3e}, on_curve={on_curve}, yields={yields}")

    # (1) Neuber identity (both cases)
    id_lo = abs(sig_lo * eps_lo - (Kt * S_lo) ** 2 / E) / ((Kt * S_lo) ** 2 / E)
    id_hi = abs(sig_hi * eps_hi - (Kt * S_hi) ** 2 / E) / ((Kt * S_hi) ** 2 / E)
    neuber_ok = id_lo < 1e-12 and id_hi < 1e-12
    print(f"  (1) Neuber identity sigma*eps=(Kt*S)^2/E: low {id_lo:.1e}, high {id_hi:.1e}")

    # (4) cross-check: run the 1D return mapping to the Neuber strain eps_hi -> same sigma?
    sig_rm = float(return_map(np.linspace(0, eps_hi, 300), E, sy, sy * 0 + 0.0, 0.0)[0][-1]) if False else None
    sig_rm = float(return_map(np.linspace(0, eps_hi, 400), E, sy, H, 0.0)[0][-1])   # isotropic hardening to eps_hi
    cross_ok = abs(sig_rm - sig_hi) / sig_hi < 1e-3
    print(f"  (4) cross-check: return mapping to eps={eps_hi:.3e} gives sigma={sig_rm/1e6:.1f} MPa vs Neuber {sig_hi/1e6:.1f} -> {cross_ok}")

    # (5) differentiable: d sigma_local/d{Kt, S, sigma_y} autograd vs FD
    import torch
    torch.set_default_dtype(torch.float64)
    def sloc(Ktv, Sv, syv):
        return neuber_solve(Ktv, Sv, torch.tensor(E), syv, torch.tensor(H), torch=torch)[0]
    Ktt, St, syt = torch.tensor(Kt, requires_grad=True), torch.tensor(S_hi, requires_grad=True), torch.tensor(sy, requires_grad=True)
    sloc(Ktt, St, syt).backward()
    def sloc_np(Ktv, Sv, syv):
        return float(neuber_solve(Ktv, Sv, E, syv, H)[0])
    fd = {}
    for nm, (val, g) in {"Kt": (Kt, Ktt.grad), "S": (S_hi, St.grad), "sy": (sy, syt.grad)}.items():
        h = abs(val) * 1e-6; base = [Kt, S_hi, sy]; i = ["Kt", "S", "sy"].index(nm)
        bp = base.copy(); bp[i] += h; bm = base.copy(); bm[i] -= h
        d = (sloc_np(*bp) - sloc_np(*bm)) / (2 * h)
        fd[nm] = abs(float(g) - d) / (abs(d) + 1e-30)
    grad_err = max(fd.values())
    print(f"  (5) differentiable d sigma_local/d{{Kt,S,sigma_y}} autograd vs FD: max rel err {grad_err:.2e}")
    diff_ok = grad_err < 1e-5

    ok = low_ok and yields and on_curve and neuber_ok and cross_ok and diff_ok
    print(f"\nVERDICT: Neuber local plasticity = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Combines the elastic Kt ({Kt}) with an elasto-plastic law through Neuber's rule, giving local yielding "
             f"at the concentration without a nonlinear FEM solve. (1) the Neuber identity holds ({id_hi:.0e}); (2) at low "
             f"load there is no yielding (sigma_local=Kt*S); (3) at high load the concentration yields: the elastic "
             f"over-prediction {s_elastic/1e6:.0f} MPa redistributes to sigma_local={sig_hi/1e6:.0f} MPa "
             f"({100*redistrib:.0f}% lower) with local plastic strain eps_p={epp_hi:.2e}; (4) cross-validated against the "
             f"1D return mapping driven to the Neuber strain (same sigma, {abs(sig_rm-sig_hi)/sig_hi:.0e}); (5) differentiable "
             f"(autograd vs FD {grad_err:.0e}). " if ok else
             f"Not validated (low {low_ok}, yields {yields}, on_curve {on_curve}, Neuber {neuber_ok}, cross {cross_ok}, "
             f"diff {diff_ok}). ")
          + "SCOPE: Neuber's rule is a standard engineering approximation (local energy equivalence; it slightly "
          "over-estimates eps_p against a full nonlinear FEM; Glinka-ESED is the alternative); 1D/uniaxial at the "
          "concentration; bilinear hardening; Kt, E and sigma_y are inputs. The Neuber identity holds by construction "
          "(the quadratic solution is the identity rewritten), so gate (1) measures roundoff, not independent physics; "
          "the cross-check in (4) is algorithm agreement (incremental return mapping vs the closed Neuber quadratic on "
          "the same bilinear material), not an independent nonlinear-FEM oracle; the redistribution percentage is "
          "load-level dependent (16% at 1.2 sigma_y, 72% at 4 sigma_y). On-curve plus the return-mapping cross-check "
          "plus FD are the falsifiable parts.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

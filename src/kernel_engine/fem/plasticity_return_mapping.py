#!/usr/bin/env python3
"""Differentiable 1D von Mises (J2) elasto-plastic return mapping: yielding and permanent deformation.

Textbook radial-return constitutive law with combined isotropic and kinematic hardening, differentiable
through the return mapping (autograd).

Gates: (1) the monotonic sigma-eps curve reproduces the analytic bilinear curve (elastic E, then tangent
modulus E_t = E*H/(E+H)); (2) under plastic flow the stress lies on the yield surface
|sigma - alpha| = sigma_y + H_iso*q; (3) unloading is elastic with slope E and leaves permanent strain
eps_p; (4) with kinematic hardening the reverse yield happens at a lower |sigma| than the peak
(Bauschinger), separating kinematic from isotropic hardening; (5) d(permanent strain)/d{E, sigma_y, H}
from autograd matches central finite differences.

  python3 plasticity_return_mapping.py
"""
import sys

import numpy as np


def return_map(strain_hist, E, sy, H_iso, H_kin, torch=None):
    """1D radial-return elasto-plastik. Returnerar (σ-historik, εp-slut, q-slut, on_surface-max-residual).
    torch=None -> numpy; otherwise torch (differentiable)."""
    lib = torch if torch is not None else np
    absf = (lambda x: lib.abs(x)) if torch is not None else np.abs
    sign = (lambda x: lib.sign(x)) if torch is not None else np.sign
    ep = (torch.zeros(()) if torch is not None else 0.0)        # plastic strain
    al = (torch.zeros(()) if torch is not None else 0.0)        # back-stress (kinematisk)
    q = (torch.zeros(()) if torch is not None else 0.0)         # accumulated plastic strain (isotropic)
    sig_out = []; surf_res = 0.0
    for eps in strain_hist:
        sig_tr = E * (eps - ep)                                 # trial stress
        eta = sig_tr - al                                       # relative stress
        f = absf(eta) - (sy + H_iso * q)                        # flytfunktion
        plastic = (float(f) > 0.0) if torch is None else (f.item() > 0.0)
        if plastic:
            dgam = f / (E + H_iso + H_kin)                      # plastisk multiplikator (1D)
            sig = sig_tr - E * dgam * sign(eta)
            ep = ep + dgam * sign(eta)
            al = al + H_kin * dgam * sign(eta)
            q = q + dgam
            if torch is None:                                  # konsistens-residual bara i numpy (undvik grad-float)
                surf_res = max(surf_res, abs(float(absf(sig - al) - (sy + H_iso * q))))
        else:
            sig = sig_tr
        sig_out.append(sig)
    return sig_out, ep, q, surf_res


def main():
    print("Plasticity: 1D J2 radial return (isotropic + kinematic hardening), differentiable")
    E = 200e9; sy = 250e6; H_iso = 2e9; H_kin = 0.0       # steel-like, isotropic hardening for gates (1)-(3)
    ey = sy / E                                            # yield strain

    # (1) monotonic bilinear: load to 4x the yield strain
    eps_mono = np.linspace(0, 4 * ey, 200)
    sig, ep_f, q_f, sres = return_map(eps_mono, E, sy, H_iso, H_kin)
    sig = np.array([float(s) for s in sig])
    Et = E * H_iso / (E + H_iso)                           # analytisk tangent-modul efter flyt
    sig_analytic = np.where(eps_mono <= ey, E * eps_mono, sy + Et * (eps_mono - ey))
    mono_err = np.max(np.abs(sig - sig_analytic)) / sy
    print(f"  (1) monotonic bilinear: sigma vs analytic (elastic E, plastic E_t={Et/1e9:.3f} GPa) max rel err {mono_err:.2e}")
    mono_ok = mono_err < 1e-9

    # (2) consistency (on the yield surface during flow)
    print(f"  (2) consistency on the yield surface |sigma-alpha|=sigma_y+H*q: max residual {sres/sy:.2e}")
    consist_ok = sres / sy < 1e-9

    # (3) unloading: load to 4 eps_y, unload to 2 eps_y (stays elastic; unloading to 0 would yield in compression,
    # the elastic range is only 2 sigma_y wide). Elastic slope E plus permanent strain by extrapolating to sigma=0.
    eps_unload = np.concatenate([np.linspace(0, 4 * ey, 100), np.linspace(4 * ey, 2 * ey, 100)])
    su, ep_u, _, _ = return_map(eps_unload, E, sy, H_iso, H_kin)
    su = np.array([float(s) for s in su])
    unload_slope = (su[120] - su[101]) / (eps_unload[120] - eps_unload[101])   # tidig avlastning (elastisk)
    eps_perm = eps_unload[-1] - su[-1] / E                 # extrapolate to sigma=0 (= permanent strain eps_p)
    slope_ok = abs(unload_slope - E) / E < 1e-9
    perm_ok = abs(eps_perm - float(ep_u)) < 1e-9
    print(f"  (3) unloading: elastic slope {unload_slope/1e9:.1f} GPa (=E {E/1e9:.0f}), permanent strain {eps_perm:.4e} "
          f"(=εp {float(ep_u):.4e}) → slope_ok={slope_ok}, perm_ok={perm_ok}")

    # (4) Bauschinger: kinematic (H_kin=H) vs isotropic (H_iso=H), cyclic load +3 eps_y -> -3 eps_y
    H = 2e9
    eps_cyc = np.concatenate([np.linspace(0, 3 * ey, 150), np.linspace(3 * ey, -3 * ey, 300)])
    sig_kin, _, _, _ = return_map(eps_cyc, E, sy, 0.0, H)      # ren kinematisk
    sig_iso, _, _, _ = return_map(eps_cyc, E, sy, H, 0.0)      # ren isotrop
    sig_kin = np.array([float(s) for s in sig_kin]); sig_iso = np.array([float(s) for s in sig_iso])
    peak = sig_kin[149]                                        # stress at load reversal
    # reverse-yield detection: the first step in the reverse branch where plastic flow resumes (the curve bends)
    def reverse_yield_stress(s):
        rev = s[150:]                                          # reverse-grenen
        # elastic unloading has slope E; yielding when |dsigma/deps| drops. Find the knee.
        de = np.diff(eps_cyc[150:]); ds = np.diff(rev); slope = ds / de
        ky = np.where(slope > 0.5 * E)[0]                      # plastiskt (lutning << E)... slope<0.5E = plastiskt
        idx = np.where(slope < 0.5 * E)[0]
        return rev[idx[0]] if len(idx) else None
    ry_kin = reverse_yield_stress(sig_kin); ry_iso = reverse_yield_stress(sig_iso)
    # Analytic: kinematic reverse yield = sigma_peak - 2 sigma_y (the 2 sigma_y wide elastic range, translated);
    # isotropic = -sigma_peak (symmetric, expanded). Bauschinger: kinematic yields at a higher value than isotropic.
    ry_kin_an = peak - 2 * sy
    kin_matches = ry_kin is not None and abs(ry_kin - ry_kin_an) < 0.05 * sy          # detektion = analytisk kinematisk
    bauschinger = kin_matches and ry_iso is not None and ry_kin > ry_iso + 0.02 * sy  # and ordering (kinematic before isotropic)
    print(f"  (4) Bauschinger: reverse yield kinematic {ry_kin/1e6 if ry_kin else 0:.0f} MPa (analytic sigma_peak-2 sigma_y="
          f"{ry_kin_an/1e6:.0f}) vs isotropic {ry_iso/1e6 if ry_iso else 0:.0f} MPa (=-sigma_peak); kinematic matches the analytic value "
          f"({kin_matches}) & flyter tidigare = {bauschinger}")

    # (5) differentiable: d(permanent strain)/d{E, sy, H_iso} autograd vs central FD
    import torch
    torch.set_default_dtype(torch.float64)
    eh = torch.tensor(eps_unload)
    def perm_strain(Ev, syv, Hv):
        _, epv, _, _ = return_map(eh, Ev, syv, Hv, torch.tensor(0.0), torch=torch)
        return epv
    Et_, syt_, Ht_ = torch.tensor(E, requires_grad=True), torch.tensor(sy, requires_grad=True), torch.tensor(H_iso, requires_grad=True)
    perm_strain(Et_, syt_, Ht_).backward()
    def perm_np(Ev, syv, Hv):
        _, epv, _, _ = return_map(eps_unload, Ev, syv, Hv, 0.0)
        return float(epv)
    fd = {}
    for nm, (val, gv) in {"E": (E, Et_.grad), "sy": (sy, syt_.grad), "H_iso": (H_iso, Ht_.grad)}.items():
        h = abs(val) * 1e-6
        base = [E, sy, H_iso]; i = ["E", "sy", "H_iso"].index(nm)
        bp = base.copy(); bp[i] += h; bm = base.copy(); bm[i] -= h
        d = (perm_np(*bp) - perm_np(*bm)) / (2 * h)
        fd[nm] = abs(float(gv) - d) / (abs(d) + 1e-30)
    grad_err = max(fd.values())
    print(f"  (5) differentiable d eps_p/d{{E,sy,H}} autograd vs FD: max rel err {grad_err:.2e} ({ {k:round(v,8) for k,v in fd.items()} })")
    diff_ok = grad_err < 1e-5

    ok = mono_ok and consist_ok and slope_ok and perm_ok and bauschinger and diff_ok
    print(f"\nVERDICT: plasticity (J2 return mapping) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"1D von Mises elasto-plasticity via radial return: (1) sigma-eps reproduces the analytic bilinear curve "
             f"({mono_err:.0e}, tangent E_t=E*H/(E+H)); (2) the stress lies on the yield surface during flow ({sres/sy:.0e}); "
             f"(3) unloading is elastic (slope E) with permanent strain eps_p; (4) Bauschinger captured -- kinematic "
             f"hardening yields in reverse at a lower |sigma| than isotropic, separating the hardening type; (5) "
             f"d eps_p / d{{E, sigma_y, H}} via autograd through the return mapping matches FD ({grad_err:.0e}). " if ok else
             f"Not validated (diff {diff_ok}). ")
          + "SCOPE: 1D (uniaxial); a full 3D J2 tensor return and pressure dependence (Drucker-Prager for polymers) are "
          "not included; rate-independent (no viscoplasticity); bilinear hardening (not Voce or power law). The analytic "
          "bilinear curve, the consistency residual, Bauschinger and FD are the falsifiable parts.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

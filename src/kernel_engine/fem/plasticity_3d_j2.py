#!/usr/bin/env python3
"""3D von Mises (J2) elasto-plasticity: full-tensor radial return (Simo-Hughes) with isotropic hardening.

Elastic predictor, yield function f = ||s_trial|| - sqrt(2/3)(sigma_y + H*alpha), plastic corrector on the
3x3 deviatoric tensor.

Gates: (1) under uniaxial stress (lateral strain bisected per step so sigma22 = sigma33 = 0) the 3D model
reproduces the 1D return mapping exactly; (2) after a plastic step the stress lies on the yield surface
||s|| = sqrt(2/3)(sigma_y + H*alpha); (3) plastic incompressibility tr(eps_p) = 0 holds exactly;
(4) objectivity/isotropy: rotating the strain history by R rotates the stress by R (R sigma R^T);
(5) d sigma11 / d sigma_y from autograd matches finite differences.

  python3 plasticity_3d_j2.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plasticity_return_mapping import return_map

R23 = np.sqrt(2.0 / 3.0)


def single_step(eps, state, K, G, sy, H, lib=np):
    """One radial-return step. eps: 3x3 total strain, state=(eps_p 3x3, alpha) -> (sigma 3x3, new state)."""
    epp_n, alpha_n = state
    I = lib.eye(3) if lib is np else lib.eye(3, dtype=eps.dtype)
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    dev_eps = eps - (tr / 3.0) * I
    s_trial = 2 * G * (dev_eps - epp_n)                       # epp_n deviatorisk → s_trial deviatorisk
    norm_st = lib.sqrt((s_trial * s_trial).sum())
    f = norm_st - R23 * (sy + H * alpha_n)
    f_val = f.detach().item() if lib is not np else float(f)  # branch selection (gradient flows through the chosen branch)
    if f_val <= 0.0:
        s, epp, alpha = s_trial, epp_n, alpha_n              # elastisk
    else:
        dgam = f / (2 * G + (2.0 / 3.0) * H)
        n = s_trial / norm_st
        s = s_trial - 2 * G * dgam * n
        epp = epp_n + dgam * n
        alpha = alpha_n + R23 * dgam
    sigma = K * tr * I + s
    return sigma, (epp, alpha)


def run_history(eps_hist, K, G, sy, H, lib=np):
    """Run a list of strain tensors sequentially -> (stress history, final state)."""
    state = (lib.zeros((3, 3), dtype=eps_hist[0].dtype) if lib is not np else np.zeros((3, 3)), 0.0)
    out = []
    for eps in eps_hist:
        sig, state = single_step(eps, state, K, G, sy, H, lib)
        out.append(sig)
    return out, state


def uniaxial_stress_drive(eps11_path, K, G, sy, H):
    """Drive the 3D model to uniaxial stress: per step, bisect the lateral strain so sigma22=sigma33=0."""
    state = (np.zeros((3, 3)), 0.0)
    sig11 = []
    for e11 in eps11_path:
        lo, hi = -e11, 0.0 if e11 >= 0 else (0.0, -e11)        # lateral contraction between -e11 and 0 (tension)
        lo, hi = (min(-abs(e11), abs(e11)), max(-abs(e11), abs(e11)))
        for _ in range(80):                                   # bisect the lateral strain so sigma22 -> 0 (from the same previous state)
            el = 0.5 * (lo + hi)
            eps = np.diag([e11, el, el])
            sig, _ = single_step(eps, state, K, G, sy, H)
            if sig[1, 1] > 0:
                hi = el
            else:
                lo = el
        el = 0.5 * (lo + hi)
        eps = np.diag([e11, el, el])
        sig, state = single_step(eps, state, K, G, sy, H)     # commit slut-εl
        sig11.append(sig[0, 0])
    return np.array(sig11)


def main():
    print("3D von Mises (J2) radial return: full tensor")
    E = 200e9; nu = 0.3; sy = 250e6; H = 2e9
    G = E / (2 * (1 + nu)); K = E / (3 * (1 - 2 * nu))
    print(f"  E={E/1e9:.0f}GPa ν={nu} σy={sy/1e6:.0f}MPa H={H/1e9:.0f}GPa → G={G/1e9:.1f} K={K/1e9:.1f}GPa")

    # (1) uniaxial stress reproduces the 1D model exactly
    e11 = np.linspace(0, 4 * sy / E, 60)                       # to 4x the yield strain
    sig3d = uniaxial_stress_drive(e11, K, G, sy, H)
    sig1d = return_map(e11, E, sy, H, 0.0)[0]                  # 1D return mapping (same E, sigma_y, H)
    uni_err = np.max(np.abs(sig3d - np.array(sig1d))) / sy
    print(f"  (1) uniaxial sigma11(eps11): 3D vs 1D return_map max rel err {uni_err:.2e} "
          f"(σ_slut 3D {sig3d[-1]/1e6:.2f} vs 1D {float(sig1d[-1])/1e6:.2f} MPa)")

    # (2)+(3) consistency and incompressibility on a multiaxial plastic history
    rng = np.random.default_rng(0)
    base = np.array([[3e-3, 1.5e-3, 0.5e-3], [1.5e-3, -1e-3, 0.8e-3], [0.5e-3, 0.8e-3, -0.5e-3]])  # multiaxiell, plastisk
    hist = [t * base for t in np.linspace(0, 1, 50)]
    sigs, (epp_f, alpha_f) = run_history(hist, K, G, sy, H)
    s_f = sigs[-1] - (np.trace(sigs[-1]) / 3) * np.eye(3)
    cons = abs(np.sqrt((s_f * s_f).sum()) - R23 * (sy + H * alpha_f)) / (R23 * sy)
    incomp = abs(np.trace(epp_f))
    plastic = alpha_f > 0
    print(f"  (2) konsistens (multiaxiell): ‖s‖−√⅔(σy+Hα) rel {cons:.2e} (α={alpha_f:.3e}, plastisk={plastic})")
    print(f"  (3) plastisk inkompressibilitet tr(εp)={incomp:.2e}")

    # (4) objectivity: rotating the whole history by R must rotate sigma by R
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    hist_rot = [Rz @ e @ Rz.T for e in hist]
    sigs_rot, _ = run_history(hist_rot, K, G, sy, H)
    obj_err = max(np.max(np.abs(sr - Rz @ s @ Rz.T)) for s, sr in zip(sigs, sigs_rot)) / sy
    print(f"  (4) objektivitet (rot {th}rad): max‖σ_rot − R·σ·Rᵀ‖/σy = {obj_err:.2e}")

    # (5) differentiable d sigma11_final/d sigma_y (uniaxial strain path, avoiding the bisection) autograd vs FD
    import torch
    torch.set_default_dtype(torch.float64)
    eps_t = [torch.tensor(np.diag([v, -nu * v, -nu * v])) for v in e11]   # approximately uniaxial strain path (differentiable, not the sigma22=0 solve)
    def s11_final(syv):
        sg, _ = run_history(eps_t, torch.tensor(K), torch.tensor(G), syv, torch.tensor(H), lib=torch)
        return sg[-1][0, 0]
    syt = torch.tensor(sy, requires_grad=True)
    s11_final(syt).backward()
    g_auto = float(syt.grad)
    h = sy * 1e-6
    def s11_np(syv):
        sg, _ = run_history([np.diag([v, -nu * v, -nu * v]) for v in e11], K, G, syv, H)
        return sg[-1][0, 0]
    g_fd = (s11_np(sy + h) - s11_np(sy - h)) / (2 * h)
    grad_err = abs(g_auto - g_fd) / (abs(g_fd) + 1e-30)
    print(f"  (5) differentiable d sigma11/d sigma_y autograd vs FD: rel err {grad_err:.2e}")

    uni_ok = uni_err < 1e-6; cons_ok = cons < 1e-12; incomp_ok = incomp < 1e-14
    obj_ok = obj_err < 1e-10; diff_ok = grad_err < 1e-5
    ok = uni_ok and cons_ok and incomp_ok and plastic and obj_ok and diff_ok
    print(f"\nVERDICT: 3D J2 plasticity = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Full-tensor von Mises radial return (Simo-Hughes). (1) under uniaxial stress (lateral solve "
             f"sigma22=sigma33=0) the 1D model is reproduced exactly (rel err {uni_err:.0e}); (2) on a multiaxial plastic "
             f"history the stress lies on the yield surface ({cons:.0e}); (3) plastic flow is incompressible "
             f"tr(eps_p)={incomp:.0e}; (4) the model is objective (sigma rotates with the strain, {obj_err:.0e}); "
             f"(5) differentiable ({grad_err:.0e}). " if ok else
             f"Not validated (diff {diff_ok}). ")
          + "SCOPE: small strain (additive eps = eps_e + eps_p, not finite deformation); isotropic hardening (kinematic "
          "hardening is the 1D script); rate-independent; J2 (pressure-independent yield, not Drucker-Prager); implicit "
          "Euler radial return. Gate (5) uses an approximately uniaxial strain path, so it tests the differentiability of "
          "the integrator rather than the uniaxial-stress solve. Exact-vs-1D, yield-surface consistency, incompressibility "
          "and objectivity are the falsifiable parts.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""TRANSVERSELY ISOTROPIC 3D ELASTICITY - a printing-correct constitutive law (addresses the disclosed 'isotropic Kt on an anisotropic part' caveat).

A genuine gap: fem3d_elasticity has ONLY D_iso (isotropic), and the anisotropy twin used a BEAM approximation (sqrt of the E ratio),
NOT a 3D orthotropic constitutive law in a stress field. But printed parts are genuinely TRANSVERSELY ISOTROPIC: the layer plane (xy) is
stiffer than the build direction (z, weaker layer adhesion). The yield/anisotropy twins FLAGGED
'isotropic Kt on an anisotropic part' as a limitation. This builds the correct constitutive law: 5 independent constants
{E_p, E_z, nu_p, nu_pz, G_zp} -> orthotropic D -> injected into the 3D FEM. It composes a MEASURED E_xy/E_z ratio.

GATE (rigorous, against ANALYTIC): (1) ISOTROPIC LIMIT: D_transiso(E,E,nu,nu,E/2(1+nu)) == D_iso(E,nu) EXACTLY (all axes);
(2) D symmetric + POSITIVE DEFINITE (a valid material); (3) FEM DIRECTIONAL PAYOFF: the same material loaded in-plane vs
the build direction give EXACTLY the effective moduli E_p and E_z respectively (uniaxial recovery u_x=t.L/E_x; orientation changes the 3D part stiffness,
not just a beam); (4) SHEAR INDEPENDENCE: G_zp independent of E_p, nu_p (build shear is not in-plane shear, unlike the isotropic case);
(5) PAYOFF: a MEASURED E_xy/E_z -> a directional stiffness ratio in a 3D field (beyond the beam approximation); (6) differentiable dD/dE_z
through the matrix inverse (design / system identification from anisotropy).

  CUDA_VISIBLE_DEVICES="" python3 fem3d_orthotropic.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('fem',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fem3d_elasticity import FEM3D, D_iso, hex_KE
try:
    from cad2simready.materials import load_tds_mechanical
except Exception:
    load_tds_mechanical = None

# Voigt-ordning [xx,yy,zz,xy,yz,zx] (matchar fem3d_elasticity.hex_KE / B-matrisen)
_SHEAR_AXES = {3: {0, 1}, 4: {1, 2}, 5: {2, 0}}   # xy, yz, zx


def D_transverse_isotropic(E_p, E_z, nu_p, nu_pz, G_zp, axis=2, lib=np):
    """Transversellt isotrop 6×6 D; symmetriaxel=axis (build-riktning). 5 oberoende konstanter.
    E_p in-plan, E_z build, ν_p in-plan-Poisson, ν_pz in-plan-stress→build-strain, G_zp transvers skjuv.
    In-plane shear G_p=E_p/(2(1+nu_p)) is DERIVED. Builds the compliance S -> D=S^-1."""
    G_p = E_p / (2 * (1 + nu_p))
    p = [i for i in (0, 1, 2) if i != axis]              # in-plan normal-index-par
    if lib is np:
        S = np.zeros((6, 6))
    else:
        S = lib.zeros((6, 6), dtype=E_z.dtype if hasattr(E_z, "dtype") else None)
    S[axis, axis] = 1.0 / E_z
    S[p[0], p[0]] = S[p[1], p[1]] = 1.0 / E_p
    S[p[0], p[1]] = S[p[1], p[0]] = -nu_p / E_p
    for q in p:
        S[axis, q] = S[q, axis] = -nu_pz / E_p           # symmetrisk per konstruktion
    for si, ax_set in _SHEAR_AXES.items():
        S[si, si] = (1.0 / G_p) if axis not in ax_set else (1.0 / G_zp)
    D = np.linalg.inv(S) if lib is np else lib.linalg.inv(S)
    # silent-partial-NaN linalg.inv guard, OUTPUT-ONLY (fleet-wide, ; D's find/fix, 10th
    # linalg-primitive site): E_z=0 legitimately gives S[axis,axis]=inf (infinite compliance), which
    # inv() correctly resolves to a FINITE D (0-stiffness) -- an INPUT guard would wrongly reject this
    # legitimate physical case. But a NaN-corrupted E_z/E_p/etc silently produces a partial-NaN D with no
    # exception (measured), corrupting the FEM stiffness matrix. Guard the OUTPUT only.
    if lib is np and not np.isfinite(D).all():
        raise ValueError("D_transverse_isotropic: inverted stiffness matrix D contains non-finite values -- material parameters are likely NaN-corrupted")
    return D


def fem_with_D(nelx, nely, nelz, h, D):
    """FEM3D with an injected constitutive D (reuses all the machinery)."""
    f = FEM3D(nelx=nelx, nely=nely, nelz=nelz, h=h)
    f.D = D; f.KE = hex_KE(h, D); f._assemble()
    return f


def eff_modulus_x(fem, t):
    """Effective x modulus from a uniaxial tensile traction: E_x = t.L / mean(u_x on the +x face)."""
    u, _, _, _ = fem.solve(t)
    ux = np.mean([u[3 * fem.nid(fem.nelx, j, k)] for k in range(fem.nz) for j in range(fem.ny)])
    return t * (fem.nelx * fem.h) / ux


def main():
    print("TRANSVERSELY ISOTROPIC 3D ELASTICITY - a printing-correct constitutive law")
    E_p, E_z, nu_p, nu_pz, G_zp = 3.0e9, 1.8e9, 0.35, 0.30, 0.65e9    # FDM-likt (build svagare)
    nelx, nely, nelz, h, t = 8, 2, 2, 1.0, 1.0e6

    # (1) ISOTROPIC LIMIT: all axes -> D_iso
    E, nu = 2.5e9, 0.33; Diso = D_iso(E, nu)
    g1 = all(np.max(np.abs(D_transverse_isotropic(E, E, nu, nu, E / (2 * (1 + nu)), axis=a) - Diso)) / np.max(np.abs(Diso)) < 1e-12
             for a in (0, 1, 2))
    print(f"  (1) isotropic limit (all 3 axes) -> D_iso: {'EXACT' if g1 else 'WRONG'}")

    # (2) symmetri + positiv-definithet
    D = D_transverse_isotropic(E_p, E_z, nu_p, nu_pz, G_zp, axis=2)
    sym = np.max(np.abs(D - D.T)) / np.max(np.abs(D)); eig = np.linalg.eigvalsh(D); g2 = sym < 1e-9 and eig.min() > 0
    print(f"  (2) D symmetrisk (rel {sym:.0e}), positiv-definit (min egv {eig.min():.2e}>0): {'OK' if g2 else 'FEL'}")

    # (3) ★FEM-RIKTNINGS-PAYOFF: ladda +x med build=z (in-plan) vs build=x (build-riktning) → E_p resp E_z exakt
    fem_inplane = fem_with_D(nelx, nely, nelz, h, D_transverse_isotropic(E_p, E_z, nu_p, nu_pz, G_zp, axis=2))
    fem_build = fem_with_D(nelx, nely, nelz, h, D_transverse_isotropic(E_p, E_z, nu_p, nu_pz, G_zp, axis=0))
    E_eff_inplane = eff_modulus_x(fem_inplane, t); E_eff_build = eff_modulus_x(fem_build, t)
    e_ip = abs(E_eff_inplane - E_p) / E_p; e_bd = abs(E_eff_build - E_z) / E_z
    g3 = e_ip < 1e-6 and e_bd < 1e-6
    print(f"  (3) FEM direction: in-plane E_eff={E_eff_inplane/1e9:.4f} GPa (=E_p, error {e_ip:.0e}), "
          f"build E_eff={E_eff_build/1e9:.4f} GPa (=E_z, error {e_bd:.0e}) -> orientation ratio {E_eff_inplane/E_eff_build:.3f}=E_p/E_z")

    # (4) SHEAR INDEPENDENCE: change G_zp -> the build shear term D[4,4] changes, the in-plane shear D[3,3] is UNCHANGED
    D2 = D_transverse_isotropic(E_p, E_z, nu_p, nu_pz, 2 * G_zp, axis=2)
    d_build = abs(D2[4, 4] - D[4, 4]) / D[4, 4]; d_inplane = abs(D2[3, 3] - D[3, 3]) / D[3, 3]
    g4 = d_build > 0.1 and d_inplane < 1e-12          # build-skjuv styrs av G_zp, in-plan av G_p (oberoende)
    print(f"  (4) shear independence: G_zp x2 -> build shear changes {d_build:.0%}, in-plane shear {d_inplane:.0e} (unchanged)")

    # (5) PAYOFF: a MEASURED E_xy/E_z -> a directional ratio in a 3D field
    a6 = "—"
    if load_tds_mechanical:
        mech = load_tds_mechanical() or {}
        cand = {k: r for k, r in mech.items() if r.get("E_z_GPa") and r.get("E_xy_GPa")}
        if cand:
            k0 = sorted(cand)[0]; r = cand[k0]; Exy = r["E_xy_GPa"] * 1e9; Ez = r["E_z_GPa"] * 1e9
            Dm_ip = fem_with_D(nelx, nely, nelz, h, D_transverse_isotropic(Exy, Ez, nu_p, nu_pz, G_zp, axis=2))
            Dm_bd = fem_with_D(nelx, nely, nelz, h, D_transverse_isotropic(Exy, Ez, nu_p, nu_pz, G_zp, axis=0))
            r_ip = eff_modulus_x(Dm_ip, t); r_bd = eff_modulus_x(Dm_bd, t)
            a6 = (f"{k0}: E_xy={r['E_xy_GPa']}GPa E_z={r['E_z_GPa']}GPa (measured={r['provenance']['measured']}) → "
                  f"3D field directional stiffness ratio {r_ip/r_bd:.2f} (in-plane / build)")
    print(f"  (5) A6-payoff: {a6}")

    # (6) differentierbar ∂D[2,2]/∂E_z genom matris-invers
    import torch
    torch.set_default_dtype(torch.float64)
    Ezt = torch.tensor(E_z, requires_grad=True)
    Dt = D_transverse_isotropic(torch.tensor(E_p), Ezt, torch.tensor(nu_p), torch.tensor(nu_pz), torch.tensor(G_zp), axis=2, lib=torch)
    Dt[2, 2].backward(); ga = float(Ezt.grad)
    hh = E_z * 1e-6
    gfd = (D_transverse_isotropic(E_p, E_z + hh, nu_p, nu_pz, G_zp)[2, 2]
           - D_transverse_isotropic(E_p, E_z - hh, nu_p, nu_pz, G_zp)[2, 2]) / (2 * hh)
    grad_err = abs(ga - gfd) / (abs(gfd) + 1e-30); g6 = grad_err < 1e-6
    print(f"  (6) differentiable dD[2,2]/dE_z through the inverse: relative error {grad_err:.2e}")

    ok = g1 and g2 and g3 and g4 and g6
    print(f"\nVERDICT: transversely isotropic 3D elasticity = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"A printing-correct constitutive law (5 independent constants) addressing the disclosed 'isotropic Kt on an anisotropic part' "
             "limitation: the isotropic limit is EXACT (all axes), D is symmetric + positive definite, and the FEM DIRECTIONAL PAYOFF: the same "
             f"material gives EXACTLY E_p in-plane / E_z in build (uniaxial recovery {max(e_ip,e_bd):.0e}), so print orientation changes "
             "3D part stiffness in a STRESS FIELD (beyond the beam approximation); shear independence (G_zp is not "
             f"isotropically determined); differentiable dD/dE_z ({grad_err:.0e}). " if ok else
             f"Not validated (iso {g1}, PD {g2}, FEM direction {g3}, shear {g4}, diff {g6}) - debug. ")
          + "CAVEAT: LINEAR elastic transverse isotropy (5 constants); E_p/E_z from measurement but nu_p/nu_pz/G_zp are handbook "
          "estimates (a shear/Poisson test is further work); SMALL strain; Q1 hex (coarse concentrations). The MEASURED "
          "part is the E_xy/E_z ratio; the full 5-constant tensor needs more measurement. Simulation only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

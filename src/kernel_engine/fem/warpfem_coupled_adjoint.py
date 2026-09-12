#!/usr/bin/env python3
"""warp.fem COUPLED ADJOINT - DIFFERENTIABLE MULTIPHYSICS (design through coupled physics).

The distinctive claim: differentiate an objective THROUGH a COUPLED multiphysics problem. Here a STRUCTURAL objective
J=u_y(target)^2 is differentiated with respect to the design rho, where rho controls the CONDUCTIVITY k(rho)=rho^p.k0 -> temperature field T(rho) ->
thermal load -> displacement u(rho) -> J. The sensitivity runs PURELY through the thermo-mechanical coupling (rho is
NOT in the structural matrix) -> it isolates the differentiability of the coupling.

CHAINED ADJOINT (derived): with thermal A_T(rho)T=b_T and structural A_u.u = C.T + d (C = coupling):
  1. strukturell adjoint  A_u λ_u = ∂J/∂u
  2. termisk adjoint-last  g_T = Cᵀλ_u  (= ∫ β'·div(λ_u)·ψ_j, β'=Eα/(1−ν))
  3. thermal adjoint  A_T(rho) lambda_T = g_T   (homogeneous Dirichlet: T's boundary DOF are fixed -> lambda_T=0 there)
  4. dJ/dρ_e = −p·ρ^(p−1)·∫_e k0·∇λ_T·∇T   (termisk kors-energi λ_T×T per cell)
GATE: FD (perturb rho_e, re-solve thermal + structural, recompute J) vs the chained adjoint per element.
Uses the analytically derived beta'=E alpha/(1-nu) (the load in thermoelastic.py was MISSING alpha, masked by u~0). Note
(audit): the FD gate validates adjoint-forward SELF-CONSISTENCY (differentiability), NOT the truth of the beta' magnitude
- a wrong beta' would pass self-consistently. beta' is correct on paper, not independently FD-validated.
Validated on design-significant cells (tiny ones are below float32 FD resolution).

  python3 warpfem_coupled_adjoint.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

E = 70.0e9
NU = 0.33
ALPHA = 23.0e-6
DELTA_T = 200.0
K0 = 1.0
LX, LY = 2.0, 1.0
NX, NY = 16, 8
SIMP_P = 3.0
FD_EPS = 5e-3
FD_TOL = 2.5e-2


@fem.integrand
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def simp_thermal_form(s: fem.Sample, T: fem.Field, w: fem.Field, rho: fem.Field, k0: float, p: float):
    return wp.pow(rho(s), p) * k0 * wp.dot(fem.grad(T, s), fem.grad(w, s))


@fem.integrand
def elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def thermal_load_form(s: fem.Sample, v: fem.Field, T: fem.Field, betap: float, tref: float):
    return betap * (T(s) - tref) * wp.trace(fem.D(v, s))


@fem.integrand
def coupling_adjoint_form(s: fem.Sample, lam_u: fem.Field, w: fem.Field, betap: float):
    # g_T_j = (Cᵀλ_u)_j = ∫ β'·div(λ_u)·ψ_j
    return betap * wp.trace(fem.D(lam_u, s)) * w(s)


@fem.integrand
def thermal_cross_form(s: fem.Sample, lam_T: fem.Field, T: fem.Field, wc: fem.Field, k0: float):
    return wc(s) * k0 * wp.dot(fem.grad(lam_T, s), fem.grad(T, s))


@fem.integrand
def scalar_proj(s: fem.Sample, T: fem.Field, w: fem.Field):
    return T(s) * w(s)


@fem.integrand
def vec_proj(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def T_bc_value(s: fem.Sample, w: fem.Field, val: float):
    return val * w(s)


@fem.integrand
def classify(s: fem.Sample, domain: fem.Domain, left: wp.array(dtype=int), right: wp.array(dtype=int)):
    nor = fem.normal(domain, s)
    if nor[0] < -0.5:
        left[s.qp_index] = 1
    if nor[0] > 0.5:
        right[s.qp_index] = 1


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    print(f"warp.fem KOPPLAD ADJOINT (differentierbar multifysik) — {NX}x{NY}, ΔT={DELTA_T}K, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    T_space = fem.make_polynomial_space(geo, degree=1, dtype=float)
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)
    domain = fem.Cells(geo)
    boundary = fem.BoundarySides(geo)
    mu = E / (2.0 * (1.0 + NU)); lam_e = E * NU / (1.0 - NU * NU); lame = wp.vec2(lam_e, mu)
    betap = E * ALPHA / (1.0 - NU)      # KORREKT termisk-stress-koeff (med α)
    tref = 0.0

    Tt = fem.make_test(T_space, domain=domain); Ttr = fem.make_trial(T_space, domain=domain)
    ut = fem.make_test(u_space, domain=domain); utr = fem.make_trial(u_space, domain=domain)
    cell_test = fem.make_test(rho_space, domain=domain)
    rho = fem.make_discrete_field(rho_space); rho.dof_values.fill_(1.0)

    # boundary classification: left (hot/clamped), right (cold)
    lm = wp.zeros(boundary.element_count(), dtype=int); rm = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify, at=boundary, values={"left": lm, "right": rm})
    left = fem.Subdomain(boundary, element_mask=lm); right = fem.Subdomain(boundary, element_mask=rm)
    lrm = wp.zeros(boundary.element_count(), dtype=int)
    wp.launch(_or_kernel, dim=lrm.shape[0], inputs=[lm, rm, lrm])
    lr = fem.Subdomain(boundary, element_mask=lrm)

    # --- thermal Dirichlet projector (left union right), values left=deltaT right=0 ---
    Tlr_test = fem.make_test(T_space, domain=lr); Tlr_trial = fem.make_trial(T_space, domain=lr)
    Tproj = fem.integrate(scalar_proj, fields={"T": Tlr_trial, "w": Tlr_test}, assembly="nodal")
    Tl_test = fem.make_test(T_space, domain=left)
    Trhs_dir = fem.integrate(T_bc_value, fields={"w": Tl_test}, values={"val": DELTA_T}, assembly="nodal")

    # --- structural Dirichlet: left edge u=0 (homogeneous, shared by u and lambda_u) ---
    ul_test = fem.make_test(u_space, domain=left); ul_trial = fem.make_trial(u_space, domain=left)
    bd_u = fem.integrate(vec_proj, fields={"u": ul_trial, "v": ul_test}, assembly="nodal", output_dtype=float)
    fem.normalize_dirichlet_projector(bd_u)

    # the structural matrix is constant (rho is NOT in the structure -> the coupling is isolated); assembled in struct_solve.
    nnode_u = u_space.node_count()
    pos_field = fem.make_discrete_field(u_space)
    fem.interpolate(node_pos, dest=pos_field)
    pos = pos_field.dof_values.numpy().reshape(nnode_u, 2)
    tgt = int(np.argmin((pos[:, 0] - 0.6 * LX) ** 2 + (pos[:, 1] - 0.5 * LY) ** 2))
    # J=u_x(target)^2: the thermal body force -beta' grad T is purely x (T linear in x) -> u_x nonzero; u_y=0 by y symmetry.
    print(f"  target node {tgt} @ ({pos[tgt,0]:.2f},{pos[tgt,1]:.2f}) - J=u_x(target)^2 (structural objective via the coupling)")

    T_field = fem.make_discrete_field(T_space)
    u_field = fem.make_discrete_field(u_space)
    lamu_field = fem.make_discrete_field(u_space)
    lamT_field = fem.make_discrete_field(T_space)

    def thermal_primary():
        AT = fem.integrate(simp_thermal_form, fields={"T": Ttr, "w": Tt, "rho": rho},
                           values={"k0": K0, "p": SIMP_P})
        rhsT = wp.zeros(T_space.node_count(), dtype=float)
        fem.project_linear_system(AT, rhsT, Tproj, Trhs_dir)
        T = wp.zeros_like(rhsT)
        fem_example_utils.bsr_cg(AT, b=rhsT, x=T, quiet=True, tol=1e-12, max_iters=4000)
        return T

    def thermal_adjoint(gT):
        AT = fem.integrate(simp_thermal_form, fields={"T": Ttr, "w": Tt, "rho": rho},
                           values={"k0": K0, "p": SIMP_P})
        rr = wp.clone(gT)
        fem.project_linear_system(AT, rr, Tproj)        # homogeneous Dirichlet (lambda_T=0 on the boundary)
        lamT = wp.zeros_like(rr)
        fem_example_utils.bsr_cg(AT, b=rr, x=lamT, quiet=True, tol=1e-12, max_iters=4000)
        return lamT

    def struct_solve(rhs_vec):
        Au = fem.integrate(elasticity_form, fields={"u": utr, "v": ut},
                           values={"lame": lame}, output_dtype=float)
        rr = wp.clone(rhs_vec)
        fem.project_linear_system(Au, rr, bd_u, normalize_projector=False)
        x = wp.zeros_like(rr)
        fem_example_utils.bsr_cg(Au, b=rr, x=x, quiet=True, tol=1e-12, max_iters=4000)
        return x

    def forward():
        T = thermal_primary(); T_field.dof_values = T
        rhs_u = fem.integrate(thermal_load_form, fields={"v": ut, "T": T_field},
                              values={"betap": betap, "tref": tref}, output_dtype=wp.vec2)
        u = struct_solve(rhs_u); u_field.dof_values = u
        ux = float(u.numpy()[tgt, 0])
        return ux * ux, ux

    J0, ux0 = forward()
    print(f"  J0=u_x(target)^2={J0:.4e} (u_x={ux0:.3e}) - u nonzero -> the load path is exercised")

    # structural adjoint: dJ/du = point load 2.u_x(target) on the target X DOF
    g = np.zeros((nnode_u, 2), np.float32); g[tgt, 0] = 2.0 * ux0
    lamu = struct_solve(wp.array(g, dtype=wp.vec2)); lamu_field.dof_values = lamu
    # termisk adjoint-last g_T = Cᵀλ_u, sedan termisk adjoint
    gT = fem.integrate(coupling_adjoint_form, fields={"lam_u": lamu_field, "w": Tt},
                       values={"betap": betap}, output_dtype=float)
    lamT = thermal_adjoint(gT); lamT_field.dof_values = lamT
    # kedjad-adjoint-sensitivitet (termisk kors-energi)
    ce = fem.integrate(thermal_cross_form, fields={"lam_T": lamT_field, "T": T_field, "wc": cell_test},
                       values={"k0": K0}, output_dtype=float).numpy()
    rho_np = rho.dof_values.numpy()
    sens = -SIMP_P * (rho_np ** (SIMP_P - 1.0)) * ce

    cells = list(np.argsort(np.abs(sens))[-5:][::-1])
    rels = []
    for e in cells:
        base = rho_np.copy()
        for sgn in (+1, -1):
            x = base.copy(); x[e] += sgn * FD_EPS; rho.dof_values.assign(x.astype(np.float32))
            J = forward()[0]
            if sgn > 0:
                Jp = J
            else:
                Jm = J
        rho.dof_values.assign(base.astype(np.float32))
        fd = (Jp - Jm) / (2 * FD_EPS)
        rel = abs(sens[e] - fd) / max(abs(fd), 1e-30)
        rels.append(rel)
        print(f"  cell {e:3d}: chained adjoint dJ/drho={sens[e]:+.4e}  FD={fd:+.4e}  relative error {rel:.2e}")

    maxrel = max(rels)
    nontrivial = abs(sens[cells].mean()) > 1e-30
    ok = maxrel < FD_TOL and nontrivial
    print(f"\nVERDICT: differentiable MULTIPHYSICS (coupled adjoint) = {'VALIDATED' if ok else 'NOT VALIDATED'} "
          f"(max rel error {maxrel:.2e} over {len(cells)} cells, tol {FD_TOL}). "
          + ("A structural objective differentiated THROUGH the thermo-elastic coupling (rho -> k -> T -> thermal load -> u -> J) "
             "via kedjad adjoint (strukturell + termisk adjoint), FD-matchad → DIFFERENTIERBAR MULTIFYSIK "
             "demonstrated: design that accounts for coupled physics. HONEST SCOPE "
             ": the FD gate validates that the chained ADJOINT is consistent with the forward "
             "model (differentiability), NOT that the beta' magnitude is physically correct - a wrong beta' "
             "would pass SELF-CONSISTENTLY (the same beta' in the adjoint and in FD). beta'=E alpha/(1-nu) is derived analytically "
             "(correct on paper), not independently FD-validated against truth here. " if ok else
             "The chained adjoint does not match FD - debug BEFORE any multiphysics-differentiability claim. ")
          + "CAVEAT: 2D P1; one-way coupling (T -> structure, no thermo-mechanical feedback); rho only in the conductivity; "
          "one objective example.")
    return 0 if ok else 1


@wp.kernel
def _or_kernel(a: wp.array(dtype=int), b: wp.array(dtype=int), out: wp.array(dtype=int)):
    i = wp.tid()
    if a[i] > 0 or b[i] > 0:
        out[i] = 1


if __name__ == "__main__":
    sys.exit(main())

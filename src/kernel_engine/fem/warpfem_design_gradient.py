#!/usr/bin/env python3
"""warp.fem DESIGN GRADIENT (design FROM physics) - topology-optimisation sensitivity, FD-gated.

Builds on the VALIDATED warp.fem elasticity (warpfem_elasticity_validate.py, equilibrium gate
8.6e-6). This is design-from-physics in its robust form: the classical SIMP topology-
optimisation SENSITIVITY - the gradient of compliance with respect to per-element density. That is
EXACTLY the mechanism generative design / topology optimisation rests on.

SIMP: K(rho) = sum rho_e^p K_e^0. Compliance C = f^T u = u^T K u. ADJOINT sensitivity (self-adjoint for
compliance, analytically known): dC/drho_e = -p.rho_e^(p-1).integral_e sigma0(u):eps(u) dV  (element strain energy x2).
GATE: FD (perturb rho_e, re-solve, deltaC) MUST match the analytic sensitivity per tested
element - otherwise the design gradient is wrong and must not drive generative design. No threshold grazing.

  python3 warpfem_design_gradient.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

E = 70.0e9
NU = 0.33
TRACTION = 1.0e6
LX, LY = 4.0, 1.0
NX, NY = 16, 4
SIMP_P = 3.0
FD_EPS = 5e-3          # balanced for float32 (too small -> roundoff; the optimum is ~(eps_f32)^(1/3) ~ 5e-3)
FD_TOL = 1e-2


@fem.integrand
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def simp_elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, rho: fem.Field, lame: wp.vec2, p: float):
    # ρ^p · σ₀:ε(v)   (SIMP-skalad styvhet)
    return wp.pow(rho(s), p) * wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def vec_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def traction_form(s: fem.Sample, v: fem.Field, t: float):
    return wp.dot(wp.vec2(t, 0.0), v(s))


@fem.integrand
def classify_sides(s: fem.Sample, domain: fem.Domain, left: wp.array(dtype=int), right: wp.array(dtype=int)):
    nor = fem.normal(domain, s)
    if nor[0] < -0.5:
        left[s.qp_index] = 1
    if nor[0] > 0.5:
        right[s.qp_index] = 1


@fem.integrand
def cell_strain_energy_form(s: fem.Sample, u: fem.Field, w: fem.Field, lame: wp.vec2):
    # ∫_e σ₀:ε dV  projicerad per cell (w = per-cell testfunktion, degree 0)
    return w(s) * wp.ddot(fem.D(u, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def compliance_form(s: fem.Sample, u: fem.Field, rho: fem.Field, lame: wp.vec2, p: float):
    return wp.pow(rho(s), p) * wp.ddot(fem.D(u, s), hooke_stress(fem.D(u, s), lame))


def main():
    wp.init()
    print(f"warp.fem DESIGN-GRADIENT (SIMP topologi-opt-sensitivitet) — {NX}x{NY}, p={SIMP_P}, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)   # per-cell densitet
    domain = fem.Cells(geo)
    u_test = fem.make_test(u_space, domain=domain)
    u_trial = fem.make_trial(u_space, domain=domain)

    mu = E / (2.0 * (1.0 + NU))
    lam = E * NU / (1.0 - NU * NU)
    lame = wp.vec2(lam, mu)

    rho = fem.make_discrete_field(rho_space)
    rho.dof_values.fill_(0.6)   # rho != 1: exercises the (p-1) exponent - at rho=1, rho^(p-1)=1 regardless of p

    # rand
    boundary = fem.BoundarySides(geo)
    left_mask = wp.zeros(boundary.element_count(), dtype=int)
    right_mask = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify_sides, at=boundary, values={"left": left_mask, "right": right_mask})
    left = fem.Subdomain(boundary, element_mask=left_mask)
    right = fem.Subdomain(boundary, element_mask=right_mask)
    right_test = fem.make_test(u_space, domain=right)
    left_test = fem.make_test(u_space, domain=left)
    left_trial = fem.make_trial(u_space, domain=left)
    bd_proj = fem.integrate(vec_projector_form, fields={"u": left_trial, "v": left_test},
                            assembly="nodal", output_dtype=float)
    fem.normalize_dirichlet_projector(bd_proj)

    u_field = fem.make_discrete_field(u_space)

    def solve_and_compliance():
        K = fem.integrate(simp_elasticity_form, fields={"u": u_trial, "v": u_test, "rho": rho},
                          values={"lame": lame, "p": SIMP_P}, output_dtype=float)
        rhs = fem.integrate(traction_form, fields={"v": right_test}, values={"t": TRACTION}, output_dtype=wp.vec2)
        fem.project_linear_system(K, rhs, bd_proj, normalize_projector=False)
        u = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(K, b=rhs, x=u, quiet=True, tol=1e-11, max_iters=3000)
        u_field.dof_values = u
        C = fem.integrate(compliance_form, fields={"u": u_field, "rho": rho},
                          values={"lame": lame, "p": SIMP_P}, domain=domain)
        return float(C)

    C0 = solve_and_compliance()
    # analytisk per-cell-sensitivitet: dC/dρ_e = −p·ρ_e^(p−1)·∫_e σ₀:ε
    cell_test = fem.make_test(rho_space, domain=domain)
    cell_energy = fem.integrate(cell_strain_energy_form, fields={"u": u_field, "w": cell_test},
                                values={"lame": lame}, output_dtype=float)
    ce = cell_energy.numpy()
    rho_np = rho.dof_values.numpy()
    sens_analytic = -SIMP_P * (rho_np ** (SIMP_P - 1.0)) * ce
    print(f"  compliance C0={C0:.6e} J, {len(ce)} celler")

    # FD gate on a selection of cells
    test_cells = [0, len(ce) // 3, len(ce) // 2, 2 * len(ce) // 3, len(ce) - 1]
    rels = []
    for e in test_cells:
        base = rho_np.copy()
        for sgn in (+1, -1):
            x = base.copy(); x[e] += sgn * FD_EPS
            rho.dof_values.assign(x)
            if sgn > 0:
                Cp = solve_and_compliance()
            else:
                Cm = solve_and_compliance()
        rho.dof_values.assign(base)
        fd = (Cp - Cm) / (2 * FD_EPS)
        rel = abs(sens_analytic[e] - fd) / max(abs(fd), 1e-30)
        rels.append(rel)
        print(f"  cell {e:3d}: analytic dC/drho={sens_analytic[e]:+.4e}  FD={fd:+.4e}  relative error {rel:.2e}")

    maxrel = max(rels)
    nontrivial = abs(sens_analytic[test_cells].mean()) > 1e-30
    ok = maxrel < FD_TOL and nontrivial
    print(f"\nVERDICT: warp.fem DESIGN GRADIENT (topology-optimisation sensitivity) = {'VALIDATED' if ok else 'NOT VALIDATED'} "
          f"(max rel error {maxrel:.2e} over {len(test_cells)} cells, tol {FD_TOL}). "
          + ("The ADJOINT sensitivity that generative design / topology optimisation rests on is now an FD-gated capability "
             "on the validated warp.fem FEM, so the design-FROM-physics gradient is genuine, not a library trick. "
             "NEXT: a topology-optimisation loop (Adam on rho) + warp.fem AD for an ARBITRARY objective (not only compliance). "
             if ok else "The sensitivity does not match FD - debug BEFORE any design claim. ")
          + "CAVEAT: 2D plane-stress P1; compliance objective (self-adjoint); AD through the solve for an arbitrary objective is next.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

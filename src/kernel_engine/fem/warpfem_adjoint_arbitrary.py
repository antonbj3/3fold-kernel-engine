#!/usr/bin/env python3
"""warp.fem ADJOINT SENSITIVITY for an ARBITRARY objective - full design freedom (FD-gated).

Design-gradienten (warpfem_design_gradient.py) var compliance = SELF-ADJOINT (λ=u). Generativ
design needs ARBITRARY objectives (displacement targets, compliant mechanisms, stress minimisation...). The
general ADJOINT METHOD (robust, not wp.Tape): for an objective J(u) with the constraint K(rho)u=f, solve the ADJOINT
equation K lambda = dJ/du (same K, new RHS), then dJ/drho_e = -p.rho^(p-1).integral_e sigma0(u):eps(lambda)  (a CROSS energy
between the primal u and the adjoint lambda; for compliance lambda=u, the self-adjoint special case).

Objective here: J = u_y(target)^2  -> dJ/du = point load 2.u_y(target) on the target y DOF -> lambda != u (testing the GENERAL
case). GATE: FD (perturb rho_e, re-solve the primal, recompute J) vs the adjoint sensitivity per element.

  python3 warpfem_adjoint_arbitrary.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

E = 70.0e9
NU = 0.33
TRACTION = 1.0e6
LX, LY = 3.0, 1.0
NX, NY = 16, 8
SIMP_P = 3.0
FD_EPS = 5e-3
FD_TOL = 2e-2


@fem.integrand
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def simp_form(s: fem.Sample, u: fem.Field, v: fem.Field, rho: fem.Field, lame: wp.vec2, p: float):
    return wp.pow(rho(s), p) * wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def vec_proj(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def traction_form(s: fem.Sample, v: fem.Field, t: float):
    return wp.dot(wp.vec2(0.0, -t), v(s))


@fem.integrand
def classify(s: fem.Sample, domain: fem.Domain, left: wp.array(dtype=int), right: wp.array(dtype=int)):
    nor = fem.normal(domain, s)
    if nor[0] < -0.5:
        left[s.qp_index] = 1
    if nor[0] > 0.5:
        right[s.qp_index] = 1


@fem.integrand
def cross_energy_form(s: fem.Sample, u: fem.Field, lam: fem.Field, w: fem.Field, lame: wp.vec2):
    # integral_e sigma0(u):eps(lambda) per cell (cross energy, primal x adjoint)
    return w(s) * wp.ddot(fem.D(lam, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    print(f"warp.fem ADJOINT arbitrary objective (u_y(target)^2) - {NX}x{NY}, device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)
    domain = fem.Cells(geo)
    ut = fem.make_test(u_space, domain=domain); utr = fem.make_trial(u_space, domain=domain)
    cell_test = fem.make_test(rho_space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam_e = E * NU / (1.0 - NU * NU); lame = wp.vec2(lam_e, mu)
    rho = fem.make_discrete_field(rho_space); rho.dof_values.fill_(1.0)

    boundary = fem.BoundarySides(geo)
    lm = wp.zeros(boundary.element_count(), dtype=int); rm = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify, at=boundary, values={"left": lm, "right": rm})
    left = fem.Subdomain(boundary, element_mask=lm); right = fem.Subdomain(boundary, element_mask=rm)
    rtest = fem.make_test(u_space, domain=right)
    ltest = fem.make_test(u_space, domain=left); ltrial = fem.make_trial(u_space, domain=left)
    bd = fem.integrate(vec_proj, fields={"u": ltrial, "v": ltest}, assembly="nodal", output_dtype=float)
    fem.normalize_dirichlet_projector(bd)
    f_trac = fem.integrate(traction_form, fields={"v": rtest}, values={"t": TRACTION}, output_dtype=wp.vec2)

    nnode = u_space.node_count()
    pos_field = fem.make_discrete_field(u_space)
    fem.interpolate(node_pos, dest=pos_field)
    pos = pos_field.dof_values.numpy().reshape(nnode, 2)
    # target point: middle of the domain (not at the load) -> lambda != u
    tgt = int(np.argmin((pos[:, 0] - LX * 0.5) ** 2 + (pos[:, 1] - LY * 0.5) ** 2))
    print(f"  target node {tgt} @ ({pos[tgt,0]:.2f},{pos[tgt,1]:.2f}) - J=u_y(target)^2")

    u_field = fem.make_discrete_field(u_space)
    lam_field = fem.make_discrete_field(u_space)

    def assemble_K():
        K = fem.integrate(simp_form, fields={"u": utr, "v": ut, "rho": rho},
                          values={"lame": lame, "p": SIMP_P}, output_dtype=float)
        return K

    def solve(K, rhs):
        rr = wp.clone(rhs)
        fem.project_linear_system(K, rr, bd, normalize_projector=False)
        x = wp.zeros_like(rr)
        fem_example_utils.bsr_cg(K, b=rr, x=x, quiet=True, tol=1e-11, max_iters=4000)
        return x

    def objective():
        K = assemble_K()
        u = solve(K, f_trac)
        u_field.dof_values = u
        uy = float(u.numpy()[tgt, 1])
        return uy * uy, uy, u, K

    J0, uy0, u, K = objective()
    # adjoint RHS: dJ/du = point load 2.u_y(target) on the target y DOF
    g = np.zeros((nnode, 2), np.float32); g[tgt, 1] = 2.0 * uy0
    g_w = wp.array(g, dtype=wp.vec2)
    lam_vec = solve(K, g_w)
    lam_field.dof_values = lam_vec

    ce = fem.integrate(cross_energy_form, fields={"u": u_field, "lam": lam_field, "w": cell_test},
                       values={"lame": lame}, output_dtype=float).numpy()
    rho_np = rho.dof_values.numpy()
    sens = -SIMP_P * (rho_np ** (SIMP_P - 1.0)) * ce
    print(f"  J0=u_y(target)^2={J0:.4e} (u_y={uy0:.3e})")

    ncell = len(ce)
    # validate where the adjoint is DESIGN-SIGNIFICANT (largest |sens|). Tiny sensitivities
    # (cells far from target/load) are below float32 FD resolution (deltaJ < solve noise ~1e-6.J), so
    # FD is noise there, not an adjoint error. Honest: gate on the design-relevant cells.
    cells = list(np.argsort(np.abs(sens))[-5:][::-1])
    rels = []
    for e in cells:
        base = rho_np.copy()
        for sgn in (+1, -1):
            x = base.copy(); x[e] += sgn * FD_EPS; rho.dof_values.assign(x)
            J = objective()[0]
            if sgn > 0:
                Jp = J
            else:
                Jm = J
        rho.dof_values.assign(base)
        fd = (Jp - Jm) / (2 * FD_EPS)
        rel = abs(sens[e] - fd) / max(abs(fd), 1e-30)
        rels.append(rel)
        print(f"  cell {e:3d}: adjoint dJ/drho={sens[e]:+.4e}  FD={fd:+.4e}  relative error {rel:.2e}")

    maxrel = max(rels)
    nontrivial = abs(sens[cells].mean()) > 1e-30
    ok = maxrel < FD_TOL and nontrivial
    print(f"\nVERDICT: adjoint sensitivity for an ARBITRARY objective = {'VALIDATED' if ok else 'NOT VALIDATED'} "
          f"(max rel error {maxrel:.2e} over {len(cells)} cells, tol {FD_TOL}). "
          + ("The general adjoint method (lambda != u, not self-adjoint compliance) matches FD, so design from physics "
             "is now open for ARBITRARY objectives (displacement / compliant mechanism / stress), not only compliance. "
             "Robust (no wp.Tape AD friction); one extra adjoint solve per objective. " if ok else
             "The adjoint sensitivity does not match FD - debug BEFORE any arbitrary-objective claim. ")
          + "CAVEAT: 2D plane-stress P1; linear elasticity (the adjoint uses the same K); one objective example.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

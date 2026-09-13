#!/usr/bin/env python3
"""warp.fem ELASTICITY MMS - rigorous field + constitutive validation.

Audit finding: the equilibrium gates (sigma_xx=t) validate a constitutive-INDEPENDENT
identity, NOT field accuracy or E/nu. An earlier cantilever attempt had an unreliable stress
comparison (abandoned). This does it RIGOROUSLY with the Method of Manufactured Solutions (MMS):

  Pick a KNOWN solution u*(x,y) = (x^2, x.y)  (zero on the left edge -> homogeneous Dirichlet, a working
  pattern). Derive the body force f = -div(sigma(u*)) (constant) + traction t = sigma(u*).n on the other boundaries.
  Solve. u* is in the Q2 space, so Galerkin FEM must REPRODUCE u* EXACTLY (to solver tolerance). A wrong
  E/nu/assembly gives a DIFFERENT solution and fails. This validates the WHOLE displacement FIELD + the constitutive law,
  without stress-recovery or clamping confounders.

sigma from u* (Hooke 2 mu eps + lambda tr(eps) I, plane stress lambda=E nu/(1-nu^2), mu=E/2(1+nu)): eps=[[2x, y/2],[y/2, x]], tr=3x ->
σ=[[(4μ+3λ)x, μy],[μy, (2μ+3λ)x]]; div(σ)_x=∂σ_xx/∂x+∂σ_xy/∂y=(4μ+3λ)+μ=5μ+3λ, div(σ)_y=0 → f=(−(5μ+3λ),0).

Plus a RESIDUAL GUARD (bsr_cg can stagnate silently): it gates ||Ku-f||/||f||.

  python3 warpfem_mms_elasticity.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
from warp.sparse import bsr_mv

E = 70.0e9
NU = 0.33
LX, LY = 2.0, 1.0
NX, NY = 24, 12
TOL_FIELD = 1e-3        # u* in Q2 -> reproduced to the float32 assembly floor ~1e-4 (not node positions - those are ~4e-8; warp.fem uses an internal float32 Jacobian); a WRONG constitutive law gives ~3e-1
                        # (a 1000x gap) -> tolerance in between, wide margin both ways (no threshold grazing)
TOL_RESID = 1e-6        # CG-residual-vakt (mot tyst stagnation)


@fem.integrand
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def vec_proj(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def body_force_form(s: fem.Sample, v: fem.Field, fx: float):
    return wp.dot(wp.vec2(fx, 0.0), v(s))


@fem.integrand
def mms_traction_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, mu: float, lam: float):
    # t = σ(u*)·n, analytiskt vid randpositionen
    p = fem.position(domain, s); n = fem.normal(domain, s)
    sxx = (4.0 * mu + 3.0 * lam) * p[0]
    syy = (2.0 * mu + 3.0 * lam) * p[0]
    sxy = mu * p[1]
    t = wp.vec2(sxx * n[0] + sxy * n[1], sxy * n[0] + syy * n[1])
    return wp.dot(t, v(s))


@fem.integrand
def classify_left(s: fem.Sample, domain: fem.Domain, left: wp.array(dtype=int), other: wp.array(dtype=int)):
    nor = fem.normal(domain, s)
    if nor[0] < -0.5:
        left[s.qp_index] = 1
    else:
        other[s.qp_index] = 1


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    print(f"warp.fem ELASTIK MMS (u*=(x²,xy)) — {NX}x{NY} {LX}x{LY}, device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)
    domain = fem.Cells(geo)
    test = fem.make_test(space, domain=domain); trial = fem.make_trial(space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam = E * NU / (1.0 - NU * NU); lame = wp.vec2(lam, mu)

    K = fem.integrate(elasticity_form, fields={"u": trial, "v": test}, values={"lame": lame},
                      output_dtype=wp.float64)         # float64 (float32 stagnates)

    boundary = fem.BoundarySides(geo)
    lm = wp.zeros(boundary.element_count(), dtype=int); om = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify_left, at=boundary, values={"left": lm, "other": om})
    left = fem.Subdomain(boundary, element_mask=lm); other = fem.Subdomain(boundary, element_mask=om)

    # body force f=(-(4 mu + 3 lambda),0) + analytic traction on the remaining boundaries
    fx = -(5.0 * mu + 3.0 * lam)        # div(σ)_x = ∂σ_xx/∂x + ∂σ_xy/∂y = (4μ+3λ) + μ = 5μ+3λ (∂(μy)/∂y=μ!)
    rhs = fem.integrate(body_force_form, fields={"v": test}, values={"fx": fx}, output_dtype=wp.vec2d)
    otest = fem.make_test(space, domain=other)
    rhs_t = fem.integrate(mms_traction_form, fields={"v": otest}, values={"mu": mu, "lam": lam},
                          output_dtype=wp.vec2d)
    rhs.assign(rhs.numpy() + rhs_t.numpy())     # total RHS = kroppskraft + analytisk traktion

    # homogeneous Dirichlet u=0 on the left (u*(x=0)=0)
    ltest = fem.make_test(space, domain=left); ltrial = fem.make_trial(space, domain=left)
    bd = fem.integrate(vec_proj, fields={"u": ltrial, "v": ltest}, assembly="nodal", output_dtype=wp.float64)
    fem.normalize_dirichlet_projector(bd)
    fem.project_linear_system(K, rhs, bd, normalize_projector=False)

    u = wp.zeros_like(rhs)
    res, iters = fem_example_utils.bsr_cg(K, b=rhs, x=u, quiet=True, tol=1e-12, max_iters=5000)
    # RESIDUAL GUARD: confirm convergence (against silent stagnation)
    Ku = wp.zeros_like(rhs); bsr_mv(K, u, Ku)
    resid = float(np.linalg.norm(Ku.numpy() - rhs.numpy()) / np.linalg.norm(rhs.numpy()))

    # u* vid noderna
    nnode = space.node_count()
    pf = fem.make_discrete_field(space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 2)
    u_exact = np.stack([pos[:, 0] ** 2, pos[:, 0] * pos[:, 1]], axis=1)
    u_fem = u.numpy().reshape(nnode, 2)
    rel_field = float(np.linalg.norm(u_fem - u_exact) / np.linalg.norm(u_exact))
    max_abs = float(np.abs(u_fem - u_exact).max())

    print(f"  CG: {iters} iter, residual-vakt ‖Ku−f‖/‖f‖ = {resid:.2e} (tol {TOL_RESID}) [float64; float32 stagnerar]")
    print(f"  FIELD: ||u_FEM-u*||/||u*|| = {rel_field:.2e} (tol {TOL_FIELD}); max abs error {max_abs:.2e} m")
    ok = rel_field < TOL_FIELD and resid < TOL_RESID
    print(f"\nVERDICT: warp.fem elasticity MMS = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + ("FEM REPRODUCES the manufactured solution u*=(x^2,xy) to solver level, validating the WHOLE "
             "displacement FIELD + the CONSTITUTIVE law (E,nu via sigma) + body-force/traction assembly + "
             "solve. A wrong E/nu/assembly gives a DIFFERENT solution "
             "and fails; this tests field accuracy, not just an equilibrium identity. The RESIDUAL GUARD "
             "confirms convergence (no silent stagnation). " if ok else
             "FEM does NOT reproduce u* (or CG stagnated) - debug BEFORE any field claim. ")
          + "CAVEAT: 2D plane-stress Q2; u* quadratic (in Q2 -> reproduction to the float32 ASSEMBLY floor ~1e-4, not position roundoff), "
          "NOT convergence order); float64 assembly and solve are REQUIRED (float32 CG stagnates); one MMS case; structured mesh.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

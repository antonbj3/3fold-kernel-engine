#!/usr/bin/env python3
"""warp.fem 3D ELASTICITY MMS - generalises the physics stack to 3D (real CAD parts are 3D).

Everything so far is 2D plane stress. Real parts are 3D, the single largest generalisation gap.
This proves the validated FEM stack works in FULL 3D, rigorously, via the Method of Manufactured
Solutions (the same method that closed field accuracy in 2D):

  Known solution u*(x,y,z) = (x^2, xy, xz)  (zero at x=0 -> homogeneous Dirichlet); derive the body force
  f = -div(sigma(u*)) + traction t = sigma(u*).n on all other faces. u* in Q2 -> FEM REPRODUCES u* exactly.

σ (3D hooke 2με+λtr(ε)I, λ=Eν/((1+ν)(1−2ν)), μ=E/2(1+ν)): ε=diag(2x,x,x)+offdiag(xy:y/2, xz:z/2),
tr=4x → σ_xx=(4μ+4λ)x, σ_yy=σ_zz=(2μ+4λ)x, σ_xy=μy, σ_xz=μz, σ_yz=0. div(σ)_x=(4μ+4λ)+μ+μ=6μ+4λ
(d sigma_xy/dy = mu AND d sigma_xz/dz = mu - the 2D MMS bug taught: include ALL off-diagonals), div_y=div_z=0 -> f=(-(6 mu + 4 lambda),0,0).

float64 + residual guard (default solver settings stagnate silently).

  python3 warpfem_mms_3d.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
from warp.sparse import bsr_mv

E = 70.0e9
NU = 0.33
N = 10                  # Grid3D N×N×N
TOL_FIELD = 5e-3
TOL_RESID = 1e-6


@fem.integrand
def hooke_stress(strain: wp.mat33, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=3, dtype=float)


@fem.integrand
def elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def vec_proj(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def body_force_form(s: fem.Sample, v: fem.Field, fx: float):
    return wp.dot(wp.vec3(fx, 0.0, 0.0), v(s))


@fem.integrand
def mms_traction_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, mu: float, lam: float):
    p = fem.position(domain, s); n = fem.normal(domain, s)
    sxx = (4.0 * mu + 4.0 * lam) * p[0]
    syy = (2.0 * mu + 4.0 * lam) * p[0]
    szz = (2.0 * mu + 4.0 * lam) * p[0]
    sxy = mu * p[1]; sxz = mu * p[2]
    t = wp.vec3(sxx * n[0] + sxy * n[1] + sxz * n[2],
                sxy * n[0] + syy * n[1],
                sxz * n[0] + szz * n[2])
    return wp.dot(t, v(s))


@fem.integrand
def classify_left(s: fem.Sample, domain: fem.Domain, left: wp.array(dtype=int), other: wp.array(dtype=int)):
    p = fem.position(domain, s)
    if p[0] < 1e-4:
        left[s.qp_index] = 1
    else:
        other[s.qp_index] = 1


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    print(f"warp.fem 3D ELASTIK MMS (u*=(x²,xy,xz)) — Grid3D {N}³={N**3} celler, device={wp.get_device()}")
    geo = fem.Grid3D(res=wp.vec3i(N, N, N))
    space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
    domain = fem.Cells(geo)
    test = fem.make_test(space, domain=domain); trial = fem.make_trial(space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)); lame = wp.vec2(lam, mu)
    print(f"  3D Lamé: μ={mu:.3e}, λ={lam:.3e} (λ_3D=Eν/((1+ν)(1−2ν)))")

    K = fem.integrate(elasticity_form, fields={"u": trial, "v": test}, values={"lame": lame},
                      output_dtype=wp.float64)
    boundary = fem.BoundarySides(geo)
    lm = wp.zeros(boundary.element_count(), dtype=int); om = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify_left, at=boundary, values={"left": lm, "other": om})
    left = fem.Subdomain(boundary, element_mask=lm); other = fem.Subdomain(boundary, element_mask=om)

    fx = -(6.0 * mu + 4.0 * lam)                       # div(σ)_x = 6μ+4λ
    rhs = fem.integrate(body_force_form, fields={"v": test}, values={"fx": fx}, output_dtype=wp.vec3d)
    otest = fem.make_test(space, domain=other)
    rhs_t = fem.integrate(mms_traction_form, fields={"v": otest}, values={"mu": mu, "lam": lam},
                          output_dtype=wp.vec3d)
    rhs.assign(rhs.numpy() + rhs_t.numpy())

    ltest = fem.make_test(space, domain=left); ltrial = fem.make_trial(space, domain=left)
    bd = fem.integrate(vec_proj, fields={"u": ltrial, "v": ltest}, assembly="nodal", output_dtype=wp.float64)
    fem.normalize_dirichlet_projector(bd)
    fem.project_linear_system(K, rhs, bd, normalize_projector=False)

    u = wp.zeros_like(rhs)
    res, iters = fem_example_utils.bsr_cg(K, b=rhs, x=u, quiet=True, tol=1e-12, max_iters=20000)
    Ku = wp.zeros_like(rhs); bsr_mv(K, u, Ku)
    resid = float(np.linalg.norm(Ku.numpy() - rhs.numpy()) / np.linalg.norm(rhs.numpy()))

    nnode = space.node_count()
    pf = fem.make_discrete_field(space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 3)
    u_exact = np.stack([pos[:, 0] ** 2, pos[:, 0] * pos[:, 1], pos[:, 0] * pos[:, 2]], axis=1)
    u_fem = u.numpy().reshape(nnode, 3)
    rel_field = float(np.linalg.norm(u_fem - u_exact) / np.linalg.norm(u_exact))

    print(f"  CG: {iters} iter, residual-vakt ‖Ku−f‖/‖f‖ = {resid:.2e} (tol {TOL_RESID}) [float64]")
    print(f"  FIELD (3D): ||u_FEM-u*||/||u*|| = {rel_field:.2e} (tol {TOL_FIELD}); {nnode} nodes x 3 DOF")
    ok = rel_field < TOL_FIELD and resid < TOL_RESID
    print(f"\nVERDICT: warp.fem 3D elasticity MMS = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + ("FEM REPRODUCES the manufactured 3D solution u*=(x^2,xy,xz) to solver level, so the whole "
             "FYSIK-STACKEN GENERALISERAD TILL 3D (full 3D-elastik, kroppskraft+traktion-assembly, "
             "Q2 tet/hex, float64 solve) is rigorously validated. This is the path to REAL CAD PARTS "
             "(3D, not 2D cross-sections) - the largest generalisation gap closed for elasticity. The careful "
             "div(sigma) derivation (ALL off-diagonal terms) is confirmed in 3D. " if ok else
             "FEM does NOT reproduce u* in 3D (or CG stagnated) - debug BEFORE any 3D claim. ")
          + "CAVEAT: full 3D elasticity Q2; u* quadratic (in Q2 -> reproduction, not convergence order); float64 "
          "required; one MMS case; structured Grid3D (a CAD tet mesh in 3D is further work, needing gmsh/netgen).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

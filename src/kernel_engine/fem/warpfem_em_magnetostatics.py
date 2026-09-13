#!/usr/bin/env python3
"""warp.fem EM / MAGNETOSTATICS - a NEW DOMAIN (electromechanics), rigorous MMS + B field.

Electromechanics/mechatronics is an explicit target domain.
This opens the EM domain: 2D magnetostatics via the vector potential A_z (out of plane), governing equation

    −∇·(ν ∇A_z) = J_z ,   ν = 1/μ (reluktivitet),   B = ∇×(A_z ẑ) = (∂A_z/∂y, −∂A_z/∂x)

Structurally a scalar Poisson problem (like the thermal/acoustic Laplacian) BUT new physics: the magnetic field B from
a current J_z. RIGOROUS METHOD OF MANUFACTURED SOLUTIONS (the same discipline as the elasticity MMS):

  Pick A_z*(x,y) = (Lx.x - x^2)(Ly.y - y^2)  -> ZERO on the WHOLE boundary (homogeneous Dirichlet, A_z=0) and in Q2.
  grad^2 A_z* = -2[(Lx x - x^2)+(Ly y - y^2)] -> derive J_z = -nu grad^2 A_z* = 2 nu [(Lx x - x^2)+(Ly y - y^2)].
  Solve. A_z* in Q2 -> Galerkin REPRODUCES A_z* EXACTLY (a wrong nu/assembly gives a DIFFERENT solution and fails).

GATE (two stages): (1) the A_z field reproduces A_z* (the MMS core) + a residual guard ||K A - J||/||J||;
(2) B = curl A_z (EM-OBSERVABELN) via L2-projektion vs analytiska B* = (A_z*_y, −A_z*_x).
float64 + residual guard. No threshold grazing.

  python3 warpfem_em_magnetostatics.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
from warp.sparse import bsr_mv

MU0 = 4.0e-7 * np.pi          # magnetisk permeabilitet (vakuum)
NU = 1.0 / MU0                # reluktivitet
LX, LY = 0.10, 0.06           # magnetic domain (m)
NX, NY = 24, 16
TOL_FIELD = 1e-3              # A_z* in Q2 -> reproduced to the float32 assembly floor ~1e-4; a wrong nu gives O(1)
TOL_RESID = 1e-6
TOL_B = 2e-3                  # B via L2 projection (B* in Q2 -> recovery is near exact)


@fem.integrand
def stiffness_form(s: fem.Sample, A: fem.Field, q: fem.Field, nu: float):
    return nu * wp.dot(fem.grad(A, s), fem.grad(q, s))            # ∫ν∇A·∇q


@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, q: fem.Field, nu: float, lx: float, ly: float):
    p = fem.position(domain, s)
    jz = 2.0 * nu * ((lx * p[0] - p[0] * p[0]) + (ly * p[1] - p[1] * p[1]))   # J_z = −ν∇²A_z*
    return jz * q(s)


@fem.integrand
def scalar_proj(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def vec_proj(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def curl_rhs(s: fem.Sample, A: fem.Field, w: fem.Field):
    g = fem.grad(A, s)                                            # ∇A_z (vec2)
    return wp.dot(wp.vec2(g[1], -g[0]), w(s))                     # B = (∂A/∂y, −∂A/∂x)


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    print(f"warp.fem EM/MAGNETOSTATICS MMS - domain {LX}x{LY} m, mu=mu0, nu={NU:.3e}, {NX}x{NY}, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    A_space = fem.make_polynomial_space(geo, degree=2, dtype=float)
    B_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)
    domain = fem.Cells(geo)
    test = fem.make_test(A_space, domain=domain); trial = fem.make_trial(A_space, domain=domain)

    K = fem.integrate(stiffness_form, fields={"A": trial, "q": test}, values={"nu": NU},
                      output_dtype=wp.float64)
    rhs = fem.integrate(source_form, fields={"q": test}, values={"nu": NU, "lx": LX, "ly": LY},
                        output_dtype=wp.float64)

    # homogeneous Dirichlet A_z=0 on the WHOLE boundary (A_z*=0 there)
    boundary = fem.BoundarySides(geo)
    btest = fem.make_test(A_space, domain=boundary); btrial = fem.make_trial(A_space, domain=boundary)
    bd = fem.integrate(scalar_proj, fields={"u": btrial, "v": btest}, assembly="nodal",
                       output_dtype=wp.float64)
    fem.normalize_dirichlet_projector(bd)
    fem.project_linear_system(K, rhs, bd, normalize_projector=False)

    A = wp.zeros_like(rhs)
    res, iters = fem_example_utils.bsr_cg(K, b=rhs, x=A, quiet=True, tol=1e-12, max_iters=5000)
    Ku = wp.zeros_like(rhs); bsr_mv(K, A, Ku)
    resid = float(np.linalg.norm(Ku.numpy() - rhs.numpy()) / np.linalg.norm(rhs.numpy()))

    nnode = A_space.node_count()
    Apos_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)   # vec2 with the same Q2 nodes
    pf = fem.make_discrete_field(Apos_space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 2)
    A_exact = (LX * pos[:, 0] - pos[:, 0] ** 2) * (LY * pos[:, 1] - pos[:, 1] ** 2)
    A_fem = A.numpy()
    rel_A = float(np.linalg.norm(A_fem - A_exact) / np.linalg.norm(A_exact))

    # B = curl A_z via L2-projektion (M B = ∫ curl(A)·w)
    A_field = fem.make_discrete_field(A_space)
    wp.utils.array_cast(in_array=A, out_array=A_field.dof_values)        # float64 -> float32 field
    btr = fem.make_test(B_space, domain=domain); btt = fem.make_trial(B_space, domain=domain)
    MB = fem.integrate(vec_proj, fields={"u": btt, "v": btr}, output_dtype=wp.float64)
    rhsB = fem.integrate(curl_rhs, fields={"A": A_field, "w": btr}, output_dtype=wp.vec2d)
    Bn = wp.zeros_like(rhsB)
    fem_example_utils.bsr_cg(MB, b=rhsB, x=Bn, quiet=True, tol=1e-12, max_iters=5000)

    nB = B_space.node_count()
    pfb = fem.make_discrete_field(B_space); fem.interpolate(node_pos, dest=pfb)
    posb = pfb.dof_values.numpy().reshape(nB, 2)
    x, y = posb[:, 0], posb[:, 1]
    Bx_ex = (LX * x - x ** 2) * (LY - 2.0 * y)            # A_z*_y
    By_ex = -(LX - 2.0 * x) * (LY * y - y ** 2)           # −A_z*_x
    B_ex = np.stack([Bx_ex, By_ex], axis=1)
    B_fem = Bn.numpy().reshape(nB, 2)
    rel_B = float(np.linalg.norm(B_fem - B_ex) / np.linalg.norm(B_ex))
    Bmax = float(np.linalg.norm(B_fem, axis=1).max())

    print(f"  CG: {iters} iter, residual-vakt ‖K A−J‖/‖J‖ = {resid:.2e} (tol {TOL_RESID}) [float64]")
    print(f"  A_z FIELD (MMS): ||A-A*||/||A*|| = {rel_A:.2e} (tol {TOL_FIELD})")
    print(f"  B=curl A (EM-observabel) L2-recovery: ‖B−B*‖/‖B*‖ = {rel_B:.2e} (tol {TOL_B}); |B|max={Bmax:.3e} T")
    ok = rel_A < TOL_FIELD and resid < TOL_RESID and rel_B < TOL_B
    print(f"\nVERDICT: warp.fem EM/magnetostatics (NEW DOMAIN) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + ("FEM REPRODUCES the manufactured A_z*(x,y) to solver level (the MMS core; a wrong nu/assembly gives "
             "a different solution and fails) AND the magnetic field B=curl(A_z z^) recovered by L2 projection matches "
             "the analytic B*, so the EM domain is open: magnetic fields from a current distribution, a differentiable substrate "
             "for electromechanics. The residual guard "
             "confirms convergence. " if ok else
             "FEM does not reproduce A_z* or the B recovery does not match - debug BEFORE any EM claim. ")
          + "CAVEAT: 2D linear magnetostatics (constant nu, NO B-H saturation), Q2, one MMS case, structured "
          "mesh; permanent magnets / eddy currents / nonlinear iron saturation + Maxwell stress force are further work.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

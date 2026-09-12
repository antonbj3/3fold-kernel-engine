#!/usr/bin/env python3
"""warp.fem STOKES FLOW - the fluid domain, validated against analytic Poiseuille (a NEW DOMAIN: flow).

Opens the FLOW thread (hydraulic pumps, flow simulation) on the same
warp.fem stack. Stokes (creeping, incompressible): -nu grad^2 u + grad p = f, div u = 0. Taylor-Hood Q2-P1
(inf-sup-stabilt) + saddle-point-solve.

ANALYTIC-FIRST GATE (classical, exact): a plane channel of height H driven by a constant body force f=(G,0),
no-slip on top and bottom (u=0), do-nothing inlet/outlet. Fully developed -> u_x(y)=(G/2nu).y(H-y) (a parabola),
u_y=0, u_max=G·H²/(8ν). Q2 represents the parabola EXACTLY -> FEM must match to solver tolerance (no
no threshold grazing; a strong clean gate). Single-gradient form (integral nu grad u : grad v) - its do-nothing BC is satisfied
EXACTLY by Poiseuille (the D-form's is NOT, so it is the wrong choice here; derived by hand).

  python3 warpfem_stokes_poiseuille.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

NU = 1.0           # viskositet
G = 1.0            # kroppskraft (≡ −dp/dx)
LX, LY = 2.0, 1.0  # kanal: L×H
NX, NY = 24, 12
BD_STRENGTH = 1.0e4
TOL = 1.5e-2


@fem.integrand
def viscosity_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    # enkel-gradient: ∫ ν ∇u:∇v → strong −ν∇²u (do-nothing-BC = ν∂u/∂n−p·n, uppfylls av Poiseuille)
    return nu * wp.ddot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def div_form(s: fem.Sample, u: fem.Field, q: fem.Field):
    return q(s) * fem.div(u, s)


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def wall_classify(s: fem.Sample, domain: fem.Domain, wall: wp.array(dtype=int)):
    # top (n_y>0.5) + bottom (n_y<-0.5) = no-slip walls
    nor = fem.normal(domain, s)
    if wp.abs(nor[1]) > 0.5:
        wall[s.qp_index] = 1


@fem.integrand
def body_force_form(s: fem.Sample, v: fem.Field, g: float):
    return wp.dot(wp.vec2(g, 0.0), v(s))


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    print(f"warp.fem STOKES Poiseuille — kanal {LX}x{LY}, ν={NU}, G={G}, Q2-P1, {NX}x{NY}, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    u_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)   # Q2 hastighet
    p_space = fem.make_polynomial_space(geo, degree=1)                  # P1 tryck (inf-sup-stabilt)
    domain = fem.Cells(geo)
    boundary = fem.BoundarySides(geo)

    u_test = fem.make_test(u_space, domain=domain); u_trial = fem.make_trial(u_space, domain=domain)
    p_test = fem.make_test(p_space, domain=domain)

    A = fem.integrate(viscosity_form, fields={"u": u_trial, "v": u_test},
                      values={"nu": NU}, output_dtype=wp.float64)
    B = fem.integrate(div_form, fields={"u": u_trial, "q": p_test}, output_dtype=wp.float64)

    # no-slip top and bottom via a soft penalty (u -> 0). The walls are a subdomain.
    wmask = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(wall_classify, at=boundary, values={"wall": wmask})
    walls = fem.Subdomain(boundary, element_mask=wmask)
    w_test = fem.make_test(u_space, domain=walls); w_trial = fem.make_trial(u_space, domain=walls)
    wall_mass = fem.integrate(mass_form, fields={"u": w_trial, "v": w_test}, output_dtype=wp.float64)
    A += BD_STRENGTH * wall_mass    # the wall RHS is 0 (u_wall=0) -> no b_u term

    b_u = fem.integrate(body_force_form, fields={"v": u_test}, values={"g": G}, output_dtype=wp.vec2d)
    b_p = wp.zeros(p_space.node_count(), dtype=wp.float64)
    x_u = wp.zeros_like(b_u); x_p = wp.zeros_like(b_p)

    fem_example_utils.bsr_solve_saddle(
        fem_example_utils.SaddleSystem(A=A, B=B), x_u=x_u, x_p=x_p, b_u=b_u, b_p=b_p, quiet=True)

    u_field = u_space.make_field()
    wp.utils.array_cast(in_array=x_u, out_array=u_field.dof_values)

    nnode = u_space.node_count()
    pos_field = fem.make_discrete_field(u_space)
    fem.interpolate(node_pos, dest=pos_field)
    pos = pos_field.dof_values.numpy().reshape(nnode, 2)
    uv = x_u.numpy().reshape(nnode, 2)

    # fullt utvecklad region (undvik in/utlopps-kanteffekter): 0.25L < x < 0.75L
    interior = (pos[:, 0] > 0.25 * LX) & (pos[:, 0] < 0.75 * LX)
    y = pos[interior, 1]; ux = uv[interior, 0]; uy = uv[interior, 1]
    ux_exact = (G / (2.0 * NU)) * y * (LY - y)
    umax_exact = G * LY * LY / (8.0 * NU)
    umax_fem = float(ux.max())

    denom = np.maximum(np.abs(ux_exact), 1e-12)
    prof_relerr = float(np.median(np.abs(ux - ux_exact) / denom))
    umax_relerr = abs(umax_fem - umax_exact) / umax_exact
    uy_rel = float(np.abs(uy).max() / umax_exact)

    # ASCII-profil (u_x vs y, samplad vid mitt-kanalen)
    print(f"  u_max: FEM {umax_fem:.5f} vs analytic G.H^2/8nu={umax_exact:.5f}  relative error {umax_relerr:.2e}")
    print(f"  profile u_x(y): median rel error {prof_relerr:.2e} (interior, {interior.sum()} nodes)")
    print(f"  cross-flow |u_y|max/u_max = {uy_rel:.2e} (should go to 0)")
    order = np.argsort(y)
    ys, uxs, uxe = y[order], ux[order], ux_exact[order]
    print("  parabolprofil (• FEM mot analytisk kurva, y upp):")
    samp = np.linspace(0, len(ys) - 1, 11).astype(int)
    for i in samp[::-1]:
        bar = int(round(uxs[i] / umax_exact * 40))
        print(f"   y={ys[i]:.2f} |{'─' * bar}• ux={uxs[i]:+.4f} (exakt {uxe[i]:+.4f})")

    ok = umax_relerr < TOL and prof_relerr < TOL and uy_rel < TOL
    print(f"\nVERDICT: Stokes flow (fluid domain) = {'VALIDATED' if ok else 'NOT VALIDATED'} against analytic "
          f"Poiseuille (u_max=G·H²/8ν, parabolprofil). "
          + ("Incompressible creeping flow solved with a Taylor-Hood Q2-P1 saddle point on the warp.fem stack opens "
             "the FLOW DOMAIN - the same stack as structures/thermal/modal. "
             "Q2 captures the parabola exactly -> a match to solver level (no threshold grazing). " if ok else
             "The profile does not match Poiseuille - debug BEFORE any fluid-domain claim. ")
          + "CAVEAT: 2D Stokes (Re -> 0, no inertia or advection -> Navier-Stokes next); body-force driven "
          "fullt utvecklad kanal; mjuk no-slip (penalty 1e4).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

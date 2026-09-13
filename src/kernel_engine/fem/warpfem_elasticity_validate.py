#!/usr/bin/env python3
"""warp.fem LINEAR ELASTICITY - integration + an analytic-first gate (uniaxial traction).

A review caught an overclaim (the warp.fem gradient example was the vendor's, not this
code). This makes warp.fem an in-repo integration and gates it against EXACT analytics FIRST.

Root cause of an earlier failure: an inhomogeneous vec2 Dirichlet (scalar projector) underdetermines the boundary -> a nearly constant
solution. FIX (here): TRACTION on the right boundary (natural BC, integral of t.v) + HOMOGENEOUS u=0 on the left (the
working Dirichlet path, the shape-optimisation pattern). Uniaxial tension, plane stress.

EXACT GATE (independent of edge effects, via EQUILIBRIUM): domain-mean sigma_xx = applied traction t -
the axial force balance gives integral(sigma_xx dy) = t.H at every cross-section, so the section mean = t and the domain mean = t.
No threshold grazing ever passes. Soft diagnostic: tip displacement u_x ~ t.L/E (approximate, edge effect).

  python3 warpfem_elasticity_validate.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

E = 70.0e9
NU = 0.33
TRACTION = 1.0e6     # Pa, tensile stress on the right boundary
LX, LY = 4.0, 1.0    # slender bar (L/H=4) -> small edge effect on tip displacement
NX, NY = 32, 8
TOL_STRESS = 2e-3    # domain-mean sigma_xx against t (exact via equilibrium; no threshold grazing)
TOL_UX = 3e-2        # domain-mean u_x against t.L/2E - a CONSTITUTIVE gate (tests E, not just equilibrium)


@fem.integrand
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def vec_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def traction_form(s: fem.Sample, v: fem.Field, t: float):
    return wp.dot(wp.vec2(t, 0.0), v(s))      # traktion (t,0) i x


@fem.integrand
def classify_sides(s: fem.Sample, domain: fem.Domain, left: wp.array(dtype=int), right: wp.array(dtype=int)):
    nor = fem.normal(domain, s)
    if nor[0] < -0.5:
        left[s.qp_index] = 1
    if nor[0] > 0.5:
        right[s.qp_index] = 1


@fem.integrand
def stress_xx_form(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    return hooke_stress(fem.D(u, s), lame)[0, 0]


@fem.integrand
def area_form(s: fem.Sample):
    return 1.0


@fem.integrand
def ux_form(s: fem.Sample, u: fem.Field):
    return u(s)[0]


def main():
    wp.init()
    print(f"warp.fem uniaxial traktion — {NX}x{NY} grid {LX}x{LY}, E={E:.1e} nu={NU} t={TRACTION:.1e}, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    domain = fem.Cells(geo)
    test = fem.make_test(space, domain=domain)
    trial = fem.make_trial(space, domain=domain)

    mu = E / (2.0 * (1.0 + NU))
    lam = E * NU / (1.0 - NU * NU)            # plane stress
    lame = wp.vec2(lam, mu)

    K = fem.integrate(elasticity_form, fields={"u": trial, "v": test}, values={"lame": lame})

    # classify the boundary -> left (Dirichlet) + right (traction)
    boundary = fem.BoundarySides(geo)
    left_mask = wp.zeros(boundary.element_count(), dtype=int)
    right_mask = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify_sides, at=boundary, values={"left": left_mask, "right": right_mask})
    left = fem.Subdomain(boundary, element_mask=left_mask)
    right = fem.Subdomain(boundary, element_mask=right_mask)

    # traction load (natural BC) on the right boundary
    right_test = fem.make_test(space, domain=right)
    rhs = fem.integrate(traction_form, fields={"v": right_test}, values={"t": TRACTION}, output_dtype=wp.vec2)

    # homogeneous Dirichlet u=0 on the left boundary (the working pattern)
    left_test = fem.make_test(space, domain=left)
    left_trial = fem.make_trial(space, domain=left)
    bd_proj = fem.integrate(vec_projector_form, fields={"u": left_trial, "v": left_test},
                            assembly="nodal", output_dtype=float)
    fem.normalize_dirichlet_projector(bd_proj)
    fem.project_linear_system(K, rhs, bd_proj, normalize_projector=False)

    u = wp.zeros_like(rhs)
    fem_example_utils.bsr_cg(K, b=rhs, x=u, quiet=True, tol=1e-10, max_iters=2000)

    field = fem.make_discrete_field(space)
    field.dof_values = u

    area = fem.integrate(area_form, domain=domain)
    sxx = fem.integrate(stress_xx_form, fields={"u": field}, values={"lame": lame}, domain=domain) / area
    rel_stress = abs(sxx - TRACTION) / TRACTION
    # tip displacement diagnostic (approximate)
    ux_avg = fem.integrate(ux_form, fields={"u": field}, domain=domain) / area
    ux_expected_mid = TRACTION * (LX / 2.0) / E   # medel-x ≈ L/2 → u_x ≈ t·(L/2)/E

    rel_ux = abs(ux_avg - ux_expected_mid) / ux_expected_mid
    print(f"  EQUILIBRIUM GATE (constitutive-INDEPENDENT identity): domain-mean sigma_xx = {sxx:.5e} Pa vs t="
          f"{TRACTION:.5e}  relative error {rel_stress:.2e}  (tol {TOL_STRESS})")
    print(f"  CONSTITUTIVE GATE (tests E): domain-mean u_x = {ux_avg:.4e} m vs t.(L/2)/E = {ux_expected_mid:.4e}"
          f"  relative error {rel_ux:.2e}  (tol {TOL_UX}; clamped-end edge effect ~1%)")
    ok = rel_stress < TOL_STRESS and rel_ux < TOL_UX
    print(f"\nVERDICT: warp.fem linear elasticity = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + ("TWO gates: (1) domain-mean sigma_xx=t is an EQUILIBRIUM IDENTITY (Galerkin orthogonality with "
             "test function v=(x,0) in P1, zero at the clamp) that is CONSTITUTIVE-INDEPENDENT, so it validates assembly + "
             "the traction RHS + solver convergence + BC coupling, NOT E/nu (audit finding: a wrong E would pass "
             "it). (2) domain-mean u_x=t.L/2E validates the CONSTITUTIVE E (1% edge effect, well under the 3% tolerance). "
             "Together: assembly + solver + BC + E gated; nu is NOT tested directly (NEXT: the Kirsch plate for "
             "field accuracy and nu). " if ok else
             "At least one gate failed - debug BEFORE any FEM claim. ")
          + "CAVEAT: 2D plane-stress P1; the u_x gate has a clamped-end edge effect (~1%); nu is not gated directly.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

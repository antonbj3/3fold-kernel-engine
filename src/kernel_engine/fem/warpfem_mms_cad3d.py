#!/usr/bin/env python3
"""warp.fem 3D MMS on a REAL CAD VOLUME - the product thesis in 3D (real part, unstructured tet mesh).

`warpfem_mms_3d.py` validated 3D elasticity on a structured Grid3D. This runs it on a REAL 3D
CAD SOLID (box minus a cylindrical hole, gmsh tet volume mesh from `cad_to_tetmesh.py`), proving the whole 3D CAD-to-sim
pipeline: 3D CAD solid -> tet volume -> 3D physics, field-validated.

Manufactured u*=(x^2,xy,xz) (zero at x=0 -> homogeneous Dirichlet); derived body force f=(-(6 mu + 4 lambda),0,0) +
analytic traction sigma(u*).n on ALL other faces INCLUDING THE HOLE WALL (arbitrary 3D normals). u* in P2(tet) ->
warp.fem reproducerar u* exakt (float64 + residual-vakt).

  python3 warpfem_mms_cad3d.py
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
from warp.sparse import bsr_mv

E = 70.0e9
NU = 0.33
MESH = Path("data/cad_meshes/cad_box3d.npz")
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
    if not MESH.exists():
        print(f"MISSING: {MESH} - run cad_to_tetmesh.py first"); return 2
    data = np.load(MESH)
    pos_np = data["positions"].astype(np.float32); tets_np = data["tets"].astype(np.int32)
    print(f"warp.fem 3D MMS on a CAD VOLUME (box minus hole, {len(pos_np)} nodes / {len(tets_np)} tets, P2) - "
          f"device={wp.get_device()}")
    geo = fem.Tetmesh(tet_vertex_indices=wp.array(tets_np, dtype=wp.int32),
                      positions=wp.array(pos_np, dtype=wp.vec3))
    space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
    domain = fem.Cells(geo)
    test = fem.make_test(space, domain=domain); trial = fem.make_trial(space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)); lame = wp.vec2(lam, mu)

    K = fem.integrate(elasticity_form, fields={"u": trial, "v": test}, values={"lame": lame},
                      output_dtype=wp.float64)
    boundary = fem.BoundarySides(geo)
    lm = wp.zeros(boundary.element_count(), dtype=int); om = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify_left, at=boundary, values={"left": lm, "other": om})
    left = fem.Subdomain(boundary, element_mask=lm); other = fem.Subdomain(boundary, element_mask=om)

    fx = -(6.0 * mu + 4.0 * lam)
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
    res, iters = fem_example_utils.bsr_cg(K, b=rhs, x=u, quiet=True, tol=1e-12, max_iters=30000)
    Ku = wp.zeros_like(rhs); bsr_mv(K, u, Ku)
    resid = float(np.linalg.norm(Ku.numpy() - rhs.numpy()) / np.linalg.norm(rhs.numpy()))

    nnode = space.node_count()
    pf = fem.make_discrete_field(space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 3)
    u_exact = np.stack([pos[:, 0] ** 2, pos[:, 0] * pos[:, 1], pos[:, 0] * pos[:, 2]], axis=1)
    u_fem = u.numpy().reshape(nnode, 3)
    rel_field = float(np.linalg.norm(u_fem - u_exact) / np.linalg.norm(u_exact))

    print(f"  CG: {iters} iter, residual-vakt ‖Ku−f‖/‖f‖ = {resid:.2e} (tol {TOL_RESID}) [float64]")
    print(f"  FIELD (3D CAD volume): ||u_FEM-u*||/||u*|| = {rel_field:.2e} (tol {TOL_FIELD}); {nnode} P2 nodes x 3 DOF")
    ok = rel_field < TOL_FIELD and resid < TOL_RESID
    print(f"\nVERDICT: warp.fem 3D MMS on a REAL CAD VOLUME = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + ("FEM reproduces u*=(x^2,xy,xz) on the UNSTRUCTURED gmsh tet VOLUME (box minus cylindrical hole, P2 tets, "
             "traction also on the HOLE WALL with arbitrary 3D normals) to solver level, so the whole 3D CAD-TO-SIM "
             "PIPELINE is proven: 3D CAD solid -> tet volume mesh -> 3D physics, field-validated ON A REAL 3D PART. "
             "The product thesis is now concrete for real 3D parts, not only 2D cross-sections. " if ok else
             "FEM does NOT reproduce u* on the 3D tet volume (or CG stagnated) - debug BEFORE any 3D CAD claim. ")
          + "CAVEAT: full 3D elasticity P2 tets; u* quadratic (in P2 -> reproduction); float64 required; one MMS case "
          "+ one CAD geometry (box minus hole); gmsh Delaunay tet volume.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

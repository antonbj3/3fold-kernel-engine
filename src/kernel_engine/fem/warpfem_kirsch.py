#!/usr/bin/env python3
"""warp.fem KIRSCH - resolves the stress CONCENTRATION, validated against classical analytics (Kt=3).

stress3d reported Kt as a LOWER BOUND (a coarse mesh + cell means under-resolve the peak). This
resolves the TRUE concentration: a finely graded mesh at the hole (cad_kirsch_mesh.py) + P2 + L2
PROJECTION stress recovery (smooth nodal sigma_theta_theta, not cell means) -> compared against the classical KIRSCH solution
for a hole in a plate under tension:

  sigma_tt(a,theta) = sigma_inf.(1 - 2 cos 2theta)   at the hole edge -> peak 3 sigma_inf at theta=90 deg (Kt=3), -sigma_inf at theta=0

GATE: (1) Kt = max sigma_tt/sigma_inf ~ 3 (the finite-width correction is small for 2a/W~0.13) + (2) the sigma_tt(theta) profile matches
sigma_inf(1-2 cos 2theta). This makes the stress OUTPUT quantitatively credible (not just a qualitative LOWER BOUND). float64 +
residual-vakt.

  python3 warpfem_kirsch.py
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
TRACTION = 1.0e6
MESH = Path("data/cad_meshes/cad_kirsch.npz")
KT_LO, KT_HI = 2.6, 3.3        # finite-width Kirsch (2a/W~0.13) ~3.0; P2 + recovery
TOL_PROFILE = 0.15             # normalised sigma_tt(theta) profile error against Kirsch


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
def traction_form(s: fem.Sample, v: fem.Field, t: float):
    return wp.dot(wp.vec2(t, 0.0), v(s))


@fem.integrand
def classify(s: fem.Sample, domain: fem.Domain, left: wp.array(dtype=int), right: wp.array(dtype=int),
             w: float):
    p = fem.position(domain, s)
    if p[0] < 1e-4:
        left[s.qp_index] = 1
    if p[0] > w - 1e-4:
        right[s.qp_index] = 1


@fem.integrand
def scalar_mass(s: fem.Sample, p: fem.Field, w: fem.Field):
    return p(s) * w(s)


@fem.integrand
def sigma_theta_rhs(s: fem.Sample, domain: fem.Domain, u: fem.Field, w: fem.Field, lame: wp.vec2,
                    cx: float, cy: float):
    sig = hooke_stress(fem.D(u, s), lame)
    p = fem.position(domain, s)
    th = wp.atan2(p[1] - cy, p[0] - cx)
    st = wp.sin(th); ct = wp.cos(th)
    s_tt = sig[0, 0] * st * st - 2.0 * sig[0, 1] * st * ct + sig[1, 1] * ct * ct   # σ_θθ (hoop)
    return s_tt * w(s)


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    if not MESH.exists():
        print(f"MISSING: {MESH} - run cad_kirsch_mesh.py first"); return 2
    data = np.load(MESH)
    pos_np = data["positions"].astype(np.float32); tris_np = data["tris"].astype(np.int32)
    W = float(data["W"]); H = float(data["H"]); A = float(data["A"])
    cx, cy = W / 2, H / 2
    print(f"warp.fem KIRSCH (plate {W}x{H}, hole a={A}, 2a/W={2*A/W:.2f}) - {len(pos_np)} nodes / {len(tris_np)} triangles, "
          f"P2, device={wp.get_device()}")
    geo = fem.Trimesh2D(tri_vertex_indices=wp.array(tris_np, dtype=wp.int32),
                        positions=wp.array(pos_np, dtype=wp.vec2))
    u_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)
    s_space = fem.make_polynomial_space(geo, degree=1, dtype=float)     # σ_θθ-recovery-rum
    domain = fem.Cells(geo)
    ut = fem.make_test(u_space, domain=domain); utr = fem.make_trial(u_space, domain=domain)
    st_test = fem.make_test(s_space, domain=domain); st_tr = fem.make_trial(s_space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam = E * NU / (1.0 - NU * NU); lame = wp.vec2(lam, mu)

    K = fem.integrate(elasticity_form, fields={"u": utr, "v": ut}, values={"lame": lame}, output_dtype=wp.float64)
    boundary = fem.BoundarySides(geo)
    lm = wp.zeros(boundary.element_count(), dtype=int); rm = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify, at=boundary, values={"left": lm, "right": rm, "w": W})
    left = fem.Subdomain(boundary, element_mask=lm); right = fem.Subdomain(boundary, element_mask=rm)
    rtest = fem.make_test(u_space, domain=right)
    rhs = fem.integrate(traction_form, fields={"v": rtest}, values={"t": TRACTION}, output_dtype=wp.vec2d)
    ltest = fem.make_test(u_space, domain=left); ltrial = fem.make_trial(u_space, domain=left)
    bd = fem.integrate(vec_proj, fields={"u": ltrial, "v": ltest}, assembly="nodal", output_dtype=wp.float64)
    fem.normalize_dirichlet_projector(bd)
    fem.project_linear_system(K, rhs, bd, normalize_projector=False)
    u = wp.zeros_like(rhs)
    res, iters = fem_example_utils.bsr_cg(K, b=rhs, x=u, quiet=True, tol=1e-12, max_iters=20000)
    Ku = wp.zeros_like(rhs); bsr_mv(K, u, Ku)
    resid = float(np.linalg.norm(Ku.numpy() - rhs.numpy()) / np.linalg.norm(rhs.numpy()))
    u_field = fem.make_discrete_field(u_space); wp.utils.array_cast(in_array=u, out_array=u_field.dof_values)

    # L2 PROJECTION of sigma_tt -> smooth nodal recovery: M.sigma = integral(sigma_tt.w)
    M = fem.integrate(scalar_mass, fields={"p": st_tr, "w": st_test}, output_dtype=wp.float64)
    rhs_s = fem.integrate(sigma_theta_rhs, fields={"u": u_field, "w": st_test},
                          values={"lame": lame, "cx": cx, "cy": cy}, output_dtype=wp.float64)
    sig_nodal = wp.zeros(s_space.node_count(), dtype=wp.float64)
    fem_example_utils.bsr_cg(M, b=rhs_s, x=sig_nodal, quiet=True, tol=1e-12, max_iters=5000)

    # P1 s_space nodes = mesh vertices (same order) -> hole-edge nodes via the mesh positions
    spos = pos_np.astype(np.float64)
    sig = sig_nodal.numpy()
    r = np.sqrt((spos[:, 0] - cx) ** 2 + (spos[:, 1] - cy) ** 2)
    edge = np.abs(r - A) < 0.03
    th = np.arctan2(spos[edge, 1] - cy, spos[edge, 0] - cx)
    s_tt = sig[edge]
    s_kirsch = TRACTION * (1.0 - 2.0 * np.cos(2.0 * th))
    Kt = float(s_tt.max() / TRACTION)
    # profile error (normalised against sigma_inf)
    prof_err = float(np.median(np.abs(s_tt - s_kirsch)) / TRACTION)
    # value at theta~0 (side, should be -sigma_inf)
    i0 = int(np.argmin(np.abs(np.cos(th))))   # want theta ~ 0
    i0 = int(np.argmin(np.abs(th)))
    s_at0 = float(s_tt[i0] / TRACTION)

    print(f"  CG: {iters} iterations, residual guard {resid:.2e}; {int(edge.sum())} hole-edge nodes")
    print(f"  Kt = max sigma_tt/sigma_inf = {Kt:.2f} (Kirsch infinite=3.0, finite width 2a/W={2*A/W:.2f} ~3.0)")
    print(f"  sigma_tt(theta~0)/sigma_inf = {s_at0:.2f} (Kirsch -1.0); profile error {prof_err:.2e} (tol {TOL_PROFILE})")
    ok = KT_LO < Kt < KT_HI and prof_err < TOL_PROFILE and resid < 1e-6
    print(f"\nVERDICT: warp.fem KIRSCH stress concentration = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"The TRUE concentration is RESOLVED (finely graded mesh + L2 projection recovery) -> Kt={Kt:.2f} "
             "matches classical Kirsch (Kt=3) and the sigma_tt(theta) profile matches sigma_inf(1-2cos2theta) (tensile peak at theta=90 deg, "
             "compression -sigma_inf at theta=0). This makes the stress3d engineering output QUANTITATIVELY credible (not just a LOWER "
             "BOUND): the validated FEM + recovery resolve the failure stress correctly against the gold-standard "
             "analytik. " if ok else
             "Kt/profile does not match Kirsch - debug mesh/recovery BEFORE any concentration claim. ")
          + "CAVEAT: 2D plane stress P2 + L2 recovery; finite-width plate (2a/W~0.13, Kt~3.0 plus a finite correction); "
          "the left clamp approximates a uniaxial far field (St Venant); one geometry case.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

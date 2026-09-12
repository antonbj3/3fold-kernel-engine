#!/usr/bin/env python3
"""warp.fem GENERATIVE DESIGN for an ARBITRARY objective - closes the design-from-physics loop (the payoff).

`warpfem_adjoint_arbitrary.py` VALIDATED the adjoint GRADIENT for an arbitrary objective (lambda != u, FD-gated)
but never used it in an optimisation. `warpfem_topopt.py` GENERATES geometry but only for compliance
(self-adjoint). This COMBINES them: topology optimisation driven by the VALIDATED arbitrary-objective
adjoint, generating a design for a NON-compliance objective via CROSS COUPLING (point load UPPER right,
minimise u_y^2 LOWER right -> isolating the objective from the load). It demonstrates "design for ANY objective FROM
physics" - and that a genuinely non-compliance objective gives a MEASURABLY DISTINCT design (difference ~0.20 vs
compliance), unlike a compliance-like objective (deflection AT the load -> ~compliance, difference ~0.06).

Objective J = u_y(target)^2; sensitivity dJ/drho_e = -p.rho^(p-1).integral sigma0(u):eps(lambda) with lambda from K lambda = dJ/du (point load 2.u_y
on the target y DOF) - a CROSS energy (lambda != u, not self-adjoint). Distinctness from compliance is MEASURED (both are run).
VERIFIED LESSONS: the OC method OSCILLATES for an arbitrary objective (OC assumes a negative
sensitivity = compliance; an arbitrary objective has mixed signs) -> a PROJECTED GRADIENT is required; float32 CG
STAGNATES on ill-conditioned SIMP (rho contrast 1e9) -> float64 + rho_min=1e-2 + a residual guard are required.

  python3 warpfem_design_objective.py
"""
import sys

import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
from warp.sparse import bsr_mv

E = 70.0e9
NU = 0.33
LOAD = 1.0e6
LX, LY = 3.0, 1.0
NX, NY = 90, 30
SIMP_P = 3.0
VOLFRAC = 0.4
RMIN = 1.5
N_ITER = 45
TOL_RESID = 1e-5


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
def load_form(s: fem.Sample, v: fem.Field, t: float):
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
    return w(s) * wp.ddot(fem.D(lam, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def cell_x(s: fem.Sample, domain: fem.Domain, w: fem.Field):
    return w(s) * fem.position(domain, s)[0]


@fem.integrand
def cell_y(s: fem.Sample, domain: fem.Domain, w: fem.Field):
    return w(s) * fem.position(domain, s)[1]


@fem.integrand
def cell_one(s: fem.Sample, w: fem.Field):
    return w(s)


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def main():
    wp.init()
    print(f"warp.fem GENERATIVE DESIGN (arbitrary objective u_y(target)^2) - {NX}x{NY}={NX*NY} cells, "
          f"volfrac={VOLFRAC}, device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)
    domain = fem.Cells(geo)
    u_test = fem.make_test(u_space, domain=domain); u_trial = fem.make_trial(u_space, domain=domain)
    cell_test = fem.make_test(rho_space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam = E * NU / (1.0 - NU * NU); lame = wp.vec2(lam, mu)

    boundary = fem.BoundarySides(geo)
    lm = wp.zeros(boundary.element_count(), dtype=int); rm = wp.zeros(boundary.element_count(), dtype=int)
    fem.interpolate(classify, at=boundary, values={"left": lm, "right": rm})
    left = fem.Subdomain(boundary, element_mask=lm); right = fem.Subdomain(boundary, element_mask=rm)
    right_test = fem.make_test(u_space, domain=right)
    left_test = fem.make_test(u_space, domain=left); left_trial = fem.make_trial(u_space, domain=left)
    bd = fem.integrate(vec_proj, fields={"u": left_trial, "v": left_test}, assembly="nodal", output_dtype=wp.float64)
    fem.normalize_dirichlet_projector(bd)

    ncell = rho_space.node_count()
    rho = fem.make_discrete_field(rho_space)
    u_field = fem.make_discrete_field(u_space); lam_field = fem.make_discrete_field(u_space)

    nnode = u_space.node_count()
    pf = fem.make_discrete_field(u_space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 2)
    # CROSS COUPLING (genuinely non-compliance): downward point load UPPER RIGHT; minimise u_y^2 LOWER RIGHT.
    # Compliance minimises the deflection WHERE the load is (upper right); this objective isolates lower right from
    # the load -> forces a DIFFERENT load path -> a genuinely distinct design.
    n_load = int(np.argmin((pos[:, 0] - LX) ** 2 + (pos[:, 1] - LY) ** 2))     # upper right
    rhs0_np = np.zeros((nnode, 2), np.float64); rhs0_np[n_load, 1] = -LOAD
    rhs0 = wp.array(rhs0_np, dtype=wp.vec2d)
    tgt = int(np.argmin((pos[:, 0] - LX) ** 2 + pos[:, 1] ** 2))               # lower right
    print(f"  point load @ ({pos[n_load,0]:.2f},{pos[n_load,1]:.2f}); objective u_y^2 @ ({pos[tgt,0]:.2f},{pos[tgt,1]:.2f}) "
          f"- CROSS COUPLING (genuinely non-compliance)")

    cx = fem.integrate(cell_x, fields={"w": cell_test}, domain=domain).numpy()
    cy = fem.integrate(cell_y, fields={"w": cell_test}, domain=domain).numpy()
    ca = fem.integrate(cell_one, fields={"w": cell_test}, domain=domain).numpy()
    cx, cy = cx / ca, cy / ca
    dx = LX / NX
    H = np.zeros((ncell, ncell), dtype=np.float64); R = RMIN * dx
    for i in range(ncell):
        H[i] = np.maximum(0.0, R - np.sqrt((cx - cx[i]) ** 2 + (cy - cy[i]) ** 2))
    Hs = H.sum(1)

    def solve(K, rhs):
        rr = wp.clone(rhs)
        fem.project_linear_system(K, rr, bd, normalize_projector=False)
        x = wp.zeros_like(rr)
        fem_example_utils.bsr_cg(K, b=rr, x=x, quiet=True, tol=1e-12, max_iters=12000)
        Ku = wp.zeros_like(rr); bsr_mv(K, x, Ku)
        res = float(np.linalg.norm(Ku.numpy() - rr.numpy()) / max(np.linalg.norm(rr.numpy()), 1e-30))
        return x, res

    def objective_and_sens(use_compliance):
        K = fem.integrate(simp_form, fields={"u": u_trial, "v": u_test, "rho": rho},
                          values={"lame": lame, "p": SIMP_P}, output_dtype=wp.float64)   # float64 (ill-kond)
        u, res1 = solve(K, rhs0)
        wp.utils.array_cast(in_array=u, out_array=u_field.dof_values)        # float64 -> float32 field
        xr = rho.dof_values.numpy()
        if use_compliance:
            # compliance = self-adjoint (λ=u): sens = self-energi (kors-energi med lam=u)
            ce = fem.integrate(cross_energy_form, fields={"u": u_field, "lam": u_field, "w": cell_test},
                               values={"lame": lame}, output_dtype=float).numpy()
            obj = float((xr ** SIMP_P * ce).sum()); res2 = 0.0
        else:
            uy = float(u.numpy()[tgt, 1]); obj = uy * uy
            # adjoint: K lambda = dJ/du = point load 2.u_y(target) on the target y DOF
            g = np.zeros((nnode, 2), np.float64); g[tgt, 1] = 2.0 * uy
            lam_vec, res2 = solve(K, wp.array(g, dtype=wp.vec2d))
            wp.utils.array_cast(in_array=lam_vec, out_array=lam_field.dof_values)
            ce = fem.integrate(cross_energy_form, fields={"u": u_field, "lam": lam_field, "w": cell_test},
                               values={"lame": lame}, output_dtype=float).numpy()
        dObj = -SIMP_P * (xr ** (SIMP_P - 1.0)) * ce
        return obj, dObj, max(res1, res2)

    def optimize(use_compliance):
        # PROJECTED GRADIENT (not OC): OC assumes a negative sensitivity (compliance); an ARBITRARY objective has
        # BLANDAT-tecken kors-energi-sensitivitet → OC oscillerar (verifierat). Move-limiterat normaliserat
        # steg + volym-projektion (Lagrange-skift via bisection) hanterar blandat tecken stabilt/monotont.
        x = np.full(ncell, VOLFRAC); hist = []; maxres = 0.0
        for it in range(N_ITER):
            rho.dof_values.assign(x.astype(np.float32))
            obj, dObj, res = objective_and_sens(use_compliance)
            hist.append(obj); maxres = max(maxres, res)
            dObj_f = (H @ (x * dObj)) / (Hs * np.maximum(x, 1e-3))
            x_trial = x - 0.04 * dObj_f / (np.abs(dObj_f).max() + 1e-30)
            lo, hi = x_trial.min() - 1.0, x_trial.max() + 1.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if np.clip(x_trial - mid, 1e-2, 1.0).mean() > VOLFRAC:
                    lo = mid
                else:
                    hi = mid
            x = np.clip(x_trial - 0.5 * (lo + hi), 1e-2, 1.0)    # ρ_min=1e-2 → CG-konditionering OK
            if (it % 9 == 0 or it == N_ITER - 1):
                tag = "C" if use_compliance else "J"
                print(f"  [{tag}] it {it:3d}: obj={obj:.4e}  vol={x.mean():.3f}  CG-res={res:.1e}")
        return x, hist, maxres

    print("--- point-stiffness objective (the arbitrary-objective adjoint) ---")
    x_obj, hist_obj, maxres_obj = optimize(False)
    print("--- compliance objective (reference for distinctness) ---")
    x_cmp, hist_cmp, maxres_cmp = optimize(True)

    # emergent design (point objective)
    AW, AH = 60, 16
    g_ascii = np.zeros((AH, AW)); cnt = np.zeros((AH, AW))
    for i in range(ncell):
        gx = min(AW - 1, int(cx[i] / LX * AW)); gy = min(AH - 1, int(cy[i] / LY * AH))
        g_ascii[gy, gx] += x_obj[i]; cnt[gy, gx] += 1
    g_ascii = g_ascii / np.maximum(cnt, 1)
    print("\nEmergent design (point-stiffness objective):")
    for row in g_ascii[::-1]:
        print("  |" + "".join("#" if v > 0.6 else (":" if v > 0.3 else " ") for v in row) + "|")

    design_diff = float(np.linalg.norm(x_obj - x_cmp) / np.linalg.norm(x_cmp))
    J0, Jf = hist_obj[0], hist_obj[-1]
    monotone = all(hist_obj[i + 1] <= hist_obj[i] * 1.02 for i in range(len(hist_obj) - 1))
    converged = (Jf < 0.5 * J0) and monotone and (maxres_obj < TOL_RESID) and (maxres_cmp < TOL_RESID)
    print(f"\nVERDICT: generative design for an ARBITRARY objective = {'VALIDATED' if converged else 'NOT VALIDATED'} "
          f"- point objective J=u_y^2 {J0:.3e} -> {Jf:.3e} ({100*(1-Jf/J0):.0f}% reduction, {'MONOTONE' if monotone else 'NOT monotone'}), "
          f"residual-vakt max {max(maxres_obj, maxres_cmp):.1e}<{TOL_RESID}. "
          + (f"The VALIDATED arbitrary-objective adjoint (lambda != u, not compliance) drives a STABLE (monotone) "
             f"projected-gradient topology optimisation -> design FROM physics for a NON-compliance objective. "
             f"DISTINCTNESS MEASURED: ||rho_point - rho_compliance||/||rho_compliance|| = {design_diff:.2f}, so the design "
             f"{'is measurably but MODESTLY different from' if design_diff > 0.15 else 'resembles'} the compliance design "
             f"(same setup, same volume). The residual guard (float64) confirms that ALL solves converged. "
             if converged else
             "Not monotone/converged or the solve stagnated - debug BEFORE any claim. ")
          + "VERIFIED LESSON: the OC method oscillates for an arbitrary objective (it assumes a negative "
          "sensitivity = compliance) -> a projected gradient is required; float32 CG stagnates on ill-conditioned "
          "SIMP -> float64 + rho_min=1e-2 + a residual guard are required. CAVEAT: 2D P1 SIMP; ONE cross-coupling objective; "
          "compliant mechanisms (maximising output MOTION, spring ports) are further work; a DEMO, not production topology optimisation.")
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())

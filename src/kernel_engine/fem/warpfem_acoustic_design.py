#!/usr/bin/env python3
"""warp.fem DIFFERENTIABLE ACOUSTIC DESIGN - design from physics in a NEW domain (NVH resonance tuning).

The distinctive claim is design-from-physics ACROSS domains. Differentiable design is already shown
in STRUCTURES (design gradient, arbitrary objective, coupled multiphysics). This shows the GENERALITY: differentiable
design in the ACOUSTIC domain - the sensitivity of a resonant eigenfrequency to the material distribution, so one can tune
NVH resonances (engine acoustics, cavity rattles) by design.

Eigenproblem K phi = lambda M(rho) phi with per-cell density rho (mass M(rho)=integral rho p q; more density -> more inertia -> lower
resonance). EIGENVALUE SENSITIVITY (M-normalised phi, phi^T M phi=1): dlambda/drho_e = phi^T(dK/drho_e - lambda dM/drho_e)phi =
-lambda integral_e phi^2  (K independent of rho; dM/drho_e = the cell mass). FD GATE (perturb rho_e, re-solve the eigenproblem,
recompute lambda) - the same guarded pattern as the design gradient. No threshold grazing passes.

  python3 warpfem_acoustic_design.py
"""
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import warp as wp
import warp.fem as fem

C = 343.0
LX, LY = 0.5, 0.3
NX, NY = 40, 24
FD_EPS = 1e-3
FD_TOL = 1e-2


@fem.integrand
def stiffness_form(s: fem.Sample, p: fem.Field, q: fem.Field):
    return wp.dot(fem.grad(p, s), fem.grad(q, s))


@fem.integrand
def mass_form(s: fem.Sample, p: fem.Field, q: fem.Field, rho: fem.Field):
    return rho(s) * p(s) * q(s)                         # densitets-viktad massa M(ρ)


@fem.integrand
def cell_phi2(s: fem.Sample, phi: fem.Field, w: fem.Field):
    return w(s) * phi(s) * phi(s)                       # ∫_e φ² (enhets-densitet cell-mass-energi)


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def to_scipy(Mat, n):
    Mat.nnz_sync()
    v = Mat.values.numpy()
    if v.ndim == 1:
        v = v.reshape(-1, 1, 1)
    return sps.bsr_matrix((v, Mat.columns.numpy(), Mat.offsets.numpy()), shape=(n, n)).tocsr()


def main():
    wp.init()
    print(f"warp.fem DIFFERENTIERBAR AKUSTISK DESIGN — kavitet {LX}x{LY}, {NX}x{NY}, device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    p_space = fem.make_polynomial_space(geo, degree=1, dtype=float)
    rho_space = fem.make_polynomial_space(geo, degree=0, dtype=float)
    domain = fem.Cells(geo)
    test = fem.make_test(p_space, domain=domain); trial = fem.make_trial(p_space, domain=domain)
    cell_test = fem.make_test(rho_space, domain=domain)
    rho = fem.make_discrete_field(rho_space); rho.dof_values.fill_(1.0)
    nnode = p_space.node_count(); ncell = rho_space.node_count()

    Kw = fem.integrate(stiffness_form, fields={"p": trial, "q": test}, output_dtype=wp.float64)
    K = to_scipy(Kw, nnode)
    pos_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    pf = fem.make_discrete_field(pos_space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 2)
    xmn, ymn = pos[:, 0].min(), pos[:, 1].min(); xmx, ymx = pos[:, 0].max(), pos[:, 1].max()
    eps = 1e-5
    on_bd = ((pos[:, 0] < xmn + eps) | (pos[:, 0] > xmx - eps) |
             (pos[:, 1] < ymn + eps) | (pos[:, 1] > ymx - eps))
    free = np.where(~on_bd)[0]
    Kff = K[np.ix_(free, free)].tocsc()
    phi_field = fem.make_discrete_field(p_space)

    def solve_eig(return_vec=False):
        Mw = fem.integrate(mass_form, fields={"p": trial, "q": test, "rho": rho}, output_dtype=wp.float64)
        M = to_scipy(Mw, nnode); Mff = M[np.ix_(free, free)].tocsc()
        if return_vec:
            vals, vecs = spsl.eigsh(Kff, k=1, M=Mff, sigma=0.0, which="LM")
            lam = float(np.abs(vals[0]))
            phi = np.zeros(nnode); phi[free] = vecs[:, 0]
            # M-normalise (phi^T M phi=1 - eigsh already does this, but make sure)
            nrm = float(phi[free] @ (Mff @ phi[free]))
            phi /= np.sqrt(max(nrm, 1e-30))
            return lam, phi
        vals = spsl.eigsh(Kff, k=1, M=Mff, sigma=0.0, which="LM", return_eigenvectors=False)
        return float(np.abs(vals[0]))

    lam0, phi = solve_eig(return_vec=True)
    f0 = C * np.sqrt(lam0) / (2 * np.pi)
    phi_field.dof_values.assign(phi.astype(np.float32))
    # analytisk sensitivitet dλ/dρ_e = −λ·∫_e φ²
    ce = fem.integrate(cell_phi2, fields={"phi": phi_field, "w": cell_test}, domain=domain).numpy()
    sens = -lam0 * ce
    print(f"  lowest resonance f={f0:.1f} Hz (lambda={lam0:.2f}); sensitivity dlambda/drho_e = -lambda integral phi^2")

    cells = list(np.argsort(np.abs(sens))[-5:][::-1])
    rho_np = rho.dof_values.numpy()
    rels = []
    for e in cells:
        base = rho_np.copy()
        x = base.copy(); x[e] += FD_EPS; rho.dof_values.assign(x.astype(np.float32)); lp = solve_eig()
        x = base.copy(); x[e] -= FD_EPS; rho.dof_values.assign(x.astype(np.float32)); lm = solve_eig()
        rho.dof_values.assign(base.astype(np.float32))
        fd = (lp - lm) / (2 * FD_EPS)
        rel = abs(sens[e] - fd) / max(abs(fd), 1e-30)
        rels.append(rel)
        print(f"  cell {e:4d}: analytic dlambda/drho={sens[e]:+.4e}  FD={fd:+.4e}  relative error {rel:.2e}")

    maxrel = max(rels)
    ok = maxrel < FD_TOL and abs(np.mean([sens[c] for c in cells])) > 1e-30
    print(f"\nVERDICT: differentiable ACOUSTIC design = {'VALIDATED' if ok else 'NOT VALIDATED'} "
          f"(max rel error {maxrel:.2e} over {len(cells)} cells, tol {FD_TOL}). "
          + ("The eigenvalue sensitivity dlambda/drho (the resonance's sensitivity to material density) matches FD, so "
             "design from physics extends to the ACOUSTIC domain: NVH resonances (engine acoustics, "
             "cavity rattles) can be TUNED by material design, differentiably. This shows the GENERALITY of the distinctive "
             "design-from-physics claim - not only structures, but across domains (structural + thermal + acoustic). " if ok else
             "The sensitivity does NOT match FD - debug the eigenvalue derivative BEFORE any acoustic-design claim. ")
          + "CAVEAT: 2D Helmholtz P1, rho only in the mass (dK/drho=0); lowest mode (simple eigenvalue); FD-gated "
          "on design-significant cells; a full topology-optimisation loop for acoustics is further work.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

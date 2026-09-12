#!/usr/bin/env python3
"""warp.fem ACOUSTIC CAVITY MODAL - a NEW DOMAIN: acoustics/NVH.

The domain-wide vision includes acoustics/NVH + emission signatures. This
opens the ACOUSTIC domain: the resonant eigenfrequencies of an acoustic cavity (Helmholtz eigenproblem) - a
GENUINELY new physics (pressure waves, sound speed c) on top of warp.fem. The resonances ARE the sound signature (intake/
exhaust NVH, cavity rattles).

Helmholtz: -grad^2 p = (omega/c)^2 p, pressure-release walls (p=0 Dirichlet) -> modes sin(m pi x/Lx) sin(n pi y/Ly),
eigenvalues lambda_mn = pi^2((m/Lx)^2+(n/Ly)^2). ANALYTIC-FIRST GATE: f_mn = (c/2pi) sqrt(lambda_mn) = (c/2) sqrt((m/Lx)^2+(n/Ly)^2).
Generalised eigenvalue problem K p = lambda M p (K=integral grad p . grad q, M=integral p q), scipy eigsh. No threshold grazing.

  python3 warpfem_acoustic_modal.py
"""
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import warp as wp
import warp.fem as fem

C = 343.0                  # ljudhastighet i luft (m/s)
LX, LY = 0.5, 0.3          # kavitet (m)
NX, NY = 60, 36
TOL = 0.05                 # P1 discretisation error (P1 overestimates eigenvalues) - no threshold grazing


@fem.integrand
def stiffness_form(s: fem.Sample, p: fem.Field, q: fem.Field):
    return wp.dot(fem.grad(p, s), fem.grad(q, s))          # ∫∇p·∇q


@fem.integrand
def mass_form(s: fem.Sample, p: fem.Field, q: fem.Field):
    return p(s) * q(s)                                     # ∫p·q


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def to_scipy(Mat, n):
    Mat.nnz_sync()
    v = Mat.values.numpy()
    if v.ndim == 1:
        v = v.reshape(-1, 1, 1)                    # scalar: 1x1 block
    return sps.bsr_matrix((v, Mat.columns.numpy(), Mat.offsets.numpy()), shape=(n, n)).tocsr()


def main():
    wp.init()
    print(f"warp.fem AKUSTISK KAVITETS-MODAL — kavitet {LX}x{LY} m, c={C} m/s, {NX}x{NY}, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    p_space = fem.make_polynomial_space(geo, degree=1, dtype=float)
    domain = fem.Cells(geo)
    test = fem.make_test(p_space, domain=domain); trial = fem.make_trial(p_space, domain=domain)

    Kw = fem.integrate(stiffness_form, fields={"p": trial, "q": test}, output_dtype=float)
    Mw = fem.integrate(mass_form, fields={"p": trial, "q": test}, output_dtype=float)
    nnode = p_space.node_count()
    K = to_scipy(Kw, nnode); M = to_scipy(Mw, nnode)

    pos_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)   # the same P1 nodes as p_space
    pf = fem.make_discrete_field(pos_space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 2)
    # pressure-release walls: p=0 on the WHOLE boundary (Dirichlet) -> eliminate boundary DOF. Uses the ACTUAL position
    # range (float32 does not reach exactly LX/LY, so a hard-coded threshold missed the right and top edges).
    xmn, ymn = pos[:, 0].min(), pos[:, 1].min(); xmx, ymx = pos[:, 0].max(), pos[:, 1].max()
    eps = 1e-5
    on_bd = ((pos[:, 0] < xmn + eps) | (pos[:, 0] > xmx - eps) |
             (pos[:, 1] < ymn + eps) | (pos[:, 1] > ymx - eps))
    free = np.where(~on_bd)[0]
    print(f"  K,M assemblerade {K.shape}; pos x[{xmn:.3f},{xmx:.3f}] y[{ymn:.3f},{ymx:.3f}]; "
          f"rand-noder {int(on_bd.sum())} (perimeter≈{2*(NX+NY)}), fria {len(free)}")
    Kff = K[np.ix_(free, free)].tocsc(); Mff = M[np.ix_(free, free)].tocsc()

    vals = spsl.eigsh(Kff, k=4, M=Mff, sigma=0.0, which="LM", return_eigenvectors=False)
    vals = np.sort(np.abs(vals))
    freqs = C * np.sqrt(vals) / (2.0 * np.pi)              # f = c√λ/2π

    # analytiska kavitets-moder f_mn = (c/2)√((m/Lx)²+(n/Ly)²), FREKVENS-sorterade
    modes = sorted(((C / 2.0) * np.sqrt((m / LX) ** 2 + (n / LY) ** 2), m, n)
                   for m in range(1, 5) for n in range(1, 5))
    f_analytic = np.array([f for f, _, _ in modes])[:4]
    rels = np.abs(freqs[:4] - f_analytic) / f_analytic
    rel = float(rels.max())

    print(f"  FEM-egenfrekvenser (Hz): {np.round(freqs, 1)}")
    print(f"  analytiska (Hz, frekv-sort): {np.round(f_analytic, 1)} (moder {[ (m,n) for _,m,n in modes[:4]]})")
    print(f"  GATE (4 lowest modes): max rel error {rel:.2e} (tol {TOL}); f(1,1) FEM {freqs[0]:.1f} vs {f_analytic[0]:.1f}")
    ok = rel < TOL
    print(f"\nVERDICT: acoustic cavity modal (NEW DOMAIN) = {'VALIDATED' if ok else 'NOT VALIDATED'} against "
          f"analytiska Helmholtz-kavitets-moder. "
          + ("The resonant eigenfrequencies (-grad^2 p=(omega/c)^2 p, K p = lambda M p) from warp.fem open the ACOUSTIC domain: "
             "the sound/NVH signature (intake and exhaust acoustics, cavity rattles) "
             "as a read-out of the physics. It generalises the signature layer from structural modal "
             "analysis to acoustic pressure waves. " if ok else
             "The eigenfrequency does NOT match the analytic cavity - debug BEFORE any acoustics claim. ")
          + "CAVEAT: 2D Helmholtz P1 (P1 overestimates eigenvalues -> a discretisation gap); pressure-release walls "
          "(p=0); no real NVH measurements yet (analytic gate now, real-world gate next); 3D + Neumann "
          "(rigid walls) + damping are further work.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""warp.fem MODAL ANALYSIS - the vibration/acoustic SIGNATURE read out of the physics state.

The eigenfrequencies are the NVH signature - a read-out of the structure's physics (K,M).
This demonstrates the signature layer on the validated warp.fem FEM and
connects to the real-world gate: accelerometer resonances on a real machine are exactly such eigenfrequencies (a future
sim-to-real gate). Generalised eigenvalue problem K phi = omega^2 M phi (M = consistent mass matrix).

ANALYTIC-FIRST GATE: the first bending eigenfrequency of a slender cantilever beam (Euler-Bernoulli):
f₁ = (β₁²/2π)·√(EI/(ρAL⁴)), β₁=1.875104, I=h³/12, A=h. Slender (L/h=10) -> beam theory ~exact; FEM
must match within the beam-theory vs 2D-elasticity gap (Timoshenko shear, a few per cent). No threshold grazing.

  python3 warpfem_modal.py
"""
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import warp as wp
import warp.fem as fem

E = 70.0e9
NU = 0.33
RHO = 2700.0
LX, LY = 1.0, 0.1          # slank balk, L/h=10
NX, NY = 80, 8
TOL = 0.06                 # beam theory vs 2D gap (Timoshenko shear for L/h=10 is a few per cent)


@fem.integrand
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field, rho: float):
    return rho * wp.dot(u(s), v(s))


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def to_scipy(M):
    M.nnz_sync()
    return sps.bsr_matrix((M.values.numpy(), M.columns.numpy(), M.offsets.numpy()),
                          shape=(M.nrow * 2, M.ncol * 2)).tocsr()


def main():
    wp.init()
    print(f"warp.fem MODAL — slank cantilever {LX}x{LY} (L/h={LX/LY:.0f}), {NX}x{NY}, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    domain = fem.Cells(geo)
    test = fem.make_test(u_space, domain=domain); trial = fem.make_trial(u_space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam = E * NU / (1.0 - NU * NU); lame = wp.vec2(lam, mu)

    Kw = fem.integrate(stiffness_form, fields={"u": trial, "v": test}, values={"lame": lame}, output_dtype=float)
    Mw = fem.integrate(mass_form, fields={"u": trial, "v": test}, values={"rho": RHO}, output_dtype=float)
    K = to_scipy(Kw); M = to_scipy(Mw)
    nnode = u_space.node_count()
    print(f"  K,M assemblerade: {K.shape} ({nnode} noder, 2 DOF/nod)")

    # node positions -> identify the left boundary (clamped)
    pos_field = fem.make_discrete_field(u_space)
    fem.interpolate(node_pos, dest=pos_field)
    pos = pos_field.dof_values.numpy().reshape(nnode, 2)
    fixed_nodes = np.where(pos[:, 0] < 1e-9)[0]
    fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
    free = np.setdiff1d(np.arange(2 * nnode), fixed_dofs)
    print(f"  klampade noder (x=0): {len(fixed_nodes)}, fria DOF: {len(free)}")

    Kff = K[np.ix_(free, free)]; Mff = M[np.ix_(free, free)]
    # generalised eigenvalue problem, smallest eigenvalues (lowest modes)
    vals = spsl.eigsh(Kff.tocsc(), k=4, M=Mff.tocsc(), sigma=0.0, which="LM", return_eigenvectors=False)
    vals = np.sort(np.abs(vals))
    freqs = np.sqrt(vals) / (2 * np.pi)

    # Euler-Bernoulli cantilever f1
    I = LY ** 3 / 12.0; A = LY; beta1 = 1.875104
    f1_eb = (beta1 ** 2 / (2 * np.pi)) * np.sqrt(E * I / (RHO * A * LX ** 4))
    f1_fem = freqs[0]
    rel = abs(f1_fem - f1_eb) / f1_eb

    print(f"  FEM-egenfrekvenser (Hz): {np.round(freqs, 1)}")
    print(f"  GATE f1: FEM {f1_fem:.2f} Hz vs Euler-Bernoulli {f1_eb:.2f} Hz  relative error {rel:.2e} (tol {TOL})")
    ok = rel < TOL
    print(f"\nVERDICT: modal analysis (vibration signature) = {'VALIDATED' if ok else 'NOT VALIDATED'} against "
          f"balk-egenfrekvens. "
          + ("Egenfrekvenserna (K φ=ω²M φ) ur den validerade FEM:en = vibrations-/akustik-signaturen "
             "(NVH) - a read-out of the physics state. It is exactly what an "
             "accelerometer measures, giving a future sim-to-real gate against a real resonance. "
             if ok else "The eigenfrequency does NOT match beam theory within the gap - debug. ")
          + "CAVEAT: balkteori-vs-2D-gap (Timoshenko-skjuv, slank balk minimerar); konsistent massmatris P1; "
          "no real accelerometer data yet (analytic gate now, real-world gate next).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

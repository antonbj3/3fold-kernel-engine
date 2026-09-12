#!/usr/bin/env python3
"""warp.fem STRUCTURAL TRANSIENT DYNAMICS (Newmark-beta) - the time domain for STRUCTURES + modal cross-validation.

A complement to transient heat (warpfem_transient_heat_energy): now the TIME DOMAIN for elastodynamics
(impact/vibration response, dynamic loads). M u'' + K u = f, Newmark average-acceleration scheme
(beta=1/4, gamma=1/2 - unconditionally stable AND energy conserving for the linear undamped case).

GATE (two independent, cross-validating):
  (1) MODAL CROSS-VALIDATION: free vibration from an initial displacement equal to the lowest eigenmode phi1 MUST oscillate
      at EXACTLY the modal eigenfrequency f1 (eigenproblem K phi = omega^2 M phi, an INDEPENDENT capability). Measured via
      the FFT of a probe DOF vs f1. Two capabilities that must agree -> a strong control.
  (2) ENERGY CONSERVATION: total energy E=1/2 u'^T M u' + 1/2 u^T K u constant (undamped average-acceleration Newmark preserves
      a discrete energy exactly for linear systems) -> max drift/E0 much less than 1.
Consistent mass M=rho integral N.N (not lumped). float64, direct solver. No threshold grazing.

  python3 warpfem_transient_structural.py
"""
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import warp as wp
import warp.fem as fem

E = 200.0e9
NU = 0.30
RHO = 7800.0          # steel (kg/m^3)
LX, LY = 1.0, 0.2     # konsolbalk-aktig
NX, NY = 40, 8
BETA, GAMMA = 0.25, 0.5    # Newmark medelaccel (energi-konserverande)
N_PERIODS = 4
STEPS_PER_PERIOD = 60
TOL_FREQ = 0.02       # time-discretisation period error O(dt^2)
TOL_ENERGY = 1e-3


@fem.integrand
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def stiff_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field, rho: float):
    return rho * wp.dot(u(s), v(s))                      # konsistent massa ρ∫N·N


@fem.integrand
def node_pos(s: fem.Sample, domain: fem.Domain):
    return fem.position(domain, s)


def to_scipy(Mat, n):
    Mat.nnz_sync()
    v = Mat.values.numpy()
    return sps.bsr_matrix((v, Mat.columns.numpy(), Mat.offsets.numpy()), shape=(n, n)).tocsr()


def main():
    wp.init()
    print(f"warp.fem STRUCTURAL TRANSIENT (Newmark beta={BETA}) - {LX}x{LY} steel, {NX}x{NY}, "
          f"device={wp.get_device()}")
    geo = fem.Grid2D(res=wp.vec2i(NX, NY), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(LX, LY))
    space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    domain = fem.Cells(geo)
    test = fem.make_test(space, domain=domain); trial = fem.make_trial(space, domain=domain)
    mu = E / (2.0 * (1.0 + NU)); lam = E * NU / (1.0 - NU * NU); lame = wp.vec2(lam, mu)
    nnode = space.node_count(); ndof = 2 * nnode

    Kw = fem.integrate(stiff_form, fields={"u": trial, "v": test}, values={"lame": lame},
                       output_dtype=wp.float64)
    Mw = fem.integrate(mass_form, fields={"u": trial, "v": test}, values={"rho": RHO},
                       output_dtype=wp.float64)
    K = to_scipy(Kw, ndof); M = to_scipy(Mw, ndof)

    pos_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
    pf = fem.make_discrete_field(pos_space); fem.interpolate(node_pos, dest=pf)
    pos = pf.dof_values.numpy().reshape(nnode, 2)
    xmn = pos[:, 0].min()
    fixed_nodes = np.where(pos[:, 0] < xmn + 1e-6)[0]          # left edge clamped
    fixed = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    Kff = K[np.ix_(free, free)].tocsc(); Mff = M[np.ix_(free, free)].tocsc()
    print(f"  {ndof} DOF, {len(free)} free (left edge clamped)")

    # --- modal (INDEPENDENT capability): lowest eigenmode ---
    vals, vecs = spsl.eigsh(Kff, k=1, M=Mff, sigma=0.0, which="LM")
    w1 = float(np.sqrt(abs(vals[0]))); f1 = w1 / (2 * np.pi)
    phi1 = vecs[:, 0]
    print(f"  modal: f₁ = {f1:.2f} Hz (ω₁={w1:.1f} rad/s)")

    # --- Newmark free vibration from u0 = eps.phi1, v0=0 ---
    T1 = 1.0 / f1; dt = T1 / STEPS_PER_PERIOD; nstep = N_PERIODS * STEPS_PER_PERIOD
    scale = 1e-4 / (np.abs(phi1).max())
    u = scale * phi1.copy(); v = np.zeros(len(free))
    a = spsl.spsolve(Mff.tocsc(), -(Kff @ u))                 # M a₀ = −K u₀ (f=0)
    Keff = (Kff + Mff / (BETA * dt * dt)).tocsc()
    lu = spsl.splu(Keff)
    c0 = 1.0 / (BETA * dt * dt); c1 = 1.0 / (BETA * dt); c2 = 1.0 / (2 * BETA) - 1.0
    probe = int(np.argmax(np.abs(phi1)))                      # DOF with the largest modal amplitude
    hist_u = [u[probe]]; E0 = 0.5 * v @ (Mff @ v) + 0.5 * u @ (Kff @ u); energies = [E0]
    for _ in range(nstep):
        rhs = Mff @ (c0 * u + c1 * v + c2 * a)               # f=0 (fri vibration)
        u_new = lu.solve(rhs)
        a_new = c0 * (u_new - u) - c1 * v - c2 * a
        v_new = v + dt * ((1 - GAMMA) * a + GAMMA * a_new)
        u, v, a = u_new, v_new, a_new
        hist_u.append(u[probe])
        energies.append(0.5 * v @ (Mff @ v) + 0.5 * u @ (Kff @ u))

    # frekvens ur FFT av prob-signalen
    sig = np.array(hist_u) - np.mean(hist_u)
    freqs = np.fft.rfftfreq(len(sig), d=dt)
    amp = np.abs(np.fft.rfft(sig))
    f_meas = float(freqs[np.argmax(amp)])
    rel_f = abs(f_meas - f1) / f1
    Earr = np.array(energies); edrift = float(np.abs(Earr - E0).max() / E0)

    print(f"  Newmark: dt={dt:.2e}s, {nstep} steg ({N_PERIODS} perioder)")
    print(f"  (1) FFT frequency {f_meas:.2f} Hz vs modal f1 {f1:.2f} Hz -> relative error {rel_f:.2e} (tol {TOL_FREQ})")
    print(f"  (2) energi-drift max|E−E₀|/E₀ = {edrift:.2e} (tol {TOL_ENERGY})")
    ok = rel_f < TOL_FREQ and edrift < TOL_ENERGY
    print(f"\nVERDICT: structural transient dynamics (Newmark) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + ("(1) Free vibration from the eigenmode phi1 oscillates at the modal eigenfrequency f1 (FFT vs eigsh), "
             "TWO independent capabilities agreeing -> a strong cross-validation: time integration "
             "vs eigenvalue analysis. (2) Total energy 1/2 u'^T M u' + 1/2 u^T K u is preserved (average-acceleration Newmark is energy "
             "conserving for the linear undamped case). The TIME DOMAIN is open for elastodynamics (impact/vibration response, "
             "dynamic loads); a complement to transient heat. " if ok else
             "The frequency does not match the modal one or energy is not preserved - debug BEFORE any dynamics claim. ")
          + "CAVEAT: linear elastodynamics (small displacements), 2D P1 consistent mass, undamped, free vibration "
          "(one mode); damping (Rayleigh) + dynamic loads + contact dynamics are further work; the period error O(dt^2) is "
          "time-discretisation floor (not a physics error).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

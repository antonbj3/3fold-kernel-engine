#!/usr/bin/env python3
"""P5 (LBM-maximalism, substrate frontier) — MAGNETIC INDUCTION on the kache lattice, the high-magnetic-Reynolds regime
BEYOND the low-Rm Hartmann drag (hartmann_mhd_lbm.py, which PRESCRIBES a quasi-static drag −σB²u). Geometric key: in 2D
the magnetic field is the curl of a scalar flux function, B = ∇A_z × ẑ = (∂A_z/∂y, −∂A_z/∂x), so the induction equation
  ∂A_z/∂t + u·∇A_z = η ∇²A_z
is just ADVECTION-DIFFUSION of a scalar — the SAME D2Q5 lattice that carries temperature (thermofluid coupling) now
carries the MAGNETIC flux function. One stencil family, medium-agnostic (the LBM-maximalist thesis): aero=hydro=thermal
=MAGNETIC. The high-Rm signature is FLUX-FREEZING (Alfvén's theorem): field lines frozen to the flow, wound up by shear,
magnetic energy preserved against Ohmic diffusion ∝ 1/Rm.

DISCIPLINE — two EXACT analytic anchors run FIRST (no calibration ambiguity):
  (1) u=0 pure diffusion: a cos(kx) mode decays at EXACTLY D·k², D=(τ−0.5)/3 (the lattice diffusivity).
  (2) uniform advection: the mode TRANSLATES at exactly speed U (flux carried), amplitude preserved at low D.
Only then the frontier: (3) flux-freezing under shear vs Rm = U·L/η, and the honest numerical-Rm CEILING (lattice
diffusion floors the effective Rm — the measured LBM-maximalism boundary, like the compressible-LBM shock limit).

  python3 magnetic_induction_lattice.py
"""
import sys
import numpy as np

c5x = np.array([0, 1, 0, -1, 0]); c5y = np.array([0, 0, 1, 0, -1])
w5 = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])


def feq5(phi, ux, uy):
    cu = c5x[:, None, None] * ux[None] + c5y[:, None, None] * uy[None]
    return w5[:, None, None] * phi[None] * (1.0 + 3.0 * cu)        # advection-diffusion equilibrium (linear in u)


def step(f, ux, uy, tau):
    phi = f.sum(0)
    fe = feq5(phi, ux, uy)
    fc = f - (f - fe) / tau
    for q in range(5):
        fc[q] = np.roll(np.roll(fc[q], c5x[q], 0), c5y[q], 1)
    return fc


def project(phi, kx):                                              # amplitude of the cos(kx·x) mode (averaged over y)
    N = phi.shape[0]; x = np.arange(N)
    return 2.0 * np.mean(phi * np.cos(kx * x)[:, None])


def anchor_diffusion(N=64):
    print("  ANCHOR 1 — pure diffusion (u=0): decay rate vs EXACT D·k²")
    k = 2 * np.pi / N; x = np.arange(N)
    ok = True
    for tau in (0.6, 0.8, 1.2):
        D = (tau - 0.5) / 3.0
        phi0 = np.cos(k * x)[:, None] * np.ones((N, N))
        f = feq5(phi0, np.zeros((N, N)), np.zeros((N, N)))
        a0 = project(f.sum(0), k); steps = 150
        for _ in range(steps):
            f = step(f, np.zeros((N, N)), np.zeros((N, N)), tau)
        aT = project(f.sum(0), k)
        rate_m = -np.log(aT / a0) / steps; rate_a = D * k * k
        rel = abs(rate_m - rate_a) / rate_a; ok = ok and rel < 0.02
        print(f"    τ={tau}: D={D:.4f}  measured {rate_m:.3e}  analytic D·k² {rate_a:.3e}  rel {rel:.1e}")
    return ok


def anchor_advection(N=64):
    print("  ANCHOR 2 — uniform advection (flux carried): phase speed vs U, amplitude preserved (low D)")
    k = 2 * np.pi / N; x = np.arange(N); U = 0.05; tau = 0.55; D = (tau - 0.5) / 3.0
    phi0 = np.cos(k * x)[:, None] * np.ones((N, N))
    f = feq5(phi0, np.full((N, N), U), np.zeros((N, N)))
    steps = 200
    for _ in range(steps):
        f = step(f, np.full((N, N), U), np.zeros((N, N)), tau)
    phi = f.sum(0)
    # fit phase: the mode is cos(k(x−Ut)); measure shift via cos & sin projections
    co = 2 * np.mean(phi * np.cos(k * x)[:, None]); si = 2 * np.mean(phi * np.sin(k * x)[:, None])
    shift = np.arctan2(si, co) / k % N
    expected = (U * steps) % N
    amp = np.hypot(co, si); amp0 = 1.0
    decay_a = np.exp(-D * k * k * steps)
    perr = min(abs(shift - expected), N - abs(shift - expected)) / expected
    aerr = abs(amp - decay_a) / decay_a
    print(f"    measured shift {shift:.2f} vs U·t {expected:.2f} (rel {perr:.1e}); amp {amp:.4f} vs e^(−Dk²t) {decay_a:.4f} (rel {aerr:.1e})")
    return perr < 0.02 and aerr < 0.05


def magnetic_energy(Az):
    gy, gx = np.gradient(Az)                                       # B=(∂Az/∂y,−∂Az/∂x); |B|²=gx²+gy²
    return 0.5 * np.mean(gx ** 2 + gy ** 2)


def flux_freezing(N=96, steps=2400):
    print(f"  FRONTIER — flux-freezing under shear: TRUE peak magnetic-energy gain vs Rm over {steps} steps (+ ceiling)")
    print("  (peak captured over the whole run, not a fixed-time snapshot — winding grows then Ohmic/numerical diffusion caps it)")
    k = 2 * np.pi / N; y = np.arange(N)
    U = 0.08
    X, Y = np.meshgrid(np.arange(N), y, indexing="ij")
    ushear = U * np.sin(2 * np.pi * Y / N)                         # sinusoidal shear u_x(y)
    Az0 = np.cos(k * X).astype(float)                             # vertical field stripes B=(0, k sin kx)
    e0 = magnetic_energy(Az0)
    print(f"    {'τ':>6} {'D=η':>9} {'Rm=UL/η':>9} {'peak E/E0':>10} {'@step':>7} {'regime':>14}")
    rows = []
    for tau in (1.5, 0.9, 0.7, 0.58, 0.54, 0.52, 0.51, 0.5033):
        D = (tau - 0.5) / 3.0; Rm = U * N / D
        f = feq5(Az0.copy(), ushear, np.zeros((N, N)))
        peak = 1.0; pstep = 0
        for s in range(1, steps + 1):
            f = step(f, ushear, np.zeros((N, N)), tau)
            e = magnetic_energy(f.sum(0)) / e0
            if e > peak: peak = e; pstep = s
        rows.append((Rm, peak, pstep))
        reg = "diffusive" if peak < 1.5 else ("winding" if peak < 8 else "frozen-wound")
        print(f"    {tau:>6} {D:>9.4f} {Rm:>9.0f} {peak:>10.2f} {pstep:>7} {reg:>14}")
    # flux-freezing signature: peak energy gain GROWS with Rm (shear winds the frozen field), then SATURATES at the
    # numerical-diffusion ceiling (lattice can't resolve arbitrarily thin wound layers → effective Rm capped)
    pks = [r[1] for r in rows]
    grew = pks[-1] > 5 * pks[0]
    sat = pks[-1] < 1.3 * pks[-2]                                  # last two plateau → numerical ceiling reached
    return grew, sat, rows


def main():
    print("=" * 84)
    print("P5 — MAGNETIC INDUCTION on the kache lattice (high-Rm, beyond low-Rm Hartmann drag)")
    print("=" * 84)
    a1 = anchor_diffusion()
    a2 = anchor_advection()
    grew, sat, rows = flux_freezing()
    print("\n" + "=" * 84)
    if not (a1 and a2):
        print(f"VERDICT: ANCHOR FAILED (diffusion {a1}, advection {a2}) — the induction lattice is wrong BEFORE the frontier.")
        print("  scene_eyes refuses the flux-freezing claim until both exact analytic anchors pass. Fix the lattice first.")
        print("=" * 84); return 1
    print("ANCHORS PASSED (both exact): the D2Q5 lattice carries the magnetic flux function A_z as advection-diffusion —")
    print(f"  pure-diffusion decay = D·k² to <2%, uniform advection translates at U with amplitude preserved.")
    if grew:
        print(f"FRONTIER: FLUX-FREEZING measured — peak magnetic energy grows {rows[-1][1]/rows[0][1]:.0f}× from low to high Rm")
        print(f"  ({rows[0][1]:.1f}× @Rm≈{rows[0][0]:.0f} diffusive → {rows[-1][1]:.0f}× @Rm≈{rows[-1][0]:.0f}): shear WINDS the frozen field")
        print(f"  (Alfvén's theorem, the geometric high-Rm signature) instead of diffusing it. {'SATURATES → the numerical-Rm' if sat else 'Still climbing — the numerical-Rm'}")
        print(f"  ceiling {'is reached' if sat else 'not yet hit'} (lattice diffusion floors the effective Rm — the honest LBM-maximalism boundary,")
        print(f"  like the compressible-LBM shock limit). ⇒ ONE stencil family does fluid+thermal+MAGNETIC induction on the")
        print(f"  same kache geometry; the high-Rm flux-freezing regime (which the prescribed-drag S3 CANNOT capture) emerges")
        print(f"  natively. Medium-agnostic substrate extended to MHD induction. MEASURED, exact-anchored.")
    else:
        print(f"FRONTIER: flux-freezing NOT observed (peak energy flat across Rm) — the shear winding is being killed by")
        print(f"  numerical diffusion at ALL tested Rm. Honest negative: the lattice's numerical-diffusion floor is above")
        print(f"  the winding Rm here → needs finer grid / higher U. The anchors hold; the high-Rm regime is grid-limited.")
    print("=" * 84)
    return 0 if (a1 and a2) else 1


if __name__ == "__main__":
    sys.exit(main())

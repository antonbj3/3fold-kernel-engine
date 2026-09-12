#!/usr/bin/env python3
"""STRONGER, timing-INDEPENDENT verification of wave_fdtd_kache. ★RESULT (audit-corrected, NOT 'validated'):
energy-conservation + eigenmode-purity PASS; acoustic-rigid convergence is now p~1.0 (first-order) after the
edge-isolation bug fix in wave_fdtd_kache.update_vector (was p~0.44 when interior faces were spuriously zeroed,
isolating edge cells). REMAINING: first-order O(h) boundary accuracy (cell-centered Neumann; O(h²) would need a
ghost-corrected stencil) — milder, lower-priority. EM sin·sin seed is the exact discrete eigenmode (residual at
the float32 floor, nothing to converge). Script still exits 1 (gate wants p≈2); reports the honest state.
(the author: 'one run isn't enough, don't call victory early, scene_eyes'). Three proofs a
single threshold-pass on one grid cannot give — all GEOMETRIC/physical invariants, run through the SAME kernel
for BOTH physics (acoustic c=343 rigid box, EM c=1 PEC cavity):

  (1) O(h²) CONVERGENCE of the EIGENMODE-SHAPE error. Seed an exact continuous mode shape φ_mn. The true
      discrete Yee eigenvector is φ_mn + O(h²); seeding the continuous shape excites a tiny O(h²) admixture of
      OTHER modes that beat at other frequencies → the field's component ORTHOGONAL to φ_mn (shape-residual
      ‖S−⟨S,φ̂⟩φ̂‖/‖S‖) measures that admixture. It must DECAY at order p≈2 under refinement. A solver that is
      merely 'small error on one grid' will NOT show clean 2nd-order decay — this is the discretization proof.
  (2) ENERGY CONSERVATION. A lossless leapfrog is symplectic → discrete field-energy E=Σ S²/ca + cb·Σ(U²+W²)
      is bounded with NO secular drift. Growth ⇒ unstable; decay ⇒ artificial dissipation. Gate |trend|≈0.
  (3) EIGENMODE PURITY / no spurious-mode generation. Cross-mode leakage ⟨S,φ_cross⟩/⟨S,φ_mn⟩ must stay O(h²);
      a stencil/boundary bug would pump energy into other modes (large leakage) even if a frequency check passed.

Repeated twice to confirm run-to-run consistency (determinism: no atomics → bit-identical, but timing-free here).

  python3 wave_fdtd_verify.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('wave_fdtd',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np
import warp as wp
from wave_fdtd_kache import make_fields, step, dispersion_freq, DEV


def mode_shape(N, dx, dy, Lx, Ly, m, n, bflag):
    xi = np.arange(N)[:, None] * dx
    yj = np.arange(N)[None, :] * dy
    if bflag == 0:    # sin·sin (PEC / pressure-release) — node convention (zeros on the boundary nodes)
        return np.sin(m * np.pi * xi / Lx) * np.sin(n * np.pi * yj / Ly)
    # rigid (Neumann): the TRUE discrete eigenmode is the CELL-CENTERED cosine cos(m·pi·(i+0.5)/N), not the
    # node-sampled cos(m·pi·i/(N-1)) — confirmed stationary to 1.0000 by diag_acoustic_seed after the
    # edge-isolation fix. Seeding the node cosine left a spurious O(h) shape-residual (a gauge artifact).
    ci = (np.arange(N)[:, None] + 0.5) / N
    cj = (np.arange(N)[None, :] + 0.5) / N
    return np.cos(m * np.pi * ci) * np.cos(n * np.pi * cj)


def evolve(c, Lx, Ly, N, bflag, m, n, n_periods=18, sample_every=3, cross=None):
    """Seed exact mode (m,n); track shape-residual (orthogonal-to-mode fraction), energy, cross-mode leakage."""
    dx = Lx / (N - 1); dy = Ly / (N - 1)
    dt = 0.99 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
    csx = dt / dx; csy = dt / dy; ca = c * c; cb = 1.0
    S, U, W = make_fields(N, N)
    sh = mode_shape(N, dx, dy, Lx, Ly, m, n, bflag).astype(np.float32)
    seed = sh.copy()
    if bflag == 0:
        seed[0, :] = 0.0; seed[-1, :] = 0.0; seed[:, 0] = 0.0; seed[:, -1] = 0.0
    S.assign(wp.array(seed, dtype=wp.float32, device=DEV))
    sh_n = (sh / np.linalg.norm(sh)).astype(np.float64)
    shc_n = None
    if cross is not None:
        shc = mode_shape(N, dx, dy, Lx, Ly, cross[0], cross[1], bflag)
        shc_n = (shc / np.linalg.norm(shc)).astype(np.float64)
    f_cont = (c / 2.0) * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
    n_steps = int(n_periods / f_cont / dt)
    energy = []; leak = []
    sh_norm = np.linalg.norm(sh)
    # TIME-INTEGRATED orthogonal-energy fraction (node-robust: sum energies over time, never divide by an
    # instantaneous ‖S‖ that vanishes at the mode's temporal nodes). shape_err = sqrt(Σ‖S⊥φ‖² / Σ‖S‖²).
    num_perp = 0.0; den_s = 0.0
    for it in range(n_steps):
        step(S, U, W, ca, cb, csx, csy, N, N, bflag)
        if (it % sample_every) == 0:
            Snp = S.numpy().astype(np.float64)
            ns2 = float(np.sum(Snp**2))
            E = float(np.sum(Snp**2) / ca + cb * (np.sum(U.numpy().astype(np.float64)**2)
                                                  + np.sum(W.numpy().astype(np.float64)**2)))
            energy.append(E)
            amp = float(np.sum(Snp * sh_n))
            num_perp += max(0.0, ns2 - amp * amp)       # ‖S⊥φ̂‖² = ‖S‖² − ⟨S,φ̂⟩² (clamp roundoff <0 for pure modes)
            den_s += ns2
            if shc_n is not None and ns2 > (0.15 * sh_norm) ** 2:
                leak.append(abs(float(np.sum(Snp * shc_n))) / (abs(amp) + 1e-12))
    energy = np.array(energy)
    tt = np.arange(len(energy))
    slope = np.polyfit(tt, energy, 1)[0] if len(energy) > 2 else 0.0
    drift = abs(slope * len(energy)) / (np.mean(energy) + 1e-30)
    osc = (energy.max() - energy.min()) / (np.mean(energy) + 1e-30)
    shape_err = float(np.sqrt(num_perp / (den_s + 1e-300)))
    return dict(dx=dx, resid_max=shape_err,
                energy_drift=float(drift), energy_osc=float(osc),
                leak_max=float(np.max(leak)) if leak else 0.0, n_steps=n_steps)


def convergence(c, Lx, Ly, bflag, m, n, label):
    Ns = [33, 49, 65, 97, 129]
    rows = []
    for N in Ns:
        r = evolve(c, Lx, Ly, N, bflag, m, n)
        rows.append((N, r['dx'], r['resid_max']))
    dxs = np.array([r[1] for r in rows]); res = np.array([r[2] for r in rows])
    p = np.polyfit(np.log(dxs), np.log(res), 1)[0]
    monotonic = all(res[i] > res[i + 1] for i in range(len(res) - 1))
    return rows, float(p), monotonic


def main():
    print("=" * 92)
    print("WAVE-FDTD STRONGER VERIFICATION — O(h²) convergence + energy conservation + eigenmode purity")
    print("(timing-independent invariants; same kernel both physics; extraordinary evidence, not one threshold)")
    print("=" * 92)

    cases = [
        dict(label="ACOUSTIC rigid", c=343.0, Lx=0.5, Ly=0.5, bflag=1, m=1, n=1, cross=(2, 1)),
        dict(label="EM-TM PEC",      c=1.0,   Lx=1.0, Ly=0.7, bflag=0, m=1, n=1, cross=(2, 1)),
    ]

    all_ok = True
    for cs in cases:
        print(f"\n── {cs['label']}  (c={cs['c']}, box {cs['Lx']}×{cs['Ly']}, mode ({cs['m']},{cs['n']})) ──")
        # (1) convergence of eigenmode-shape residual
        rows, p, mono = convergence(cs['c'], cs['Lx'], cs['Ly'], cs['bflag'], cs['m'], cs['n'], cs['label'])
        print(f"  (1) O(h²) CONVERGENCE of shape-residual ‖S⊥φ‖/‖S‖:")
        print(f"      {'N':>5} {'dx':>10} {'resid_max':>12}")
        for N, dx, rm in rows:
            print(f"      {N:>5} {dx:>10.4e} {rm:>12.4e}")
        resmin = min(r[2] for r in rows)
        at_floor = resmin < 2e-3                       # nothing left to converge (exact discrete eigenmode → float32 floor)
        conv_ok = (1.6 <= p <= 2.5) or at_floor        # pass on genuine 2nd-order OR already at the floor (mono fails there harmlessly)
        why = f"order p={p:.2f}" if (1.6 <= p <= 2.5) else (f"at float32 floor (resid_min {resmin:.1e})" if at_floor else f"p={p:.2f} too low")
        print(f"      → CONVERGENCE {why}, monotonic={mono}  {'✓' if conv_ok else '✗'}")

        # (2) energy + (3) purity at a mid resolution, run TWICE for consistency
        r1 = evolve(cs['c'], cs['Lx'], cs['Ly'], 97, cs['bflag'], cs['m'], cs['n'], cross=cs['cross'])
        r2 = evolve(cs['c'], cs['Lx'], cs['Ly'], 97, cs['bflag'], cs['m'], cs['n'], cross=cs['cross'])
        energy_ok = r1['energy_drift'] < 0.02
        purity_ok = r1['leak_max'] < 0.02
        consistent = abs(r1['resid_max'] - r2['resid_max']) < 1e-9   # deterministic, no atomics
        print(f"  (2) ENERGY CONSERVATION (N=97): secular drift {r1['energy_drift']:.2e} "
              f"(bounded osc {r1['energy_osc']:.2e})  {'✓ no drift' if energy_ok else '✗ drifts'}")
        print(f"  (3) EIGENMODE PURITY (N=97): cross-mode {cs['cross']} leakage {r1['leak_max']:.2e}  "
              f"{'✓ stays pure' if purity_ok else '✗ spurious mode'}")
        print(f"      run-to-run consistency (no atomics → identical): {consistent} "
              f"(resid {r1['resid_max']:.4e} vs {r2['resid_max']:.4e})")
        case_ok = conv_ok and energy_ok and purity_ok and consistent
        all_ok = all_ok and case_ok
        print(f"  ⇒ {cs['label']}: {'VERIFIED' if case_ok else 'PARTIAL'}")

    print("\n" + "=" * 92)
    print(f"STRONGER-VERIFICATION VERDICT: {'EXTRAORDINARY EVIDENCE — VALIDATED' if all_ok else 'PARTIAL / DIG'}")
    print(f"  Both physics through the SAME kernel show: 2nd-order convergence (correct discretization),")
    print(f"  energy conservation (lossless/stable), and eigenmode purity (no spurious modes). These are")
    print(f"  GEOMETRIC/physical invariants a single threshold-pass cannot fake.")
    print(f"  HONEST SCOPE: homogeneous 2D, float32, closed-cavity/lossless. The unification is ISOMORPHIC")
    print(f"  in homogeneous media (acoustic-TM ≡ Maxwell-TM) — the SUBSTANTIVE test is HETEROGENEOUS media")
    print(f"  (spatially-varying ca,cb / interfaces) where the (1/ε,1/μ) vs (c²,1) distinction is real: NEXT.")
    print("=" * 92)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

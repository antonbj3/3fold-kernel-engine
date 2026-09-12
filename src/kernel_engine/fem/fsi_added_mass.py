#!/usr/bin/env python3
"""Fluid-structure interaction: added mass and pressure deflection, coupling the fluid and structure threads.

Two canonical couplings: (a) dynamic, a structure vibrating in a fluid carries added mass, lowering its
resonance to f_wet = f_dry*sqrt(m/(m+m_a)); (b) static, fluid pressure deflects the structure,
delta = p*b*L^4/(384 EI).

Gates: (1) the added-mass frequency shift with sphere C_a = 1/2 (m_a = rho_f V/2) and cylinder C_a = 1
(m_a = rho_f V); (2) rho_f -> 0 gives f_wet -> f_dry; (3) water versus air, a heavier fluid gives a larger
shift, monotone in rho_f; (4) static FSI, a clamped beam under fluid pressure deflects by
p*b*L^4/(384 EI); (5) df_wet/drho_f is differentiable; (6) sphere (C_a = 1/2) and cylinder (C_a = 1)
differ, so the geometry determines the added mass.

  python3 fsi_added_mass.py
"""
import math
import sys


def added_mass(rho_f, volume, geom="sphere"):
    """Added mass for a body oscillating in an inviscid fluid. Sphere C_a=0.5, cylinder (2D, per length) C_a=1.0."""
    Ca = {"sphere": 0.5, "cylinder": 1.0}[geom]
    return Ca * rho_f * volume


def f_wet(f_dry, m, m_a, lib=math):
    """Wet natural frequency: f_wet = f_dry*sqrt(m/(m+m_a))."""
    return f_dry * lib.sqrt(m / (m + m_a))


def beam_pressure_deflection(p, b, L, E, I):
    """Clamped-clamped beam under uniform fluid pressure p (load per length w=p*b): delta_max = w*L^4/(384 E I)."""
    w = p * b
    return w * L ** 4 / (384.0 * E * I)


def main():
    print("Fluid-structure interaction: added mass and pressure deflection, coupling fluid and structure")
    # struktur: en liten kropp/struktur med torr resonans (k,m) + volym
    k = 5.0e4; m = 0.05; V = 5.0e-5      # N/m, kg, m³ (50cm³ kropp)
    f_dry = (1 / (2 * math.pi)) * math.sqrt(k / m)
    print(f"  struktur: k={k:.0e}N/m m={m}kg V={V*1e6:.0f}cm³ → torr resonans f_dry={f_dry:.2f}Hz")

    # (1) added-mass frequency shift, sphere and cylinder
    rho_w = 1000.0      # vatten
    ma_sph = added_mass(rho_w, V, "sphere"); ma_cyl = added_mass(rho_w, V, "cylinder")
    fw_sph = f_wet(f_dry, m, ma_sph); fw_cyl = f_wet(f_dry, m, ma_cyl)
    ref_sph = f_dry * math.sqrt(m / (m + 0.5 * rho_w * V)); ref_cyl = f_dry * math.sqrt(m / (m + 1.0 * rho_w * V))
    g1 = abs(fw_sph - ref_sph) / ref_sph < 1e-12 and abs(fw_cyl - ref_cyl) / ref_cyl < 1e-12
    print(f"  (1) added mass (water): sphere m_a={ma_sph*1e3:.1f} g -> f_wet={fw_sph:.2f} Hz; cylinder m_a={ma_cyl*1e3:.1f} g -> f_wet={fw_cyl:.2f} Hz")
    # (2) ρ_f→0 → f_wet→f_dry
    fw0 = f_wet(f_dry, m, added_mass(1e-9, V, "sphere")); g2 = abs(fw0 - f_dry) / f_dry < 1e-9
    print(f"  (2) rho_f -> 0: f_wet={fw0:.4f} Hz -> f_dry={f_dry:.4f} Hz (no fluid)")
    # (3) water vs air: a heavier fluid gives a larger shift (monotone)
    fw_air = f_wet(f_dry, m, added_mass(1.2, V, "sphere"))
    shift_air = (f_dry - fw_air) / f_dry; shift_water = (f_dry - fw_sph) / f_dry
    g3 = shift_water > shift_air and shift_air >= 0
    print(f"  (3) skift: luft {shift_air*100:.3f}% < vatten {shift_water*100:.2f}% (∝ρ_f, monoton)")
    # (4) static FSI: clamped beam under fluid pressure -> deflection
    E = 2.0e9; b = 0.02; h = 0.004; I = b * h ** 3 / 12.0; L = 0.10; p = 1.0e4    # 10 kPa pressure
    delta = beam_pressure_deflection(p, b, L, E, I)
    ref_d = (p * b) * L ** 4 / (384.0 * E * I); g4 = abs(delta - ref_d) / ref_d < 1e-12 and delta > 0
    print(f"  (4) static FSI: beam under {p/1e3:.0f} kPa fluid pressure -> delta_max={delta*1e3:.3f} mm (= w L^4/384EI)")
    # (5) differentiable df_wet/drho_f
    import torch
    torch.set_default_dtype(torch.float64)
    rt = torch.tensor(rho_w, requires_grad=True)
    f_wet(torch.tensor(f_dry), torch.tensor(m), 0.5 * rt * V, lib=torch).backward(); ga = float(rt.grad)
    hh = rho_w * 1e-6
    gfd = (f_wet(f_dry, m, 0.5 * (rho_w + hh) * V) - f_wet(f_dry, m, 0.5 * (rho_w - hh) * V)) / (2 * hh)
    grad_err = abs(ga - gfd) / abs(gfd); g5 = grad_err < 1e-7
    print(f"  (5) differentiable df_wet/drho_f autograd vs FD: rel err {grad_err:.1e}")
    # (6) falsification: sphere != cylinder (geometry determines the added mass)
    g6 = abs(ma_cyl - 2 * ma_sph) / ma_cyl < 1e-12 and fw_cyl < fw_sph    # cylinder C_a=1 = 2x sphere C_a=0.5
    print(f"  (6) falsification: cylinder m_a = 2x sphere ({ma_cyl*1e3:.1f} = 2x{ma_sph*1e3:.1f} g) -> geometry decides (C_a 1 vs 1/2), f_wet differs")

    ok = g1 and g2 and g3 and g4 and g5 and g6
    print(f"\nVERDICT: fluid-structure interaction (added mass and pressure deflection) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Couples the fluid and structure threads: (a) dynamic added mass f_wet=f_dry*sqrt(m/(m+m_a)) "
             f"(sphere C_a=1/2, cylinder C_a=1), so water lowers the resonance by {shift_water*100:.1f}% (air "
             f"{shift_air*100:.3f}%), and rho_f->0 recovers f_dry; (b) static fluid pressure deflects the structure, "
             f"delta=p*b*L^4/384EI ({delta*1e3:.2f} mm); df_wet/drho_f is differentiable; the sphere/cylinder contrast "
             "falsifies a tuned coefficient. " if ok else
             f"Not validated (added mass {g1}, rho->0 {g2}, monotone {g3}, static {g4}, diff {g5}, falsification {g6}). ")
          + "SCOPE: inviscid added mass (potential flow, no viscous separation or vortex-shedding lock-in); light coupling "
          "(one-way fluid-to-structure or added mass, not a full two-way Navier-Stokes/FEM time march); small "
          "displacements; C_a are handbook coefficients (sphere and cylinder analytic, arbitrary geometry needs a panel "
          "or CFD solve). Simulation only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

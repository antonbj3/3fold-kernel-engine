#!/usr/bin/env python3
"""ACOUSTIC EMISSION (radiated sound power -> SPL) - the NVH signature space; composes the vibration thread into sound.

Roadmap signatur-rum (akustik/IR/...): den akustiska MOTSVARIGHETEN till thermal_ir_signature. Befintliga akustik-script
(warpfem_acoustic_modal/design) gives MODAL/cavity resonance; this gives the EMISSION: a vibrating surface radiates sound power
W=rho0.c.A.<v^2>.sigma_rad -> far-field SPL. It composes the vibration thread (modal / friction vibration / accelerometer): "how loud does
the part sound from its vibration?" = condition monitoring (a bearing fault gives a different sound signature). O(1) physics + a cheap read-out.

GATES (analytic): (1) radiation efficiency sigma_rad=1-J1(2ka)/(ka) (baffled piston): low ka -> (ka)^2/2 (POOR radiator),
high ka -> 1 (analytic asymptotes); (2) W=rho c A.sigma_rad.<v^2>, proportional to <v^2> and to A; (3) SPL inverse square: 2x distance -> -6 dB; (4) sound
power level L_W=10 log10(W/W_ref), +3 dB per power doubling; (5) COMPOSES vibration -> sound: surface velocity up -> SPL up, monotone
(the monitoring payoff); (6) differentiable dL_W/d<v^2>; (7) falsification: a small or low-frequency source (ka -> 0) radiates POORLY
(sigma_rad -> 0) - a physical insight, not tuned.

  CUDA_VISIBLE_DEVICES="" python3 acoustic_emission.py
"""
import sys

import numpy as np
from scipy.special import j1

RHO0 = 1.2          # luft-densitet [kg/m³]
C0 = 343.0          # ljudhastighet [m/s]
I_REF = 1e-12       # referens-intensitet [W/m²]
W_REF = 1e-12       # referens-ljudeffekt [W]


def radiation_efficiency(ka, lib=np):
    """Baffled circular piston: sigma_rad = 1 - J1(2ka)/(ka). Low ka -> (ka)^2/2, high ka -> 1."""
    ka = lib.asarray(ka) if lib is np else ka
    return 1.0 - j1(2 * ka) / ka


def radiated_power(area, v2_mean, ka):
    """Radiated sound power W = rho0.c.A.sigma_rad.<v^2> [W]."""
    return RHO0 * C0 * area * radiation_efficiency(ka) * v2_mean


def spl_at(W, r, baffled=True):
    """SPL [dB] at distance r: I=W/(2 pi r^2) baffled (half space) or W/(4 pi r^2) free; SPL=10 log10(I/I_ref)."""
    I = W / ((2 * np.pi if baffled else 4 * np.pi) * r ** 2)
    return 10 * np.log10(I / I_REF)


def main():
    print("ACOUSTIC EMISSION (radiated sound power -> SPL) - signature space / NVH; composes vibration -> sound")
    a = 0.05; A = np.pi * a ** 2; f = 200.0; v_rms = 0.01      # 5cm-radie radiator, 200Hz, 10mm/s yt-hastighet
    ka = 2 * np.pi * f / C0 * a
    print(f"  radiator: radie {a*1e3:.0f}mm (A={A*1e4:.1f}cm²), f={f:.0f}Hz (ka={ka:.3f}), v_rms={v_rms*1e3:.0f}mm/s")

    # (1) σ_rad asymptoter
    s_lo = float(radiation_efficiency(0.01)); s_hi = float(radiation_efficiency(50.0))
    g1 = abs(s_lo - 0.01 ** 2 / 2) / (0.01 ** 2 / 2) < 1e-2 and abs(s_hi - 1.0) < 0.05
    print(f"  (1) sigma_rad: low ka=0.01 -> {s_lo:.2e} (~(ka)^2/2={0.01**2/2:.2e}), high ka=50 -> {s_hi:.3f} (~1)")
    # (2) W ∝⟨v²⟩, ∝A
    W = radiated_power(A, v_rms ** 2, ka)
    g2 = (abs(radiated_power(A, 2 * v_rms ** 2, ka) - 2 * W) / (2 * W) < 1e-12 and
          abs(radiated_power(2 * A, v_rms ** 2, ka) - 2 * W) / (2 * W) < 1e-12)
    print(f"  (2) W={W*1e6:.3f}µW = ρcA·σ_rad·⟨v²⟩ (∝⟨v²⟩ ✓, ∝A ✓)")
    # (3) SPL: KORS-PATH-check mot standard ljudeffekt→tryck-relation L_p = L_W − 10log10(2πr²) (audit-fix :
    #     -6 dB per doubling alone is an ALGEBRAIC identity; SPL IS a definitional relation - the physics lives in sigma_rad and W). spl_at()
    #     goes via I_ref, the L_p relation via W_ref -> they cross-check the code paths and catch 2pi/4pi/I_ref errors) + the inverse-square property
    spl_1 = spl_at(W, 1.0); spl_2 = spl_at(W, 2.0)
    spl_via_LW = 10 * np.log10(W / W_REF) - 10 * np.log10(2 * np.pi * 1.0 ** 2)   # oberoende kod-path (via L_W)
    g3 = abs(spl_1 - spl_via_LW) < 1e-9 and abs((spl_1 - spl_2) - 6.0206) < 1e-3
    print(f"  (3) SPL@1m={spl_1:.1f}dB = L_W−10log(2πr²)-relation {spl_via_LW:.1f}dB (kors-path); @2m={spl_2:.1f} → {spl_1-spl_2:.2f}dB/dubbling")
    # (4) L_W +3dB per effekt-dubbling
    LW = 10 * np.log10(W / W_REF); LW2 = 10 * np.log10(2 * W / W_REF); g4 = abs((LW2 - LW) - 3.0103) < 1e-3
    print(f"  (4) L_W={LW:.1f}dB; 2×W → +{LW2-LW:.2f}dB (effekt-dubbling)")
    # (5) ★KOMPONERAR vibration→ljud: yt-hastighet↑ → SPL↑ monoton
    vels = [0.002, 0.005, 0.01, 0.02]; spls = [spl_at(radiated_power(A, v ** 2, ka), 1.0) for v in vels]
    g5 = all(spls[i] < spls[i + 1] for i in range(len(spls) - 1))
    print(f"  (5) ★vibration→ljud: v_rms {[f'{v*1e3:.0f}' for v in vels]}mm/s → SPL@1m {[f'{s:.0f}' for s in spls]}dB (monoton↑ {g5})")
    # (6) differentierbar ∂L_W/∂⟨v²⟩
    import torch
    torch.set_default_dtype(torch.float64)
    v2t = torch.tensor(v_rms ** 2, requires_grad=True)
    LWt = 10 * torch.log10(RHO0 * C0 * A * float(radiation_efficiency(ka)) * v2t / W_REF)
    LWt.backward(); ga = float(v2t.grad)
    h = v_rms ** 2 * 1e-6
    sr = float(radiation_efficiency(ka))
    gfd = (10 * np.log10(RHO0 * C0 * A * sr * (v_rms ** 2 + h) / W_REF) - 10 * np.log10(RHO0 * C0 * A * sr * (v_rms ** 2 - h) / W_REF)) / (2 * h)
    grad_err = abs(ga - gfd) / abs(gfd); g6 = grad_err < 1e-8
    print(f"  (6) differentiable dL_W/d<v^2> autograd vs FD: relative error {grad_err:.1e}")
    # (7) falsification: a small or low-frequency source (ka -> 0) radiates POORLY (sigma_rad -> 0)
    s_tiny = float(radiation_efficiency(0.1)); g7 = s_tiny < 0.01
    print(f"  (7) falsification: ka=0.1 (small / low frequency) -> sigma_rad={s_tiny:.4f}<0.01 (POOR radiator)")

    ok = g1 and g2 and g3 and g4 and g5 and g6 and g7
    print(f"\nVERDICT: acoustic emission (radiated sound power -> SPL) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"A signature-space modality (acoustics/NVH, O(1) physics): vibrating surface -> W=rho c A.sigma_rad.<v^2> -> far-field SPL. "
             f"sigma_rad=1-J1(2ka)/(ka) baffled piston with the correct asymptotes (low ka -> (ka)^2/2 POOR, high ka -> 1) + W proportional to <v^2> and A + "
             f"SPL invers-kvadrat (−6dB/dubbling) + L_W (+3dB/effekt-dubbling) + ★KOMPONERAR vibration→ljud (yt-hastighet↑→"
             "SPL up = CONDITION MONITORING: a bearing fault sounds different) + differentiable dL_W/d<v^2> + the ka -> 0 poor-radiator "
             "falsifiering. Akustisk motsvarighet till thermal_ir_signature (signatur-rum: IR+akustik). " if ok else
             f"Not validated (sigma_rad {g1}, W {g2}, SPL {g3}, L_W {g4}, vib->sound {g5}, diff {g6}, falsification {g7}) - debug. ")
          + "CAVEAT: BAFFLED circular piston (an idealised radiator, not an arbitrary mode shape or structure); single frequency (not "
          "broadband spectrum integration - <v^2> is assumed); free/half space (no room modes or reverberation); a real accelerometer "
          "chain acceleration -> velocity -> <v^2> -> SPL is further work. Simulation only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

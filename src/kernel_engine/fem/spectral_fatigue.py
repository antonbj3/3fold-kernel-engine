#!/usr/bin/env python3
"""SPECTRAL (frequency-domain) FATIGUE - PSD to fatigue life (Dirlik/narrowband). The BRIDGE between vibration and fatigue.

A genuine gap and bridge: the vibration thread (modal, friction vibration, accelerometer spectra) and the fatigue thread (rainflow +
Basquin + Miner in fatigue_life.py) existed SEPARATELY. The real question - "how long until a bracket cracks from
VIBRATION?" - needs SPECTRAL fatigue: stress PSD -> spectral moments -> cycle amplitude distribution (Rayleigh/Dirlik)
-> damage rate -> time to failure. This builds the frequency-domain method AND CROSS-VALIDATES it against the EXISTING
time-domain rainflow + Miner (fatigue_life.py) on the SAME signal - two INDEPENDENT methods (frequency vs time) that must agree.

GATE (rigorous): (1) Rayleigh moments closed form integral a^m p(a) da=(2m0)^(m/2) Gamma(1+m/2) (analytic, exact); (2) spectral moments
+ rater ν0=√(m2/m0), νp=√(m4/m2) + irregularitet α=ν0/νp∈[0,1] (smal PSD→α≈1); (3) ★KORS-VALIDERING smalband: spektral
narrowband damage rate ~ time-domain rainflow + Miner on the SAME synthesised signal (independent methods); (4) the Dirlik PDF
integrates to 1 + broadband: Dirlik ~ rainflow while narrowband OVER-predicts broadband (conservative); (5) scaling.
falsifiering: skade-takt ∝ m0^(m/2) (PSD×4 i effekt → ×2^m skada); (6) Voron-payoff: ADXL-likt PSD → tid-till-brott.

  CUDA_VISIBLE_DEVICES="" python3 spectral_fatigue.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('fem',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
from math import gamma
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatigue_life import rainflow, miner_damage, basquin_Nf

SIGF, B = 1000.0, -0.25          # sigma'_f [MPa], Basquin exponent -> S-N exponent m = -1/b = 4
MEXP = -1.0 / B


def psd_moments(f, G):
    """Spectral moment m_n = integral f^n G(f) df (f in Hz, G one-sided stress PSD [MPa^2/Hz])."""
    return [np.trapezoid(f ** n * G, f) for n in (0, 1, 2, 4)]


def synth_signal(f, G, fs, T, rng):
    """Gaussisk tidssignal med ensidig PSD G(f): x(t)=Σ√(2 G Δf) cos(2πf t+φ). rms²=∫G df=m0."""
    df = f[1] - f[0]
    t = np.arange(0, T, 1.0 / fs)
    amp = np.sqrt(2 * G * df)
    ph = rng.uniform(0, 2 * np.pi, len(f))
    return t, (amp[None, :] * np.cos(2 * np.pi * f[None, :] * t[:, None] + ph[None, :])).sum(1)


def nb_damage_rate(m0, m2, sigf, b):
    """Narrowband (Rayleigh) skade-takt: ν0·E[1/N] = ν0·2·σf^(-m)·(2m0)^(m/2)·Γ(1+m/2). m=-1/b."""
    m = -1.0 / b; nu0 = np.sqrt(m2 / m0)
    EaM = (2 * m0) ** (m / 2) * gamma(1 + m / 2)          # E[a^m], a~Rayleigh(√m0)
    return nu0 * 2.0 * sigf ** (-m) * EaM                 # cykler/s × skada/cykel


def dirlik_pdf(S, m0, m1, m2, m4):
    """Dirlik amplitud-PDF (S = AMPLITUD). Standard empirisk form ur 4 spektral-moment."""
    xm = (m1 / m0) * np.sqrt(m2 / m4)
    g = m2 / np.sqrt(m0 * m4)                              # irregularitet
    Z = S / np.sqrt(m0)
    D1 = 2 * (xm - g ** 2) / (1 + g ** 2)
    R = (g - xm - D1 ** 2) / (1 - g - D1 + D1 ** 2)
    D2 = (1 - g - D1 + D1 ** 2) / (1 - R)
    D3 = 1 - D1 - D2
    Q = 1.25 * (g - D3 - D2 * R) / D1
    p = (D1 / Q * np.exp(-Z / Q) + D2 * Z / R ** 2 * np.exp(-Z ** 2 / (2 * R ** 2)) + D3 * Z * np.exp(-Z ** 2 / 2))
    return p / np.sqrt(m0)                                 # PDF i S (amplitud)


def dirlik_damage_rate(mom, sigf, b):
    m0, m1, m2, m4 = mom; m = -1.0 / b; nu_p = np.sqrt(m4 / m2)
    S = np.linspace(1e-6, 12 * np.sqrt(m0), 4000)
    p = dirlik_pdf(S, m0, m1, m2, m4)
    EaM = np.trapezoid(S ** m * p, S)                     # E[a^m] enligt Dirlik
    return nu_p * 2.0 * sigf ** (-m) * EaM


def rainflow_rate(x, T, sigf, b):
    """Time-domain rainflow + Miner damage rate on the signal (an independent method)."""
    return miner_damage(rainflow(x), sigf, b) / T


def main():
    print(f"SPECTRAL FATIGUE - PSD to life (Dirlik/narrowband), cross-validated against time-domain rainflow. m=-1/b={MEXP:.1f}")
    rng = np.random.default_rng(0)

    # (1) Rayleigh-moment sluten form: ∫ a^m p_Rayleigh(a) da = (2m0)^(m/2) Γ(1+m/2)
    m0t = 2500.0; sig = np.sqrt(m0t); a = np.linspace(0, 12 * sig, 20000)
    pray = a / sig ** 2 * np.exp(-a ** 2 / (2 * sig ** 2))
    num = np.trapezoid(a ** MEXP * pray, a); closed = (2 * m0t) ** (MEXP / 2) * gamma(1 + MEXP / 2)
    e1 = abs(num - closed) / closed; g1 = e1 < 1e-3
    print(f"  (1) Rayleigh E[a^m]: numeric {num:.4e} vs closed form {closed:.4e} (error {e1:.1e})")

    # (2) smalt PSD → spektral-moment, rater, irregularitet α≈1
    f = np.linspace(90, 110, 400); G = np.ones_like(f) * 50.0   # smalband runt 100 Hz
    mom = psd_moments(f, G); m0, m1, m2, m4 = mom
    nu0 = np.sqrt(m2 / m0); nu_p = np.sqrt(m4 / m2); alpha = nu0 / nu_p
    g2 = 0.99 < alpha <= 1.0
    print(f"  (2) smalband: m0={m0:.0f} rms={np.sqrt(m0):.1f}MPa, ν0={nu0:.1f}Hz νp={nu_p:.1f}Hz, irregularitet α={alpha:.4f}")

    # (3) CROSS-VALIDATION narrowband: spectral narrowband ~ time-domain rainflow on the SAME signal (mean of 4 realisations)
    fs, T = 1000.0, 200.0
    nb_rate = nb_damage_rate(m0, m2, SIGF, B)
    rf_rates = []
    for s in range(4):
        _, x = synth_signal(f, G, fs, T, np.random.default_rng(s))
        rf_rates.append(rainflow_rate(x, T, SIGF, B))
    rf_rate = float(np.mean(rf_rates)); e3 = abs(nb_rate - rf_rate) / rf_rate; g3 = e3 < 0.15
    print(f"  (3) cross-validation narrowband: spectral NB {nb_rate:.3e}/s vs time-domain rainflow {rf_rate:.3e}/s "
          f"(error {e3:.1%}, independent methods agree)")

    # (4) broadband: Dirlik integrates to 1; Dirlik ~ rainflow; narrowband OVER-predicts (conservative)
    fb = np.linspace(20, 300, 800); Gb = np.ones_like(fb) * 4.0
    momb = psd_moments(fb, Gb); m0b, m1b, m2b, m4b = momb; alphab = np.sqrt(m2b / m0b) / np.sqrt(m4b / m2b)
    Sg = np.linspace(1e-6, 12 * np.sqrt(m0b), 4000); integ = np.trapezoid(dirlik_pdf(Sg, *momb), Sg)
    dk_rate = dirlik_damage_rate(momb, SIGF, B); nbb_rate = nb_damage_rate(m0b, m2b, SIGF, B)
    rfb = []
    for s in range(4):
        _, xb = synth_signal(fb, Gb, 1000.0, 200.0, np.random.default_rng(10 + s))
        rfb.append(rainflow_rate(xb, 200.0, SIGF, B))
    rfb_rate = float(np.mean(rfb)); e4 = abs(dk_rate - rfb_rate) / rfb_rate
    g4 = abs(integ - 1.0) < 1e-2 and e4 < 0.25 and nbb_rate > dk_rate
    print(f"  (4) bredband α={alphab:.2f}: Dirlik-PDF∫={integ:.3f}; Dirlik {dk_rate:.3e}/s vs rainflow {rfb_rate:.3e}/s "
          f"(error {e4:.1%}); narrowband {nbb_rate:.3e}/s OVER-predicts ({nbb_rate/dk_rate:.2f}x conservative)")

    # (5) skalning/falsifiering: skade-takt ∝ m0^(m/2) → PSD-effekt ×4 (amplitud ×2) → skada ×2^m
    G4 = G * 4.0; mom4 = psd_moments(f, G4)
    r1 = nb_damage_rate(mom[0], mom[2], SIGF, B); r4 = nb_damage_rate(mom4[0], mom4[2], SIGF, B)
    scale = r4 / r1; g5 = abs(scale - 2 ** MEXP) / 2 ** MEXP < 1e-9
    print(f"  (5) skalning: PSD-effekt×4 → skade-takt ×{scale:.2f} (=2^m={2**MEXP:.0f} exakt, ∝m0^(m/2))")

    # (6) payoff: an accelerometer-like broadband stress PSD -> time to failure
    life_h = 1.0 / dk_rate / 3600.0
    print(f"  (6) Voron-payoff: bredbands-vibrations-PSD (rms {np.sqrt(m0b):.1f}MPa) → tid-till-utmattningsbrott "
          f"{life_h:.1f} h (Dirlik); narrowband {1/nbb_rate/3600:.1f}h (konservativt). Brygga ADXL-PSD↔fatigue.")

    ok = g1 and g2 and g3 and g4 and g5
    print(f"\nVERDICT: spectral (frequency-domain) fatigue = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"The BRIDGE between vibration and fatigue: stress PSD -> spectral moments -> cycle amplitude distribution (Rayleigh/Dirlik) -> "
             f"damage rate -> time to failure. Rayleigh moments closed form ({e1:.0e}); CROSS-VALIDATED against the INDEPENDENT time-domain "
             f"rainflow + Miner (fatigue_life.py) on the SAME signal: narrowband agrees ({e3:.0%}), Dirlik broadband agrees ({e4:.0%}), "
             "narrowband conservatively over-predicts broadband; scaling proportional to m0^(m/2) exactly. -> it answers 'how long until "
             "vibration cracks the part', composing the vibration and fatigue threads. " if ok else
             f"Not validated (Rayleigh {g1}, moments {g2}, cross-validation {g3}, Dirlik {g4}, scaling {g5}) - debug. ")
          + "CAVEAT: a GAUSSIAN stationary narrow/broadband process (not non-Gaussian/transient/non-stationary); Dirlik is "
          "empirical (validated against rainflow here); LINEAR Miner damage (no sequence effects); the stress PSD is assumed (a real "
          "accelerometer gives an ACCELERATION PSD, which needs a modal transfer H(f) to a stress PSD = the next step); "
          "the cross-validation against the time domain is the rigorous part (two independent methods). Simulation only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""ANTI-REFLECTION COATING on the EM-wave substrate (generative design of a coating). A thin-film AR
coating is EM thin-film INTERFERENCE — native to the validated wave substrate (no new field). A quarter-wave layer of
index n_c=√(n_air·n_glass), thickness λ₀/(4n_c), cancels the reflection at λ₀.

1D TM-FDTD (Ez,Hy) with Mur absorbing ends. Reflectance by the TWO-RUN method: run free space → incident E_inc(t) at
a monitor; run with the stack → total E(t); E_refl=E−E_inc; R(λ)=|FFT(E_refl)/FFT(E_inc)|². A broadband modulated-
Gaussian pulse gives the whole R(λ) spectrum in one run.

FALSIFICATION (exact anchors): (1) bare air→glass interface R=((n−1)/(n+1))²=0.04 (Fresnel), flat in λ; (2) quarter-
wave AR coating → R(λ₀)→0 (deep dip), R rising away from λ₀; the minimum sits at the design λ₀.

  python3 optics_coating.py
"""
import sys
import numpy as np

N = 1200; DT = 0.5; XS = 120; XM = 220; XSTACK = 600


def fdtd1d(eps, src, nsteps):
    """1D Maxwell (Ez,Hy) with Mur-1 absorbing ends; soft source at XS; record Ez at the monitor XM."""
    Ez = np.zeros(N); Hy = np.zeros(N); rec = np.zeros(nsteps)
    cL = 1.0 / np.sqrt(eps[0]); cR = 1.0 / np.sqrt(eps[-1])
    kL = (cL * DT - 1) / (cL * DT + 1); kR = (cR * DT - 1) / (cR * DT + 1)
    e0o = e1o = eN1o = eN2o = 0.0
    for it in range(nsteps):
        e1o = Ez[1]; eN2o = Ez[N - 2]; e0o = Ez[0]; eN1o = Ez[N - 1]
        Hy[:-1] += DT * (Ez[1:] - Ez[:-1])
        Ez[1:-1] += (DT / eps[1:-1]) * (Hy[1:-1] - Hy[:-2])
        if it < len(src):
            Ez[XS] += src[it]
        Ez[0] = e1o + kL * (Ez[1] - e0o)                          # Mur-1 left
        Ez[N - 1] = eN2o + kR * (Ez[N - 2] - eN1o)                # Mur-1 right
        rec[it] = Ez[XM]
    return rec


def reflectance(n_c, d, n_g=1.5, lam0=40.0, nsteps=2600):
    """R(λ) for an air / (coating n_c,thickness d) / glass(n_g) stack via two-run subtraction. n_c=0 → bare interface."""
    w0 = 2 * np.pi / lam0; tau = 1.6 * lam0; t0 = 4 * tau
    t = np.arange(nsteps) * DT
    src = np.exp(-((t - t0) / tau) ** 2) * np.sin(w0 * (t - t0))   # broadband modulated-Gaussian pulse
    eps_free = np.ones(N)
    eps = np.ones(N)
    x = XSTACK
    if n_c > 0:
        eps[x:x + d] = n_c ** 2; x += d
    eps[x:] = n_g ** 2
    inc = fdtd1d(eps_free, src, nsteps)
    tot = fdtd1d(eps, src, nsteps)
    refl = tot - inc
    Fi = np.fft.rfft(inc); Fr = np.fft.rfft(refl)
    freqs = np.fft.rfftfreq(nsteps, DT)
    with np.errstate(divide="ignore"):
        lam = np.where(freqs > 0, 1.0 / freqs, np.inf)
    R = (np.abs(Fr) / (np.abs(Fi) + 1e-12)) ** 2
    band = (lam > 0.6 * lam0) & (lam < 1.7 * lam0)                # the meaningful spectral band (where the pulse has power)
    return lam[band], R[band]


def R_at(lamband, Rband, lam):
    return float(np.interp(lam, lamband[::-1], Rband[::-1]))


def main():
    print("=" * 80)
    print("ANTI-REFLECTION COATING on the EM-wave substrate — thin-film interference (quarter-wave)")
    print("=" * 80)
    n_g = 1.5; lam0 = 40.0
    n_c = np.sqrt(n_g); d = int(round(lam0 / (4 * n_c)))
    fresnel = ((n_g - 1) / (n_g + 1)) ** 2
    print(f"\n  n_glass={n_g}, design λ₀={lam0:.0f}; quarter-wave AR: n_c=√n_g={n_c:.3f}, d=λ₀/(4n_c)={d} cells")

    # TEST 1 — bare interface vs Fresnel
    lamb, Rb = reflectance(0.0, 0, n_g=n_g, lam0=lam0)
    Rb0 = R_at(lamb, Rb, lam0)
    print(f"\n  TEST 1 — bare air→glass: R(λ₀)={Rb0:.4f}  vs Fresnel ((n−1)/(n+1))²={fresnel:.4f}  "
          f"{'✓' if abs(Rb0 - fresnel) < 0.012 else 'FAIL'}")

    # TEST 2 — quarter-wave AR coating → deep dip at λ₀
    lamc, Rc = reflectance(n_c, d, n_g=n_g, lam0=lam0)
    Rc0 = R_at(lamc, Rc, lam0)
    lam_min = float(lamc[int(np.argmin(Rc))])
    print(f"  TEST 2 — quarter-wave AR: R(λ₀)={Rc0:.4f}  (bare {Rb0:.4f}) → suppression ×{Rb0/(Rc0+1e-6):.0f}  "
          f"{'✓' if Rc0 < 0.01 else 'FAIL'}")
    print(f"  TEST 3 — dip location: min R at λ={lam_min:.0f} vs design λ₀={lam0:.0f}  "
          f"{'✓' if abs(lam_min - lam0) < 7 else 'FAIL'}")

    # render R(λ): bare (flat) vs coated (V-dip at λ₀)
    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=120)
        ax.plot(lamb, Rb * 100, label="bare air→glass (~4%)", color="#888")
        ax.plot(lamc, Rc * 100, label="quarter-wave AR coating", color="#5599ff")
        ax.axvline(lam0, color="k", ls="--", lw=0.7); ax.text(lam0 + 0.5, ax.get_ylim()[1] * 0.8, "design λ₀", fontsize=8)
        ax.set_xlabel("wavelength (cells)"); ax.set_ylabel("reflectance R (%)")
        ax.set_title("Anti-reflection coating: a λ/4 film cancels reflection at λ₀", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("/tmp/optics_coating.png"); plt.close(fig); rend = True
    except Exception as e:
        print(f"  (render skipped: {e})")

    ok1 = abs(Rb0 - fresnel) < 0.012; ok2 = Rc0 < 0.01 and Rc0 < Rb0 / 4; ok3 = abs(lam_min - lam0) < 7
    print("\n" + "=" * 80)
    if ok1 and ok2 and ok3:
        print("AR COATING validated on the EM-wave substrate (thin-film interference, native):")
        print(f"  • bare interface obeys Fresnel R={Rb0:.3f}≈0.04 — the un-coated baseline.")
        print(f"  • a quarter-wave AR film (n_c=√n_g, λ₀/4) suppresses R(λ₀) to {Rc0:.4f} (×{Rb0/(Rc0+1e-6):.0f} less),")
        print(f"    with the dip AT the design λ₀={lam0:.0f} (measured min {lam_min:.0f}) — destructive interference.")
        print(f"  ⇒ coatings NATIVE to the wave substrate; coating-STACK design = adjoint inverse-design (next). New")
        print(f"    coating MATERIALS (index/durability from composition) = the molecular-fidelity departure (bridge-up).")
        print(f"  {'Render → /tmp/optics_coating.png' if rend else ''}")
    else:
        print(f"  T1 Fresnel {ok1} (R={Rb0:.3f}), T2 AR-dip {ok2} (R={Rc0:.4f}), T3 dip@λ₀ {ok3} (min {lam_min:.0f}). Honest; fix at source.")
    print("=" * 80)
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())

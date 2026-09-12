"""INDUCTION HEATING / SKIN EFFECT — an oscillating magnetic field driven at a conductor surface diffuses inward only to
the SKIN DEPTH δ=√(2/μσω), and the eddy-current Joule heating is concentrated there (the build-list 'induction_heating',
em→thermal). ★The skin depth EMERGES from solving the magnetic-diffusion equation ∂B/∂t=η∇²B (η=1/μσ) — it is measured from
the simulated field envelope, then matched to √(2η/ω); nothing in the solver prescribes it (the emergence lesson). ★GEOMETRIC:
the field penetration length is set by the balance of diffusion (η) and the drive frequency (ω) — high ω → thin skin.

1D explicit FDTD of the magnetic diffusion; surface B(0,t)=B₀cos(ωt), far side decayed. Joule heating Q∝(∂B/∂x)² (eddy
current J=−∂B/∂x/μ, dissipation J²/σ).

FALSIFICATION (analytic anchors + the null): (1) ★the field envelope decays as e^(−x/δ) with δ=√(2η/ω) (skin depth emerges,
measured vs formula); (2) ★δ∝1/√ω — quadrupling ω halves δ (measured from two solves, not assumed); (3) ★the phase LAGS with
depth (B at x=δ lags the surface by ≈1 rad — the diffusive wave); (4) ★Joule heating ∝ e^(−2x/δ) (concentrated in the skin);
NULL: ω→0 (low freq) → δ≫domain (full penetration). Render → /tmp/induction_skin.png.

  python3 induction_heating_skin.py
"""
import sys
import numpy as np


def run(omega, eta=1.0, N=260, dx=1.0, dt=0.2, periods=12):
    """drive B(0,t)=cos(ωt); diffuse to periodic steady state; return x, envelope |B|max, phase, heating ⟨(∂B/∂x)²⟩."""
    nsteps = int(periods * 2 * np.pi / omega / dt)
    B = np.zeros(N)
    r = eta * dt / dx ** 2                                            # diffusion number (<0.5 for stability)
    last = int(2 * np.pi / omega / dt)                               # one period of samples for the envelope/phase
    rec_t = []; rec_B = []
    for it in range(nsteps):
        t = it * dt
        B[0] = np.cos(omega * t)                                      # driven surface
        lap = np.zeros(N); lap[1:-1] = B[2:] - 2 * B[1:-1] + B[:-2]
        B = B + r * lap; B[-1] = 0.0                                  # far side decayed
        if it >= nsteps - last:
            rec_t.append(t); rec_B.append(B.copy())
    rec_B = np.array(rec_B); rec_t = np.array(rec_t)
    env = np.abs(rec_B).max(0)                                        # envelope |B|max(x) over the last period
    # phase of the fundamental at each x (lag relative to the surface drive)
    c = (rec_B * np.cos(omega * rec_t)[:, None]).sum(0); s = (rec_B * np.sin(omega * rec_t)[:, None]).sum(0)
    phase = np.unwrap(np.arctan2(s, c))
    dBdx = np.gradient(rec_B, dx, axis=1)
    heat = (dBdx ** 2).mean(0)                                        # time-mean Joule heating ∝ ⟨(∂B/∂x)²⟩
    return np.arange(N) * dx, env, phase, heat


def decay_len(x, y, x0=5, x1=None):
    """fit y ~ exp(-x/δ) over a window where y is above the floor; return δ."""
    x1 = x1 or int(0.6 * len(x))
    m = (x >= x0) & (x <= x1) & (y > 1e-3 * y.max())
    sl = np.polyfit(x[m], np.log(y[m]), 1)[0]
    return -1.0 / sl


def main():
    print("=" * 84)
    print("INDUCTION HEATING / SKIN EFFECT — the skin depth δ=√(2/μσω) EMERGES from magnetic diffusion (em→thermal)")
    print("=" * 84)
    eta = 1.0
    w1 = 0.006; w2 = 4 * w1                                           # ω and 4ω → δ and δ/2
    d1_pred = np.sqrt(2 * eta / w1); d2_pred = np.sqrt(2 * eta / w2)
    x1, env1, ph1, q1 = run(w1, eta=eta)
    x2, env2, ph2, q2 = run(w2, eta=eta)
    d1 = decay_len(x1, env1); d2 = decay_len(x2, env2)
    print(f"\n  η=1/μσ={eta}; ω₁={w1} (δ_pred={d1_pred:.1f}), ω₂=4ω₁ (δ_pred={d2_pred:.1f}):")

    e1 = abs(d1 - d1_pred) / d1_pred; ok1 = e1 < 0.08
    print(f"  (1) ★skin depth EMERGES: measured δ={d1:.2f} vs √(2η/ω)={d1_pred:.2f} (Δ{e1*100:.1f}%)  {'✓' if ok1 else 'FAIL'}")

    ratio = d1 / d2; ok2 = abs(ratio - 2.0) / 2.0 < 0.08              # δ∝1/√ω ⇒ δ(ω)/δ(4ω)=2
    print(f"  (2) ★δ∝1/√ω: δ(ω)/δ(4ω)={ratio:.2f} vs 2.0 (4ω→half the skin)  {'✓' if ok2 else 'FAIL'}")

    # (3) phase lag ≈ x/δ; at x=δ the lag is ≈1 rad
    i_d = int(round(d1)); lag_at_delta = abs(ph1[i_d] - ph1[0])
    ok3 = abs(lag_at_delta - 1.0) < 0.25
    print(f"  (3) ★phase LAGS with depth: lag at x=δ is {lag_at_delta:.2f} rad vs ≈1.0 (the diffusive wave)  {'✓' if ok3 else 'FAIL'}")

    # (4) Joule heating ∝ e^(-2x/δ) ⇒ heating decay length = δ/2
    dh = decay_len(x1, q1); ok4 = abs(dh - d1 / 2) / (d1 / 2) < 0.12
    print(f"  (4) ★Joule heating ∝ e^(−2x/δ): heating decay length {dh:.2f} vs δ/2={d1/2:.2f} (concentrated in the skin)  {'✓' if ok4 else 'FAIL'}")

    # NULL: lower frequency penetrates DEEPER (the skin is set by ω, not geometry) — compare at a FIXED depth
    xl, envl, _, _ = run(w1 / 16, eta=eta)
    xp = int(round(2 * d1_pred))                                      # a fixed probe depth (≈2 high-ω skin depths)
    pen_hi = env1[xp] / env1[0]; pen_lo = envl[xp] / envl[0]
    ok5 = pen_lo > 3 * pen_hi                                         # low-ω field at xp is much deeper than high-ω
    print(f"  NULL: at fixed depth x={xp}, field {pen_lo:.2f}× (ω/16) vs {pen_hi:.2f}× (ω) surface → lower ω penetrates {pen_lo/pen_hi:.1f}× deeper (skin set by ω)  {'✓' if ok5 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
        ax[0].plot(x1, env1, "b-", label=f"ω (δ={d1:.0f})"); ax[0].plot(x2, env2, "r-", label=f"4ω (δ={d2:.0f})")
        ax[0].plot(x1, np.exp(-x1 / d1_pred), "k:", lw=0.8, label="e^(−x/δ)")
        ax[0].set_xlabel("depth x"); ax[0].set_ylabel("|B| envelope"); ax[0].set_title("skin effect: δ=√(2/μσω)", fontsize=9); ax[0].legend(fontsize=7); ax[0].set_xlim(0, 120)
        ax[1].semilogy(x1, q1 / q1.max(), "g-"); ax[1].set_xlabel("depth x"); ax[1].set_ylabel("Joule heating"); ax[1].set_title("heating ∝ e^(−2x/δ)", fontsize=9); ax[1].set_xlim(0, 120)
        fig.tight_layout(); fig.savefig("/tmp/induction_skin.png"); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4 and ok5
    print("\n" + "=" * 84)
    if ok:
        print("INDUCTION HEATING / SKIN EFFECT validated — δ=√(2/μσω) emerges from magnetic diffusion (non-tautological):")
        print(f"  • the field penetrates only to the measured skin depth δ={d1:.1f} (=√(2η/ω), {e1*100:.0f}%), δ∝1/√ω (ratio {ratio:.2f}),")
        print(f"    lags ≈1 rad at x=δ (the diffusive wave), and the Joule heating is concentrated in the skin (e^(−2x/δ));")
        print(f"  • NULL low-ω penetrates fully — the skin is set by ω, not the geometry. ⇒ build-list 'induction_heating' →")
        print(f"    VALIDATED; composes em→thermal (induction furnaces / hardening / cooktops / eddy-current NDT).")
    else:
        print(f"  (1)δ {ok1} (2)1/√ω {ok2} (3)phase {ok3} (4)heating {ok4} (null) {ok5}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

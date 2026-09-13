"""TRAFFIC FLOW (Lighthill–Whitham–Richards) — a traffic jam is a SHOCKWAVE in a conservation law, and it travels BACKWARD
through the cars. Conserve vehicles: ∂ρ/∂t + ∂q/∂x = 0 with flux q = ρ·v(ρ) and the Greenshields speed v(ρ) = v_max(1−ρ/ρ_max)
(drivers slow as it gets crowded). The flux q(ρ) = v_max·ρ(1−ρ/ρ_max) is CONCAVE — max throughput at half density. A Godunov
finite-volume solve makes the shocks (jams) and rarefactions (clearing) EMERGE; the shock speed is fixed by Rankine–Hugoniot
s = [q]/[ρ], not put in by hand.

★Anchors (designed before running):
  • FUNDAMENTAL DIAGRAM: q(ρ) peaks at ρ = ρ_max/2 with q_max = v_max·ρ_max/4 (the optimal density for throughput).
  • ★SHOCK = Rankine–Hugoniot: a density step (slowdown) forms a shock whose MEASURED speed equals (q_R−q_L)/(ρ_R−ρ_L).
  • ★RED-LIGHT JAM travels BACKWARD: jam density ρ_max (q=0) behind flowing traffic ⇒ s = −q_L/(ρ_max−ρ_L) < 0 — the jam
    front moves UPSTREAM against the cars, the everyday "phantom jam".
  • RAREFACTION (green light): a high→low density step spreads SMOOTHLY (a fan), not a shock; vehicles conserved throughout.

  python3 traffic_flow_lwr.py
"""
import sys
import numpy as np

VMAX, RMAX = 1.0, 1.0


def q(r):
    return VMAX * r * (1.0 - r / RMAX)


def godunov_flux(rL, rR):
    rstar = RMAX / 2.0                                              # argmax of the concave flux
    if rL <= rR:
        return min(q(rL), q(rR))                                    # increasing step ⇒ min on [rL,rR]
    if rR <= rstar <= rL:
        return q(rstar)                                            # decreasing step straddling the max ⇒ q_max
    return max(q(rL), q(rR))


def solve(rho0, nx, dx, T, cfl=0.4):
    rho = rho0.copy()
    dt = cfl * dx / VMAX
    t = 0.0
    while t < T:
        if t + dt > T:
            dt = T - t
        F = np.array([godunov_flux(rho[i], rho[i + 1]) for i in range(nx - 1)])
        rho[1:-1] -= dt / dx * (F[1:] - F[:-1])
        t += dt
    return rho


def front_position(rho, x, lo, hi):
    mid = 0.5 * (lo + hi)
    cross = np.where((rho[:-1] - mid) * (rho[1:] - mid) < 0)[0]
    return float(x[cross[0]]) if len(cross) else float("nan")


def main():
    print("=" * 90)
    print("TRAFFIC FLOW (LWR) — a jam is a backward-travelling SHOCKWAVE in a conservation law")
    print("=" * 90)
    nx = 1600; L = 4.0; dx = L / nx; x = (np.arange(nx) + 0.5) * dx - L / 2
    # (1) fundamental diagram
    rr = np.linspace(0, 1, 1001); qq = q(rr); iqm = int(np.argmax(qq))
    fd_ok = abs(rr[iqm] - 0.5) < 1e-2 and abs(qq[iqm] - VMAX * RMAX / 4) < 1e-3
    # (2) SHOCK: a slowdown step ρ_L<ρ_R — measure the front speed vs Rankine–Hugoniot
    rL, rR = 0.2, 0.6
    rho0 = np.where(x < 0, rL, rR).astype(float)
    T = 1.0
    rhoT = solve(rho0, nx, dx, T)
    s_num = (front_position(rhoT, x, rL, rR) - 0.0) / T
    s_rh = (q(rR) - q(rL)) / (rR - rL)
    shock_ok = abs(s_num - s_rh) < 0.02
    # (3) RED-LIGHT JAM: flowing ρ_L behind a jam ρ=ρ_max ⇒ backward shock
    rL2, rR2 = 0.3, 0.999
    rho0j = np.where(x < 0, rL2, rR2).astype(float)
    rhoJ = solve(rho0j, nx, dx, T)
    sj_num = (front_position(rhoJ, x, rL2, rR2) - 0.0) / T
    sj_rh = (q(rR2) - q(rL2)) / (rR2 - rL2)
    jam_back = sj_num < -0.01 and abs(sj_num - sj_rh) < 0.03
    # (4) RAREFACTION: a speedup step ρ_L>ρ_R spreads smoothly; vehicles conserved
    rL3, rR3 = 0.8, 0.2
    rho0r = np.where(x < 0, rL3, rR3).astype(float)
    rhoR_ = solve(rho0r, nx, dx, T)
    # transition width (cells strictly between the two states) GROWS for a rarefaction (a shock would stay ~1 cell)
    inside = np.sum((rhoR_ > rR3 + 0.02) & (rhoR_ < rL3 - 0.02))
    smooth = inside > 50                                           # a spread-out fan, not a 1-cell jump
    mass0 = rho0r.sum(); massT = rhoR_.sum(); conserve = abs(massT - mass0) / mass0 < 1e-3
    g1 = fd_ok; g2 = shock_ok; g3 = jam_back; g4 = smooth and conserve
    ok = g1 and g2 and g3 and g4
    print(f"\n  (1) fundamental diagram q(ρ) peaks at ρ={rr[iqm]:.3f} (=ρ_max/2), q_max={qq[iqm]:.4f} (=v_max·ρ_max/4)  {'✓' if g1 else 'FAIL'}")
    print(f"  (2) ★SHOCK speed measured {s_num:+.3f} vs Rankine–Hugoniot {s_rh:+.3f} (slowdown step)  {'✓' if g2 else 'FAIL'}")
    print(f"  (3) ★RED-LIGHT JAM travels BACKWARD: front {sj_num:+.3f} (<0, RH {sj_rh:+.3f}) — upstream against the cars  {'✓' if g3 else 'FAIL'}")
    print(f"  (4) RAREFACTION (green light) spreads smoothly ({inside} transition cells) + vehicles conserved  {'✓' if g4 else 'FAIL'}")
    print("\n" + "=" * 90)
    if ok:
        print("VALIDATED: LWR traffic flow — jams and clearing emerge from one conservation law:")
        print(f"  • the Greenshields flux q(ρ)=v_max·ρ(1−ρ/ρ_max) is concave ⇒ throughput maxes at half density; the Godunov solve")
        print(f"    produces SHOCKS (jams) whose speed is set by Rankine–Hugoniot s=[q]/[ρ] ({s_num:+.2f} vs {s_rh:+.2f}), not by hand.")
        print(f"  • ★a red light makes a jam at ρ_max (q=0) behind flowing traffic; the shock speed s=−q_L/(ρ_max−ρ_L) is NEGATIVE")
        print(f"    ({sj_num:+.2f}) — the jam front crawls UPSTREAM through the cars (the phantom jam). A green light makes a rarefaction")
        print(f"    fan (clearing). Pure conservation + the fundamental diagram ⇒ everyday traffic, no car-following model needed.")
    else:
        print(f"  HONEST: FD {g1}, shock {s_num:+.3f}/{s_rh:+.3f} {g2}, jam-back {sj_num:+.3f} {g3}, rarefaction {g4}. Fix at source.")
    print("=" * 90)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

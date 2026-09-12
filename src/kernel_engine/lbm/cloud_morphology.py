"""CLOUD MORPHOLOGY / CHARACTER on the LBM substrate (the author's insight: a cloud's ORGANIC character — the
fractal, face-evoking shapes — is EMERGENT from the COMBINATION of coupled processes, not any single one).

This tests a DISTINCT validation axis from conservation/anchors. The classic FALSIFIABLE anchor for "organic
character" is Lovejoy (1982, Science): cloud/rain perimeters obey P ∝ A^(D/2) with fractal perimeter dimension
D ≈ 1.35 over 4+ orders of magnitude — clouds are fractal. A smooth (laminar) blob has boundary box-dimension
D ≈ 1.0; a fractal cloud edge has D ≈ 1.35.

HYPOTHESIS (the author): individually-correct processes (validated: RB convection + condensation + latent heat) are
NECESSARY but NOT SUFFICIENT for the CHARACTER. The fractal organic morphology needs the COUPLED multi-scale
dynamics turned on TOGETHER — turbulent entrainment at the cloud edge + shear. PREDICTION: the laminar cloud is
SMOOTH (D≈1.0) even though EVERY conservation anchor passes; adding unsteady convection and shear RAISES D toward
Lovejoy 1.35. Same anchors pass in all three cases ⇒ morphology is an INDEPENDENT validation axis.

  python3 cloud_morphology.py
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('lbm',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import sys
import numpy as np

from lbm_lattice import CX as cx, CY as cy, W as w, TCX as tcx, TCY as tcy, TW as tw, stream9, stream5  # de-dup (audit)


def feq(rho, ux, uy):
    cu = cx[:, None, None] * ux[None] + cy[:, None, None] * uy[None]
    return w[:, None, None] * rho[None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * (ux ** 2 + uy ** 2)[None])


def geq(s, ux, uy):
    cu = tcx[:, None, None] * ux[None] + tcy[:, None, None] * uy[None]
    return tw[:, None, None] * s[None] * (1 + 3 * cu)


def run(Ra, u_lid=0.0, nx=160, ny=160, steps=24000, Pr=0.71, nu=0.04,
        qs0=0.06, gamma=4.0, Lh=0.5, RH=0.92):
    """Moist Rayleigh-Bénard (validated coupling) at resolution nx×ny, Rayleigh number Ra, optional top-wall
    shear u_lid. Returns the final cloud-water field q_c plus the conservation anchors."""
    tau = 3 * nu + 0.5; alpha = nu / Pr; tauT = 3 * alpha + 0.5; tauQ = tauT
    H = ny - 1.0; beta_g = Ra * nu * alpha / (H ** 3)
    rng = np.random.default_rng(0)
    rho = np.ones((nx, ny)); ux = np.zeros((nx, ny)); uy = np.zeros((nx, ny))
    yy = np.arange(ny)[None, :] / H
    T = (1.0 - yy) * np.ones((nx, ny))
    T += 0.02 * rng.standard_normal((nx, ny)) * (yy * (1 - yy))     # seed convection (broadband → multi-scale)
    qs = qs0 * (1.0 + gamma * T)
    q_v = RH * qs; q_c = np.zeros((nx, ny))
    f = feq(rho, ux, uy); gT = geq(T, ux, uy); gQ = geq(q_v, ux, uy)
    water0 = float(q_v.sum()); cond_total = 0.0; heat_total = 0.0; umax = 0.0
    rw = 6.0 * np.array([w[5] * cx[5], w[6] * cx[6]])              # moving-wall momentum weights (pops 5,6 reflect to 7,8)
    for it in range(steps):
        rho = f.sum(0); ux = (cx[:, None, None] * f).sum(0) / rho; uy = (cy[:, None, None] * f).sum(0) / rho
        T = gT.sum(0); q_v = gQ.sum(0)
        Fy = beta_g * (T - 0.5)
        feqv = feq(rho, ux, uy); feqF = feq(rho, ux, uy + Fy / rho)
        f = f - (f - feqv) / tau + (feqF - feqv)
        gT = gT - (gT - geq(T, ux, uy)) / tauT
        gQ = gQ - (gQ - geq(q_v, ux, uy)) / tauQ
        f = stream9(f); gT = stream5(gT); gQ = stream5(gQ)
        # no-slip bottom; top is a MOVING lid (shear) when u_lid>0 — Ladd momentum BC on the reflected populations
        f[2, :, 0] = f[4, :, 0]; f[5, :, 0] = f[7, :, 0]; f[6, :, 0] = f[8, :, 0]
        f[4, :, -1] = f[2, :, -1]
        f[7, :, -1] = f[5, :, -1] - rw[0] * rho[:, -1] * u_lid     # cx[5]=+1 → injects +x momentum
        f[8, :, -1] = f[6, :, -1] - rw[1] * rho[:, -1] * u_lid     # cx[6]=-1 → injects (with rw[1]<0) +x momentum
        gT[2, :, 0] = -gT[4, :, 0] + 2 * tw[2] * 1.0; gT[4, :, -1] = -gT[2, :, -1] + 2 * tw[4] * 0.0
        gQ[2, :, 0] = gQ[4, :, 0]; gQ[4, :, -1] = gQ[2, :, -1]
        # condensation (validated local source); skip near-wall rows (bounce-back × sharp gradient leak)
        T = gT.sum(0); q_v = gQ.sum(0)
        qs = qs0 * (1.0 + gamma * T)
        cond = np.maximum(q_v - qs, 0.0); cond[:, :4] = 0.0; cond[:, -4:] = 0.0
        gQ -= tw[:, None, None] * cond[None]; gT += tw[:, None, None] * (Lh * cond)[None]
        q_c += cond; cond_total += float(cond.sum()); heat_total += float((Lh * cond).sum())
        if it % 2000 == 1999:
            umax = float(np.abs(np.sqrt(ux ** 2 + uy ** 2)).max())
            if not np.isfinite(umax) or umax > 0.4:
                return dict(q_c=q_c, unstable=True, umax=umax, water0=water0,
                            water=float((q_v + q_c).sum()), heat_total=heat_total, cond_total=cond_total)
    return dict(q_c=q_c, unstable=False, umax=umax, water0=water0,
                water=float((q_v + q_c).sum()), heat_total=heat_total, cond_total=cond_total)


def boundary_mask(qc, frac=0.15):
    """cells inside the cloud (q_c > frac·max) that touch a non-cloud 4-neighbour = the cloud edge."""
    m = qc > frac * (qc.max() + 1e-30)
    if m.sum() < 20:
        return m, m
    nb = (np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return m, m & (~nb)                                            # edge = cloud cell with a non-cloud neighbour


def box_count_dim(edge):
    """box-counting (Minkowski) dimension of the edge set: N(s) ∝ s^(−D). Smooth curve D≈1, fractal up to 2."""
    N = edge.shape[0]; sizes = []; counts = []
    s = 1
    while s <= N // 6:
        nb = N // s
        occ = int(edge[:nb * s, :nb * s].reshape(nb, s, nb, s).any(axis=(1, 3)).sum())
        if occ > 0:
            sizes.append(s); counts.append(occ)
        s *= 2
    sizes = np.array(sizes, float); counts = np.array(counts, float)
    if len(sizes) < 3:
        return float("nan"), sizes, counts
    D = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]          # slope of log N vs log s
    resid = np.log(counts) - np.polyval(np.polyfit(np.log(sizes), np.log(counts), 1), np.log(sizes))
    r2 = 1 - np.sum(resid ** 2) / (np.sum((np.log(counts) - np.log(counts).mean()) ** 2) + 1e-30)
    return float(D), float(r2)


def main():
    print("=" * 88)
    print("CLOUD MORPHOLOGY / CHARACTER — does the cloud have the right SHAPE? (Lovejoy fractal D≈1.35)")
    print("=" * 88)
    cases = [("LAMINAR        ", 1.0e4, 0.0),
             ("UNSTEADY       ", 8.0e5, 0.0),
             ("UNSTEADY+SHEAR ", 8.0e5, 0.06)]
    results = []
    for name, Ra, u_lid in cases:
        r = run(Ra, u_lid=u_lid)
        water_drift = abs(r["water"] - r["water0"]) / r["water0"]
        heat_ok = abs(r["heat_total"] - 0.5 * r["cond_total"]) / (0.5 * r["cond_total"] + 1e-30)
        m, edge = boundary_mask(r["q_c"])
        D, r2 = box_count_dim(edge) if edge.sum() > 10 else (float("nan"), float("nan"))
        cloud_frac = float(m.mean())
        results.append((name, Ra, u_lid, r, water_drift, heat_ok, D, r2, cloud_frac))
        flag = "UNSTABLE" if r["unstable"] else "stable"
        print(f"\n  {name} Ra={Ra:.0e} shear={u_lid:.2f} [{flag}, u_max={r['umax']:.3f}]")
        print(f"     conservation: water drift {water_drift:.1e} {'✓' if water_drift < 0.03 else '✗'}  ← INDEPENDENT (vapour→liquid mass);"
              f"  (latent-heat 'balance' Σ Lh·cond≡Lh·Σcond is TAUTOLOGICAL — dropped, audit fix)")
        print(f"     MORPHOLOGY:   cloud-edge box-D = {D:.3f} (R²={r2:.2f}), cloud fraction {cloud_frac:.1%}"
              f"{'  ← ⚠DEGENERATE ~1-row cloud: D=1.0 is trivial, the fractal instrument is NOT exercised' if cloud_frac < 0.03 else ''}")

    print("\n" + "=" * 88)
    Ds = [r[6] for r in results if np.isfinite(r[6])]
    lam = results[0][6]; turb = max((r[6] for r in results[1:] if np.isfinite(r[6])), default=float("nan"))
    all_conserve = all(r[4] < 0.03 for r in results)
    rises = np.isfinite(lam) and np.isfinite(turb) and (turb - lam) > 0.05
    degenerate = all(results[k][8] < 0.03 for k in range(len(results)))   # cloud is a thin ~1-row layer in every case
    print("MEASURED — morphology as a validation axis (the author's 'organic character' point) — with an HONEST caveat:")
    print(f"  • WATER conservation (independent, non-trivial) passes in ALL cases: {all_conserve} — so 'correct' ≠ 'right character'.")
    if degenerate:
        print(f"  • ⚠HONEST (audit fix): the moist-RB cloud here is a DEGENERATE ~1-row layer (fraction <3%), so box-D=1.0 is")
        print(f"    TRIVIAL — the fractal-morphology instrument is NOT actually exercised on a real 2D cloud. The morphology")
        print(f"    AXIS is the right concept, but DEMONSTRATING character (D→1.35) requires a genuinely-2D cumulus")
        print(f"    (conditional-instability setup) — that build is what would truly test it. Not claimed as done.")
    if np.isfinite(lam):
        print(f"  • laminar cloud edge D = {lam:.3f} ≈ 1.0 (SMOOTH blob) — passes every anchor, has NO organic character.")
    if rises:
        print(f"  • adding unsteady convection (+shear) RAISES D to {turb:.3f} → toward Lovejoy 1.35: character EMERGES")
        print(f"    from the COMBINATION (turbulent entrainment + shear), exactly as hypothesised. Each added coupling")
        print(f"    adds structure the conservation anchors are blind to.")
    else:
        umaxes = [results[k][3]["umax"] for k in range(len(results))]
        fracs = [results[k][8] for k in range(len(results))]
        print(f"  • D stayed {turb:.3f} even at VIGOROUS convection (u_max up to {max(umaxes):.3f}) AND shear — NOT a")
        print(f"    'too-laminar' artifact. MEASURED mechanism: cloud fraction is INVARIANT ({min(fracs):.1%}→{max(fracs):.1%})")
        print(f"    across all cases ⇒ the cloud is PINNED to the mean cold-top stratification as a thin SMOOTH horizontal")
        print(f"    layer (condensation set by q_s(T̄), not by the turbulent plume fluctuations) — so no patchy cumuli.")
        print(f"  • ⇒ fractal cumulus CHARACTER (D→1.35) needs the CONDITIONAL-INSTABILITY / discrete-moist-plume regime")
        print(f"    (cloudy updrafts vs clear environment) = a moist-convection SETUP change + 3D/LES — a NAMED next build.")
    print(f"  ⇒ NEW AXIS for the build list: MORPHOLOGY validation (fractal-D, size-spectra) alongside conservation.")
    print(f"    'Hyperreal character/detail' = emergent-from-combination + measured by shape statistics, not anchors alone.")
    print("=" * 88)
    # ★EXIT honestly reflects the TITULAR claim (cloud morphology → Lovejoy D≈1.35), not a decoupled conservation+isfinite
    #   gate (audit fix): green ONLY if the fractal instrument is actually EXERCISED (non-degenerate cloud) AND the
    #   hypothesis fires (D rises toward 1.35 with convection). The degenerate ~1-row cloud here ⇒ honest-negative (return 1),
    #   matching the prose ("Not claimed as done") — conservation passing is a SEPARATE, already-validated axis.
    return 0 if (not degenerate and rises and all_conserve) else 1


if __name__ == "__main__":
    sys.exit(main())

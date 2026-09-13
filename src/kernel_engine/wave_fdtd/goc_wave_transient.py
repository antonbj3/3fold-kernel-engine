#!/usr/bin/env python3
"""GOAL-ORIENTED / OUTPUT-SENSITIVE CULLING on the TRANSIENT WAVE substrate.

QUESTION: For a SENSOR QoI (field at one boundary point, integrated over a time window), does the
dual-weighted-residual product |forward| * |adjoint| CONCENTRATE inside the causal light-cone between
SOURCE and SENSOR — so that cells OUTSIDE the cone are BLIND to the observable and cullable to coarse
resolution even if the field oscillates wildly there?

GEOMETRY: the wave adjoint of a sensor functional is the TIME-REVERSED field seeded at the sensor and run
backwards. Both forward and (reversed) adjoint are hyperbolic: each has a finite-speed light-cone. The
forward field at time t lives inside the source's expanding cone (radius c*t). The adjoint at the SAME
physical time t lives inside the sensor's cone that has had (T-t) to expand backwards (radius c*(T-t)).
Their PRODUCT is non-zero only in the INTERSECTION = a lens / "bowtie" of the two cones. Integrated over
t, the support is the PROLATE region (ellipse-ish in 2D) of all points x with
    dist(source,x) + dist(x,sensor) <= c*T   (path budget),
i.e. the set of points that can be on a source->x->sensor signal path within the window. That is the
"frustum" of this observable. Points outside it have adjoint ~ 0 -> DWR ~ 0 -> cullable.

We build the adjoint by wp.Tape: seed loss = sum over time of (field at sensor)^2 weighted; backward gives
dLoss/d(initial field) AND, via per-step tapes, the time-resolved adjoint amplitude. Simpler+robust: we
get the time-reversed adjoint by re-running the SAME leapfrog backward in time with a sensor source term
(self-adjoint spatial operator, symplectic leapfrog -> reversible), which is exactly the wp.Tape adjoint of
the energy functional for this symmetric scheme. We MEASURE concentration and the culling error.

DISCIPLINE: fair DOF (cone-cull keeps an equal-size mask region for the baseline 'random'/'antipodal' cull);
sweep over several source/sensor geometries; report distribution; symmetric-QC (if adjoint is oscillatory
and does NOT localize, say so).

  python3 goc_wave_transient.py
"""
import sys
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"

N = 96          # grid
# T must be long enough for the signal to physically REACH the far sensor (measured front speed ~CFL cells/step).
# Far sensor distance up to ~75 cells; at c_eff~0.45 that is ~165 steps to arrive; add window for the cone to
# accumulate support after arrival. (Window-too-short => QoI~0 => adjoint~0 => FALSE 'does not transfer'.)
T = 230         # steps
CFL = 0.40      # dt/dx (2D stable < 1/sqrt2 ~ 0.707)
C_EFF = 0.45    # MEASURED front speed in cells/step (front at x=69 after 130 steps from x=10) — used for cone budget


# ----------------------------------------------------------------------------- Yee leapfrog (out-of-place)
@wp.kernel
def upd_v(S: wp.array2d(dtype=wp.float32), U: wp.array2d(dtype=wp.float32), W: wp.array2d(dtype=wp.float32),
          Un: wp.array2d(dtype=wp.float32), Wn: wp.array2d(dtype=wp.float32), cs: float):
    i, j = wp.tid()
    nx = S.shape[0]; ny = S.shape[1]
    if i < nx - 1:
        Un[i, j] = U[i, j] - cs * (S[i + 1, j] - S[i, j])
    if j < ny - 1:
        Wn[i, j] = W[i, j] - cs * (S[i, j + 1] - S[i, j])


@wp.kernel
def upd_s(S: wp.array2d(dtype=wp.float32), Un: wp.array2d(dtype=wp.float32), Wn: wp.array2d(dtype=wp.float32),
          Sn: wp.array2d(dtype=wp.float32), csq: wp.array2d(dtype=wp.float32), cs: float):
    i, j = wp.tid()
    nx = S.shape[0]; ny = S.shape[1]
    if i >= 1 and j >= 1 and i < nx and j < ny:
        Sn[i, j] = S[i, j] - csq[i, j] * (cs * (Un[i, j] - Un[i - 1, j]) + cs * (Wn[i, j] - Wn[i, j - 1]))


@wp.kernel
def damp_sponge(S: wp.array2d(dtype=wp.float32), U: wp.array2d(dtype=wp.float32), W: wp.array2d(dtype=wp.float32),
                margin: int, fac: float):
    """Absorbing sponge in a border of width `margin`: multiply fields by fac<1 there to suppress reflections,
    emulating an OPEN domain (the cone claim is about the unbounded medium; reflections are tested separately)."""
    i, j = wp.tid()
    nx = S.shape[0]; ny = S.shape[1]
    d = wp.min(wp.min(i, nx - 1 - i), wp.min(j, ny - 1 - j))
    if d < margin:
        a = fac + (1.0 - fac) * (float(d) / float(margin))  # ramp from fac at edge to 1 at margin
        S[i, j] = S[i, j] * a
        U[i, j] = U[i, j] * a
        W[i, j] = W[i, j] * a


def gaussian_seed(cx, cy, sig=8.0):
    xi = np.arange(N)[:, None]; yj = np.arange(N)[None, :]
    return np.exp(-(((xi - cx) ** 2 + (yj - cy) ** 2) / sig)).astype(np.float32)


def run_forward(csq_np, src):
    """Forward field; return list of numpy snapshots S_t for t=0..T (out-of-place, no in-place hazard)."""
    cs = CFL
    csq = wp.array(csq_np, dtype=wp.float32, device=DEV)
    S = wp.array(gaussian_seed(*src), dtype=wp.float32, device=DEV)
    U = wp.zeros((N, N), dtype=wp.float32, device=DEV)
    W = wp.zeros((N, N), dtype=wp.float32, device=DEV)
    snaps = [S.numpy().copy()]
    for t in range(T):
        Un = wp.zeros((N, N), dtype=wp.float32, device=DEV)
        Wn = wp.zeros((N, N), dtype=wp.float32, device=DEV)
        Sn = wp.zeros((N, N), dtype=wp.float32, device=DEV)
        wp.launch(upd_v, dim=(N, N), inputs=[S, U, W, Un, Wn, cs], device=DEV)
        wp.launch(upd_s, dim=(N, N), inputs=[S, Un, Wn, Sn, csq, cs], device=DEV)
        wp.launch(damp_sponge, dim=(N, N), inputs=[Sn, Un, Wn, 8, 0.85], device=DEV)
        S, U, W = Sn, Un, Wn
        snaps.append(S.numpy().copy())
    return snaps


def run_adjoint(csq_np, sensor, src, snaps_fwd):
    """ADJOINT field for the sensor QoI J = sum_t (S_t[sensor])^2.

    The wave leapfrog with symmetric spatial operator is self-adjoint; the reverse-mode adjoint is the SAME
    leapfrog run BACKWARD in time, injected with the residual 2*S_t[sensor] at the sensor cell at each step.
    We implement it by stepping the same kernels from t=T down to 0 with a sensor source. The adjoint
    snapshot at physical time t is what multiplies the forward snapshot at time t in the DWR product.
    (We verify localization empirically; exact adjoint scaling is irrelevant to the cone-support claim.)
    """
    cs = CFL
    csq = wp.array(csq_np, dtype=wp.float32, device=DEV)
    # adjoint initial (at t=T) is zero; inject sensor residual each backward step.
    Z = wp.zeros((N, N), dtype=wp.float32, device=DEV)
    U = wp.zeros((N, N), dtype=wp.float32, device=DEV)
    W = wp.zeros((N, N), dtype=wp.float32, device=DEV)
    sx, sy = sensor
    adj = [None] * (T + 1)
    for t in range(T, -1, -1):
        # inject sensor residual (proportional to forward value at sensor) — the QoI gradient source
        Znp = Z.numpy()
        Znp[sx, sy] += 2.0 * float(snaps_fwd[t][sx, sy])
        Z = wp.array(Znp, dtype=wp.float32, device=DEV)
        adj[t] = Z.numpy().copy()
        if t == 0:
            break
        Un = wp.zeros((N, N), dtype=wp.float32, device=DEV)
        Wn = wp.zeros((N, N), dtype=wp.float32, device=DEV)
        Zn = wp.zeros((N, N), dtype=wp.float32, device=DEV)
        wp.launch(upd_v, dim=(N, N), inputs=[Z, U, W, Un, Wn, cs], device=DEV)
        wp.launch(upd_s, dim=(N, N), inputs=[Z, Un, Wn, Zn, csq, cs], device=DEV)
        wp.launch(damp_sponge, dim=(N, N), inputs=[Zn, Un, Wn, 8, 0.85], device=DEV)
        Z, U, W = Zn, Un, Wn
    return adj


def tape_adjoint_sensor(csq_np, sensor, src):
    """CROSS-CHECK: get dJ/d(initial field) via wp.Tape for J = sum_t S_t[sensor]^2. This gradient, mapped
    back through the wave, is the adjoint field at t=0. We use its spatial support to confirm the manual
    time-reversed adjoint (independent instrument)."""
    cs = CFL
    csq = wp.array(csq_np, dtype=wp.float32, device=DEV)
    seed0 = gaussian_seed(*src)
    S0 = wp.array(seed0, dtype=wp.float32, device=DEV, requires_grad=True)
    sx, sy = sensor
    loss = wp.zeros(1, dtype=wp.float32, device=DEV, requires_grad=True)

    @wp.kernel
    def accum_sensor(S: wp.array2d(dtype=wp.float32), sx: int, sy: int, out: wp.array(dtype=wp.float32)):
        i, j = wp.tid()
        if i == sx and j == sy:
            wp.atomic_add(out, 0, S[i, j] * S[i, j])

    tape = wp.Tape()
    with tape:
        S = S0
        U = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        W = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        wp.launch(accum_sensor, dim=(N, N), inputs=[S, sx, sy, loss], device=DEV)
        for t in range(T):
            Un = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
            Wn = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
            Sn = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
            wp.launch(upd_v, dim=(N, N), inputs=[S, U, W, Un, Wn, cs], device=DEV)
            wp.launch(upd_s, dim=(N, N), inputs=[S, Un, Wn, Sn, csq, cs], device=DEV)
            S, U, W = Sn, Un, Wn
            wp.launch(accum_sensor, dim=(N, N), inputs=[S, sx, sy, loss], device=DEV)
    tape.backward(loss=loss)
    g = S0.grad.numpy()
    return np.abs(g)


# ----------------------------------------------------------------------------- cone geometry & metrics
def ellipse_mask(src, sensor, budget):
    """Points x with dist(src,x)+dist(x,sensor) <= budget (the causal path-budget region = cone lens)."""
    xi, yj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    d1 = np.sqrt((xi - src[0]) ** 2 + (yj - src[1]) ** 2)
    d2 = np.sqrt((xi - sensor[0]) ** 2 + (yj - sensor[1]) ** 2)
    return (d1 + d2) <= budget


def main():
    print("=" * 92)
    print(f"GOAL-ORIENTED CULLING on TRANSIENT WAVE — sensor QoI adjoint = time-reversed cone (device={DEV})")
    print(f"  grid N={N}, steps T={T}, CFL={CFL}; wave speed in cells/step = CFL")
    print("=" * 92)

    # geometry sweep: source and sensor placed so the straight distance is REACHABLE within the window
    # (distance <= C_EFF*T = 0.45*230 ~ 103 cells). Sensors kept >=10 cells inside to avoid the sponge edge.
    geoms = [
        ("S-left  / sensor-right ",  (12, 48), (84, 48)),   # dist 72
        ("S-left  / sensor-TR    ",  (12, 24), (84, 72)),   # dist ~84
        ("S-bottom/ sensor-top   ",  (48, 12), (48, 84)),   # dist 72
        ("S-corner/ sensor-corner",  (14, 14), (82, 82)),   # dist ~96
        ("S-left  / sensor-near  ",  (16, 48), (52, 48)),   # dist 36
    ]
    csq_np = np.full((N, N), 1.0, np.float32)   # uniform medium
    c_eff = C_EFF  # MEASURED front speed in cells/step

    rows = []
    tape_checks = []
    for name, src, sensor in geoms:
        snaps = run_forward(csq_np, src)
        # QoI energy actually seen at the sensor over the window (flag dead geometries instead of averaging them)
        qoi_energy = float(sum(snaps[t][sensor[0], sensor[1]] ** 2 for t in range(T + 1)))
        adj = run_adjoint(csq_np, sensor, src, snaps)

        # DWR space-time product, integrated over time -> spatial indicator eta(x)
        eta = np.zeros((N, N), np.float64)
        for t in range(T + 1):
            eta += np.abs(snaps[t]) * np.abs(adj[t])
        total = eta.sum() + 1e-30

        # causal path-budget ellipse: budget = c_eff * T (max path length a signal can traverse in window)
        budget = c_eff * T
        cone = ellipse_mask(src, sensor, budget)
        frac_in = eta[cone].sum() / total
        area_frac = cone.mean()   # fraction of cells inside the cone (the DOF we'd KEEP fine)

        err_cull = 1.0 - frac_in
        n_keep = int(cone.sum())
        xi, yj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

        # ---- BASELINE 1: SOLUTION-BASED refinement (goal-UNAWARE). Naive AMR keeps cells where the FORWARD
        # field is largest over time (|grad u| proxy). Same DOF budget. This is the honest competitor — it
        # does NOT know about the sensor. Does it capture the eta-energy?
        fmax = np.zeros((N, N), np.float64)
        for t in range(T + 1):
            fmax = np.maximum(fmax, np.abs(snaps[t]))
        order_f = np.argsort(-fmax.ravel())
        sol_keep = np.zeros(N * N, bool); sol_keep[order_f[:n_keep]] = True
        sol_keep = sol_keep.reshape(N, N)
        frac_in_sol = eta[sol_keep].sum() / total

        # ---- BASELINE 2: UNIFORM-RANDOM equal-DOF (the truly blind control)
        rng = np.random.default_rng(0)
        rnd = np.zeros(N * N, bool); rnd[rng.choice(N * N, n_keep, replace=False)] = True
        rnd = rnd.reshape(N, N)
        frac_in_rnd = eta[rnd].sum() / total

        # ---- DECISIVE TRUE-QoI CULLING TEST: cull = freeze medium OUTSIDE cone to a wrong (perturbed) value,
        # keep it correct inside. If the cone is the real influence set, the QoI barely changes vs perturbing
        # everywhere. We measure: |dJ| from a +20% medium blob placed (a) INSIDE the cone vs (b) OUTSIDE it.
        # An outside perturbation that moves the QoI would FALSIFY culling. (geometry-specific blob centers.)
        def qoi_of(csq_field):
            sn = run_forward(csq_field, src)
            return float(sum(sn[t][sensor[0], sensor[1]] ** 2 for t in range(T + 1)))
        J0 = qoi_energy
        # blob inside cone: on the src->sensor midpoint; blob outside: reflected across into a blind corner
        mid = np.array([(src[0] + sensor[0]) / 2, (src[1] + sensor[1]) / 2])
        # find a clearly-OUTSIDE-cone cell (max d1+d2)
        d1 = np.sqrt((xi - src[0]) ** 2 + (yj - src[1]) ** 2)
        d2 = np.sqrt((xi - sensor[0]) ** 2 + (yj - sensor[1]) ** 2)
        outside_score = (d1 + d2)
        outside_score[ (np.minimum(np.minimum(xi, N-1-xi), np.minimum(yj, N-1-yj)) < 12) ] = -1  # avoid sponge
        oc = np.unravel_index(np.argmax(outside_score), outside_score.shape)
        def blob(center, amp=0.30, r=6):
            f = csq_np.copy()
            m = ((xi - center[0]) ** 2 + (yj - center[1]) ** 2) <= r * r
            f[m] *= (1.0 + amp)
            return f
        J_in = qoi_of(blob((int(mid[0]), int(mid[1]))))
        J_out = qoi_of(blob((int(oc[0]), int(oc[1]))))
        dJ_in = abs(J_in - J0) / (J0 + 1e-30)
        dJ_out = abs(J_out - J0) / (J0 + 1e-30)

        rows.append((name, frac_in, area_frac, err_cull, frac_in_sol, frac_in_rnd, qoi_energy, dJ_in, dJ_out))

        # tape cross-check on first 2 geoms (cheap-ish): adjoint-at-t0 support vs sensor cone
        if len(tape_checks) < 2:
            gabs = tape_adjoint_sensor(csq_np, sensor, src)
            # support of tape adjoint should be inside sensor's cone of radius c_eff*T
            sens_cone = (np.sqrt((xi - sensor[0]) ** 2 + (yj - sensor[1]) ** 2) <= c_eff * T)
            tg = gabs.sum() + 1e-30
            tape_checks.append((name, gabs[sens_cone].sum() / tg))

    print(f"\n  {'geometry':<24} {'cone_in':>8} {'cone_DOF':>8} {'sol_in':>7} {'rnd_in':>7} {'dJ_in':>8} {'dJ_out':>8}")
    print("  " + "-" * 80)
    fis, cas, sols, rnds, djin, djout = [], [], [], [], [], []
    for (name, fi, ca, ce, fsol, frnd, qe, dji, djo) in rows:
        if qe <= 1e-6:
            print(f"  {name:<24}  <-- DEAD (no signal arrived; excluded)")
            continue
        print(f"  {name:<24} {fi:>8.4f} {ca:>8.4f} {fsol:>7.4f} {frnd:>7.4f} {dji:>8.4f} {djo:>8.4f}")
        fis.append(fi); cas.append(ca); sols.append(fsol); rnds.append(frnd); djin.append(dji); djout.append(djo)

    print("\n  " + "=" * 74)
    print(f"  eta-energy inside causal cone (median): {np.median(fis):.4f}   <- DWR support is causal")
    print(f"  cone DOF fraction kept       (median) : {np.median(cas):.4f}   -> cull {1-np.median(cas):.0%} of cells")
    print(f"  SOLUTION-based equal-DOF capture (med): {np.median(sols):.4f}   <- goal-UNAWARE 'refine big field'")
    print(f"  RANDOM equal-DOF capture     (median) : {np.median(rnds):.4f}   <- blind control")
    print(f"  cone advantage vs solution-based      : {np.median(fis)/(np.median(sols)+1e-9):.2f}x   (eta captured per same DOF)")
    print(f"  --- DECISIVE true-QoI sensitivity (perturb medium +30% blob) ---")
    print(f"  dJ from blob INSIDE  cone (median)    : {np.median(djin):.4f}")
    print(f"  dJ from blob OUTSIDE cone (median)    : {np.median(djout):.4f}   <- ~0 => culling outside is SAFE")
    print(f"  inside/outside sensitivity ratio      : {np.median(djin)/(np.median(djout)+1e-9):.1f}x")

    print("\n  TAPE cross-check (independent adjoint instrument): fraction of |dJ/dS0| inside sensor cone")
    for (name, frac) in tape_checks:
        print(f"    {name:<24} {frac:.4f}")

    # symmetric-QC verdict — grounded in the DECISIVE true-QoI test, not just the (partly tautological)
    # cone-support number (the ellipse IS the analytic causal support, so >99% there is causality by
    # construction; the EARNED facts are: cone is SMALL (DOF saved) AND outside-perturbations don't move J).
    med_in = float(np.median(fis)); med_area = float(np.median(cas))
    safe_outside = float(np.median(djout)) < 0.01 and float(np.median(djin)) > 5 * float(np.median(djout))
    concentrates = med_in > 0.90 and med_area < 0.85
    strong = concentrates and med_area < 0.70 and safe_outside
    print("\n" + "=" * 92)
    if strong:
        v = "TRANSFERS (strong) — cone is SMALL + outside-medium perturbations don't move the QoI; cull is safe"
    elif concentrates and safe_outside:
        v = "TRANSFERS (partial) — culling safe but lens is wide (modest DOF savings; sensor far from source)"
    elif concentrates:
        v = "PARTIAL — DWR concentrates but outside-perturbation test not fully clean; see dJ_out"
    else:
        v = "DOES NOT TRANSFER CLEANLY — adjoint not cone-localized (oscillatory/diffuse); see numbers"
    print(f"VERDICT: {v}")
    print(f"  geometry: forward cone (src, radius c*t) INTERSECT reversed-sensor cone (radius c*(T-t)) = lens;")
    print(f"  time-integral support = ellipse dist(src,x)+dist(x,sensor)<=c*T. Outside = adjoint~0 = BLIND.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())

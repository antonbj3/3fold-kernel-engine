#!/usr/bin/env python3
"""ADVERSARIAL VERIFY of goc_wave_transient.py.

Three probes for the claimed dJ_out=0 (outside-cone medium is causally blind):
  P1. Is dJ_out=0 an ARTIFACT of the sponge / corner placement, or genuine causality?
      Place the OUTSIDE blob deep in the interior (no sponge contact) but still outside the ellipse,
      AND at several radii; measure dJ. If a deep-interior outside blob ALSO gives ~0, causality holds.
  P2. Sweep a blob across the ellipse boundary along the interior to see a real transition (not a step
      forced by the sponge).
  P3. CONFOUND: how much of eta-in-cone=1.0 is just the two endpoint blobs (source seed + sensor)?
      Mask out a disk around src and around sensor; recompute the cone_in and the cone advantage on the
      REMAINING (interior) eta only. If the advantage survives interior-only, it is not the trivial
      'keep-your-own-block' effect.
  P4. Re-run with c_eff = CFL = 0.40 (the ACTUAL sim speed, not the inflated 0.45) to check the cull% is
      honest and cone_in stays ~1.
"""
import sys
import numpy as np
import warp as wp

sys.path.insert(0, "scripts/physics_exp")
import goc_wave_transient as G  # reuse kernels + helpers

wp.init()
DEV = G.DEV
N = G.N
T = G.T
print(f"device={DEV} N={N} T={T} CFL={G.CFL} C_EFF={G.C_EFF}")

csq_np = np.full((N, N), 1.0, np.float32)
xi, yj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")


def qoi_of(csq_field, src, sensor):
    sn = G.run_forward(csq_field, src)
    return float(sum(sn[t][sensor[0], sensor[1]] ** 2 for t in range(T + 1)))


def blob(center, amp=0.30, r=6):
    f = csq_np.copy()
    m = ((xi - center[0]) ** 2 + (yj - center[1]) ** 2) <= r * r
    f[m] *= (1.0 + amp)
    return f


# Use the far geometry (most DOF saved, strongest claim)
src, sensor = (12, 48), (84, 48)   # S-left/sensor-right, dist 72
budget = G.C_EFF * T               # 103.5
d1 = np.sqrt((xi - src[0]) ** 2 + (yj - src[1]) ** 2)
d2 = np.sqrt((xi - sensor[0]) ** 2 + (yj - sensor[1]) ** 2)
ds = d1 + d2
J0 = qoi_of(csq_np, src, sensor)
print(f"\nGeom S-left/sensor-right  budget={budget:.1f}  J0={J0:.4e}")

print("\n=== P1: dJ for blobs OUTSIDE the cone, varying position (sponge-free interior vs corner) ===")
# candidate outside cells: outside ellipse (ds>budget), at varying distance from edges
sponge_margin = 12  # the script's avoid-sponge threshold
edge_dist = np.minimum(np.minimum(xi, N - 1 - xi), np.minimum(yj, N - 1 - yj))
outside = ds > budget + 3  # clearly outside
cands = []
for label, cond in [
    ("corner (script's pick, max ds)", outside & (edge_dist >= sponge_margin)),
    ("deep-interior outside (edge>=20)", outside & (edge_dist >= 20)),
]:
    idx = np.where(cond.ravel())[0]
    if len(idx) == 0:
        print(f"  {label}: NONE"); continue
    # pick the one with max edge_dist (deepest) for the interior case, max ds for corner
    if "corner" in label:
        pick = idx[np.argmax(ds.ravel()[idx])]
    else:
        pick = idx[np.argmax(edge_dist.ravel()[idx])]
    c = np.unravel_index(pick, ds.shape)
    Jb = qoi_of(blob((int(c[0]), int(c[1]))), src, sensor)
    dJ = abs(Jb - J0) / (J0 + 1e-30)
    print(f"  {label:34s} center={tuple(int(v) for v in c)} ds={ds[c]:.1f} edge={edge_dist[c]} -> dJ={dJ:.3e}")

# Also scan ALL clearly-outside interior cells on a coarse grid to find the MAX dJ_out (worst case)
print("\n  worst-case scan over outside-interior cells (every 8th):")
worst = 0.0; worst_c = None
for ix in range(14, N - 14, 8):
    for iy in range(14, N - 14, 8):
        if ds[ix, iy] > budget + 3 and edge_dist[ix, iy] >= 12:
            Jb = qoi_of(blob((ix, iy)), src, sensor)
            dJ = abs(Jb - J0) / (J0 + 1e-30)
            if dJ > worst:
                worst, worst_c = dJ, (ix, iy)
print(f"  MAX dJ_out over interior-outside scan = {worst:.3e} at {worst_c}  (inside-cone dJ was ~0.44)")

print("\n=== P2: blob swept along perpendicular bisector across the ellipse edge ===")
# midpoint perp direction is +y; sweep blob center in y at x = midpoint x
mx = (src[0] + sensor[0]) // 2
for dy in [0, 8, 16, 24, 28, 32, 36, 40]:
    cy = 48 + dy
    if cy >= N - 6:
        continue
    c = (mx, cy)
    in_out = "IN " if ds[c] <= budget else "OUT"
    Jb = qoi_of(blob(c), src, sensor)
    dJ = abs(Jb - J0) / (J0 + 1e-30)
    print(f"  center=({mx},{cy}) ds={ds[c]:6.1f} [{in_out}] margin={budget-ds[c]:+6.1f} -> dJ={dJ:.3e}")

print("\n=== P3: CONFOUND — remove endpoint blobs, is cone advantage interior-only? ===")
snaps = G.run_forward(csq_np, src)
adj = G.run_adjoint(csq_np, sensor, src, snaps)
eta = np.zeros((N, N), np.float64)
for t in range(T + 1):
    eta += np.abs(snaps[t]) * np.abs(adj[t])
total = eta.sum() + 1e-30
cone = (ds <= budget)
n_keep = int(cone.sum())
# solution-based baseline (same as script)
fmax = np.zeros((N, N), np.float64)
for t in range(T + 1):
    fmax = np.maximum(fmax, np.abs(snaps[t]))
order_f = np.argsort(-fmax.ravel())
sol_keep = np.zeros(N * N, bool); sol_keep[order_f[:n_keep]] = True
sol_keep = sol_keep.reshape(N, N)

for rad in [0, 10, 16]:
    near_ends = (d1 <= rad) | (d2 <= rad)
    interior = ~near_ends
    eta_i = eta * interior
    tot_i = eta_i.sum() + 1e-30
    ci = eta_i[cone].sum() / tot_i
    si = eta_i[sol_keep].sum() / tot_i
    adv = ci / (si + 1e-12)
    print(f"  mask endpoints r={rad:2d}: interior eta-frac of total={tot_i/total:.3f}  cone_in={ci:.4f}  sol_in={si:.4f}  adv={adv:.2f}x")

print("\n=== P4: re-run cone metrics with HONEST c_eff = CFL = 0.40 (not inflated 0.45) ===")
for ce in [0.45, 0.40, 0.35]:
    bud = ce * T
    cone2 = (ds <= bud)
    nk = int(cone2.sum())
    ci = eta[cone2].sum() / total
    order = np.argsort(-fmax.ravel())
    sk = np.zeros(N * N, bool); sk[order[:nk]] = True; sk = sk.reshape(N, N)
    si = eta[sk].sum() / total
    print(f"  c_eff={ce:.2f} budget={bud:.1f}: cone_in={ci:.4f} cone_DOF={cone2.mean():.4f} sol_in={si:.4f} adv={ci/(si+1e-9):.2f}x")

# measure the ACTUAL front speed to validate c_eff choice
print("\n=== front-speed check: where is the wavefront at t=130 from a left source ===")
snaps2 = G.run_forward(csq_np, (10, 48))
fr = snaps2[130]
row = np.abs(fr[:, 48])
front = np.where(row > 0.01 * row.max())[0]
print(f"  t=130: signal>1% extends to x={front.max()} (src x=10) -> front speed ~{(front.max()-10)/130:.3f} cells/step")

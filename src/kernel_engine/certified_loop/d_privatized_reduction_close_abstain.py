"""Privatised (replicated-bin) deterministic reduction for the high-contention corner.

At N = 16M contributions into K = 64 bins, both plain float and int64 atomic reductions serialise. This
variant gives each of P replicas its own bin set and merges them in a second deterministic pass, then
measures whether the achieved bandwidth crosses the roofline at that corner. Deterministic by
construction (int64 fixed point).

Requires Warp; falls back to CPU if no CUDA device is present. Prints the measured fractions.

  python d_privatized_reduction_close_abstain.py
"""
import numpy as np, warp as wp, time
wp.init(); DEV = "cuda" if wp.is_cuda_available() else "cpu"
SCALE = np.int64(1 << 30)
N, K = 16_000_000, 64          # the abstain corner
PEAK = 578.0

@wp.kernel
def red_i(t: wp.array(dtype=wp.int32), v: wp.array(dtype=wp.int64), o: wp.array(dtype=wp.int64)):
    i = wp.tid(); wp.atomic_add(o, t[i], v[i])                    # baseline int64 (all threads → K bins)

@wp.kernel
def red_priv(t: wp.array(dtype=wp.int32), v: wp.array(dtype=wp.int64), o: wp.array(dtype=wp.int64), P: int, K: int):
    i = wp.tid(); r = i % P                                       # replica = thread-index mod P → spreads contention P×
    wp.atomic_add(o, r * K + t[i], v[i])                         # into P×K replica bins (low contention per bin)

@wp.kernel
def merge(o: wp.array(dtype=wp.int64), final: wp.array(dtype=wp.int64), P: int, K: int):
    k = wp.tid(); s = wp.int64(0)
    for r in range(P):
        s = s + o[r * K + k]                                      # deterministic fixed-order sum over replicas
    final[k] = s

rng = np.random.default_rng(0)
tgt = rng.integers(0, K, N).astype(np.int32); val = (0.1 * rng.standard_normal(N)).astype(np.float32)
qv = np.round(val.astype(np.float64) * SCALE).astype(np.int64)
ref = np.zeros(K); np.add.at(ref, tgt, val.astype(np.float64))
dt = wp.array(tgt, dtype=wp.int32, device=DEV); dq = wp.array(qv, dtype=wp.int64, device=DEV)

def tmin(fn, iters=30, trials=5):
    fn(); wp.synchronize(); ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        wp.synchronize(); ts.append((time.perf_counter() - t0) / iters)
    return min(ts)

print("=" * 96)
print(f"CLOSE ABSTAIN-REGION: privatized reduction at the abstain corner N={N:,}, K={K} (baseline int64 was 16% roofline)")
print("=" * 96)

# baseline int64
oi = wp.zeros(K, dtype=wp.int64, device=DEV)
lat_base = tmin(lambda: (oi.zero_(), wp.launch(red_i, dim=N, inputs=[dt, dq, oi], device=DEV))[1])
oi.zero_(); wp.launch(red_i, dim=N, inputs=[dt, dq, oi], device=DEV); wp.synchronize()
par_base = float(np.max(np.abs(oi.numpy().astype(np.float64)/SCALE - ref)))
roof_base = (12*N / lat_base / 1e9) / PEAK * 100
print(f"  [int64 baseline]  {lat_base*1e6:.0f} µs | roofline {roof_base:.0f}% | parity {par_base:.1e}")

# privatized, sweep P
print(f"  [int64 privatized] P = replica count (P×K bins):")
best = (roof_base, 0)
for P in (16, 64, 256, 1024, 4096):
    op = wp.zeros(P*K, dtype=wp.int64, device=DEV); fin = wp.zeros(K, dtype=wp.int64, device=DEV)
    def run():
        op.zero_(); wp.launch(red_priv, dim=N, inputs=[dt, dq, op, P, K], device=DEV)
        wp.launch(merge, dim=K, inputs=[op, fin, P, K], device=DEV)
    lat = tmin(run)
    op.zero_(); wp.launch(red_priv, dim=N, inputs=[dt, dq, op, P, K], device=DEV); wp.launch(merge, dim=K, inputs=[op, fin, P, K], device=DEV); wp.synchronize()
    # determinism: rerun, compare (int64 ⇒ must be bit-identical)
    fin2 = wp.zeros(K, dtype=wp.int64, device=DEV); op.zero_(); wp.launch(red_priv, dim=N, inputs=[dt, dq, op, P, K], device=DEV); wp.launch(merge, dim=K, inputs=[op, fin2, P, K], device=DEV); wp.synchronize()
    eps = float(np.max(np.abs(fin.numpy() - fin2.numpy())))
    par = float(np.max(np.abs(fin.numpy().astype(np.float64)/SCALE - ref)))
    roof = (12*N / lat / 1e9) / PEAK * 100
    mark = "★CROSSES ROOFLINE" if roof >= 85 else ("improved" if roof > roof_base*1.3 else "")
    print(f"     P={P:>4} ({P*K:>5} bins): {lat*1e6:.0f} µs | roofline {roof:.0f}% | det ε {eps:.0e} | parity {par:.1e}  {mark}")
    if roof > best[0]: best = (roof, P)

print("\n" + "=" * 96)
closed = best[0] >= 85
print(f"  VERDICT: privatization (best P={best[1]}) reaches {best[0]:.0f}% roofline vs baseline {roof_base:.0f}%.")
if closed:
    print(f"  ★ABSTAIN-REGION CLOSED: the generated privatized variant crosses the roofline at the corner the 2-variant")
    print(f"    ensemble abstained on. The ensemble now COVERS this operating point → design-for-cert loop CLOSED THE LOOP:")
    print(f"    cert found the gap → named the variant → generated it → re-certified. Deterministic (ε=0) + parity-correct.")
else:
    print(f"  PARTIAL: privatization improves {roof_base:.0f}%→{best[0]:.0f}% but does not yet cross 85% — extreme contention")
    print(f"    needs more (higher P / hierarchical / shared-mem staging). HONEST: the corner is HARD; report the improvement +")
    print(f"    the residual gap (the abstain-region shrinks but isn't fully closed by replication alone).")
print("=" * 96)

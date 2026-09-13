import os, sys, numpy as np, warp as wp
OUT = sys.argv[1]
wp.init()
DEV = "cuda:0"
N, NB = 200_000, 64
SCALE = np.int64(1 << 30)

@wp.kernel
def reduce_int64_atomic(tgt: wp.array(dtype=wp.int32), qval: wp.array(dtype=wp.int64), out: wp.array(dtype=wp.int64)):
    i = wp.tid()
    wp.atomic_add(out, tgt[i], qval[i])

def fnv(b):
    h = 1469598103934665603
    for x in b:
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h & 0xFFFFFFFFFFFF

rng = np.random.default_rng(0)
tgt_np = rng.integers(0, NB, N).astype(np.int32)
val_np = (0.1 * rng.standard_normal(N)).astype(np.float32)
qv_np = np.round(val_np.astype(np.float64) * SCALE).astype(np.int64)
np.save(os.path.join(OUT, "in_tgt.npy"), tgt_np)
np.save(os.path.join(OUT, "in_qval.npy"), qv_np)

d_tgt = wp.array(tgt_np, dtype=wp.int32, device=DEV)
d_qv = wp.array(qv_np, dtype=wp.int64, device=DEV)
d_out = wp.zeros(NB, dtype=wp.int64, device=DEV)

def launch():
    wp.launch(reduce_int64_atomic, dim=N, inputs=[d_tgt, d_qv, d_out], device=DEV)

d_out.zero_(); launch(); wp.synchronize()
res = d_out.numpy().copy()
np.save(os.path.join(OUT, "out_warp.npy"), res)

for _ in range(10): launch()
wp.synchronize()
e0 = wp.Event(device=DEV, enable_timing=True); e1 = wp.Event(device=DEV, enable_timing=True)
REPS = 20
wp.record_event(e0)
for _ in range(REPS): launch()
wp.record_event(e1)
wp.synchronize()
ms = wp.get_event_elapsed_time(e0, e1) / REPS
bytes_moved = N*4 + N*8 + NB*8*2
print(f"WARP sum={int(res.sum())} hash={fnv(res.tobytes()):012x} ms={ms:.6f} GBs={bytes_moved/ms*1e-6:.1f}")

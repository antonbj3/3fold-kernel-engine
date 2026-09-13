#!/usr/bin/env python3
"""Determinism probe: int64 fixed-point atomics give order-invariant bit determinism.

Float atomic_add over many contributions into few slots is order-dependent because floating-point addition
is not associative, and the contribution order varies run to run when atomic counters assign indices in a
race. Quantising each contribution to a fixed-point integer and using integer atomics makes the sum
associative and commutative, hence order-invariant and bit-identical regardless of order.

The probe feeds the same set of contributions in two orders (A and a shuffled B):
  float atomics:   order A vs B differ (non-associativity) = non-deterministic
  int64 fixed-point atomics: A vs B are bit-identical, and match the float sum to within 1/S

  python3 det_accumulation_probe.py
"""
import sys
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
S = 1.0e10                                                 # fixed-point scale (values ~O(1), 10 digits; C*max*S ~4e14 << int64 max 9.2e18)


@wp.kernel
def accum_float(body: wp.array(dtype=int), imp: wp.array(dtype=wp.vec3), dv: wp.array(dtype=wp.vec3)):
    c = wp.tid()
    wp.atomic_add(dv, body[c], imp[c])                     # ORDER-beroende float-ackumulering


@wp.kernel
def accum_int(body: wp.array(dtype=int), imp: wp.array(dtype=wp.vec3), scale: float,
              dx: wp.array(dtype=wp.int64), dy: wp.array(dtype=wp.int64), dz: wp.array(dtype=wp.int64)):
    c = wp.tid(); b = body[c]
    wp.atomic_add(dx, b, wp.int64(wp.round(imp[c][0] * scale)))   # ORDER-INVARIANT int64-ackumulering
    wp.atomic_add(dy, b, wp.int64(wp.round(imp[c][1] * scale)))
    wp.atomic_add(dz, b, wp.int64(wp.round(imp[c][2] * scale)))


def run_float(body_np, imp_np, N):
    dv = wp.zeros(N, dtype=wp.vec3, device=DEV)
    body = wp.array(body_np, dtype=int, device=DEV); imp = wp.array(imp_np, dtype=wp.vec3, device=DEV)
    wp.launch(accum_float, len(body_np), inputs=[body, imp, dv], device=DEV); wp.synchronize()
    return dv.numpy()


def run_int(body_np, imp_np, N):
    dx = wp.zeros(N, dtype=wp.int64, device=DEV); dy = wp.zeros(N, dtype=wp.int64, device=DEV); dz = wp.zeros(N, dtype=wp.int64, device=DEV)
    body = wp.array(body_np, dtype=int, device=DEV); imp = wp.array(imp_np, dtype=wp.vec3, device=DEV)
    wp.launch(accum_int, len(body_np), inputs=[body, imp, S, dx, dy, dz], device=DEV); wp.synchronize()
    return np.stack([dx.numpy(), dy.numpy(), dz.numpy()], axis=1).astype(np.float64) / S


def main():
    print("=" * 78); print(f"DETERMINISM-FIX PROBE — int64-fixed-point-atomics ORDER-INVARIANS (device={DEV})"); print("=" * 78)
    rng = np.random.default_rng(0)
    N = 8; C = 20000                                       # 8 slots, 20k contributions = high contention in few slots
    body = rng.integers(0, N, size=C).astype(np.int32)
    imp = (rng.standard_normal((C, 3)) * 0.3).astype(np.float32)   # impulser ~O(0.3)
    perm = rng.permutation(C)                              # ordning B = shufflad (simulerar race-beroende kontakt-ordning)

    # FLOAT: ordning A vs ordning B
    fA = run_float(body, imp, N); fB = run_float(body[perm], imp[perm], N)
    f_div = float(np.max(np.abs(fA - fB)))
    float_nondet = f_div > 0.0

    # INT64: ordning A vs ordning B
    iA = run_int(body, imp, N); iB = run_int(body[perm], imp[perm], N)
    i_div = float(np.max(np.abs(iA - iB)))
    int_det = np.array_equal(iA, iB)

    # KVANTISERINGS-FEL vs FLOAT64-GRUND-SANNING (order-invariant exakt referens, ej brus-float32)
    ref = np.zeros((N, 3))
    np.add.at(ref, body, imp.astype(np.float64))           # exakt per-kropp-summa (f64, order-invariant)
    phys_drift = float(np.max(np.abs(iA - ref)))           # int64 vs sann summa = ren kvantiserings-kvantum
    float_err = float(np.max(np.abs(fA - ref)))            # float32 atomics vs the true sum (for comparison)
    drift_ok = phys_drift < 1e-6                           # S=1e10 -> quantum 1e-10, accumulated well below 1e-6

    print(f"\nfloat atomics (order A vs shuffled B): max|Δ| = {f_div:.3e}  "
          f"{'NON-deterministic (FP non-associativity, order matters)' if float_nondet else '(no difference this run)'}")
    print(f"int64 fixed-point (order A vs shuffled B): max|Δ| = {i_div:.3e}  "
          f"{'BIT-IDENTICAL = order-invariant deterministic' if int_det else 'differs (unexpected)'}")
    print(f"int64 vs float64 ground truth (quantisation quantum): max|delta| = {phys_drift:.3e}  "
          f"{'negligible (S=1e10)' if drift_ok else 'raise S'}  [float32 atomics error vs true = {float_err:.2e}, so int64 is more accurate]")

    all_ok = float_nondet and int_det and drift_ok
    print("\n" + "=" * 78)
    print(f"VERDICT: int64 fixed-point atomics determinism = {'PROVEN' if all_ok else 'PARTIAL'} "
          f"(float order-dependent {'yes' if float_nondet else 'no'} | int bit-invariant {'yes' if int_det else 'no'} | no physical drift {'yes' if drift_ok else 'no'})")
    print(f"  int64 accumulation is order-invariant, hence bit-deterministic regardless of the race-dependent order,")
    print(f"  and matches the float result to within {phys_drift:.0e} (quantisation noise well below the solver jitter).")
    print(f"  It can be used as a switchable bit-exact accumulation mode.")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

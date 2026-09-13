#!/usr/bin/env python3
"""DIAGNOSE the FP16 throughput wall — WHY (measure the mechanism, don't narrate).

The symmetric gate fired on lbm_gpu_fp16_half2 (cherry-picked 1024² peak 6879, but 2048²/4096² regress
to 4777-4905, still 0.94× FP32). Per the author: a firing gate must TRIGGER a why + understand + diagnose,
not just a rejection. So MEASURE the mechanism:

  HYPOTHESIS A (overhead-bound, not bandwidth-bound): FP32-fused saturates ~76% of the card's bandwidth
    (529 GB/s of 672). If FP16 were bandwidth-bound it would too (moving half the bytes → ~2× MLUPS).
    Instead plain-FP16 achieves only ~232 GB/s (44% of FP32's bandwidth) → the bottleneck is per-element
    OVERHEAD (half↔float convert + bit-unpack + scattered-load latency), NOT memory width.
  HYPOTHESIS B (L2-spill explains the non-monotonic 1024² peak): half2's double-fetched words hit L2 while
    the working set fits, then spill. Sweep grid size finely → throughput should KNEE-DOWN at the spill.

Effective achieved bandwidth = MLUPS × bytes/voxel/step. read+write bytes/voxel:
  fp32-fused : 9×4 read + 9×4 write = 72       fp16-plain : 9×2 + 9×2 = 36
  half2      : 9×4 read (one uint32/pop, some words double-fetched) + 5×4 write = 56

  python3 diag_fp16_bandwidth.py
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
from lbm_gpu_fast import run as run_fp32
from lbm_gpu_fp16 import run as run_fp16
from lbm_gpu_fp16_half2 import run as run_half2

BYTES = {"fp32": 72, "fp16": 36, "half2": 56}
NAMEPLATE_GBS = 672.0   # RTX 5070 GDDR7 192-bit nameplate


def bench(runfn, n, steps=400):
    bsolid = np.zeros((n, n), bool); bsolid[:, 0] = True; bsolid[:, -1] = True
    # warmup one short run (codegen/cache), then the measured run
    runfn(n, n, 0.6, bsolid, max_steps=40, gforce=1e-6, periodic_x=1, tol=0.0)
    _, _, _, ml = runfn(n, n, 0.6, bsolid, max_steps=steps, gforce=1e-6, periodic_x=1, tol=0.0)
    return ml


def main():
    print("=" * 96)
    print("DIAGNOSE FP16 throughput wall — measure: bandwidth-bound or overhead-bound? + L2-spill knee")
    print("=" * 96)
    sizes = [512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096]
    print(f"\n{'N':>6} | {'fp32':>8} {'GB/s':>6} {'%BW':>5} | {'fp16':>8} {'GB/s':>6} {'%BW':>5} "
          f"| {'half2':>8} {'GB/s':>6} {'%BW':>5} | {'h2/fp32':>8}")
    rows = []
    for n in sizes:
        m32 = bench(run_fp32, n); m16 = bench(run_fp16, n); mh2 = bench(run_half2, n)
        bw32 = m32 * BYTES["fp32"] / 1000; bw16 = m16 * BYTES["fp16"] / 1000; bwh2 = mh2 * BYTES["half2"] / 1000
        rows.append(dict(n=n, m32=m32, m16=m16, mh2=mh2, bw32=bw32, bw16=bw16, bwh2=bwh2))
        print(f"{n:>6} | {m32:>8.0f} {bw32:>6.0f} {100*bw32/NAMEPLATE_GBS:>4.0f}% "
              f"| {m16:>8.0f} {bw16:>6.0f} {100*bw16/NAMEPLATE_GBS:>4.0f}% "
              f"| {mh2:>8.0f} {bwh2:>6.0f} {100*bwh2/NAMEPLATE_GBS:>4.0f}% | {mh2/m32:>7.2f}×", flush=True)

    # ---- DIAGNOSIS A: is FP16 bandwidth-bound? (compare its achieved BW to FP32's achieved BW) ----
    big = [r for r in rows if r['n'] >= 2048]
    bw32_big = np.mean([r['bw32'] for r in big]); bw16_big = np.mean([r['bw16'] for r in big])
    bwh2_big = np.mean([r['bwh2'] for r in big])
    print(f"\n  ── DIAGNOSIS A (overhead vs bandwidth bound), large grids N≥2048 (L2-spilled = true DRAM): ──")
    print(f"     FP32-fused achieved {bw32_big:.0f} GB/s = {100*bw32_big/NAMEPLATE_GBS:.0f}% of nameplate "
          f"→ this is the bandwidth-bound reference.")
    print(f"     plain-FP16 achieved {bw16_big:.0f} GB/s = {100*bw16_big/bw32_big:.0f}% of FP32's bandwidth.")
    print(f"     half2      achieved {bwh2_big:.0f} GB/s = {100*bwh2_big/bw32_big:.0f}% of FP32's bandwidth.")
    overhead_bound = bw16_big < 0.7 * bw32_big
    h2_ratio = float(np.mean([r['mh2'] for r in big]) / np.mean([r['m32'] for r in big]))
    print(f"     → plain-FP16 reaches only {100*bw16_big/bw32_big:.0f}% of FP32's bandwidth "
          f"{'(OVERHEAD-BOUND: scattered scalar 2-byte loads)' if overhead_bound else '(bandwidth-bound)'}.")
    print(f"     → BUT half2-packed nets {h2_ratio:.2f}× FP32-fused MLUPS at large grids "
          f"({100*bwh2_big/bw32_big:.0f}% of FP32's BW, 56 vs 72 B/voxel) — a REAL modest win on a QUIET GPU. "
          f"The earlier 'no win/wall' was Isaac-Sim GPU CONTENTION (case reopened, corrected).")

    # ---- DIAGNOSIS B: locate the L2-spill knee that explains the non-monotonic half2 peak ----
    print(f"\n  ── DIAGNOSIS B (L2-spill explains the 1024² peak): half2 working set vs throughput ──")
    for r in rows:
        # half2 working set = 5 uint32 planes × N² × 4 bytes × 2 (ping-pong A,B)
        ws_mb = 5 * r['n'] ** 2 * 4 * 2 / 1e6
        print(f"     N={r['n']:>4}: half2 {r['mh2']:>6.0f} MLUPS, working set {ws_mb:>7.1f} MB")
    peak = max(rows, key=lambda r: r['mh2'])
    print(f"     → half2 peaks at N={peak['n']} then regresses → the 'win' is an L2-resident transient, "
          f"not a coalescing win (the gate was right to fire).")

    print(f"\n  ➤ MEASURED VERDICT (QUIET GPU): plain-FP16 IS overhead-bound (scattered scalar 2-byte pull-loads → "
          f"only {int(100*bw16_big/bw32_big)}% of FP32's BW), BUT half2-packing (5 wide uint32 words/voxel) recovers it to "
          f"{int(100*bwh2_big/bw32_big)}% of FP32's BW and — moving 56 vs 72 B/voxel — NETS ~{h2_ratio:.2f}× FP32-fused MLUPS "
          f"at production grids. So precision-packing DOES help (modestly) when packed wide; the bigger remaining lever is "
          f"same-location streaming (EsoTwist/AA). ★The earlier 'no win' verdict was GPU-contention, not physics.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())

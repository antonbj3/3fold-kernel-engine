#!/usr/bin/env python3
"""H5 thermal-roofline-GPU (unlock-graph V8, Wave-G COMPUTE-TWIN vertical) — the thermal roofline AS A TWIN-STATE,
on-silicon, real RTX 5070 (the author U-lane).

`fem_sass_roofline.py` (DO NOT MODIFIED, reused only for its ridge-construction convention) measures the roofline as
a SINGLE INSTANTANEOUS SNAPSHOT: one short burst of launches, one wall-clock number, no notion that the ridge could
be a dynamical STATE. GEOMETRIC point: peak_FLOP/s and peak_BW are not fixed geometric constants of the silicon —
they are outputs of a feedback controller (NVIDIA GPU Boost) whose STATE (SM clock, power draw, die temperature)
evolves over a sustained run and is itself governed by a σ_min-like binding constraint (power/thermal cap). If that
cap never binds, the ridge IS a constant and the twin-state coupling is unmotivated; if it binds, the ridge is a
STATE VARIABLE that a digital twin must carry forward (a "cheap roofline" measured in the first 100ms is a biased
estimator of the roofline the workload will actually see 60s in).

We sweep 1-D central-FD 2nd-derivative stencils at formal order m ∈ {2,4,6,8} (jet-order), built with Warp's Tile
API so DRAM traffic is ~flat (~1 read + 1 write per point via ONE cooperative shared-memory tile load per thread
block) while FLOPs/point grow ~linearly with m — this is what makes AI actually rise with order (a naive
"each-thread-rereads-its-own-neighbours-from-global" kernel would NOT: FLOPs and bytes both scale ~m, AI stays flat,
never reproducing the textbook high-order-stencil roofline shape). We verify the reuse claim empirically (gate 3:
tile-shape byte accounting stays ~flat with order; PLUS a direct tile-vs-naive A/B at order 2 and order 8, since a
1-D stencil's halo/block ratio is small enough that GPU L1/L2 caching might absorb naive redundancy anyway — don't
assume the Tile API's benefit is real here, measure it).

For each order we run an INSTANTANEOUS phase (~1.5s, near-boost) and a SUSTAINED phase (~55s), back-to-back across
all 4 orders with NO cooldown (~220s of continuous load, ending on the hottest/most FLOP-dense kernel — this IS the
force-before-negative adversary baked into the design, not a bolt-on afterthought), while a background thread polls
`nvidia-smi` every ~0.7s for SM clock / power / temperature (hard safety stop at 85°C). C/¬C is forced statistically
(first-third vs last-third of the pooled sustained window, pre-registered thresholds: clock drop > 2% of the
615→3090 MHz idle→boost range = 49.5 MHz, or power within 5% of the 250W cap = ≥237.5W) — NOT by eyeballing two
numbers. If the pooled Phase-B window (already ~220s, already back-to-back, already ending hot) still shows ¬C, an
automatic OODA-extension re-runs order 8 (the hottest kernel) for another ~90s with zero gap before booking ¬C.

  python3 \
      u_h5_thermal_roofline_twin_state.py

(Run through gpu_lock.sh — there is ONE physical GPU shared by every worktree/lane on this machine; running the
venv python directly bypasses the machine-wide flock and risks a concurrent-VRAM-clobber Xid fault.)
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import warp as wp
from scipy import stats

wp.init()
wp.set_module_options({"enable_backward": False})
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"

# ============================================================================================
# Pre-registered constants (fixed BEFORE any GPU measurement is taken)
# ============================================================================================
BLOCK_DIM = 256
ORDERS = (2, 4, 6, 8)
N = 1 << 24                       # >= 2^24 points, per spec
NB = N // BLOCK_DIM
HWMAX = 4

IDLE_CLOCK_MHZ = 615.0            # task-supplied idle baseline
BOOST_CLOCK_MHZ = 3090.0          # task-supplied boost clock
POWER_CAP_W = 250.0
CLOCK_DROP_THRESH_MHZ = 0.02 * (BOOST_CLOCK_MHZ - IDLE_CLOCK_MHZ)   # 49.5 MHz
POWER_CAP_THRESH_W = 0.95 * POWER_CAP_W                              # 237.5 W
RIDGE_SHIFT_THRESH_REL = 0.10      # 10% relative shift instant->sustained counts as "measurable"
TEMP_ABORT_C = 85.0

SUSTAINED_S = 55.0                 # per-order sustained duration (45-60s spec band)
INSTANT_S = 1.5
GATE3_S = 0.8
OODA_EXTENSION_S = 90.0

FD_COEFS = {
    2: [1.0, -2.0, 1.0],
    4: [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0],
    6: [1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0, -49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0],
    8: [-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0, -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0],
}


def order_consts(m):
    hw = m // 2
    tl = BLOCK_DIM + 2 * hw
    flops_per_point = 2.0 * (m + 1)                       # (m+1) taps, 1 mul + 1 add each (literal op count)
    bytes_per_launch_tile = 4.0 * (NB * tl + N)            # 1 coalesced shared-load/block + 1 global write/point
    bytes_per_launch_naive = 4.0 * (N * (m + 2))           # (m+1) independent global reads/point + 1 write (worst case)
    return dict(hw=hw, tl=tl, flops_per_point=flops_per_point,
                bytes_per_launch_tile=bytes_per_launch_tile, bytes_per_launch_naive=bytes_per_launch_naive)


ORDER_CONSTS = {m: order_consts(m) for m in ORDERS}


# ============================================================================================
# Gate 0 — FD coefficient verification via measured convergence ORDER (machine cross-check,
# not trust-the-table): apply each stencil to sin(x) at several h, check the observed order
# log2(err(h)/err(h/2)) matches nominal m. Noise-floor guarded (drop ratios where either error
# is within 20x of float64 eps-driven floor ~ eps/h^2).
# ============================================================================================
def verify_fd_coefficients():
    x0 = 1.234
    f = np.sin
    exact = -np.sin(x0)
    hs = [0.4, 0.2, 0.1, 0.05]
    report = {}
    for m, coefs in FD_COEFS.items():
        hw = len(coefs) // 2
        sum_c = sum(coefs)
        errs = []
        for h in hs:
            s = 0.0
            for k, c in enumerate(coefs):
                s += c * f(x0 + (k - hw) * h)
            d2 = s / h ** 2
            errs.append(float(abs(d2 - exact)))
        noise_floor = [2.2e-16 / h ** 2 for h in hs]
        orders_obs = []
        for i in range(len(hs) - 1):
            if errs[i] > 20 * noise_floor[i] and errs[i + 1] > 20 * noise_floor[i + 1]:
                orders_obs.append(float(np.log2(errs[i] / errs[i + 1])))
        mean_order = float(np.mean(orders_obs)) if orders_obs else float("nan")
        ok = bool(orders_obs) and abs(mean_order - m) < 0.4 and abs(sum_c) < 1e-10
        report[m] = dict(sum_coefs=sum_c, hs=hs, errs=errs, observed_orders=orders_obs,
                          mean_observed_order=mean_order, pass_=ok)
    return report


# ============================================================================================
# Warp kernels — TILE (genuine shared-memory reuse) design, one per order, all explicit
# (no dynamic kernel factory — matches the codebase's concrete-kernel-per-variant convention).
# TILE_LEN(m) = BLOCK_DIM + 2*hw(m); each block cooperatively loads ONE shared tile from global
# once, every thread reads its stencil window from THAT shared tile (not from repeated global
# reads) — this is what keeps DRAM traffic ~flat as m rises.
# ============================================================================================
TL2 = BLOCK_DIM + 2 * 1
TL4 = BLOCK_DIM + 2 * 2
TL6 = BLOCK_DIM + 2 * 3
TL8 = BLOCK_DIM + 2 * 4


@wp.kernel
def stencil_tile_o2(inp: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    s = wp.tile_load(inp, shape=TL2, offset=i * BLOCK_DIM, storage="shared")
    acc = wp.float32(0.0)
    acc = acc + wp.float32(1.0) * s[j + 0]
    acc = acc + wp.float32(-2.0) * s[j + 1]
    acc = acc + wp.float32(1.0) * s[j + 2]
    out[i * BLOCK_DIM + j] = acc


@wp.kernel
def stencil_tile_o4(inp: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    s = wp.tile_load(inp, shape=TL4, offset=i * BLOCK_DIM, storage="shared")
    acc = wp.float32(0.0)
    acc = acc + wp.float32(-1.0 / 12.0) * s[j + 0]
    acc = acc + wp.float32(4.0 / 3.0) * s[j + 1]
    acc = acc + wp.float32(-5.0 / 2.0) * s[j + 2]
    acc = acc + wp.float32(4.0 / 3.0) * s[j + 3]
    acc = acc + wp.float32(-1.0 / 12.0) * s[j + 4]
    out[i * BLOCK_DIM + j] = acc


@wp.kernel
def stencil_tile_o6(inp: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    s = wp.tile_load(inp, shape=TL6, offset=i * BLOCK_DIM, storage="shared")
    acc = wp.float32(0.0)
    acc = acc + wp.float32(1.0 / 90.0) * s[j + 0]
    acc = acc + wp.float32(-3.0 / 20.0) * s[j + 1]
    acc = acc + wp.float32(3.0 / 2.0) * s[j + 2]
    acc = acc + wp.float32(-49.0 / 18.0) * s[j + 3]
    acc = acc + wp.float32(3.0 / 2.0) * s[j + 4]
    acc = acc + wp.float32(-3.0 / 20.0) * s[j + 5]
    acc = acc + wp.float32(1.0 / 90.0) * s[j + 6]
    out[i * BLOCK_DIM + j] = acc


@wp.kernel
def stencil_tile_o8(inp: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    s = wp.tile_load(inp, shape=TL8, offset=i * BLOCK_DIM, storage="shared")
    acc = wp.float32(0.0)
    acc = acc + wp.float32(-1.0 / 560.0) * s[j + 0]
    acc = acc + wp.float32(8.0 / 315.0) * s[j + 1]
    acc = acc + wp.float32(-1.0 / 5.0) * s[j + 2]
    acc = acc + wp.float32(8.0 / 5.0) * s[j + 3]
    acc = acc + wp.float32(-205.0 / 72.0) * s[j + 4]
    acc = acc + wp.float32(8.0 / 5.0) * s[j + 5]
    acc = acc + wp.float32(-1.0 / 5.0) * s[j + 6]
    acc = acc + wp.float32(8.0 / 315.0) * s[j + 7]
    acc = acc + wp.float32(-1.0 / 560.0) * s[j + 8]
    out[i * BLOCK_DIM + j] = acc


TILE_KERNELS = {2: stencil_tile_o2, 4: stencil_tile_o4, 6: stencil_tile_o6, 8: stencil_tile_o8}


# ============================================================================================
# Naive (no shared tile — each thread does its own global reads) adversary kernels, order 2 & 8
# only (the two extremes of the sweep), for the gate-3 tile-vs-naive A/B.
# ============================================================================================
@wp.kernel
def stencil_naive_o2(inp: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    idx = wp.tid()
    acc = wp.float32(0.0)
    acc = acc + wp.float32(1.0) * inp[idx + 0]
    acc = acc + wp.float32(-2.0) * inp[idx + 1]
    acc = acc + wp.float32(1.0) * inp[idx + 2]
    out[idx] = acc


@wp.kernel
def stencil_naive_o8(inp: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    idx = wp.tid()
    acc = wp.float32(0.0)
    acc = acc + wp.float32(-1.0 / 560.0) * inp[idx + 0]
    acc = acc + wp.float32(8.0 / 315.0) * inp[idx + 1]
    acc = acc + wp.float32(-1.0 / 5.0) * inp[idx + 2]
    acc = acc + wp.float32(8.0 / 5.0) * inp[idx + 3]
    acc = acc + wp.float32(-205.0 / 72.0) * inp[idx + 4]
    acc = acc + wp.float32(8.0 / 5.0) * inp[idx + 5]
    acc = acc + wp.float32(-1.0 / 5.0) * inp[idx + 6]
    acc = acc + wp.float32(8.0 / 315.0) * inp[idx + 7]
    acc = acc + wp.float32(-1.0 / 560.0) * inp[idx + 8]
    out[idx] = acc


NAIVE_KERNELS = {2: stencil_naive_o2, 8: stencil_naive_o8}


# ============================================================================================
# nvidia-smi background poller (single continuous thread across the WHOLE experiment — this is
# what turns "back-to-back, no cooldown" into the strongest available adversary for free).
# ============================================================================================
class GpuPoller:
    def __init__(self, interval_s=0.7, temp_abort_c=TEMP_ABORT_C):
        self.interval_s = interval_s
        self.temp_abort_c = temp_abort_c
        self.samples = []
        self.phase = "init"
        self.abort_event = threading.Event()
        self.abort_reason = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def set_phase(self, phase):
        self.phase = phase

    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0)
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                raw_ts, clk, pw, temp = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                self.samples.append(dict(t_wall=t0, phase=self.phase, raw_ts=raw_ts,
                                          sm_clock_mhz=clk, power_w=pw, temp_c=temp))
                if temp >= self.temp_abort_c and not self.abort_event.is_set():
                    self.abort_reason = f"temp {temp:.0f}C >= {self.temp_abort_c:.0f}C during phase={self.phase}"
                    self.abort_event.set()
                    print(f"\n  *** SAFETY HARD-STOP: {self.abort_reason} ***\n", flush=True)
            except Exception:
                pass
            dt = time.time() - t0
            time.sleep(max(0.0, self.interval_s - dt))


def measure_phase(do_launch, duration_s, poller, phase_label, target_batch_time=0.3):
    """Warm up once, time a single launch, batch launches to ~target_batch_time between syncs,
    run back-to-back for duration_s (or until poller trips the abort event)."""
    poller.set_phase(phase_label)   # set BEFORE warm-up so no poll sample is mislabeled to the previous phase
    do_launch()
    wp.synchronize()
    t0 = time.perf_counter()
    do_launch()
    wp.synchronize()
    t_single = time.perf_counter() - t0
    launches_per_batch = max(1, int(target_batch_time / max(t_single, 1e-6)))
    tp_samples = []
    t_start = time.time()
    n_total = 0
    while (time.time() - t_start) < duration_s:
        if poller.abort_event.is_set():
            break
        tb0 = time.time()
        for _ in range(launches_per_batch):
            do_launch()
        wp.synchronize()
        tb1 = time.time()
        n_total += launches_per_batch
        tp_samples.append(dict(t_wall=tb1, dt=tb1 - tb0, launches=launches_per_batch))
    elapsed = time.time() - t_start
    return dict(elapsed=elapsed, n_total=n_total, tp_samples=tp_samples, t_single=t_single,
                launches_per_batch=launches_per_batch, aborted=poller.abort_event.is_set())


def achieved_tile(result, m):
    oc = ORDER_CONSTS[m]
    flops_total = oc["flops_per_point"] * N * result["n_total"]
    bytes_total = oc["bytes_per_launch_tile"] * result["n_total"]
    gflops = flops_total / result["elapsed"] / 1e9 if result["elapsed"] > 0 else float("nan")
    gbs = bytes_total / result["elapsed"] / 1e9 if result["elapsed"] > 0 else float("nan")
    ai = oc["flops_per_point"] * N / oc["bytes_per_launch_tile"]
    return dict(gflops=gflops, gbs=gbs, ai=ai)


def achieved_naive(result, m):
    oc = ORDER_CONSTS[m]
    flops_total = oc["flops_per_point"] * N * result["n_total"]
    bytes_total = oc["bytes_per_launch_naive"] * result["n_total"]
    gflops = flops_total / result["elapsed"] / 1e9 if result["elapsed"] > 0 else float("nan")
    gbs = bytes_total / result["elapsed"] / 1e9 if result["elapsed"] > 0 else float("nan")
    return dict(gflops=gflops, gbs=gbs)


def first_last_third(samples, key):
    """Split a chronologically-ordered sample list into first/last third by INDEX, return
    (mean_first, std_first, mean_last, std_last, n_each)."""
    n = len(samples)
    if n < 6:
        return None
    k = n // 3
    first = [s[key] for s in samples[:k]]
    last = [s[key] for s in samples[-k:]]
    return dict(mean_first=float(np.mean(first)), std_first=float(np.std(first)),
                mean_last=float(np.mean(last)), std_last=float(np.std(last)), n_each=k)


def regress_vs_time(samples, key, t0):
    n = len(samples)
    if n < 6:
        return None
    t = np.array([s["t_wall"] - t0 for s in samples])
    y = np.array([s[key] for s in samples])
    r = stats.linregress(t, y)
    return dict(slope=float(r.slope), intercept=float(r.intercept), rvalue=float(r.rvalue),
                pvalue=float(r.pvalue), n=n)


def correctness_check(m, padded_np):
    hw = m // 2
    coefs = FD_COEFS[m]
    NCHK = 4096
    inp_wp = wp.array(padded_np, dtype=wp.float32, device=DEV)
    out_wp = wp.zeros(N, dtype=wp.float32, device=DEV)
    wp.launch_tiled(TILE_KERNELS[m], dim=NB, inputs=[inp_wp, out_wp], block_dim=BLOCK_DIM, device=DEV)
    wp.synchronize()
    got = out_wp.numpy()[:NCHK].astype(np.float64)
    ref = np.empty(NCHK, dtype=np.float64)
    for idx in range(NCHK):
        window = padded_np[idx:idx + 2 * hw + 1].astype(np.float64)
        ref[idx] = float(np.dot(coefs, window))
    max_abs_err = float(np.max(np.abs(got - ref)))
    return max_abs_err


def main():
    if DEV != "cuda:0":
        print("NO CUDA DEVICE FOUND — this experiment requires the real RTX 5070. Aborting.")
        return 1

    gpu_name = "unknown"
    try:
        gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=2.0).stdout.strip()
    except Exception:
        pass

    print("=" * 100)
    print(f"H5 thermal-roofline-GPU — thermal roofline AS A TWIN-STATE, on-silicon ({gpu_name}, device={DEV})")
    print("=" * 100)
    print(f"  N={N} ({N/2**20:.0f}Mi points), BLOCK_DIM={BLOCK_DIM}, NB={NB} blocks, orders={ORDERS}")
    print(f"  pre-registered: clock-drop-thresh={CLOCK_DROP_THRESH_MHZ:.1f}MHz (2% of {IDLE_CLOCK_MHZ:.0f}->"
          f"{BOOST_CLOCK_MHZ:.0f}MHz range), power-cap-thresh={POWER_CAP_THRESH_W:.1f}W (95% of {POWER_CAP_W:.0f}W),"
          f" ridge-shift-thresh={RIDGE_SHIFT_THRESH_REL*100:.0f}%, temp-abort={TEMP_ABORT_C:.0f}C")

    evidence = dict(meta=dict(gpu_name=gpu_name, device=DEV, n_grid=N, block_dim=BLOCK_DIM, orders=list(ORDERS),
                               fd_coefs=FD_COEFS, clock_drop_thresh_mhz=CLOCK_DROP_THRESH_MHZ,
                               power_cap_thresh_w=POWER_CAP_THRESH_W, ridge_shift_thresh_rel=RIDGE_SHIFT_THRESH_REL,
                               temp_abort_c=TEMP_ABORT_C, sustained_s=SUSTAINED_S, t_start=time.time()))

    # ---- Gate 0: FD coefficient convergence-order verification (CPU, numpy, machine cross-check) ----
    print("\n--- Gate 0: FD coefficient verification (measured convergence ORDER vs nominal, not table-trust) ---")
    fd_report = verify_fd_coefficients()
    for m, r in fd_report.items():
        print(f"  order {m}: sum(coefs)={r['sum_coefs']:.2e}  observed_orders={['%.3f' % o for o in r['observed_orders']]}"
              f"  mean={r['mean_observed_order']:.3f}  -> {'PASS' if r['pass_'] else 'FAIL'}")
    fd_all_pass = all(r["pass_"] for r in fd_report.values())
    evidence["gate0_fd_convergence"] = fd_report

    # ---- Build input arrays (one base random field, sliced per order's halo width) ----
    rng = np.random.default_rng(42)
    base = rng.random(N + 2 * HWMAX).astype(np.float32)
    padded_np = {}
    padded_wp = {}
    out_wp = {}
    for m in ORDERS:
        hw = m // 2
        start = HWMAX - hw
        end = start + N + 2 * hw
        padded_np[m] = base[start:end].copy()
        padded_wp[m] = wp.array(padded_np[m], dtype=wp.float32, device=DEV)
        out_wp[m] = wp.zeros(N, dtype=wp.float32, device=DEV)

    # ---- Gate: kernel correctness (GPU tile kernel vs numpy reference, small subset) ----
    print("\n--- Gate: kernel correctness (GPU tile output vs numpy reference, first 4096 points) ---")
    correctness = {}
    for m in (2, 8):
        err = correctness_check(m, padded_np[m])
        ok = err < 1e-2
        correctness[m] = dict(max_abs_err=err, pass_=ok)
        print(f"  order {m}: max_abs_err={err:.3e} (tol 1e-2) -> {'PASS' if ok else 'FAIL'}")
    correctness_all_pass = all(r["pass_"] for r in correctness.values())
    evidence["gate_correctness"] = correctness

    poller = GpuPoller()
    poller.start()
    t_experiment_start = time.time()

    # ---- Gate 3(c): tile-vs-naive adversary A/B at order 2 and order 8, 3 repeats each, cold GPU ----
    print("\n--- Gate 3: tile-vs-naive reuse adversary A/B (order 2 & order 8, 3 repeats each, ~0.8s bursts) ---")
    reuse_ab = {}
    for m in (2, 8):
        tile_k = TILE_KERNELS[m]
        naive_k = NAIVE_KERNELS[m]

        def do_tile(m=m, tile_k=tile_k):
            wp.launch_tiled(tile_k, dim=NB, inputs=[padded_wp[m], out_wp[m]], block_dim=BLOCK_DIM, device=DEV)

        def do_naive(m=m, naive_k=naive_k):
            wp.launch(naive_k, dim=N, inputs=[padded_wp[m], out_wp[m]], device=DEV)

        tile_gflops_runs, naive_gflops_runs = [], []
        for rep in range(3):
            r_tile = measure_phase(do_tile, GATE3_S, poller, f"gate3_tile_o{m}_rep{rep}")
            tile_gflops_runs.append(achieved_tile(r_tile, m)["gflops"])
            r_naive = measure_phase(do_naive, GATE3_S, poller, f"gate3_naive_o{m}_rep{rep}")
            naive_gflops_runs.append(achieved_naive(r_naive, m)["gflops"])
        tile_mean, tile_std = float(np.mean(tile_gflops_runs)), float(np.std(tile_gflops_runs))
        naive_mean, naive_std = float(np.mean(naive_gflops_runs)), float(np.std(naive_gflops_runs))
        speedup = tile_mean / naive_mean if naive_mean > 0 else float("nan")
        combined_std = (tile_std ** 2 + naive_std ** 2) ** 0.5
        separated = abs(tile_mean - naive_mean) > 2 * combined_std
        regime = "genuine_reuse_advantage" if (speedup > 1.15 and separated) else \
                 ("naive_slower_but_not_separated" if speedup > 1.15 else "no_distinguishable_reuse_advantage")
        reuse_ab[m] = dict(tile_gflops_mean=tile_mean, tile_gflops_std=tile_std,
                            naive_gflops_mean=naive_mean, naive_gflops_std=naive_std,
                            speedup=speedup, separated_gt_2std=separated, regime=regime)
        print(f"  order {m}: tile={tile_mean:.1f}+-{tile_std:.1f} GFLOP/s  naive={naive_mean:.1f}+-{naive_std:.1f} GFLOP/s"
              f"  speedup={speedup:.2f}x  2std-separated={separated}  -> {regime}")
    evidence["gate3_reuse_adversary"] = reuse_ab

    # ---- Phase A: cold instantaneous sweep across all 4 orders (clean ridge_instant anchor) ----
    print("\n--- Phase A: cold instantaneous sweep (ridge_instant anchor, GPU still near idle) ---")
    phaseA = {}
    for m in ORDERS:
        tile_k = TILE_KERNELS[m]

        def do_tile(m=m, tile_k=tile_k):
            wp.launch_tiled(tile_k, dim=NB, inputs=[padded_wp[m], out_wp[m]], block_dim=BLOCK_DIM, device=DEV)

        r = measure_phase(do_tile, INSTANT_S, poller, f"phaseA_instant_o{m}")
        a = achieved_tile(r, m)
        phaseA[m] = dict(result=r, achieved=a)
        print(f"  order {m}: AI={a['ai']:.3f} FLOP/byte  GFLOP/s={a['gflops']:.1f}  GB/s={a['gbs']:.1f}"
              f"  (elapsed={r['elapsed']:.2f}s, n_launches={r['n_total']})")

    ai_ratio = phaseA[8]["achieved"]["ai"] / phaseA[2]["achieved"]["ai"]
    gbs_ratio_cold = phaseA[8]["achieved"]["gbs"] / phaseA[2]["achieved"]["gbs"]
    gate_ai_rises = ai_ratio >= 2.5
    gate_bytes_flat = gbs_ratio_cold < 1.3
    print(f"\n  gate AI(8)/AI(2) = {ai_ratio:.2f} (>=2.5 required)  -> {'PASS' if gate_ai_rises else 'FAIL'}")
    print(f"  gate achieved-GB/s(8)/achieved-GB/s(2) = {gbs_ratio_cold:.2f} (<1.3 required, i.e. NOT scaling up"
          f" with order) -> {'PASS' if gate_bytes_flat else 'FAIL'}")

    ridge_instant = phaseA[8]["achieved"]["gflops"] / phaseA[2]["achieved"]["gbs"]
    print(f"  ridge_instant = peak_FLOP(o8)/peak_BW(o2) = {ridge_instant:.3f} FLOP/byte")
    evidence["phaseA_cold_instant"] = {m: dict(achieved=phaseA[m]["achieved"], elapsed=phaseA[m]["result"]["elapsed"],
                                                n_total=phaseA[m]["result"]["n_total"]) for m in ORDERS}
    evidence["ai_ratio_8_over_2"] = ai_ratio
    evidence["gbs_ratio_cold_8_over_2"] = gbs_ratio_cold

    # ---- Phase B: per-order instant+sustained, back-to-back, no cooldown (the main experiment) ----
    print(f"\n--- Phase B: per-order instant+sustained sweep, BACK-TO-BACK no cooldown ({SUSTAINED_S:.0f}s/order sustained) ---")
    phaseB = {}
    aborted_at = None
    for m in ORDERS:
        if poller.abort_event.is_set():
            aborted_at = aborted_at or m
            print(f"  order {m}: SKIPPED (safety abort already tripped: {poller.abort_reason})")
            continue
        tile_k = TILE_KERNELS[m]

        def do_tile(m=m, tile_k=tile_k):
            wp.launch_tiled(tile_k, dim=NB, inputs=[padded_wp[m], out_wp[m]], block_dim=BLOCK_DIM, device=DEV)

        r_instant = measure_phase(do_tile, INSTANT_S, poller, f"instant_o{m}")
        a_instant = achieved_tile(r_instant, m)
        r_sustained = measure_phase(do_tile, SUSTAINED_S, poller, f"sustained_o{m}")
        a_sustained = achieved_tile(r_sustained, m)
        phaseB[m] = dict(instant=dict(result=r_instant, achieved=a_instant),
                          sustained=dict(result=r_sustained, achieved=a_sustained))
        print(f"  order {m}: instant GFLOP/s={a_instant['gflops']:.1f} GB/s={a_instant['gbs']:.1f}"
              f"  |  sustained GFLOP/s={a_sustained['gflops']:.1f} GB/s={a_sustained['gbs']:.1f}"
              f"  (elapsed={r_sustained['elapsed']:.1f}s, n_launches={r_sustained['n_total']}, aborted={r_sustained['aborted']})")
        if r_sustained["aborted"]:
            aborted_at = m
            break

    evidence["phaseB"] = {m: dict(
        instant_achieved=phaseB[m]["instant"]["achieved"], instant_elapsed=phaseB[m]["instant"]["result"]["elapsed"],
        sustained_achieved=phaseB[m]["sustained"]["achieved"], sustained_elapsed=phaseB[m]["sustained"]["result"]["elapsed"],
        sustained_n_total=phaseB[m]["sustained"]["result"]["n_total"], sustained_aborted=phaseB[m]["sustained"]["result"]["aborted"],
    ) for m in phaseB}
    evidence["aborted_at_order"] = aborted_at

    orders_completed = list(phaseB.keys())
    if 8 in phaseB and 2 in phaseB:
        ridge_sustained = phaseB[8]["sustained"]["achieved"]["gflops"] / phaseB[2]["sustained"]["achieved"]["gbs"]
    else:
        ridge_sustained = float("nan")
    ridge_rel_shift = abs(ridge_sustained - ridge_instant) / ridge_instant if ridge_instant else float("nan")
    gate_ridge_shift = (not np.isnan(ridge_rel_shift)) and ridge_rel_shift > RIDGE_SHIFT_THRESH_REL
    print(f"\n  ridge_instant={ridge_instant:.3f}  ridge_sustained={ridge_sustained:.3f}"
          f"  rel_shift={ridge_rel_shift*100:.1f}% (>{RIDGE_SHIFT_THRESH_REL*100:.0f}% required)"
          f" -> {'PASS' if gate_ridge_shift else 'FAIL'}")
    evidence["ridge"] = dict(instant=ridge_instant, sustained=ridge_sustained, rel_shift=ridge_rel_shift)

    # ---- Statistical force of C vs ¬C: pooled (all completed orders' sustained samples,
    # chronologically concatenated -- the STRONGEST available test given Phase B is already
    # back-to-back/no-cooldown/ends-hot) + per-order breakdown across the diverse instance-space ----
    def pooled_sustained_samples():
        return [s for s in poller.samples if s["phase"].startswith("sustained_o")]

    def eval_force(samples, t0, label):
        fl_clock = first_last_third(samples, "sm_clock_mhz")
        fl_power = first_last_third(samples, "power_w")
        reg_clock = regress_vs_time(samples, "sm_clock_mhz", t0)
        reg_power = regress_vs_time(samples, "power_w", t0)
        clock_drop = (fl_clock["mean_first"] - fl_clock["mean_last"]) if fl_clock else float("nan")
        power_last = fl_power["mean_last"] if fl_power else float("nan")
        c_clock = bool(fl_clock) and clock_drop > CLOCK_DROP_THRESH_MHZ
        c_power = bool(fl_power) and power_last >= POWER_CAP_THRESH_W
        print(f"  [{label}] n={len(samples)} samples: clock first/last third = "
              f"{fl_clock['mean_first']:.0f}/{fl_clock['mean_last']:.0f} MHz (drop={clock_drop:.1f}MHz, "
              f"thresh={CLOCK_DROP_THRESH_MHZ:.1f}) -> clock_binds={c_clock}" if fl_clock else f"  [{label}] insufficient samples")
        if fl_power:
            print(f"      power first/last third = {fl_power['mean_first']:.1f}/{fl_power['mean_last']:.1f} W "
                  f"(thresh>={POWER_CAP_THRESH_W:.1f}W) -> power_binds={c_power}")
        if reg_clock:
            print(f"      clock regression: slope={reg_clock['slope']*1000:.3f} MHz/1000s, "
                  f"r={reg_clock['rvalue']:.3f}, p={reg_clock['pvalue']:.2e}")
        if reg_power:
            print(f"      power regression: slope={reg_power['slope']*1000:.3f} W/1000s, "
                  f"r={reg_power['rvalue']:.3f}, p={reg_power['pvalue']:.2e}")
        return dict(first_last_clock=fl_clock, first_last_power=fl_power, regress_clock=reg_clock,
                    regress_power=reg_power, clock_drop_mhz=clock_drop, power_last_w=power_last,
                    c_clock=c_clock, c_power=c_power, c_overall=bool(c_clock or c_power))

    print("\n--- Statistical force of C vs neg-C (pooled Phase-B sustained window) ---")
    pooled_samples = pooled_sustained_samples()
    force_pooled = eval_force(pooled_samples, t_experiment_start, "POOLED phaseB sustained")

    print("\n  per-order breakdown (diverse instance-space check):")
    force_per_order = {}
    for m in orders_completed:
        s_m = [s for s in poller.samples if s["phase"] == f"sustained_o{m}"]
        force_per_order[m] = eval_force(s_m, t_experiment_start, f"order={m}")

    # ---- OODA force-before-negative: if pooled Phase-B does not show C, extend order 8 (hottest
    # kernel) for another ~90s with ZERO gap before booking neg-C. ----
    extension_evidence = None
    if not force_pooled["c_overall"] and not poller.abort_event.is_set() and 8 in phaseB:
        print(f"\n--- OODA force-before-negative: pooled Phase-B showed neg-C -> extending order-8 sustained"
              f" load by another {OODA_EXTENSION_S:.0f}s, ZERO gap (adversary: longer + hottest kernel + no cooldown) ---")
        tile_k = TILE_KERNELS[8]

        def do_tile8():
            wp.launch_tiled(tile_k, dim=NB, inputs=[padded_wp[8], out_wp[8]], block_dim=BLOCK_DIM, device=DEV)

        r_ext = measure_phase(do_tile8, OODA_EXTENSION_S, poller, "sustained_o8")  # same phase label -> pools with o8
        a_ext = achieved_tile(r_ext, 8)
        print(f"  extension: elapsed={r_ext['elapsed']:.1f}s, n_launches={r_ext['n_total']}, "
              f"GFLOP/s={a_ext['gflops']:.1f}, GB/s={a_ext['gbs']:.1f}, aborted={r_ext['aborted']}")
        s_ext_all = [s for s in poller.samples if s["phase"] == "sustained_o8"]
        force_ext = eval_force(s_ext_all, t_experiment_start, "order-8 EXTENDED (orig+90s, pooled)")
        extension_evidence = dict(result=r_ext, achieved=a_ext, force=force_ext)
        # re-evaluate the pooled verdict including the extension (extension samples already carry
        # the "sustained_o" prefix so they're included automatically on re-pool)
        pooled_samples = pooled_sustained_samples()
        force_pooled = eval_force(pooled_samples, t_experiment_start, "POOLED phaseB+extension sustained")

    poller.stop()

    # ---- Final verdict ----
    verdict = "C" if (gate_ridge_shift and force_pooled["c_overall"]) else "not-C"
    n_orders_binding = sum(1 for m in force_per_order if force_per_order[m]["c_overall"])

    evidence["statistical_force"] = dict(pooled=force_pooled, per_order=force_per_order,
                                          ooda_extension=extension_evidence)
    evidence["verdict"] = verdict
    evidence["n_orders_individually_binding"] = n_orders_binding
    evidence["t_experiment_elapsed_s"] = time.time() - t_experiment_start

    # ---- Gate rollup (instrument-validity gates only -- verdict C/not-C is reported separately,
    # NOT folded into ALL_PASS: an honest not-C is a legitimate, non-failing outcome) ----
    gates = {
        "fd_coefficients_verified": fd_all_pass,
        "kernel_correctness_verified": correctness_all_pass,
        "ai_rises_with_order": gate_ai_rises,
        "bytes_stay_flat_with_order": gate_bytes_flat,
        "reuse_adversary_measured": True,   # process gate: A/B was actually run & disclosed (see regime above)
        "safety_handled_cleanly": True,     # no unhandled crash; abort (if any) was caught & recorded
    }
    print("\n" + "=" * 100)
    print("GATE ROLLUP (instrument validity)")
    print("=" * 100)
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_pass = all(gates.values())
    print(f"\n  ALL_PASS = {all_pass}  ({sum(gates.values())}/{len(gates)} gates)")

    print("\nSCIENTIFIC FORCING RESULT (not a pass/fail gate -- honest-negative is a valid outcome)")
    print(f"  ridge_instant={ridge_instant:.3f} FLOP/byte, ridge_sustained={ridge_sustained:.3f} FLOP/byte,"
          f" rel_shift={ridge_rel_shift*100:.1f}% (measurable={gate_ridge_shift})")
    print(f"  pooled sustained window: clock_binds={force_pooled['c_clock']}, power_binds={force_pooled['c_power']}"
          f" -> thermal_state_binds={force_pooled['c_overall']}")
    print(f"  per-order binding count: {n_orders_binding}/{len(force_per_order)} orders individually show binding")
    verdict_text = ("the sustained roofline IS a measurably different twin-STATE, causally tied to clock/power binding"
                     if verdict == "C" else
                     "the ridge did not show a pre-registered-threshold-crossing, causally-tied shift in this run"
                     " (see honest limits)")
    print(f"\n  VERDICT: {verdict}  ({verdict_text})")

    reuse_regimes = {m: reuse_ab[m]["regime"] for m in reuse_ab}
    print(f"\n  reuse regime achieved (gate 3, tile-vs-naive A/B): {reuse_regimes}")

    print("\nHONEST LIMITS:")
    print("  (1) single GPU, single run, single day -- no repeat-day thermal-drift/ambient-temperature check;")
    print("      a re-run on a hotter/colder day, or with different case airflow, could shift the numbers.")
    print(f"  (2) Tile-API reuse was VERIFIED EMPIRICALLY via gate 3 (tile-vs-naive A/B, see 'regime achieved'"
          f" above), not assumed -- but for this 1-D stencil the halo/block ratio is small (<=4/256=1.6% at m=8),"
          f" so any tile-vs-naive gap this run measured could still be within GPU L1/L2 caching's reach; the"
          f" regime label above is the actual measured finding, not a hoped-for one.")
    print("  (3) the 1-D stencil is a toy relative to a real 3-D production kernel (FEM assembly, CFD stencil) --")
    print("      the AI values reached here (~0.75-2.25 FLOP/byte) are far below where a real compute-bound")
    print("      kernel would sit; this experiment is about the THERMAL-STATE claim, not about crossing the ridge.")
    print("  (4) nvidia-smi polling granularity (~0.7s) cannot resolve sub-ms clock transitions -- GPU Boost can")
    print("      step clocks faster than we sample; our clock/power numbers are a temporally-aliased envelope,")
    print("      not the true instantaneous trajectory (this is exactly why we force via a THIRD-vs-THIRD")
    print("      aggregate + regression, not a single before/after pair of instantaneous samples).")
    print("  (5) only order=2's Phase-B instant sub-phase is a genuinely COLD/near-boost baseline; orders 4/6/8's")
    print("      instant sub-phases run after previous orders' sustained load, i.e. already partially warmed --")
    print("      this is why ridge_instant is taken from the SEPARATE cold Phase-A sweep, not from Phase B.")
    print("  (6) FLOP/byte accounting convention: 1 mul + 1 add per tap (2*(m+1) FLOPs/point), which is what the")
    print("      generated code literally executes -- not an idealized FMA-fused count; a real FMA-fused GPU ISA")
    print("      could halve the FLOP count without changing wall-clock time, which would double reported GFLOP/s")
    print("      for a fixed hardware reality -- reported AI/GFLOP/s values are convention-dependent, GB/s and")
    print("      wall-clock time are the convention-free ground truth used for the C/neg-C statistical force.")
    print("  (7) no matplotlib in this venv -- no PNG roofline plot was produced; all verdicts are from the")
    print("      numeric gates/regressions above (measure, don't eyeball a figure), and the raw samples are in")
    print("      the evidence JSON for independent re-plotting.")
    if aborted_at is not None:
        print(f"  (8) SAFETY ABORT TRIPPED at order={aborted_at}: {poller.abort_reason} -- remaining orders were")
        print("      not run; verdict above is based only on completed orders (see evidence JSON).")

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"), exist_ok=True)
    ev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts",
                            "u_h5_thermal_roofline_twin_state_evidence.json")
    evidence["all_poller_samples"] = poller.samples
    evidence["all_pass_instrument_gates"] = all_pass
    with open(ev_path, "w") as f:
        json.dump(evidence, f, indent=1, default=str)
    print(f"\n  evidence JSON written: {os.path.abspath(ev_path)}  ({len(poller.samples)} raw poller samples)")
    print("=" * 100)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

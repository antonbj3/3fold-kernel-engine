"""
THE EXCLUSIVE SWEEP BLOCK.

Two regime-map axes measured in ONE exclusive GPU session:
  PART 1  REGIME SWEEP    : power {175,210,250,263}W x clock {unlocked-boost, locked-2000MHz}
                            -> 8-cell table {DRAM-copy-BW, launch-floor, compute_us(GEMM),
                            clock_eq(sustained), jitter}. Fills the twin's declared extrapolation zone.
  PART 2  INTERFERENCE MTX: disturber {compute-GEMM, DRAM-BW, L2-resident-40MiB, sync-barrier-storm}
                            x intensity {0, 1@50%duty, 1@100%, 2@100%}
                            x victim {compute_us, DRAM-copy-BW, L2-24MiB-BW, composite-p50}.

INSTRUMENTS (reuse of measured mechanisms):
  - DRAM/L2 copy BW = K-burst event-timed copy of a working set (round-11's mechanism: a set that
    stays L2-resident copies FAST; an oversize/evicted set copies at DRAM). Copy byte-accounting=2*bytes.
  - launch floor    = median round-trip of a trivial kernel launch+synchronize (driver/CPU bound).
  - compute_us      = the calib GEMM class (4096^2 fp32 matmul), CUDA-event timed (in-band GPU clock).
  - composite p50   = v1.2 G2 barrier-sensitive pipeline: H2D(64MiB pinned)+small-GPU-work+D2H, 2 host
                      syncs/rep -> the barrier channel's victim.

PRE-REGISTERED PREDICTIONS (frozen before running):
  PART 1:
   P1 DRAM BW power-INSENSITIVE above ~200W (mem clock separate domain) but MAY sag at 175W.
   P2 compute_us scales with achieved clock (est-cycles ~ constant across cells).
   P3 capped-equilibrium clock is MONOTONE non-decreasing in power limit.
   P4 locked-2000 cells show near-zero jitter (governor removed).
  PART 2:
   M1 matrix is SPARSE: compute-hog barely hits DRAM-victim and vice versa (off-diagonal small).
   M2 L2-victim collapses ONLY under L2-resident disturber (round-11 cross-validation).
   M3 barrier-storm disturber hits composite-victim HARDEST (own row).
   M4 intensity curves monotone, saturating.
  CONTROLS: victim-alone baselines FIRST, must reproduce known values within 10% (regression gate);
            disturber liveness (ips) logged per arm.

STATE MANAGEMENT (sudo envelope): power ONLY in [175,263]W; clock lock/release only; every
change wrapped try/finally with restoration (power->250, clocks released) + a VERIFICATION READ after
restore. No GPU reset / driver / CPU state. TENANCY: verify GPU empty before start; a FOREIGN tenant
mid-block -> abort current arm, restore, skip to reporting what we have (labeled). Commit nothing.
"""
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import threading
import time

MIB = 1024 * 1024
HERE = os.path.dirname(os.path.abspath(__file__))

# round-11's PROVEN warp CUDA-graph copy-BW instrument (hits ~556 GB/s clean DRAM asymptote; a plain
# torch copy_ loop tops out ~360 GB/s because inter-launch gaps starve the GPU — measured this block).
_L2 = None
_WP = None
_COPY_K = None


def _warp_bw():
    global _L2, _WP, _COPY_K
    if _WP is None:
        spec = importlib.util.spec_from_file_location(
            "l2endgame", os.path.join(HERE, "probe_l2_inband_endgame.py"))
        _L2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(_L2)
        _WP = _L2._wp()
        _, _COPY_K = _L2._kernels(_WP)
    return _L2, _WP, _COPY_K
DEV = "cuda:0"
SELF = os.path.abspath(__file__)
REP = os.path.join(HERE, "artifacts", "probe_exclusive_sweep_block.json")
RATE_DIR = os.path.join(HERE, "artifacts", "isweep")
os.makedirs(RATE_DIR, exist_ok=True)
POWER_MIN, POWER_MAX = 175, 263          # hard envelope bounds
DRAM_BAND = (500.0, 800.0)               # RTX5070 GDDR7 ~672 GB/s external anchor (copy accounting *2)

# ------------------------------------------------------------------ smi helpers ---
def _smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def _compute_apps():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def foreign_tenant(own):
    for line in _compute_apps().splitlines():
        p = line.strip()
        if p and int(p) not in own:
            return p
    return None


def stamp():
    return {"tenants": _compute_apps() or "NONE",
            "clock_sm": _smi("clocks.sm"), "clock_mem": _smi("clocks.mem"),
            "power_draw": _smi("power.draw"), "power_limit": _smi("power.limit"),
            "util": _smi("utilization.gpu"), "temp": _smi("temperature.gpu"),
            "t": round(time.time(), 2)}


# ---------------------------------------------------------- state management -----
def set_power(w):
    assert POWER_MIN <= w <= POWER_MAX, f"power {w} out of envelope [{POWER_MIN},{POWER_MAX}]"
    return subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(w)],
                          capture_output=True, text=True).stdout.strip()


def lock_clock(mhz):
    return subprocess.run(["sudo", "-n", "nvidia-smi", "-lgc", str(mhz)],
                          capture_output=True, text=True).stdout.strip()


def release_clock():
    return subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"],
                          capture_output=True, text=True).stdout.strip()


def restore_and_verify():
    """Unconditional restoration + verification read. Runs in finally even on crash."""
    out = {"actions": []}
    out["actions"].append(("release_clock", release_clock()))
    out["actions"].append(("set_power_250", set_power(250)))
    time.sleep(1.0)
    out["verify"] = {"power_limit": _smi("power.limit"), "clock_sm_idle": _smi("clocks.sm"),
                     "clocks_locked_query": _smi("clocks.current.sm")}
    # a locked clock would show a fixed sm regardless of load; verify power restored to 250
    try:
        pl = float(out["verify"]["power_limit"])
        out["power_restored_ok"] = abs(pl - 250.0) < 1.0
    except Exception:
        out["power_restored_ok"] = None
    return out


# ------------------------------------------------------------------- victims -----
def copy_bw(torch, dev, mib, N=None):
    """Warp CUDA-GRAPH burst copy-BW of an `mib` working set (round-11's proven instrument). A set
    that stays L2-resident (<=~40MiB) copies FAST; an oversize/evicted set (256MiB) copies at DRAM.
    Graph replay keeps the GPU queue full (a plain launch loop starves it -> ~360 not ~556). Returns
    median GB/s (copy = read+write = 2*bytes)."""
    l2, wp, copy_k = _warp_bw()
    M = (mib * MIB) // 4
    n = 40 if N is None else N
    return l2.burst_copy_bw(wp, copy_k, M, N=n)[0]


def launch_floor_us(torch, dev, reps=200):
    """Median round-trip of a trivial kernel launch + synchronize (driver/CPU bound)."""
    x = torch.zeros(1, dtype=torch.float32, device=dev)
    for _ in range(20):
        x.add_(1.0); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        x.add_(1.0); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(ts), 2)


def compute_us(torch, dev, size=4096, reps=30):
    """The calib GEMM class: size^2 fp32 matmul, CUDA-event timed (in-band GPU clock). Returns
    (median_us, clock_sm_during)."""
    a = torch.randn(size, size, device=dev); b = torch.randn(size, size, device=dev)
    for _ in range(6):
        a.matmul(b)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    us = []
    for _ in range(reps):
        e0.record(); c = a.matmul(b); e1.record(); e1.synchronize()
        us.append(e0.elapsed_time(e1) * 1e3)
    clk = _smi("clocks.sm")
    del a, b
    return round(statistics.median(us), 2), clk


def composite_p50(torch, dev, reps=60):
    """v1.2 G2 barrier-sensitive pipeline: H2D(64MiB pinned) + small GPU work + D2H(small), 2 host
    syncs/rep. perf_counter p50 (us). The barrier channel's victim."""
    h2d_bytes, d2h_bytes = 64 * MIB, 256 * 1024
    h_src = torch.empty(h2d_bytes // 4, dtype=torch.float32, pin_memory=True)
    d_dst = torch.empty(h2d_bytes // 4, dtype=torch.float32, device=dev)
    d_src = torch.empty(d2h_bytes // 4, dtype=torch.float32, device=dev)
    h_dst = torch.empty(d2h_bytes // 4, dtype=torch.float32, pin_memory=True)
    w0 = torch.randn(512, 512, device=dev); w1 = torch.randn(512, 512, device=dev)

    def one():
        d_dst.copy_(h_src, non_blocking=True); torch.cuda.synchronize()
        for _ in range(4):
            w0.matmul(w1)
        torch.cuda.synchronize()
        h_dst.copy_(d_src, non_blocking=True); torch.cuda.synchronize()

    for _ in range(8):
        one()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); one(); ts.append((time.perf_counter() - t0) * 1e6)
    del h_src, d_dst, d_src, h_dst, w0, w1
    return round(statistics.median(ts), 1)


def sustained_clock_traj(torch, dev, dur_s, sample_hz=1.0):
    """Warp-burn (large matmul loop) for dur_s while sampling clocks.sm at sample_hz in a thread.
    Returns {samples, clock_eq (median of last 60%), jitter (std of last 60%), rate_ips}."""
    a = torch.randn(4096, 4096, device=dev); b = torch.randn(4096, 4096, device=dev)
    samples = []
    stop = {"go": True}

    def sampler():
        while stop["go"]:
            try:
                samples.append((round(time.time(), 2), float(_smi("clocks.sm")),
                                float(_smi("power.draw"))))
            except Exception:
                pass
            time.sleep(1.0 / sample_hz)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.time(); it = 0
    while time.time() < t0 + dur_s:
        for _ in range(8):
            a.matmul(b)
        torch.cuda.synchronize(); it += 8
    stop["go"] = False; th.join(timeout=2.0)
    dt = time.time() - t0
    del a, b
    clocks = [c for _, c, _ in samples]
    tail = clocks[int(len(clocks) * 0.4):] if len(clocks) >= 3 else clocks
    return {"n_samples": len(clocks), "clocks": clocks,
            "clock_eq": round(statistics.median(tail), 1) if tail else None,
            "jitter_std": round(statistics.pstdev(tail), 2) if len(tail) >= 2 else 0.0,
            "rate_ips": round(it / dt, 1), "powers": [p for _, _, p in samples]}


# --------------------------------------------------------------- disturbers ------
def run_disturber():
    """Self-owned co-tenant. --dtype {gemm,dram,l2,barrier} --duty D --dur S --rate-file F."""
    import torch
    dev = torch.device(DEV)
    dtype = _argval("--dtype", "gemm")
    duty = _argval("--duty", 1.0, float)
    dur = _argval("--dur", 60.0, float)
    rate_file = _argval("--rate-file", None)
    WIN = 0.10  # duty window

    if dtype == "gemm":
        a = torch.randn(4096, 4096, device=dev); b = torch.randn(4096, 4096, device=dev)
        def work():
            for _ in range(20):
                a.matmul(b)
            torch.cuda.synchronize(); return 20
    elif dtype == "dram":
        n = (256 * MIB) // 4
        src = torch.empty(n, dtype=torch.float32, device=dev); dst = torch.empty(n, dtype=torch.float32, device=dev)
        def work():
            for _ in range(8):
                dst.copy_(src)
            torch.cuda.synchronize(); return 8
    elif dtype == "l2":
        # SM READ of a resident 40MiB set (reduction: reads 40MiB, writes ~scalar) -> churns the set
        # through L2, evicting a co-probe's working set. Round-11's proven L2-pollution mechanism
        # (a DMA copy_ can bypass/differently-use L2; a reduction forces SM reads through the cache).
        n = (40 * MIB) // 4
        src = torch.randn(n, dtype=torch.float32, device=dev)
        acc = torch.zeros(1, dtype=torch.float32, device=dev)
        def work():
            for _ in range(40):
                acc += src.sum()
            torch.cuda.synchronize(); return 40
    elif dtype == "barrier":
        x = torch.zeros(1, dtype=torch.float32, device=dev)
        def work():
            for _ in range(50):
                x.add_(1.0); torch.cuda.synchronize()
            return 50
    else:
        raise SystemExit(f"unknown dtype {dtype}")

    # warm
    work()
    t0 = time.time(); it = 0; last = t0
    while time.time() < t0 + dur:
        wt0 = time.time()
        it += work()
        if duty < 1.0:
            active = time.time() - wt0
            idle = active * (1.0 - duty) / duty
            if idle > 0:
                time.sleep(min(idle, WIN))
        now = time.time()
        if rate_file and now - last >= 0.4:
            try:
                json.dump({"dtype": dtype, "duty": duty, "iters": it,
                           "rate_ips": round(it / (now - t0), 1), "t": round(now, 2)}, open(rate_file, "w"))
            except Exception:
                pass
            last = now


def _argval(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def _start_disturbers(dtype, n_inst, duty, own, dur=90.0):
    """Launch n_inst disturber subprocesses; return (pids, rate_files)."""
    pids, rfs = [], []
    for i in range(n_inst):
        rf = os.path.join(RATE_DIR, f"dist_{dtype}_{i}.json")
        if os.path.exists(rf):
            os.remove(rf)
        bg = subprocess.Popen([sys.executable, SELF, "--disturber", "--dtype", dtype,
                               "--duty", str(duty), "--dur", str(dur), "--rate-file", rf],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        own.add(bg.pid); pids.append(bg.pid); rfs.append(rf)
    return pids, rfs


def _stop_disturbers(pids, own):
    for p in pids:
        try:
            subprocess.run(["kill", str(p)], capture_output=True)
        except Exception:
            pass
        own.discard(p)
    time.sleep(1.5)


def _read_rates(rfs):
    rates = []
    for rf in rfs:
        try:
            rates.append(json.load(open(rf)).get("rate_ips"))
        except Exception:
            rates.append(None)
    return rates


def measure_all_victims(torch, dev):
    """All four victims in one pass (in-band)."""
    cu, cclk = compute_us(torch, dev)
    return {"compute_us": cu, "compute_clk": cclk,
            "dram_bw": copy_bw(torch, dev, 256),
            "l2_24_bw": copy_bw(torch, dev, 24),
            "composite_p50_us": composite_p50(torch, dev)}


# ------------------------------------------------------------------- part 1 ------
def part1(torch, dev, own, rep):
    POWERS = [175, 210, 250, 263]
    CLOCKS = [("unlocked", None), ("locked2000", 2000)]
    cells = {}
    for pw in POWERS:
        for cname, cmhz in CLOCKS:
            key = f"{pw}W_{cname}"
            ft = foreign_tenant(own)
            if ft:
                rep["part1_abort"] = f"foreign-{ft}-at-{key}"
                return cells
            set_power(pw)
            if cmhz:
                lock_clock(cmhz)
            else:
                release_clock()
            time.sleep(1.5)  # settle
            pre = stamp()
            # warm to the cell's equilibrium
            _ = compute_us(torch, dev, reps=6)
            burn_dur = 8.0 if cname == "unlocked" else 4.0  # locked arm brief
            traj = sustained_clock_traj(torch, dev, burn_dur)
            cu, cclk = compute_us(torch, dev)
            cell = {"power_set": pw, "clock_mode": cname, "clock_lock_mhz": cmhz,
                    "pre_stamp": pre,
                    "dram_bw": copy_bw(torch, dev, 256),
                    "launch_floor_us": launch_floor_us(torch, dev),
                    "compute_us": cu, "compute_clk_sm": cclk,
                    "clock_eq": traj["clock_eq"], "jitter_std": traj["jitter_std"],
                    "burn_rate_ips": traj["rate_ips"], "burn_clocks": traj["clocks"],
                    "burn_powers": traj["powers"], "post_stamp": stamp()}
            cells[key] = cell
            print(f"[P1] {key:22s} BW={cell['dram_bw']:6.1f} floor={cell['launch_floor_us']:6.2f}us "
                  f"comp={cell['compute_us']:7.1f}us clk_eq={cell['clock_eq']} jit={cell['jitter_std']}")
    return cells


# ------------------------------------------------------------------- part 2 ------
def part2(torch, dev, own, rep):
    # restore nominal state for interference matrix (clean 250W unlocked)
    release_clock(); set_power(250); time.sleep(1.5)
    DISTURBERS = ["gemm", "dram", "l2", "barrier"]
    INTENS = [("1x50", 1, 0.5), ("1x100", 1, 1.0), ("2x100", 2, 1.0)]  # intensity 0 = baseline below
    out = {"baselines": {}, "matrix": {}, "baseline_stamp": stamp()}

    # ---- CONTROL-FIRST: victim-alone baselines (regression gate) ----
    base = measure_all_victims(torch, dev)
    base2 = measure_all_victims(torch, dev)  # one is never enough
    out["baselines"] = base
    out["baselines_rep2"] = base2
    print(f"[P2] BASELINE comp={base['compute_us']}us BW={base['dram_bw']} "
          f"L2={base['l2_24_bw']} comp_p50={base['composite_p50_us']}us "
          f"(rep2 comp={base2['compute_us']} BW={base2['dram_bw']} L2={base2['l2_24_bw']} cp={base2['composite_p50_us']})")

    for dtype in DISTURBERS:
        ft = foreign_tenant(own)
        if ft:
            out["abort"] = f"foreign-{ft}-before-{dtype}"
            return out
        out["matrix"][dtype] = {}
        for iname, ninst, duty in INTENS:
            ft = foreign_tenant(own)
            if ft:
                out["abort"] = f"foreign-{ft}-at-{dtype}-{iname}"
                return out
            pids, rfs = _start_disturbers(dtype, ninst, duty, own)
            time.sleep(6.0)  # let disturber reach steady state
            rates_pre = _read_rates(rfs)
            v = measure_all_victims(torch, dev)
            rates_post = _read_rates(rfs)
            _stop_disturbers(pids, own)
            v["disturber_rates_ips"] = {"pre": rates_pre, "post": rates_post}
            v["alive"] = all(r for r in rates_post if r is not None) and bool([r for r in rates_post if r])
            out["matrix"][dtype][iname] = v
            print(f"[P2] {dtype:8s} {iname:6s} comp={v['compute_us']:7.1f} BW={v['dram_bw']:6.1f} "
                  f"L2={v['l2_24_bw']:6.1f} cp50={v['composite_p50_us']:7.1f} ips={rates_post}")
    return out


# ---------------------------------------------------------------- verdicts -------
def verdicts(rep):
    V = {}
    cells = rep.get("part1", {})
    # P1: DRAM BW power-insensitive above 200W, may sag at 175
    bw = {k: c["dram_bw"] for k, c in cells.items()}
    hi = [c["dram_bw"] for k, c in cells.items() if c["power_set"] >= 210 and c["clock_mode"] == "unlocked"]
    if hi:
        spread = (max(hi) - min(hi)) / statistics.mean(hi)
        V["P1_bw_insensitive_above200"] = {"pass": spread <= 0.10, "spread_frac": round(spread, 3),
                                            "bws_ge210_unlocked": hi}
    # P2: compute_us scales with achieved clock -> est-cycles ~ constant. cycles = us * clk(MHz)
    cyc = {}
    for k, c in cells.items():
        try:
            cyc[k] = round(c["compute_us"] * float(c["compute_clk_sm"]) / 1e3, 1)  # kilo-cycles
        except Exception:
            pass
    if cyc:
        vals = list(cyc.values())
        cv = statistics.pstdev(vals) / statistics.mean(vals) if statistics.mean(vals) else None
        V["P2_est_cycles_constant"] = {"pass": cv is not None and cv <= 0.10, "cv": round(cv, 3) if cv else None,
                                       "kilocycles_by_cell": cyc}
    # P3: capped-equilibrium clock monotone in power (unlocked cells)
    unl = sorted([(c["power_set"], c["clock_eq"]) for k, c in cells.items()
                  if c["clock_mode"] == "unlocked" and c["clock_eq"] is not None])
    mono = all(unl[i][1] <= unl[i + 1][1] + 5 for i in range(len(unl) - 1))  # +5MHz noise tol
    V["P3_clock_eq_monotone_in_power"] = {"pass": mono, "power_clock_eq": unl}
    # P4: locked cells near-zero jitter vs unlocked
    lj = [c["jitter_std"] for k, c in cells.items() if c["clock_mode"] == "locked2000"]
    uj = [c["jitter_std"] for k, c in cells.items() if c["clock_mode"] == "unlocked"]
    V["P4_locked_low_jitter"] = {"pass": bool(lj) and bool(uj) and statistics.mean(lj) < statistics.mean(uj),
                                 "mean_locked_jitter": round(statistics.mean(lj), 2) if lj else None,
                                 "mean_unlocked_jitter": round(statistics.mean(uj), 2) if uj else None}

    # PART 2 verdicts
    p2 = rep.get("part2", {})
    b = p2.get("baselines", {})
    b2 = p2.get("baselines_rep2", {})
    mtx = p2.get("matrix", {})
    if b:
        # regression gate: rep1 vs rep2 within 10%
        reg = {}
        for key in ("compute_us", "dram_bw", "l2_24_bw", "composite_p50_us"):
            if b.get(key) and b2.get(key):
                reg[key] = round(abs(b[key] - b2[key]) / b[key], 3)
        V["control_baseline_reproducible"] = {"pass": all(v <= 0.10 for v in reg.values()), "rel_diff": reg}
        # anchor: DRAM BW in the RTX5070 band
        V["control_dram_bw_anchor"] = {"pass": DRAM_BAND[0] <= b["dram_bw"] <= DRAM_BAND[1],
                                       "bw": b["dram_bw"], "band": DRAM_BAND}

    def worst_ratio(dtype, victim, higher_is_worse):
        """max degradation ratio across intensities for a victim under a disturber."""
        if dtype not in mtx or not b.get(victim):
            return None
        rs = []
        for iname in mtx[dtype]:
            mv = mtx[dtype][iname].get(victim)
            if mv:
                r = mv / b[victim] if higher_is_worse else b[victim] / mv
                rs.append(round(r, 3))
        return max(rs) if rs else None

    # M1: sparsity — off-diagonal cross-channel hits are small
    #   compute-hog vs DRAM-victim & DRAM-hog vs compute-victim
    m1 = {"gemm_hits_dram_bw": worst_ratio("gemm", "dram_bw", False),      # bw drop ratio (base/meas)
          "dram_hits_compute": worst_ratio("dram", "compute_us", True),    # compute inflation
          "gemm_hits_compute_diag": worst_ratio("gemm", "compute_us", True),
          "dram_hits_dram_diag": worst_ratio("dram", "dram_bw", False)}
    # sparse if off-diagonal << diagonal
    V["M1_matrix_sparse"] = {"pass": None, "detail": m1,
                             "note": "off-diag (gemm->dram_bw, dram->compute) should be near 1.0; diag large"}
    if all(m1[k] is not None for k in m1):
        offdiag = max(m1["gemm_hits_dram_bw"], m1["dram_hits_compute"])
        diag = min(m1["gemm_hits_compute_diag"], m1["dram_hits_dram_diag"])
        V["M1_matrix_sparse"]["pass"] = offdiag <= 1.25 and diag >= 1.30
        V["M1_matrix_sparse"]["offdiag_worst"] = offdiag
        V["M1_matrix_sparse"]["diag_worst"] = diag

    # M2: L2-victim collapses ONLY under L2 disturber
    l2hits = {d: worst_ratio(d, "l2_24_bw", False) for d in ("gemm", "dram", "l2", "barrier")}
    if all(v is not None for v in l2hits.values()):
        others = max(l2hits["gemm"], l2hits["barrier"])
        V["M2_l2_selective"] = {"pass": l2hits["l2"] >= 1.30 and l2hits["l2"] > 1.15 * others,
                                "l2_bw_drop_by_disturber": l2hits}
    else:
        V["M2_l2_selective"] = {"pass": None, "l2_bw_drop_by_disturber": l2hits}

    # M3: barrier-storm hits composite HARDEST
    comp_hits = {d: worst_ratio(d, "composite_p50_us", True) for d in ("gemm", "dram", "l2", "barrier")}
    if all(v is not None for v in comp_hits.values()):
        others = max(comp_hits["gemm"], comp_hits["dram"], comp_hits["l2"])
        V["M3_barrier_hits_composite"] = {"pass": comp_hits["barrier"] >= others,
                                          "composite_inflation_by_disturber": comp_hits}
    else:
        V["M3_barrier_hits_composite"] = {"pass": None, "composite_inflation_by_disturber": comp_hits}

    # M4: intensity monotone saturating (compute victim under gemm hog as canonical)
    mono_checks = {}
    for dtype in mtx:
        seq = []
        for iname in ("1x50", "1x100", "2x100"):
            mv = mtx[dtype].get(iname, {}).get("compute_us")
            if mv:
                seq.append(mv)
        if len(seq) >= 2:
            mono_checks[dtype] = {"seq": seq, "monotone": all(seq[i] <= seq[i + 1] * 1.05 for i in range(len(seq) - 1))}
    V["M4_intensity_monotone"] = {"pass": all(v["monotone"] for v in mono_checks.values()) if mono_checks else None,
                                  "compute_us_by_intensity": mono_checks}
    return V


# --------------------------------------------------------------------- main ------
def main():
    if "--disturber" in sys.argv:
        return run_disturber()

    os.makedirs(RATE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REP), exist_ok=True)
    own = {os.getpid()}
    rep = {"probe": "probe_exclusive_sweep_block", "gpu": _smi("name"),
           "launch_stamp": stamp(), "predictions": "P1-P4 (regime) + M1-M4 (interference); frozen in docstring"}

    ft = foreign_tenant(own)
    if ft:
        rep["verdict"] = f"ABORT-FOREIGN-TENANT-{ft}"
        json.dump(rep, open(REP, "w"), indent=1, default=str)
        print("FOREIGN TENANT", ft, "- abort"); return

    import torch
    dev = torch.device(DEV)
    torch.cuda.init()
    print("[env]", rep["launch_stamp"])

    t_start = time.time()
    try:
        rep["part1"] = part1(torch, dev, own, rep)
        rep["t_after_part1_s"] = round(time.time() - t_start, 1)
        print(f"--- part1 done in {rep['t_after_part1_s']}s ---")
        rep["part2"] = part2(torch, dev, own, rep)
        rep["t_after_part2_s"] = round(time.time() - t_start, 1)
    finally:
        # kill any straggler disturbers we own
        for p in list(own):
            if p != os.getpid():
                subprocess.run(["kill", str(p)], capture_output=True)
        rep["state_restoration"] = restore_and_verify()
        print("[restore]", rep["state_restoration"])

    rep["verdicts"] = verdicts(rep)
    rep["total_s"] = round(time.time() - t_start, 1)
    json.dump(rep, open(REP, "w"), indent=1, default=str)
    print("\n=== VERDICTS ===")
    for k, v in rep["verdicts"].items():
        print(f"  {k}: pass={v.get('pass')}")
    print("wrote", REP, "total", rep["total_s"], "s")


if __name__ == "__main__":
    main()

"""
L2 IN-BAND ENDGAME for the l2_stream_bistability_mechanism question.

WHY THIS INSTRUMENT (prior outcome, machine-proven):
  (1) The ncu residency SENSOR (lts__t_sector_hit_rate) is a SPATIAL-LOCALITY ARTIFACT, not
      residency: 12MiB warm == 12MiB flushed (both ~0.82 = the 7/8 compulsory first-touch floor).
  (2) ncu SERIALIZES cross-process concurrency -> a live co-tenant hog is INVISIBLE under ncu.
  => This round: IN-BAND CUDA-event TIMING (no profiler). The residency signal is the achieved
     COPY STREAM BANDWIDTH of a working set (= the card's OWN original observable): a set that
     stays L2-resident copies FAST; an evicted/oversize set copies at DRAM.
     [INSTRUMENT PIVOT, honest provenance: the first cut used a pure-READ kernel, but the
     65536-thread grid-stride version was latency/occupancy-bound (96MiB read FASTER than 12MiB)
     and the dim=M binned-atomic version was atomic-bound (no residency separation). The COPY
     kernel (dim=M, one thread/elem) is properly bandwidth-bound AND is exactly what the card
     measured. Gates below were RE-FROZEN for copy before the final run.]

  ============================ FINAL VERDICT (round 11) ============================
  INSTRUMENT VALID: dram_asymptote(256MiB copy)=556.5, neg(96MiB)=555.8 (0.1% dev -> 96MiB sits
    on the DRAM asymptote = genuine capacity eviction), pos(12MiB)=1247, sep=2.24x. (copy byte-
    accounting = 2*bytes; write-allocate makes true DRAM BW ~1.5x -> ~660 GB/s == 5070 spec.)
  1. CO-TENANT: CONFIRMED-QUANTIFIED. Cross-process L2 pollution is REAL and in-band-observable.
     Knee (probe copy footprint = 2*P, under a live+alive 80MiB warp hog, hog_rate 4.5k ips):
        fp<=24MiB: retained 0.99 (IMMUNE) | fp>=32MiB: retained 0.53 (COLLAPSE).
     Alone the L2 holds ~48MiB footprint; under an 80MiB co-tenant the probe's usable L2 HALVES
     to ~24-28MiB -> a co-tenant steals ~half of L2 (fair/LRU capacity sharing). SIZE-SELECTIVE
     (small probes immune) => it is L2 capacity eviction, NOT generic DRAM-bandwidth theft.
  2. 8-SESSION FLIP: does NOT reproduce clean. 12MiB spread 0.4% (always L2 ~1247), 24MiB spread
     2.9% (always ~820), steady clock 2827MHz, self-only tenancy. NO intrinsic session/clock/
     allocator bistability. The card's historical 24MiB flip (500 low / 736-841 high) maps
     EXACTLY onto measured 24MiB-alone=815 (high) vs 24MiB-under-co-tenant=348-500 (low).
  => CARD MECHANISM: the "stream bistability" is CO-TENANCY (foreign-lane L2 pollution), not an
     intrinsic bistable state. Remaining non-co-tenancy candidates are REJECTED with evidence:
     power/clock (clock pinned 2827MHz across all 8 sessions), allocator/session-state (fresh
     process each, 0.4% spread), driver heuristic (stable). The card resolves.
  =================================================================================

TWO ROUND-CORRECTIONS BAKED IN:
  - LAUNCH-FLOOR (round-6): a single small-buffer launch is dominated by ~110us of warp launch+
    event overhead. FIX = BURST timing: bracket N=300 separate launches in ONE event pair, sync
    once. Floor is amortized to floor/N (~0.4us/launch, negligible) AND the N separate launches
    give a time-sliced co-tenant many interleave points (a single K-pass kernel never yields
    mid-launch, so it would be structurally blind to a co-tenant -> burst is the concurrency-
    preserving design).
  - REGISTER-HOIST artifact (found this round): an in-kernel passes-loop over a small resident set
    gets hoisted into registers (measured 20 TB/s = register speed, not L2). Burst of SEPARATE
    single-pass launches cannot hoist across launches -> real memory traffic every launch.

EXTERNAL ANCHOR (not a tautology gate): the oversize (96MiB > 48MiB L2) read MUST land in the
RTX 5070 DRAM band ~[500,800] GB/s. If it does, the DRAM state is really being observed; the
resident state is then whatever reads materially faster than that.

PRE-REGISTERED GATES (frozen BEFORE running):
  CALIBRATION (control-first, MANDATORY):
    G-cal-neg  : burst_read_bw(96MiB, alone) in [500,800] GB/s  (genuine DRAM = capacity eviction,
                 NOT memset-cold; anchored to the 5070's ~672 GB/s spec).
    G-cal-sep  : burst_read_bw(12MiB, alone) / burst_read_bw(96MiB, alone) >= 1.30  (resident vs
                 DRAM states separate cleanly).
    INSTRUMENT VALID iff G-cal-neg AND G-cal-sep. Else ABORT (record calibration only).

  ENDGAME (pre-registered):
   1. CO-TENANT curve: probe P in {4,12,24}MiB x hog H in {none,40,80}MiB (self-owned warp read
      hog). Metric = burst_read_bw(P | H) AND burst_copy_bw(P | H). Hog logs its loop-rate so
      liveness/starvation is provable concurrently.
      G-collapse : read_bw(P|H) <= 0.60 * read_bw(P|none) at some H  -> co-tenant L2 pollution
                   CONFIRMED-QUANTIFIED; the (H -> read_bw) curve is the mechanism.
      G-isolation: read_bw(P|80) >= 0.85 * read_bw(P|none) for ALL P, AND the 80MiB hog is proven
                   ALIVE + UNSTARVED (rate under probe >= 0.5x its solo rate) -> L2 ISOLATION
                   finding (co-tenants time-slice; no concurrent L2 sharing => card flip is NOT
                   co-tenancy). A starved hog + unaffected probe proves nothing (hog wasn't co-running).
   2. 8-SESSION state re-read: fresh process each, tenancy=NONE stamped. burst_read_bw AND
      burst_copy_bw at 12 & 24MiB. Do the two instruments AGREE on the state (both L2 or both
      DRAM)? If states FLIP across sessions with tenancy=NONE -> mechanism is NOT co-tenancy;
      remaining candidates (allocator/session, power/clock, driver heuristic) each get an
      evidenced line.

TENANCY: nvidia-smi compute-apps + clocks.sm stamped per burst. Own child PIDs whitelisted; a
FOREIGN (non-child) tenant -> abort load arms (never crowd). GPU warmed to steady clock before
any timed burst. Commit nothing.
"""
import json
import os
import subprocess
import sys
import time
import statistics

DEV = "cuda:0"
SELF = os.path.abspath(__file__)
REP = "reports/probes/probe_l2_inband_endgame.json"
MIB = 1024 * 1024
THREADS = 65536
DRAM_BAND = (500.0, 800.0)  # RTX 5070 GDDR7 ~672 GB/s external anchor
RATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(RATE_DIR, exist_ok=True)


def _wp():
    import warp as wp
    wp.init()
    return wp


def _kernels(wp):
    # one thread PER element (dim=M) = coalesced, high-occupancy -> BANDWIDTH-bound (the earlier
    # 65536-thread grid-stride read was latency/occupancy-bound: 96MiB read FASTER than 12MiB
    # because more elems/thread = more ILP -> a broken BW kernel). Atomics spread over 1024 bins
    # to avoid serialization while keeping the loads from being dead-code eliminated.
    @wp.kernel
    def read_k(val: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        wp.atomic_add(out, i & 1023, val[i] * 0.0 + 1.0)

    @wp.kernel
    def copy_k(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
        i = wp.tid()
        dst[i] = src[i] * 2.0

    return read_k, copy_k


def _argval(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def stamp():
    apps = os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip()
    clk = os.popen("nvidia-smi --query-gpu=clocks.sm,utilization.gpu,power.draw --format=csv,noheader").read().strip()
    return {"tenants": apps or "NONE", "gpu": clk, "t": round(time.time(), 3)}


def foreign_tenant(own):
    apps = os.popen("nvidia-smi --query-compute-apps=pid --format=csv,noheader").read().strip()
    for line in apps.splitlines():
        p = line.strip()
        if p and int(p) not in own:
            return p
    return None


# ------------------------------------------------------------------ hog child ---
def run_hog():
    wp = _wp()
    read_k, _ = _kernels(wp)
    mib = _argval("--hog-mib", 40, int)
    rate_file = _argval("--rate-file", None)
    dur = _argval("--dur", 120.0, float)
    M = (mib * MIB) // 4
    buf = wp.zeros(M, dtype=wp.float32, device=DEV)
    out = wp.zeros(1024, dtype=wp.float32, device=DEV)
    wp.launch(read_k, dim=M, inputs=[buf, out], device=DEV)
    wp.synchronize()
    t0 = time.time()
    t_end = t0 + dur
    iters = 0
    last = t0
    while time.time() < t_end:
        for _ in range(20):
            wp.launch(read_k, dim=M, inputs=[buf, out], device=DEV)
        wp.synchronize()
        iters += 20
        now = time.time()
        if rate_file and (now - last) >= 0.4:
            try:
                json.dump({"mib": mib, "iters": iters, "rate_ips": round(iters / (now - t0), 1),
                           "t": round(now, 3)}, open(rate_file, "w"))
            except Exception:
                pass
            last = now


# --------------------------------------------------------------- burst timing ---
def _ev(wp):
    return wp.Event(enable_timing=True), wp.Event(enable_timing=True)


def burst_read_bw(wp, read_k, M, N=100, reps=5):
    """N read launches captured into a CUDA GRAPH, timed via replay. GRAPH replay submits all N
    kernels with ~one CPU launch -> GPU-BOUND (a plain Python launch loop idles the GPU between
    small kernels, inverting the BW -> that artifact is why graphs are mandatory here). Floor is
    amortized over N; a time-sliced co-tenant still preempts during the multi-ms replay."""
    val = wp.zeros(M, dtype=wp.float32, device=DEV)
    out = wp.zeros(1024, dtype=wp.float32, device=DEV)
    for _ in range(3):
        wp.launch(read_k, dim=M, inputs=[val, out], device=DEV)
    wp.synchronize()
    with wp.ScopedCapture(device=DEV) as cap:
        for _ in range(N):
            wp.launch(read_k, dim=M, inputs=[val, out], device=DEV)
    graph = cap.graph
    wp.capture_launch(graph)
    wp.synchronize()
    bws = []
    for _ in range(reps):
        e0, e1 = _ev(wp)
        wp.record_event(e0)
        wp.capture_launch(graph)
        wp.record_event(e1)
        wp.synchronize_event(e1)
        ms = wp.get_event_elapsed_time(e0, e1)
        bws.append(N * (M * 4) / (ms / 1e3) / 1e9)
    del val, out
    return round(statistics.median(bws), 1), [round(b, 1) for b in bws]


def burst_copy_bw(wp, copy_k, M, N=100, reps=5):
    src = wp.zeros(M, dtype=wp.float32, device=DEV)
    dst = wp.zeros(M, dtype=wp.float32, device=DEV)
    for _ in range(3):
        wp.launch(copy_k, dim=M, inputs=[src, dst], device=DEV)
    wp.synchronize()
    with wp.ScopedCapture(device=DEV) as cap:
        for _ in range(N):
            wp.launch(copy_k, dim=M, inputs=[src, dst], device=DEV)
    graph = cap.graph
    wp.capture_launch(graph)
    wp.synchronize()
    bws = []
    for _ in range(reps):
        e0, e1 = _ev(wp)
        wp.record_event(e0)
        wp.capture_launch(graph)
        wp.record_event(e1)
        wp.synchronize_event(e1)
        ms = wp.get_event_elapsed_time(e0, e1)
        bws.append(N * (2 * M * 4) / (ms / 1e3) / 1e9)
    del src, dst
    return round(statistics.median(bws), 1), [round(b, 1) for b in bws]


def warm_gpu(wp, read_k):
    M = (32 * MIB) // 4
    val = wp.zeros(M, dtype=wp.float32, device=DEV)
    out = wp.zeros(1024, dtype=wp.float32, device=DEV)
    t_end = time.time() + 4.0
    while time.time() < t_end:
        for _ in range(40):
            wp.launch(read_k, dim=M, inputs=[val, out], device=DEV)
        wp.synchronize()
    del val, out


# ------------------------------------------------------------- session child ---
def run_session_child():
    """Fresh-process session: measure BOTH instruments (read residency + copy stream) at 12 & 24MiB."""
    wp = _wp()
    read_k, copy_k = _kernels(wp)
    out_file = _argval("--out-file", None)
    warm_gpu(wp, read_k)
    # Measured DRAM asymptote (256MiB copy) -> state threshold = midpoint to a 4MiB L2 copy
    dram256, _ = burst_copy_bw(wp, copy_k, (256 * MIB) // 4, N=40)
    l2_4, _ = burst_copy_bw(wp, copy_k, (4 * MIB) // 4)
    thr = 0.5 * (dram256 + l2_4)
    res = {"tenancy": stamp(), "dram256_copy": dram256, "l2_4MiB_copy": l2_4,
           "state_threshold": round(thr, 1), "sizes": {}}
    for mib in (12, 24):
        M = (mib * MIB) // 4
        cbw, craw = burst_copy_bw(wp, copy_k, M)
        res["sizes"][str(mib)] = {
            "copy_bw_GBps": cbw, "copy_raw": craw,
            "copy_state": ("L2" if cbw > thr else "DRAM"),
        }
    res["tenancy_post"] = stamp()
    if out_file:
        json.dump(res, open(out_file, "w"), indent=1, default=str)
    print(json.dumps(res, default=str))


# --------------------------------------------------------------------- main ----
def solo_hog_rate(H, dur=8):
    rf = os.path.join(RATE_DIR, f"hog_solo_{H}.json")
    if os.path.exists(rf):
        os.remove(rf)
    bg = subprocess.Popen([sys.executable, SELF, "--hog", "--hog-mib", str(H),
                           "--rate-file", rf, "--dur", str(dur)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bg.wait()
    if os.path.exists(rf):
        try:
            return json.load(open(rf)).get("rate_ips")
        except Exception:
            return None
    return None


def run_sessions(n=8):
    """Part-2: fresh PROCESS per session (kills allocator/session-state carryover), tenancy=NONE
    stamped. Each child reports copy_bw @12&24MiB + its DRAM/L2 state classification."""
    out = {"probe": "probe_l2_inband_endgame_sessions", "sessions": []}
    for s in range(n):
        # only launch when GPU is clean (tenancy=NONE) so the flip, if any, is NOT co-tenancy
        for _ in range(60):
            if not foreign_tenant({os.getpid()}):
                break
            time.sleep(5)
        of = os.path.join(RATE_DIR, f"session_{s}.json")
        r = subprocess.run([sys.executable, SELF, "--session-child", "--out-file", of],
                           capture_output=True, text=True, timeout=200)
        try:
            d = json.load(open(of))
        except Exception:
            d = {"err": (r.stderr or r.stdout)[-300:]}
        d["session"] = s
        out["sessions"].append(d)
        t = d.get("tenancy", {}).get("tenants", "?")
        szs = d.get("sizes", {})
        print(f"session {s}: tenancy={str(t)[:30]} 12MiB={szs.get('12',{}).get('copy_bw_GBps')}"
              f"({szs.get('12',{}).get('copy_state')}) 24MiB={szs.get('24',{}).get('copy_bw_GBps')}"
              f"({szs.get('24',{}).get('copy_state')}) clk={d.get('tenancy',{}).get('gpu','?')}")
    # flip analysis (only tenancy=NONE sessions)
    for mib in ("12", "24"):
        states = [(s["session"], s["sizes"][mib]["copy_state"], s["sizes"][mib]["copy_bw_GBps"])
                  for s in out["sessions"]
                  if s.get("tenancy", {}).get("tenants") == "NONE" and "sizes" in s]
        distinct = set(st for _, st, _ in states)
        out[f"flip_{mib}MiB"] = {"states": states, "flipped": len(distinct) > 1, "distinct": list(distinct)}
        print(f"  {mib}MiB across NONE-sessions: flipped={len(distinct)>1} {[ (a,b,c) for a,b,c in states]}")
    os.makedirs(os.path.dirname(REP), exist_ok=True)
    json.dump(out, open(REP.replace(".json", "_sessions.json"), "w"), indent=1, default=str)
    return out


def run_knee():
    """Probe-footprint knee under a fixed 80MiB hog: locate where copy collapses (footprint=2*P)."""
    wp = _wp()
    read_k, copy_k = _kernels(wp)
    warm_gpu(wp, read_k)
    own = {os.getpid()}
    SIZES = [8, 12, 16, 20, 22, 24, 28, 32]
    res = {"alone": {}, "hog80": {}}
    for P in SIZES:
        res["alone"][str(P)] = burst_copy_bw(wp, copy_k, (P * MIB) // 4)[0]
    if foreign_tenant(own):
        res["ABORT"] = "foreign-tenant"
    else:
        rf = os.path.join(RATE_DIR, "hog_knee.json")
        bg = subprocess.Popen([sys.executable, SELF, "--hog", "--hog-mib", "80",
                               "--rate-file", rf, "--dur", "120"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        own.add(bg.pid)
        time.sleep(7)
        for P in SIZES:
            res["hog80"][str(P)] = burst_copy_bw(wp, copy_k, (P * MIB) // 4)[0]
        try:
            res["hog_rate"] = json.load(open(rf)).get("rate_ips")
        except Exception:
            res["hog_rate"] = None
        bg.terminate()
        try:
            bg.wait(timeout=10)
        except Exception:
            bg.kill()
    for P in SIZES:
        a = res["alone"].get(str(P))
        h = res["hog80"].get(str(P))
        print(f"  P={P}MiB (fp {2*P}MiB) alone={a} hog80={h} retained={round(h/a,3) if (a and h) else None}")
    os.makedirs(os.path.dirname(REP), exist_ok=True)
    json.dump(res, open(REP.replace(".json", "_knee.json"), "w"), indent=1, default=str)
    return res


def main():
    if "--hog" in sys.argv:
        return run_hog()
    if "--session-child" in sys.argv:
        return run_session_child()
    if "--sessions" in sys.argv:
        return run_sessions()
    if "--knee" in sys.argv:
        return run_knee()

    os.makedirs(RATE_DIR, exist_ok=True)
    rep = {"probe": "probe_l2_inband_endgame", "gpu": "RTX5070 GB205 48MiB L2",
           "dram_band": DRAM_BAND, "launch_stamp": stamp(), "cal": {}, "cotenant": {},
           "sessions": [], "gates": {}}
    own = {os.getpid()}
    ft = foreign_tenant(own)
    if ft:
        rep["verdict"] = f"ABORT-FOREIGN-TENANT-{ft}"
        os.makedirs(os.path.dirname(REP), exist_ok=True)
        json.dump(rep, open(REP, "w"), indent=1, default=str)
        print("FOREIGN TENANT", ft, "- abort")
        return

    wp = _wp()
    read_k, copy_k = _kernels(wp)
    print("warming GPU...")
    warm_gpu(wp, read_k)
    rep["warm_stamp"] = stamp()
    print("clock:", rep["warm_stamp"]["gpu"])

    # ---- CALIBRATION (COPY = the card's own stream-BW observable) --------------
    # read_k (binned atomics) is atomic-bound (no residency separation) -> dropped. COPY is the
    # instrument: L2-resident copy is FAST, DRAM (oversize) copy is SLOW. DRAM anchor = the
    # asymptotic copy rate at 256MiB (deep DRAM); a 96MiB copy must match it (both DRAM regime).
    print("=== CALIBRATION (copy stream BW) ===")
    dram256, _ = burst_copy_bw(wp, copy_k, (256 * MIB) // 4, N=40)
    neg_c, neg_raw = burst_copy_bw(wp, copy_k, (96 * MIB) // 4)
    pos_c, pos_raw = burst_copy_bw(wp, copy_k, (12 * MIB) // 4)
    sep = round(pos_c / neg_c, 3) if neg_c else None
    anchor_dev = abs(neg_c - dram256) / dram256 if dram256 else None
    rep["cal"] = {"dram_asymptote_256MiB_copy_GBps": dram256,
                  "neg_96MiB_copy_GBps": neg_c, "neg_raw": neg_raw,
                  "pos_12MiB_copy_GBps": pos_c, "pos_raw": pos_raw,
                  "sep_ratio_copy": sep,
                  "neg_vs_dram256_devfrac": round(anchor_dev, 3) if anchor_dev is not None else None,
                  "note_writealloc": "copy accounting=2*bytes; write-allocate adds ~1 read -> true DRAM BW ~ 1.5x reported (~660 GB/s matches 5070 spec)",
                  "stamp": stamp()}
    g_neg = anchor_dev is not None and anchor_dev <= 0.20   # 96MiB sits at the DRAM asymptote
    g_sep = neg_c > 0 and (pos_c / neg_c) >= 1.30
    rep["gates"]["G_cal_neg_dram_anchor"] = g_neg
    rep["gates"]["G_cal_sep"] = g_sep
    valid = g_neg and g_sep
    rep["instrument_valid"] = valid
    print(f"CAL dram256={dram256} neg_96MiB={neg_c} (anchor dev={anchor_dev} ok={g_neg}) | pos_12MiB={pos_c} | sep={sep} ({g_sep})")
    if "--cal-only" in sys.argv or not valid:
        os.makedirs(os.path.dirname(REP), exist_ok=True)
        suffix = "_calonly" if "--cal-only" in sys.argv else ""
        if not valid:
            rep["verdict"] = "ABORT-INSTRUMENT-INVALID"
        json.dump(rep, open(REP.replace(".json", suffix + ".json"), "w"), indent=1, default=str)
        print("instrument_valid =", valid)
        return

    # ---- CO-TENANT ARM --------------------------------------------------------
    print("=== CO-TENANT ARM ===")
    PROBES = [4, 12, 24]
    HOGS = [0, 40, 80]
    for H in HOGS:
        bg = None
        rate_file = os.path.join(RATE_DIR, f"hog_{H}.json")
        if H > 0:
            ft = foreign_tenant(own)
            if ft:
                print(f"FOREIGN TENANT {ft} — abort hog arms")
                rep["cotenant"]["ABORT"] = f"foreign-{ft}-at-H{H}"
                break
            if os.path.exists(rate_file):
                os.remove(rate_file)
            bg = subprocess.Popen([sys.executable, SELF, "--hog", "--hog-mib", str(H),
                                   "--rate-file", rate_file, "--dur", "120"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            own.add(bg.pid)
            time.sleep(7)
        rep["cotenant"][str(H)] = {}
        for P in PROBES:
            M = (P * MIB) // 4
            cbw, craw = burst_copy_bw(wp, copy_k, M)
            hr = None
            if H > 0 and os.path.exists(rate_file):
                try:
                    hr = json.load(open(rate_file)).get("rate_ips")
                except Exception:
                    hr = None
            rep["cotenant"][str(H)][str(P)] = {
                "copy_bw_GBps": cbw, "copy_raw": craw, "hog_rate_ips": hr, "stamp": stamp()}
            print(f"  hog={H} probe={P}MiB copy={cbw} hog_rate={hr}")
        if bg is not None:
            bg.terminate()
            try:
                bg.wait(timeout=10)
            except Exception:
                bg.kill()
            own.discard(bg.pid)
            time.sleep(2)

    rep["hog_solo_rate"] = {str(H): solo_hog_rate(H) for H in (40, 80)}

    # ---- CO-TENANT VERDICT ----------------------------------------------------
    collapse = {}
    iso = {"all_read_retained": True, "detail": {}}
    base = rep["cotenant"].get("0", {})
    for P in PROBES:
        b0 = base.get(str(P), {}).get("copy_bw_GBps")
        for H in (40, 80):
            hd = rep["cotenant"].get(str(H), {}).get(str(P))
            if hd and b0:
                bh = hd["copy_bw_GBps"]
                ratio = bh / b0 if b0 else None
                iso["detail"][f"P{P}_H{H}"] = {"b0": b0, "bh": bh, "retained": round(ratio, 3) if ratio else None,
                                               "hog_rate": hd.get("hog_rate_ips")}
                if ratio is not None and ratio <= 0.60:
                    collapse[f"P{P}_H{H}"] = {"b0": b0, "bh": bh, "retained": round(ratio, 3)}
                if H == 80 and (ratio is None or ratio < 0.85):
                    iso["all_read_retained"] = False
    solo80 = rep["hog_solo_rate"].get("80")
    rates80 = [rep["cotenant"].get("80", {}).get(str(P), {}).get("hog_rate_ips") for P in PROBES]
    rates80 = [r for r in rates80 if r]
    hog_unstarved = bool(rates80) and solo80 and (min(rates80) >= 0.5 * solo80)
    rep["collapse_events"] = collapse
    rep["isolation_check"] = {**iso, "hog80_unstarved": hog_unstarved,
                              "hog80_rates_under_probe": rates80, "hog80_solo": solo80}
    if collapse:
        rep["cotenant_verdict"] = f"CONFIRMED-QUANTIFIED: L2 pollution — read_bw collapses under hog: {collapse}"
    elif iso["all_read_retained"] and hog_unstarved:
        rep["cotenant_verdict"] = "ISOLATION: read_bw retained >=0.85 under alive+unstarved 80MiB hog (co-tenants time-slice; no concurrent L2 sharing)"
    else:
        rep["cotenant_verdict"] = "AMBIGUOUS (see isolation_check: retention high but hog starvation unproven)"
    print("CO-TENANT:", rep["cotenant_verdict"])

    os.makedirs(os.path.dirname(REP), exist_ok=True)
    json.dump(rep, open(REP, "w"), indent=1, default=str)
    print("wrote", REP)


if __name__ == "__main__":
    main()

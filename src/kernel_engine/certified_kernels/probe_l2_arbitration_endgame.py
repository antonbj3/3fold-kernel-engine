"""
!!! OUTCOME — READ FIRST, this matrix design is INVALID as-shipped:
  The pre-registered co-residency matrix below CANNOT measure what it claims. Two independent
  breaks, both machine-proven (see the JSON this module writes under artifacts/):
   (1) The residency SENSOR is a SPATIAL-LOCALITY ARTIFACT, not residency. 12MiB warm reads
       hit=0.84; the SAME 12MiB after a 200MiB flush (guaranteed evicted from 48MiB L2) reads
       hit=0.82. Warm == evicted. The ~0.82 is the compulsory first-touch-per-sector miss floor
       of a contiguous stream (~7/8), independent of pre-residency. The earlier round's '0.80 clean / 0.80
       under hog' were BOTH this floor -> the '51MiB-in-48MiB' paradox dissolves.
   (2) ncu profiling SERIALIZES cross-process concurrency. A concurrent 80MiB warp hog (alive
       tenant) leaves a fitting reuse kernel's hit_rate UNCHANGED (0.34->0.34), while the same
       instrument drops 0.32->0.01 for in-context oversize. The co-tenant is invisible under ncu.
  VERDICT: co-tenant L2 pollution is NOT confirmable via ncu lts__t_sector_hit_rate. The card's
  cross-session flip needs a NON-ncu concurrency-preserving instrument. The valid pieces here are
  the CONTROLS (warm-vs-FLUSHED known-negative) and the WITHIN-kernel two-pass reuse capacity
  probe. The matrix arms are retained for provenance only.
!!!

L2-ARBITRATION ENDGAME for l2_stream_bistability_mechanism.

STATE inherited: the residency SENSOR works (one-pass re-read; hit_rate = residency fraction
directly; 0.80-0.85 clean, validated earlier). The earlier hog was UNCALIBRATED (torch .sum()
streams, never verified L2-occupying). THIS round: a WARP read+write hog that provably keeps
its N-MiB working set resident (its own sensor reads it warm), swept in size, with SYMMETRIC
both-sides observation (ncu profiles one process, so we alternate the profiled party while the
other party runs as unprofiled GPU-background).

GPU: RTX 5070 (Blackwell GB205), 48 MiB L2 (matches card's 48MiB assumption). Verified via
nvidia-smi at launch: single tenant = this experiment.

PRE-REGISTERED (frozen before running):
 CONTROLS (control-first, MANDATORY — this card ate 2 uncalibrated-instrument classes):
  C-hog+  : hog-probe warm, ALONE   -> PREDICT hit HIGH (>=0.6). Instrument positive.
  C-hog-  : hog-probe COLD, ALONE   -> PREDICT hit LOW  (<=0.3). Instrument negative.
  C-str+  : stream-probe, ALONE     -> PREDICT hit ~0.80 (reproduce the earlier run). Positive.
  C-str-  : stream-probe COLD, ALONE-> PREDICT hit LOW  (<=0.3). Negative.
  Both sensors must pass +/- separation >=0.4 or the matrix is uninterpretable (ABORT).

 CO-RESIDENCY MATRIX, hog size N in {16,32,40,56,80} MiB:
  stream_under_hog[N] : hog-bg(N) background, profile stream sensor  -> stream residency.
  hog_under_stream[N] : stream-bg background,  profile hog sensor(N)  -> hog residency.
  baselines: stream_clean, hog_clean[N] (no background).
  PREDICTION (finite 48MiB LRU): sum of resident bytes <= 48MiB. As N grows past ~ (48 - 24)=24MiB
   SOMEONE must lose. Replacement law = WHO loses and HOW:
     - stream collapses as N grows, hog stays high      -> recency/hog-favored (round-robin/LRU win by hog).
     - hog capped, stream stays high                    -> stream-favored / hog can't exceed a share.
     - both degrade proportionally                      -> proportional/random replacement.
     - NEITHER loses even at 80MiB (>48)                -> partitioned / non-LRU / set-limited L2 (isolation IS the finding).

 CEILING ANATOMY (why clean stream reads 0.80 not 1.0 — where do ~18% of lines go):
  (a) full-copy vs src-only-reread (removes 12MiB dst write-allocate) -> does ceiling rise?
  (b) stream 4MiB vs 12MiB (capacity-unrelated confounds shrink)      -> does ceiling rise?
  (c) sensor-immediate vs sensor-after-0.3s-delay                     -> time-decay component?

GATES (frozen):
  G-ctrl: both sensors pass +/- separation >= 0.4.
  G-collapse: stream_under_hog drops >= 0.3 absolute vs stream_clean at some N -> co-tenant
    occupancy MECHANISM CONFIRMED-QUANTIFIED; report the collapse threshold N*.
  G-isolation: stream_under_hog stays within 0.15 of stream_clean for ALL N incl 80MiB, while
    hog_clean[80] itself is >=0.6 (hog really is occupying) -> L2 ISOLATION finding (non-LRU/partitioned).
  Exactly one of G-collapse / G-isolation should fire; if neither -> ambiguous, report raw + orient.

TENANCY: stamp nvidia-smi compute-apps + clocks.sm per burst. Own child PIDs whitelisted; a
foreign (non-child) tenant -> abort background arms (never crowd). --cache-control none (observe
natural cache state, no flush).
"""
import json
import os
import subprocess
import sys
import time
import csv as _csv
import io

NCU = "/usr/local/cuda-12.8/bin/ncu"
NCU_PATH = os.environ.get("PATH", "")
SELF = os.path.abspath(__file__)
REP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts",
                   "probe_l2_arbitration_endgame.json")
METRIC = "lts__t_sector_hit_rate"
HOG_SIZES = [16, 32, 40, 56, 80]


# ---------------------------------------------------------------- children ---
def _warp():
    import warp as wp
    wp.init()
    return wp


def _kernels(wp):
    @wp.kernel
    def copy_k(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
        i = wp.tid()
        dst[i] = src[i]

    # read+write same buffer: keeps working set resident, NO extra memory footprint
    @wp.kernel
    def touch_k(buf: wp.array(dtype=wp.float32)):
        i = wp.tid()
        buf[i] = buf[i] * 1.0 + 0.0

    # pure single READ pass -> hit_rate = residency of `val`. VERBATIM round-9 proven sensor:
    # atomic to a SINGLE out location (size-1 array). Spreading the atomic over 256 lines
    # broke discrimination (pinned 0.50 = out-read/val-read 1:1 artifact); the single hot
    # sector is register/L1-coalesced so val reads dominate the L2 hit_rate -> discriminates
    # 0.80 warm / 0.00 streaming (round-9 calibration).
    @wp.kernel
    def streamsensor_k(val: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        wp.atomic_add(out, 0, val[i] * 0.0 + 1.0)

    @wp.kernel
    def hogsensor_k(val: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        wp.atomic_add(out, 0, val[i] * 0.0 + 1.0)

    return copy_k, touch_k, streamsensor_k, hogsensor_k


def _argval(flag, default=None, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def run_stream_probe():
    wp = _warp()
    copy_k, touch_k, streamsensor_k, hogsensor_k = _kernels(wp)
    mib = _argval("--stream-mib", 12, int)
    cold = "--cold" in sys.argv
    src_only = "--src-only" in sys.argv
    no_delay = "--no-delay" in sys.argv
    N = (mib * 1024 * 1024) // 4
    src = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    dst = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    out = wp.zeros(1, dtype=wp.float32, device="cuda:0")
    if not cold:
        for _ in range(10):
            if src_only:
                wp.launch(touch_k, dim=N, inputs=[src])   # read+write src, no dst allocate-traffic
            else:
                wp.launch(copy_k, dim=N, inputs=[src, dst])
        wp.synchronize()
        if not no_delay:
            time.sleep(0.3)
    wp.launch(streamsensor_k, dim=N, inputs=[src, out])
    wp.synchronize()


def run_hog_probe():
    wp = _warp()
    copy_k, touch_k, streamsensor_k, hogsensor_k = _kernels(wp)
    mib = _argval("--hog-mib", 40, int)
    cold = "--cold" in sys.argv
    N = (mib * 1024 * 1024) // 4
    buf = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    out = wp.zeros(1, dtype=wp.float32, device="cuda:0")
    if not cold:
        for _ in range(200):
            wp.launch(touch_k, dim=N, inputs=[buf])
        wp.synchronize()
        time.sleep(0.3)
    wp.launch(hogsensor_k, dim=N, inputs=[buf, out])
    wp.synchronize()


def run_hog_bg():
    wp = _warp()
    copy_k, touch_k, streamsensor_k, hogsensor_k = _kernels(wp)
    mib = _argval("--hog-mib", 40, int)
    N = (mib * 1024 * 1024) // 4
    buf = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    t_end = time.time() + 180
    while time.time() < t_end:
        for _ in range(50):
            wp.launch(touch_k, dim=N, inputs=[buf])
        wp.synchronize()


def run_stream_bg():
    wp = _warp()
    copy_k, touch_k, streamsensor_k, hogsensor_k = _kernels(wp)
    mib = _argval("--stream-mib", 12, int)
    N = (mib * 1024 * 1024) // 4
    src = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    dst = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    t_end = time.time() + 180
    while time.time() < t_end:
        for _ in range(50):
            wp.launch(copy_k, dim=N, inputs=[src, dst])
        wp.synchronize()


# ------------------------------------------------------------------ parent ---
def stamp():
    apps = os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip()
    clk = os.popen("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader").read().strip()
    return {"tenants": apps or "NONE", "clock": clk, "t": time.time()}


def foreign_tenant(own_pids):
    apps = os.popen("nvidia-smi --query-compute-apps=pid --format=csv,noheader").read().strip()
    for line in apps.splitlines():
        p = line.strip()
        if p and int(p) not in own_pids and p != str(os.getpid()):
            return p
    return None


def ncu_hit(child_args, regex):
    cmd = ["sudo", "-n", "env", f"PATH={NCU_PATH}", NCU, "--launch-count", "1",
           "--cache-control", "none", "--kernel-name", f"regex:{regex}",
           "--metrics", METRIC, "--csv", sys.executable, SELF] + child_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    vals = []
    for row in _csv.reader(io.StringIO(r.stdout + "\n" + r.stderr)):
        if len(row) > 3 and METRIC in ",".join(row):
            for cell in row:
                try:
                    v = float(cell.replace(",", ""))
                    if 0 <= v <= 100:
                        vals.append(v)
                except ValueError:
                    pass
    if vals:
        v = vals[-1]
        return v / 100.0 if v > 1.0 else v
    return {"err": (r.stderr or r.stdout)[-400:]}


def measure(child_args, regex, reps=2):
    """profile `reps` times, return list of hit_rates + tenancy stamps."""
    out = []
    for _ in range(reps):
        s0 = stamp()
        hr = ncu_hit(child_args, regex)
        out.append({"hit": hr, "stamp": s0})
    return out


def bg_launch(child_args):
    p = subprocess.Popen([sys.executable, SELF] + child_args,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p


def hits(lst):
    return [d["hit"] for d in lst if isinstance(d["hit"], float)]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    # dispatch children
    if "--stream-probe" in sys.argv:
        return run_stream_probe()
    if "--hog-probe" in sys.argv:
        return run_hog_probe()
    if "--hog-bg" in sys.argv:
        return run_hog_bg()
    if "--stream-bg" in sys.argv:
        return run_stream_bg()

    rep = {"probe": "probe_l2_arbitration_endgame", "gpu": "RTX5070 GB205 48MiB L2",
           "launch_stamp": stamp(), "controls": {}, "matrix": {}, "ceiling": {}}

    # ---- CONTROLS (control-first) --------------------------------------------
    print("=== CONTROLS ===")
    rep["controls"]["hog_pos_warm_40"] = measure(["--hog-probe", "--hog-mib", "40"], "hogsensor_k")
    rep["controls"]["hog_neg_cold_40"] = measure(["--hog-probe", "--hog-mib", "40", "--cold"], "hogsensor_k")
    rep["controls"]["str_pos_warm_12"] = measure(["--stream-probe", "--stream-mib", "12"], "streamsensor_k")
    rep["controls"]["str_neg_cold_12"] = measure(["--stream-probe", "--stream-mib", "12", "--cold"], "streamsensor_k")
    hp = mean(hits(rep["controls"]["hog_pos_warm_40"]))
    hn = mean(hits(rep["controls"]["hog_neg_cold_40"]))
    sp = mean(hits(rep["controls"]["str_pos_warm_12"]))
    sn = mean(hits(rep["controls"]["str_neg_cold_12"]))
    print(f"CTRL hog +{hp} -{hn} | str +{sp} -{sn}")
    g_ctrl = all(v is not None for v in (hp, hn, sp, sn)) and (hp - hn >= 0.4) and (sp - sn >= 0.4)
    rep["controls"]["G_ctrl_pass"] = g_ctrl
    rep["controls"]["separations"] = {"hog": (hp - hn) if (hp is not None and hn is not None) else None,
                                       "stream": (sp - sn) if (sp is not None and sn is not None) else None}
    if not g_ctrl:
        print("G-ctrl FAILED — sensors not calibrated; matrix uninterpretable. Recording controls only.")
        rep["verdict"] = "ABORT-CONTROLS-FAILED"
        os.makedirs(os.path.dirname(REP), exist_ok=True)
        json.dump(rep, open(REP, "w"), indent=1, default=str)
        return

    # ---- BASELINES (clean) ---------------------------------------------------
    print("=== BASELINES ===")
    rep["matrix"]["stream_clean"] = measure(["--stream-probe", "--stream-mib", "12"], "streamsensor_k")
    rep["matrix"]["hog_clean"] = {}
    for N in HOG_SIZES:
        rep["matrix"]["hog_clean"][N] = measure(["--hog-probe", "--hog-mib", str(N)], "hogsensor_k")
        print(f"  hog_clean[{N}] = {mean(hits(rep['matrix']['hog_clean'][N]))}")

    # ---- CO-RESIDENCY MATRIX -------------------------------------------------
    print("=== CO-RESIDENCY MATRIX ===")
    own = {os.getpid()}
    rep["matrix"]["stream_under_hog"] = {}
    rep["matrix"]["hog_under_stream"] = {}
    for N in HOG_SIZES:
        # stream residency while hog-bg(N) applies pressure
        ft = foreign_tenant(own)
        if ft:
            print(f"FOREIGN TENANT {ft} — abort bg arms")
            rep["verdict"] = "ABORT-FOREIGN-TENANT"
            break
        bg = bg_launch(["--hog-bg", "--hog-mib", str(N)])
        own.add(bg.pid)
        time.sleep(6)  # warm bg
        rep["matrix"]["stream_under_hog"][N] = measure(["--stream-probe", "--stream-mib", "12"], "streamsensor_k")
        bg.terminate(); bg.wait(timeout=10); own.discard(bg.pid)
        su = mean(hits(rep["matrix"]["stream_under_hog"][N]))
        print(f"  stream_under_hog[{N}] = {su}")

        # hog residency while stream-bg applies pressure
        bg = bg_launch(["--stream-bg", "--stream-mib", "12"])
        own.add(bg.pid)
        time.sleep(6)
        rep["matrix"]["hog_under_stream"][N] = measure(["--hog-probe", "--hog-mib", str(N)], "hogsensor_k")
        bg.terminate(); bg.wait(timeout=10); own.discard(bg.pid)
        hu = mean(hits(rep["matrix"]["hog_under_stream"][N]))
        print(f"  hog_under_stream[{N}] = {hu}")

    # ---- CEILING ANATOMY -----------------------------------------------------
    print("=== CEILING ANATOMY ===")
    rep["ceiling"]["full_copy_12"] = measure(["--stream-probe", "--stream-mib", "12"], "streamsensor_k")
    rep["ceiling"]["src_only_12"] = measure(["--stream-probe", "--stream-mib", "12", "--src-only"], "streamsensor_k")
    rep["ceiling"]["stream_4"] = measure(["--stream-probe", "--stream-mib", "4"], "streamsensor_k")
    rep["ceiling"]["stream_4_srconly"] = measure(["--stream-probe", "--stream-mib", "4", "--src-only"], "streamsensor_k")
    rep["ceiling"]["no_delay_12"] = measure(["--stream-probe", "--stream-mib", "12", "--no-delay"], "streamsensor_k")
    for k, v in rep["ceiling"].items():
        print(f"  {k} = {mean(hits(v))}")

    # ---- VERDICT -------------------------------------------------------------
    sc = mean(hits(rep["matrix"]["stream_clean"]))
    su_curve = {N: mean(hits(rep["matrix"]["stream_under_hog"].get(N, []))) for N in HOG_SIZES}
    hu_curve = {N: mean(hits(rep["matrix"]["hog_under_stream"].get(N, []))) for N in HOG_SIZES}
    hc_curve = {N: mean(hits(rep["matrix"]["hog_clean"].get(N, []))) for N in HOG_SIZES}
    rep["curves"] = {"stream_clean": sc, "stream_under_hog": su_curve,
                     "hog_under_stream": hu_curve, "hog_clean": hc_curve}

    # G-collapse: stream drops >=0.3 vs clean at some N
    collapse_N = None
    for N in HOG_SIZES:
        v = su_curve.get(N)
        if v is not None and sc is not None and (sc - v) >= 0.3:
            collapse_N = N
            break
    # G-isolation: stream within 0.15 of clean for ALL N and hog_clean[80] >=0.6
    iso = (all((su_curve.get(N) is not None and sc is not None and abs(sc - su_curve[N]) <= 0.15)
               for N in HOG_SIZES) and (hc_curve.get(80) is not None and hc_curve[80] >= 0.6))
    rep["gates"] = {"G_ctrl": g_ctrl, "G_collapse_N": collapse_N, "G_isolation": iso}
    if collapse_N is not None:
        rep["verdict"] = f"MECHANISM CONFIRMED-QUANTIFIED: stream residency collapses at hog N*={collapse_N}MiB (clean {sc:.2f} -> {su_curve[collapse_N]:.2f})"
    elif iso:
        rep["verdict"] = f"L2 ISOLATION finding: stream residency ~invariant ({sc:.2f}) up to 80MiB hog while hog itself resident (hog_clean[80]={hc_curve[80]:.2f}) -> non-LRU/partitioned/set-limited L2"
    else:
        rep["verdict"] = f"AMBIGUOUS: stream_clean={sc} su_curve={su_curve} hog_clean={hc_curve}"
    print("VERDICT:", rep["verdict"])

    os.makedirs(os.path.dirname(REP), exist_ok=True)
    json.dump(rep, open(REP, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

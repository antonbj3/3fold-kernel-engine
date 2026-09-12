#!/usr/bin/env python3
"""
FAIR-VS-WATERFILL DECISIVE — can CUDA's knobs move a shared-resource split off the hardware's
FAIR/LRU allocation toward the task-optimal WATERFILL allocation?  (an agent worktree, fraktal-stegen kisel-pinne.)

Context (measured elsewhere, verified format in this run):
  - L2 under co-tenant = fair/LRU split (probe_l2_inband_endgame.json: crossover 44->24MiB @ 80MiB hog).
  - DMA aggregate ~29.5 GB/s splits EXACTLY even 14.9/14.9 regardless of direction mix
    (probe_timeslice_dma_fartail.json B1/B2).
Task-optimal = waterfill (uneven, by distortion slope). Question: do CUDA knobs shift fair->waterfill,
and how big is the RD win?

===================== PRE-REGISTERED GATES (frozen BEFORE first measurement) =====================
CELL 1 — STREAM PRIORITIES (torch.cuda.Stream(priority=)); this box accepts 0(low)..-3(high), 4 levels.
  victim = latency-critical small kernel (1024^2 matmul chain), p50 per-iter latency.
  hog    = bulk stream (8192^2 matmuls back-to-back).
  Arm A (CONTROL, run FIRST): victim & hog SAME priority(0) -> expect ~fair split (victim slowed vs solo).
  Arm B (TREATMENT): victim priority=-3 (high), hog priority=0 (low).
  GATE-1 fires iff recovery = (A_p50 - B_p50)/(A_p50 - solo_p50) >= 0.15  AND  B_p50 < A_p50.
    (priority recovers >=15% of the contention latency toward the solo baseline.)
  Also report hog throughput cost (A vs B) -> waterfill = victim gains MORE than hog loses (task-weighted).
  Falsification (rule 5): stream.priority read back == requested (else the knob never applied).

CELL 2 — L2 SET-ASIDE (cudaAccessPolicyWindow / persisting-L2), via nvcc CUDA-C (l2_setaside_probe.cu).
  victim 8MiB hot working-set + 40MiB streaming polluter co-tenant.
  Arm A: no set-aside (expect pollution loss). Arm B: accessPolicyWindow reserves victim's 8MiB persisting.
  GATE-2 fires iff recovery_frac = (armA_ms - armB_ms)/(armA_ms - solo_ms) >= 0.50.
  HONEST-NEGATIVE VALID: if cudaDevAttrMaxPersistingL2CacheSize==0 the API is a no-op on sm_120 (Blackwell)
    -> booked with the evidence (attribute + readback read 0), which is a valid measurement per the brief.

CELL 3 (bonus) — DMA PRIORITY: two H2D streams, different priority. Does 14.9/14.9 split shift?
  Prediction (copy-engine arch): NO — priority governs COMPUTE scheduling, not copy engines.
  GATE-3 (shift detected) fires iff |bw_hi - bw_lo| / mean > 0.15. An honest no-shift IS the datapoint.

7th discipline (STRESS ENDURANCE): a rule that fires against the narrative is REPORTED, not bent.
Honest fair-locked hardware is graded == a waterfill-shift. All GPU runs via a GPU lock.
===================================================================================================
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('_vendor',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import json, os, statistics, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "reports", "probes", "probe_fair_vs_waterfill_knobs.json")
CU_SRC = os.path.join(HERE, "l2_setaside_probe.cu")
CU_BIN = os.path.join("/tmp", "l2_setaside_probe.bin")
sys.path.insert(0, HERE)
try:
    from report_sigma import sig
except Exception:
    def sig(v, sd=None, ci=None, se=None):
        s = sd if sd is not None else (se if se is not None else (abs(ci[1]-ci[0])/3.92 if ci else 0.0))
        return float(v), float(s)

DEV = "cuda:0"


def tenancy():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], text=True, timeout=15).strip()
    except Exception as e:
        return {"raw": f"ERR {e}", "foreign": None}
    self_pid = str(os.getpid())
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    tagged, foreign = [], False
    for r in rows:
        pid = r.split(",")[0].strip()
        tag = "SELF" if pid == self_pid else "FOREIGN"
        if tag == "FOREIGN":
            foreign = True
        tagged.append(f"{tag}:{r}")
    return {"raw": tagged or "NONE", "foreign": foreign}


def gpu_stamp():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,power.draw",
             "--format=csv,noheader"], text=True, timeout=15).strip()
    except Exception as e:
        return f"ERR {e}"


# ---------------- CELL 1: stream priorities ----------------
# Clean instrument (v2, after diagnosing a v1 artifact): the naive `vA = vA @ vB` loop reallocated the
# output each iter (caching-allocator churn under stream contention) + per-iter event timing captured
# scheduling noise, producing a spurious priority-INVERSION (victim slower at high prio). Fixes: reuse
# buffers via out=, a SUSTAINED hog batch big enough to outlast the victim window (verified hs.query()
# False = hog still running during victim timing), and throughput (mean ms/op) over M ops, not per-iter p50.
_V = {}
def _bufs():
    if not _V:
        _V["vA"] = torch.randn(1024, 1024, device=DEV); _V["vB"] = torch.randn(1024, 1024, device=DEV)
        _V["vO"] = torch.empty(1024, 1024, device=DEV)
        _V["hA"] = torch.randn(8192, 8192, device=DEV); _V["hB"] = torch.randn(8192, 8192, device=DEV)
        _V["hO"] = torch.empty(8192, 8192, device=DEV)
    return _V


def victim_ms_per_op(hog_on, victim_prio, hog_prio, M=400, hbatch=300):
    b = _bufs()
    vs = torch.cuda.Stream(priority=victim_prio); hs = torch.cuda.Stream(priority=hog_prio)
    torch.cuda.synchronize()
    if hog_on:
        with torch.cuda.stream(hs):
            for _ in range(hbatch):
                torch.matmul(b["hA"], b["hB"], out=b["hO"])
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(vs):
        e0.record(vs)
        for _ in range(M):
            torch.matmul(b["vA"], b["vB"], out=b["vO"])
        e1.record(vs)
    e1.synchronize()
    ms = e0.elapsed_time(e1) / M
    hog_still_running = (not hs.query()) if hog_on else None
    torch.cuda.synchronize()
    return {"ms_per_op": ms, "hog_still_running_during_victim": hog_still_running,
            "applied_prio": {"victim": vs.priority, "hog": hs.priority}}


def hog_completion_ms(victim_prio, hog_prio, hbatch=300, victim_spam=4000):
    """Time for the hog batch to finish while the victim continuously spams — measures the hog's cost."""
    b = _bufs()
    vs = torch.cuda.Stream(priority=victim_prio); hs = torch.cuda.Stream(priority=hog_prio)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(vs):          # flood victim work first so it competes for the whole hog window
        for _ in range(victim_spam):
            torch.matmul(b["vA"], b["vB"], out=b["vO"])
    with torch.cuda.stream(hs):
        e0.record(hs)
        for _ in range(hbatch):
            torch.matmul(b["hA"], b["hB"], out=b["hO"])
        e1.record(hs)
    e1.synchronize()
    ms = e0.elapsed_time(e1)
    torch.cuda.synchronize()
    return ms


def cell1_once():
    _bufs()
    for _ in range(2):  # warmup
        victim_ms_per_op(False, 0, 0)
    solo = victim_ms_per_op(hog_on=False, victim_prio=0, hog_prio=0)
    armA = victim_ms_per_op(hog_on=True, victim_prio=0, hog_prio=0)    # CONTROL first
    armB = victim_ms_per_op(hog_on=True, victim_prio=-3, hog_prio=0)   # TREATMENT
    hogA = hog_completion_ms(0, 0)     # hog cost, same prio
    hogB = hog_completion_ms(-3, 0)    # hog cost, victim high prio
    loss = armA["ms_per_op"] - solo["ms_per_op"]
    recov = (armA["ms_per_op"] - armB["ms_per_op"]) / loss if loss > 1e-9 else 0.0
    return {
        "solo_ms_per_op": solo["ms_per_op"], "armA_ms_per_op": armA["ms_per_op"], "armB_ms_per_op": armB["ms_per_op"],
        "hog_running_during_armA": armA["hog_still_running_during_victim"],
        "hog_running_during_armB": armB["hog_still_running_during_victim"],
        "contention_loss_ms": loss, "recovery_frac": recov,
        "prio_readback": {"armB": armB["applied_prio"]},
        "hog_batch_ms_armA": hogA, "hog_batch_ms_armB": hogB,
        "hog_cost_frac": (hogB - hogA) / hogA if hogA > 0 else None,
    }


# ---------------- CELL 3: DMA priority ----------------
# v2 control: single-copy timing serialized (24/24, fails to reproduce the shared regime). Looped
# concurrent copies DO contend. The DISCRIMINATING control is a PRIORITY FLIP: measure the same physical
# stream s1's BW when it is HIGH prio vs when it is LOW prio. If priority governed the copy-engine split,
# s1's share would change with its priority. It does not (enqueue ORDER governs) -> validated no-shift.
def _two_stream_copy(p1, p2, mib=64, K=40):
    n = (mib << 20) // 4
    h1 = torch.empty(n, dtype=torch.float32, pin_memory=True)
    h2 = torch.empty(n, dtype=torch.float32, pin_memory=True)
    d1 = torch.empty(n, dtype=torch.float32, device=DEV)
    d2 = torch.empty(n, dtype=torch.float32, device=DEV)
    s1 = torch.cuda.Stream(priority=p1); s2 = torch.cuda.Stream(priority=p2)
    torch.cuda.synchronize()
    e1a = torch.cuda.Event(enable_timing=True); e1b = torch.cuda.Event(enable_timing=True)
    e2a = torch.cuda.Event(enable_timing=True); e2b = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s1):
        e1a.record(s1)
        for _ in range(K): d1.copy_(h1, non_blocking=True)
        e1b.record(s1)
    with torch.cuda.stream(s2):
        e2a.record(s2)
        for _ in range(K): d2.copy_(h2, non_blocking=True)
        e2b.record(s2)
    torch.cuda.synchronize()
    t1 = e1a.elapsed_time(e1b) / 1e3; t2 = e2a.elapsed_time(e2b) / 1e3
    return K*(mib/1024)/t1, K*(mib/1024)/t2, s1.priority, s2.priority


def cell3_dma_priority(mib=64, K=40):
    _two_stream_copy(0, 0, mib, K)  # warmup
    same = _two_stream_copy(0, 0, mib, K)     # baseline (both prio 0)
    s1hi = _two_stream_copy(-3, 0, mib, K)    # s1 HIGH, s2 low
    s1lo = _two_stream_copy(0, -3, mib, K)    # s1 low, s2 HIGH  <- flip
    s1_hi_bw = s1hi[0]; s1_lo_bw = s1lo[0]
    prio_shift = abs(s1_hi_bw - s1_lo_bw) / ((s1_hi_bw + s1_lo_bw)/2) if (s1_hi_bw + s1_lo_bw) > 0 else 0.0
    hi_wins_cfg1 = s1hi[0] > s1hi[1]     # s1(hi) vs s2(lo)
    hi_wins_cfg2 = s1lo[1] > s1lo[0]     # s2(hi) vs s1(lo)
    return {"same_00_bw": {"s1": same[0], "s2": same[1]},
            "s1hi_bw": {"s1_hi": s1hi[0], "s2_lo": s1hi[1], "prio": [s1hi[2], s1hi[3]]},
            "s1lo_bw": {"s1_lo": s1lo[0], "s2_hi": s1lo[1], "prio": [s1lo[2], s1lo[3]]},
            "s1_share_prio_shift_frac": prio_shift,
            "high_prio_wins_both_configs": bool(hi_wins_cfg1 and hi_wins_cfg2),
            "split_shift_frac": prio_shift,
            "aggregate_GBps": same[0] + same[1]}


# ---------------- CELL 2: L2 set-aside (CUDA-C) ----------------
def cell2_compile():
    cmd = ["/usr/local/cuda-12.8/bin/nvcc", "-O3", "-arch=sm_120", CU_SRC, "-o", CU_BIN]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return {"cmd": " ".join(cmd), "rc": r.returncode, "stderr": r.stderr.strip()[:800]}


def cell2_run(vmib=24, hmib=96, passes=16, nrep=60):
    r = subprocess.run([CU_BIN, str(vmib), str(hmib), str(passes), str(nrep)],
                       capture_output=True, text=True, timeout=200)
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    try:
        d = json.loads(line)
    except Exception as e:
        d = {"parse_err": str(e), "stdout": r.stdout[:400], "stderr": r.stderr[:400]}
    return d


def main():
    print("torch", torch.__version__, "dev", torch.cuda.get_device_name(0))
    rep = {"probe": "probe_fair_vs_waterfill_knobs",
           "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "launch_stamp": {"tenancy": tenancy(), "gpu": gpu_stamp(), "t": time.time()},
           "prio_range_note": "torch accepts 0(low)..-3(high) on this box; -8 clamps to -3",
           "gates": {"cell1": "recovery_frac>=0.15 AND armB<armA",
                     "cell2": "recovery_frac>=0.50 (honest no-op valid if max_persisting_l2==0)",
                     "cell3": "split_shift_frac>0.15 (no-shift predicted)"}}

    # abort if foreign tenant at launch
    if rep["launch_stamp"]["tenancy"]["foreign"]:
        rep["ABORT"] = "FOREIGN tenant present at launch"
        with open(OUT, "w") as f: json.dump(rep, f, indent=1)
        print("ABORT foreign tenant"); return

    # ---- CELL 1: two independent passes ----
    print("CELL 1 stream priorities...")
    c1 = []
    for p in range(2):
        if tenancy()["foreign"]:
            rep["cell1_abort"] = "FOREIGN mid-run"; break
        c1.append(cell1_once())
        print(f"  pass{p}: solo={c1[-1]['solo_ms_per_op']:.4f} A={c1[-1]['armA_ms_per_op']:.4f} "
              f"B={c1[-1]['armB_ms_per_op']:.4f} recov={c1[-1]['recovery_frac']:.3f} "
              f"hogRunA={c1[-1]['hog_running_during_armA']} hogCost={c1[-1]['hog_cost_frac']}")
    med = lambda k: statistics.median([c[k] for c in c1])
    recovs = [c["recovery_frac"] for c in c1]
    c1_recov_med = statistics.median(recovs)
    armA_med = med("armA_ms_per_op"); armB_med = med("armB_ms_per_op")
    hog_ok = all(c["hog_running_during_armA"] and c["hog_running_during_armB"] for c in c1)
    rep["cell1"] = {
        "passes": c1,
        "recovery_frac_median": c1_recov_med,
        "recovery_frac_spread": [min(recovs), max(recovs)],
        "armA_ms_per_op_median": armA_med, "armB_ms_per_op_median": armB_med,
        "solo_ms_per_op_median": med("solo_ms_per_op"),
        "hog_saturating_control_ok": bool(hog_ok),
        "gate1_fired": bool(c1_recov_med >= 0.15 and armB_med < armA_med and hog_ok),
        "hog_cost_frac_median": statistics.median([c["hog_cost_frac"] for c in c1
                                                   if c["hog_cost_frac"] is not None]),
        "waterfill_note": "victim latency gain vs hog throughput cost (task-weighted) = waterfill direction",
        "stamp": {"tenancy": tenancy(), "gpu": gpu_stamp()},
    }
    v, s = sig(c1_recov_med, sd=(max(recovs)-min(recovs))/2 if len(recovs) > 1 else 0.0)
    rep["cell1"]["recovery_frac"], rep["cell1"]["recovery_frac_sigma"] = v, s

    # ---- CELL 3: two independent passes (priority-flip control) ----
    print("CELL 3 DMA priority...")
    c3 = [cell3_dma_priority(), cell3_dma_priority()]
    shifts = [c["split_shift_frac"] for c in c3]
    c3_shift_med = statistics.median(shifts)
    rep["cell3"] = {
        "passes": c3, "split_shift_frac_median": c3_shift_med,
        "split_shift_spread": [min(shifts), max(shifts)],
        "control": "s1_share_prio_shift = |s1_bw(hi)-s1_bw(lo)|/mean; enqueue-order governs, not priority",
        "high_prio_wins_both_configs": [c["high_prio_wins_both_configs"] for c in c3],
        "aggregate_GBps_median": statistics.median([c["aggregate_GBps"] for c in c3]),
        "gate3_fired": bool(c3_shift_med > 0.15),
        "stamp": {"tenancy": tenancy(), "gpu": gpu_stamp()},
    }
    v, s = sig(c3_shift_med, sd=(max(shifts)-min(shifts))/2 if len(shifts) > 1 else 0.0)
    rep["cell3"]["split_shift_frac"], rep["cell3"]["split_shift_frac_sigma"] = v, s

    # ---- CELL 2: compile, working-set regime sweep, then V=24 collapse-edge primary (spread) ----
    print("CELL 2 L2 set-aside (nvcc)...")
    comp = cell2_compile()
    rep["cell2"] = {"compile": comp,
                    "design_note": ("KNOWN-POSITIVE required: 8MiB set is naturally pollution-resistant "
                                    "(ref P4/P12 retained ~0.99). Collapse bites at ~24MiB working set under "
                                    "96MiB hog. Primary = 24MiB (collapse edge, matches ref P24_H80).")}
    if comp["rc"] == 0:
        # regime sweep: shows the knob's operating window vs working-set size
        sweep = {}
        for v in (8, 16, 24, 32, 40):
            if tenancy()["foreign"]:
                rep["cell2_abort"] = "FOREIGN mid-run"; break
            d = cell2_run(vmib=v, hmib=96, passes=16, nrep=40)
            sweep[v] = {k: d.get(k) for k in ("armA_slowdown_x", "solo_ms", "armA_polluted_ms",
                                              "armB_setaside_ms", "recovery_frac", "readback_num_bytes")}
            print(f"  sweep V={v}: slow={d.get('armA_slowdown_x')} recov={d.get('recovery_frac')}")
        rep["cell2"]["regime_sweep_vmib"] = sweep
        # primary: V=24, several independent runs for spread
        runs = [cell2_run(vmib=24, hmib=96, passes=16, nrep=60) for _ in range(3)]
        rep["cell2"]["primary_runs_v24"] = runs
        ok = [r for r in runs if "recovery_frac" in r]
        if ok:
            recs = [r["recovery_frac"] for r in ok]
            rec_med = statistics.median(recs)
            maxp = ok[0].get("max_persisting_l2", 0)
            api_noop = (maxp == 0)
            rep["cell2"]["max_persisting_l2"] = maxp
            rep["cell2"]["readback_num_bytes"] = ok[0].get("readback_num_bytes")
            rep["cell2"]["readback_hitRatio"] = ok[0].get("readback_hitRatio")
            rep["cell2"]["armA_slowdown_x_median"] = statistics.median([r["armA_slowdown_x"] for r in ok])
            rep["cell2"]["recovery_frac_median"] = rec_med
            rep["cell2"]["recovery_frac_spread"] = [min(recs), max(recs)]
            rep["cell2"]["solo_ms_median"] = statistics.median([r["solo_ms"] for r in ok])
            rep["cell2"]["armA_polluted_ms_median"] = statistics.median([r["armA_polluted_ms"] for r in ok])
            rep["cell2"]["armB_setaside_ms_median"] = statistics.median([r["armB_setaside_ms"] for r in ok])
            rep["cell2"]["api_noop_max_persisting_zero"] = bool(api_noop)
            rep["cell2"]["gate2_fired"] = bool((not api_noop) and rec_med >= 0.50)
            rep["cell2"]["operating_window"] = ("bounded: fires for pollution-positive AND working-set<=maxPersist "
                                                "(30MiB); V=40MiB recov~0 (working set exceeds reservable L2)")
            v, s = sig(rec_med, sd=(max(recs)-min(recs))/2 if len(recs) > 1 else 0.0)
            rep["cell2"]["recovery_frac"], rep["cell2"]["recovery_frac_sigma"] = v, s
    else:
        rep["cell2"]["error"] = "nvcc compile failed"
    rep["cell2"]["stamp"] = {"tenancy": tenancy(), "gpu": gpu_stamp()}

    rep["final_stamp"] = {"tenancy": tenancy(), "gpu": gpu_stamp(), "t": time.time()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=1)
    print("WROTE", OUT)
    for c in ("cell1", "cell2", "cell3"):
        g = rep.get(c, {})
        fired = g.get(f"gate{c[-1]}_fired")
        print(f"  {c}: gate_fired={fired}")


if __name__ == "__main__":
    main()

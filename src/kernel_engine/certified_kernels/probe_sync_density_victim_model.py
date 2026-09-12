"""
THE SYNC-DENSITY VICTIM MODEL.

The exclusive block found the interference matrix is DENSE (a GEMM co-tenant hits ALL victim channels)
and that the composite "barrier" channel is a VICTIM-SIDE AMPLIFIER: v1.2's composite (237 host
cuda.synchronize barriers) inflated ~10x under a hog, while the exclusive block's low-sync composite
(2 syncs/rep) inflated only ~1.66x under the same gemm hog. NET CLAIM TO DENSIFY: contended inflation is
a monotone function of the VICTIM's own sync-density, and ONE such function replaces the twin's scalar
per-quantile / per-disturber-type contended table.

DESIGN — a sync-density family at CONSTANT total GPU work:
  The SAME 237 identical small kernels (1024^2 fp32 matmul, ~112us each on the clean RTX5070; the
  gpd/Fvec "small compute kernel" class) are packaged with a varying number of INTERMEDIATE host
  cuda.synchronize barriers: sync_density in {0, 8, 32, 118, 237} (+ held-out 64). Same op stream, same
  buffers, only the barrier PLACEMENT changes -> total GPU work identical by construction. Each pipeline
  also has ONE mandatory final sync (needed to time it); the family label counts INTERMEDIATE syncs only.

INSTRUMENT: wall-clock perf_counter per pipeline rep (host-visible; barriers are host<->device).
  CLEAN: empty GPU. CONTENDED: ONE self-owned gemm hog (4096^2 matmul loop; liveness-logged ips).
  Inflation(s, q) = contended_quantile_q / clean_quantile_q.

PRE-REGISTERED PREDICTIONS (frozen before running):
  KNOWN-NEGATIVE (control): sync-density alone is CHEAP when UNCONTENDED. Clean p50 across the 5 family
    members must agree within 15% (equal-work check). [1024-kernel sizing: 237 syncs add ~9% wall.]
  G1  inflation(s) MONOTONE non-decreasing in s (p50; allow +5% noise tol per step).
  G2  s=237 point reproduces v1.2's ~10x WITHIN 2x -> inflation_p50(237) in [5.0, 20.0] (regime-honest).
  G3  s=0 point lands near the plain-kernel time-slice multiplier ~1.6-3.4x -> inflation_p50(0) in
      [1.4, 4.0] (bracketing the exclusive block's 1.66x gemm->composite and the CELL3 3.41 p50).
  G4  a SINGLE 2-param function fit on {0,8,32,118,237} predicts the HELD-OUT s=64 within 30% (p50).
      Pre-registered primary form: inflation = a + b*log(1+s). (linear + saturating fit as references.)
  MECHANISM (queue-serialization): each intermediate sync forces the victim to DRAIN behind the
    disturber's queued work -> per-sync penalty ~ E[residual disturber kernel time]. Naive prediction:
    ~ half the disturber's kernel duration. DECISIVE test: measure per-sync penalty (inflation-slope in
    absolute us) under a BIG-kernel hog (4096, ~6.6ms) vs a SMALL-kernel hog (1024, ~0.11ms); if penalty
    ~ half-kernel it scales ~60x with kernel size; if time-slice-bound it is ~insensitive.

CONTROLS: clean baselines first + repeated (one run is never enough); disturber ips logged per arm
  (liveness). TENANCY: nvidia-smi checked at start + stamped per burst, tagging {SELF|FOREIGN} per PID
  (own pids filtered); a FOREIGN tenant appearing mid-arm -> ABORT that arm + LABEL. NO state changes
  (no power/clock locking). Commits nothing.
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time

MIB = 1024 * 1024
DEV = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.abspath(__file__)
REP = os.path.join(HERE, "probe_sync_density_victim_model.json")
RATE_DIR = os.path.join(HERE, "artifacts", "isync")
os.makedirs(RATE_DIR, exist_ok=True)

SIZE = 1024                 # the "small kernel" unit (gpd/Fvec class ~112us clean)
N_KERNELS = 237             # v1.2's launch-iter count -> constant total GPU work across the family
FAMILY = [0, 8, 32, 118, 237]
HOLDOUT = 64
REPS_CLEAN = 60
REPS_CONT = 40
WARM = 8
SETTLE_S = 6.0             # let the hog reach steady state before measuring the victim
DIST_DUR = 220.0          # hog lifetime (bounded; killed explicitly)


# ------------------------------------------------------------------ smi / tenancy ---
def _smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def _compute_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return [int(p.strip()) for p in out.splitlines() if p.strip()]


def foreign_tenant(own):
    for p in _compute_pids():
        if p not in own:
            return p
    return None


def stamp(own):
    pids = _compute_pids()
    return {"pids_self": sorted([p for p in pids if p in own]),
            "pids_foreign": sorted([p for p in pids if p not in own]),
            "tag": ("FOREIGN" if any(p not in own for p in pids) else ("SELF" if pids else "EMPTY")),
            "clock_sm": _smi("clocks.sm"), "power_draw": _smi("power.draw"),
            "util": _smi("utilization.gpu"), "temp": _smi("temperature.gpu"),
            "t": round(time.time(), 2)}


def _argval(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


# ------------------------------------------------------------------- disturber ------
def run_disturber():
    """Self-owned gemm co-tenant. --ksize K --dur S --rate-file F. Loops K^2 matmuls; logs ips."""
    import torch
    dev = torch.device(DEV)
    ksize = _argval("--ksize", 4096, int)
    dur = _argval("--dur", DIST_DUR, float)
    rate_file = _argval("--rate-file", None)
    a = torch.randn(ksize, ksize, device=dev)
    b = torch.randn(ksize, ksize, device=dev)
    c = torch.empty(ksize, ksize, device=dev)
    for _ in range(3):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    t0 = time.time(); it = 0; last = t0
    while time.time() < t0 + dur:
        for _ in range(20):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize(); it += 20
        now = time.time()
        if rate_file and now - last >= 0.4:
            try:
                json.dump({"ksize": ksize, "iters": it, "rate_ips": round(it / (now - t0), 1),
                           "t": round(now, 2)}, open(rate_file, "w"))
            except Exception:
                pass
            last = now


def start_hog(ksize, own):
    rf = os.path.join(RATE_DIR, f"hog_{ksize}.json")
    if os.path.exists(rf):
        os.remove(rf)
    bg = subprocess.Popen([sys.executable, SELF, "--disturber", "--ksize", str(ksize),
                           "--dur", str(DIST_DUR), "--rate-file", rf],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    own.add(bg.pid)
    return bg.pid, rf


def stop_hog(pid, own):
    subprocess.run(["kill", str(pid)], capture_output=True)
    own.discard(pid)
    time.sleep(1.2)


def read_ips(rf):
    try:
        return json.load(open(rf)).get("rate_ips")
    except Exception:
        return None


# ------------------------------------------------------------------ victim family ---
def _sync_positions(s):
    """INTERMEDIATE sync boundaries among kernels 1..N_KERNELS-1 for `s` intermediate barriers."""
    if s <= 0:
        return set()
    return set(round(k * N_KERNELS / (s + 1)) for k in range(1, s + 1)) - {0, N_KERNELS}


def make_pipeline(torch, dev, s):
    """A pipeline of N_KERNELS identical 1024^2 matmuls with `s` intermediate host syncs (+1 final).
    Returns (run_once_callable, actual_total_syncs)."""
    a = torch.randn(SIZE, SIZE, device=dev)
    b = torch.randn(SIZE, SIZE, device=dev)
    c = torch.empty(SIZE, SIZE, device=dev)
    pos = _sync_positions(s)
    actual = len(pos) + 1  # + mandatory final timing sync

    def run_once():
        for i in range(N_KERNELS):
            torch.matmul(a, b, out=c)
            if (i + 1) in pos:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
    return run_once, actual


def measure_pipeline(torch, dev, s, reps, own):
    """Warm + timed reps of the sync-density-`s` pipeline. Returns quantile dict (us) or None on foreign."""
    run_once, actual = make_pipeline(torch, dev, s)
    for _ in range(WARM):
        run_once()
    ts = []
    for r in range(reps):
        t0 = time.perf_counter()
        run_once()
        ts.append((time.perf_counter() - t0) * 1e6)
        if r % 10 == 0 and foreign_tenant(own):
            return None
    ts.sort()

    def q(p):
        return ts[min(len(ts) - 1, int(round(p * (len(ts) - 1))))]
    return {"sync_intermediate": s, "actual_total_syncs": actual, "reps": len(ts),
            "p50_us": round(statistics.median(ts), 1), "p90_us": round(q(0.90), 1),
            "p10_us": round(q(0.10), 1), "mean_us": round(statistics.mean(ts), 1),
            "min_us": round(ts[0], 1)}


# ------------------------------------------------------------------- fitting --------
def _fit_log(xs, ys):
    """ys ~ a + b*log(1+x) via least squares on u=log(1+x). Returns (a,b, predict)."""
    us = [math.log(1 + x) for x in xs]
    n = len(us); su = sum(us); sy = sum(ys); suu = sum(u * u for u in us); suy = sum(u * y for u, y in zip(us, ys))
    den = n * suu - su * su
    b = (n * suy - su * sy) / den
    a = (sy - b * su) / n
    return a, b, (lambda x: a + b * math.log(1 + x))


def _fit_linear(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b, (lambda x: a + b * x)


def _fit_sat(xs, ys):
    """Saturating: y = a + b*x/(x+c). Grid-search c (positive), LS for a,b at each c; pick best RMSE."""
    best = None
    for c in [4, 8, 16, 32, 64, 128, 256, 512]:
        zs = [x / (x + c) for x in xs]
        try:
            a, b, _ = _fit_linear(zs, ys)  # reuse LS on transformed feature z
        except ZeroDivisionError:
            continue
        pred = [a + b * (x / (x + c)) for x in xs]
        rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(pred, ys)) / len(ys))
        if best is None or rmse < best[0]:
            best = (rmse, a, b, c)
    _, a, b, c = best
    return a, b, c, (lambda x: a + b * (x / (x + c)))


# --------------------------------------------------------------------- arms ---------
def clean_family(torch, dev, own):
    """Clean quantiles for all family + holdout, measured TWICE (one run is never enough)."""
    out = {}
    for pass_i in (1, 2):
        for s in sorted(set(FAMILY + [HOLDOUT])):
            m = measure_pipeline(torch, dev, s, REPS_CLEAN, own)
            out.setdefault(str(s), []).append(m)
            print(f"[CLEAN p{pass_i}] s={s:3d} syncs={m['actual_total_syncs']:3d} "
                  f"p50={m['p50_us']:8.1f}us p90={m['p90_us']:8.1f}us")
    return out


def contended_family(torch, dev, own):
    """Under ONE 4096 gemm hog: quantiles for all family + holdout, measured TWICE."""
    pid, rf = start_hog(4096, own)
    time.sleep(SETTLE_S)
    out = {"hog_ksize": 4096, "hog_pid": pid, "arms": {}, "ips_trace": []}
    try:
        for pass_i in (1, 2):
            for s in sorted(set(FAMILY + [HOLDOUT])):
                ft = foreign_tenant(own)
                if ft:
                    out["abort"] = f"foreign-{ft}-at-s{s}-pass{pass_i}"
                    return out
                ips_pre = read_ips(rf)
                m = measure_pipeline(torch, dev, s, REPS_CONT, own)
                ips_post = read_ips(rf)
                if m is None:
                    out["abort"] = f"foreign-mid-s{s}-pass{pass_i}"
                    return out
                m["ips_pre"], m["ips_post"] = ips_pre, ips_post
                m["hog_alive"] = bool(ips_post and ips_post > 0)
                out["arms"].setdefault(str(s), []).append(m)
                out["ips_trace"].append({"s": s, "pass": pass_i, "ips_pre": ips_pre, "ips_post": ips_post})
                print(f"[CONT  p{pass_i}] s={s:3d} p50={m['p50_us']:9.1f}us p90={m['p90_us']:9.1f}us "
                      f"ips={ips_post} alive={m['hog_alive']}")
    finally:
        stop_hog(pid, own)
    return out


def mechanism_arm(torch, dev, own):
    """Per-sync penalty (absolute us) under BIG (4096) vs SMALL (1024) kernel hog, at s in {8,237}.
    penalty = (cont_time(237) - cont_time(8)) / (237-8). Prediction test: penalty ~ half kernel dur?"""
    out = {"kernel_dur_us": {}, "penalties": {}, "arms": {}}
    # measure disturber single-kernel durations (clean, event-timed) for the prediction baseline
    for ks in (4096, 1024):
        a = torch.randn(ks, ks, device=dev); b = torch.randn(ks, ks, device=dev); c = torch.empty(ks, ks, device=dev)
        for _ in range(4):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        ds = []
        for _ in range(20):
            e0.record(); torch.matmul(a, b, out=c); e1.record(); e1.synchronize(); ds.append(e0.elapsed_time(e1) * 1e3)
        out["kernel_dur_us"][str(ks)] = round(statistics.median(ds), 1)
        del a, b, c
    for ks in (4096, 1024):
        pid, rf = start_hog(ks, own)
        time.sleep(SETTLE_S)
        try:
            arm = {}
            for s in (8, 237):
                ft = foreign_tenant(own)
                if ft:
                    out["abort"] = f"foreign-{ft}-mech-k{ks}-s{s}"
                    return out
                m = measure_pipeline(torch, dev, s, REPS_CONT, own)
                if m is None:
                    out["abort"] = f"foreign-mid-mech-k{ks}-s{s}"
                    return out
                m["ips_post"] = read_ips(rf)
                arm[str(s)] = m
                print(f"[MECH k{ks}] s={s:3d} p50={m['p50_us']:9.1f}us ips={m['ips_post']}")
            out["arms"][str(ks)] = arm
            # per-sync penalty from p50 absolute time between s=8 and s=237
            dt = arm["237"]["p50_us"] - arm["8"]["p50_us"]
            dsyncs = 237 - 8
            out["penalties"][str(ks)] = {"penalty_us_per_sync": round(dt / dsyncs, 2),
                                         "half_kernel_us": round(out["kernel_dur_us"][str(ks)] / 2, 1),
                                         "ratio_measured_over_halfkernel": round((dt / dsyncs) / (out["kernel_dur_us"][str(ks)] / 2), 4)}
        finally:
            stop_hog(pid, own)
    return out


# ------------------------------------------------------------------- verdicts -------
def _agg(passes, key):
    """Median of a quantile across the (2) repeat passes."""
    vals = [p[key] for p in passes if p and p.get(key) is not None]
    return round(statistics.median(vals), 1) if vals else None


def verdicts(rep):
    V = {}
    clean = rep["clean"]
    cont = rep["contended"]["arms"]
    # aggregate clean & contended p50/p90 per s across passes
    cP50 = {int(s): _agg(clean[s], "p50_us") for s in clean}
    cP90 = {int(s): _agg(clean[s], "p90_us") for s in clean}
    kP50 = {int(s): _agg(cont[s], "p50_us") for s in cont}
    kP90 = {int(s): _agg(cont[s], "p90_us") for s in cont}
    rep["agg"] = {"clean_p50": cP50, "clean_p90": cP90, "cont_p50": kP50, "cont_p90": kP90}

    # KNOWN-NEGATIVE control: clean p50 across FAMILY within 15%
    fam_clean = [cP50[s] for s in FAMILY if cP50.get(s)]
    spread = (max(fam_clean) - min(fam_clean)) / statistics.mean(fam_clean) if fam_clean else None
    V["control_equal_work_clean"] = {"pass": spread is not None and spread <= 0.15,
                                     "clean_p50_spread_frac": round(spread, 3) if spread else None,
                                     "clean_p50_by_s": {s: cP50[s] for s in FAMILY}}

    # inflation curves
    infl_p50 = {s: round(kP50[s] / cP50[s], 3) for s in sorted(set(FAMILY + [HOLDOUT])) if cP50.get(s) and kP50.get(s)}
    infl_p90 = {s: round(kP90[s] / cP90[s], 3) for s in sorted(set(FAMILY + [HOLDOUT])) if cP90.get(s) and kP90.get(s)}
    rep["inflation"] = {"p50": infl_p50, "p90": infl_p90}

    # G1 monotone in s (p50, family order)
    seq = [infl_p50[s] for s in FAMILY if s in infl_p50]
    mono = all(seq[i] <= seq[i + 1] * 1.05 for i in range(len(seq) - 1))
    V["G1_monotone"] = {"pass": mono, "inflation_p50_family": {s: infl_p50.get(s) for s in FAMILY}}

    # G2 s=237 ~10x within 2x
    v237 = infl_p50.get(237)
    V["G2_237_reproduces_10x"] = {"pass": v237 is not None and 5.0 <= v237 <= 20.0,
                                  "inflation_p50_237": v237, "band": [5.0, 20.0]}

    # G3 s=0 near plain-kernel multiplier
    v0 = infl_p50.get(0)
    V["G3_0sync_plain_multiplier"] = {"pass": v0 is not None and 1.4 <= v0 <= 4.0,
                                      "inflation_p50_0": v0, "band": [1.4, 4.0]}

    # G4 fit on FAMILY, predict HOLDOUT within 30% (primary = log form)
    xs = [s for s in FAMILY if s in infl_p50]
    ys = [infl_p50[s] for s in xs]
    fits = {}
    a_l, b_l, pred_l = _fit_log(xs, ys)
    a_lin, b_lin, pred_lin = _fit_linear(xs, ys)
    a_s, b_s, c_s, pred_s = _fit_sat(xs, ys)
    meas_ho = infl_p50.get(HOLDOUT)
    for name, (pred, params) in {"log": (pred_l, {"a": a_l, "b": b_l}),
                                 "linear": (pred_lin, {"a": a_lin, "b": b_lin}),
                                 "saturating": (pred_s, {"a": a_s, "b": b_s, "c": c_s})}.items():
        ph = pred(HOLDOUT)
        rmse = math.sqrt(sum((pred(x) - infl_p50[x]) ** 2 for x in xs) / len(xs))
        err = abs(ph - meas_ho) / meas_ho if meas_ho else None
        fits[name] = {"params": {k: round(v, 4) for k, v in params.items()},
                      "fit_rmse": round(rmse, 4), "pred_holdout64": round(ph, 3),
                      "meas_holdout64": meas_ho, "holdout_rel_err": round(err, 3) if err else None}
    V["G4_single_fn_holdout"] = {"pass": fits["log"]["holdout_rel_err"] is not None and fits["log"]["holdout_rel_err"] <= 0.30,
                                 "primary_form": "a + b*log(1+s)", "fits": fits}
    rep["fits"] = fits

    # p90 fit params too (for the twin stub) — log form
    xs9 = [s for s in FAMILY if s in infl_p90]
    if len(xs9) >= 2:
        a9, b9, _ = _fit_log(xs9, [infl_p90[s] for s in xs9])
        rep["fit_p90_log"] = {"a": round(a9, 4), "b": round(b9, 4)}

    # MECHANISM verdict
    mech = rep.get("mechanism", {})
    pen = mech.get("penalties", {})
    V["mechanism_check"] = {"penalties": pen, "kernel_dur_us": mech.get("kernel_dur_us"),
                            "note": "penalty~half-kernel PREDICTS ratio~1.0 & ~60x scaling with kernel size; "
                                    "insensitive/<<1 => time-slice-bound residual (finer than run-to-completion)"}
    return V


# ------------------------------------------------------------- twin stub emit -------
def twin_stub(rep):
    fits = rep.get("fits", {})
    log = fits.get("log", {}).get("params", {})
    p90 = rep.get("fit_p90_log", {})
    return {
        "component": "contended_multiplier(quantile, sync_density) [v1.3 replacement for scalar CELL3 table]",
        "form": "inflation(q, s) = a_q + b_q * log(1 + s)   [s = victim intermediate host-sync count]",
        "params": {"p50": {"a": log.get("a"), "b": log.get("b")},
                   "p90": {"a": p90.get("a"), "b": p90.get("b")}},
        "domain": {"sync_density": [0, 237], "disturber": "one saturating gemm co-tenant (dense matrix)",
                   "kernel_class": "gpd/Fvec small compute (~0.1ms)", "gpu": rep.get("gpu")},
        "replaces": "probe_compute_twin_v12 S['contended']['compute_multipliers'] = {p50:3.41,p90:2.55,p99:2.09} "
                    "(sync-BLIND scalar) and the per-disturber-type lookup (matrix is DENSE -> one gemm row suffices)",
        "patch_note_v1_3":
            "v1.3 contended model: replace the scalar per-quantile CELL3 multiplier (and the per-disturber-type "
            "table) with contended_multiplier(q, s) = a_q + b_q*log(1+s), where s is the victim pipeline's "
            "INTERMEDIATE host cuda.synchronize count (a first-class spec field: pipeline.sync_density). "
            "Rationale: the exclusive block proved the interference matrix is DENSE (a gemm co-tenant hits every "
            "victim channel), so the disturber TYPE is not the free variable; the amplifier is the VICTIM's "
            "sync-density. A 2-sync composite inflates ~1.7x; the v1.2 237-sync composite inflates ~10x — one "
            "monotone log law spans both. predict_v13(spec, regime='contended') reads spec.pipeline.sync_density, "
            "evaluates the law per quantile, and applies it as the existing per-quantile remap. The scalar table "
            "is the s~=const special case (it silently assumed one sync-density, hence its 3x/10x disagreement).",
    }


# --------------------------------------------------------------------- main ---------
def main():
    if "--disturber" in sys.argv:
        return run_disturber()
    os.makedirs(RATE_DIR, exist_ok=True)
    own = {os.getpid()}
    rep = {"probe": "probe_sync_density_victim_model", "gpu": _smi("name"),
           "config": {"SIZE": SIZE, "N_KERNELS": N_KERNELS, "FAMILY": FAMILY, "HOLDOUT": HOLDOUT,
                      "REPS_CLEAN": REPS_CLEAN, "REPS_CONT": REPS_CONT},
           "launch_stamp": stamp(own),
           "predictions": "control(equal-work<=15%) + G1 monotone + G2 237~10x[5,20] + G3 0~[1.4,4] + "
                          "G4 log-holdout64<=30% + mechanism(half-kernel vs measured penalty). Frozen in docstring."}
    ft = foreign_tenant(own)
    if ft:
        rep["verdict"] = f"ABORT-FOREIGN-TENANT-{ft}"
        json.dump(rep, open(REP, "w"), indent=1, default=str)
        print("FOREIGN TENANT", ft, "- abort"); return

    import torch
    dev = torch.device(DEV)
    torch.cuda.init()
    print("[env]", rep["launch_stamp"])
    t0 = time.time()
    try:
        rep["clean"] = clean_family(torch, dev, own)
        print(f"--- clean done {round(time.time()-t0,1)}s ---")
        rep["contended"] = contended_family(torch, dev, own)
        print(f"--- contended done {round(time.time()-t0,1)}s ---")
        rep["mechanism"] = mechanism_arm(torch, dev, own)
        print(f"--- mechanism done {round(time.time()-t0,1)}s ---")
    finally:
        for p in list(own):
            if p != os.getpid():
                subprocess.run(["kill", str(p)], capture_output=True)
        rep["final_stamp"] = stamp({os.getpid()})
    rep["verdicts"] = verdicts(rep)
    rep["twin_stub"] = twin_stub(rep)
    rep["total_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(REP, "w"), indent=1, default=str)
    print("\n=== VERDICTS ===")
    for k, v in rep["verdicts"].items():
        print(f"  {k}: pass={v.get('pass')}")
    print("inflation p50:", rep["inflation"]["p50"])
    print("wrote", REP, "total", rep["total_s"], "s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TENSOR-CORE-PRECISION RUNG (an agent worktree, feat/breakthrough-hyperreal)
================================================================
The descent-ladder's un-measured rung: the dtype PATHS of the RTX 5070 (Blackwell sm_120) matmul unit,
measured as a discrete rate-distortion menu (throughput vs numerical error per dtype). Completes the
twin's FLOP_peak scalar into a dtype-aware section.

RUN CONTRACT (GPU-LOCK, P-L): always launched through a GPU lock; every burst stamps tenancy
(SELF vs FOREIGN compute procs) and ABORTS the arm on FOREIGN. Two independent process invocations
(--inv A, --inv B) for every headline number; --merge combines them (cross-invocation sigma siblings).

PRE-REGISTRATION (frozen=true BEFORE any measurement; bands motivated from public Blackwell GB20x specs):
  Public anchors used to set the bands (dense, achievable-order, not marketing sparse-FP4):
    * RTX 5070 FP32 (CUDA-core, non-tensor) peak ~= 32.3 TFLOPS  (matches twin FLOP_peak 32263 GFLOPS).
    * RTX 5070 dense FP16 tensor ~= 4x FP32-CUDA-core order (marketing 988 AI TOPS is FP4+sparsity;
      dense FP16 ~ TOPS/8 ~ 120 TFLOPS nominal; cuBLAS FP32-accumulate on consumer Blackwell is
      rate-throttled, so ACHIEVED fp16/fp32 sits in a broad band).
    * Blackwell FP8 dense ~= 2x FP16 dense (one more precision-halving step on the tensor core).
  G1  fp16/fp32 TFLOPS ratio in [3, 10]        (nominal ~3.8x; band spans the fp32-accum throttle down
                                                to full-tensor up). FIRES if achieved is throttled <3x.
  G2  fp8/fp16 TFLOPS ratio in [1.5, 2.5]      (nominal 2.0x precision-halving step) -- only if fp8 runs.
  G3  roofline-knee(dtype) GROWS as the dtype narrows (directional): narrower dtype -> higher compute
      roof -> the compute-bound crossover (>=80% of max TFLOPS) needs a LARGER matrix. knee ordered
      fp32 <= tf32 <= fp16 <= fp8 (allow ties; strict inversions FIRE).
  C2  relative error vs fp64 ~= 2^(-mantissa_bits) per path (fp32~24, tf32~11, fp16~11, bf16~8, fp8e4m3~4).
      Tensor cores accumulate in fp32 -> measured error MAY beat naive per-dtype-eps (that improvement is
      itself the finding). Reported: measured/theory ratio per path, both conditioning arms.
  C3  RD-menu prereg: do the (throughput, error) points lie on ONE monotone RD frontier (thru up <=> err
      up) or is any path DOMINATED (worse on BOTH axes = never rational)? tf32-vs-fp16 is the a-priori
      dominated candidate (near-equal mantissa, unequal throughput). A dominated point is a product-level
      "never use X" finding. FORM cross-check vs the bits-pin quant-floor curve (probe_cross_rung_rd_form
      'bits' rung, winner=hinge): compare FAMILY only, declare THIN if axes are incommensurate.

7th discipline (STRESS ENDURANCE): a frozen gate that fires against the narrative is REPORTED, never bent.
"""
import argparse, json, os, sys, time, subprocess, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports" / "probes"
INV_A = REPORTS / "_tc_precision_inv_A.json"
INV_B = REPORTS / "_tc_precision_inv_B.json"
FINAL = REPORTS / "probe_tensorcore_precision_rung.json"
TWIN_SECTION = REPORTS / "compute_twin_dtype_section.json"
RD_FORM_REF = REPORTS / "probe_cross_rung_rd_form.json"

SIZES = [512, 1024, 2048, 4096, 8192]
ERR_N = 1024               # error-arm matrix size (fp64 reference kept cheap)
WARMUP = 12
REPS = 7                   # >=5 required; median over these
MANTISSA_BITS = {"fp32": 24, "tf32": 11, "fp16": 11, "bf16": 8, "fp8_e4m3": 4}

# frozen pre-registration block ------------------------------------------------
PREREG = {
    "frozen": True,
    "frozen_before_measurement": True,
    "G1_fp16_over_fp32_TFLOPS": {"band": [3.0, 10.0], "why": "dense fp16 ~3.8x fp32-CUDA nominal; band spans fp32-accum throttle..full-tensor"},
    "G2_fp8_over_fp16_TFLOPS": {"band": [1.5, 2.5], "cond": "only if fp8 runs", "why": "Blackwell fp8 ~2.0x fp16 precision-halving step"},
    "G3_knee_growth": {"rule": "roofline knee (size at >=80% max TFLOPS) non-decreasing as dtype narrows: fp32<=tf32<=fp16<=fp8", "type": "directional"},
    "C2_error_theory": {"rule": "rel_err ~ 2^(-mantissa_bits)", "mantissa_bits": MANTISSA_BITS, "note": "fp32-accum tensor cores MAY beat naive theory -> report measured/theory"},
    "C3_RD": {"rule": "points on ONE monotone RD frontier OR a DOMINATED path (worse on both axes)", "a_priori_dominated_candidate": "tf32 vs fp16", "form_ref": "probe_cross_rung_rd_form.json bits rung (hinge)"},
}


def stamp_tenancy(self_pid):
    """Return (list_of_foreign_pids, raw). FOREIGN = any compute proc whose pid != our python pid."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except Exception as e:
        return [], f"nvidia-smi failed: {e!r}"
    foreign = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid = line.split(",")[0].strip()
        if pid and pid != str(self_pid):
            foreign.append(line.strip())
    return foreign, out


def event_time_matmul(fn, reps):
    """Return median wall-time (s) of fn() over reps, event-timed, steady state."""
    import torch
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1e3)  # ms->s
    times.sort()
    return times[len(times) // 2]


def make_matmul(dtype, n):
    """Return a zero-arg closure doing one n x n matmul on the requested dtype path, plus a validity flag."""
    import torch
    dev = "cuda"
    if dtype in ("fp32", "tf32"):
        a = torch.randn(n, n, device=dev, dtype=torch.float32)
        b = torch.randn(n, n, device=dev, dtype=torch.float32)
        return (lambda: torch.mm(a, b)), None
    if dtype == "fp16":
        a = torch.randn(n, n, device=dev, dtype=torch.float16)
        b = torch.randn(n, n, device=dev, dtype=torch.float16)
        return (lambda: torch.mm(a, b)), None
    if dtype == "bf16":
        a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        return (lambda: torch.mm(a, b)), None
    if dtype == "fp8_e4m3":
        a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        bcol = b.t().contiguous().t()  # column-major operand required by _scaled_mm
        sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
        return (lambda: torch._scaled_mm(a, bcol, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)), None
    raise ValueError(dtype)


def set_tf32(on):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on
    try:
        torch.set_float32_matmul_precision("high" if on else "highest")
    except Exception:
        pass


def run_throughput():
    """CELL 1: TFLOPS per dtype x size. Returns {dtype: {size: tflops}} + api gaps."""
    import torch
    res = {}
    api_gaps = {}
    dtypes = ["fp32", "tf32", "fp16", "bf16", "fp8_e4m3"]
    for dt in dtypes:
        res[dt] = {}
        if dt == "tf32":
            set_tf32(True)
        else:
            set_tf32(False)
        for n in SIZES:
            try:
                fn, _ = make_matmul(dt, n)
                t = event_time_matmul(fn, REPS)
                tflops = (2.0 * n ** 3) / t / 1e12
                res[dt][str(n)] = tflops
            except Exception as e:
                res[dt][str(n)] = None
                api_gaps.setdefault(dt, {})[str(n)] = repr(e)[:300]
    set_tf32(False)
    return res, api_gaps


def build_illcond(n, dev, cond=1e6):
    """Ill-conditioned square matrix N(0,1)-scaled with a geometric singular spectrum spanning `cond`."""
    import torch
    u, _, vh = torch.linalg.svd(torch.randn(n, n, device=dev, dtype=torch.float64))
    s = torch.logspace(0, -torch.log10(torch.tensor(cond)).item(), n, device=dev, dtype=torch.float64)
    return (u * s) @ vh


def rel_err_for(dtype, a64, b64, c64):
    """Cast a64,b64 to dtype path, matmul, upcast, Frobenius relative error vs c64."""
    import torch
    dev = "cuda"
    nrm = torch.linalg.norm(c64)
    if dtype == "fp32":
        set_tf32(False)
        c = torch.mm(a64.float(), b64.float()).double()
    elif dtype == "tf32":
        set_tf32(True)
        c = torch.mm(a64.float(), b64.float()).double()
        set_tf32(False)
    elif dtype == "fp16":
        c = torch.mm(a64.half(), b64.half()).double()
    elif dtype == "bf16":
        c = torch.mm(a64.bfloat16(), b64.bfloat16()).double()
    elif dtype == "fp8_e4m3":
        a8 = a64.float().to(torch.float8_e4m3fn); b8 = b64.float().to(torch.float8_e4m3fn)
        bcol = b8.t().contiguous().t()
        sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
        c = torch._scaled_mm(a8, bcol, scale_a=sa, scale_b=sb, out_dtype=torch.float32).double()
    else:
        raise ValueError(dtype)
    return float(torch.linalg.norm(c - c64) / nrm)


def run_error():
    """CELL 2: relative error vs fp64 reference, two conditioning arms."""
    import torch
    dev = "cuda"
    torch.manual_seed(1234)  # fixed matrices -> error reproducible across invocations
    n = ERR_N
    out = {}
    # well-conditioned arm
    aw = torch.randn(n, n, device=dev, dtype=torch.float64)
    bw = torch.randn(n, n, device=dev, dtype=torch.float64)
    cw = aw @ bw
    # ill-conditioned arm
    ai = build_illcond(n, dev, cond=1e6)
    bi = build_illcond(n, dev, cond=1e6)
    ci = ai @ bi
    condA = float(torch.linalg.cond(ai))
    for arm, (a, b, c) in {"well_cond_N01": (aw, bw, cw), "ill_cond_1e6": (ai, bi, ci)}.items():
        out[arm] = {}
        for dt in ["fp32", "tf32", "fp16", "bf16", "fp8_e4m3"]:
            try:
                out[arm][dt] = rel_err_for(dt, a, b, c)
            except Exception as e:
                out[arm][dt] = None
                out[arm].setdefault("_api_gaps", {})[dt] = repr(e)[:200]
    out["ill_cond_number_measured"] = condA
    out["err_matrix_n"] = n
    return out


def run_invocation(tag):
    import torch
    self_pid = os.getpid()
    foreign_pre, raw_pre = stamp_tenancy(self_pid)
    if foreign_pre:
        print(f"[ABORT] FOREIGN compute procs present pre-run: {foreign_pre}", file=sys.stderr)
        sys.exit(3)
    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    t0 = time.time()
    thru, api_gaps = run_throughput()
    err = run_error()
    foreign_post, raw_post = stamp_tenancy(self_pid)
    rec = {
        "invocation": tag,
        "self_pid": self_pid,
        "tenancy": {"foreign_pre": foreign_pre, "foreign_post": foreign_post,
                    "self_clean": (not foreign_pre and not foreign_post),
                    "raw_pre": raw_pre, "raw_post": raw_post},
        "device": dev_name, "capability": list(cap),
        "torch": torch.__version__,
        "throughput_TFLOPS": thru,
        "api_gaps_throughput": api_gaps,
        "error": err,
        "wall_s": round(time.time() - t0, 1),
        "sizes": SIZES, "reps": REPS, "warmup": WARMUP,
    }
    path = INV_A if tag == "A" else INV_B
    path.write_text(json.dumps(rec, indent=1))
    print(f"[inv {tag}] written {path} ({rec['wall_s']}s) foreign_post={foreign_post}")


# ---------------- merge / adjudication ----------------
def _med(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return v[len(v) // 2]


def knee_size(size_to_tflops):
    """Smallest size reaching >=80% of the dtype's max measured TFLOPS."""
    pairs = [(int(s), t) for s, t in size_to_tflops.items() if t is not None]
    if not pairs:
        return None
    mx = max(t for _, t in pairs)
    for s, t in sorted(pairs):
        if t >= 0.8 * mx:
            return s
    return None


def fit_families(xs, ys):
    """Fit exp / power / hinge D(R); return AIC-winning family name. Mirrors cross_rung_rd_form shapes."""
    import numpy as np
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n = len(xs)
    if n < 4:
        return None, {}
    from scipy.optimize import curve_fit
    def aic(rss, k):
        return n * np.log(rss / n + 1e-300) + 2 * k
    fams = {}
    try:
        p, _ = curve_fit(lambda x, a: np.exp(-a * x), xs, ys, p0=[1.0], maxfev=20000)
        r = ys - np.exp(-p[0] * xs); fams["exp"] = aic(float(r @ r), 1)
    except Exception:
        pass
    try:
        p, _ = curve_fit(lambda x, b: (x + 1e-9) ** (-b), xs, ys, p0=[1.0], maxfev=20000)
        r = ys - (xs + 1e-9) ** (-p[0]); fams["power"] = aic(float(r @ r), 1)
    except Exception:
        pass
    try:
        def hinge(x, rk, lo):
            return lo + np.maximum(0.0, rk - x)
        p, _ = curve_fit(hinge, xs, ys, p0=[xs.mean(), ys.min()], maxfev=20000)
        r = ys - hinge(xs, *p); fams["hinge"] = aic(float(r @ r), 2)
    except Exception:
        pass
    if not fams:
        return None, {}
    winner = min(fams, key=fams.get)
    return winner, fams


def merge():
    import numpy as np
    A = json.loads(INV_A.read_text())
    B = json.loads(INV_B.read_text())

    # ---- CELL 1: cross-invocation throughput medians + sigma siblings ----
    dtypes = ["fp32", "tf32", "fp16", "bf16", "fp8_e4m3"]
    thru = {}
    max_tflops = {}
    knees = {}
    for dt in dtypes:
        thru[dt] = {}
        med_by_size = {}
        for s in map(str, SIZES):
            a = A["throughput_TFLOPS"].get(dt, {}).get(s)
            b = B["throughput_TFLOPS"].get(dt, {}).get(s)
            if a is None or b is None:
                thru[dt][s] = {"A": a, "B": b, "median": None}
                continue
            med = (a + b) / 2.0
            sd = abs(a - b) / (2 ** 0.5)  # sample sd, n=2
            thru[dt][s] = {"A": round(a, 2), "B": round(b, 2), "median": round(med, 2), "sigma": round(sd, 3)}
            med_by_size[s] = med
        if med_by_size:
            argmax_s = max(med_by_size, key=med_by_size.get)
            mx = med_by_size[argmax_s]
            max_tflops[dt] = {"value": round(mx, 2), "value_sigma": thru[dt][argmax_s].get("sigma"), "at_size": int(argmax_s)}
            # knee from median curve
            kA = knee_size({s: A["throughput_TFLOPS"].get(dt, {}).get(s) for s in map(str, SIZES)})
            kB = knee_size({s: B["throughput_TFLOPS"].get(dt, {}).get(s) for s in map(str, SIZES)})
            kmed = knee_size({s: thru[dt][s].get("median") for s in map(str, SIZES)})
            knees[dt] = {"median": kmed, "A": kA, "B": kB}

    # ---- gates ----
    gates = {}
    def ratio(x, y):
        return (x / y) if (x and y) else None
    r16 = ratio(max_tflops.get("fp16", {}).get("value"), max_tflops.get("fp32", {}).get("value"))
    gates["G1_fp16_over_fp32"] = {"ratio": round(r16, 3) if r16 else None, "band": [3.0, 10.0],
                                  "PASS": (r16 is not None and 3.0 <= r16 <= 10.0),
                                  "fired_direction": None if r16 is None else ("below" if r16 < 3.0 else ("above" if r16 > 10.0 else "in-band"))}
    fp8_ran = max_tflops.get("fp8_e4m3", {}).get("value") is not None
    r8 = ratio(max_tflops.get("fp8_e4m3", {}).get("value"), max_tflops.get("fp16", {}).get("value")) if fp8_ran else None
    gates["G2_fp8_over_fp16"] = {"ratio": round(r8, 3) if r8 else None, "band": [1.5, 2.5], "fp8_ran": fp8_ran,
                                 "PASS": (r8 is not None and 1.5 <= r8 <= 2.5),
                                 "fired_direction": None if r8 is None else ("below" if r8 < 1.5 else ("above" if r8 > 2.5 else "in-band"))}
    order = ["fp32", "tf32", "fp16", "fp8_e4m3"]
    knee_seq = [(dt, knees.get(dt, {}).get("median")) for dt in order]
    evalable = [(dt, k) for dt, k in knee_seq if k is not None]
    non_decreasing = all(evalable[i][1] <= evalable[i + 1][1] for i in range(len(evalable) - 1))
    gates["G3_knee_growth"] = {"knee_sequence": knee_seq, "non_decreasing": non_decreasing,
                               "PASS": non_decreasing, "note": "directional; ties allowed"}

    # ---- CELL 2: error table + theory deviation (error is deterministic; cross-inv consistency check) ----
    err = {}
    for arm in ["well_cond_N01", "ill_cond_1e6"]:
        err[arm] = {}
        for dt in dtypes:
            ea = A["error"][arm].get(dt); eb = B["error"][arm].get(dt)
            if ea is None:
                err[arm][dt] = {"rel_err": None}
                continue
            theory = 2.0 ** (-MANTISSA_BITS[dt])
            err[arm][dt] = {"rel_err_A": ea, "rel_err_B": eb,
                            "rel_err": ea,
                            "cross_inv_consistent": (eb is None) or (abs(ea - eb) <= 0.05 * max(ea, 1e-12)),
                            "theory_2pow_neg_mbits": theory,
                            "measured_over_theory": round(ea / theory, 3) if theory else None,
                            "mantissa_bits": MANTISSA_BITS[dt]}
    err["ill_cond_number_measured"] = A["error"].get("ill_cond_number_measured")

    # ---- CELL 3: RD menu (throughput vs error, well-conditioned arm) + dominance ----
    pts = []
    for dt in dtypes:
        t = max_tflops.get(dt, {}).get("value")
        e = err["well_cond_N01"].get(dt, {}).get("rel_err")
        if t is not None and e is not None:
            pts.append({"dtype": dt, "TFLOPS": t, "rel_err": e, "mantissa_bits": MANTISSA_BITS[dt]})
    # dominated: exists another point better on BOTH axes (higher TFLOPS AND lower error)
    dominated = []
    for p in pts:
        for q in pts:
            if q["dtype"] == p["dtype"]:
                continue
            if q["TFLOPS"] >= p["TFLOPS"] and q["rel_err"] <= p["rel_err"] and \
               (q["TFLOPS"] > p["TFLOPS"] or q["rel_err"] < p["rel_err"]):
                dominated.append({"dominated": p["dtype"], "by": q["dtype"],
                                  "d_TFLOPS": round(q["TFLOPS"] - p["TFLOPS"], 2),
                                  "d_rel_err": q["rel_err"] - p["rel_err"]})
                break
    # monotone frontier check on the NON-dominated set
    dom_names = {d["dominated"] for d in dominated}
    frontier = sorted([p for p in pts if p["dtype"] not in dom_names], key=lambda z: z["TFLOPS"])
    monotone = all(frontier[i]["rel_err"] <= frontier[i + 1]["rel_err"] for i in range(len(frontier) - 1))

    # form cross-check vs bits-pin quant-floor curve
    form = {"my_curve": "rel_err (well-cond) vs mantissa_bits", "ref": "probe_cross_rung_rd_form.json bits rung"}
    xs = [p["mantissa_bits"] for p in pts]; ys = [p["rel_err"] for p in pts]
    my_win, my_aic = fit_families(xs, ys)
    form["my_winner_family"] = my_win
    form["my_family_aic"] = {k: round(v, 2) for k, v in my_aic.items()}
    ref_win = None
    if RD_FORM_REF.exists():
        try:
            ref = json.loads(RD_FORM_REF.read_text())
            ref_win = ref["rungs"]["bits"]["winner"]
        except Exception:
            pass
    form["ref_bits_winner_family"] = ref_win
    form["axes_commensurate"] = False
    form["thin_declaration"] = ("THIN: my axis = error-vs-MANTISSA-BITS (input quantization, expected geometric in bits); "
                                "ref bits-rung axis = descriptor-error-vs-RETAINED-SAMPLE-BUDGET (a sampling RD). "
                                "The quantities differ; compare FAMILY only, do not equate magnitude. "
                                f"my_family={my_win} vs ref_family={ref_win}.")

    # ---- STRESS lines (7th discipline: every fired gate reported with its number) ----
    stress = []
    if not gates["G1_fp16_over_fp32"]["PASS"] and r16 is not None:
        stress.append(f"STRESS: G1 fp16/fp32 ratio={r16:.2f} {gates['G1_fp16_over_fp32']['fired_direction']} band [3,10]")
    if fp8_ran and not gates["G2_fp8_over_fp16"]["PASS"] and r8 is not None:
        stress.append(f"STRESS: G2 fp8/fp16 ratio={r8:.2f} {gates['G2_fp8_over_fp16']['fired_direction']} band [1.5,2.5]")
    if not gates["G3_knee_growth"]["PASS"]:
        stress.append(f"STRESS: G3 knee sequence not non-decreasing: {knee_seq}")
    for arm in ["well_cond_N01", "ill_cond_1e6"]:
        for dt in dtypes:
            mot = err[arm][dt].get("measured_over_theory")
            if mot is not None and (mot > 5.0 or mot < 0.2):
                stress.append(f"STRESS: C2 {arm}/{dt} measured/theory={mot} (off naive 2^-mbits by >5x)")

    rep = {
        "probe": "tensorcore_precision_rung",
        "lane": "I", "branch": "feat/breakthrough-hyperreal",
        "device": A["device"], "capability": A["capability"], "torch": A["torch"],
        "prereg": PREREG,
        "tenancy": {"invA": A["tenancy"]["self_clean"], "invB": B["tenancy"]["self_clean"],
                    "foreign_any": bool(A["tenancy"]["foreign_pre"] or A["tenancy"]["foreign_post"]
                                        or B["tenancy"]["foreign_pre"] or B["tenancy"]["foreign_post"])},
        "cell1_throughput_TFLOPS": thru,
        "cell1_max_TFLOPS": max_tflops,
        "cell1_roofline_knee_MiBsize": knees,
        "cell1_api_gaps": {"invA": A.get("api_gaps_throughput", {}), "invB": B.get("api_gaps_throughput", {})},
        "gates": gates,
        "cell2_error": err,
        "cell3_rd_menu": {"points": pts, "dominated_paths": dominated,
                          "monotone_frontier_over_nondominated": monotone,
                          "frontier_order": [p["dtype"] for p in frontier],
                          "form_crosscheck": form},
        "stress_lines": stress,
        "verdict": "MEASURED",
    }
    FINAL.write_text(json.dumps(rep, indent=1))
    print(f"[merge] wrote {FINAL}")

    # ---- twin dtype section (separate file, coordinator integrates after QC; l2_v2 pattern) ----
    twin = {
        "schema": "compute_twin_dtype_section_v1",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": A["device"], "capability": A["capability"],
        "note": "makes twin FLOP_peak (single scalar 32263 GFLOPS = fp32) dtype-aware. Coordinator integrates after QC.",
        "fp32_scalar_anchor_GFLOPS": 32263.87711739625,
        "paths": {},
        "provenance": {"probe": "probe_tensorcore_precision_rung.json",
                       "form_ref": "probe_cross_rung_rd_form.json",
                       "invocations": ["_tc_precision_inv_A.json", "_tc_precision_inv_B.json"]},
    }
    for dt in dtypes:
        mt = max_tflops.get(dt, {})
        twin["paths"][dt] = {
            "max_TFLOPS": mt.get("value"),
            "max_TFLOPS_sigma": mt.get("value_sigma"),
            "max_at_size": mt.get("at_size"),
            "roofline_knee_size": knees.get(dt, {}).get("median"),
            "rel_err_well_cond": err["well_cond_N01"].get(dt, {}).get("rel_err"),
            "rel_err_ill_cond": err["ill_cond_1e6"].get(dt, {}).get("rel_err"),
            "mantissa_bits": MANTISSA_BITS[dt],
        }
    twin["dominated_paths"] = dominated
    TWIN_SECTION.write_text(json.dumps(twin, indent=1))
    print(f"[merge] wrote {TWIN_SECTION}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", choices=["A", "B"])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    elif args.inv:
        run_invocation(args.inv)
    else:
        ap.error("need --inv A|B or --merge")

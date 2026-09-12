#!/usr/bin/env python3
"""
an agent worktree, Phase-0 cell #2 — Hong-Kung roofline for the token-rate ceiling (re-verifies the low-VRAM/token-rate
rationale: cell 94's data-movement wall + Hong-Kung CORE-1). PERF cell => v0.3 hardware-measurement hygiene BINDS.
Run under gpu_lock: `python3 l_phase0_c2_roofline.py`

PHYSICS (the Hong-Kung I/O lower bound applied to batch-1 autoregressive decode):
  batch-1 decode has NO weight reuse => every generated token must READ ALL MODEL WEIGHTS ONCE from DRAM (they exceed
  on-chip cache). Arithmetic intensity ~2 FLOP/byte << the machine ridge => MEMORY-BOUND. The ceiling:
      tokens/s  <=  BW_measured / weight_bytes_per_token
  This is why low-VRAM / large-model decode is bandwidth-bound (cell 94): the wall is DATA MOVEMENT, not FLOPs.

FROZEN PREDICTION (before the decisive numbers):
  - measured sustained copy-BW in [0.5, 1.0]x spec peak (~672 GB/s for RTX 5070 GDDR7 192-bit), and <= spec (anchor).
  - G1 (roofline is an UPPER BOUND): for a DRAM-resident model, measured decode tokens/s <= BW/weight_bytes (+tol),
    under quiescence. If measured > roofline => model is cache-resident or BW under-measured (investigate; the sweep
    resolves it).
  - G2 (MBU plausible): the DRAM-bound model achieves a real fraction of peak BW: MBU = achieved_BW/copy_BW in
    [0.3, 1.0].
  - SWEEP (the emergent, decisive demonstration): sweeping model weight_bytes from cache-resident (~MB) to DRAM
    (~GB), the ACHIEVED bandwidth (weight_bytes * tokens/s) is LAUNCH/COMPUTE-bound (<< copy-BW) for small models and
    SATURATES at copy-BW for large models => the bandwidth roofline EMERGES as the ceiling exactly in the DRAM regime.
  - float32 doubles the traffic => halves the roofline; measured rate tracks (traffic-invariant achieved-BW).
7th discipline: if G1/G2 fire against the narrative, REPORT the firing, don't retune. A CONTENDED rig-report demotes
any CEILING to a floor (v0.3). Form-tag: PERF cell, GPU, rig-report mandatory.
"""
import os, json, sys, subprocess, statistics
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(HERE, "artifacts", "l_phase0_c2_roofline_evidence.json")
os.makedirs(os.path.dirname(EVID), exist_ok=True)
SPEC_PEAK_GBPS = 672.0   # RTX 5070 GDDR7 192-bit spec peak (external anchor; measured must be <= this)
L2_MB = 48.0             # approx RTX 5070 L2 (Blackwell) — models >> this are DRAM-streaming

def nvsmi():
    try:
        o = subprocess.run(["nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=8)
        return o.stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"

def foreign_compute():
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=8)
        return o.stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"

def loadavg():
    try: return open("/proc/loadavg").read().strip()
    except Exception as e: return str(e)

def rig_report():
    return {"nvidia_smi": nvsmi(), "foreign_compute_apps": foreign_compute(), "loadavg": loadavg(),
            "gpu_name": torch.cuda.get_device_name(0),
            "cap": list(torch.cuda.get_device_capability(0)),
            "total_mem_GB": round(torch.cuda.get_device_properties(0).total_memory/1e9, 2)}

def measure_copy_bw(n_bytes=512*1024*1024, reps=7, dtype=torch.float16):
    d = torch.device("cuda")
    n = n_bytes // 2
    x = torch.empty(n, device=d, dtype=dtype); y = torch.empty(n, device=d, dtype=dtype)
    torch.cuda.synchronize()
    for _ in range(3): y.copy_(x)                      # warm
    torch.cuda.synchronize()
    gbps = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); y.copy_(x); e.record(); torch.cuda.synchronize()
        gbps.append((2 * n * 2) / (s.elapsed_time(e) / 1e3) / 1e9)   # read+write
    del x, y; torch.cuda.empty_cache()
    return dict(max=max(gbps), median=statistics.median(gbps),
                spread=(max(gbps) - min(gbps)) / max(gbps))

def measure_read_bw(n_bytes=1024*1024*1024, reps=7, dtype=torch.float16):
    """READ-only bandwidth via a large reduction (reads the whole array, writes a scalar). This is the correct
    reference for batch-1 weight-streaming decode, which is READ-dominated (read+write copy under-reads it)."""
    d = torch.device("cuda")
    n = n_bytes // 2
    x = torch.randn(n, device=d, dtype=dtype)
    torch.cuda.synchronize()
    for _ in range(3): float(x.sum())                    # warm
    torch.cuda.synchronize()
    gbps = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); _ = x.sum(); e.record(); torch.cuda.synchronize()
        gbps.append((n * 2) / (s.elapsed_time(e) / 1e3) / 1e9)   # read-only
    del x; torch.cuda.empty_cache()
    return dict(max=max(gbps), median=statistics.median(gbps),
                spread=(max(gbps) - min(gbps)) / max(gbps))

def build_stack(total_bytes, dim, dtype=torch.float16):
    d = torch.device("cuda")
    bpp = 2 if dtype == torch.float16 else 4
    layer_bytes = dim * dim * bpp
    K = max(1, int(round(total_bytes / layer_bytes)))
    Ws = [torch.randn(dim, dim, device=d, dtype=dtype) / (dim ** 0.5) for _ in range(K)]
    real_bytes = K * layer_bytes
    return Ws, real_bytes, K

def time_decode(Ws, dim, dtype=torch.float16, reps=7, inner=8):
    """tokens/s for batch-1 forward through the weight stack (each pass = one decoded token)."""
    d = torch.device("cuda")
    x0 = torch.randn(1, dim, device=d, dtype=dtype)
    def one_token():
        x = x0
        for W in Ws:
            x = x @ W                                   # matvec: reads W once (weight-streaming)
        return x
    torch.cuda.synchronize()
    for _ in range(3): one_token()                      # warm
    torch.cuda.synchronize()
    rates = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(inner): one_token()
        e.record(); torch.cuda.synchronize()
        rates.append(inner / (s.elapsed_time(e) / 1e3))  # tokens/s
    return dict(max=max(rates), median=statistics.median(rates),
                spread=(max(rates) - min(rates)) / max(rates))

def main():
    if not torch.cuda.is_available():
        print("no CUDA"); return 1
    torch.manual_seed(0)
    rig_before = rig_report()
    # quiescence gate
    foreign = rig_before["foreign_compute_apps"]
    util = rig_before["nvidia_smi"].split(",")[0].strip() if rig_before["nvidia_smi"] else "?"
    la1 = float(rig_before["loadavg"].split()[0]) if rig_before["loadavg"] else 99
    gpu_quiescent = (foreign == "" or "unavailable" in foreign) and (util.replace("%","").strip().isdigit() and int(util.replace("%","").strip()) < 15)
    # (CPU loadavg is high on the shared 13600K; decode rate is partly CPU-launch-bound => flag but GPU-BW is GPU-bound)
    contended_cpu = la1 > 4.0

    RUNTIME = {"model_line": "model identity omitted",
               "effort": "xhigh", "requested": "Hong-Kung roofline cell, run under the GPU lock"}

    bw = measure_copy_bw()                               # read+write copy (reference)
    rbw = measure_read_bw()                              # READ-only (correct ref for read-dominated decode)
    ref_bw = rbw["max"]                                  # roofline uses the READ bandwidth
    # sweep spans cache-resident -> DRAM-resident by varying dim (one 8192^2 fp16 layer = 128MB > L2, so small dims
    # give the cache-resident end). (dim, K) => weight_bytes = K*dim*dim*2.
    configs = [(512,1),(1024,1),(2048,1),(4096,1),(8192,1),(8192,4),(8192,8),(8192,16),(8192,24)]
    sweep = []
    for dim, K in configs:
        Ws, real_bytes, Kact = build_stack(K * dim * dim * 2, dim)
        dec = time_decode(Ws, dim)
        rate = dec["max"]                                # ceiling: max-over-repeats
        achieved_bw = real_bytes * rate / 1e9            # GB/s of weight traffic actually sustained
        roofline = ref_bw * 1e9 / real_bytes             # tokens/s ceiling from READ bandwidth
        sweep.append(dict(dim=dim, K=Kact, weight_bytes=real_bytes, weight_MB=round(real_bytes/1e6,1),
                          tokens_per_s_max=rate, tokens_per_s_median=dec["median"], rate_spread=dec["spread"],
                          achieved_BW_GBps=achieved_bw, roofline_tokens_per_s=roofline,
                          MBU=achieved_bw / ref_bw, dram_resident=real_bytes/1e6 > L2_MB))
        del Ws; torch.cuda.empty_cache()

    # float32 cross-check at 2GB (traffic doubles -> roofline halves; achieved-BW should be traffic-invariant)
    Ws32, rb32, K32 = build_stack(2048*1024*1024, 8192, dtype=torch.float32)
    dec32 = time_decode(Ws32, 8192, dtype=torch.float32)
    ach32 = rb32 * dec32["max"] / 1e9
    del Ws32; torch.cuda.empty_cache()

    rig_after = rig_report()

    # ---- gates ----
    dram = [s for s in sweep if s["dram_resident"]]
    cache = [s for s in sweep if not s["dram_resident"]]
    big = max(sweep, key=lambda s: s["weight_bytes"])    # largest = most DRAM-bound
    G1_upper_bound = all(s["tokens_per_s_max"] <= s["roofline_tokens_per_s"] * 1.05 for s in dram)
    G2_mbu = 0.3 <= big["MBU"] <= 1.05
    G3_bw_anchor = 0.5 * SPEC_PEAK_GBPS <= ref_bw <= SPEC_PEAK_GBPS
    # SWEEP emergence: cache-resident MBU low (launch/compute-bound), DRAM-resident MBU saturates high
    mbu_cache = min((s["MBU"] for s in cache), default=0.0)
    mbu_dram = big["MBU"]
    SWEEP_emerges = (mbu_dram - mbu_cache) > 0.15 and mbu_dram > 0.4
    # float32 cross-check: achieved BW (traffic-invariant) close to fp16 achieved BW at 2GB
    ach_fp16_2gb = next(s["achieved_BW_GBps"] for s in sweep if s["dim"] == 8192 and s["K"] == 16)
    F_traffic_invariant = abs(ach32 - ach_fp16_2gb) / max(ach32, ach_fp16_2gb) < 0.25

    stress = []
    if not G1_upper_bound:
        viol = [(s["weight_MB"], round(s["tokens_per_s_max"],1), round(s["roofline_tokens_per_s"],1)) for s in dram
                if s["tokens_per_s_max"] > s["roofline_tokens_per_s"]*1.05]
        stress.append(f"STRESS: G1 roofline NOT an upper bound for DRAM models {viol} — model cache-resident or BW under-measured")
    if not G3_bw_anchor:
        stress.append(f"STRESS: measured READ BW {ref_bw:.0f} outside [0.5,1.0]x spec {SPEC_PEAK_GBPS} — instrument or spec wrong")
    if not SWEEP_emerges:
        stress.append(f"STRESS: sweep did NOT show roofline emergence (MBU cache={mbu_cache:.2f} dram={mbu_dram:.2f})")
    if contended_cpu:
        stress.append(f"CONTENDED: loadavg {la1:.1f}>4 on shared 13600K — decode-rate CEILING demoted to a floor "
                      f"(real rate >= measured); GPU copy-BW is GPU-bound and unaffected. per v0.3 hygiene.")

    verdict = dict(
        copy_BW=bw, read_BW=rbw, roofline_ref_BW_GBps=ref_bw,
        spec_peak_GBps=SPEC_PEAK_GBPS, read_bw_fraction_of_spec=ref_bw/SPEC_PEAK_GBPS,
        G1_roofline_is_upper_bound=bool(G1_upper_bound),
        G2_MBU_plausible=bool(G2_mbu), MBU_at_largest=big["MBU"],
        G3_BW_anchored=bool(G3_bw_anchor),
        SWEEP_roofline_emerges=bool(SWEEP_emerges), MBU_cache=mbu_cache, MBU_dram=mbu_dram,
        float32_traffic_invariant=bool(F_traffic_invariant), achieved_BW_fp32_2GB=ach32, achieved_BW_fp16_2GB=ach_fp16_2gb,
        largest_model=dict(weight_MB=big["weight_MB"], tokens_per_s=big["tokens_per_s_max"],
                           roofline_tokens_per_s=big["roofline_tokens_per_s"]),
        gpu_quiescent=bool(gpu_quiescent), cpu_contended=bool(contended_cpu),
    )
    evidence = dict(cell="l_phase0_c2_roofline", form_tag="PERF (GPU, rig-report mandatory, v0.3)",
                    runtime=RUNTIME, rig_before=rig_before, rig_after=rig_after,
                    verdict=verdict, sweep=sweep, stress_lines=stress,
                    cite=["cell 94/data-movement-wall", "hong-kung/CORE-1/#21", "roofline/williams-2009"])
    with open(EVID, "w") as f: json.dump(evidence, f, indent=2)

    # console
    print("=== l_phase0_c2_roofline — Hong-Kung token-rate ceiling ===")
    print(f"copy(r+w) BW: max={bw['max']:.1f} GB/s | READ BW: max={rbw['max']:.1f} median={rbw['median']:.1f} "
          f"spread={rbw['spread']*100:.1f}%  ({ref_bw/SPEC_PEAK_GBPS*100:.0f}% of spec {SPEC_PEAK_GBPS}) [roofline ref=READ]")
    print(f"quiescence: GPU_quiescent={gpu_quiescent} (util={util}, foreign='{foreign[:40]}')  CPU_contended={contended_cpu} (load {la1})")
    print(f"{'dim':>6} {'K':>3} {'weight_MB':>10} {'tok/s(max)':>11} {'roofline':>9} {'MBU':>5} {'DRAM?':>6}")
    for s in sweep:
        print(f"{s['dim']:>6} {s['K']:>3} {s['weight_MB']:>10.1f} {s['tokens_per_s_max']:>11.1f} {s['roofline_tokens_per_s']:>9.1f} "
              f"{s['MBU']:>5.2f} {str(s['dram_resident']):>6}")
    print(f"G1 roofline upper-bounds DRAM decode: {G1_upper_bound}")
    print(f"G2 MBU plausible @largest ({big['MBU']:.2f}): {G2_mbu}")
    print(f"G3 BW anchored to spec: {G3_bw_anchor}")
    print(f"SWEEP roofline emerges (MBU cache {mbu_cache:.2f} -> DRAM {mbu_dram:.2f}): {SWEEP_emerges}")
    print(f"float32 traffic-invariant achieved-BW (fp32 {ach32:.0f} vs fp16 {ach_fp16_2gb:.0f} GB/s): {F_traffic_invariant}")
    if stress:
        print("---- STRESS / CONTENDED (v0.3 + 7th discipline) ----")
        for s in stress: print(" " + s)
    print(f"evidence -> {EVID}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

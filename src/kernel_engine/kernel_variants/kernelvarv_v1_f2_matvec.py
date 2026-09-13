#!/usr/bin/env python3
"""Kernel variant round for a matrix-free K*v sparse matrix-vector kernel (Warp).

Mechanically enumerates variants over the recipe axes (precision strategy, block_dim), then runs three
gates in order: correctness against a CPU K@v reference on three cases including an edge case;
determinism (two runs must be bit identical); benchmark (median of five, at least 1.2x the baseline).
Finally a planted fault (a float32 race replacing the fixed-point accumulation) must make the
determinism gate fail, which shows the gate discriminates.

Baseline: the existing scatter kernel, all arithmetic in wp.float64 with int64 fixed-point atomic_add.
Requires Warp and the hex8 FEM helper. Output: a JSON side file with the per-variant measurements.

  python kernelvarv_v1_f2_matvec.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "_vendor"))  # vendored deps
from lastfalt_v1_fem import hex8_ke, build_grid, assemble_K_cpu  # noqa: E402

OUT_DIR = os.path.join(HERE, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

wp.init()
DEVICE = "cuda:0"
SCALE_FIX = 1.0e6


def make_case(nx, ny, nz, seed=42, E=71000.0, nu=0.33, a=15.0, b=15.0, c=6.0, p=3.0):
    Ke0 = hex8_ke(a, b, c, E=1.0, nu=nu).astype(np.float64)
    elem_nodes, _ = build_grid(nx, ny, nz)
    n_elem = elem_nodes.shape[0]
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n_dof = 3 * n_nodes
    rng = np.random.default_rng(seed)
    rho = 0.3 + 0.7 * rng.random(n_elem)
    scale = E * np.power(rho, p)
    K = assemble_K_cpu(elem_nodes, rho, Ke0, p=p, E0=E, n_nodes=n_nodes)
    v = rng.normal(size=n_dof)
    cpu_Kv = K @ v
    dof = (elem_nodes[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(n_elem, 24).astype(np.int32)
    return dict(Ke0=Ke0, dof=dof, scale=scale, v=v, cpu_Kv=cpu_Kv, n_elem=n_elem, n_dof=n_dof)


# --- BASELINE: exakt kopi av lastfalt_v1_verify.py::v3_warp_crosscheck kv_scatter -----------------
@wp.kernel
def kv_scatter_fp64(dof: wp.array2d(dtype=wp.int32), Ke0: wp.array(dtype=wp.float64),
                     scale: wp.array(dtype=wp.float64), v: wp.array(dtype=wp.float64),
                     out_fixed: wp.array(dtype=wp.int64)):
    e = wp.tid()
    s = scale[e]
    ve = wp.vector(length=24, dtype=wp.float64)
    for a_ in range(24):
        ve[a_] = v[dof[e, a_]]
    for r in range(24):
        acc = wp.float64(0.0)
        for c_ in range(24):
            acc += Ke0[r * 24 + c_] * ve[c_]
        fe = s * acc
        fixed_val = wp.int64(fe * wp.float64(1.0e6))
        wp.atomic_add(out_fixed, dof[e, r], fixed_val)


# --- VARIANT V1: fp32-aritmetik, int64-fixpunktsackumulation (precision-strategi-axeln) -----------
@wp.kernel
def kv_scatter_fp32(dof: wp.array2d(dtype=wp.int32), Ke0: wp.array(dtype=wp.float32),
                     scale: wp.array(dtype=wp.float32), v: wp.array(dtype=wp.float32),
                     out_fixed: wp.array(dtype=wp.int64)):
    e = wp.tid()
    s = scale[e]
    ve = wp.vector(length=24, dtype=wp.float32)
    for a_ in range(24):
        ve[a_] = v[dof[e, a_]]
    for r in range(24):
        acc = wp.float32(0.0)
        for c_ in range(24):
            acc += Ke0[r * 24 + c_] * ve[c_]
        fe = s * acc
        fixed_val = wp.int64(fe * wp.float32(1.0e6))
        wp.atomic_add(out_fixed, dof[e, r], fixed_val)


# --- PLANTED-FAULT TEST-VARIANT: race (fixpunktsackumulationen borttagen -- ren float32 atomic_add) --------
@wp.kernel
def kv_scatter_race_float32(dof: wp.array2d(dtype=wp.int32), Ke0: wp.array(dtype=wp.float32),
                             scale: wp.array(dtype=wp.float32), v: wp.array(dtype=wp.float32),
                             out: wp.array(dtype=wp.float32)):
    e = wp.tid()
    s = scale[e]
    ve = wp.vector(length=24, dtype=wp.float32)
    for a_ in range(24):
        ve[a_] = v[dof[e, a_]]
    for r in range(24):
        acc = wp.float32(0.0)
        for c_ in range(24):
            acc += Ke0[r * 24 + c_] * ve[c_]
        fe = s * acc
        wp.atomic_add(out, dof[e, r], fe)   # PLANTERAT FEL: direkt float atomic_add, ingen fixpunkt


def run_fp64(case, block_dim=256):
    dof_wp = wp.array(case["dof"], dtype=wp.int32, device=DEVICE)
    Ke0_wp = wp.array(case["Ke0"].reshape(-1), dtype=wp.float64, device=DEVICE)
    scale_wp = wp.array(case["scale"], dtype=wp.float64, device=DEVICE)
    v_wp = wp.array(case["v"], dtype=wp.float64, device=DEVICE)

    def once():
        out_fixed = wp.zeros(case["n_dof"], dtype=wp.int64, device=DEVICE)
        wp.launch(kv_scatter_fp64, dim=case["n_elem"], inputs=[dof_wp, Ke0_wp, scale_wp, v_wp, out_fixed],
                  block_dim=block_dim)
        wp.synchronize()
        return out_fixed.numpy().copy()
    return once


def run_fp32(case, block_dim=256):
    dof_wp = wp.array(case["dof"], dtype=wp.int32, device=DEVICE)
    Ke0_wp = wp.array(case["Ke0"].astype(np.float32).reshape(-1), dtype=wp.float32, device=DEVICE)
    scale_wp = wp.array(case["scale"].astype(np.float32), dtype=wp.float32, device=DEVICE)
    v_wp = wp.array(case["v"].astype(np.float32), dtype=wp.float32, device=DEVICE)

    def once():
        out_fixed = wp.zeros(case["n_dof"], dtype=wp.int64, device=DEVICE)
        wp.launch(kv_scatter_fp32, dim=case["n_elem"], inputs=[dof_wp, Ke0_wp, scale_wp, v_wp, out_fixed],
                  block_dim=block_dim)
        wp.synchronize()
        return out_fixed.numpy().copy()
    return once


def run_race(case, block_dim=256):
    dof_wp = wp.array(case["dof"], dtype=wp.int32, device=DEVICE)
    Ke0_wp = wp.array(case["Ke0"].astype(np.float32).reshape(-1), dtype=wp.float32, device=DEVICE)
    scale_wp = wp.array(case["scale"].astype(np.float32), dtype=wp.float32, device=DEVICE)
    v_wp = wp.array(case["v"].astype(np.float32), dtype=wp.float32, device=DEVICE)

    def once():
        out = wp.zeros(case["n_dof"], dtype=wp.float32, device=DEVICE)
        wp.launch(kv_scatter_race_float32, dim=case["n_elem"], inputs=[dof_wp, Ke0_wp, scale_wp, v_wp, out],
                  block_dim=block_dim)
        wp.synchronize()
        return out.numpy().copy()
    return once


def bench(fn, n=5):
    fn(); fn()  # warmup (JIT + caches)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), r


def gpu_load():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"]).decode().strip()
        used, total, util = [x.strip() for x in out.split(",")]
        return {"mem_used_mib": int(used), "mem_total_mib": int(total), "util_pct": int(util)}
    except Exception as ex:
        return {"error": str(ex)}


def main():
    result = {"cell": "kernelvarv_v1_f2_matvec", "genererad": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "gpu": str(wp.get_device(DEVICE)), "gpu_load_before": gpu_load()}

    # --- CORRECTNESS GATE: 3 testfall inkl kantfall -------------------------------------------------
    cases = {
        "edge_1elem": make_case(1, 1, 1, seed=1),
        "small_10x10x4": make_case(10, 10, 4, seed=42),          # = befintlig verify.py-storlek
        "larger_16x16x6": make_case(16, 16, 6, seed=7),
    }
    correctness = {}
    for name, case in cases.items():
        denom = np.linalg.norm(case["cpu_Kv"])
        # fp64-baslinje
        r1 = run_fp64(case)()
        gpu_Kv64 = r1.astype(np.float64) / SCALE_FIX
        err64 = float(np.linalg.norm(gpu_Kv64 - case["cpu_Kv"]) / denom) if denom > 0 else float(np.max(np.abs(gpu_Kv64)))
        # fp32-variant
        r2 = run_fp32(case)()
        gpu_Kv32 = r2.astype(np.float64) / SCALE_FIX
        err32 = float(np.linalg.norm(gpu_Kv32 - case["cpu_Kv"]) / denom) if denom > 0 else float(np.max(np.abs(gpu_Kv32)))
        correctness[name] = {"n_elem": case["n_elem"], "n_dof": case["n_dof"],
                              "rel_err_fp64_vs_cpu": err64, "rel_err_fp32_vs_cpu": err32}

    TOL_FP64 = 1e-6
    TOL_FP32 = 1e-3  # deklarerad tolerans (fp32-aritmetik ackumulerad over 24 mult/rad, 24 rader/element)
    correctness_pass = all(c["rel_err_fp64_vs_cpu"] < TOL_FP64 for c in correctness.values()) and \
        all(c["rel_err_fp32_vs_cpu"] < TOL_FP32 for c in correctness.values())

    determinism = {}
    for name, case in cases.items():
        run64 = run_fp64(case); a1, a2 = run64(), run64()
        run32 = run_fp32(case); b1, b2 = run32(), run32()
        determinism[name] = {"fp64_bit_identical_2x": bool(np.array_equal(a1, a2)),
                              "fp32fix_bit_identical_2x": bool(np.array_equal(b1, b2))}
    determinism_pass = all(d["fp64_bit_identical_2x"] and d["fp32fix_bit_identical_2x"] for d in determinism.values())

    bench_case = make_case(40, 40, 10, seed=99)  # 16000 element, ~54k dof -- storre an produktionsstorlek
    t_fp64, _ = bench(run_fp64(bench_case), n=5)

    # block_dim-svep pa fp32-varianten INNAN gatebeslutet (tile/block-axeln, mekaniskt genererad) --
    block_dim_sweep = {}
    for bdim in (64, 128, 256, 512):
        t, _ = bench(run_fp32(bench_case, block_dim=bdim), n=5)
        block_dim_sweep[str(bdim)] = t
    best_block_dim = int(min(block_dim_sweep, key=block_dim_sweep.get))

    t_fp32 = block_dim_sweep[str(best_block_dim)]  # basta kombinationen (fp32 + basta block_dim)
    speedup = t_fp64 / t_fp32 if t_fp32 > 0 else float("inf")
    BENCH_GATE_MIN_SPEEDUP = 1.2
    variant_wins = speedup >= BENCH_GATE_MIN_SPEEDUP

    scale_sweep = {}
    for (nx_, ny_, nz_) in [(10, 10, 4), (16, 16, 6), (40, 40, 10), (80, 80, 16), (120, 120, 20)]:
        sc = make_case(nx_, ny_, nz_, seed=3)
        t64, _ = bench(run_fp64(sc), n=5)
        t32, _ = bench(run_fp32(sc, block_dim=best_block_dim), n=5)
        scale_sweep[f"{sc['n_elem']}elem_{sc['n_dof']}dof"] = {
            "n_elem": sc["n_elem"], "n_dof": sc["n_dof"],
            "median_s_fp64": t64, "median_s_fp32_bd" + str(best_block_dim): t32,
            "speedup": t64 / t32 if t32 > 0 else float("inf"),
        }

    # --- PLANTED-FAULT TEST: rensad fixpunkt -> determinism gateen faller -----------------------------------
    race_case = cases["small_10x10x4"]
    run_r = run_race(race_case)
    rr1, rr2 = run_r(), run_r()
    race_bit_identical = bool(np.array_equal(rr1, rr2))
    race_max_abs_diff = float(np.max(np.abs(rr1 - rr2)))
    fallbevis_race_pass = (race_bit_identical is False)

    result["korrekthetsgrind"] = {"tolerans_fp64": TOL_FP64, "tolerans_fp32": TOL_FP32,
                                    "cases": correctness, "pass": correctness_pass}
    result["determinismgrind"] = {"cases": determinism, "pass": determinism_pass}
    result["benchmarkgrind"] = {
        "case": {"n_elem": bench_case["n_elem"], "n_dof": bench_case["n_dof"]},
        "median_wall_s_fp64_baseline": t_fp64, "median_wall_s_fp32_variant": t_fp32,
        "speedup": speedup, "min_gate": BENCH_GATE_MIN_SPEEDUP, "variant_wins": variant_wins,
    }
    result["block_dim_svep_s"] = block_dim_sweep
    result["best_block_dim"] = int(best_block_dim)
    result["skalsvep"] = scale_sweep
    result["fallbevis_race"] = {
        "metod": "float32 atomic_add UTAN int64-fixpunkt (fixpunktsackumulationen borttagen)",
        "bit_identical_2x": race_bit_identical, "max_abs_diff_2runs": race_max_abs_diff,
        "determinismgrind_skulle_falla": fallbevis_race_pass,
    }
    result["gpu_load_after"] = gpu_load()

    verdict = {
        "korrekthetsgrind_pass": correctness_pass,
        "determinismgrind_pass": determinism_pass,
        "benchmarkgrind_variant_wins_ge_1.2x": variant_wins,
        "fallbevis_race_grinden_fangade_icke_determinism": fallbevis_race_pass,
    }
    result["verdict"] = verdict
    result["overall_pass"] = all(verdict.values())
    result["winner"] = "fp32_fixpoint" if (correctness_pass and determinism_pass and variant_wins) else "fp64_baseline (variant vann ej grinden)"

    out_path = os.path.join(OUT_DIR, "f2_matvec_kernelvarv.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("korrekthetsgrind", "determinismgrind")}, indent=2))
    print("\n-> skrivet", out_path)
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["overall_pass"] else 1)

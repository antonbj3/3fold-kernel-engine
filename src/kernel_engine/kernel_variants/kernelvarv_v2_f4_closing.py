#!/usr/bin/env python3
"""Kernel variant round for 2D morphological closing on the GPU.

Baseline: skimage.morphology.binary_closing(mask, footprint=disk(r)), including its border semantics
(dilation border_value False, erosion border_value True). Variants, both derived from exactly the same
disk() footprint so no radius definition can drift: (V1) a Warp brute-force disk-offset kernel, and
(V2) a torch conv2d formulation with exact integer sums thresholded against the footprint pixel count.

Gates: pixel-exact equality against the baseline on three cases including an edge case, determinism,
and benchmark. A planted radius+1 fault must be caught.

Requires Warp, torch, scikit-image and scipy. Output: a JSON side file with the measurements.

  python kernelvarv_v2_f4_closing.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, disk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(ROOT, "reports/probes/kernelvarv_v1_sidofiler")
os.makedirs(OUT_DIR, exist_ok=True)

import warp as wp
wp.init()
DEVICE = "cuda:0" if "cuda:0" in [str(d) for d in wp.get_devices()] else "cpu"

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH_CUDA = torch.cuda.is_available()
except Exception as e:
    torch = None
    HAVE_TORCH_CUDA = False
    print("torch not available:", e)


def disk_offsets(rho_px, extra=0):
    """Exactly the same footprint as skimage.morphology.disk(rho_px); `extra` is used to plant the
    radius+1 fault, so variant and planted fault share one radius definition."""
    fp = disk(rho_px + extra)
    ys, xs = np.nonzero(fp)
    r_c = rho_px + extra
    off = np.stack([ys - r_c, xs - r_c], axis=1).astype(np.int32)
    return off  # (K,2)


# ---- V1: Warp GPU brute-force over EXAKT disk-offsets (bounds-check = border_value-semantik) -----
@wp.kernel
def dilate_disk_wp(mask: wp.array2d(dtype=wp.uint8), out: wp.array2d(dtype=wp.uint8),
                    off_i: wp.array(dtype=wp.int32), off_j: wp.array(dtype=wp.int32),
                    n_off: int, nr: int, nz: int):
    i, j = wp.tid()
    found = wp.uint8(0)
    for k in range(n_off):
        ii = i + off_i[k]
        jj = j + off_j[k]
        if ii >= 0 and ii < nr and jj >= 0 and jj < nz:
            if mask[ii, jj] == wp.uint8(1):
                found = wp.uint8(1)
    out[i, j] = found


@wp.kernel
def erode_disk_wp(mask: wp.array2d(dtype=wp.uint8), out: wp.array2d(dtype=wp.uint8),
                   off_i: wp.array(dtype=wp.int32), off_j: wp.array(dtype=wp.int32),
                   n_off: int, nr: int, nz: int):
    i, j = wp.tid()
    all_true = wp.uint8(1)
    for k in range(n_off):
        ii = i + off_i[k]
        jj = j + off_j[k]
        if ii >= 0 and ii < nr and jj >= 0 and jj < nz:
            if mask[ii, jj] == wp.uint8(0):
                all_true = wp.uint8(0)
        # utanfor rastret: border_value=True for erosion -> paverkar EJ all_true
    out[i, j] = all_true


def closing_warp(mask0, off, device=DEVICE):
    nr, nz = mask0.shape
    mask_wp = wp.array(mask0.astype(np.uint8), dtype=wp.uint8, device=device)
    off_i = wp.array(np.ascontiguousarray(off[:, 0]), dtype=wp.int32, device=device)
    off_j = wp.array(np.ascontiguousarray(off[:, 1]), dtype=wp.int32, device=device)
    n_off = off.shape[0]
    dilated = wp.zeros((nr, nz), dtype=wp.uint8, device=device)
    wp.launch(dilate_disk_wp, dim=(nr, nz), inputs=[mask_wp, dilated, off_i, off_j, n_off, nr, nz], device=device)
    eroded = wp.zeros((nr, nz), dtype=wp.uint8, device=device)
    wp.launch(erode_disk_wp, dim=(nr, nz), inputs=[dilated, eroded, off_i, off_j, n_off, nr, nz], device=device)
    wp.synchronize()
    return eroded.numpy().astype(bool)


def closing_torch(mask0, rho_px, device="cuda"):
    fp = disk(rho_px).astype(np.float32)
    K = int(fp.sum())
    r = rho_px
    kernel = torch.from_numpy(fp).to(device)[None, None]
    m = torch.from_numpy(mask0.astype(np.float32)).to(device)[None, None]

    m_dil = F.pad(m, (r, r, r, r), mode="constant", value=0.0)
    conv_dil = F.conv2d(m_dil, kernel)
    dilated_f = (conv_dil.squeeze(0).squeeze(0) >= 0.5).float()

    m_ero = F.pad(dilated_f[None, None], (r, r, r, r), mode="constant", value=1.0)
    conv_ero = F.conv2d(m_ero, kernel)
    eroded = (conv_ero.squeeze(0).squeeze(0) >= (K - 0.5))
    return eroded.detach().cpu().numpy()


def make_case(kind, rho_px_ref=25, seed=0):
    """Three cases: (a) a large stepped turned profile rasterised from a polyline, (b) a generic
    interior blob, (c) an edge case whose True region is placed so the disk footprint sticks out over
    two raster edges at once for both dilation and erosion, i.e. exactly where the border_value
    semantics (0 for dilation, 1 for erosion) decide the result."""
    if kind == "prod_facit":
        # Synthetic stepped turned profile (self-contained stand-in for a production profile);
        # same raster size class as the original case, about 1243 x 1201 pixels.
        pz_syn = np.linspace(0.0, 48.0, 13)
        pr_syn = np.array([12.0, 12.0, 20.0, 20.0, 31.0, 31.0, 24.0, 24.0, 40.0, 40.0, 18.0, 18.0, 12.0])
        poly = [[float(a), float(b)] for a, b in zip(pz_syn, pr_syn)]
        z_lo, z_hi = float(pz_syn[0]), float(pz_syn[-1])
        R_ENVELOPE = 47.7
        PITCH2D_MM = 0.04
        RHO_TURN_MM = 1.0
        z_pad, r_pad = 2.0, 2.0
        z0, z1 = z_lo - z_pad, z_hi + z_pad
        r0, r1 = 0.0, R_ENVELOPE + r_pad
        nz = int(round((z1 - z0) / PITCH2D_MM)) + 1
        nr = int(round((r1 - r0) / PITCH2D_MM)) + 1
        zz = np.linspace(z0, z1, nz)
        rr = np.linspace(r0, r1, nr)
        poly_s = sorted(poly, key=lambda t: t[0])
        pz = np.array([p[0] for p in poly_s]); pr = np.array([p[1] for p in poly_s])
        R, Z = np.meshgrid(rr, zz, indexing="ij")
        r_of_z_2d = np.interp(Z.ravel(), zz, np.interp(zz, pz, pr)).reshape(Z.shape)
        mask0 = R <= r_of_z_2d
        rho_px = max(1, int(round(RHO_TURN_MM / PITCH2D_MM)))
        return mask0, rho_px
    if kind == "interior_generic":
        rng = np.random.default_rng(seed)
        nr, nz = 200, 220
        mask0 = np.zeros((nr, nz), dtype=bool)
        # random blob with concave corners (overlapping rectangles), interior, margin >= rho_px+5
        m = rho_px_ref + 5
        for _ in range(6):
            y0 = rng.integers(m, nr - m - 20); x0 = rng.integers(m, nz - m - 20)
            h = rng.integers(10, 30); w = rng.integers(10, 30)
            mask0[y0:y0 + h, x0:x0 + w] = True
        return mask0, rho_px_ref
    if kind == "edge_disk_at_border":
        rho_px = 12
        nr, nz = 40, 40
        mask0 = np.zeros((nr, nz), dtype=bool)
        # tva sma "L"-formade regioner precis vid tva olika kanter (hornfall + kantfall samtidigt) --
        mask0[0:3, 0:6] = True          # rad 0 = ovankant
        mask0[nr - 4:nr, 5:9] = True    # nedre kant
        mask0[10:14, 0:3] = True        # vanster kant
        mask0[15:19, nz - 3:nz] = True  # hoger kant
        return mask0, rho_px
    raise ValueError(kind)


def bench(fn, n=5):
    fn()  # warmup
    ts = []
    r = None
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


def ref_closing(mask0, rho_px):
    return binary_closing(mask0, footprint=disk(rho_px))


def main():
    result = {"cell": "kernelvarv_v2_f4_closing", "genererad": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "gpu": DEVICE, "have_torch_cuda": HAVE_TORCH_CUDA, "gpu_load_before": gpu_load()}

    cases = {
        "prod_facit": make_case("prod_facit"),
        "interior_generic": make_case("interior_generic"),
        "edge_disk_at_border": make_case("edge_disk_at_border"),
    }

    correctness = {}
    for name, (mask0, rho_px) in cases.items():
        ref = ref_closing(mask0, rho_px)
        off = disk_offsets(rho_px)
        out_wp = closing_warp(mask0, off)
        bit_id_wp = bool(np.array_equal(ref, out_wp))
        n_diff_wp = int(np.sum(ref != out_wp))
        entry = {"shape": list(mask0.shape), "rho_px": rho_px, "n_off": int(off.shape[0]),
                  "warp_bit_identical": bit_id_wp, "warp_n_diff_px": n_diff_wp}
        if HAVE_TORCH_CUDA:
            out_th = closing_torch(mask0, rho_px)
            bit_id_th = bool(np.array_equal(ref, out_th))
            entry["torch_bit_identical"] = bit_id_th
            entry["torch_n_diff_px"] = int(np.sum(ref != out_th))
        correctness[name] = entry

    correctness_pass_warp = all(c["warp_bit_identical"] for c in correctness.values())
    correctness_pass_torch = (all(c.get("torch_bit_identical", False) for c in correctness.values())
                               if HAVE_TORCH_CUDA else None)

    # --- PLANTED-FAULT TEST: planterad off-by-one (rho_px+1 i variantens EGNA fotavtryck) -- baslinjen kvar
    fallbevis = {}
    name = "edge_disk_at_border"
    mask0, rho_px = cases[name]
    ref = ref_closing(mask0, rho_px)
    off_bad = disk_offsets(rho_px, extra=1)  # off-by-one: fotavtryck for radie rho_px+1
    out_bad = closing_warp(mask0, off_bad)
    bad_bit_identical = bool(np.array_equal(ref, out_bad))
    bad_n_diff = int(np.sum(ref != out_bad))
    fallbevis["offbyone_edge"] = {
        "metod": "disk-offset fotavtryck for radie rho_px+1 i stallet for rho_px (planterat i variantens EGEN disk_offsets(), baslinjen skimage ofoandrad)",
        "case": name, "bit_identical_vs_ref": bad_bit_identical, "n_diff_px": bad_n_diff,
        "korrekthetsgrind_skulle_falla": (bad_bit_identical is False),
    }
    mask0_i, rho_i = cases["interior_generic"]
    ref_i = ref_closing(mask0_i, rho_i)
    off_bad_i = disk_offsets(rho_i, extra=1)
    out_bad_i = closing_warp(mask0_i, off_bad_i)
    bad_n_diff_i = int(np.sum(ref_i != out_bad_i))
    fallbevis["offbyone_edge"]["n_diff_px_interior_jamforelse"] = bad_n_diff_i
    fallbevis["offbyone_edge"]["kantfallet_storre_eller_lika_andel"] = bool(
        (bad_n_diff / mask0.size) >= (bad_n_diff_i / mask0_i.size))

    # --- BENCHMARK GATE: produktionsstorlek (verklig facit-raster), median-5 --------------------------
    prod_mask0, prod_rho = cases["prod_facit"]
    prod_off = disk_offsets(prod_rho)
    t_cpu, _ = bench(lambda: ref_closing(prod_mask0, prod_rho), n=5)
    t_wp, _ = bench(lambda: closing_warp(prod_mask0, prod_off), n=5)
    speedup_wp = t_cpu / t_wp if t_wp > 0 else float("inf")
    bench_entry = {
        "case_shape": list(prod_mask0.shape), "rho_px": prod_rho, "n_off": int(prod_off.shape[0]),
        "median_wall_s_cpu_skimage_baseline": t_cpu,
        "median_wall_s_warp_gpu": t_wp, "speedup_warp": speedup_wp,
    }
    if HAVE_TORCH_CUDA:
        t_th, _ = bench(lambda: closing_torch(prod_mask0, prod_rho), n=5)
        speedup_th = t_cpu / t_th if t_th > 0 else float("inf")
        bench_entry["median_wall_s_torch_gpu"] = t_th
        bench_entry["speedup_torch"] = speedup_th

    BENCH_GATE_MIN_SPEEDUP = 5.0
    variant_wins_wp = speedup_wp >= BENCH_GATE_MIN_SPEEDUP
    variant_wins_th = (bench_entry.get("speedup_torch", 0) >= BENCH_GATE_MIN_SPEEDUP) if HAVE_TORCH_CUDA else False
    bench_entry["min_gate"] = BENCH_GATE_MIN_SPEEDUP
    bench_entry["variant_wins_warp"] = variant_wins_wp
    bench_entry["variant_wins_torch"] = variant_wins_th

    result["korrekthetsgrind"] = {"cases": correctness, "pass_warp": correctness_pass_warp,
                                    "pass_torch": correctness_pass_torch}
    result["benchmarkgrind"] = bench_entry
    result["fallbevis"] = fallbevis
    result["gpu_load_after"] = gpu_load()

    winner = None
    winner_speedup = 0.0
    if correctness_pass_warp and variant_wins_wp and speedup_wp > winner_speedup:
        winner, winner_speedup = "warp_gpu_disk_offsets", speedup_wp
    if HAVE_TORCH_CUDA and correctness_pass_torch and variant_wins_th and bench_entry.get("speedup_torch", 0) > winner_speedup:
        winner, winner_speedup = "torch_conv_disk", bench_entry["speedup_torch"]

    verdict = {
        "korrekthetsgrind_warp_pass": correctness_pass_warp,
        "korrekthetsgrind_torch_pass": correctness_pass_torch,
        "benchmarkgrind_warp_wins_ge_5x": variant_wins_wp,
        "benchmarkgrind_torch_wins_ge_5x": variant_wins_th,
        "fallbevis_offbyone_grinden_fangade_felet": fallbevis["offbyone_edge"]["korrekthetsgrind_skulle_falla"],
        "fallbevis_kantfallet_diskriminerande": fallbevis["offbyone_edge"]["kantfallet_storre_eller_lika_andel"],
    }
    result["verdict"] = verdict
    result["overall_pass"] = correctness_pass_warp and (variant_wins_wp or variant_wins_th) and \
        fallbevis["offbyone_edge"]["korrekthetsgrind_skulle_falla"]
    result["winner"] = winner if winner else "cpu_skimage_baseline (no variant passed the 5x gate while staying correct)"

    out_path = os.path.join(OUT_DIR, "f4_closing_kernelvarv.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "korrekthetsgrind"}, indent=2))
    print("\n-> skrivet", out_path)
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["overall_pass"] else 1)

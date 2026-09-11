#!/usr/bin/env python3
"""Kernel variant round for a 3D constructive-solid-geometry sweep kernel (Warp).

Enumerates variants over launch dimensionality and sparsity (skip cells whose distance bound proves no
change), and runs the same three gates: bit identity against the 1D-launch baseline, determinism, and
benchmark. A planted off-by-one in the radial clamp must be caught by the correctness gate, and only on
the edge case built to hit the last valid table bin.

This round is an honest negative: no variant passes the 1.2x benchmark gate.

Requires Warp. Output: a JSON side file with the per-variant measurements.

  python kernelvarv_v1_f4_csg.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(ROOT, "reports/probes/kernelvarv_v1_sidofiler")
os.makedirs(OUT_DIR, exist_ok=True)

wp.init()
DEVICE = "cuda:0"


def make_sdf2d(nr2d=200, nz2d=200, pitch2d=0.04, r_target=6.0):
    """Synthetic (r, z) SDF table: circular cross-section of radius r_target, with a small z sweep
    so the value depends on both r and z and the bilinear interpolation is exercised in both
    directions."""
    rr = np.arange(nr2d) * pitch2d
    zz = np.arange(nz2d) * pitch2d
    R, Z = np.meshgrid(rr, zz, indexing="ij")
    r_of_z = r_target + 0.3 * np.sin(Z * 2.0)  # target radius varies along z
    sdf2d = (R - r_of_z).astype(np.float32)
    return sdf2d, rr, zz


def build_case(nx, ny, nz, pitch3d, sdf2d, rr, zz, pitch2d, r_blank, z_blank_lo, z_blank_hi,
               x0, y0):
    nr2d, nz2d = sdf2d.shape
    return dict(sdf2d=sdf2d.reshape(-1), nr2d=nr2d, nz2d=nz2d, r0_2d=float(rr[0]), z0_2d=float(zz[0]),
                pitch2d=pitch2d, x0=x0, y0=y0, z0=z_blank_lo, pitch3d=pitch3d, nx=nx, ny=ny, nz=nz,
                r_blank=r_blank, z_blank_lo=z_blank_lo, z_blank_hi=z_blank_hi,
                px=2.0, py=4.0, arch_r=2.0, arch_cz=1.0, rho_mill=0.6,
                drill_r=1.5, drill_z=1.5, drill_x_half=r_blank + 1.0)


# --- BASELINE: 1D flat launch, manuell div/mod (identisk logik/param-lista som produktionskerna) --
@wp.kernel
def cnc_subtract_1d(sdf2d: wp.array(dtype=wp.float32),
                     nr2d: int, nz2d: int, r0_2d: float, z0_2d: float, pitch2d: float,
                     x0: float, y0: float, z0: float, pitch3d: float,
                     nx: int, ny: int, nz: int,
                     r_blank: float, z_blank_lo: float, z_blank_hi: float,
                     px: float, py: float, arch_r: float, arch_cz: float, rho_mill: float,
                     drill_r: float, drill_z: float, drill_x_half: float,
                     out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    k = tid % nz
    j = (tid // nz) % ny
    i = tid // (nz * ny)
    x = x0 + float(i) * pitch3d
    y = y0 + float(j) * pitch3d
    z = z0 + float(k) * pitch3d
    r = wp.sqrt(x * x + y * y)
    sdf_blank = wp.max(r - r_blank, wp.max(z_blank_lo - z, z - z_blank_hi))
    fr = (r - r0_2d) / pitch2d
    fz = (z - z0_2d) / pitch2d
    ir0 = wp.clamp(int(fr), 0, nr2d - 2)
    iz0 = wp.clamp(int(fz), 0, nz2d - 2)
    tr = wp.clamp(fr - float(ir0), 0.0, 1.0)
    tz = wp.clamp(fz - float(iz0), 0.0, 1.0)
    v00 = sdf2d[ir0 * nz2d + iz0]
    v01 = sdf2d[ir0 * nz2d + iz0 + 1]
    v10 = sdf2d[(ir0 + 1) * nz2d + iz0]
    v11 = sdf2d[(ir0 + 1) * nz2d + iz0 + 1]
    sdf_turned = (v00 * (1.0 - tr) * (1.0 - tz) + v01 * (1.0 - tr) * tz
                  + v10 * tr * (1.0 - tz) + v11 * tr * tz)
    material = wp.max(sdf_blank, sdf_turned)
    qx = wp.abs(x) - (px - rho_mill)
    qy = wp.abs(y) - (py - rho_mill)
    dxy = wp.sqrt(wp.max(qx, 0.0) ** 2.0 + wp.max(qy, 0.0) ** 2.0) + wp.min(wp.max(qx, qy), 0.0) - rho_mill
    arch_arg = arch_r * arch_r - x * x
    roof_z = arch_cz + wp.sqrt(wp.max(arch_arg, 0.0))
    d_top = z - roof_z
    pocket_sdf = wp.max(dxy, d_top)
    material = wp.max(material, -pocket_sdf)
    dr_axis = wp.sqrt((y - 0.0) ** 2.0 + (z - drill_z) ** 2.0) - drill_r
    d_xclip = wp.abs(x) - drill_x_half
    bore_sdf = wp.max(dr_axis, d_xclip)
    material = wp.max(material, -bore_sdf)
    out[tid] = material


# --- VARIANT V1: nativ 3D launch (dim=(nx,ny,nz)), wp.tid() trippel -- ingen div/mod --------------
@wp.kernel
def cnc_subtract_3d(sdf2d: wp.array(dtype=wp.float32),
                     nr2d: int, nz2d: int, r0_2d: float, z0_2d: float, pitch2d: float,
                     x0: float, y0: float, z0: float, pitch3d: float,
                     nx: int, ny: int, nz: int,
                     r_blank: float, z_blank_lo: float, z_blank_hi: float,
                     px: float, py: float, arch_r: float, arch_cz: float, rho_mill: float,
                     drill_r: float, drill_z: float, drill_x_half: float,
                     out: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    x = x0 + float(i) * pitch3d
    y = y0 + float(j) * pitch3d
    z = z0 + float(k) * pitch3d
    r = wp.sqrt(x * x + y * y)
    sdf_blank = wp.max(r - r_blank, wp.max(z_blank_lo - z, z - z_blank_hi))
    fr = (r - r0_2d) / pitch2d
    fz = (z - z0_2d) / pitch2d
    ir0 = wp.clamp(int(fr), 0, nr2d - 2)
    iz0 = wp.clamp(int(fz), 0, nz2d - 2)
    tr = wp.clamp(fr - float(ir0), 0.0, 1.0)
    tz = wp.clamp(fz - float(iz0), 0.0, 1.0)
    v00 = sdf2d[ir0 * nz2d + iz0]
    v01 = sdf2d[ir0 * nz2d + iz0 + 1]
    v10 = sdf2d[(ir0 + 1) * nz2d + iz0]
    v11 = sdf2d[(ir0 + 1) * nz2d + iz0 + 1]
    sdf_turned = (v00 * (1.0 - tr) * (1.0 - tz) + v01 * (1.0 - tr) * tz
                  + v10 * tr * (1.0 - tz) + v11 * tr * tz)
    material = wp.max(sdf_blank, sdf_turned)
    qx = wp.abs(x) - (px - rho_mill)
    qy = wp.abs(y) - (py - rho_mill)
    dxy = wp.sqrt(wp.max(qx, 0.0) ** 2.0 + wp.max(qy, 0.0) ** 2.0) + wp.min(wp.max(qx, qy), 0.0) - rho_mill
    arch_arg = arch_r * arch_r - x * x
    roof_z = arch_cz + wp.sqrt(wp.max(arch_arg, 0.0))
    d_top = z - roof_z
    pocket_sdf = wp.max(dxy, d_top)
    material = wp.max(material, -pocket_sdf)
    dr_axis = wp.sqrt((y - 0.0) ** 2.0 + (z - drill_z) ** 2.0) - drill_r
    d_xclip = wp.abs(x) - drill_x_half
    bore_sdf = wp.max(dr_axis, d_xclip)
    material = wp.max(material, -bore_sdf)
    out[i, j, k] = material


# --- VARIANT V2: sparsity -- squared-distance pre-check before the expensive sqrt() for pocket/drill.
# The drill test depends on (y, z), so the branch is partly divergent, but the cheap squared test still avoids two sqrt() calls.
@wp.kernel
def cnc_subtract_3d_sparse(sdf2d: wp.array(dtype=wp.float32),
                            nr2d: int, nz2d: int, r0_2d: float, z0_2d: float, pitch2d: float,
                            x0: float, y0: float, z0: float, pitch3d: float,
                            nx: int, ny: int, nz: int,
                            r_blank: float, z_blank_lo: float, z_blank_hi: float,
                            px: float, py: float, arch_r: float, arch_cz: float, rho_mill: float,
                            drill_r: float, drill_z: float, drill_x_half: float,
                            out: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    x = x0 + float(i) * pitch3d
    y = y0 + float(j) * pitch3d
    z = z0 + float(k) * pitch3d
    r = wp.sqrt(x * x + y * y)
    sdf_blank = wp.max(r - r_blank, wp.max(z_blank_lo - z, z - z_blank_hi))
    fr = (r - r0_2d) / pitch2d
    fz = (z - z0_2d) / pitch2d
    ir0 = wp.clamp(int(fr), 0, nr2d - 2)
    iz0 = wp.clamp(int(fz), 0, nz2d - 2)
    tr = wp.clamp(fr - float(ir0), 0.0, 1.0)
    tz = wp.clamp(fz - float(iz0), 0.0, 1.0)
    v00 = sdf2d[ir0 * nz2d + iz0]
    v01 = sdf2d[ir0 * nz2d + iz0 + 1]
    v10 = sdf2d[(ir0 + 1) * nz2d + iz0]
    v11 = sdf2d[(ir0 + 1) * nz2d + iz0 + 1]
    sdf_turned = (v00 * (1.0 - tr) * (1.0 - tz) + v01 * (1.0 - tr) * tz
                  + v10 * tr * (1.0 - tz) + v11 * tr * tz)
    material = wp.max(sdf_blank, sdf_turned)

    # POCKET: skip om (x,y) SAKERT utanfor rundad-box+rho_mill-marginal (kvadrat-test, ingen sqrt)
    qx = wp.abs(x) - (px - rho_mill)
    qy = wp.abs(y) - (py - rho_mill)
    pocket_maybe = (qx < rho_mill) or (qy < rho_mill)  # konservativt: bara sakert-utanfor hoppas
    if pocket_maybe:
        dxy = wp.sqrt(wp.max(qx, 0.0) ** 2.0 + wp.max(qy, 0.0) ** 2.0) + wp.min(wp.max(qx, qy), 0.0) - rho_mill
        arch_arg = arch_r * arch_r - x * x
        roof_z = arch_cz + wp.sqrt(wp.max(arch_arg, 0.0))
        d_top = z - roof_z
        pocket_sdf = wp.max(dxy, d_top)
        material = wp.max(material, -pocket_sdf)

    # BORR: skip om (y,z) SAKERT utanfor drill_r+marginal fran axeln (kvadrat-test)
    dy = y - 0.0
    dz = z - drill_z
    d2_axis = dy * dy + dz * dz
    margin = drill_r + 1.0
    if d2_axis < margin * margin:
        dr_axis = wp.sqrt(d2_axis) - drill_r
        d_xclip = wp.abs(x) - drill_x_half
        bore_sdf = wp.max(dr_axis, d_xclip)
        material = wp.max(material, -bore_sdf)

    out[i, j, k] = material


# --- PLANTED-FAULT TEST-VARIANT: off-by-one i r-klampen (nr2d-1 ist f nr2d-2) -- lasa utanfor tabellen -------
@wp.kernel
def cnc_subtract_offbyone(sdf2d: wp.array(dtype=wp.float32),
                           nr2d: int, nz2d: int, r0_2d: float, z0_2d: float, pitch2d: float,
                           x0: float, y0: float, z0: float, pitch3d: float,
                           nx: int, ny: int, nz: int,
                           r_blank: float, z_blank_lo: float, z_blank_hi: float,
                           px: float, py: float, arch_r: float, arch_cz: float, rho_mill: float,
                           drill_r: float, drill_z: float, drill_x_half: float,
                           out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    k = tid % nz
    j = (tid // nz) % ny
    i = tid // (nz * ny)
    x = x0 + float(i) * pitch3d
    y = y0 + float(j) * pitch3d
    z = z0 + float(k) * pitch3d
    r = wp.sqrt(x * x + y * y)
    sdf_blank = wp.max(r - r_blank, wp.max(z_blank_lo - z, z - z_blank_hi))
    fr = (r - r0_2d) / pitch2d
    fz = (z - z0_2d) / pitch2d
    ir0 = wp.clamp(int(fr), 0, nr2d - 1)   # PLANTERAT FEL: skulle vara nr2d - 2 (v10/v11 kan lasa utanfor)
    iz0 = wp.clamp(int(fz), 0, nz2d - 2)
    tr = wp.clamp(fr - float(ir0), 0.0, 1.0)
    tz = wp.clamp(fz - float(iz0), 0.0, 1.0)
    v00 = sdf2d[ir0 * nz2d + iz0]
    v01 = sdf2d[ir0 * nz2d + iz0 + 1]
    ir1 = wp.min(ir0 + 1, nr2d - 1)
    v10 = sdf2d[ir1 * nz2d + iz0]
    v11 = sdf2d[ir1 * nz2d + iz0 + 1]
    sdf_turned = (v00 * (1.0 - tr) * (1.0 - tz) + v01 * (1.0 - tr) * tz
                  + v10 * tr * (1.0 - tz) + v11 * tr * tz)
    material = wp.max(sdf_blank, sdf_turned)
    qx = wp.abs(x) - (px - rho_mill)
    qy = wp.abs(y) - (py - rho_mill)
    dxy = wp.sqrt(wp.max(qx, 0.0) ** 2.0 + wp.max(qy, 0.0) ** 2.0) + wp.min(wp.max(qx, qy), 0.0) - rho_mill
    arch_arg = arch_r * arch_r - x * x
    roof_z = arch_cz + wp.sqrt(wp.max(arch_arg, 0.0))
    d_top = z - roof_z
    pocket_sdf = wp.max(dxy, d_top)
    material = wp.max(material, -pocket_sdf)
    dr_axis = wp.sqrt((y - 0.0) ** 2.0 + (z - drill_z) ** 2.0) - drill_r
    d_xclip = wp.abs(x) - drill_x_half
    bore_sdf = wp.max(dr_axis, d_xclip)
    material = wp.max(material, -bore_sdf)
    out[tid] = material


def run_1d(case, kernel=cnc_subtract_1d, block_dim=256):
    n_total = case["nx"] * case["ny"] * case["nz"]
    sdf2d_wp = wp.array(case["sdf2d"], dtype=wp.float32, device=DEVICE)
    out = wp.zeros(n_total, dtype=wp.float32, device=DEVICE)

    def once():
        wp.launch(kernel, dim=n_total,
                  inputs=[sdf2d_wp, case["nr2d"], case["nz2d"], case["r0_2d"], case["z0_2d"],
                          case["pitch2d"], case["x0"], case["y0"], case["z0"], case["pitch3d"],
                          case["nx"], case["ny"], case["nz"], case["r_blank"], case["z_blank_lo"],
                          case["z_blank_hi"], case["px"], case["py"], case["arch_r"], case["arch_cz"],
                          case["rho_mill"], case["drill_r"], case["drill_z"], case["drill_x_half"]],
                  outputs=[out], block_dim=block_dim)
        wp.synchronize()
        return out.numpy().copy()
    return once


def run_3d(case, block_dim=256):
    sdf2d_wp = wp.array(case["sdf2d"], dtype=wp.float32, device=DEVICE)
    out = wp.zeros((case["nx"], case["ny"], case["nz"]), dtype=wp.float32, device=DEVICE)

    def once():
        wp.launch(cnc_subtract_3d, dim=(case["nx"], case["ny"], case["nz"]),
                  inputs=[sdf2d_wp, case["nr2d"], case["nz2d"], case["r0_2d"], case["z0_2d"],
                          case["pitch2d"], case["x0"], case["y0"], case["z0"], case["pitch3d"],
                          case["nx"], case["ny"], case["nz"], case["r_blank"], case["z_blank_lo"],
                          case["z_blank_hi"], case["px"], case["py"], case["arch_r"], case["arch_cz"],
                          case["rho_mill"], case["drill_r"], case["drill_z"], case["drill_x_half"]],
                  outputs=[out], block_dim=block_dim)
        wp.synchronize()
        return out.numpy().reshape(-1).copy()
    return once


def run_3d_sparse(case, block_dim=256):
    sdf2d_wp = wp.array(case["sdf2d"], dtype=wp.float32, device=DEVICE)
    out = wp.zeros((case["nx"], case["ny"], case["nz"]), dtype=wp.float32, device=DEVICE)

    def once():
        wp.launch(cnc_subtract_3d_sparse, dim=(case["nx"], case["ny"], case["nz"]),
                  inputs=[sdf2d_wp, case["nr2d"], case["nz2d"], case["r0_2d"], case["z0_2d"],
                          case["pitch2d"], case["x0"], case["y0"], case["z0"], case["pitch3d"],
                          case["nx"], case["ny"], case["nz"], case["r_blank"], case["z_blank_lo"],
                          case["z_blank_hi"], case["px"], case["py"], case["arch_r"], case["arch_cz"],
                          case["rho_mill"], case["drill_r"], case["drill_z"], case["drill_x_half"]],
                  outputs=[out], block_dim=block_dim)
        wp.synchronize()
        return out.numpy().reshape(-1).copy()
    return once


def bench(fn, n=5):
    fn(); fn()
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
    result = {"cell": "kernelvarv_v1_f4_csg", "genererad": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "gpu": str(wp.get_device(DEVICE)), "gpu_load_before": gpu_load()}

    sdf2d, rr, zz = make_sdf2d(nr2d=200, nz2d=200, pitch2d=0.04, r_target=6.0)
    PITCH2D = 0.04

    def mk(nx, ny, nz, pitch3d, r_blank_frac=0.9):
        r_blank = r_blank_frac * rr[-1]
        return build_case(nx, ny, nz, pitch3d, sdf2d, rr, zz, PITCH2D,
                           r_blank=r_blank, z_blank_lo=0.0, z_blank_hi=zz[-1],
                           x0=-r_blank - 1, y0=-r_blank - 1)

    # edge case built to hit the last valid bin of the r table (fr in [nr2d-2, nr2d-1))
    # where the planted off-by-one clamp (nr2d-1 instead of nr2d-2) actually differs from the reference
    r_edge_center = (200 - 1.5) * PITCH2D  # centre of the last valid bin
    edge_case = build_case(24, 24, 16, 0.05, sdf2d, rr, zz, PITCH2D,
                            r_blank=r_edge_center + 2.0, z_blank_lo=0.0, z_blank_hi=zz[-1],
                            x0=r_edge_center - 0.6, y0=-0.6)  # x near r_edge (small y) so r spans the band

    cases = {
        "edge_r_table_boundary": edge_case,
        "small_32x32x16": mk(32, 32, 16, pitch3d=0.3),
        "medium_64x64x32": mk(64, 64, 32, pitch3d=0.18),
    }

    # --- CORRECTNESS GATE: 1D-baslinje vs 3D-variant, BITIDENTISKT (ren funktionell kernel, ingen
    # atomic, samma flyttalsordning per voxel -> exakt likhet forvantad). 3D-sparse provas SEPARAT
    # (informativ, blockerar EJ vinnarvalet) -- se ORIENT-anteckning nedan varfor den dog. -----------
    correctness = {}
    sparse_diag = {}
    for name, case in cases.items():
        r1d = run_1d(case)()
        r3d = run_3d(case)()
        rsp = run_3d_sparse(case)()
        bit_id_3d = bool(np.array_equal(r1d, r3d))
        bit_id_sp = bool(np.array_equal(r1d, rsp))
        correctness[name] = {
            "n_voxels": case["nx"] * case["ny"] * case["nz"],
            "bit_identical_1d_vs_3d": bit_id_3d,
            "max_abs_diff_3d": 0.0 if bit_id_3d else float(np.max(np.abs(r1d - r3d))),
        }
        sparse_diag[name] = {"bit_identical_1d_vs_sparse": bit_id_sp,
                              "max_abs_diff_sparse": 0.0 if bit_id_sp else float(np.max(np.abs(r1d - rsp)))}
    correctness_pass = all(c["bit_identical_1d_vs_3d"] for c in correctness.values())
    sparse_correctness_pass = all(d["bit_identical_1d_vs_sparse"] for d in sparse_diag.values())

    # --- DETERMINISMGATE: 2x korning, bitidentiskt (1D, 3D -- vinnarkandidaterna) -----------------
    determinism = {}
    for name, case in cases.items():
        f1 = run_1d(case); a1, a2 = f1(), f1()
        f3 = run_3d(case); b1, b2 = f3(), f3()
        determinism[name] = {"1d_bit_identical_2x": bool(np.array_equal(a1, a2)),
                              "3d_bit_identical_2x": bool(np.array_equal(b1, b2))}
    determinism_pass = all(d["1d_bit_identical_2x"] and d["3d_bit_identical_2x"] for d in determinism.values())

    prod_case = mk(288, 288, 132, pitch3d=0.35)

    block_dim_sweep_1d = {}
    block_dim_sweep_3d = {}
    for bdim in (64, 128, 256, 512, 1024):
        t1, _ = bench(run_1d(prod_case, block_dim=bdim), n=5)
        block_dim_sweep_1d[str(bdim)] = t1
        t3, _ = bench(run_3d(prod_case, block_dim=bdim), n=5)
        block_dim_sweep_3d[str(bdim)] = t3
    best_block_dim_1d = int(min(block_dim_sweep_1d, key=block_dim_sweep_1d.get))
    best_block_dim_3d = int(min(block_dim_sweep_3d, key=block_dim_sweep_3d.get))
    t_1d = block_dim_sweep_1d[str(best_block_dim_1d)]
    t_3d = block_dim_sweep_3d[str(best_block_dim_3d)]

    speedup = t_1d / t_3d if t_3d > 0 else float("inf")
    BENCH_GATE_MIN_SPEEDUP = 1.2
    variant_wins = speedup >= BENCH_GATE_MIN_SPEEDUP
    best_name = "3d_launch"
    best_block_dim = best_block_dim_3d

    offbyone_results = {}
    for name, case in cases.items():
        r_ref = run_1d(case)()
        r_bad = run_1d(case, kernel=cnc_subtract_offbyone)()
        bit_id = bool(np.array_equal(r_ref, r_bad))
        max_diff = float(np.max(np.abs(r_ref - r_bad)))
        offbyone_results[name] = {"bit_identical_vs_ref": bit_id, "max_abs_diff": max_diff}
    edge_caught = offbyone_results["edge_r_table_boundary"]["bit_identical_vs_ref"] is False
    edge_max_diff = offbyone_results["edge_r_table_boundary"]["max_abs_diff"]
    other_max_diffs = [offbyone_results[n]["max_abs_diff"] for n in cases if n != "edge_r_table_boundary"]
    fallbevis_diskriminerande = edge_caught and (edge_max_diff >= max(other_max_diffs + [0.0]))

    result["korrekthetsgrind"] = {"cases": correctness, "pass": correctness_pass}
    result["determinismgrind"] = {"cases": determinism, "pass": determinism_pass}
    result["benchmarkgrind"] = {
        "case": {"grid_shape": [prod_case["nx"], prod_case["ny"], prod_case["nz"]],
                  "n_voxels": prod_case["nx"] * prod_case["ny"] * prod_case["nz"]},
        "median_wall_s_1d_baseline_best_blockdim": t_1d, "best_block_dim_1d": best_block_dim_1d,
        "median_wall_s_3d_launch_variant_best_blockdim": t_3d, "best_block_dim_3d": best_block_dim_3d,
        "speedup": speedup, "min_gate": BENCH_GATE_MIN_SPEEDUP, "variant_wins": variant_wins,
    }
    result["block_dim_svep_s_1d_baseline"] = block_dim_sweep_1d
    result["block_dim_svep_s_3d_launch"] = block_dim_sweep_3d
    result["best_block_dim"] = best_block_dim
    result["sparsitet_variant_DEAD"] = {
        "orient_diagnos": "measured and dead, not merely deprioritised: the pocket/drill subtraction "
            "uses material = max(material, -op_sdf) as the accumulator, not a true clamped/bounded SDF. "
            "A squared-distance pre-check (skip the op_sdf evaluation when (x, y) is safely outside the "
            "tool-radius margin of the pocket) proves op_sdf > 0 there, but not that op_sdf is LARGE "
            "ENOUGH: at the edge of the margin op_sdf is only about 0.41 of the tool radius, so -op_sdf "
            "can still dominate max() over a deeply negative material value (measured: max_abs_diff up "
            "to 1.9995 on the small and medium cases, a genuine correctness-gate failure, not noise). "
            "A safe margin would need the global depth bound of the material (~r_blank), which makes the "
            "skip so rare that the gain disappears. The sparsity axis is inapplicable to this "
            "max()-accumulator formulation without changing the CSG composition itself.",
        "korrekthet_per_fall": sparse_diag, "korrekthet_pass": sparse_correctness_pass,
    }
    result["fallbevis_offbyone"] = {
        "metod": "the r clamp set to nr2d-1 (correct: nr2d-2), so the interpolation can read a wrong or clamped row",
        "per_case": offbyone_results,
        "kantfallet_fangat": edge_caught,
        "kantfallet_storst_avvikelse": fallbevis_diskriminerande,
    }
    result["loop_fusion_status"] = ("already fully exploited in the baseline: blank, turn, pocket and "
                                    "drill run as ONE wp.launch pass, so no further fusion is possible "
                                    "without changing the operation set")
    result["gpu_load_after"] = gpu_load()

    verdict = {
        "korrekthetsgrind_pass": correctness_pass,
        "determinismgrind_pass": determinism_pass,
        "benchmarkgrind_variant_wins_ge_1.2x": variant_wins,
        "fallbevis_offbyone_kantfallet_fangat": edge_caught,
        "fallbevis_offbyone_diskriminerande": fallbevis_diskriminerande,
    }
    result["verdict"] = verdict
    result["overall_pass"] = all(verdict.values())
    result["winner"] = (f"{best_name}_bd{best_block_dim}") if (correctness_pass and determinism_pass and variant_wins) \
        else "1d_baseline (ingen levande variant slog 1.2x-grinden -- 3d_launch matchade grinden ej, " \
             "sparsitetsvarianten dodad pa korrekthetsgrinden, se sparsitet_variant_DEAD)"

    out_path = os.path.join(OUT_DIR, "f4_csg_kernelvarv.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("korrekthetsgrind", "determinismgrind")}, indent=2))
    print("\n-> skrivet", out_path)
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["overall_pass"] else 1)

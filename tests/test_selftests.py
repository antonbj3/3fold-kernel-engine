"""Pytest wrappers around the self-tests each module already carries.

Every module is run as a script in a fresh subprocess; the test asserts exit code 0.
Modules that need a CUDA device are skipped when no CUDA device is present.
"""
import os
import shutil
import subprocess
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "kernel_engine")


def _cuda_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


CUDA = _cuda_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="requires a CUDA device")
slow = pytest.mark.skipif(not CUDA, reason="exceeds the time budget without a CUDA device")


def run(rel, args=(), timeout=600):
    path = os.path.join(SRC, rel)
    proc = subprocess.run([sys.executable, path, *args], capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return proc.stdout


# --- allocation law (CPU, seconds) -------------------------------------------------------------
def test_waterfilling_unification():
    run("allocation/d_poxel_waterfilling_unification.py")


def test_bitwidth_waterfilling():
    run("allocation/d_fpga_bitwidth_waterfilling.py")


def test_self_tuning_sensitivity_kernel():
    run("allocation/d_self_tuning_sensitivity_kernel.py")


def test_goal_derived_representation():
    run("allocation/d_goal_derived_representation.py")


# --- vendored helper ---------------------------------------------------------------------------
def test_render_match_scaffold():
    run("_vendor/render_match_scaffold.py")


# --- LBM ---------------------------------------------------------------------------------------
def test_lbm3d_poiseuille_numpy():
    out = run("lbm/lbm3d_poiseuille.py")
    assert "CLOSES" in out


def test_lbm3d_gpu():
    out = run("lbm/lbm3d_gpu.py")
    assert "CLOSES" in out


@slow
def test_lbm_gpu_fast():
    run("lbm/lbm_gpu_fast.py")


@cuda_only
def test_airflow_lbm_d3q19_poiseuille():
    run("lbm/gpu_lbm_luftflode_v1.py", ["poiseuille"])


# --- certified kernel loop ---------------------------------------------------------------------
def test_fixed_point_reduction_determinism():
    out = run("certified_loop/d_cuda_scene_eyes_determinism_real_gpu.py")
    assert "DETERMINISM" in out


def test_privatized_reduction():
    run("certified_loop/d_privatized_reduction_close_abstain.py")


@cuda_only
def test_autonomy_loop_real_cuda():
    run("certified_loop/d_1c_v_autonomy_loop_end_to_end_real_cuda.py")


@cuda_only
def test_lbm_soa_certified_roofline():
    run("certified_loop/d_lbm_soa_certified_roofline_close.py")


@pytest.mark.skipif(not CUDA or shutil.which("vulkaninfo") is None,
                    reason="requires a CUDA device and a Vulkan driver (plus the wgpu package)")
def test_vulkan_port_certified_kernel():
    run("certified_loop/d_vulkan_port_certified_kernel.py")


# --- kernel variant rounds ---------------------------------------------------------------------
@cuda_only
def test_variant_round_matvec():
    run("kernel_variants/kernelvarv_v1_f2_matvec.py")


@cuda_only
def test_variant_round_csg():
    run("kernel_variants/kernelvarv_v1_f4_csg.py")


@cuda_only
def test_variant_round_closing():
    run("kernel_variants/kernelvarv_v2_f4_closing.py")

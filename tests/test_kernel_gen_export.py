"""Warp -> CUDA C++ export check for the int64 fixed-point reduction.

Builds the exported kernel with nvcc into a Warp-free host program, runs it on the same input the
Warp driver wrote, and asserts the 64 int64 output slots are bit-identical to the Warp-launched result.
Skips when nvcc, a CUDA device or warp is missing.
"""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "src", "kernel_engine", "kernel_gen", "export")


def _cuda_device_available():
    try:
        import warp as wp
    except Exception:
        return False
    try:
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not installed")
@pytest.mark.skipif(not _cuda_device_available(), reason="no CUDA device / warp not importable")
def test_exported_kernel_is_bit_identical(tmp_path):
    np = pytest.importorskip("numpy")
    work = str(tmp_path)

    ref = subprocess.run([sys.executable, os.path.join(EXPORT, "warp_reference_run.py"), work],
                         capture_output=True, text=True, timeout=600)
    assert ref.returncode == 0, ref.stderr[-2000:]
    assert "WARP sum=" in ref.stdout, ref.stdout

    build_dir = os.path.join(work, "build")
    build = subprocess.run([os.path.join(EXPORT, "build.sh"), build_dir],
                           capture_output=True, text=True, timeout=600)
    assert build.returncode == 0, build.stderr[-4000:]

    run = subprocess.run([os.path.join(build_dir, "host_int64_reduce"), work],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "NVCC sum=" in run.stdout, run.stdout

    a = np.load(os.path.join(work, "out_warp.npy"))
    b = np.fromfile(os.path.join(work, "out_nvcc.bin"), dtype=np.int64)
    assert a.shape == b.shape
    assert np.array_equal(a, b), "exported kernel is not bit-identical to the Warp-launched original"

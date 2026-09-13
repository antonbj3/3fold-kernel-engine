"""Pytest wrapper for the int64 fixed-point determinism sweep (skips without a CUDA device)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
CELL = os.path.join(SRC, "kernel_engine", "certified_kernels", "determinism_sweep_int64.py")


def _cuda_available():
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="requires a CUDA device")
def test_determinism_sweep_int64():
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, CELL], capture_output=True, text=True, env=env, timeout=900)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    assert "DETERMINISTIC" in p.stdout

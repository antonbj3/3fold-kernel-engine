"""Pytest wrapper for the chunkable-recurrence rule (the fifth a-priori requirement).

The module's own selftest is run as a script by the parametrized CPU sweep in test_pass3_selftests.py;
these tests check the two claims directly and check that wiring the rule into the a-priori cert leaves the
cert's own gates untouched (the `--no-chunkable-rule` flag reproduces the four-requirement cert exactly).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "src", "kernel_engine", "certified_kernels")
sys.path.insert(0, MOD)

from chunkable_recurrence_rule import (CHUNKABLE, NO_RECURRENCE, SEQUENTIAL, byte_floors,  # noqa: E402
                                       detect_recurrence, gated_delta_chunked, gated_delta_sequential,
                                       suggest_chunk_size)


def test_chunked_equals_sequential_at_fp32_roundoff():
    """gated delta rule, per-channel diagonal gate, d=64, T=512, chunk 64."""
    rng = np.random.default_rng(0)
    T, d, chunk = 512, 64, 64
    q = rng.standard_normal((T, d)).astype(np.float32)
    k = rng.standard_normal((T, d)).astype(np.float32)
    k = (k / np.linalg.norm(k, axis=1, keepdims=True)).astype(np.float32)
    v = rng.standard_normal((T, d)).astype(np.float32)
    g = (0.95 + 0.05 * rng.random((T, d))).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=T).astype(np.float32)
    O_seq, S_seq = gated_delta_sequential(q, k, v, g, beta)
    O_chk, S_chk = gated_delta_chunked(q, k, v, g, beta, chunk)
    eps = np.finfo(np.float32).eps
    assert np.max(np.abs(O_seq - O_chk)) <= eps * np.max(np.abs(O_seq)) * np.sqrt(chunk * d)
    assert np.max(np.abs(S_seq - S_chk)) <= eps * np.max(np.abs(S_seq)) * np.sqrt(T * d)


@pytest.mark.parametrize("src,expected", [
    ("h = torch.tanh(self.W @ h + x[t])", SEQUENTIAL),
    ("self.lstm = nn.LSTM(input_size, hidden_size)", SEQUENTIAL),
    ("self.gru = nn.GRU(input_size, hidden_size)", SEQUENTIAL),
    ("return torch.cumsum(x, dim=self.dim)", CHUNKABLE),
    ("A_cumsum = torch.cumsum(A_blocks, dim=-1)", CHUNKABLE),
    ("S = gated_delta_rule(q, k, v, beta)", CHUNKABLE),
    ("return torch.matmul(A, B)", NO_RECURRENCE),
])
def test_transition_classes(src, expected):
    assert detect_recurrence(src)[0] == expected


def test_byte_floors_shrink_state_traffic_by_the_chunk_size():
    fl = byte_floors(512, 64, 64, 64)
    assert fl["state_traffic_ratio"] == pytest.approx(64.0)
    assert fl["chunked_bytes"] < fl["sequential_bytes"]
    assert suggest_chunk_size(64, 64, 512) == 64


def test_apriori_cert_gates_unchanged_by_the_fifth_requirement():
    """the cert passes with the rule on (default) and with `--no-chunkable-rule`; the R5 block only appears
    in the first run."""
    script = os.path.join(MOD, "apriori_requirement_cert_on_real_kernelbench.py")
    on = subprocess.run([sys.executable, script], cwd=MOD, capture_output=True, text=True, timeout=600)
    off = subprocess.run([sys.executable, script, "--no-chunkable-rule"], cwd=MOD, capture_output=True,
                         text=True, timeout=600)
    assert on.returncode == 0 and off.returncode == 0, on.stdout[-2000:] + off.stdout[-2000:]
    verdict_on = [ln for ln in on.stdout.splitlines() if ln.startswith("VERDICT:")]
    verdict_off = [ln for ln in off.stdout.splitlines() if ln.startswith("VERDICT:")]
    assert verdict_on == verdict_off
    assert "[R5]" in on.stdout and "[R5]" not in off.stdout

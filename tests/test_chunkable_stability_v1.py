"""Refusal controls for the separate numerical chunk audit."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/kernel_engine/certified_kernels'))
from chunkable_roundoff_guard_v2 import SOURCE, experiment, transition_guard


def test_measured_expansion_and_stable_roundoff_failure():
    report, _ = experiment()
    assert all(report['gates'].values())


def test_unknown_and_nonfinite_transitions_refused():
    assert not transition_guard(SOURCE)['accepted']
    assert not transition_guard(SOURCE, [[1.0]], [[float('nan')]], [.1])['accepted']


def test_spectral_radius_below_one_is_not_enough():
    k = np.array([[1, 1]]) / np.sqrt(2)
    g = np.array([[.45, 1.35]])
    operator = np.diag(g[0]) @ (np.eye(2) - np.outer(k[0], k[0]))
    assert max(abs(np.linalg.eigvals(operator))) < 1
    assert not transition_guard(SOURCE, k, g, [1.0])['accepted']

"""Regression for numerical failure despite composable transition syntax."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/kernel_engine/certified_kernels'))
from chunkable_stability_v1 import report


def test_expansive_recurrence_and_nonnormal_control():
    result = report()
    assert all(result['gates'].values()), result['gates']

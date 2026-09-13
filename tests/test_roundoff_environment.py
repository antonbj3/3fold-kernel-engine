"""Observer integrity and temporary instrumentation restoration controls."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import roundoff_environment_v1 as observer
import roundoff_environment_compare_v1 as comparison


def test_capture_preserves_an_existing_trace():
    q=np.ones((4,1),np.float32);inputs=(q,q,q,q,np.ones(4,np.float32),np.zeros((1,1),np.float32))
    old=sys.gettrace()
    def previous(*args):return previous
    try:
        sys.settrace(previous);observer.capture(inputs)
        assert sys.gettrace() is previous
    finally:sys.settrace(old)


def test_transplant_restores_solver_when_function_raises(monkeypatch):
    original=observer.frozen.solve_triangular
    def failure(*args):raise RuntimeError('controlled')
    monkeypatch.setattr(observer.frozen,'_chunk_forward',failure)
    with pytest.raises(RuntimeError,match='controlled'):
        comparison.transplant([],{'Cmat':np.ones((1,1)),'Umat':np.ones((1,1))})
    assert observer.frozen.solve_triangular is original


def test_tampered_array_evidence_refuses(tmp_path):
    source=observer.ROOT/'reports/roundoff_environment_v1'
    shutil.copyfile(source/'report.json',tmp_path/'report.json')
    with np.load(source/'arrays.npz') as data:arrays={k:data[k].copy() for k in data.files}
    arrays['Cmat'][0,0]+=1
    np.savez_compressed(tmp_path/'arrays.npz',**arrays)
    with pytest.raises(ValueError,match='evidence mismatch'):comparison.load(tmp_path)


def test_source_drift_refuses_before_transplant(tmp_path):
    source=observer.ROOT/'reports/roundoff_environment_v1'
    shutil.copyfile(source/'arrays.npz',tmp_path/'arrays.npz')
    d=json.loads((source/'report.json').read_text());d['observer_sha256']='changed'
    (tmp_path/'report.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='drift'):comparison.compare(source,tmp_path)

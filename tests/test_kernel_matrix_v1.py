"""Small audit fixtures exercise refusals; hardware evidence lives in capsules."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/kernel_engine/certified_kernels'))
import kernel_matrix_report_v1 as report


def captures(monkeypatch):
    monkeypatch.setattr(report,'EXPECTED_COUNTS',{s:2 for s in report.EXPECTED_SLOTS})
    records=[dict(ordinal=i,role='readback',shape=[1],dtype='float32',bytes=4,sha256='0'*64) for i in range(2)]
    leg=dict(records=records,returncode=0,exception_classes=[],finite=True)
    sites=[]
    for module,site in sorted(report.EXPECTED_SITES):
        raw=dict(dtype='int64',shape=[1],values=[17],sha256=hashlib.sha256(np.array([17],dtype=np.int64).tobytes()).hexdigest())
        sites.append(dict(module=module,site=site,kernel=site,input_hashes=[],legs=[raw,copy.deepcopy(raw)],exact=True,int64_delta=0.))
    export=[dict(side=n,input_sha256='1'*64,native_sha256='2'*64,warp_sha256='2'*64,expected_sha256='2'*64,bytes=9*n*n*4,
                 exact_reference=True,exact_replays=True) for n in (512,1024)]
    return [dict(device=device+', driver',source_hashes={'frozen.py':'a'*64},warp='1.17.0',
        selftests=[dict(slot=slot,legs=copy.deepcopy([leg,leg]),gates={'passed':True}) for slot in sorted(report.EXPECTED_SLOTS)],
        sites=copy.deepcopy(sites),exports=copy.deepcopy([export,export]),gates={'passed':True},export_performance={})
        for device in sorted(report.EXPECTED_DEVICES)]


def test_complete_matrix_and_missing_local_are_explicit(monkeypatch):
    result=report.compare(captures(monkeypatch))
    assert all(result['gates'].values())
    assert len(result['rows'])==20
    assert result['unmeasured_devices']==['RTX 5070']


@pytest.mark.parametrize('mutation',('missing_slot','truncated_arrays','source','integer_values','same_device','cross_difference'))
def test_corrupt_or_incomplete_capsules_refuse(monkeypatch,mutation):
    data=captures(monkeypatch)
    if mutation=='missing_slot':data[0]['selftests'].pop()
    if mutation=='truncated_arrays':
        for c in data:
            for leg in c['selftests'][0]['legs']:leg['records'].pop()
    if mutation=='source':data[1]['source_hashes']['frozen.py']='b'*64
    if mutation=='integer_values':
        for leg in data[1]['sites'][0]['legs']:leg['values']=[18]
    if mutation=='same_device':data[1]['device']=data[0]['device']
    if mutation=='cross_difference':
        for leg in data[1]['selftests'][0]['legs']:leg['records'][0]['sha256']='9'*64
    result=report.compare(data)
    assert result['status']=='OWN-GATE-FAIL'
    if mutation=='cross_difference':
        differing=[p for row in result['rows'] for p in row['comparisons'] if not p['exact']]
        assert differing and all(p['max_absolute_delta'] is None and p['first_difference']['ordinal']==0 for p in differing)


def test_capsule_checksum_rejects_corruption(tmp_path):
    raw=b'{}'
    (tmp_path/'capture.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    (tmp_path/'summary.json').write_text(json.dumps({'capture_sha256':'0'*64}))
    with pytest.raises(ValueError,match='checksum'):report.load(tmp_path)

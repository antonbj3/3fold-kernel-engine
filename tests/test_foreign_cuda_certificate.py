"""Reject incomplete or tampered retained cross-SKU numerical evidence."""
import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/kernel_engine/kernel_gen'))
from foreign_cuda_cross_sku_v1 import audit


def reports():
    paths=[ROOT/'reports/foreign_cuda_v1'/sku for sku in ('l4','a10g')]
    return ([json.loads((p/'certificate.json').read_text()) for p in paths],
            [json.loads((p/'measurement.json').read_text()) for p in paths])


def test_retained_evidence_passes():
    assert all(audit(*reports())['gates'].values())


@pytest.mark.parametrize('mutation',('hash','missing','failed_gate','same_sku','infinite_time'))
def test_corrupted_evidence_refuses(mutation):
    c,m=copy.deepcopy(reports())
    if mutation=='hash':c[1]['cases'][0]['output_sha256']='0'*64
    if mutation=='missing':
        for item in c:item['cases'].pop()
    if mutation=='failed_gate':
        for item in c:item['gates']['G1_reference']=False
    if mutation=='same_sku':m[1]['device']=m[0]['device']
    if mutation=='infinite_time':m[1]['timings'][-1]['legs'][0]['kernel_ms']=float('inf')
    assert audit(c,m)['status']=='OWN-GATE-FAIL'

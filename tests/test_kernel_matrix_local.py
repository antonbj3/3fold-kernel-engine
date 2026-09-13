"""Four-device extension keeps strict completeness and identity refusals."""
import copy
from pathlib import Path
import sys
import pytest
from test_kernel_matrix_v1 import captures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import kernel_matrix_local_report_v1 as local

@pytest.mark.parametrize('mutation',('none','missing_local','duplicate_local','local_difference','source_drift','truncated_local'))
def test_four_device_audit(monkeypatch,mutation):
    data=captures(monkeypatch)
    fourth=copy.deepcopy(data[0]);fourth['device']='NVIDIA GeForce RTX 5070, driver';data.append(fourth)
    if mutation=='missing_local':data.pop()
    if mutation=='duplicate_local':fourth['device']=data[0]['device']
    if mutation=='local_difference':
        for leg in fourth['selftests'][0]['legs']:leg['records'][0]['sha256']='e'*64
    if mutation=='source_drift':fourth['source_hashes']['frozen.py']='c'*64
    if mutation=='truncated_local':
        for leg in fourth['selftests'][0]['legs']:leg['records'].pop()
    result=local.compare(data)
    assert all(result['gates'].values())==(mutation=='none')
    if mutation=='none':assert len(result['rows'])==20 and all(len(r['comparisons'])==6 for r in result['rows'])

"""Never turn a hash mismatch into an invented instruction-level attribution."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import kernel_float_provenance_v1 as provenance


def fixture():
    pair=dict(left='L4',right='5070',exact=True,first_difference=None,max_absolute_delta=0)
    return dict(gates={'integrity':True,'cross_sku_exact':True},rows=[dict(module=str(i),comparisons=[copy.deepcopy(pair) for _ in range(6)]) for i in range(20)])


def test_empty_and_mismatch_catalogues():
    data=fixture();assert provenance.catalogue(data)['status']=='VERIFIED-FRESH'
    data['rows'][0]['comparisons'][0].update(exact=False,first_difference={'ordinal':19},max_absolute_delta=None)
    result=provenance.catalogue(data)
    assert result['status']=='OWN-GATE-FAIL'
    assert result['differences'][0]['operation_class']=='unresolved'
    assert result['differences'][0]['first_difference']=={'ordinal':19}
    assert all(not v for v in result['operation_classes'].values())


def test_incomplete_or_invalid_matrix_refused():
    data=fixture();data['rows'].pop();assert provenance.catalogue(data)['status']=='OWN-GATE-FAIL'
    data=fixture();data['gates']['integrity']=False;assert provenance.catalogue(data)['status']=='OWN-GATE-FAIL'

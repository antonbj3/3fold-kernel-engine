import copy
from pathlib import Path
import pytest


def test_symmetric_probe_bands_cover_each_cell_and_preserve_parent_groups(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from measure_lbm3d_channel_sensitivity import probe_layout
    for h in [32,64,96,128]:
        width,steps,sample=probe_layout(h)
        cells=[]
        for band in range(8):
            lo,hi=band*width,(band+1)*width
            assert lo%2==hi%2==0
            cells.extend(range(lo,hi));cells.extend(range(h-hi,h-lo))
        assert sorted(cells)==list(range(h))
        assert steps/(h/2)==8 and sample/(h/2)==.5
        assert (steps//2)//sample==8
    with pytest.raises(ValueError):probe_layout(48)


def test_replay_rejects_changed_interior_checkpoint_or_statistics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from measure_lbm3d_resolution_sensitivity import verify_replay
    p=Path(__file__).resolve().parents[1]/'reports/lbm3d_recursive_resolution_v1/20260914T103313Z/report.json'
    import json
    control=json.loads(p.read_text())
    assert verify_replay(control,copy.deepcopy(control))
    for key,value in [('state_sha256','wrong'),('plane_integer_statistics',[])]:
        changed=copy.deepcopy(control);changed['blocks'][5][key]=value
        assert not verify_replay(control,changed)
    changed=copy.deepcopy(control);changed['passed']=False
    assert not verify_replay(control,changed)

"""Refuse corrupted arrays and misreported runtime dispatch."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import roundoff_dispatch_compare_v1 as audit


def fixture(root):
    source=audit.ROOT/'reports/roundoff_dispatch_v1'
    for i in range(2):
        for ext in ['json','npz']:
            name='Haswell_Haswell_'+str(i)+'.'+ext
            shutil.copyfile(source/name,root/name)
    return root


def test_complete_case_and_missing_array_refusal(tmp_path):
    root=fixture(tmp_path)
    record,arrays=audit.load_case(root,'Haswell_Haswell');assert len(arrays)==56
    arrays.pop('own_Cmat')
    np.savez_compressed(root/'Haswell_Haswell_0.npz',**arrays)
    with pytest.raises(ValueError,match='Incomplete'):audit.load_case(root,'Haswell_Haswell')


def test_changed_array_refuses(tmp_path):
    root=fixture(tmp_path)
    with np.load(root/'Haswell_Haswell_0.npz') as data:arrays={k:data[k].copy() for k in data.files}
    arrays['own_Cmat'][0,0]+=1
    np.savez_compressed(root/'Haswell_Haswell_0.npz',**arrays)
    with pytest.raises(ValueError,match='Array mismatch'):audit.load_case(root,'Haswell_Haswell')


def test_reported_success_does_not_override_actual_core(tmp_path):
    root=fixture(tmp_path);p=root/'Haswell_Haswell_0.json';r=json.loads(p.read_text())
    r['libraries']['numpy']['architecture']='SkylakeX';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Dispatch contract'):audit.load_case(root,'Haswell_Haswell')

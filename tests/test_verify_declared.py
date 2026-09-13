"""Successful exits do not replace decisive report checks."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_declared_v1 import check_expected


def test_decisive_value_and_json_types(tmp_path):
    p=tmp_path/'report.json';p.write_text(json.dumps({'gates':{'ok':True},'rows':[1,2]}))
    expected=dict(exit_code=0,report='report.json',json_equal={'/gates/ok':True},json_length={'/rows':2})
    assert all(check_expected(tmp_path,expected,0,'').values())
    p.write_text(json.dumps({'gates':{'ok':1},'rows':[1,2]}))
    assert not all(check_expected(tmp_path,expected,0,'').values())


def test_exit_counts_and_missing_evidence(tmp_path):
    assert not all(check_expected(tmp_path,dict(exit_code=0),1,'').values())
    assert not all(check_expected(tmp_path,dict(exit_code=0,pytest_passed=4),0,'3 passed').values())
    with pytest.raises(FileNotFoundError):check_expected(tmp_path,dict(exit_code=0,report='missing.json'),0,'')

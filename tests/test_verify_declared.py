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

def test_shared_execution_keeps_each_rows_own_gate(tmp_path):
    from verify_declared_v1 import run_recipes
    (tmp_path/'reports').mkdir()
    # Stale success must be removed before the one fresh child writes its result.
    (tmp_path/'reports/result.json').write_text('{"value":99}')
    program="from pathlib import Path; p=Path('count'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); Path('reports/result.json').write_text('{\"value\":7}')"
    recipe=dict(cwd='.',runtime='cpu',argv=['python','-c',program])
    rows=run_recipes(tmp_path,[dict(recipe,key=str(i),expected=dict(exit_code=0,report='reports/result.json',json_equal={'/value':v})) for i,v in enumerate([7,99,7])],'cpu')
    assert (tmp_path/'count').read_text()=='1'
    assert [all(r['checks'].values()) for r in rows]==[True,False,True]
    assert [r['reused_execution'] for r in rows]==[False,True,True]
    assert all(r['executed_by']=='0' for r in rows)
    run_recipes(tmp_path,[dict(recipe,key='fresh',expected=dict(exit_code=0,report='reports/result.json',json_equal={'/value':7}))],'cpu')
    assert (tmp_path/'count').read_text()=='2'


def test_changed_command_cannot_reuse_or_overwrite_shared_evidence(tmp_path):
    from verify_declared_v1 import run_recipes
    (tmp_path/'reports').mkdir()
    def recipe(key,value):
        return dict(key=key,cwd='.',runtime='cpu',argv=['python','-c',f"from pathlib import Path; Path('reports/result.json').write_text('{{\"value\":{value}}}')"],expected=dict(exit_code=0,report='reports/result.json',json_equal={'/value':value}))
    a=recipe('a',7);b=recipe('b',8)
    rows=run_recipes(tmp_path,[a,b,dict(a,key='c')],'cpu')
    assert [r['reused_execution'] for r in rows]==[False,False,True]
    assert not rows[2]['checks']['fresh_execution_evidence']
    assert not all(rows[2]['checks'].values())


def test_shared_child_failure_is_never_promoted(tmp_path):
    from verify_declared_v1 import run_recipes
    recipe=dict(cwd='.',runtime='cpu',argv=['python','-c','raise SystemExit(3)'],expected={'exit_code':0})
    rows=run_recipes(tmp_path,[dict(recipe,key='a'),dict(recipe,key='b')],'cpu')
    assert [r['returncode'] for r in rows]==[3,3]
    assert all(not all(r['checks'].values()) for r in rows)

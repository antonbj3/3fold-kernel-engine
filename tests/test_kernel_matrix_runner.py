"""Transport orchestration controls; no cloud calls are made by these tests."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import kernel_matrix_modal_v1 as runner


@pytest.mark.parametrize('missing_capsule',(False,True))
def test_sequential_fetch_and_numeric_failure_propagation(tmp_path,monkeypatch,missing_capsule):
    external=tmp_path/'external';external.mkdir();(external/'run.py').write_text('# configured test runner')
    result=tmp_path/'fetched/reports/kernel_matrix_v1';result.mkdir(parents=True)
    (result/'summary.json').write_text('{}')
    if not missing_capsule:(result/'capture.json.gz').write_bytes(b'fixture')
    repo=tmp_path/'repo';repo.mkdir();calls=[]
    monkeypatch.setattr(runner,'ROOT',repo)
    monkeypatch.setattr(sys,'argv',['kernel_matrix_modal_v1.py','--runner-root',str(external)])
    def fake_run(command,**kwargs):
        calls.append(command)
        if command[0]=='modal':
            assert kwargs['timeout']==3000
            return SimpleNamespace(stdout='results: '+str(tmp_path/'fetched')+'\n',stderr='',returncode=0)
        target=repo/'reports/kernel_matrix_v1'
        (target/'matrix.json').write_text('{"status":"OWN-GATE-FAIL"}')
        (target/'matrix.md').write_text('FAIL')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess,'run',fake_run)
    if missing_capsule:
        with pytest.raises(RuntimeError,match='incomplete'):runner.main()
        assert len(calls)==1
    else:
        assert runner.main()==1
        assert [c[c.index('--gpu')+1] for c in calls[:3]]==['L4','A10G','H100']
        assert len(calls)==4
        assert len(list((repo/'reports/kernel_matrix_v1/runs').glob('*/matrix.json')))==1

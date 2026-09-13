"""A fresh replay must match signed data, sources, inventory and device runtime."""
import copy
from pathlib import Path
import sys
import pytest
from test_kernel_matrix_v1 import captures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import kernel_certificate_replay_v2 as replay

@pytest.mark.parametrize('mutation',('none','output','input','source','driver','warp','arch','missing_row','failed_gate'))
def test_fresh_binding(monkeypatch,mutation):
    data=captures(monkeypatch)
    for c in data:c['arch']=89
    data.append(copy.deepcopy(data[0]));data[-1]['device']='NVIDIA GeForce RTX 5070, driver'
    payloads,sources=replay.cert.payloads(data);fresh=copy.deepcopy(data[0])
    if mutation=='output':
        for leg in fresh['selftests'][0]['legs']:leg['records'][0]['sha256']='f'*64
    if mutation=='input':fresh['sites'][0]['input_hashes']=[{'sha256':'e'*64}]
    if mutation=='source':fresh['source_hashes']['frozen.py']='d'*64
    if mutation=='driver':fresh['device']=fresh['device'].split(',')[0]+', new driver'
    if mutation=='warp':fresh['warp']='0.0'
    if mutation=='arch':fresh['arch']=120
    if mutation=='missing_row':fresh['selftests'].pop()
    if mutation=='failed_gate':fresh['gates']['passed']=False
    report=replay.compare_fresh(fresh,payloads,sources)
    assert all(report['gates'].values())==(mutation in ('none','driver'))


def test_failed_rerun_removes_stale_success(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();(out/'verification.json').write_text('{"status":"VERIFIED-FRESH"}')
    monkeypatch.setattr(sys,'argv',['replay','--trusted-key',str(tmp_path/'missing.pem'),'--output',str(out)])
    def refuse(*args,**kwargs):raise ValueError('Untrusted bundle')
    monkeypatch.setattr(replay.cert,'verify',refuse)
    with pytest.raises(ValueError,match='Untrusted'):replay.main()
    assert not (out/'verification.json').exists()

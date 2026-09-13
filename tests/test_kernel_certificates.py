"""Cryptographic tamper controls use synthetic fixture records, never a GPU."""
import base64
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from test_kernel_matrix_v1 import captures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import kernel_certificates_v1 as cert

@pytest.fixture
def bundle(tmp_path,monkeypatch):
    data=captures(monkeypatch)
    data.append(copy.deepcopy(data[0]));data[-1]['device']='NVIDIA GeForce RTX 5070, driver'
    for c in data:c['arch']=89
    monkeypatch.setattr(cert,'observations',lambda:data)
    key=tmp_path/'private.pem'
    key.write_bytes(cert.run(['openssl','genpkey','-algorithm','ED25519']))
    out=tmp_path/'bundle';cert.build(out,key)
    return out,key


def test_two_builds_exact_and_evidence_verified(bundle,tmp_path):
    out,key=bundle;other=tmp_path/'repeat';cert.build(other,key)
    assert {p.name:p.read_bytes() for p in out.iterdir()}=={p.name:p.read_bytes() for p in other.iterdir()}
    assert len(cert.verify(out,out/'public_key.pem'))==20


@pytest.mark.parametrize('mutation',('payload','signature','wrong_key','missing_row','duplicate_row','source_manifest','signed_false_gate'))
def test_corruption_refused(bundle,tmp_path,mutation):
    out,key=bundle;trusted=out/'public_key.pem';path=out/'00.json';e=json.loads(path.read_text())
    if mutation=='payload':e['payload']['devices'][0]['driver']='tampered'
    if mutation=='signature':e['signature']=base64.b64encode(bytes(64)).decode()
    if mutation=='wrong_key':
        other=tmp_path/'other.pem';other.write_bytes(cert.run(['openssl','genpkey','-algorithm','ED25519']))
        trusted=tmp_path/'other_public.pem';trusted.write_bytes(cert.run(['openssl','pkey','-in',str(other),'-pubout']))
    if mutation in ('missing_row','duplicate_row'):
        index=json.loads((out/'index.json').read_text())
        if mutation=='missing_row':index.pop()
        else:index[-1]=index[0]
        (out/'index.json').write_text(json.dumps(index))
    if mutation=='source_manifest':(out/'sources.json').write_text('{}')
    if mutation=='signed_false_gate':
        e['payload']['devices'][0]['own_gates']=False
        e['signature']=base64.b64encode(cert.crypto(cert.canonical(e['payload']),key)).decode()
    path.write_text(json.dumps(e))
    with pytest.raises((ValueError,subprocess.CalledProcessError)):cert.verify(out,trusted)

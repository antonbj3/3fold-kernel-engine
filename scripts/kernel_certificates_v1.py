"""Sign and verify finite four-device observations; OpenSSL Ed25519 is required."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src/kernel_engine/certified_kernels'))
import kernel_matrix_local_report_v1 as matrix
import kernel_matrix_report_v1 as audit
DEFAULT=ROOT/'reports/kernel_certificates_v1'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def run(args):return subprocess.run(args,capture_output=True,check=True).stdout


def crypto(data,key,signature=None):
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp);(folder/'message').write_bytes(data)
        args=['openssl','pkeyutl','-rawin','-in',str(folder/'message'),'-inkey',str(key)]
        if signature is None:return run(args+['-sign'])
        (folder/'signature').write_bytes(signature)
        run(args+['-verify','-pubin','-sigfile',str(folder/'signature')])


def observations():
    folders=[ROOT/'reports/kernel_matrix_v1'/s for s in ('l4','a10g','h100')]+[ROOT/'reports/kernel_matrix_local_v1']
    captures=[audit.load(p) for p in folders]
    if not all(matrix.compare(captures)['gates'].values()):raise ValueError('Uncertified source matrix')
    return captures


def input_identity(key,row,sources):
    if key.startswith('selftest:'):
        return dict(kind='deterministic_fixture_definition',sha256=sha(canonical(dict(slot=key,sources=sources,
                    worker='fixed_order_selftests_v1',mode='RUN_TO_RUN'))),
                    limitation='Pinned deterministic source-defined fixture; raw input arrays were not captured separately')
    if key.startswith('int64:'):
        return dict(kind='observed_input_array_manifest',sha256=sha(canonical(row['records'][0]['inputs'])))
    return dict(kind='observed_input_bytes',sha256=row['records'][0]['input_sha256'])


def payloads(captures):
    sources=captures[0]['source_hashes'];tables=[audit.rows(c) for c in captures]
    output=[]
    for key in sorted(tables[0]):
        devices=[]
        for c,t in zip(captures,tables):
            model,driver=c['device'].split(',',1);row=t[key]
            devices.append(dict(device=model.strip(),driver=driver.strip(),arch=c['arch'],warp=c['warp'],
                                input=input_identity(key,row,sources),output_sha256=sha(canonical(row['records'])),
                                output_kind=row['kind'],own_gates=row['valid']))
        output.append(dict(schema=1,row=key,status='VERIFIED-FRESH',source_manifest_sha256=sha(canonical(sources)),
            devices=devices,exact_across_devices=True,
            scope='Finite frozen fixture observations, not universal determinism or performance; eight selftests, ten reduction sites and two export sizes are twenty rows, not twenty distinct modules'))
    return output,sources


def validate_payload(p):
    if p['schema']!=1 or p['status']!='VERIFIED-FRESH' or not p['exact_across_devices']:raise ValueError('Invalid certificate status')
    devices=p['devices']
    if len(devices)!=4 or {d['device'] for d in devices}!=audit.EXPECTED_DEVICES|{'NVIDIA GeForce RTX 5070'}:raise ValueError('Incomplete devices')
    if not all(d['own_gates'] and d['warp']=='1.17.0' for d in devices):raise ValueError('Failed numerical/runtime gate')
    if len({d['output_sha256'] for d in devices})!=1 or len({canonical(d['input']) for d in devices})!=1:raise ValueError('Nonidentical observations')


def verify_file(path,trusted_key):
    envelope=json.loads(path.read_text())
    if envelope['algorithm']!='Ed25519' or envelope['key_sha256']!=sha(trusted_key.read_bytes()):raise ValueError('Untrusted key or algorithm')
    crypto(canonical(envelope['payload']),trusted_key,base64.b64decode(envelope['signature'],validate=True))
    validate_payload(envelope['payload']);return envelope['payload']


def build(directory,private_key):
    directory.mkdir(parents=True,exist_ok=True)
    public=run(['openssl','pkey','-in',str(private_key),'-pubout'])
    values,sources=payloads(observations());index=[]
    for i,value in enumerate(values):
        validate_payload(value);name=f'{i:02d}.json'
        envelope=dict(algorithm='Ed25519',key_sha256=sha(public),payload=value,
                      signature=base64.b64encode(crypto(canonical(value),private_key)).decode())
        (directory/name).write_bytes(canonical(envelope)+b'\n');index.append(dict(row=value['row'],file=name))
    (directory/'public_key.pem').write_bytes(public)
    (directory/'sources.json').write_bytes(canonical(sources)+b'\n')
    (directory/'index.json').write_bytes(canonical(index)+b'\n')
    return index


def verify(directory,trusted_key,evidence=True):
    index=json.loads((directory/'index.json').read_text())
    sources=json.loads((directory/'sources.json').read_text())
    expected,original=payloads(observations()) if evidence else (None,sources)
    if sources!=original:raise ValueError('Stored source manifest differs from evidence')
    rows=[]
    for entry in index:
        name=entry['file']
        if Path(name).name!=name:raise ValueError('Invalid certificate filename')
        value=verify_file(directory/name,trusted_key)
        if value['row']!=entry['row'] or value['source_manifest_sha256']!=sha(canonical(sources)):raise ValueError('Manifest mismatch')
        rows.append(value)
    expected_keys={'selftest:'+s for s in audit.EXPECTED_SLOTS}|{'int64:'+m+':'+s for m,s in audit.EXPECTED_SITES}|{'export:512','export:1024'}
    if len(rows)!=20 or {p['row'] for p in rows}!=expected_keys:raise ValueError('Incomplete certificate inventory')
    if evidence and rows!=expected:raise ValueError('Certificate differs from retained evidence')
    return rows


def replay(directory,trusted_key,row):
    values=verify(directory,trusted_key);certificate=next(p for p in values if p['row']==row)
    sources=json.loads((directory/'sources.json').read_text())
    if any(sha((ROOT/path).read_bytes())!=value for path,value in sources.items()):raise ValueError('Frozen source drift')
    # Replay the two original independent native/Warp export legs, including both
    # sizes; only the selected row is certified by this command.
    if row not in ('export:512','export:1024'):raise ValueError('Replay currently supports the native export rows only')
    import kernel_matrix_capture_v1 as capture
    device=run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']).decode().strip()
    meta=run([sys.executable,'-c',"import warp as w;w.init();print('META',w.config.version,w.get_device('cuda:0').arch)"]).decode()
    _,version,arch=next(x for x in meta.splitlines() if x.startswith('META ')).split()
    if version!='1.17.0':raise ValueError('Runtime mismatch')
    exports=capture.run_export(arch);side=int(row.split(':')[1]);records=[]
    fields=('side','input_sha256','native_sha256','warp_sha256','expected_sha256','bytes')
    for leg in exports:
        e=next(e for e in leg if e['side']==side)
        if not e['exact_reference'] or not e['exact_replays'] or len({e[k] for k in ('native_sha256','warp_sha256','expected_sha256')})!=1:raise ValueError('Replay numerical gate failed')
        records.append({k:e[k] for k in fields})
    model,driver=device.split(',',1);reference=next(d for d in certificate['devices'] if d['device']==model.strip())
    gates=dict(signature_and_retained_evidence=True,source_identity=True,runtime=version==reference['warp'] and int(arch)==reference['arch'],
               own_numerical_gates=True,independent_repeat=records[0]==records[1],
               input_identity=records[0]['input_sha256']==reference['input']['sha256'],
               output_identity=sha(canonical([records[0]]))==reference['output_sha256'])
    result=dict(row=row,device=model.strip(),driver=driver.strip(),warp=version,arch=int(arch),gates=gates,
                records=records,certificate_sha256=sha(canonical(certificate)),
                scope='Fresh selected-row replay; other certificates verified against retained four-device observations, not freshly rerun')
    (directory/'replay.json').write_bytes(canonical(result)+b'\n')
    print(json.dumps(gates));return int(not all(gates.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','verify','replay'])
    p.add_argument('--directory',type=Path,default=DEFAULT);p.add_argument('--private-key',type=Path)
    p.add_argument('--trusted-key',type=Path);p.add_argument('--row',default='export:512');p.add_argument('--signature-only',action='store_true');a=p.parse_args()
    if a.action=='build':
        if a.private_key is None:p.error('--private-key required')
        print('Built',len(build(a.directory,a.private_key)),'certificates');return 0
    if a.trusted_key is None:p.error('Supply an independently trusted --trusted-key; embedded keys are not automatically trusted')
    if a.action=='replay':return replay(a.directory,a.trusted_key,a.row)
    print('Verified',len(verify(a.directory,a.trusted_key,not a.signature_only)),'certificates');return 0

if __name__=='__main__':raise SystemExit(main())

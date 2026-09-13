"""Fresh complete twenty-row replay of signed kernel observations on one GPU."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import kernel_certificates_v1 as cert
ROOT=Path(__file__).resolve().parents[1]


def compare_fresh(capture,payloads,sources):
    table=cert.audit.rows(capture);expected={p['row']:p for p in payloads}
    device=capture['device'].split(',')[0].strip();rows=[]
    source_ok=all(capture['source_hashes'].get(k)==v for k,v in sources.items())
    for key,payload in expected.items():
        reference=next((d for d in payload['devices'] if d['device']==device),None);row=table.get(key)
        gates=dict(device_and_runtime=reference is not None and capture['warp']==reference['warp'] and capture['arch']==reference['arch'],
                   numerical_and_repeat=row is not None and row['valid'],
                   input_identity=reference is not None and row is not None and cert.input_identity(key,row,sources)==reference['input'],
                   output_identity=reference is not None and row is not None and cert.sha(cert.canonical(row['records']))==reference['output_sha256'])
        rows.append(dict(row=key,gates=gates,observed_output_sha256=cert.sha(cert.canonical(row['records'])) if row else None,
                         certificate_payload_sha256=cert.sha(cert.canonical(payload))))
    gates=dict(complete_inventory=len(expected)==20 and len(payloads)==20 and set(table)==set(expected),
               original_source_identity=source_ok,capture_own_gates=all(capture['gates'].values()),
               all_rows_match=all(all(r['gates'].values()) for r in rows))
    return dict(schema=2,gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',device=capture['device'],
        warp=capture['warp'],arch=capture['arch'],rows=rows,
        additional_captured_sources={k:v for k,v in capture['source_hashes'].items() if k not in sources},
        scope='Fresh complete certificate-row replay on the recorded device; original four-device observations remain signed and unchanged. Extra captured observer sources are listed separately. Selftest input identity remains the source-defined fixture, not a newly recorded raw-input stream.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trusted-key',type=Path,required=True)
    parser.add_argument('--directory',type=Path,default=cert.DEFAULT)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/kernel_certificate_replay_v2')
    parser.add_argument('--capture',type=Path,help='Audit an existing freshly captured folder on CPU instead of launching GPU work')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # A failed rerun must not leave a prior successful verification visible.
    (args.output/'verification.json').unlink(missing_ok=True)
    payloads=cert.verify(args.directory,args.trusted_key)
    sources=json.loads((args.directory/'sources.json').read_text())
    if any(cert.sha((ROOT/p).read_bytes())!=h for p,h in sources.items()):raise ValueError('Original source drift before replay')
    if args.capture is None:
        script=ROOT/'src/kernel_engine/certified_kernels/kernel_matrix_capture_v1.py'
        print('Trusted certificates and frozen sources verified; starting all frozen capture workers',flush=True)
        p=subprocess.run([sys.executable,str(script)],cwd=ROOT,capture_output=True,timeout=1800)
        (args.output/'capture_stdout.txt').write_bytes(p.stdout)
        (args.output/'capture_stderr.txt').write_bytes(p.stderr)
        if p.returncode:raise RuntimeError('Fresh capture failed; logs retained, no success report issued')
        source=ROOT/'reports/kernel_matrix_v1'
    else:source=args.capture
    for name in ('capture.json.gz','summary.json'):
        if (source/name).resolve()!=(args.output/name).resolve():shutil.copyfile(source/name,args.output/name)
    fresh=cert.audit.load(args.output);report=compare_fresh(fresh,payloads,sources)
    report['trusted_key_sha256']=cert.sha(args.trusted_key.read_bytes())
    report['capture_sha256']=cert.sha(gzip.decompress((args.output/'capture.json.gz').read_bytes()))
    report['runner_sha256']=cert.sha(Path(__file__).read_bytes())
    (args.output/'verification.json').write_bytes(cert.canonical(report)+b'\n')
    print(json.dumps(dict(gates=report['gates'],status=report['status'],rows=len(report['rows']))),flush=True)
    return int(not all(report['gates'].values()))

if __name__=='__main__':raise SystemExit(main())

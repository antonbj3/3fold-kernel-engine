"""Audit cloud kernel capsules without importing Warp or opening a GPU context."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

EXPECTED_SLOTS={
 'lbm/differentiable_flow_control','lbm/differentiable_lbm_probe','lbm/differentiable_fsi_chain','lbm/lbm3d_immersed_boundary',
 'wave_fdtd/diff_wave_3d','wave_fdtd/diff_wave_substrate','wave_fdtd/xray_tomography_sigma','wave_fdtd/xray_3d_dda'}
EXPECTED_COUNTS=dict(zip(('lbm/differentiable_flow_control','lbm/differentiable_lbm_probe','lbm/differentiable_fsi_chain','lbm/lbm3d_immersed_boundary','wave_fdtd/diff_wave_3d','wave_fdtd/diff_wave_substrate','wave_fdtd/xray_tomography_sigma','wave_fdtd/xray_3d_dda'),(28269,402,84608,2,910,1384,5622,183)))
EXPECTED_SITES={('wave_fdtd/xray_tomography_sigma', 'sensitivity'), ('wave_fdtd/diff_wave_substrate', 'focus_loss'), ('lbm/differentiable_fsi_chain', 'drag_force'), ('lbm/differentiable_lbm_probe', 'objective'), ('lbm/differentiable_flow_control', 'probe_ux'), ('lbm/lbm3d_immersed_boundary', 'spread'), ('wave_fdtd/xray_tomography_sigma', 'sq_resid'), ('wave_fdtd/xray_3d_dda', 'backproject'), ('lbm/differentiable_flow_control', 'track_loss'), ('wave_fdtd/diff_wave_3d', 'energy')}
EXPECTED_DEVICES={'NVIDIA L4','NVIDIA A10','NVIDIA H100 80GB HBM3'}


def load(folder):
    data=gzip.decompress((folder/'capture.json.gz').read_bytes())
    summary=json.loads((folder/'summary.json').read_text())
    if hashlib.sha256(data).hexdigest()!=summary['capture_sha256']:raise ValueError('Capsule checksum mismatch')
    result=json.loads(data)
    if any(result[k]!=summary[k] for k in ('device','arch','warp','gates')):raise ValueError('Capsule summary mismatch')
    return result


def rows(capture):
    output={}
    for s in capture['selftests']:
        if len(s['legs'])!=2:raise ValueError('Expected two independent selftest legs')
        key='selftest:'+s['slot']
        if key in output:raise ValueError('Duplicate selftest slot')
        output[key]=dict(records=s['legs'][0]['records'],
            valid=all(s['gates'].values()) and s['legs'][0]['records']==s['legs'][1]['records'] and all(
                x['returncode']==0 and not x['exception_classes'] and x['finite'] and len(x['records'])==EXPECTED_COUNTS[s['slot']] and all(r['ordinal']==i for i,r in enumerate(x['records'])) for x in s['legs']),
            kind='array_record_sequence')
    for s in capture['sites']:
        key='int64:'+s['module']+':'+s['site']
        if key in output:raise ValueError('Duplicate integer site')
        if len(s['legs'])!=2:raise ValueError('Expected two integer launches')
        output[key]=dict(records=[dict(kernel=s['kernel'],inputs=s['input_hashes'],output=s['legs'][0])],
            valid=s['exact'] and s['int64_delta']==0 and s['legs'][0]==s['legs'][1] and all(leg['dtype']=='int64' and list(np.asarray(leg['values'],dtype=np.int64).shape)==leg['shape'] and hashlib.sha256(np.asarray(leg['values'],dtype=np.int64).tobytes()).hexdigest()==leg['sha256'] for leg in s['legs']),kind='int64')
    if len(capture['exports'])!=2:raise ValueError('Expected two export workers')
    a,b=capture['exports']
    if {x['side'] for x in a}!={512,1024} or {x['side'] for x in b}!={512,1024} or len(a)!=2 or len(b)!=2:
        raise ValueError('Incomplete export sizes')
    by_side={x['side']:x for x in b}
    for e in a:
        other=by_side[e['side']];fields=('side','input_sha256','native_sha256','warp_sha256','expected_sha256','bytes')
        signature={k:e[k] for k in fields}
        output['export:'+str(e['side'])]=dict(records=[signature],kind='exact_oracle_hash',valid=
            all(x['exact_reference'] and x['exact_replays'] and x['native_sha256']==x['warp_sha256']==x['expected_sha256'] for x in (e,other)) and
            signature=={k:other[k] for k in fields})
    return output


def compare(captures):
    if not captures:raise ValueError('No captures')
    tables=[rows(c) for c in captures];keys=set(tables[0])
    coverage=len(captures)==3 and all(set(t)==keys and len(t)==20 for t in tables) and all(
        {s['slot'] for s in c['selftests']}==EXPECTED_SLOTS and len(c['selftests'])==8 and len(c['sites'])==10 and {(s['module'],s['site']) for s in c['sites']}==EXPECTED_SITES for c in captures)
    devices=[c['device'].split(',')[0] for c in captures]
    exact_devices=set(devices)==EXPECTED_DEVICES and len(set(devices))==3
    source_identity=all(c['source_hashes']==captures[0]['source_hashes'] and c['warp']=='1.17.0' for c in captures)
    matrix=[]
    for key in sorted(keys):
        comparisons=[]
        for i,j in itertools.combinations(range(len(tables)),2):
            a=tables[i].get(key);b=tables[j].get(key)
            equal=a is not None and b is not None and a['records']==b['records']
            first=None
            if a is not None and b is not None and not equal:
                for n,(x,y) in enumerate(itertools.zip_longest(a['records'],b['records'])):
                    if x!=y:first=dict(ordinal=n,left=x,right=y);break
            delta=0. if equal else None
            if not equal and a is not None and b is not None and a['kind']==b['kind']=='int64':
                left=a['records'][0];right=b['records'][0]
                av=np.asarray(left['output']['values'],dtype=np.int64);bv=np.asarray(right['output']['values'],dtype=np.int64)
                if av.shape==bv.shape:
                    delta=int(np.max(np.abs(av.astype(object)-bv.astype(object))))
                first=dict(left_sha256=left['output']['sha256'],right_sha256=right['output']['sha256'],
                    input_identity=left['inputs']==right['inputs'],left_shape=list(av.shape),right_shape=list(bv.shape))
            comparisons.append(dict(left=devices[i],right=devices[j],exact=equal,
                max_absolute_delta=delta,first_difference=first))
        matrix.append(dict(module=key,own_gates=all(t.get(key,{}).get('valid',False) for t in tables),comparisons=comparisons))
    gates=dict(full_coverage=coverage,three_distinct_skus=exact_devices,source_and_version_identity=source_identity,
        all_own_numerical_gates=all(all(c['gates'].values()) for c in captures) and all(r['own_gates'] for r in matrix),
        cross_sku_exact=coverage and all(all(p['exact'] for p in r['comparisons']) for r in matrix))
    return dict(schema=1,scope='E kernel-family cloud observations; no all-repository or local-SKU claim',gates=gates,
        status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',unmeasured_devices=['RTX 5070'],rows=matrix,
        export_performance=[dict(device=c['device'],evidence=c['export_performance']) for c in captures],
        delta_convention='Zero from exact full-byte hashes; int64 differences computed from stored values; other nonidentical records require raw-array replay')


def main():
    if len(sys.argv)!=4:raise SystemExit('usage: kernel_matrix_report_v1.py L4_DIR A10_DIR H100_DIR')
    report=compare([load(Path(p)) for p in sys.argv[1:]])
    path=Path('reports/kernel_matrix_v1/matrix.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    lines=['# Kernel cloud determinism matrix','', '| Module | Own gates | L4 / A10 / H100 observed max delta |','|---|---|---|']
    for row in report['rows']:
        exact=all(p['exact'] for p in row['comparisons'])
        lines.append('| '+row['module']+' | '+('PASS' if row['own_gates'] else 'FAIL')+' | '+('0' if exact else 'DIFF; see first records in JSON')+' |')
    lines+=['','Local RTX 5070: unmeasured. Export performance criteria are reported separately in JSON.','']
    (path.parent/'matrix.md').write_text('\n'.join(lines))
    print(json.dumps(dict(gates=report['gates'],status=report['status'],rows=len(report['rows']))))
    return 0 if all(report['gates'].values()) else 1


if __name__=='__main__':raise SystemExit(main())

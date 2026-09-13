"""Assemble split local observations and audit four devices without a GPU context."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/kernel_engine/certified_kernels'))
import kernel_matrix_report_v1 as audit
OUT=ROOT/'reports/kernel_matrix_local_v1'


def compare(captures):
    report=audit.compare(captures)
    tables=[audit.rows(c) for c in captures]
    coverage=len(captures)==4 and all(len(t)==20 for t in tables) and all(
        len(c['selftests'])==8 and {s['slot'] for s in c['selftests']}==audit.EXPECTED_SLOTS and
        len(c['sites'])==10 and {(s['module'],s['site']) for s in c['sites']}==audit.EXPECTED_SITES for c in captures)
    devices=[c['device'].split(',')[0] for c in captures]
    gates=report['gates'];gates.pop('three_distinct_skus')
    gates['four_distinct_skus']=len(set(devices))==4 and set(devices)==audit.EXPECTED_DEVICES|{'NVIDIA GeForce RTX 5070'}
    gates['full_coverage']=coverage
    gates['cross_sku_exact']=coverage and all(p['exact'] for row in report['rows'] for p in row['comparisons'])
    report.update(scope='Frozen kernel-family observations on L4, A10, H100 and RTX 5070; no other repository or universal-input claim',
                  unmeasured_devices=[],status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    return report


def assemble():
    cloud=[audit.load(ROOT/'reports/kernel_matrix_v1'/sku) for sku in ('l4','a10g','h100')]
    parts=[]
    def part(name):
        p=json.loads(gzip.decompress((OUT/'parts'/(name+'.json.gz')).read_bytes()));parts.append(p);return p['value']
    suite=[]
    for original in cloud[0]['selftests']:
        slot=original['slot'];legs=[part(slot.replace('/','_')+'_'+str(i)) for i in range(2)]
        gates=dict(selftests=all(x['returncode']==0 and not x['exception_classes'] for x in legs),
                   exact_arrays=legs[0]['records']==legs[1]['records'],finite=all(x['finite'] for x in legs),
                   coverage=all(len(x['records'])==audit.EXPECTED_COUNTS[slot] for x in legs))
        suite.append(dict(slot=slot,legs=legs,gates=gates))
    sites=part('sites');exports=part('exports')
    first=parts[0]
    for p in parts:
        if any(p[k]!=first[k] for k in ('device','warp','arch','source_hashes','observer_sha256')):raise ValueError('Part identity drift')
    gates=dict(eight_slots=len(suite)==8,all_slot_gates=all(all(s['gates'].values()) for s in suite),
        int64_sites=len(sites)==10 and all(s['exact'] and s['int64_delta']==0 for s in sites),
        export_oracle=all(r['exact_reference'] and r['exact_replays'] for leg in exports for r in leg),
        export_repeat=all(a['native_sha256']==b['native_sha256'] and a['warp_sha256']==b['warp_sha256'] for a,b in zip(*exports)))
    local={k:first[k] for k in ('device','warp','arch','source_hashes','observer_sha256')}
    local.update(schema=1,selftests=suite,sites=sites,exports=exports,gates=gates,
                 export_performance=dict(scope='Uncontrolled desktop background; timings diagnostic only, no local performance acceptance claim'))
    data=json.dumps(local,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    (OUT/'capture.json.gz').write_bytes(gzip.compress(data,mtime=0))
    summary={k:local[k] for k in ('device','warp','arch','gates')};summary['capture_sha256']=hashlib.sha256(data).hexdigest()
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=compare(cloud+[local]);(OUT/'matrix.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    lines=['# Four-device kernel matrix','','| Module | Own gates | All six device pairs |','|---|---|---|']
    for row in report['rows']:
        lines.append('| '+row['module']+' | '+('PASS' if row['own_gates'] else 'FAIL')+' | '+('exact, delta 0' if all(p['exact'] for p in row['comparisons']) else 'DIFF: see JSON')+' |')
    (OUT/'matrix.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=report['gates'],status=report['status'])))
    return int(not all(report['gates'].values()))

if __name__=='__main__':raise SystemExit(assemble())

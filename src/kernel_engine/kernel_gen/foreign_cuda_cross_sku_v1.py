"""Audit complete foreign-kernel certificate equality across two GPU SKUs."""
import hashlib
import json
import math
from pathlib import Path
import sys

EXPECTED={(n,c) for n in (0,1,255,256,257,65539,16777219) for c in ('add','zero','cancel')}


def audit(certificates,measurements):
    coverage=all(len(c['cases'])==21 and {(r['n'],r['case']) for r in c['cases']}==EXPECTED for c in certificates)
    numerical=all(c['gates']==dict(G1_reference=True,G2_repeat=True,G3_null_and_bounds=True)
        and c['status']=='VERIFIED-FRESH' and all(r['oracle_exact'] and r['repeat_exact'] and r['guard_intact'] for r in c['cases']) for c in certificates)
    identity=len(certificates)==2 and certificates[0]==certificates[1]
    skus=len(measurements)==2 and len({m['device'].split(',')[0] for m in measurements})==2
    timings=all(len(m['timings'])==21 and {(r['n'],r['case']) for r in m['timings']}==EXPECTED and all(len(r['legs'])==2 and all(v['kernel_ms']>0 and v['copy_ms']>0 and
        v['useful_gbps']>0 and v['bandwidth_fraction']>0 and all(math.isfinite(x) for x in v.values()) for v in r['legs']) for r in m['timings'] if r['n']>0) for m in measurements)
    gates=dict(complete_coverage=coverage,three_numerical_gates=numerical,cross_sku_exact=identity,two_distinct_skus=skus,bandwidth_recorded=timings)
    return dict(gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
        certificate_sha256=[hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest() for c in certificates],
        devices=[m['device'] for m in measurements],bandwidth=[dict(device=m['device'],rows=[r for r in m['timings'] if r['n']==16777219]) for m in measurements])


if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: foreign_cuda_cross_sku_v1.py REPORT_DIR_A REPORT_DIR_B')
    paths=[Path(x) for x in sys.argv[1:]]
    result=audit([json.loads((p/'certificate.json').read_text()) for p in paths],
                 [json.loads((p/'measurement.json').read_text()) for p in paths])
    target=Path('reports/foreign_cuda_cross_sku_v1.json');target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if all(result['gates'].values()) else 1)

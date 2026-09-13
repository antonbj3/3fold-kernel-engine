"""Audit the independent dispatch interventions from complete captured arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np
import roundoff_dispatch_v1 as probe

ROOT=Path(__file__).resolve().parents[1]
CASES=[a+'_'+b for a,b in probe.CASES]


def load_case(root,case):
    legs=[];records=[]
    for i in range(2):
        record=json.loads((root/(case+'_'+str(i)+'.json')).read_text())
        with np.load(root/(case+'_'+str(i)+'.npz'),allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
        if set(arrays)!=set(record['arrays']):raise ValueError('Incomplete array inventory')
        for k,a in arrays.items():
            m=record['arrays'][k]
            if hashlib.sha256(a.tobytes()).hexdigest()!=m['sha256'] or list(a.shape)!=m['shape'] or str(a.dtype)!=m['dtype']:raise ValueError('Array mismatch')
        if not all(record['gates'].values()):raise ValueError('Worker gate failure')
        if record['requested']!=dict(zip(('numpy','scipy'),case.split('_'))):raise ValueError('Case identity mismatch')
        for lib,requested in record['requested'].items():
            if record['libraries'][lib]['num_threads']!=1 or requested!='native' and record['libraries'][lib]['architecture']!=requested:raise ValueError('Dispatch contract mismatch')
        legs.append(arrays);records.append(record)
    if records[0]!=records[1] or any(legs[0][k].tobytes()!=legs[1][k].tobytes() for k in legs[0]):raise ValueError('Independent repeat mismatch')
    return records[0],legs[0]


def compare(local,cloud):
    data={host:{case:load_case(root,case) for case in CASES} for host,root in [('local',local),('cloud',cloud)]}
    records=[r for host in data.values() for r,a in host.values()]
    if any(r['sources']!=records[0]['sources'] or r['probe_sha256']!=records[0]['probe_sha256'] or r['observer_sha256']!=records[0]['observer_sha256'] for r in records):raise ValueError('Source drift')
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in records[0]['sources'].items()):raise ValueError('Frozen source drift')
    if hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()!=records[0]['probe_sha256']:raise ValueError('Probe source drift')
    if hashlib.sha256((ROOT/'scripts/roundoff_environment_v1.py').read_bytes()).hexdigest()!=records[0]['observer_sha256']:raise ValueError('Observer source drift')
    c={case:pair[1] for case,pair in data['cloud'].items()};l=data['local']['Haswell_Haswell'][1]
    def same(a,b,names):return all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in names)
    solve=['own_Cmat','own_Umat'];fixed=['cloud_solves_Cmat','cloud_solves_Umat','cloud_solves_B','cloud_solves_state_times_C','cloud_solves_correction']
    gates=dict(complete_arrays=all(set(a)==set(l) and len(a)==56 for host in data.values() for r,a in host.values()),
        fixed_builds_within_hosts=all(len({tuple(r['libraries'][k]['binary_sha256'] for k in ('numpy','scipy')) for r,a in host.values()})==1 for host in data.values()),
        scipy_binary_shared_across_hosts=data['local']['native_native'][0]['libraries']['scipy']['binary_sha256']==data['cloud']['native_native'][0]['libraries']['scipy']['binary_sha256'],
        scipy_switch_isolates_solves=same(c['native_native'],c['Haswell_native'],solve) and same(c['native_Haswell'],c['Haswell_Haswell'],solve) and not same(c['native_native'],c['native_Haswell'],solve),
        fixed_endpoint_operands=all(same(c['native_native'],a,fixed) for a in c.values()),
        numpy_switch_isolates_forcing=same(c['native_native'],c['native_Haswell'],['cloud_solves_forcing']) and same(c['Haswell_native'],c['Haswell_Haswell'],['cloud_solves_forcing']) and not same(c['native_native'],c['Haswell_native'],['cloud_solves_forcing']),
        matched_dispatch_crosshost_exact=same(l,c['Haswell_Haswell'],l),
        native_difference_retained=not same(l,c['native_native'],l))
    rows=[]
    for host,cases in data.items():
        for case,(r,a) in cases.items():
            rows.append(dict(host=host,case=case,libraries=r['libraries'],environment=r['environment'],error=r['error'],
                own_endpoint=a['own_S_end'].tolist(),fixed_forcing=a['cloud_solves_forcing'].tolist(),fixed_endpoint=a['cloud_solves_S_end'].tolist(),
                arrays_differing_from_local=sum(l[k].tobytes()!=a[k].tobytes() for k in l)))
    return dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
        arrays_per_leg=56,independent_workers_per_host=8,matched_dispatch_arrays=56,
        comparator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Within-host library binaries and frozen inputs are fixed while actual NumPy/SciPy dispatch is varied separately. This establishes dispatch dependence for the captured fixture, not a unique instruction-level explanation or universal CPU determinism. Original native-environment failures remain unchanged.')


def main():
    root=ROOT/'reports/roundoff_dispatch_v1';report=compare(root,root/'cloud')
    (root/'comparison.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],gates=report['gates'])))
    return int(not all(report['gates'].values()))

if __name__=='__main__':raise SystemExit(main())

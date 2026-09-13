"""Audit two captured environments and transplant only the observed solve outputs."""
import json
from pathlib import Path
import numpy as np
import roundoff_environment_v1 as observer

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    report=json.loads((path/'report.json').read_text())
    with np.load(path/'arrays.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    if set(arrays)!=set(report['arrays']):raise ValueError('Incomplete array inventory')
    for key,a in arrays.items():
        expected=report['arrays'][key]
        if observer.sha(a.tobytes())!=expected['sha256'] or list(a.shape)!=expected['shape'] or str(a.dtype)!=expected['dtype']:
            raise ValueError('Array evidence mismatch')
    return report,arrays


def transplant(inputs,source):
    original=observer.frozen.solve_triangular
    values=iter([source['Cmat'].T.copy(),source['Umat'].T.copy()])
    try:
        observer.frozen.solve_triangular=lambda *a,**kw:next(values)
        result=observer.frozen._chunk_forward(*inputs)
    finally:observer.frozen.solve_triangular=original
    return result


def compare(local,remote):
    a,x=load(local);b,y=load(remote)
    if a['sources']!=b['sources'] or a['observer_sha256']!=b['observer_sha256'] or a['stage_order']!=b['stage_order']:
        raise ValueError('Capture source or stage drift')
    if observer.sha(Path(observer.__file__).read_bytes())!=a['observer_sha256']:raise ValueError('Observer source drift')
    if any(observer.sha((ROOT/p).read_bytes())!=v for p,v in a['sources'].items()):raise ValueError('Frozen source drift')
    rows=[]
    for key in a['stage_order']+['sequential_output','sequential_state']:
        if x[key].shape!=y[key].shape or x[key].dtype!=y[key].dtype:raise ValueError('Array schema drift')
        equal=x[key].tobytes()==y[key].tobytes();row=dict(stage=key,exact=equal)
        if not equal:
            bits_x=np.ascontiguousarray(x[key]).view(np.uint8).reshape(x[key].shape+(x[key].dtype.itemsize,))
            bits_y=np.ascontiguousarray(y[key]).view(np.uint8).reshape(y[key].shape+(y[key].dtype.itemsize,))
            indices=np.argwhere(np.any(bits_x!=bits_y,axis=-1));idx=tuple(indices[0]);row.update(count=len(indices),first_index=list(map(int,idx)),
                local_value=float(x[key][idx]),cloud_value=float(y[key][idx]),max_absolute_delta=float(np.max(np.abs(x[key]-y[key]))))
        rows.append(row)
    first=next((r['stage'] for r in rows if not r['exact']),None)
    inputs=[x[k] for k in observer.STAGES[:6]];control=transplant(inputs,x);transfer=transplant(inputs,y)
    gates=dict(capture_own_gates=all(a['gates'].values()) and all(b['gates'].values()),
        first_difference_is_triangular_solve=first=='Cmat',
        local_injection_control=control[0].tobytes()==x['O'].tobytes() and control[1].tobytes()==x['S_end'].tobytes(),
        output_transplant_exact=transfer[0].tobytes()==y['O'].tobytes(),
        endpoint_transplant_exact=transfer[1].tobytes()==y['S_end'].tobytes(),
        full_environment_identity=all(r['exact'] for r in rows))
    return dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,first_differing_stage=first,rows=rows,
        environments=dict(local=a['environment'],cloud=b['environment']),
        replay_environment=dict(python=observer.platform.python_version(),numpy=np.__version__,scipy=observer.scipy.__version__,machine=observer.platform.machine()),
        transplanted_endpoint=transfer[1].tolist(),cloud_endpoint=y['S_end'].tolist(),
        original_error=dict(local=a['max_output_error'],cloud=b['max_output_error']),
        observer_sha256=a['observer_sha256'],comparison_sha256=observer.sha(Path(__file__).read_bytes()),
        scope='Frozen arithmetic is unchanged. Replacing only the two solve outputs reproduces the cloud output array on the local host, but endpoint disagreement remains. Python, NumPy wheel and CPU dispatch are not independently varied; no unique instruction/library-version diagnosis follows. Original recipe failure is retained.')


def main():
    path=ROOT/'reports/roundoff_environment_v1';report=compare(path,path/'cloud')
    (path/'comparison.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(gates=report['gates'],first=report['first_differing_stage'],endpoint=report['transplanted_endpoint'])))
    return int(not all(report['gates'].values()))

if __name__=='__main__':raise SystemExit(main())

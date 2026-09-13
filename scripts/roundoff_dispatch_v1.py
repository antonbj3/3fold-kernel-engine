"""Separate NumPy and SciPy BLAS dispatch in fresh CPU workers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
CASES=[('native','native'),('Haswell','native'),('native','Haswell'),('Haswell','Haswell')]


def worker(np_core,sp_core,target):
    def select(core):
        if core=='native':os.environ.pop('OPENBLAS_CORETYPE',None)
        else:os.environ['OPENBLAS_CORETYPE']=core
    select(np_core)
    import numpy as np
    from threadpoolctl import threadpool_info
    np.dot(np.ones((2,2)),np.ones((2,2)))
    np_paths={r['filepath'] for r in threadpool_info() if r['internal_api']=='openblas'}
    select(sp_core)
    import scipy.linalg
    import roundoff_environment_v1 as observer
    import roundoff_environment_compare_v1 as comparison
    pools=threadpool_info();libraries={}
    for name,paths in [('numpy',np_paths),('scipy',{r['filepath'] for r in pools}-np_paths)]:
        entries=[r for r in pools if r['filepath'] in paths and r['internal_api']=='openblas']
        if len(entries)!=1:raise ValueError('Exactly one OpenBLAS runtime per library required')
        row=entries[0]
        libraries[name]={k:row[k] for k in ('architecture','version','num_threads')}
        libraries[name]['binary_sha256']=hashlib.sha256(Path(row['filepath']).read_bytes()).hexdigest()
    report,own=observer.observe()
    cloud_report,cloud=comparison.load(ROOT/'reports/roundoff_environment_v1/cloud')
    inputs=[own[k] for k in observer.STAGES[:6]]
    old=observer.frozen.solve_triangular
    values=iter([cloud['Cmat'].T.copy(),cloud['Umat'].T.copy()])
    try:
        observer.frozen.solve_triangular=lambda *a,**kw:next(values)
        _,transfer=observer.capture(inputs)
    finally:observer.frozen.solve_triangular=old
    arrays={};identities=[]
    for label,data in [('own',own),('cloud_solves',transfer)]:
        sc=data['S0']@data['Cmat']
        correction=sc@data['B'];forcing=data['Umat']@data['B']
        remainder=data['S0']-correction;pre=remainder+forcing
        reconstructed=(pre*data['gamma'][-1][None,:]).astype(np.float32)
        terms=dict(state_times_C=sc,correction=correction,forcing=forcing,remainder=remainder,pre_scale=pre,reconstructed_endpoint=reconstructed)
        identities.append(reconstructed.tobytes()==data['S_end'].tobytes())
        arrays.update({label+'_'+k:v for k,v in {**data,**terms}.items()})
    requested=dict(numpy=np_core,scipy=sp_core)
    gates=dict(capture_contracts=all(report['gates'].values()),endpoint_expression_exact=all(identities),
        requested_dispatch_enforced=all(v=='native' or libraries[k]['architecture']==v for k,v in requested.items()),
        one_thread=all(r['num_threads']==1 for r in libraries.values()),
        frozen_sources=report['sources']==cloud_report['sources'],
        identical_cloud_inputs=all(own[k].tobytes()==cloud[k].tobytes() for k in observer.STAGES[:6]))
    result=dict(requested=requested,libraries=libraries,environment=report['environment'],gates=gates,
        error=report['max_output_error'],sources=report['sources'],observer_sha256=report['observer_sha256'],
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arrays={k:dict(sha256=hashlib.sha256(v.tobytes()).hexdigest(),shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})
    Path(target+'.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    np.savez_compressed(target+'.npz',**arrays)
    if not all(gates.values()):raise SystemExit(1)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='--worker':
        worker(*sys.argv[2:]);return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker NUMPY SCIPY TARGET')
    out=ROOT/'reports/roundoff_dispatch_v1';out.mkdir(exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='dispatch-') as tmp:
        for a,b in CASES:
            legs=[]
            for i in range(2):
                name=a+'_'+b+'_'+str(i);target=str(Path(tmp)/name)
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',a,b,target],check=True,timeout=120,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
                legs.append(Path(target+'.json').read_bytes())
                # Each leg carries full arrays, not only comparison summaries.
                for suffix in ('.json','.npz'):(out/(name+suffix)).write_bytes(Path(target+suffix).read_bytes())
            first=json.loads(legs[0]);rows.append(dict(case=a+'_'+b,exact_repeat=legs[0]==legs[1],record=first))
            print(a,b,first['error'],first['libraries'],flush=True)
    gates=dict(four_dispatch_cases=len(rows)==4,all_worker_contracts=all(all(r['record']['gates'].values()) for r in rows),
        independent_repeats=all(r['exact_repeat'] for r in rows),
        library_builds_fixed=len({json.dumps({k:v['binary_sha256'] for k,v in r['record']['libraries'].items()},sort_keys=True) for r in rows})==1)
    result=dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,rows=rows,
        scope='Within-host dispatch intervention with identical library binaries, frozen inputs and numerical source. NumPy is loaded before selecting SciPy dispatch; actual runtime architectures are verified. No original exact-number recipe is changed.')
    (out/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return int(not all(gates.values()))

if __name__=='__main__':raise SystemExit(main())

"""Capture actual frozen recurrence intermediates without rewriting arithmetic."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/kernel_engine/certified_kernels'))
import chunkable_recurrence_rule as frozen
import chunkable_roundoff_guard_v2 as guard

STAGES=['q','k','v','g','beta','S0','gamma','A','B','Qt','M','L','Ident','rhs_a','rhs_v','Cmat','Umat','Z','Zm','O','S_end']


def sha(value):return hashlib.sha256(value).hexdigest()


def capture(inputs):
    arrays={}
    def trace(frame,event,arg):
        if frame.f_code is frozen._chunk_forward.__code__ and event=='return':
            arrays.update({name:frame.f_locals[name].copy() for name in STAGES})
        return trace
    previous=sys.gettrace()
    try:
        sys.settrace(trace)
        output=frozen._chunk_forward(*inputs)
    finally:sys.settrace(previous)
    return output,arrays


def observe():
    q=np.ones((128,1),np.float32)
    inputs=(q,q.copy(),q.copy(),np.full_like(q,1.2),np.full(128,.15,np.float32),np.array([[.25]],np.float32))
    plain=frozen._chunk_forward(*inputs)
    first,arrays=capture(inputs);second,again=capture(inputs)
    sequential=frozen.gated_delta_sequential(*inputs)
    policy=guard.transition_guard(guard.SOURCE,inputs[1],inputs[3],inputs[4])
    gates=dict(complete_stages=set(arrays)==set(STAGES),
        exact_trace_repeat=all(arrays[k].tobytes()==again[k].tobytes() for k in STAGES),
        tracing_preserves_output=all(a.tobytes()==b.tobytes()==c.tobytes() for a,b,c in zip(plain,first,second)),
        unchanged_input_arrays=all(arrays[k].tobytes()==v.tobytes() for k,v in zip(STAGES[:6],inputs)),
        expansive_refusal=not policy['accepted'] and policy['reason']=='expansive_transition')
    arrays.update(sequential_output=sequential[0],sequential_state=sequential[1])
    libraries={}
    for name,module in [('numpy',np),('scipy',scipy)]:
        blas=getattr(module.__config__,'CONFIG',{}).get('Build Dependencies',{}).get('blas',{})
        libraries[name]=dict(version=module.__version__,blas={k:blas[k] for k in ('name','version') if k in blas})
    report=dict(schema=1,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,
        stage_order=STAGES,arrays={k:dict(sha256=sha(v.tobytes()),shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
        environment=dict(python=platform.python_version(),machine=platform.machine(),libraries=libraries),
        sources={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__).read_bytes()) for m in (frozen,guard)},
        observer_sha256=sha(Path(__file__).read_bytes()),policy=policy,
        max_output_error=float(np.max(np.abs(sequential[0]-plain[0]))),
        scope='Actual frozen function locals captured at return, exact against an uninstrumented call and an independent repeat. Stage differences localize environmental arithmetic effects, not a specific CPU instruction or library version cause. Original numerical policy remains refused.')
    return report,arrays


def main():
    report,arrays=observe();out=ROOT/'reports/roundoff_environment_v1';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'arrays.npz',**arrays)
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','gates','environment','max_output_error')}))
    return int(not all(report['gates'].values()))

if __name__=='__main__':raise SystemExit(main())

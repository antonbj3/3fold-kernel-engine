"""Certify one pinned foreign vecAdd kernel; no arbitrary-kernel correctness claim."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent/'foreign_cuda_v1'


def digest(a):
    return hashlib.sha256(a if isinstance(a,bytes) else a.tobytes()).hexdigest()


def certify():
    provenance=json.loads((HERE/'provenance.json').read_text())
    if digest((HERE/'vectorAdd.cu').read_bytes())!=provenance['vendored_sha256']:
        raise ValueError('Foreign source does not match pinned provenance')
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    compiler=subprocess.check_output(['nvcc','--version'],text=True).strip()
    rows=[];timings=[]
    with tempfile.TemporaryDirectory(prefix='foreign-cuda-') as tmp:
        lib=Path(tmp)/'kernel.so'
        flags=['-O3','--ftz=false','--fmad=false','-arch=native','-shared','-Xcompiler','-fPIC']
        subprocess.run(['nvcc',*flags,str(HERE/'harness.cu'),'-o',str(lib)],check=True)
        run=ctypes.CDLL(str(lib)).run_foreign
        fp=ctypes.POINTER(ctypes.c_float)
        run.argtypes=[fp,fp,fp,ctypes.c_int,ctypes.c_int,fp,fp];run.restype=ctypes.c_int
        for n in (0,1,255,256,257,65539,16777219):
            rng=np.random.default_rng(1300+n)
            a=rng.normal(size=n+32).astype(np.float32);b=rng.normal(size=n+32).astype(np.float32)
            if n>=8:
                a[:8]=np.array([0.,-0.,1.,-1.,np.finfo('f').tiny,np.nextafter(np.float32(0),np.float32(1)),2**24,-2**24],dtype='f')
                b[:8]=np.array([-0.,0.,-1.,1.,-np.finfo('f').tiny,0.,1.,-1.],dtype='f')
            for case in ('add','zero','cancel'):
                other=b if case=='add' else np.zeros_like(b) if case=='zero' else -a
                expected=(a[:n]+other[:n]).astype(np.float32)
                outputs=[];leg_times=[]
                for leg in range(2):
                    out=np.empty(n+32,np.float32);ms=ctypes.c_float();cm=ctypes.c_float()
                    rc=run(a.ctypes.data_as(fp),other.ctypes.data_as(fp),out.ctypes.data_as(fp),n,30,ctypes.byref(ms),ctypes.byref(cm))
                    if rc:raise RuntimeError(f'CUDA harness error {rc}')
                    outputs.append(out)
                    leg_times.append(dict(kernel_ms=ms.value,copy_ms=cm.value,
                        useful_gbps=12*n/(ms.value*1e6),copy_gbps=8*(n+32)/(cm.value*1e6),
                        bandwidth_fraction=(12*n/(ms.value*1e6))/(8*(n+32)/(cm.value*1e6))))
                guards=all(np.all(x[n:].view(np.uint32)==0x5a5a5a5a) for x in outputs)
                exact=all(x[:n].tobytes()==expected.tobytes() for x in outputs)
                rows.append(dict(n=n,case=case,input_hashes=[digest(a),digest(other)],output_sha256=digest(outputs[0]),
                    reference_sha256=digest(expected),oracle_exact=exact,repeat_exact=outputs[0].tobytes()==outputs[1].tobytes(),guard_intact=bool(guards)))
                timings.append(dict(n=n,case=case,legs=leg_times))
    gates=dict(G1_reference=all(r['oracle_exact'] for r in rows),G2_repeat=all(r['repeat_exact'] for r in rows),
        G3_null_and_bounds=all(r['guard_intact'] and r['oracle_exact'] for r in rows))
    certificate=dict(schema=1,kernel='NVIDIA cuda-samples vecAdd',provenance=provenance,
        harness_sha256=digest((HERE/'harness.cu').read_bytes()),runner_sha256=digest(Path(__file__).read_bytes()),
        domain='Finite fp32 vectors including signed zero, subnormal and cancellation controls; tested sizes only',
        compile_flags=flags,cases=rows,gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    return certificate,dict(device=device,compiler=compiler,bandwidth_reference='Measured same-worker cudaMemcpy D2D read+write bytes; not theoretical DRAM peak',timings=timings)


if __name__=='__main__':
    certificate,measurement=certify()
    target=Path('reports/foreign_cuda_v1');target.mkdir(parents=True,exist_ok=True)
    for name,value in (('certificate',certificate),('measurement',measurement)):
        (target/(name+'.json')).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(gates=certificate['gates'],device=measurement['device'],large=measurement['timings'][-3:]),allow_nan=False))
    raise SystemExit(0 if all(certificate['gates'].values()) else 1)

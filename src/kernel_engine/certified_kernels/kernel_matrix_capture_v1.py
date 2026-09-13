"""Cloud capture capsule for frozen kernel selftests, int64 sites and exports."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
GEN=ROOT/'src/kernel_engine/kernel_gen'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(GEN))
from fixed_order_selftests_v1 import child, TAPE_SLOTS
from innovation_runtime_adjoint import config_from_reference
from innovation_lbm_stream_baseline import idle


def digest(data):return hashlib.sha256(data).hexdigest()


def sites_worker():
    import warp as wp
    wp.config.deterministic=wp.DeterministicMode.NOT_GUARANTEED
    from kernel_engine.certified_kernels import determinism_sweep_int64 as frozen
    observed=[]
    def measure(kernel,dim,inputs,out):
        values=[];raw=[];input_hashes=[]
        if out.dtype==wp.int64:
            input_hashes=[dict(shape=list(a.shape),dtype=str(a.dtype),sha256=digest(a.numpy().tobytes()))
                          for a in inputs if isinstance(a,wp.array) and a is not out]
        for _ in range(2):
            out.zero_();wp.launch(kernel,dim,inputs=inputs,device=frozen.DEV);wp.synchronize()
            a=out.numpy().copy();values.append(a.astype(np.float64))
            raw.append(dict(shape=list(a.shape),dtype=str(a.dtype),sha256=digest(a.tobytes()),values=a.tolist()))
        if out.dtype==wp.int64:observed.append(dict(kernel=kernel.key,input_hashes=input_hashes,legs=raw,exact=raw[0]==raw[1]))
        return float(np.max(np.abs(values[0]-values[1])))
    frozen._two_runs=measure
    rows=frozen.site_checks()
    if len(rows)!=10 or len(observed)!=10:raise RuntimeError('Incomplete frozen site coverage')
    for r,o in zip(rows,observed):o.update(module=r[0],site=r[1],float_delta=r[2],int64_delta=r[3])
    print('MATRIX_SITES '+json.dumps(observed),flush=True)


def run_export(arch):
    script=GEN/'innovation_lbm_stream_graph_export.py';legs=[]
    with tempfile.TemporaryDirectory(prefix='matrix-export-') as tmp:
        folder=Path(tmp)
        subprocess.run(['bash',str(GEN/'stream_graph_export_v1/build.sh'),str(folder),'sm_'+str(arch)],check=True)
        for _ in range(2):
            idle()
            p=subprocess.run([sys.executable,str(script),'--worker',str(folder)],capture_output=True,text=True,timeout=600)
            if p.returncode:raise RuntimeError(p.stdout[-2000:]+p.stderr[-2000:])
            idle();lines=[x[7:] for x in p.stdout.splitlines() if x.startswith('RESULT ')]
            if len(lines)!=1:raise RuntimeError('Missing frozen export result')
            rows=[]
            for reference in json.loads(lines[0]):
                n=reference['side'];output=folder/f'native_{n}.bin'
                native=subprocess.run([str(folder/'host_stream'),str(n),str(folder/f'input_{n}.bin'),str(output)],capture_output=True,text=True,check=True,timeout=300)
                idle();timing=json.loads(native.stdout);data=output.read_bytes();expected=(folder/f'expected_{n}.bin').read_bytes()
                rows.append(dict(side=n,input_sha256=digest((folder/f'input_{n}.bin').read_bytes()),
                    native_sha256=digest(data),warp_sha256=reference['sha256'],expected_sha256=digest(expected),bytes=len(data),
                    exact_reference=data==expected and digest(data)==reference['sha256'] and reference['exact_cpu_reference'],
                    exact_replays=reference['exact_replays'] and timing['two_launch_identity'],
                    bandwidth_ratio=reference['event_ms']/timing['event_ms']))
            legs.append(rows)
    return legs


def main():
    if sys.argv[1:]==['--sites']:sites_worker();return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --sites')
    target=ROOT/'reports/kernel_matrix_v1';target.mkdir(parents=True,exist_ok=True)
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    meta=subprocess.check_output([sys.executable,'-c',"import warp as w;w.init();print('META',w.config.version,w.get_device('cuda:0').arch)"],text=True)
    line=next(x for x in meta.splitlines() if x.startswith('META '));_,version,arch=line.split()
    if version!='1.17.0':raise RuntimeError('Warp1.17.0 required for this matrix')
    source_hashes={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted((ROOT/'src/kernel_engine').rglob('*.py'))}
    # Unrelated lane-owned Vulkan files are not part of this numerical capsule.
    source_hashes={k:v for k,v in source_hashes.items() if '/vulkan_winding_v1/' not in k}
    for folder in ('stream_graph_export_v1','stream_export_v1'):
        for p in sorted((GEN/folder).glob('*')):
            if p.is_file():source_hashes[str(p.relative_to(ROOT))]=digest(p.read_bytes())
    slots=[m for m,_ in config_from_reference()[0]];suite=[]
    for slot in slots:
        legs=[child(['--worker',slot]) for _ in range(2)]
        gates=dict(selftests=all(x['returncode']==0 and not x['exception_classes'] for x in legs),
            exact_arrays=legs[0]['records']==legs[1]['records'],finite=all(x['finite'] for x in legs),
            coverage=all(x['records'] and (slot not in TAPE_SLOTS or x['gradient_arrays']>0 and x['backward_calls']>0) for x in legs))
        suite.append(dict(slot=slot,legs=legs,gates=gates));print('SLOT '+slot+' '+json.dumps(gates),flush=True)
    idle()
    p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--sites'],capture_output=True,text=True,check=True,timeout=600)
    idle();lines=[x[13:] for x in p.stdout.splitlines() if x.startswith('MATRIX_SITES ')]
    if len(lines)!=1:raise RuntimeError('Missing numeric int64 capture')
    sites=json.loads(lines[0]);exports=run_export(arch)
    gates=dict(eight_slots=len(suite)==8,all_slot_gates=all(all(s['gates'].values()) for s in suite),
        int64_sites=len(sites)==10 and all(s['exact'] and s['int64_delta']==0 for s in sites),
        export_oracle=all(r['exact_reference'] and r['exact_replays'] for leg in exports for r in leg),
        export_repeat=all(a['native_sha256']==b['native_sha256'] and a['warp_sha256']==b['warp_sha256'] for a,b in zip(*exports)))
    report=dict(schema=1,device=device,arch=int(arch),warp=version,source_hashes=source_hashes,selftests=suite,sites=sites,exports=exports,gates=gates,
        scope='Cloud kernel-family observations only; local SKU and other repositories not measured',
        export_performance=dict(within_ten_percent=all(.9<=r['bandwidth_ratio']<=1.1 for leg in exports for r in leg),
                                scope='Diagnostic repeat of frozen timing protocol; not a determinism acceptance gate'))
    data=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    (target/'capture.json.gz').write_bytes(gzip.compress(data,mtime=0))
    (target/'summary.json').write_text(json.dumps(dict(device=device,arch=int(arch),warp=version,gates=gates,
        capture_sha256=digest(data),arrays_per_leg=sum(len(s['legs'][0]['records']) for s in suite),
        bytes_per_leg=sum(s['legs'][0]['observed_bytes'] for s in suite),export_performance=report['export_performance']),indent=2)+'\n')
    print(json.dumps(gates),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())

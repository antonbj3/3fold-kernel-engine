"""Capture one bounded local matrix part; caller must hold the shared GPU lock."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/kernel_engine/certified_kernels'))
import kernel_matrix_capture_v1 as frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['slot','sites','exports'])
    parser.add_argument('--slot');parser.add_argument('--leg',type=int,choices=[0,1])
    args=parser.parse_args()
    reference=json.loads(gzip.decompress((ROOT/'reports/kernel_matrix_v1/l4/capture.json.gz').read_bytes()))
    # Validate every original captured source. New observers are separately hashed.
    sources={k:frozen.digest((ROOT/k).read_bytes()) for k in reference['source_hashes']}
    if sources!=reference['source_hashes']:raise RuntimeError('Frozen source drift')
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    meta=subprocess.check_output([sys.executable,'-c',"import warp as w;w.init();print('META',w.config.version,w.get_device('cuda:0').arch)"],text=True)
    _,version,arch=next(x for x in meta.splitlines() if x.startswith('META ')).split()
    if version!='1.17.0' or 'RTX 5070' not in device:raise RuntimeError('Wrong runtime/device')
    if args.part=='slot':
        if args.slot not in [s['slot'] for s in reference['selftests']] or args.leg is None:raise ValueError('Slot and leg required')
        value=frozen.child(['--worker',args.slot]);name=args.slot.replace('/','_')+'_'+str(args.leg)
    elif args.part=='sites':
        p=subprocess.run([sys.executable,str(Path(frozen.__file__)),'--sites'],capture_output=True,text=True,check=True)
        value=json.loads(next(x[13:] for x in p.stdout.splitlines() if x.startswith('MATRIX_SITES ')));name='sites'
    else:value=frozen.run_export(arch);name='exports'
    data=dict(device=device,warp=version,arch=int(arch),source_hashes=sources,value=value,
              observer_sha256=frozen.digest(Path(__file__).read_bytes()))
    out=ROOT/'reports/kernel_matrix_local_v1/parts';out.mkdir(parents=True,exist_ok=True)
    (out/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(data,sort_keys=True,allow_nan=False).encode(),mtime=0))
    print('PART_COMPLETE '+name,flush=True)

if __name__=='__main__':main()

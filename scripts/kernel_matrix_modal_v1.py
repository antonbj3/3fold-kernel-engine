"""Run the E kernel-family cloud matrix sequentially through a configured Modal runner."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner-root',type=Path,default=os.environ.get('MODAL_RUNNER_ROOT'))
    args=parser.parse_args()
    if args.runner_root is None or not (args.runner_root/'run.py').is_file():
        parser.error('Set MODAL_RUNNER_ROOT or --runner-root to the configured external runner directory')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=ROOT/'reports/kernel_matrix_v1/runs'/stamp;destination.mkdir(parents=True)
    paths=[];transport=[]
    command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/kernel_engine/certified_kernels/kernel_matrix_capture_v1.py'
    for gpu,label in (('L4','l4'),('A10G','a10g'),('H100','h100')):
        print('Capturing '+gpu,flush=True)
        proc=subprocess.run(['modal','run','run.py::run','--repo','3fold-kernel-engine','--cmd',command,'--gpu',gpu,'--timeout-s','2400'],
            cwd=args.runner_root,capture_output=True,text=True,timeout=3000)
        output=proc.stdout+proc.stderr
        # The external runner prints its fetched directory; never infer success
        # solely from the transport exit status, which may not mirror the job.
        found=re.findall(r'^results:\s*(.+?)\s*$',output,re.MULTILINE)
        if len(found)!=1:raise RuntimeError('Missing unique fetched result directory: '+output[-1500:])
        source=Path(found[0])/'reports/kernel_matrix_v1'
        target=destination/label;target.mkdir()
        for name in ('capture.json.gz','summary.json'):
            if not (source/name).is_file():raise RuntimeError('Capture incomplete on '+gpu)
            shutil.copyfile(source/name,target/name)
        paths.append(target);transport.append(dict(gpu=gpu,returncode=proc.returncode))
    reporter=ROOT/'src/kernel_engine/certified_kernels/kernel_matrix_report_v1.py'
    result=subprocess.run([sys.executable,str(reporter),*[str(p) for p in paths]],cwd=ROOT)
    exit_code=result.returncode or int(any(r['returncode']!=0 for r in transport))
    (destination/'run.json').write_text(json.dumps(dict(timestamp=stamp,transport=transport,matrix_returncode=result.returncode,returncode=exit_code),indent=2)+'\n')
    for name in ('matrix.json','matrix.md'):
        shutil.copyfile(ROOT/'reports/kernel_matrix_v1'/name,destination/name)
    return exit_code


if __name__=='__main__':raise SystemExit(main())

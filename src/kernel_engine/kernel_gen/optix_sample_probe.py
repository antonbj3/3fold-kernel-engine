"""Measure an externally built, unmodified headless OptiX sample twice.

The caller holds the shared GPU lock. This proves the toolchain only, not a
shared engine backend, ray-distance agreement, SER or throughput.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]


def ppm(blob):
    position=0;tokens=[]
    while len(tokens)<4:
        while position<len(blob) and blob[position] in b' \t\r\n':position+=1
        if blob[position:position+1]==b'#':
            position=blob.index(b'\n',position)+1;continue
        start=position
        while position<len(blob) and blob[position] not in b' \t\r\n':position+=1
        tokens.append(blob[start:position])
    if tokens!=[b'P6',b'128',b'96',b'255']:raise ValueError('Unexpected PPM contract')
    if blob[position:position+2]==b'\r\n':position+=2
    else:position+=1
    pixels=blob[position:]
    if len(pixels)!=128*96*3:raise ValueError('Unexpected pixel payload length')
    colors={pixels[i:i+3] for i in range(0,len(pixels),3)}
    blue=pixels[2::3]
    return {'distinct_colors':len(colors),'hit_blue_pixels':blue.count(255),'non_hit_blue_pixels':len(blue)-blue.count(255)}


def fault_seen():
    if '--journal-guard' not in sys.argv:return False
    out=subprocess.run(['journalctl','-k','-b','--grep=NVRM: Xid','--no-pager','-o','cat'],capture_output=True,text=True,timeout=10)
    if out.returncode not in (0,1):raise RuntimeError('Journal guard unavailable')
    if 'permission' in out.stderr.lower() or 'not seeing messages' in out.stderr.lower():raise RuntimeError('Journal guard lacks access')
    return bool(re.search(r'NVRM: Xid',out.stdout))


def main():
    binary=Path(os.environ['OPTIX_TRIANGLE_BINARY']).resolve()
    output=ROOT/'reports';output.mkdir(exist_ok=True);rows=[];fault=False
    for leg in range(2):
        if fault_seen():fault=True;break
        target=output/f'optix_sample_{leg+1}.ppm'
        if target.exists():raise FileExistsError('Use a fresh result directory; never accept stale sample output')
        run=subprocess.run([str(binary),'--dim=128x96','--file',str(target)],capture_output=True,timeout=45)
        if os.environ.get('OPTIX_SAMPLE_LOG_DIR'):
            logs=Path(os.environ['OPTIX_SAMPLE_LOG_DIR']);logs.mkdir(parents=True,exist_ok=True)
            (logs/f'run_{leg+1}.stdout').write_bytes(run.stdout)
            (logs/f'run_{leg+1}.stderr').write_bytes(run.stderr)
        row={'exit_code':run.returncode,'stdout_sha256':hashlib.sha256(run.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(run.stderr).hexdigest()}
        if run.returncode==0 and target.exists():
            blob=target.read_bytes();row.update(ppm(blob));row['image_sha256']=hashlib.sha256(blob).hexdigest();row['bytes']=len(blob)
        rows.append(row)
        if run.returncode!=0:break
    fault=fault or fault_seen()
    gates={'both_processes_pass':len(rows)==2 and all(r['exit_code']==0 for r in rows),
        'valid_nontrivial_images':len(rows)==2 and all(r.get('distinct_colors',0)>=16 and r.get('hit_blue_pixels',0)>0 and r.get('non_hit_blue_pixels',0)>0 for r in rows),
        'complete_output_bytes_exact':len(rows)==2 and rows[0].get('image_sha256') is not None and rows[0].get('image_sha256')==rows[1].get('image_sha256'),
        'no_observed_current_boot_fault':not fault}
    report={'legs':rows,'gates':gates,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'journal_guard_requested':'--journal-guard' in sys.argv,'scope':'external unmodified SDK headless triangle toolchain only; no engine correctness or performance claim'}
    (output/'optix_sample_proof.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())

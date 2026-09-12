"""Measure all eight frozen selftests without exempting adjoint-carrying outputs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from kernel_engine.certified_kernels import determinism_sweep_int64 as baseline

NUMBER=re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def execute(module):
    child=subprocess.run([sys.executable,str(ROOT/"src"/"kernel_engine"/(module+".py"))],
                         capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/"src")))
    body="\n".join(line for line in (child.stdout+child.stderr).splitlines() if not baseline.SKIP.search(line))
    return child.returncode,body


def main():
    if baseline.DEV != "cuda:0":
        raise RuntimeError("CUDA required")
    sites=[{"module":m,"site":s,"float_delta":f,"int64_delta":i} for m,s,f,i in baseline.site_checks()]
    rows=[]
    for module,adjoint in baseline.MODULES:
        rc1,a=execute(module)
        rc2,b=execute(module)
        aligned=NUMBER.sub("#",a)==NUMBER.sub("#",b)
        numbers_a=[float(x) for x in NUMBER.findall(a)]
        numbers_b=[float(x) for x in NUMBER.findall(b)]
        delta=max((abs(x-y) for x,y in zip(numbers_a,numbers_b)),default=0.) if aligned else None
        row={"module":module,"adjoint_carrying":adjoint,"returncodes":[rc1,rc2],
             "hashes":[hashlib.sha256(a.encode()).hexdigest(),hashlib.sha256(b.encode()).hexdigest()],
             "identical":a==b,"numeric_outputs_aligned":aligned,"max_printed_numeric_delta":delta}
        rows.append(row)
        print(json.dumps(row),flush=True)
    gates={"int64_sites_identical":all(x["int64_delta"]==0. for x in sites),
           "all_selftests_exit_zero":all(x["returncodes"]==[0,0] for x in rows),
           "all_eight_selftests_identical":len(rows)==8 and all(x["identical"] for x in rows)}
    result={"sites":sites,"selftests":rows,"gates":gates,"performance_measured":False}
    (ROOT/"reports"/"innovation_adjoint_baseline.json").write_text(json.dumps(result,indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())

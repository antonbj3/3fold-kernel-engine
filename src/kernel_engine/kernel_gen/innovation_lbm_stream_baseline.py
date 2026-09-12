"""Measure a frozen bandwidth-oriented LBM stream before native export design."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))


def idle():
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],
                          capture_output=True,text=True,check=True)
    if result.stdout.strip():
        raise RuntimeError("Compute context present outside isolated worker")


def worker():
    import warp as wp
    from kernel_engine.lbm import differentiable_lbm_probe as baseline
    if baseline.DEV!="cuda:0":
        raise RuntimeError("CUDA required")
    rows=[]
    for n in (512,1024):
        host=np.random.default_rng(20260912).random((9,n,n),dtype=np.float32)
        solid=np.zeros((n,n),np.int32);solid[:,0]=1;solid[:,-1]=1
        source=wp.array(host,dtype=float,device=baseline.DEV)
        out=wp.empty_like(source)
        mask=wp.array(solid,dtype=int,device=baseline.DEV)
        cx=wp.array(baseline.CX,dtype=float,device=baseline.DEV)
        cy=wp.array(baseline.CY,dtype=float,device=baseline.DEV)
        opp=wp.array(baseline.OP,dtype=int,device=baseline.DEV)
        def launch():
            wp.launch(baseline.stream,(n,n),inputs=[source,out,mask,cx,cy,opp,n,n],device=baseline.DEV)
        launch();a=out.numpy()
        launch();b=out.numpy()
        i,j=np.indices((n,n))
        expected=np.empty_like(host)
        for q in range(9):
            si=(i-int(baseline.CX[q]))%n
            sj=np.clip(j-int(baseline.CY[q]),0,n-1)
            expected[q]=np.where(solid[si,sj]==1,host[baseline.OP[q],i,j],host[q,si,sj])
        for _ in range(10):
            launch()
        wp.synchronize()
        begin=wp.Event(device=baseline.DEV,enable_timing=True)
        end=wp.Event(device=baseline.DEV,enable_timing=True)
        start=time.perf_counter();wp.record_event(begin)
        for _ in range(30):
            launch()
        wp.record_event(end);wp.synchronize()
        wall=(time.perf_counter()-start)*1000/30
        event=wp.get_event_elapsed_time(begin,end)/30
        payload=2*host.nbytes
        rows.append({"side":n,"payload_bytes":payload,"event_ms":event,"wall_ms":wall,
                     "payload_GB_per_s":payload/event*1e-6,"two_launch_identity":a.tobytes()==b.tobytes(),
                     "exact_cpu_reference":a.tobytes()==expected.tobytes(),
                     "sha256":hashlib.sha256(a.tobytes()).hexdigest()})
    print("RESULT "+json.dumps(rows),flush=True)


def main():
    if sys.argv[1:]==["--worker"]:
        worker();return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker")
    legs=[]
    for _ in range(2):
        idle()
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker"],cwd=ROOT,capture_output=True,text=True)
        if child.returncode:
            print(child.stdout);print(child.stderr,file=sys.stderr)
            raise RuntimeError(f"Worker exit {child.returncode}")
        idle()
        records=[line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if len(records)!=1:
            raise RuntimeError("Expected exactly one worker result")
        rows=json.loads(records[0]);legs.append(rows);print(json.dumps(rows),flush=True)
    gates={"two_worker_identity":all(a["sha256"]==b["sha256"] for a,b in zip(*legs)),
           "exact_cpu_reference":all(r["exact_cpu_reference"] for leg in legs for r in leg),
           "two_launch_identity":all(r["two_launch_identity"] for leg in legs for r in leg)}
    report={"legs":legs,"gates":gates,"warm":10,"timed":30,
            "byte_model":"population read+write only; excludes mask/direction reads and cache effects"}
    (ROOT/"reports"/"innovation_lbm_stream_baseline.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())

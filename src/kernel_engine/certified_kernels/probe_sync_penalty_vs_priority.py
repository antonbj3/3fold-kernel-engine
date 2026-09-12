"""SYNC-LAW x PRIORITY coupling decisive: the sync-density law (penalty ~2.26ms/sync,
measured at DEFAULT priority) vs cell-1's finding (victim priority -3 -> recovery 0.90 on compute).
PRE-REG: H-A priority reduces per-sync penalty >=30% (twin needs a priority predicate; solvers can
defend with priority). H-B penalty unchanged <15% (the quantum is a DRIVER arbitration constant,
below the priority-scheduling layer -> sync-law priority-invariant). Victim: 40 matmul launches with
s=32 syncs, contended by saturating gemm hog; arms: default-prio vs prio -3. 2 passes each."""
import json, os, time
import torch

def stamp():
    return os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip() or "NONE"

def victim_run(prio, dev):
    s = torch.cuda.Stream(priority=prio)
    a = torch.randn(1024, 1024, device=dev); b = torch.randn(1024, 1024, device=dev)
    out = torch.empty_like(a)
    NK, S = 40, 32
    with torch.cuda.stream(s):
        for _ in range(4):  # warm
            torch.matmul(a, b, out=out)
    s.synchronize()
    t0 = time.perf_counter()
    k_per_sync = max(1, NK // S)
    with torch.cuda.stream(s):
        for i in range(S):
            for _ in range(k_per_sync):
                torch.matmul(a, b, out=out)
            s.synchronize()  # the host sync = the law's lever
    return time.perf_counter() - t0

dev = torch.device("cuda")
lo, hi = torch.cuda.stream_priority_range() if hasattr(torch.cuda, "stream_priority_range") else (-3, 0)
rep = {"probe": "probe_sync_penalty_vs_priority", "pre": stamp(), "prio_range": [lo, hi]}
# clean references
rep["clean_default_s"] = [victim_run(0, dev) for _ in range(2)]
# hog: saturating gemms on default-prio stream, kept alive during arms
hs = torch.cuda.Stream(priority=0)
H = torch.randn(4096, 4096, device=dev); Ho = torch.empty_like(H)
def hog_burst(n=200):
    with torch.cuda.stream(hs):
        for _ in range(n):
            torch.matmul(H, H, out=Ho)
res = {}
for label, prio in [("contended_prio0", 0), ("contended_prioHigh", lo)]:
    runs = []
    for _ in range(2):
        hog_burst(400)
        assert not hs.query(), "hog must be running"
        runs.append(victim_run(prio, dev))
        torch.cuda.synchronize()
    res[label] = runs
rep["arms"] = res
rep["post"] = stamp()
import statistics as st
c = st.median(rep["clean_default_s"]); a0 = st.median(res["contended_prio0"]); ah = st.median(res["contended_prioHigh"])
pen0 = (a0 - c) / 32 * 1000; penh = (ah - c) / 32 * 1000  # ms/sync
red = (pen0 - penh) / pen0 if pen0 > 0 else 0.0
rep["derived"] = {"clean_s": c, "penalty_prio0_ms_per_sync": round(pen0, 3),
                  "penalty_prioHigh_ms_per_sync": round(penh, 3), "reduction_frac": round(red, 3),
                  "H_A_priority_reduces_30pct": bool(red >= 0.30), "H_B_invariant_15pct": bool(abs(red) < 0.15)}
json.dump(rep, open("reports/probes/probe_sync_penalty_vs_priority.json", "w"), indent=1)
print(json.dumps(rep["derived"], indent=1))

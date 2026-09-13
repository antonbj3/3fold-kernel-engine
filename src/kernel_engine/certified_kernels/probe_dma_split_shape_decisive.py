"""dma_split_shape_disputed DECISIVE: the windowed instrument (64MiB chunks) saw aggregate 29.5
EVEN split (14.9/14.9); the fair-vs-waterfill instrument (looped copies) saw 36.4 UNEVEN (24.6/11.8).
Same invocation, both instruments, same pinned buffers. PRE-REG: if instrument determines the answer,
the difference is INSTRUMENT (chunk scheduling / measurement window), and the twin carries the config
predicate. Gates: G1 both reproduce their own prior numbers within 15%; G2 identify the discriminating
config axis (chunk size / stream creation order / timing method)."""
import json, time, os
import torch

def _out(name):
    """Path of the evidence JSON under the repository's artifacts/ directory."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def stamp():
    return os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip() or "NONE"

def wave1_style(chunk_mib=64, window_s=4.0):
    n = chunk_mib * 1024 * 1024 // 4
    h1 = torch.empty(n, dtype=torch.float32, pin_memory=True); d1 = torch.empty(n, dtype=torch.float32, device="cuda")
    h2 = torch.empty(n, dtype=torch.float32, pin_memory=True); d2 = torch.empty(n, dtype=torch.float32, device="cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    c1 = c2 = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    # interleaved enqueue, count completed chunks per stream via events polled at end
    ev1, ev2 = [], []
    while time.perf_counter() - t0 < window_s:
        with torch.cuda.stream(s1):
            d1.copy_(h1, non_blocking=True); e = torch.cuda.Event(); e.record(s1); ev1.append(e)
        with torch.cuda.stream(s2):
            d2.copy_(h2, non_blocking=True); e = torch.cuda.Event(); e.record(s2); ev2.append(e)
        if len(ev1) % 8 == 0:
            time.sleep(0.005)  # let queue drain a bit (paced by the measurement window)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    g1 = len(ev1) * chunk_mib / 1024 / dt; g2 = len(ev2) * chunk_mib / 1024 / dt
    return {"s1_GBps": round(g1, 2), "s2_GBps": round(g2, 2), "agg": round(g1 + g2, 2), "mode": "interleaved-enqueue-windowed"}

def cell3_style(chunk_mib=64, iters=40):
    n = chunk_mib * 1024 * 1024 // 4
    h1 = torch.empty(n, dtype=torch.float32, pin_memory=True); d1 = torch.empty(n, dtype=torch.float32, device="cuda")
    h2 = torch.empty(n, dtype=torch.float32, pin_memory=True); d2 = torch.empty(n, dtype=torch.float32, device="cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    # batch-enqueue: ALL of s1's loop first, then all of s2's (cell3-style loop per stream)
    with torch.cuda.stream(s1):
        for _ in range(iters): d1.copy_(h1, non_blocking=True)
    with torch.cuda.stream(s2):
        for _ in range(iters): d2.copy_(h2, non_blocking=True)
    e1 = torch.cuda.Event(); e1.record(s1); e2 = torch.cuda.Event(); e2.record(s2)
    e1.synchronize(); t1 = time.perf_counter() - t0
    e2.synchronize(); t2 = time.perf_counter() - t0
    g1 = iters * chunk_mib / 1024 / t1; g2 = iters * chunk_mib / 1024 / t2
    return {"s1_GBps": round(g1, 2), "s2_GBps": round(g2, 2), "agg_wall": round(2 * iters * chunk_mib / 1024 / max(t1, t2), 2), "mode": "batch-enqueue-per-stream"}

rep = {"probe": "probe_dma_split_shape_decisive", "pre": stamp()}
rep["wave1_style"] = [wave1_style() for _ in range(2)]
rep["cell3_style"] = [cell3_style() for _ in range(2)]
rep["post"] = stamp()
json.dump(rep, open(_out("probe_dma_split_shape_decisive.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))

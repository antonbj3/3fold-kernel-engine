"""DUAL-CEILING DECISIVE (PRE-REGISTERED before running):
The tension: single H2D = 24.7 GB/s but two concurrent H2D = 29.5 aggregate -> 24.7 is NOT an
engine cap. Q: what limits a single stream?
PRE-REG (frozen): H1 per-transfer-overhead -> BW rises monotonically with chunk size, 256MiB chunk
  reaches >= 27.5 GB/s (>=93% of the 29.5 shared cap). H2 stream-depth cap -> BW flat-ish (<5%
  gain 64->256MiB), stays <= 26 GB/s. Disambiguator arm: single stream, 64MiB chunks but 4 transfers
  queued back-to-back before sync (deeper pipeline, same chunk) -> H1 predicts gain, H2 predicts none.
GATES: G1 = chunk-sweep monotone direction decided; G2 = queued-depth arm agrees with the same H.
Tenancy: nvidia-smi compute-app stamps."""
import json, os, time
import torch

def _out(name):
    """Path of the evidence JSON under the repository's artifacts/ directory."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def stamp():
    return os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip() or "NONE"

def bw_gbps(chunk_mib, n_queued=1, total_mib=1024, reps=3):
    n = chunk_mib * 1024 * 1024 // 4
    h = torch.empty(n, dtype=torch.float32, pin_memory=True)
    d = torch.empty(n, dtype=torch.float32, device="cuda")
    s = torch.cuda.Stream()
    iters = max(1, total_mib // (chunk_mib * n_queued))
    outs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(iters):
                for _ in range(n_queued):
                    d.copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        outs.append(iters * n_queued * chunk_mib / 1024 / dt)
    return sorted(outs)[len(outs)//2]

rep = {"probe": "probe_dma_single_stream_gap", "pre": stamp(), "sweep": {}, "queued": {}}
for c in [16, 64, 256]:
    rep["sweep"][c] = round(bw_gbps(c), 2)
    print(f"chunk {c}MiB: {rep['sweep'][c]} GB/s")
for q in [1, 4]:
    rep["queued"][q] = round(bw_gbps(64, n_queued=q), 2)
    print(f"64MiB x{q} queued: {rep['queued'][q]} GB/s")
rep["post"] = stamp()
b256, b64, b16 = rep["sweep"][256], rep["sweep"][64], rep["sweep"][16]
h1 = b256 >= 27.5 and b256 > b64 > b16
h2 = (b256 - b64) / b64 < 0.05 and b256 <= 26.0
qgain = (rep["queued"][4] - rep["queued"][1]) / rep["queued"][1]
rep["verdict"] = {"H1_per_transfer_overhead": h1, "H2_stream_depth_cap": h2,
                  "queued_gain_frac": round(qgain, 3),
                  "G2_agrees": bool((h1 and qgain > 0.03) or (h2 and qgain <= 0.03))}
json.dump(rep, open(_out("probe_dma_single_stream_gap.json"), "w"), indent=1)
print("VERDICT:", rep["verdict"])

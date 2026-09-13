"""Measure direct versus captured dispatch of the unchanged LBM stream."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from innovation_lbm_stream_baseline import ROOT, idle


def worker():
    import warp as wp
    from kernel_engine.lbm import differentiable_lbm_probe as baseline
    if baseline.DEV != "cuda:0":
        raise RuntimeError("CUDA required")
    rows = []
    for n in (512, 1024):
        host = np.random.default_rng(20260912).random((9, n, n), dtype=np.float32)
        solid = np.zeros((n, n), np.int32)
        solid[:, 0] = 1
        solid[:, -1] = 1
        source = wp.array(host, dtype=float, device=baseline.DEV)
        out = wp.empty_like(source)
        mask = wp.array(solid, dtype=int, device=baseline.DEV)
        cx = wp.array(baseline.CX, dtype=float, device=baseline.DEV)
        cy = wp.array(baseline.CY, dtype=float, device=baseline.DEV)
        opp = wp.array(baseline.OP, dtype=int, device=baseline.DEV)

        def launch():
            wp.launch(baseline.stream, (n, n), inputs=[source, out, mask, cx, cy, opp, n, n], device=baseline.DEV)

        launch()
        initial = out.numpy()
        i, j = np.indices((n, n))
        expected = np.empty_like(host)
        for q in range(9):
            si = (i - int(baseline.CX[q])) % n
            sj = np.clip(j - int(baseline.CY[q]), 0, n - 1)
            expected[q] = np.where(solid[si, sj] == 1, host[baseline.OP[q], i, j], host[q, si, sj])
        start = time.perf_counter()
        with wp.ScopedCapture(device=baseline.DEV) as capture:
            for _ in range(30):
                launch()
        capture_ms = (time.perf_counter() - start) * 1000
        measurements = {}
        for mode in ("direct", "graph"):
            for _ in range(10):
                launch()
            wp.synchronize()
            begin = wp.Event(device=baseline.DEV, enable_timing=True)
            end = wp.Event(device=baseline.DEV, enable_timing=True)
            start = time.perf_counter()
            wp.record_event(begin)
            if mode == "direct":
                for _ in range(30):
                    launch()
            else:
                wp.capture_launch(capture.graph)
            wp.record_event(end)
            wp.synchronize()
            wall = (time.perf_counter() - start) * 1000 / 30
            event = wp.get_event_elapsed_time(begin, end) / 30
            result = out.numpy()
            measurements[mode] = dict(event_ms=event, wall_ms=wall,
                                      sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                                      exact_cpu=result.tobytes() == expected.tobytes(),
                                      exact_initial=result.tobytes() == initial.tobytes())
        rows.append(dict(side=n, payload_bytes=2 * host.nbytes, modes=measurements, capture_ms=capture_ms))
    print("RESULT " + json.dumps(rows), flush=True)


def main():
    if sys.argv[1:] == ["--worker"]:
        worker()
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker")
    legs = []
    for _ in range(2):
        idle()
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                           cwd=ROOT, capture_output=True, text=True)
        if p.returncode:
            print(p.stdout)
            print(p.stderr, file=sys.stderr)
            raise RuntimeError("Worker failed")
        idle()
        lines = [line[7:] for line in p.stdout.splitlines() if line.startswith("RESULT ")]
        if len(lines) != 1:
            raise RuntimeError("Expected one result")
        legs.append(json.loads(lines[0]))
        print(lines[0], flush=True)
    gates = dict(all_outputs_exact=all(m["exact_cpu"] and m["exact_initial"] for leg in legs for r in leg for m in r["modes"].values()),
                 exact_repeats=all(a["modes"][mode]["sha256"] == b["modes"][mode]["sha256"] for a, b in zip(*legs) for mode in ("direct", "graph")),
                 finite_positive_times=all(np.isfinite(m[k]) and m[k] > 0 for leg in legs for r in leg for m in r["modes"].values() for k in ("event_ms", "wall_ms")))
    (ROOT / "reports/innovation_lbm_dispatch_probe.json").write_text(json.dumps(dict(legs=legs, gates=gates), indent=2) + "\n")
    print(json.dumps(gates), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

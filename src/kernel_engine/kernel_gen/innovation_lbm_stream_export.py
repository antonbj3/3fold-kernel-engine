"""Compare an isolated frozen Warp launch with a standalone compiled C caller."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from innovation_lbm_stream_baseline import ROOT, idle, worker as baseline_worker

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "stream_export_v1"


def forward_text(source):
    marker = 'extern "C" __global__ void stream_3bfa5cbb_cuda_kernel_forward('
    body = source[source.index(marker):]
    if 'extern "C" __global__ void' in body[len(marker):]:
        body = body[:body.index('extern "C" __global__ void', len(marker))]
    return "\n".join(line for line in body.splitlines()
                     if line.strip() and not line.lstrip().startswith("//"))


def worker(folder):
    import warp as wp
    from kernel_engine.lbm import differentiable_lbm_probe as reference
    for n in (512, 1024):
        host = np.random.default_rng(20260912).random((9, n, n), dtype=np.float32)
        solid = np.zeros((n, n), np.int32)
        solid[:, 0] = 1; solid[:, -1] = 1
        with (folder / f"input_{n}.bin").open("wb") as stream:
            for array in (host, solid, reference.CX, reference.CY, reference.OP):
                stream.write(array.tobytes())
        i, j = np.indices((n, n))
        expected = np.empty_like(host)
        for q in range(9):
            si = (i - int(reference.CX[q])) % n
            sj = np.clip(j - int(reference.CY[q]), 0, n - 1)
            expected[q] = np.where(solid[si, sj] == 1, host[reference.OP[q], i, j], host[q, si, sj])
        (folder / f"expected_{n}.bin").write_bytes(expected.tobytes())
    baseline_worker()
    # Ensure the checked-in export is the current frozen kernel's generated body.
    paths = list(Path(wp.config.kernel_cache_dir).rglob("*differentiable_lbm_probe*.cu"))
    expected = forward_text((EXPORT / "stream_generated.cu").read_text())
    if not any(forward_text(p.read_text()) == expected for p in paths):
        raise RuntimeError("Generated stream body differs from checked-in export")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2])); return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker DIRECTORY")
    legs = []
    with tempfile.TemporaryDirectory(prefix="stream-export-") as tmp:
        folder = Path(tmp)
        subprocess.run(["bash", str(EXPORT / "build.sh"), str(folder), "sm_89"], check=True)
        for _ in range(2):
            idle()
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(folder)],
                                   cwd=ROOT, capture_output=True, text=True)
            if child.returncode:
                print(child.stdout); print(child.stderr, file=sys.stderr)
                raise RuntimeError(f"Warp worker exit {child.returncode}")
            idle()
            lines = [line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
            if len(lines) != 1:
                raise RuntimeError("Expected one frozen worker result")
            rows = []
            for reference in json.loads(lines[0]):
                n = reference["side"]
                idle()
                output = folder / f"native_{n}.bin"
                native = subprocess.run([str(folder / "host_stream"), str(n),
                                         str(folder / f"input_{n}.bin"), str(output)],
                                        capture_output=True, text=True, check=True)
                idle()
                measured = json.loads(native.stdout)
                data = output.read_bytes()
                if len(data) != reference["payload_bytes"] // 2:
                    raise RuntimeError("Native output length mismatch")
                measured["sha256"] = hashlib.sha256(data).hexdigest()
                measured["payload_GB_per_s"] = reference["payload_bytes"] / measured["event_ms"] * 1e-6
                rows.append({"side": n, "warp": reference, "native": measured,
                             "exact_warp_output": measured["sha256"] == reference["sha256"]
                                 and data == (folder / f"expected_{n}.bin").read_bytes(),
                             "bandwidth_ratio": reference["event_ms"] / measured["event_ms"]})
            legs.append(rows)
            print(json.dumps(rows), flush=True)
    gates = {"exact_warp_outputs": all(r["exact_warp_output"] for leg in legs for r in leg),
             "frozen_cpu_reference": all(r["warp"]["exact_cpu_reference"] for leg in legs for r in leg),
             "two_native_runs_identical": all(a["native"]["sha256"] == b["native"]["sha256"] for a,b in zip(*legs)),
             "two_warp_runs_identical": all(a["warp"]["sha256"] == b["warp"]["sha256"] for a,b in zip(*legs)),
             "bandwidth_within_ten_percent": all(0.9 <= r["bandwidth_ratio"] <= 1.1 for leg in legs for r in leg)}
    report = {"legs": legs, "gates": gates, "warm": 10, "timed": 30,
              "host_language": "C11 compiled with gcc; CUDA shim compiled with nvcc",
              "arch": "sm_89", "byte_model": "population read+write floor; no peak DRAM claim"}
    (ROOT / "reports/innovation_lbm_stream_export.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bound the frozen tomography loop's deterministic records, without changing it."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

from innovation_runtime_adjoint import ROOT, SRC, config_from_reference

REFERENCE = SRC / "kernel_engine/wave_fdtd/xray_tomography_sigma.py"


def record_bound():
    nodes = ast.parse(REFERENCE.read_text()).body
    samples = next(ast.literal_eval(n.value) for n in nodes
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "NS" for t in n.targets))
    if samples != 110:
        raise RuntimeError("Frozen loop changed; re-derive the record bound")
    # Four field reads in each bilinear sample yield at most four field-adjoint
    # records. The project loop has exactly NS samples per ray/thread.
    return 4 * samples


def main():
    bound = record_bound()
    if sys.argv[1:] == ["--worker"]:
        import warp as wp
        wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
        wp.config.deterministic_max_records = bound
        runpy.run_path(str(REFERENCE), run_name="__main__")
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker")
    _, skip = config_from_reference()
    legs = []
    for _ in range(2):
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               cwd=ROOT, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONPATH=str(SRC)))
        normalized = "\n".join(line for line in (child.stdout + child.stderr).splitlines()
                               if not skip.search(line))
        metrics = {}
        for key, pattern in {
            "printed_max_fd_relative_error": r"max rel ([\deE.+-]+)",
            "printed_covered_error_percent": r"COVERED region = ([\d.]+)%",
            "printed_quartile_error_ratio": r"error band = ([\d.]+)",
        }.items():
            match = re.search(pattern, child.stdout)
            metrics[key] = float(match[1]) if match else None
        leg = {"returncode": child.returncode,
               "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
               "exception_classes": re.findall(r"^(\w+(?:Error|Exception)):", child.stderr, re.MULTILINE),
               "printed_metrics": metrics}
        legs.append(leg)
        print(json.dumps(leg), flush=True)
    gates = {"two_normalized_outputs_identical": legs[0]["sha256"] == legs[1]["sha256"],
             "both_original_selftests_pass": all(x["returncode"] == 0 for x in legs),
             "no_runtime_exceptions": all(not x["exception_classes"] for x in legs),
             "all_metrics_present": all(v is not None for x in legs for v in x["printed_metrics"].values())}
    report = {"runtime_mode": "RUN_TO_RUN", "max_records": bound,
              "derivation": "4 bilinear field reads per sample * 110 samples per ray/thread",
              "legs": legs, "gates": gates, "performance_measured": False}
    (ROOT / "reports/innovation_tomography_record_bound.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

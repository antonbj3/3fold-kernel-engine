"""Compose the measured runtime mode and bounded tomography across eight cases."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from innovation_runtime_adjoint import ROOT, SRC, config_from_reference


def main():
    if sys.argv[1:]:
        raise ValueError("Expected no arguments")
    modules, skip = config_from_reference()
    rows = []
    for module, adjoint in modules:
        if module == "wave_fdtd/xray_tomography_sigma":
            script = "innovation_tomography_record_bound.py"
            args = ["--worker"]
        else:
            script = "innovation_runtime_adjoint.py"
            args = ["--worker", module]
        legs = []
        for _ in range(2):
            child = subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *args],
                                   cwd=ROOT, capture_output=True, text=True,
                                   env=dict(os.environ, PYTHONPATH=str(SRC)))
            text = "\n".join(line for line in (child.stdout + child.stderr).splitlines()
                             if not skip.search(line))
            legs.append({"returncode": child.returncode,
                         "sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "exception_classes": re.findall(r"^(\w+(?:Error|Exception)):", child.stderr, re.MULTILINE)})
        row = {"module": module, "adjoint_carrying": adjoint, "legs": legs,
               "identical": legs[0]["sha256"] == legs[1]["sha256"],
               "both_exit_zero": all(x["returncode"] == 0 for x in legs)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    gates = {"all_eight_identical": len(rows) == 8 and all(x["identical"] for x in rows),
             "no_runtime_exceptions": all(not q["exception_classes"] for r in rows for q in r["legs"]),
             "all_eight_physics_pass": len(rows) == 8 and all(x["both_exit_zero"] for x in rows)}
    result = {"runtime_mode": "RUN_TO_RUN", "tomography_max_records": 440,
              "selftests": rows, "gates": gates, "performance_measured": False,
              "scope": "normalized whole-selftest text, not every hidden array"}
    (ROOT / "reports/innovation_runtime_bounded_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

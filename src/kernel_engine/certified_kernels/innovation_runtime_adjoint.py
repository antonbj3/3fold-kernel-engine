"""Test runtime scatter/sort/reduce mode on the same eight frozen selftests."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))
REFERENCE=Path(__file__).with_name("determinism_sweep_int64.py")


def config_from_reference():
    # Read the exact frozen module list and normalizer without importing Warp or
    # creating baseline modules before the worker selects its runtime mode.
    nodes=ast.parse(REFERENCE.read_text()).body
    modules=None;pattern=None
    for node in nodes:
        if isinstance(node,ast.Assign):
            names=[x.id for x in node.targets if isinstance(x,ast.Name)]
            if "MODULES" in names:
                modules=ast.literal_eval(node.value)
            if "SKIP" in names:
                pattern=ast.literal_eval(node.value.args[0])
    if modules is None or pattern is None:
        raise RuntimeError("Frozen reference metadata not found")
    return modules,re.compile(pattern)


def main():
    modules,skip=config_from_reference()
    if len(sys.argv)==3 and sys.argv[1]=="--worker":
        module=sys.argv[2]
        if module not in dict(modules):
            raise ValueError("Unknown reference module")
        import warp as wp
        wp.config.deterministic=wp.DeterministicMode.RUN_TO_RUN
        # Keep the runtime-generated record bounds; overflow/unsupported kernels
        # are measured failures. No silent fallback or PID/exception whitelist.
        runpy.run_path(str(SRC/"kernel_engine"/(module+".py")),run_name="__main__")
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker MODULE")
    rows=[]
    for module,adjoint in modules:
        legs=[]
        for _ in range(2):
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",module],
                                 cwd=ROOT,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(SRC)))
            text="\n".join(line for line in (child.stdout+child.stderr).splitlines() if not skip.search(line))
            classes=re.findall(r"^(\w+(?:Error|Exception)):",child.stderr,re.MULTILINE)
            legs.append({"returncode":child.returncode,"sha256":hashlib.sha256(text.encode()).hexdigest(),
                         "exception_classes":classes})
        row={"module":module,"adjoint_carrying":adjoint,"legs":legs,
             "identical":legs[0]["sha256"]==legs[1]["sha256"],
             "both_exit_zero":all(x["returncode"]==0 for x in legs)}
        rows.append(row);print(json.dumps(row),flush=True)
    gates={"all_eight_identical":len(rows)==8 and all(x["identical"] for x in rows),
           "all_eight_exit_zero":len(rows)==8 and all(x["both_exit_zero"] for x in rows),
           "no_runtime_exceptions":all(not q["exception_classes"] for r in rows for q in r["legs"])}
    result={"runtime_mode":"RUN_TO_RUN","max_records":"runtime default",
            "selftests":rows,"gates":gates,"performance_measured":False}
    (ROOT/"reports"/"innovation_runtime_adjoint.json").write_text(json.dumps(result,indent=2)+"\n")
    return 0 if all(gates.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())

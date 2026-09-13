"""Run the unchanged recurrence guard and publish its fresh JSON receipt."""
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    module=root/'src/kernel_engine/certified_kernels/chunkable_roundoff_guard_v2.py'
    artifact=module.parent/'artifacts/chunkable_roundoff_guard_v2.json'
    output=root/'reports/chunkable_roundoff_guard_v2.json'
    output.parent.mkdir(exist_ok=True)
    output.unlink(missing_ok=True)
    artifact.unlink(missing_ok=True)
    result=subprocess.run([sys.executable,str(module)],cwd=root,check=False)
    if artifact.is_file():shutil.copyfile(artifact,output)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

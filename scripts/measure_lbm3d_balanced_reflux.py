"""Balanced conserved reflux with expanded fine-wall allocation, gated DNS."""
import subprocess,sys
from measure_lbm3d_refinement_sgs import main as numerical_gate


if __name__=='__main__':
    status=numerical_gate(recursive=True,bulk_tau_fine=1.,conserved_reflux=True,balanced_reflux=True,
                          output='reports/lbm3d_balanced_reflux_gates_v1/report.json')
    if status:raise SystemExit(status)
    cmd=[sys.executable,'scripts/measure_lbm3d_refined_dns.py',
         '--uniform-report','data/lbm3d_resolution_control.json',
         '--sensitivity-report','data/lbm3d_resolution_sensitivity.json',
         '--wall-cells','24','--conserved-reflux','--balanced-reflux']
    status=subprocess.run(cmd+['--diagnostic'],check=False).returncode
    if status:raise SystemExit(status)
    raise SystemExit(subprocess.run(cmd,check=False).returncode)

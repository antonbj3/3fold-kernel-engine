"""Bracket the accepted wall allocation with an 18-layer candidate.

Admission requires completed pressure/shear screens, exact ledgers, clear
flags and <=5% wave error. Improvement over the unbalanced control remains
reported separately; it is not assumed for shear transport.
"""
import json,subprocess,sys
from pathlib import Path

if __name__=='__main__':
    for mode,waves in [('acoustics',(96,32,16)),('shear',(192,32,16))]:
        for wave in waves:
            d=json.loads(Path(f'data/lbm3d_{mode}_wall18_wave{wave}.json').read_text())
            assert d['wall_cells']==18 and all(d['gates'][k] for k in ('complete','exact_ledgers_and_flags','balanced_error_below_5percent'))
    cmd=[sys.executable,'scripts/measure_lbm3d_refined_dns.py','--uniform-report','data/lbm3d_resolution_control.json','--sensitivity-report','data/lbm3d_resolution_sensitivity.json','--wall-cells','18','--conserved-reflux','--balanced-reflux','--specialized']
    status=subprocess.run(cmd+['--diagnostic'],check=False).returncode
    if status:raise SystemExit(status)
    raise SystemExit(subprocess.run(cmd,check=False).returncode)

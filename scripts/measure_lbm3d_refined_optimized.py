"""Gate specialization before a full replay of the accepted refined DNS."""
import json,subprocess,sys
from pathlib import Path
from measure_lbm3d_refined_specialized import main as gate

if __name__=='__main__':
    status=gate()
    if status:raise SystemExit(status)
    cmd=[sys.executable,'scripts/measure_lbm3d_refined_dns.py','--uniform-report','data/lbm3d_resolution_control.json','--sensitivity-report','data/lbm3d_resolution_sensitivity.json','--wall-cells','24','--conserved-reflux','--balanced-reflux','--specialized']
    status=subprocess.run(cmd,check=False).returncode
    if status:raise SystemExit(status)
    old=json.loads(Path('data/lbm3d_refined_control.json').read_text())
    new=json.loads(Path('reports/lbm3d_refined_dns_v1/report.json').read_text())
    keys=('step','mass_exact','momentum_residual','flags','wall_int64_limbs')
    exact=len(old['blocks'])==len(new['blocks'])==12 and all(all(a[k]==b[k] for k in keys) for a,b in zip(old['blocks'],new['blocks']))
    statistic_pairs=[(a,b) for a,b in zip(old['blocks'],new['blocks']) if a['step']>old['burn_steps']]
    exact=exact and len(statistic_pairs)==8 and all(a['plane_integer_statistics']==b['plane_integer_statistics'] for a,b in statistic_pairs)
    result={'scope':'full DNS integer statistics and ledgers replay; full population parity tested separately in short gate','gates':{'both_DNS_pass':bool(old['passed'] and new['passed']),'all12block_ledgers_and8averaging_block_statistics_exact':exact},'original_seconds':old['seconds'],'specialized_seconds':new['seconds'],'separate_run_time_ratio':old['seconds']/new['seconds']}
    result['passed']=all(result['gates'].values())
    p=Path('reports/lbm3d_refined_optimized_replay_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['passed'] else 1)

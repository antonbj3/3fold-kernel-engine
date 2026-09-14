"""Reuse a measured channel response for different observable goals."""
import hashlib,json
from pathlib import Path
import numpy as np
from kernel_engine.lbm.lbm3d_goal_allocation import allocate_for_goal,binary_band_choice


def main():
    source=Path('reports/lbm3d_channel_sensitivity_v1/receipt/reports/lbm3d_channel_sensitivity_v1/report.json')
    r=json.loads(source.read_text());j=np.asarray(r['weighted_jacobian']);n=j.shape[0]//5
    masks={'mean_U_plus_only':np.r_[np.ones(n),np.zeros(4*n)],
           'Reynolds_stresses_only':np.r_[np.zeros(n),np.ones(4*n)],
           'mean_and_stresses':np.ones(5*n)}
    goals={}
    for name,weights in masks.items():
        result=allocate_for_goal(j,tolerances=np.ones(j.shape[0]),weights=weights)
        result['two_fine_band_candidate']=binary_band_choice(result['sensitivities'],fine_bands=2)
        goals[name]=result
    report={'scope':'goal reweighting at one failed uniform DNS endpoint; no new PDE solve or certified mesh',
        'input_jacobian_already_tolerance_scaled':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'physical_state':r['base_state_sha256'],'goals':goals,
        'limitation':'Only goals formed from the measured observables at this state/horizon are covered. New geometry or operating point requires additional sensitivity evidence.'}
    p=Path('reports/lbm3d_goal_reweighting_v1/report.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['two_fine_band_candidate'] for k,v in goals.items()},indent=2))

if __name__=='__main__':main()

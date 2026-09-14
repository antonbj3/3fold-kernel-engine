"""Postprocess accepted profiles on the same physical wall-distance grid.

Historical native-plane receipts are preserved. These are interpolated profile
metrics, not new simulations or proof of spatial convergence.
"""
import hashlib,json
from pathlib import Path
import numpy as np


def main(refined_report='reports/lbm3d_refined_dns_v1/20260914T165405Z/report.json',output='reports/lbm3d_common_grid_v1/report.json'):
    paths={
      'uniform':Path('reports/lbm3d_recursive_resolution_v1/20260914T103313Z/report.json'),
      'refined':Path(refined_report)}
    data={k:json.loads(p.read_text()) for k,p in paths.items()}
    data['uniform']['u_tau']=data['uniform']['u_tau_measured']
    assert data['uniform']['grid']==[288,98,144]
    assert data['refined']['fine_equivalent_extent']==[288,96,144]
    assert data['uniform']['tau']==data['refined']['fine_tau']==.5032
    nu=(data['uniform']['tau']-.5)/3;y=np.arange(48)+.5
    means=np.loadtxt('tests/data/lbm_channel/chan180.means')
    stress=np.loadtxt('tests/data/lbm_channel/chan180.reystress')
    rows={};profiles={}
    for name,d in data.items():
        assert d['passed']
        ut=d['u_tau'];source_y=np.array(d['y_plus'])*nu/ut;yp=y*ut/nu
        up=np.interp(y,source_y,d['U_plus'])
        rp=np.array([np.interp(y,source_y,r) for r in d['reynolds_plus'][:4]])
        ru=np.interp(yp,means[:,1],means[:,2]);rr=np.array([np.interp(yp,stress[:,1],stress[:,i]) for i in range(2,6)])
        peaks=np.max(np.abs(rr),axis=1);log=(yp>=30)&(yp<=100)
        metrics={'log_mean_relative_max':float(np.max(np.abs(up[log]/ru[log]-1))),
                 'mean_relative_L2':float(np.linalg.norm(up-ru)/np.linalg.norm(ru)),
                 'stress_peak_RMS':(np.sqrt(np.mean((rp-rr)**2,axis=1))/peaks).tolist()}
        metrics['accuracy_pass']=metrics['log_mean_relative_max']<=.05 and metrics['mean_relative_L2']<=.1 and max(metrics['stress_peak_RMS'])<=.2
        rows[name]=metrics;profiles[name]=(up*ut,rp*ut**2)
    assert np.isclose(rows['uniform']['log_mean_relative_max'],data['uniform']['log_mean_relative_max_error'],rtol=1e-12)
    assert np.allclose(rows['uniform']['stress_peak_RMS'],data['uniform']['stress_peak_normalized_RMS'],rtol=1e-12)
    u,r=profiles['uniform'];v,s=profiles['refined']
    peaks=np.max(np.abs(stress[:,2:6]),axis=0)*data['uniform']['u_tau']**2
    parity={'dimensional_mean_relative_L2':float(np.linalg.norm(u-v)/np.linalg.norm(u)),
            'dimensional_stress_peak_RMS':(np.sqrt(np.mean((r-s)**2,axis=1))/peaks).tolist()}
    parity['pass']=parity['dimensional_mean_relative_L2']<=.05 and max(parity['dimensional_stress_peak_RMS'])<=.1
    result={'scope':__doc__,'wall_distance_fine_cells':y.tolist(),'metrics':rows,'refined_vs_uniform':parity,
            'passed':all(row['accuracy_pass'] for row in rows.values()) and parity['pass'],
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'metrics':rows,'parity':parity,'passed':result['passed']}))

if __name__=='__main__':main()

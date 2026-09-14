import itertools
import numpy as np
from kernel_engine.lbm.lbm3d_goal_allocation import allocate_for_goal,binary_band_choice


def test_goal_reweighting_moves_resources_and_zero_goal_drops_all():
    j=np.diag([4.,3.,2.,1.])
    a=allocate_for_goal(j,tolerances=np.ones(4),weights=[1,0,0,0])
    b=allocate_for_goal(j,tolerances=np.ones(4),weights=[0,0,0,1])
    assert a['order'][0]==0 and b['order'][0]==3
    assert a['continuous_resources'][0]>1 and b['continuous_resources'][3]>1
    for r in (a,b):assert r['model_error_squared']<=r['model_error_budget_squared']*(1+1e-12)
    assert allocate_for_goal(j,tolerances=np.ones(4),weights=np.zeros(4))['continuous_resources']==[1.,1.,1.,1.]


def test_discrete_band_selection_matches_exhaustive_budget_search():
    c=np.array([1.,7.,2.,5.,3.,4.]);result=binary_band_choice(c,fine_bands=2)
    errors=[]
    for pair in itertools.combinations(range(6),2):
        resources=np.ones(6);resources[list(pair)]=8
        errors.append(float(np.sum((c/resources)**2)))
    assert result['model_error_squared']==min(errors)
    assert result['selected_bands']==[1,3]

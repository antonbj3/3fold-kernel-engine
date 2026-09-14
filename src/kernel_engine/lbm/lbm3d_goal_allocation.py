"""Goal-specific allocation from a measured local response Jacobian.

This predicts resource allocation under the existing water-filling error
model. It does not certify a mesh or predict sensitivities in a new flow.
"""
import numpy as np
from kernel_engine.allocation.d_poxel_waterfilling_unification import unified_alloc,err2


def allocate_for_goal(jacobian,*,tolerances,weights=None,error_fraction=.25):
    """Whiten observable rows by goal tolerances and allocate mesh resources.

    jacobian[:,i] is the measured response to the coarsening defect in band i.
    Zero row weight removes an observable from this goal. Reweighting does
    not require another PDE solve while the state and observables are fixed.
    """
    j=np.asarray(jacobian,dtype=float);tol=np.asarray(tolerances,dtype=float)
    if j.ndim!=2 or min(j.shape)<1 or tol.shape!=(j.shape[0],):raise ValueError('Jacobian and per-observable tolerances required')
    w=np.ones(j.shape[0]) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=tol.shape or not np.isfinite(j).all() or not np.isfinite(tol).all() or not np.isfinite(w).all() or np.any(tol<=0) or np.any(w<0) or not 0<error_fraction<=1:
        raise ValueError('finite Jacobian, positive tolerances and nonnegative weights required')
    c=np.linalg.norm(j*(np.sqrt(w)/tol)[:,None],axis=0)
    budget=float(np.sum(c*c)*error_fraction)
    levels,waterlevel=(np.ones(j.shape[1]),0.) if budget==0 else unified_alloc(c,1,budget)
    return {'sensitivities':c.tolist(),'order':np.argsort(-c,kind='stable').tolist(),
            'continuous_resources':levels.tolist(),'waterlevel':waterlevel,
            'model_error_budget_squared':budget,'model_error_squared':err2(levels,c)}


def binary_band_choice(sensitivities,*,fine_bands,fine_resource=8):
    """Equal-cost two-level band selection under the same separable model."""
    c=np.asarray(sensitivities,float)
    if c.ndim!=1 or not np.isfinite(c).all() or np.any(c<0) or not isinstance(fine_bands,int) or not 0<=fine_bands<=c.size or not np.isfinite(fine_resource) or fine_resource<=1:
        raise ValueError('invalid band resources')
    chosen=np.argsort(-c,kind='stable')[:fine_bands]
    resources=np.ones(c.size);resources[chosen]=fine_resource
    wall_resources=np.ones(c.size);wall_resources[:fine_bands]=fine_resource
    return {'selected_bands':sorted(chosen.tolist()),'model_error_squared':err2(resources,c),
            'wall_prefix_model_error_squared':err2(wall_resources,c),
            'fine_equivalent_cell_fraction':float(resources.sum()/(fine_resource*c.size))}

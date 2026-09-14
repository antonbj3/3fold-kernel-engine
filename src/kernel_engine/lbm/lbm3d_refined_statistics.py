"""Plane statistics on the active portions of two wall blocks and a coarse core."""
import numpy as np


def fold_wall_blocks(raw,*,nx,height,nz,wall_cells,samples):
    """Normalize each plane by its own sample/cell count, then reflect walls.

    raw order: bottom fine block, top fine block, full coarse background.
    Covered coarse cells and interface ghosts never enter the observable.
    """
    if len(raw)!=3 or samples<1 or min(nx,nz,wall_cells)<2 or any(n%2 for n in (nx,height,nz,wall_cells)) or 2*wall_cells>=height:
        raise ValueError('invalid two-level statistics geometry')
    shapes=[(10,wall_cells+2),(10,wall_cells+2),(10,height//2+2)]
    if any(np.shape(a)!=shape for a,shape in zip(raw,shapes)):raise ValueError('statistics shape mismatch')
    fine=[np.asarray(a,dtype=float)/(2**32*samples*nx*nz) for a in raw[:2]]
    coarse=np.asarray(raw[2],dtype=float)/(2**32*samples*(nx//2)*(nz//2))
    b=wall_cells//2+1;t=height//2-wall_cells//2
    values=np.concatenate([fine[0][:,1:-1],coarse[:,b:t+1],fine[1][:,1:-1]],axis=1)
    positions=np.r_[np.arange(wall_cells)+.5,np.arange(wall_cells+1,height-wall_cells,2),np.arange(height-wall_cells,height)+.5]
    half=(values.shape[1]+1)//2;sign=np.array([1,1,-1,1,1,1,1,-1,1,-1])[:,None]
    folded=.5*(values[:,:half]+sign*values[:,::-1][:,:half])
    stress=np.array([folded[4]-folded[1]**2,folded[5]-folded[2]**2,folded[6]-folded[3]**2,
                     folded[7]-folded[1]*folded[2],folded[8]-folded[1]*folded[3],folded[9]-folded[2]*folded[3]])
    return positions[:half],folded[1],stress,folded[0]

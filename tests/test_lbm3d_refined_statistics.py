import numpy as np
from kernel_engine.lbm.lbm3d_refined_statistics import fold_wall_blocks


def test_plane_normalization_excludes_ghosts_and_covered_cells():
    nx,h,nz,nf,samples=8,32,8,8,3
    raw=[np.full((10,nf+2),2**50,np.int64),np.full((10,nf+2),2**50,np.int64),np.full((10,h//2+2),2**50,np.int64)]
    for side in range(2):
        raw[side][:,1:-1]=0;raw[side][0,1:-1]=samples*nx*nz*2**32
        raw[side][1,1:-1]=samples*nx*nz*2**28
        raw[side][4,1:-1]=samples*nx*nz*2**24
    raw[2][:,nf//2+1:h//2-nf//2+1]=0
    raw[2][0,nf//2+1:h//2-nf//2+1]=samples*(nx//2)*(nz//2)*2**32
    raw[2][1,nf//2+1:h//2-nf//2+1]=samples*(nx//2)*(nz//2)*2**28
    raw[2][4,nf//2+1:h//2-nf//2+1]=samples*(nx//2)*(nz//2)*2**24
    y,u,stress,rho=fold_wall_blocks(raw,nx=nx,height=h,nz=nz,wall_cells=nf,samples=samples)
    np.testing.assert_array_equal(y,np.r_[np.arange(8)+.5,np.arange(9,16,2)])
    np.testing.assert_array_equal(u,np.full(len(y),1/16));np.testing.assert_array_equal(rho,np.ones(len(y)))
    np.testing.assert_array_equal(stress,np.zeros((6,len(y))))

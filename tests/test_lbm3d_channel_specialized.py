import numpy as np
import pytest
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
from kernel_engine.lbm.lbm3d_channel_specialized import SpecializedChannelSimulation


@pytest.mark.parametrize('recursive',[False,True])
def test_specialized_matrix_collision_is_byte_exact_with_force_walls(recursive):
    rng=np.random.default_rng(447);shape=(6,8,6);solid=np.zeros(shape,np.int32);solid[:,0,:]=1;solid[:,-1,:]=1
    u=rng.normal(0,.008,(3,)+shape);q=lb.quantize(lb.equilibrium(np.ones(shape),u))
    kwargs=dict(force_density=1e-5,tau=.8,cs=.1,bulk_tau=1.,recursive=recursive,solid=solid)
    a=ChannelSimulation(q,**kwargs);b=SpecializedChannelSimulation(q,**kwargs)
    for _ in range(20):a.step();b.step()
    np.testing.assert_array_equal(a.numpy(),b.numpy())
    np.testing.assert_array_equal(a.wall_impulse.numpy(),b.wall_impulse.numpy())
    assert a.failure.numpy()[0]==b.failure.numpy()[0]==0


def test_specialized_channel_factory_preserves_resident_refinement_states():
    from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
    kwargs=dict(device='cpu',nx=4,height=16,nz=4,wall_cells=4,recursive=True,cs_fine=.1)
    a=RefinedChannelGPU(**kwargs);b=RefinedChannelGPU(channel_factory=SpecializedChannelSimulation,**kwargs)
    for _ in range(10):a.step();b.step()
    for x,y in zip([*a.fine,a.coarse],[*b.fine,b.coarse]):np.testing.assert_array_equal(x.numpy(),y.numpy())
    assert a.ledger()==b.ledger()

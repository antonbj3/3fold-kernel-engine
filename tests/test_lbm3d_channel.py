import numpy as np
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_channel import ChannelSimulation


def test_forcing_and_wall_impulse_close_integer_momentum():
    shape=(4,6,4);solid=np.zeros(shape,np.int32);solid[:,0,:]=1;solid[:,-1,:]=1
    q=lb.quantize(lb.equilibrium(np.ones(shape),np.zeros((3,)+shape)))
    sim=ChannelSimulation(q,force_density=2e-5,tau=.8,cs=0,solid=solid)
    initial=lb.ledger(q);sim.step(17);final=lb.ledger(sim.numpy());impulse=sim.wall_impulse_numpy()
    assert final[0]==initial[0]
    assert final[1]-initial[1]+int(impulse[0])==17*np.sum(solid==0)*sim.force_units
    assert final[2]-initial[2]+int(impulse[1])==0
    assert final[3]-initial[3]+int(impulse[2])==0
    assert sim.failure.numpy()[0]==0


def test_zero_force_matches_original_complete_state():
    rng=np.random.default_rng(33);shape=(4,6,4)
    solid=np.zeros(shape,np.int32);solid[:,0,:]=1;solid[:,-1,:]=1
    u=rng.uniform(-.01,.01,(3,)+shape)
    q=lb.quantize(lb.equilibrium(np.ones(shape),u))
    a=lb.Simulation(q,tau=.7,mode='hermite_mrt',solid=solid)
    b=ChannelSimulation(q,force_density=0,tau=.7,solid=solid)
    a.step(13);b.step(13)
    np.testing.assert_array_equal(a.numpy(),b.numpy())


def test_periodic_force_delivers_physical_half_step_velocity():
    shape=(4,4,4);q=lb.quantize(lb.equilibrium(np.ones(shape),np.zeros((3,)+shape)))
    sim=ChannelSimulation(q,force_density=1e-5,tau=.8,cs=0);sim.step(23)
    rho,u=lb.fields(sim.numpy());physical=u[0]+.5*sim.force_density/rho
    np.testing.assert_allclose(physical,23.5*sim.force_density,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(sim.wall_impulse_numpy(),0)


def test_plane_statistics_match_independent_fields():
    import warp as wp
    from kernel_engine.lbm.lbm3d_channel_statistics import accumulate
    rng=np.random.default_rng(18);shape=(4,6,4)
    mask=np.zeros(shape,np.int32);mask[:,0,:]=1;mask[:,-1,:]=1
    q=lb.quantize(lb.equilibrium(np.ones(shape),rng.uniform(-.03,.03,(3,)+shape)))
    s=ChannelSimulation(q,force_density=1e-5,tau=.8,solid=mask)
    stats=wp.zeros((10,shape[1]),dtype=wp.int64,device='cpu')
    wp.launch(accumulate,shape,[s.a,s.args[0],s.args[1],wp.float64(2**40),wp.int64(s.force_units),stats,s.failure],device='cpu')
    rho,u=lb.fields(q);u[0]+=.5*s.force_density/rho
    values=np.array([rho,*u,u[0]**2,u[1]**2,u[2]**2,u[0]*u[1],u[0]*u[2],u[1]*u[2]])
    scaled=values*2**32
    # Warp round follows C/CUDA: halfway values round away from zero.
    expected=np.copysign(np.floor(np.abs(scaled)+.5),scaled).astype(np.int64)
    expected[:,mask==1]=0
    np.testing.assert_array_equal(stats.numpy(),expected.sum(axis=(1,3)))

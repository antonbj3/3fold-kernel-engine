import pytest
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


def test_independent_bulk_rate_removes_trace_preserves_shear_relaxation():
    from kernel_engine.lbm import lbm3d_mrt_les as lb
    from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
    shape=(4,4,4)
    eq=lb.equilibrium(np.ones(shape),np.zeros((3,)+shape))
    c=lb.C.astype(float);h=np.einsum('qa,qb->qab',c,c)-np.eye(3)[None]/3
    pi=np.eye(3)*.0003;pi[0,1]=pi[1,0]=.0002
    f=eq+4.5*lb.W[:,None,None,None]*np.einsum('qab,ab->q',h,pi)[:,None,None,None]
    q=lb.quantize(f);s=ChannelSimulation(q,force_density=0,tau=.8,cs=0,bulk_tau=1.)
    s.step();after=s.numpy()/2**40
    stress=np.einsum('qa,qb,qxyz->ab',c,c,after-eq)/np.prod(shape)
    assert abs(np.trace(stress))<2e-11
    np.testing.assert_allclose(stress[0,1],(1-1/.8)*pi[0,1],atol=2e-11,rtol=0)
    assert lb.ledger(s.numpy())==lb.ledger(q)


def test_bulk_rate_uses_matching_guo_source_in_moment_space():
    from kernel_engine.lbm import lbm3d_mrt_les as lb
    from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
    shape=(4,4,4);u=np.zeros((3,)+shape);u[0]=.031;u[1]=.017;u[2]=-.013
    q=lb.quantize(lb.equilibrium(np.ones(shape),u));sim=ChannelSimulation(q,force_density=1e-5,tau=.8,cs=0,bulk_tau=1.)
    f=q[:,0,0,0]/2**40;rho=f.sum();vel=lb.C.T@f/rho;vel[0]+=.5*sim.force_density/rho
    eq=lb.equilibrium(np.array([[[rho]]]),vel[:,None,None,None])[:,0,0,0]
    cu=lb.C@vel
    source=lb.W*sim.force_density*(3*(lb.C[:,0]-vel[0])+9*cu*lb.C[:,0])
    rates=np.r_[np.zeros(4),1.,np.full(5,1/.8),np.ones(9)]
    expected=f-lb.MI_HERMITE@(rates*(lb.M_HERMITE@(f-eq)))+lb.MI_HERMITE@((1-.5*rates)*(lb.M_HERMITE@source))
    sim.step();actual=sim.numpy()[:,0,0,0]/2**40
    np.testing.assert_allclose(actual,expected,atol=5e-12,rtol=0)
    before=lb.ledger(q);after=lb.ledger(sim.numpy())
    assert after[0]==before[0] and after[1]-before[1]==sim.force_units*np.prod(shape)


@pytest.mark.parametrize('force',[0.,1e-5])
@pytest.mark.parametrize('bulk',[None,1.])
def test_recursive_forced_collision_matches_third_hermite_tensor(force,bulk):
    from kernel_engine.lbm import lbm3d_mrt_les as lb
    from kernel_engine.lbm.lbm3d_channel import ChannelSimulation
    c=lb.C.astype(float);shape=(4,4,4);velocity=np.array([.07,.02,-.01]);u=np.broadcast_to(velocity[:,None,None,None],(3,)+shape)
    f=lb.equilibrium(np.ones(shape),u)+lb.W[:,None,None,None]*(c[:,0]*c[:,1])[:,None,None,None]*.001
    q=lb.quantize(f);sim=ChannelSimulation(q,force_density=force,tau=.8,cs=0,bulk_tau=bulk,recursive=True)
    g=q[:,0,0,0]/2**40;rho=g.sum();v=c.T@g/rho;v[0]+=.5*sim.force_density/rho
    cu=c@v;v2=v@v
    eq=lb.W*rho*(1+3*cu+4.5*cu**2-1.5*v2+4.5*cu**3-4.5*cu*v2)
    force_vector=np.array([sim.force_density,0.,0.]);source2=np.outer(v,force_vector)+np.outer(force_vector,v)
    pi=np.einsum('qa,qb,q->ab',c,c,g-eq)+.5*source2
    post_pi=(1-1/.8)*pi
    if bulk is not None:post_pi+=np.eye(3)*(1/.8-1/bulk)*np.trace(pi)/3
    h3=np.einsum('qa,qb,qc->qabc',c,c,c)-(np.einsum('qa,bc->qabc',c,np.eye(3))+np.einsum('qb,ac->qabc',c,np.eye(3))+np.einsum('qc,ab->qabc',c,np.eye(3)))/3
    third=np.einsum('a,bc->abc',v,post_pi)+np.einsum('b,ac->abc',v,post_pi)+np.einsum('c,ab->abc',v,post_pi)
    recursive=4.5*lb.W*np.einsum('qabc,abc->q',h3,third)
    source=lb.W*sim.force_density*(3*(c[:,0]-v[0])+9*cu*c[:,0]+(13.5*cu**2-4.5*v2)*c[:,0]-9*cu*v[0])
    rates=np.r_[np.zeros(4),np.full(6,1/.8),np.ones(9)]
    if bulk is not None:rates[4]=1/bulk
    expected=g-lb.MI_HERMITE@(rates*(lb.M_HERMITE@(g-eq)))+lb.MI_HERMITE@((1-.5*rates)*(lb.M_HERMITE@source))+recursive
    sim.step();np.testing.assert_allclose(sim.numpy()[:,0,0,0]/2**40,expected,atol=5e-12,rtol=0)
    before=lb.ledger(q);after=lb.ledger(sim.numpy())
    assert after[0]==before[0] and after[1]-before[1]==sim.force_units*np.prod(shape)

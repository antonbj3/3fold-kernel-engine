"""Independent moment, streaming, quantization and invalid-state controls."""
import numpy as np
import pytest
from kernel_engine.lbm import lbm3d_mrt_les as lb


def state(n=4):
    rng=np.random.default_rng(712)
    rho=1+rng.uniform(-.01,.01,(n,n,n))
    u=rng.uniform(-.02,.02,(3,n,n,n))
    return lb.quantize(lb.equilibrium(rho,u))


def test_lattice_and_second_moments():
    np.testing.assert_array_equal(lb.C[lb.OPP],-lb.C)
    np.testing.assert_allclose(lb.MI@lb.M,np.eye(19),atol=2e-15)
    rho=np.ones((4,4,4));u=np.full((3,4,4,4),.023)
    f=lb.equilibrium(rho,u)
    np.testing.assert_allclose(f.sum(0),rho,atol=2e-15)
    for i in range(3):
        for j in range(3):
            stress=np.einsum('q,q...->...',lb.C[:,i]*lb.C[:,j],f)
            np.testing.assert_allclose(stress,rho*(u[i]*u[j]+(i==j)/3),atol=2e-15)


@pytest.mark.parametrize('mode',['bgk','mrt'])
def test_collision_and_periodic_stream_conserve(mode):
    q=state(); sim=lb.Simulation(q,tau=.67,mode=mode)
    sim.step(11)
    assert lb.ledger(sim.numpy())==lb.ledger(q)
    assert sim.failure.numpy()[0]==0
    sim2=lb.Simulation(q,tau=.67,mode=mode);sim2.step(11)
    np.testing.assert_array_equal(sim.numpy(),sim2.numpy())


def test_numpy_collision_and_stream_reference():
    q=state();f=q.astype(float)/2**40
    rho,u=lb.fields(q); eq=lb.equilibrium(rho,u)
    dm=np.einsum('hq,q...->h...',lb.M,f-eq)
    pi=np.einsum('qa,qb,q...->ab...',lb.C,lb.C,f-eq)
    norm=np.sqrt(np.sum(pi*pi,axis=(0,1)))
    tau=(.67+np.sqrt(.67**2+18*np.sqrt(2)*.1**2*norm/rho))/2
    dm[:4]=0;dm[4:10]/=tau
    post=f-np.einsum('qh,h...->q...',lb.MI,dm)
    expected=lb.quantize(post)
    # quantize uses the target floating moments: explicitly restore input
    # integer invariants, matching the scheme rather than floating sums.
    delta_j=np.einsum('qd,q...->d...',lb.C.astype(np.int64),q-expected)
    expected[1]+=delta_j[0];expected[3]+=delta_j[1];expected[5]+=delta_j[2]
    expected[0]+=q.sum(0)-expected.sum(0)
    expected=np.array([np.roll(expected[h],tuple(lb.C[h]),axis=(0,1,2)) for h in range(19)])
    sim=lb.Simulation(q,tau=.67);sim.step()
    assert np.max(np.abs(sim.numpy()-expected))<=4


def test_solid_links_mass_and_sentinels():
    q=state();solid=np.zeros((4,4,4),np.int32);solid[:,0,:]=1;solid[2,2,2]=1
    sim=lb.Simulation(q,tau=.8,solid=solid);sim.step(7);after=sim.numpy()
    assert lb.ledger(after)[0]==lb.ledger(q)[0]
    np.testing.assert_array_equal(after[:,solid==1],q[:,solid==1])
    assert sim.failure.numpy()[0]==0


def test_bgk_recovery_and_smagorinsky_parameter():
    q=state();a=lb.Simulation(q,tau=1.,cs=0,mode='bgk');b=lb.Simulation(q,tau=1.,cs=0)
    a.step();b.step();assert np.max(np.abs(a.numpy()-b.numpy()))<=4
    c=lb.Simulation(q,tau=.67,cs=0);d=lb.Simulation(q,tau=.67,cs=.1)
    c.step(2);d.step(2);assert not np.array_equal(c.numpy(),d.numpy())


def test_invalid_state_sticky_failure():
    q=state();q[:,0,0,0]=0
    sim=lb.Simulation(q,tau=.8);sim.step();assert sim.failure.numpy()[0]==1
    sim.step();assert sim.failure.numpy()[0]==1
    with pytest.raises(ValueError):lb.Simulation(q,tau=.5)
    with pytest.raises(ValueError):lb.Simulation(q,tau=.8,cs=-1)
    with pytest.raises(ValueError):lb.Simulation(q,tau=.8,bits=63)
    with pytest.raises(OverflowError):lb.ledger(np.full((19,2,2,2),2**62,dtype=np.int64))

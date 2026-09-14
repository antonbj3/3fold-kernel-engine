import numpy as np
import pytest
from kernel_engine.lbm import lbm3d_mrt_les as lb
from kernel_engine.lbm.lbm3d_refinement import split_parent_mass,restrict_child_mass,rescale_incoming_stress,reflux_population_mass


def test_signed_octant_mass_and_momentum_exact():
    rng=np.random.default_rng(702)
    parent=rng.integers(-2**42,2**42,size=(19,3,2,4),dtype=np.int64)
    children=split_parent_mass(parent)
    np.testing.assert_array_equal(restrict_child_mass(children),parent)
    for octant in range(8):
        dx,dy,dz=octant//4,(octant//2)%2,octant%2
        assert np.max(np.abs(children[:,dx::2,dy::2,dz::2].astype(float)-parent/8))<1
    np.testing.assert_array_equal(np.einsum('qd,qxyz->d',lb.C.astype(np.int64),parent),
                                  np.einsum('qd,qxyz->d',lb.C.astype(np.int64),children))


def test_stress_rescaling_preserves_physical_strain():
    rho=np.ones((4,4,4));u=np.zeros((3,4,4,4));u[0]=.03
    eq=lb.equilibrium(rho,u)
    # A prescribed xy stress; independent second Hermite tensor construction.
    weights=lb.W*9*lb.C[:,0]*lb.C[:,1]
    f=eq+weights[:,None,None,None]*.0002
    fine=rescale_incoming_stress(f,tau_source=.8,tau_target=1.1,dt_ratio=.5)
    np.testing.assert_allclose(fine.sum(0),rho,atol=1e-15)
    np.testing.assert_allclose(np.einsum('qd,q...->d...',lb.C,fine),rho*u,atol=1e-15)
    coarse_stress=np.einsum('q,q...->...',lb.C[:,0]*lb.C[:,1],f-eq)
    fine_stress=np.einsum('q,q...->...',lb.C[:,0]*lb.C[:,1],fine-eq)
    np.testing.assert_allclose(fine_stress/(1.1*.5),coarse_stress/.8,rtol=1e-11,atol=1e-16)
    back=rescale_incoming_stress(fine,tau_source=1.1,tau_target=.8,dt_ratio=2)
    np.testing.assert_allclose(back,f,atol=2e-15)


def test_reflux_closes_actual_interface_exchange():
    rng=np.random.default_rng(9);shape=(19,4,4)
    start=rng.integers(1000000,2000000,size=shape,dtype=np.int64)
    coarse_net=rng.integers(-1000,1000,size=shape,dtype=np.int64)
    fine_net=rng.integers(-1000,1000,size=shape,dtype=np.int64)
    after=reflux_population_mass(start+coarse_net,fine_net,coarse_net)
    np.testing.assert_array_equal(after-start,fine_net)
    with pytest.raises(OverflowError):reflux_population_mass(np.full(shape,2**62,np.int64),np.full(shape,2**62,np.int64),np.zeros(shape,np.int64))


def test_coupled_reference_conserves_every_macro_step():
    from kernel_engine.lbm.lbm3d_refinement import ReferenceRefinedChannel
    sim=ReferenceRefinedChannel(nx=4,height=16,nz=4,wall_cells=4)
    initial=sim.initial_ledger
    for _ in range(5):
        sim.step();l=sim.ledger();wall=sim.wall_impulse()
        assert l[0]==initial[0]
        expected=sim.steps*sim.nx*sim.h*sim.nz*sim.force_units
        assert l[1]-initial[1]+int(wall[0])==expected
        assert l[2]-initial[2]+int(wall[1])==0
        assert l[3]-initial[3]+int(wall[2])==0


def test_resident_coupling_matches_reference():
    from kernel_engine.lbm.lbm3d_refinement import ReferenceRefinedChannel
    from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
    a=ReferenceRefinedChannel(nx=4,height=16,nz=4,wall_cells=4)
    b=RefinedChannelGPU(device='cpu',nx=4,height=16,nz=4,wall_cells=4)
    for _ in range(10):a.step();b.step()
    assert a.ledger()==b.ledger()
    np.testing.assert_allclose(a.profile()[1],b.profile()[1],atol=1e-10,rtol=0)
    assert all(s.failure.numpy()[0]==0 for s in [*b.fine,b.coarse])


def test_resident_coupling_spatial_flux_and_wall_momentum():
    from kernel_engine.lbm.lbm3d_refinement import ReferenceRefinedChannel
    from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
    a=ReferenceRefinedChannel(nx=8,height=16,nz=8,wall_cells=4)
    b=RefinedChannelGPU(device='cpu',nx=8,height=16,nz=8,wall_cells=4)
    for sa,sb in zip([*a.fine,a.coarse],[*b.fine,b.coarse]):
        q=sa.numpy();x,y,z=np.indices(sa.shape)
        # Vary all momentum components across both periodic directions and the
        # interface; opposite population pairs preserve local mass exactly.
        perturb=np.rint(2**24*np.sin(2*np.pi*x/sa.shape[0])*np.cos(2*np.pi*z/sa.shape[2])*(1+y)).astype(np.int64)
        perturb[sa.args[0].numpy()!=0]=0
        for positive,negative in [(1,2),(3,4),(5,6)]:
            q[positive]+=perturb;q[negative]-=perturb
        a._replace(sa,q);b._replace(sb,q)
    initial=a.ledger()
    for _ in range(10):
        a.step();b.step()
        for sim in (a,b):
            state=sim.ledger();wall=sim.wall_impulse()
            assert state[0]==initial[0]
            assert [state[k+1]-initial[k+1]+int(wall[k]) for k in range(3)]==[sim.steps*sim.nx*sim.h*sim.nz*sim.force_units,0,0]
    for sa,sb in zip([*a.fine,a.coarse],[*b.fine,b.coarse]):
        np.testing.assert_allclose(sa.numpy()/2**sa.bits,sb.numpy()/2**sb.bits,atol=1e-10,rtol=0)

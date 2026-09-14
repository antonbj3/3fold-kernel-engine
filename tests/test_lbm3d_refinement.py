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


@pytest.mark.parametrize('bulk',[None,1.])
@pytest.mark.parametrize('cs',[0.,.1])
@pytest.mark.parametrize('recursive',[False,True])
def test_resident_coupling_spatial_flux_and_wall_momentum(cs,recursive,bulk):
    from kernel_engine.lbm.lbm3d_refinement import ReferenceRefinedChannel
    from kernel_engine.lbm.lbm3d_refinement_gpu import RefinedChannelGPU
    a=ReferenceRefinedChannel(nx=8,height=16,nz=8,wall_cells=4,cs_fine=cs,recursive=recursive,bulk_tau_fine=bulk)
    b=RefinedChannelGPU(device='cpu',nx=8,height=16,nz=8,wall_cells=4,cs_fine=cs,recursive=recursive,bulk_tau_fine=bulk)
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


@pytest.mark.parametrize('ratio',[.5,2.0])
def test_sgs_transfer_preserves_physical_viscosity_and_strain(ratio):
    from kernel_engine.lbm.lbm3d_refinement import sgs_transfer_taus
    rho=np.array([.95,1.,1.05]);norm=np.array([0.,.002,.02])
    ts=.5008;tt=.5+(ts-.5)/ratio;cs=.1
    source,target=sgs_transfer_taus(rho,norm,ts,tt,cs,cs/ratio,ratio)
    transferred=ratio*target/source*norm
    independently_computed=.5*(tt+np.sqrt(tt**2+18*np.sqrt(2)*(cs/ratio)**2*transferred/rho))
    np.testing.assert_allclose(target,independently_computed,rtol=1e-15)
    np.testing.assert_allclose((target-.5)*ratio,source-.5,atol=2e-16,rtol=1e-12)
    np.testing.assert_allclose(transferred/(target*ratio),norm/source,rtol=1e-15)
    back_source,back_target=sgs_transfer_taus(rho,transferred,tt,ts,cs/ratio,cs,1/ratio)
    np.testing.assert_allclose(transferred/ratio*back_target/back_source,norm,atol=1e-17)


def test_guarded_reduction_handles_cancellation_without_wrap():
    from kernel_engine.lbm.lbm3d_refinement import checked_int64_sum,ReferenceRefinedChannel
    # A direct NumPy reduction would overflow before the cancellation.
    values=np.array([2**62,2**62,-2**62,-2**62,17],dtype=np.int64)
    assert checked_int64_sum(values)==17
    assert checked_int64_sum(np.array([-2**63],dtype=np.int64))==-2**63
    with pytest.raises(OverflowError):checked_int64_sum(np.array([2**62,2**62],dtype=np.int64))
    with pytest.raises(OverflowError):ReferenceRefinedChannel(nx=384,height=128,nz=192,wall_cells=16)


@pytest.mark.parametrize('ratio',[.5,2.])
@pytest.mark.parametrize('cs',[0.,.1])
def test_split_bulk_transfer_strain_closure_and_roundtrip(ratio,cs):
    from kernel_engine.lbm.lbm3d_refinement import split_relaxation_stress
    rng=np.random.default_rng(513)
    pi=rng.normal(0,.002,(3,3,7));pi=(pi+pi.swapaxes(0,1))/2
    rho=np.linspace(.98,1.02,7);ts=.503;tt=.5+(ts-.5)/ratio
    bs=1.;bt=.5+(bs-.5)/ratio
    out,source,target=split_relaxation_stress(rho,pi,ts,tt,cs,cs/ratio,ratio,bs,bt)
    tr=np.trace(pi,axis1=0,axis2=1);otr=np.trace(out,axis1=0,axis2=1)
    np.testing.assert_allclose(otr/(ratio*bt),tr/bs,atol=1e-17)
    dev=pi-np.eye(3)[:,:,None]*tr/3
    odev=out-np.eye(3)[:,:,None]*otr/3
    np.testing.assert_allclose(odev/(ratio*target),dev/source,atol=1e-17)
    computed=.5*(tt+np.sqrt(tt*tt+18*np.sqrt(2)*(cs/ratio)**2*np.sqrt(np.sum(out*out,axis=(0,1)))/rho))
    np.testing.assert_allclose(target,computed,atol=2e-16,rtol=1e-15)
    back,_,_=split_relaxation_stress(rho,out,tt,ts,cs/ratio,cs,1/ratio,bt,bs)
    np.testing.assert_allclose(back,pi,atol=1e-17)

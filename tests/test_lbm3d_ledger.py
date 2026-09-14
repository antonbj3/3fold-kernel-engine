import numpy as np
import pytest
from kernel_engine.lbm.lbm3d_ledger import exact_sum,int64_limbs,population_ledger,RADIX
from kernel_engine.lbm.lbm3d_mrt_les import C,Simulation


def test_signed_limb_carries_and_limits():
    for value in [0,1,-1,RADIX-1,RADIX,-RADIX,-RADIX-1,7*RADIX+9,-7*RADIX-9]:
        high,low=int64_limbs(value)
        assert -RADIX<=high<RADIX and 0<=low<RADIX
        assert high*RADIX+low==value
    with pytest.raises(OverflowError):int64_limbs(RADIX**2)


def test_sum_never_wraps_with_large_cancelling_terms():
    a=np.array([RADIX-1,RADIX-1,-RADIX,-RADIX,41],np.int64)
    assert exact_sum(a)==39
    assert exact_sum(np.full(20,RADIX-1,np.int64))==20*(RADIX-1)


def test_population_totals_match_independent_object_reduction():
    rng=np.random.default_rng(93)
    arrays=[rng.integers(-2**62,2**62,size=(19,2,3,2),dtype=np.int64) for _ in range(3)]
    totals=sum((a.astype(object).sum(axis=(1,2,3)) for a in arrays))
    expected=[sum(totals),*(C.astype(object).T@totals)]
    assert population_ledger(arrays,C)==expected


def test_explicit_wide_mode_preserves_population_bytes_and_default_rejection():
    q=np.full((19,32,32,32),20*2**40,np.int64)
    with pytest.raises(OverflowError):Simulation(q,tau=.8)
    sim=Simulation(q,tau=.8,ledger_mode='int64_limbs')
    np.testing.assert_array_equal(sim.numpy(),q)
    assert population_ledger([q],C)[0]>RADIX


def test_quantizer_explicit_wide_total_and_unchanged_small_population_bytes():
    from kernel_engine.lbm.lbm3d_mrt_les import quantize
    f=np.ones((19,32,32,32))
    with pytest.raises(OverflowError):quantize(f,44)
    q=quantize(f,44,ledger_mode='int64_limbs')
    assert population_ledger([q],C)[0]>RADIX
    np.testing.assert_array_equal(quantize(f[:,:2,:2,:2],44),quantize(f[:,:2,:2,:2],44,ledger_mode='int64_limbs'))

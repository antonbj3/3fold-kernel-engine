"""Exact population ledgers with explicit signed-int64 limbs.

Population storage stays int64. A total is encoded as high*2**63 + low,
where both words fit int64 and 0 <= low < 2**63. Bounded int64 partial
reductions precede Python-integer carry handling; no floating reductions.
"""
import numpy as np

RADIX=2**63


def int64_limbs(value):
    high,low=divmod(int(value),RADIX)
    if not -RADIX<=high<RADIX:
        raise OverflowError('two-word ledger range exceeded')
    return [high,low]


def exact_sum(values):
    a=np.asarray(values)
    if a.dtype!=np.int64 or not a.size:
        raise ValueError('nonempty int64 values required')
    bound=max(abs(int(a.min())),abs(int(a.max())))
    chunk=max(1,(RADIX-1)//max(1,bound))
    flat=a.ravel()
    total=sum(int(flat[i:i+chunk].sum(dtype=np.int64)) for i in range(0,flat.size,chunk))
    int64_limbs(total)
    return total


def population_ledger(arrays,c):
    """Combine disjoint population blocks in their shared physical mass unit."""
    totals=[0]*19
    for a in arrays:
        a=np.asarray(a)
        if a.dtype!=np.int64 or a.ndim!=4 or a.shape[0]!=19:
            raise ValueError('int64 D3Q19 population blocks required')
        for q in range(19):totals[q]+=exact_sum(a[q])
    result=[sum(totals),*[sum(int(c[q,k])*totals[q] for q in range(19)) for k in range(3)]]
    for value in result:int64_limbs(value)
    return result

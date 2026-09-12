#!/usr/bin/env python3
"""L 2nd axis: WATER-FILLING bit-allocation DEPLOYABILITY-GAP (generalizes L-3 feature-visibility law to D's poxel spine).
Frozen prediction in docs/L_DEPLOYABILITY_GAP_DOCTRINE.md (## Second axis).

D owns the ORACLE (water-filling in the eigenbasis, reality-anchored WF>uniform ~2-3x). The decorrelated unmeasured
piece = the DEPLOYABILITY-GAP: does a CHEAP detector (per-channel diagonal variance, no eigendecomposition) capture it?
  uniform  = B/n bits/channel (goal-blind allocator)                    MSE = 2^(-2B/n)*trace(Cov)  [basis-independent]
  oracle   = rotate to eigenbasis (KLT) + water-filling on eigenvalues  MSE = sum_i lam_i 2^(-2 b_i^WF(lam))
  cheap    = NO rotation + water-filling on diag(Cov)                   MSE = sum_i d_i 2^(-2 b_i^WF(d))
  axis_value = MSE_uniform - MSE_oracle ; dq_cheap = (MSE_uniform - MSE_cheap)/(MSE_uniform - MSE_oracle)
FROZEN: dq_cheap tracks channel-basis<->eigenbasis alignment -> 1 when Cov near-diagonal, DROPS as off-diagonal
correlation hides the allocation structure from the cheap diagonal detector (= L-3's feature-visibility law).
NULL: dq_cheap flat vs off-diagonal mass -> law is federation-specific, book honestly. CPU-only, seconds.
"""
import os
import numpy as np


def _evid(name):
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)




def bit_alloc(variances, B_total):
    """Reverse water-filling: minimize sum var_i 2^(-2 b_i) s.t. sum b_i = B_total, b_i>=0. Returns b (bits/channel)."""
    v = np.asarray(variances, float); v = np.maximum(v, 1e-300)
    idx = np.argsort(v)[::-1]; vs = v[idx]; n = len(v)
    logv = np.log2(vs)
    b_sorted = np.zeros(n)
    for k in range(n, 0, -1):                      # try k active channels (the k largest variances)
        log_theta = (logv[:k].sum() - 2.0 * B_total) / k
        b = 0.5 * (logv[:k] - log_theta)           # b_i for the k active
        if b[-1] >= -1e-12:                        # smallest active still non-negative -> valid active set
            b_sorted[:k] = np.maximum(b, 0.0); break
    out = np.zeros(n); out[idx] = b_sorted
    return out


def mse_alloc(true_vars, alloc_vars, B_total):
    """MSE when bits are allocated by WF on alloc_vars but distortion is paid on true_vars (high-rate D_i=var_i 2^-2b_i)."""
    b = bit_alloc(alloc_vars, B_total)
    return float(np.sum(np.asarray(true_vars) * 2.0 ** (-2.0 * b)))


def measure(Cov, B_per_ch=2.0):
    n = Cov.shape[0]; B = B_per_ch * n
    lam = np.linalg.eigvalsh(Cov); lam = np.maximum(lam, 0.0)     # eigen-variances (decorrelated)
    d = np.clip(np.diag(Cov), 0, None)                            # channel variances (cheap feature)
    tr = float(np.trace(Cov))
    mse_uniform = 2.0 ** (-2.0 * B_per_ch) * tr                   # basis-independent
    mse_oracle = mse_alloc(lam, lam, B)                           # KLT + WF on eigenvalues
    mse_cheap = mse_alloc(d, d, B)                                # no rotation, WF on diag
    axis_value = mse_uniform - mse_oracle
    dq_cheap = (mse_uniform - mse_cheap) / axis_value if axis_value > 1e-12 else float("nan")
    offdiag = float(np.linalg.norm(Cov - np.diag(np.diag(Cov)))) / (float(np.linalg.norm(Cov)) + 1e-30)
    # basis-alignment: how concentrated is each eigenvector on one channel (1 = channel basis = eigenbasis)
    w, U = np.linalg.eigh(Cov)
    align = float(np.mean(np.max(U ** 2, axis=0)))
    return dict(n=n, offdiag_mass=round(offdiag, 4), basis_align=round(align, 4),
                mse_uniform=round(mse_uniform, 5), mse_oracle=round(mse_oracle, 5), mse_cheap=round(mse_cheap, 5),
                axis_value=round(axis_value, 5), dq_cheap=round(dq_cheap, 4) if dq_cheap == dq_cheap else None)


def ar1_cov(n, rho, seed):
    """Correlated covariance via AR(1) Toeplitz correlation x heterogeneous variances (controllable off-diagonal)."""
    g = np.random.default_rng(seed)
    sd = np.exp(g.normal(0, 0.8, n))                              # heterogeneous channel std
    R = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return (sd[:, None] * sd[None, :]) * R


def main():
    print("=== WATER-FILLING deployability-gap: dq_cheap vs off-diagonal correlation (frozen H-WF-2) ===")
    print(f"{'rho':>5} {'offdiag':>8} {'align':>6} {'axis_val':>9} {'dq_cheap':>9}")
    rows = []
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95):
        # average over seeds for stability (conditioning-class -> low variance expected)
        rs = [measure(ar1_cov(32, rho, s)) for s in range(5)]
        dq = np.mean([r['dq_cheap'] for r in rs]); av = np.mean([r['axis_value'] for r in rs])
        od = np.mean([r['offdiag_mass'] for r in rs]); al = np.mean([r['basis_align'] for r in rs])
        print(f"{rho:>5} {od:>8.3f} {al:>6.3f} {av:>9.4f} {dq:>9.4f}")
        rows.append(dict(rho=rho, offdiag=round(od, 4), align=round(al, 4), axis_value=round(av, 4), dq_cheap=round(dq, 4)))
    # REAL anchor: sklearn digits covariance (D's reality-anchor dataset)
    try:
        from sklearn.datasets import load_digits
        X = load_digits().data; X = X[:, X.std(0) > 1e-6]          # drop dead pixels (zero-variance -> nullspace)
        Cov = np.cov(X, rowvar=False)
        r = measure(Cov)
        print(f"\nREAL sklearn-digits ({Cov.shape[0]} live px): offdiag={r['offdiag_mass']} align={r['basis_align']} "
              f"axis_value={r['axis_value']} dq_cheap={r['dq_cheap']}")
        rows.append(dict(real="digits", **r))
    except Exception as e:
        print("digits anchor skipped:", e)
    import json
    json.dump(rows, open(_evid("l_waterfill_deploygap_evidence.json"), "w"), indent=1)
    # verdict
    dqs = [r['dq_cheap'] for r in rows if r.get('dq_cheap') is not None and 'rho' in r]
    print(f"\nH-WF-2 (dq DROPS with correlation): dq at rho=0 -> {dqs[0]:.3f}, at rho=0.95 -> {dqs[-1]:.3f}; "
          f"monotone-decreasing={all(dqs[i] >= dqs[i+1] - 0.02 for i in range(len(dqs)-1))}")


if __name__ == "__main__":
    main()

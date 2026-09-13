#!/usr/bin/env python3
"""L 2nd-axis DEEPENING: the DETECTOR-COST SPECTRUM — is the deployability-gap dq = feature-SUFFICIENCY?
Frozen H-suff (docs/L_DEPLOYABILITY_GAP_DOCTRINE.md): unifies federation (sufficient feature -> dq high) and
water-filling-diagonal (insufficient -> dq low) as ONE information-theoretic law: dq = how sufficient the cheap
detector's feature is for the oracle allocation.

Detector of cost k = "resolve the top-k eigenvalues (power-iteration cost ~k), approximate the tail as flat (mean)".
  k=0  -> flat spectrum = uniform (dq~0 region);  k=n -> exact = oracle (dq=1).  dq(k) = the detector-cost curve.
FROZEN: dq(k) monotone-climbs dq_diag->1; KNEE (min k for dq>=0.9) small for low-rank/high-corr, larger for spread
spectra (digits). NULL: dq(k) flat/non-monotone -> sufficiency doesn't govern. CPU, seconds.
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('allocation',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import numpy as np


def _evid(name):
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


from l_waterfill_deploygap import bit_alloc, mse_alloc, ar1_cov


def dq_curve(Cov, B_per_ch=2.0, ks=None):
    n = Cov.shape[0]; B = B_per_ch * n
    lam = np.sort(np.maximum(np.linalg.eigvalsh(Cov), 0.0))[::-1]     # eigen-variances, descending
    d = np.clip(np.diag(Cov), 0, None)
    tr = float(lam.sum())
    mse_uniform = 2.0 ** (-2.0 * B_per_ch) * tr
    mse_oracle = mse_alloc(lam, lam, B)
    mse_diag = mse_alloc(d, d, B)                                     # k=0 cheap diagonal proxy (the L-3-analog detector)
    denom = mse_uniform - mse_oracle
    dq_diag = (mse_uniform - mse_diag) / denom if denom > 1e-12 else float("nan")
    if ks is None:
        ks = [k for k in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, n] if k <= n]
    out = {}
    for k in ks:
        if k >= n:
            av = lam
        else:
            mu = lam[k:].mean() if k < n else 0.0
            av = np.concatenate([lam[:k], np.full(n - k, mu)])       # resolve top-k eigenvalues, flatten the tail
        mse_k = mse_alloc(lam, av, B)
        out[k] = (mse_uniform - mse_k) / denom if denom > 1e-12 else float("nan")
    knee = next((k for k in ks if out[k] >= 0.9), n)
    return dict(n=n, dq_diag=round(float(dq_diag), 4), dq_curve={k: round(float(v), 4) for k, v in out.items()},
                knee_k_for_0p9=knee, knee_frac=round(knee / n, 3))


def main():
    print("=== DETECTOR-COST SPECTRUM: dq vs #resolved eigen-components k (frozen H-suff) ===")
    # synthetic: low-rank/high-corr (small knee expected) vs spread (large knee)
    for tag, Cov in [("ar1_rho0.4 (mild corr)", ar1_cov(48, 0.4, 0)),
                     ("ar1_rho0.9 (high corr)", ar1_cov(48, 0.9, 0)),
                     ("lowrank r=3+noise", None)]:
        if Cov is None:
            g = np.random.default_rng(1); F = g.normal(0, 1, (48, 3)); Cov = F @ F.T + 0.05 * np.eye(48)
        r = dq_curve(Cov)
        pts = " ".join(f"k{k}:{v:.2f}" for k, v in r['dq_curve'].items() if k in (0, 1, 2, 4, 8, 16, r['n']))
        print(f"\n{tag} (n={r['n']}): dq_diag={r['dq_diag']}  knee@dq0.9: k={r['knee_k_for_0p9']} ({r['knee_frac']*100:.0f}% of n)")
        print(f"  {pts}")
    # REAL anchor: sklearn digits
    try:
        from sklearn.datasets import load_digits
        X = load_digits().data; X = X[:, X.std(0) > 1e-6]
        Cov = np.cov(X, rowvar=False); r = dq_curve(Cov)
        pts = " ".join(f"k{k}:{v:.2f}" for k, v in r['dq_curve'].items() if k in (0, 1, 2, 4, 8, 16, 32, r['n']))
        print(f"\nREAL digits (n={r['n']}): dq_diag={r['dq_diag']}  knee@dq0.9: k={r['knee_k_for_0p9']} ({r['knee_frac']*100:.0f}% of n)")
        print(f"  {pts}")
        # monotonicity check
        vals = list(r['dq_curve'].values())
        mono = all(vals[i] <= vals[i+1] + 1e-6 for i in range(len(vals)-1))
        print(f"\nH-suff-1 (dq(k) monotone-climbs): {mono}; dq_diag={r['dq_diag']} -> dq(k=n)={vals[-1]:.3f}")
        import json
        json.dump({"real_digits": r}, open(_evid("l_waterfill_sufficiency_evidence.json"), "w"), indent=1)
    except Exception as e:
        print("digits skipped:", e)


if __name__ == "__main__":
    main()

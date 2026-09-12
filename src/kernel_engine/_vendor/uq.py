"""Conformal-prediction UQ guard - a distribution-free error bound for surrogates.

Level 3 of the compute backbone (UQ-guided full physics). Split conformal: calibrate the residuals on a
HELD-OUT set to get a distribution-free guarantee that unseen errors are covered with probability >= 1-alpha. It wraps
ANY surrogate (world model, ROM, neural operator) so that uncertainty becomes a MEASURABLE GATE, not a claim.
No distributional assumption (it holds for heavy tails too), finite-sample exact under exchangeability (Vovk).

Use in the compute backbone: the surrogate predicts y_hat +- q; if q (or the per-sample band) exceeds
a tolerance, fall back to full physics (FEM/Newton). A direct defence against false positives.

"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShuffleMargin:
    """Resultat av shuffle_margin (STEG1 fantom-signal-vakt)."""
    margin: float           # skill_real − skill_shuffle_mean  (>0 = genuin signal)
    skill_real: float       # explained variance (R^2), model vs true targets
    skill_shuffle_mean: float
    skill_shuffle_std: float
    z: float                # margin / (std + eps) - how many null standard deviations above the shuffle zero
    refuse: bool            # margin ≤ refuse_thr ⇒ True (≙ BaselineVerdict.BEATS_NEITHER)


def _ev_skill(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance (R^2) aggregated over all elements: 1 - SS_res/SS_tot."""
    yt = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    yp = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean(0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-30)


def shuffle_margin(y_true, y_pred, n_shuffle: int = 25, seed: int = 0,
                   refuse_thr: float = 0.0, z_thr: float = 3.0) -> ShuffleMargin:
    """An O(1)-per-evaluation negative control against IMAGINED signal.

    Compares the model's skill (explained variance) against the skill when the targets are SHUFFLED across
    sampel (label-permutation). Genuin signal ⇒ margin = skill_real − skill_shuffle
    > 0 (the prediction tracks the TRUE targets, not a permuted copy). A false positive (the world model
    sees patterns in noise / fits structure that survives shuffling) gives margin <= 0 and
    refusal.


    Unlike an exhaustive permutation p-value (O(n!), valid only for small-n
    transfer studies), this is
    O(n_shuffle) INDEPENDENT of n, so it scales to trajectories / rollouts. The
    principle: a cheap gate that catches false-positive signal on EVERY world model.

    y_true, y_pred: (N,) eller (N, d). n_shuffle slumpade radpermutationer av y_true.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) != len(yp):
        raise ValueError(f"y_true ({len(yt)}) and y_pred ({len(yp)}) have different lengths")
    # FINITENESS GUARD: empty or NaN/inf input gives skill=NaN, and NaN<=0 / NaN<z are both False, so refuse=False
    # would slip through.
    # tystar vakten. Default-deny: icke-finit/tom ⇒ REFUSA.
    if yt.size == 0 or not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        nan = float("nan")
        return ShuffleMargin(margin=nan, skill_real=nan, skill_shuffle_mean=nan,
                             skill_shuffle_std=nan, z=nan, refuse=True)
    skill_real = _ev_skill(yt, yp)
    rng = np.random.default_rng(seed)
    sh = np.array([_ev_skill(yt[rng.permutation(len(yt))], yp) for _ in range(n_shuffle)])
    sh_mean, sh_std = float(sh.mean()), float(sh.std())
    margin = skill_real - sh_mean
    z = margin / (sh_std + 1e-12)
    # REFUSE if margin <= threshold OR it is not z_thr standard deviations above the shuffle zero.
    # The z condition catches a chance-positive margin on pure noise (margin>0 but z~0) -
    # the literal "<=0" rule is necessary but not sufficient (anti false positive).
    refuse = bool(margin <= refuse_thr or z < z_thr)
    return ShuffleMargin(margin=margin, skill_real=skill_real,
                         skill_shuffle_mean=sh_mean, skill_shuffle_std=sh_std,
                         z=z, refuse=refuse)


def split_conformal_quantile(cal_residuals, alpha: float = 0.1) -> float:
    """(1−α)-konform kvantil med finite-sample-korrektion.

    cal_residuals: |y_true - y_pred| (optionally normalised) on the CALIBRATION set (NOT training).
    Returns q such that P(|y - y_hat| <= q) >= 1-alpha on unseen data, DISTRIBUTION-FREE (assuming only
    utbytbarhet). Garantin kommer av rang-statistik: kvantilen tas vid rang ⌈(n+1)(1−α)⌉.
    """
    # C/H197/H199 degenerate-threshold-VALUE class: an out-of-(0,1) alpha (e.g. 1.5) makes rank<=0, and Python's
    # negative indexing on r[rank-1] then SILENTLY returns a plausible-looking element from the end of the array
    # instead of failing loudly -- a fabricated quantile, not a real conformal guarantee. Fail loud on invalid alpha.
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    r = np.sort(np.abs(np.asarray(cal_residuals, dtype=float)))
    n = len(r)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))   # 1-indexerad
    if rank > n:
        return float("inf")                         # calibration set too small for the guarantee
    return float(r[rank - 1])


def conformal_coverage(test_residuals, q: float) -> float:
    """Empirical coverage: the fraction of test |residuals| <= q (should be >= 1-alpha if the guarantee holds)."""
    r = np.abs(np.asarray(test_residuals, dtype=float))
    if len(r) == 0:
        return 1.0
    return float((r <= q).mean())


def needs_full_physics(pred, q: float, rel_tol: float) -> np.ndarray:
    """Fallback policy (UQ-guided full physics): True where the conformal error bound q is too large
    relative to the prediction (q/|pred| > rel_tol) -> run full physics instead of the surrogate. Per sample
    if pred is an array. This is the gate that makes the surrogate safe (it knows when it is uncertain)."""
    pred = np.abs(np.asarray(pred, dtype=float))
    # H199 class (D/H201/J's find, 3rd instance in uq.py): `ratio > rel_tol` is False for a NaN rel_tol (or a NaN
    # ratio) -- NaN comparisons always False -- so a corrupted rel_tol silently returns "no fallback needed" even
    # at an enormous q/pred ratio, blindly trusting the surrogate exactly when it's least verified. Fail-closed:
    # a non-finite rel_tol or ratio must ALWAYS trigger the fallback.
    ratio = q / np.maximum(pred, 1e-30)
    return (ratio > rel_tol) | ~np.isfinite(ratio) | ~np.isfinite(rel_tol)


# -- Clopper-Pearson exact binomial CI (small-n wrapper for coverage) --


def clopper_pearson(k: int, n: int, alpha: float = 0.1):
    """Exact binomial (1-alpha) confidence interval for a proportion k/n (Clopper-Pearson).

    Distribution-free exact via Beta quantiles (coverage guarantee >= 1-alpha, never
    below - conservative). For small n this is the honest uncertainty on a
    coverage-/pass-frekvens som en normal-approximation (Wald) underskattar grovt.
    Returnerar (lo, hi). k=0 ⇒ lo=0; k=n ⇒ hi=1.
    """
    from scipy.stats import beta
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"requires 0<=k<=n and n>0 (k={k}, n={n})")
    # same degenerate-threshold-VALUE class as split_conformal_quantile (H199/D/F469's 2nd-function find): an
    # out-of-(0,1) alpha (e.g. 1.5) makes alpha/2>0.5, silently producing an INVERTED interval (lo>hi) from
    # beta.ppf instead of a real confidence interval -- no crash, just a fabricated/nonsensical result.
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def conformal_coverage_ci(test_residuals, q: float, alpha_ci: float = 0.1):
    """Empirisk coverage + Clopper-Pearson-CI. Returnerar (coverage, lo, hi).

    coverage = the fraction of test |residuals| <= q; (lo,hi) = the exact binomial CI on that
    frequency. It makes coverage a MEASUREMENT WITH ERROR BARS, not a bare number."""
    r = np.abs(np.asarray(test_residuals, dtype=float))
    n = len(r)
    if n == 0:
        return 1.0, 0.0, 1.0
    k = int((r <= q).sum())
    lo, hi = clopper_pearson(k, n, alpha_ci)
    return float(k / n), lo, hi


def coverage_meets_target(test_residuals, q: float, target: float = 0.9,
                          alpha_ci: float = 0.1) -> bool:
    """MARGIN-REQUIREMENT gate: the coverage TARGET is reached ONLY if the CI's
    LOWER bound >= target. A point estimate that grazes the threshold (lo < target)
    NEVER counts as a pass - small-n uncertainty must not hide a false positive.
    """
    _, lo, _ = conformal_coverage_ci(test_residuals, q, alpha_ci)
    return bool(lo >= target)

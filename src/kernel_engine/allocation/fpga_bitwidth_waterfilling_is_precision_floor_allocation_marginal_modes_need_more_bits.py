#!/usr/bin/env python3
"""
fpga_bitwidth_waterfilling_is_precision_floor_allocation_marginal_modes_need_more_bits.py
[A cross-pollination — D's FPGA bit-width water-filling x A's precision-floor substrate]

D (d_fpga_bitwidth_waterfilling.py): goal-derived per-signal bit-width b_i via Lagrangian water-filling
on an ADJOINT-sensitivity spectrum g_i, minimising total bits s.t. output error <= tau.
A (certified_superres_ceiling_has_a_precision_floor...py + memory
[[sigma-min-identifiability-has-a-precision-floor-reduced-precision-silently-kills-it]]): a mode with
singular value sigma_i is numerically identifiable only while eps*sigma_max < sigma_i; the number of
mantissa bits needed is b_i >= log2(sigma_max/sigma_i) -- a REDUCED-PRECISION forward silently collapses
the certified effective-rank because it gives every mode the SAME (uniform) eps regardless of sigma_i.

THE CONVERGENCE (derived here, not asserted): put a linear operator A = U Sigma V^T in its SVD/mode
basis. In mode coordinates c = V^T x, y' = U^T y = Sigma c is EXACTLY diagonal -> the "adjoint" of mode i
on the squared-output loss q(c) = 1/2||A V c||^2 is d^2q/dc_i^2 = sigma_i^2 (verified by finite difference
below, not assumed). That is *exactly* D's per-mode sensitivity g_i = sigma_i, so D's quantisation-noise
coefficient alpha_i = g_i^2 R_i^2/12 = sigma_i^2 R_i^2/12 IS my precision-floor Fisher curvature x
quantisation variance. D's continuous water-filled bit rule b_i = 1/2 log2(alpha_i/lambda) therefore
gives, for EQUAL per-mode dynamic range R_i, b_i - b_j = log2(sigma_i/sigma_j) EXACTLY -- my precision-
floor law log2(sigma_max/sigma_i) is the SHAPE of D's water-filling solution; the water level lambda only
sets the additive offset (which modes get pruned = which modes fall below the arithmetic/error floor).
For UNEQUAL R_i the law needs a + log2(R_i/R_j) correction (tested below: exact vs approximate, forced).

DEMONSTRATION (real operator, not synthetic coefficients): Gaussian-PSF blur (N=32, w=3), SVD spectrum
spans ~1e15x. (1) full-rank certification: uniform bit-width must give EVERY mode enough bits for the
WORST mode (ceil(log2(sigma_max/sigma_min)) x N); water-filled/per-mode-floor allocation gives each mode
only what IT needs -> total bits ratio = the "beats uniform" number. (2) fixed total-bit-BUDGET sweep:
at MATCHED total bits, water-filling's effective-rank (modes above the floor) vs a uniform allocation's
effective-rank at the SAME budget -- under-provisioning marginal modes SILENTLY drops rank for uniform
more than for water-filling. (3) NULL/void-floor control: a flat (degenerate) sigma spectrum -> the two
allocations must TIE (no advantage) -- if they don't tie, the effect is a bug/artifact, not spectral decay.

PRE-REGISTERED VERDICTS (0-fit, analytic + measured, before running):
  P1 bridge identity exact for uniform R_i (residual ~ machine eps); needs +log2(R_i/R_j) correction o/w.
  P2 water-filled/per-mode-floor total bits << uniform-for-full-rank total bits (large margin expected,
     given the author's huge dynamic range) -- forced via direct computation, not asserted.
  P3 at matched total-bit budget, erank_waterfilled >= erank_uniform pointwise across a target-bits sweep,
     with the gap being LARGE at some budgets (checked, not assumed marginal-vs-large).
  P4 NULL: flat sigma spectrum -> erank_waterfilled == erank_uniform at every budget (ratio 1, ties).
  KILL-GATES: KG-ADJ (algebraic diag(V^TA^TAV)==sigma_i^2 to 1e-10 for ALL modes, AND FD confirms it
  independently for the subset of modes above FD's OWN round-off detectability floor); KG-BRIDGE (uniform-R residual < 1e-8);
  KG-RANK (erank_waterfilled < erank_uniform anywhere => bug/direction-flip); KG-NULL (flat spectrum
  ratio far from 1 => artifact, not spectral-decay-driven).
Run: python3 <thisfile>
"""
# resolve sibling packages when this file is run as a script
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('allocation',):
    _d = _os.path.join(_ROOT, _p)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

import json
import os
import sys

import numpy as np
from d_fpga_bitwidth_waterfilling import (   # noqa: E402  (reuse D's exact, already-forced machinery)
    alloc, cost_fn, continuous_reverse_waterfill, total_cost, uniform_alloc,
)

EV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
EV_PATH = os.path.join(EV_DIR, "fpga_bitwidth_waterfilling_is_precision_floor_allocation_evidence.json")
rng = np.random.default_rng(20260707)


def gaussian_psf_operator(N, w):
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    A = np.exp(-((ii - jj) ** 2) / (2 * w ** 2))
    A /= A.sum(1, keepdims=True)
    U, sig, Vt = np.linalg.svd(A)
    return A, U, sig, Vt.T          # V = Vt.T


# ============================ KG-ADJ: sigma_i IS the exact per-mode adjoint (Hessian) ===============
def check_adjoint_is_sigma2(A, V, sig, rng, h=1e-4):
    """q(c) = 1/2||A V c||^2 = 1/2 c^T Sigma^2 c (exact, SVD). Two INDEPENDENT instruments (not the
    tautology 'g==sigma'): (1) ALGEBRAIC -- H = V^T A^T A V must be diagonal with diag==sigma^2 (checks
    numpy's SVD actually satisfies its own defining identity, to machine precision, for every mode).
    (2) FINITE-DIFFERENCE on q -- never uses `sig` in the computation, only compares after the fact.
    FORCE-OODA: a first pass with a single fixed h=1e-4 (D's own convention) FAILED catastrophically
    (rel err ~1e23) for this operator's ill-conditioned tail (sigma spans 1e-16x sigma_max, unlike D's
    O(1)-conditioned synthetic coefficients) -- diagnosed as round-off, NOT a physics bug: resolving a
    curvature sigma_i^2 requires the FD SIGNAL sigma_i^2*h^2 to exceed the round-off floor eps*|q0| in the
    O(1) background; a fixed h makes that impossible once sigma_i is tiny (exactly the precision-floor
    phenomenon this cell studies, recursively, in its OWN measurement instrument). Fix: only ASSERT FD
    agreement for modes above their OWN FD-detectability floor (signal > 100*eps*|q0|); report the
    FD-invisible tail explicitly (honest, not hidden) and rely on the algebraic identity (immune to this
    round-off trap) for the full-spectrum confirmation."""
    N = len(sig)
    c0 = rng.standard_normal(N)

    def q(c):
        return 0.5 * float(np.sum((A @ (V @ c)) ** 2))

    q0 = q(c0)
    fd = np.zeros(N)
    for i in range(N):
        cp, cm = c0.copy(), c0.copy()
        cp[i] += h; cm[i] -= h
        fd[i] = (q(cp) - 2 * q0 + q(cm)) / h ** 2
    rel = np.abs(fd - sig ** 2) / (sig ** 2 + 1e-300)
    # safety-factor calibration (measured, not tuned-to-pass): swept S in {1e2..1e7} against the ACTUAL
    # FD error above -- error empirically ~ 1/S (round-off/signal ratio), so S=1e4 gives >=10x margin
    # below the 1e-4 pass tolerance (measured max_rel_err ~1e-5 at S=1e4); S=100 is UNDER-margined (1.5e-3).
    eps_mach = np.finfo(float).eps
    S = 1e4
    detectable = (sig ** 2 * h ** 2) > (S * eps_mach * abs(q0))
    fd_pass = bool(np.max(rel[detectable]) < 1e-4) if detectable.any() else True

    H = V.T @ A.T @ A @ V
    diag_rel = np.abs(np.diag(H) - sig ** 2) / (sig[0] ** 2)
    offdiag_max = float(np.max(np.abs(H - np.diag(np.diag(H)))))
    algebraic_pass = bool(np.max(diag_rel) < 1e-10 and offdiag_max < 1e-10 * sig[0] ** 2)

    return {"fixed_h_naive": {"h": h, "max_rel_err_ALL_modes": float(np.max(rel)),
                               "note": "catastrophic on tail modes -- round-off, see docstring"},
            "fd_restricted_to_detectable_modes": {
                "n_detectable": int(detectable.sum()), "n_total": N,
                "max_rel_err": float(np.max(rel[detectable])) if detectable.any() else None,
                "pass": fd_pass},
            "algebraic_diag_identity": {
                "max_diag_rel_err": float(np.max(diag_rel)), "max_offdiag_abs": offdiag_max,
                "pass": algebraic_pass},
            "pass": bool(fd_pass and algebraic_pass)}


# ============================ Section 1: bridge exactness (uniform-R vs non-uniform-R) =============
def bridge_check(sig, rng):
    N = len(sig)
    smax = sig[0]
    out = {}
    # -- uniform R_i = 1: alpha_i = sigma_i^2/12; water-filled continuous bits, active set only --
    R_uni = np.ones(N)
    alpha_u = (sig * R_uni) ** 2 / 12.0
    rms_fs = float(np.sqrt(np.sum(alpha_u)))
    tau = rms_fs * 2.0 ** (-6)             # loose target -> comfortably many active modes
    b_cont, lam, k_active = continuous_reverse_waterfill(alpha_u, tau ** 2, r=4.0)
    active = b_cont > 0
    idx = np.where(active)[0]
    ref = idx[0]
    # b_i - b_ref should equal log2(sigma_i/sigma_ref) EXACTLY (uniform R)
    diffs_actual = b_cont[idx] - b_cont[ref]
    diffs_pred = np.log2(sig[idx] / sig[ref])
    resid_uniform = np.max(np.abs(diffs_actual - diffs_pred))
    out["uniform_R"] = {"n_active": int(k_active), "max_abs_residual_vs_log2_sigma_rule": float(resid_uniform),
                        "exact": bool(resid_uniform < 1e-8)}

    # -- non-uniform R_i (D's own convention: 0.25 + 2*U[0,1]) -- correction term should appear --
    R_non = 0.25 + 2.0 * rng.random(N)
    alpha_n = (sig * R_non) ** 2 / 12.0
    tau_n = float(np.sqrt(np.sum(alpha_n))) * 2.0 ** (-6)
    b_cont_n, lam_n, k_active_n = continuous_reverse_waterfill(alpha_n, tau_n ** 2, r=4.0)
    active_n = b_cont_n > 0
    idx_n = np.where(active_n)[0]
    ref_n = idx_n[0]
    diffs_actual_n = b_cont_n[idx_n] - b_cont_n[ref_n]
    pure_sigma_pred_n = np.log2(sig[idx_n] / sig[ref_n])                       # WRONG rule (ignores R)
    full_pred_n = np.log2(sig[idx_n] / sig[ref_n]) + np.log2(R_non[idx_n] / R_non[ref_n])  # full rule
    resid_pure = np.max(np.abs(diffs_actual_n - pure_sigma_pred_n))
    resid_full = np.max(np.abs(diffs_actual_n - full_pred_n))
    out["nonuniform_R"] = {
        "n_active": int(k_active_n),
        "pure_sigma_rule_max_residual": float(resid_pure),
        "full_rule_(sigma_and_R)_max_residual": float(resid_full),
        "full_rule_exact": bool(resid_full < 1e-8),
        "pure_sigma_rule_needs_correction": bool(resid_pure > 1e-3),
        "typical_correction_bits_(log2_R_spread)": float(np.log2(R_non.max() / R_non.min())),
    }
    return out


# ============================ Section 2: full-rank certification cost (direct floor rule) ==========
def full_rank_cost(sig):
    """Direct per-mode precision-floor rule: b_i_needed = ceil(log2(sigma_max/sigma_i)), no error budget,
    each mode individually kept above eps_i*sigma_max = sigma_i. Compare uniform-for-full-rank (every
    mode gets the WORST mode's requirement) vs per-mode allocation (each gets only what it needs)."""
    smax = sig[0]
    b_needed = np.ceil(np.log2(smax / sig)).astype(int)
    b_needed = np.maximum(b_needed, 0)
    N = len(sig)
    total_perm = int(np.sum(b_needed))
    b_uniform_full = int(np.max(b_needed))
    total_uniform = N * b_uniform_full
    return {
        "N": N, "b_needed_per_mode_minmax": [int(b_needed.min()), int(b_needed.max())],
        "total_bits_per_mode_floor_(waterfilled)": total_perm,
        "b_uniform_needed_for_full_rank": b_uniform_full,
        "total_bits_uniform_full_rank": total_uniform,
        "saving_x": float(total_uniform / total_perm) if total_perm > 0 else None,
    }


# ============================ Section 3: matched total-bit-budget -> effective-rank comparison ======
def erank_uniform_bits(sig, b_uniform):
    """Effective rank a UNIFORM bitwidth resolves: modes with sigma_i/sigma_max > eps=2^-b_uniform."""
    if b_uniform <= 0:
        return 0
    eps = 2.0 ** (-b_uniform)
    return int(np.sum(sig / sig[0] > eps))


def budget_sweep(sig, target_bits_list, R=None):
    N = len(sig)
    R = np.ones(N) if R is None else R
    alpha = (sig * R) ** 2 / 12.0
    rms_fs = float(np.sqrt(np.sum(alpha)))
    rows = []
    for tb in target_bits_list:
        tau = rms_fs * 2.0 ** (-tb)
        b_wf = alloc(alpha, tau ** 2, cost_fn("adder"), r=4.0, bmin=0, bmax=64)
        B_total = int(np.sum(b_wf))
        erank_wf = int(np.sum(b_wf > 0))
        # matched-budget uniform: same total bits B_total spread evenly over N modes
        b_uniform_matched = int(round(B_total / N)) if N > 0 else 0
        erank_uni_matched = erank_uniform_bits(sig, b_uniform_matched)
        # D's own same-TAU smallest uniform (for the complementary "fewer bits at same tau" statement)
        bu_sametau = uniform_alloc(alpha, tau ** 2, r=4.0, bmax=64)
        total_uni_sametau = N * bu_sametau
        rows.append({
            "target_bits": tb, "tau_over_rmsfs": 2.0 ** (-tb), "B_total_waterfilled": B_total,
            "erank_waterfilled": erank_wf, "b_uniform_matched_budget": b_uniform_matched,
            "erank_uniform_matched_budget": erank_uni_matched,
            "rank_gap_(wf_minus_uniform)_at_matched_bits": erank_wf - erank_uni_matched,
            "uniform_bits_for_same_tau": total_uni_sametau,
            "wf_bits_saved_vs_uniform_same_tau_pct":
                float(100 * (total_uni_sametau - B_total) / total_uni_sametau) if total_uni_sametau > 0 else None,
        })
    return rows


# ============================ main =================================================================
def main():
    os.makedirs(EV_DIR, exist_ok=True)
    ev = {"cell": "fpga_bitwidth_waterfilling_is_precision_floor_allocation",
          "idea": "D's FPGA bit-width water-filling on adjoint g_i == A's precision-floor "
                  "log2(sigma_max/sigma_i) bit-need law, for g_i = sigma_i of a real operator's SVD"}

    N, w = 32, 3.0
    A, U, sig, V = gaussian_psf_operator(N, w)
    ev["operator"] = {"N": N, "psf_width": w, "sigma_max": float(sig[0]), "sigma_min": float(sig[-1]),
                       "decay_ratio": float(sig[0] / sig[-1])}

    # KG-ADJ
    adj = check_adjoint_is_sigma2(A, V, sig, rng)
    ev["KG_ADJ_sigma_is_hessian"] = adj

    # bridge exactness
    ev["bridge"] = bridge_check(sig, rng)

    # full-rank cost (direct floor rule, large-margin case)
    ev["full_rank_cost"] = full_rank_cost(sig)

    # matched-budget sweep, real (decaying) spectrum
    tb_list = [4, 6, 8, 10, 12, 14, 16, 20, 24, 30]
    ev["budget_sweep_decaying"] = budget_sweep(sig, tb_list)

    # NULL / void-floor control: flat spectrum (degenerate, same N, unit sigma)
    sig_flat = np.ones(N)
    ev["budget_sweep_flat_NULL"] = budget_sweep(sig_flat, tb_list)

    # ---------------- verdicts ----------------
    kills = []
    if not adj["pass"]:
        kills.append("KG-ADJ: FD Hessian != sigma_i^2")
    if not ev["bridge"]["uniform_R"]["exact"]:
        kills.append("KG-BRIDGE: uniform-R residual not machine-exact")
    if not ev["bridge"]["nonuniform_R"]["full_rule_exact"]:
        kills.append("KG-BRIDGE: full (sigma+R) rule not machine-exact under non-uniform R")

    gaps = [r["rank_gap_(wf_minus_uniform)_at_matched_bits"] for r in ev["budget_sweep_decaying"]]
    if any(g < 0 for g in gaps):
        kills.append(f"KG-RANK: water-filled erank < uniform erank at matched budget somewhere (gaps={gaps})")
    max_gap = max(gaps) if gaps else 0
    max_gap_frac = max_gap / N if N else 0

    flat_gaps = [r["rank_gap_(wf_minus_uniform)_at_matched_bits"] for r in ev["budget_sweep_flat_NULL"]]
    null_ties = all(abs(g) <= 1 for g in flat_gaps)   # allow +-1 mode rounding slack (b_uniform=round())
    if not null_ties:
        kills.append(f"KG-NULL: flat-spectrum gaps not ~0 (gaps={flat_gaps}) -> effect not spectral-decay-specific")

    p1 = "CONFIRMED" if (ev["bridge"]["uniform_R"]["exact"] and ev["bridge"]["nonuniform_R"]["full_rule_exact"]
                          and ev["bridge"]["nonuniform_R"]["pure_sigma_rule_needs_correction"]) else "CHECK"
    p2 = "CONFIRMED" if ev["full_rank_cost"]["saving_x"] and ev["full_rank_cost"]["saving_x"] > 1.5 else "CHECK"
    p3 = "CONFIRMED" if (not any("KG-RANK" in k for k in kills) and max_gap >= 3) else \
         ("MARGINAL" if not any("KG-RANK" in k for k in kills) else "FAIL")
    p4 = "CONFIRMED" if null_ties else "FAIL"

    verd = {
        "kills": kills,
        "P1_bridge_exact_uniformR_needs_R_correction_nonuniform": p1,
        "P2_full_rank_saving_x": ev["full_rank_cost"]["saving_x"], "P2_verdict": p2,
        "P3_max_rank_gap_modes": max_gap, "P3_max_rank_gap_frac_of_N": max_gap_frac, "P3_verdict": p3,
        "P4_null_flat_spectrum_ties": null_ties, "P4_verdict": p4,
    }
    if kills:
        verd["headline"] = f"KILLED ({len(kills)}): {kills}"
    else:
        verd["headline"] = (
            f"CONFIRMED: D's water-filling on g_i=sigma_i IS the precision-floor bit-need law "
            f"(bridge exact for uniform R, residual {ev['bridge']['uniform_R']['max_abs_residual_vs_log2_sigma_rule']:.1e}; "
            f"needs +log2(R_i/R_j) correction under non-uniform R, {ev['bridge']['nonuniform_R']['typical_correction_bits_(log2_R_spread)']:.2f} "
            f"bits typical spread). Full-rank cert: per-mode floor uses {ev['full_rank_cost']['saving_x']:.1f}x fewer "
            f"total bits than uniform-for-worst-mode ({ev['full_rank_cost']['total_bits_per_mode_floor_(waterfilled)']} vs "
            f"{ev['full_rank_cost']['total_bits_uniform_full_rank']}). At MATCHED total-bit budget, water-filling "
            f"certifies up to {max_gap} more modes ({100*max_gap_frac:.0f}% of N) than uniform allocation "
            f"(silent rank-collapse of uniform under-provisioning). NULL control (flat spectrum): gap ties "
            f"({flat_gaps}) -- confirms the advantage is spectral-decay-specific, not a water-filling artifact.")

    ev["verdicts"] = verd
    json.dump(ev, open(EV_PATH, "w"), indent=1, default=float)

    print("=" * 100)
    print("FPGA BIT-WIDTH WATER-FILLING == PRECISION-FLOOR ALLOCATION (marginal modes need more bits)")
    print("=" * 100)
    print(f"operator: Gaussian-PSF blur N={N} w={w}  sigma_max={sig[0]:.4e} sigma_min={sig[-1]:.4e} "
          f"decay {sig[0]/sig[-1]:.2e}x")
    print(f"KG-ADJ  algebraic diag(V^T A^T A V)==sigma_i^2: pass={adj['algebraic_diag_identity']['pass']} "
          f"(max_diag_relerr={adj['algebraic_diag_identity']['max_diag_rel_err']:.2e}, "
          f"max_offdiag={adj['algebraic_diag_identity']['max_offdiag_abs']:.2e})")
    print(f"        FD (naive fixed h=1e-4, ALL modes): max_rel_err={adj['fixed_h_naive']['max_rel_err_ALL_modes']:.2e} "
          f"-- catastrophic on tail (round-off, see docstring OODA)")
    print(f"        FD restricted to FD-detectable modes ({adj['fd_restricted_to_detectable_modes']['n_detectable']}"
          f"/{adj['fd_restricted_to_detectable_modes']['n_total']}): "
          f"max_rel_err={adj['fd_restricted_to_detectable_modes']['max_rel_err']:.2e}  "
          f"pass={adj['fd_restricted_to_detectable_modes']['pass']}")
    print(f"        KG-ADJ composite pass: {adj['pass']}")
    print("-" * 100)
    br = ev["bridge"]
    print(f"BRIDGE  uniform-R exact:      residual={br['uniform_R']['max_abs_residual_vs_log2_sigma_rule']:.2e}  "
          f"exact={br['uniform_R']['exact']}")
    print(f"BRIDGE  non-uniform-R:  pure-sigma-rule residual={br['nonuniform_R']['pure_sigma_rule_max_residual']:.3f}  "
          f"full(sigma+R)-rule residual={br['nonuniform_R']['full_rule_(sigma_and_R)_max_residual']:.2e}  "
          f"full_exact={br['nonuniform_R']['full_rule_exact']}  "
          f"correction_needed={br['nonuniform_R']['pure_sigma_rule_needs_correction']}")
    print("-" * 100)
    fr = ev["full_rank_cost"]
    print(f"FULL-RANK CERT COST:  per-mode-floor bits={fr['total_bits_per_mode_floor_(waterfilled)']}  "
          f"uniform-for-full-rank bits={fr['total_bits_uniform_full_rank']} "
          f"(b_uniform={fr['b_uniform_needed_for_full_rank']} x N={fr['N']})  "
          f"SAVING={fr['saving_x']:.2f}x")
    print("-" * 100)
    print(f"{'target_bits':>11} {'B_total_wf':>10} {'erank_wf':>8} {'b_uni_matched':>13} {'erank_uni':>9} "
          f"{'gap':>4} {'wf_bits_saved_@sameTau_%':>24}")
    for r in ev["budget_sweep_decaying"]:
        print(f"{r['target_bits']:>11} {r['B_total_waterfilled']:>10} {r['erank_waterfilled']:>8} "
              f"{r['b_uniform_matched_budget']:>13} {r['erank_uniform_matched_budget']:>9} "
              f"{r['rank_gap_(wf_minus_uniform)_at_matched_bits']:>4} "
              f"{r['wf_bits_saved_vs_uniform_same_tau_pct']:>23.1f}%")
    print(f"-- max rank gap at matched budget: {max_gap} modes ({100*max_gap_frac:.0f}% of N={N}) --")
    print("-" * 100)
    print("NULL control (FLAT sigma spectrum -- should TIE, no gap):")
    print(f"{'target_bits':>11} {'B_total_wf':>10} {'erank_wf':>8} {'b_uni_matched':>13} {'erank_uni':>9} {'gap':>4}")
    for r in ev["budget_sweep_flat_NULL"]:
        print(f"{r['target_bits']:>11} {r['B_total_waterfilled']:>10} {r['erank_waterfilled']:>8} "
              f"{r['b_uniform_matched_budget']:>13} {r['erank_uniform_matched_budget']:>9} "
              f"{r['rank_gap_(wf_minus_uniform)_at_matched_bits']:>4}")
    print(f"NULL ties (all |gap|<=1): {null_ties}")
    print("-" * 100)
    print(f"P1 bridge (exact-uniformR / correction-nonuniformR): {p1}")
    print(f"P2 full-rank-cost saving: {p2}  ({fr['saving_x']:.2f}x)")
    print(f"P3 matched-budget rank gap: {p3}  (max_gap={max_gap} modes, {100*max_gap_frac:.0f}% of N)")
    print(f"P4 null flat-spectrum tie: {p4}")
    print(f"KILLS: {kills if kills else 'none'}")
    print(f"\nHEADLINE: {verd['headline']}")
    print(f"\nevidence -> {EV_PATH}")


if __name__ == "__main__":
    main()

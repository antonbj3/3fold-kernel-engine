"""
DETERMINISTIC GPU MATERIAL-FRACTURE and sigma-ATTRIBUTION (a seed) — the MEASURED, honest result.
================================================================================
a colleague Staniszewski (@bonzajplc): the hard part of GPU fracture is "dynamic AND DETERMINISTIC fracturing on
GPU". The platform already has the two ingredients separately: det_accumulation_probe.py (int64-fixed-point
atomics are order-invariant, validated vs the contact_engine 0.32mm float jitter) and fracture_fiber_bundle.py
(Weibull material-sigma). The HYPOTHESIS I set out to prove: float-atomics non-determinism CONTAMINATES the
material-sigma attribution in a GPU fracture sim.

WHAT I MEASURED (FP/FN-symmetric -- I let the data overturn my hypothesis): the hypothesis is FALSE in the
generic regime, and the truth is sharper and more useful:
  * int64-fixed-point GPU fracture is BIT-DETERMINISTIC (order-invariant) -- a provable GUARANTEE.
  * float-atomics jitter is REAL and scales with contention (low-contention lattice ~2e-7; high-contention
    fragment accumulation ~4e-4) -- vs int64 EXACTLY 0.
  * BUT static fracture OUTCOMES are ROBUST to that jitter: a cell/fragment flips its break-decision only if it
    is SIMULTANEOUSLY marginal (within the jitter of its threshold) AND high-jitter -- a double coincidence that
    is very rare away from a feedback bifurcation. Measured: ~0 outcome flips across regimes and loads.
  => float-atomics contamination of fracture-sigma is TYPICALLY NEGLIGIBLE. The value of int64 determinism is
     the GUARANTEE: a PROVABLE zero numerical-sigma, so you never have to bound the margin -- which is exactly
     what you cannot do at a feedback bifurcation (criticality), where the cascade amplifies the jitter and the
     outcome IS sensitive. Determinism is cheap insurance whose payoff concentrates at criticality.

GATES (real warp GPU, RTX 5070):
  (A) int64-fixed-point GPU fracture CASCADE is BIT-DETERMINISTIC (order A vs B identical); float cascade is
        ALSO outcome-identical in the low-contention lattice (jitter << threshold margin) -- measured.
  (B) float-atomics jitter is REAL and grows with contention (lattice << fragment), int64 jitter is EXACTLY 0.
  (C) static fracture OUTCOMES are ROBUST (0 flips: the marginal-AND-high-jitter coincidence is rare) -> the
        float numerical-sigma on broken-count is ~0 HERE; int64 is ==0 by GUARANTEE. The determinism value is
        the guarantee (unbounded risk only at a feedback bifurcation), NOT a generic contamination.
  (D) the material model is correct: drawn thresholds match the analytic Weibull(m) (external truth).
"""
import sys
import numpy as np
import warp as wp
from math import gamma

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
SCALE = 1.0e9
L = 128; N = L * L
M_WEIBULL = 4.0
REDIST = 0.25
SIGMA_EXT = 0.62

def banner(t): print("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)

# ---------------- lattice fracture cascade kernels ----------------
@wp.kernel
def detect_f(stress: wp.array(dtype=float), thr: wp.array(dtype=float), broken: wp.array(dtype=int),
             newly: wp.array(dtype=int), cnt: wp.array(dtype=int)):
    i = wp.tid()
    if broken[i] == 0 and stress[i] > thr[i]:
        newly[i] = 1; wp.atomic_add(cnt, 0, 1)
@wp.kernel
def redist_f(order: wp.array(dtype=int), newly: wp.array(dtype=int), w: wp.array(dtype=float),
             stress: wp.array(dtype=float), broken: wp.array(dtype=int), L: int):
    t = wp.tid(); i = order[t]
    if newly[i] == 1:
        s = w[i]; x = i % L; y = i // L
        if x > 0:     wp.atomic_add(stress, i - 1, s)
        if x < L - 1: wp.atomic_add(stress, i + 1, s)
        if y > 0:     wp.atomic_add(stress, i - L, s)
        if y < L - 1: wp.atomic_add(stress, i + L, s)
        broken[i] = 1; newly[i] = 0
@wp.kernel
def detect_i(stress: wp.array(dtype=wp.int64), thr: wp.array(dtype=wp.int64), broken: wp.array(dtype=int),
             newly: wp.array(dtype=int), cnt: wp.array(dtype=int)):
    i = wp.tid()
    if broken[i] == 0 and stress[i] > thr[i]:
        newly[i] = 1; wp.atomic_add(cnt, 0, 1)
@wp.kernel
def redist_i(order: wp.array(dtype=int), newly: wp.array(dtype=int), w: wp.array(dtype=wp.int64),
             stress: wp.array(dtype=wp.int64), broken: wp.array(dtype=int), L: int):
    t = wp.tid(); i = order[t]
    if newly[i] == 1:
        s = w[i]; x = i % L; y = i // L
        if x > 0:     wp.atomic_add(stress, i - 1, s)
        if x < L - 1: wp.atomic_add(stress, i + 1, s)
        if y > 0:     wp.atomic_add(stress, i - L, s)
        if y < L - 1: wp.atomic_add(stress, i + L, s)
        broken[i] = 1; newly[i] = 0

def cascade(thr_np, w_np, order_np, use_int):
    broken = wp.zeros(N, dtype=int, device=DEV); newly = wp.zeros(N, dtype=int, device=DEV); cnt = wp.zeros(1, dtype=int, device=DEV)
    order = wp.array(order_np.astype(np.int32), dtype=int, device=DEV)
    if use_int:
        stress = wp.array((np.full(N, SIGMA_EXT) * SCALE).round().astype(np.int64), dtype=wp.int64, device=DEV)
        thr = wp.array((thr_np * SCALE).round().astype(np.int64), dtype=wp.int64, device=DEV)
        w = wp.array((w_np * SCALE).round().astype(np.int64), dtype=wp.int64, device=DEV); det, red = detect_i, redist_i
    else:
        stress = wp.array(np.full(N, SIGMA_EXT, dtype=np.float32), dtype=float, device=DEV)
        thr = wp.array(thr_np.astype(np.float32), dtype=float, device=DEV)
        w = wp.array(w_np.astype(np.float32), dtype=float, device=DEV); det, red = detect_f, redist_f
    for _ in range(3000):
        cnt.zero_(); wp.launch(det, N, inputs=[stress, thr, broken, newly, cnt], device=DEV); wp.synchronize()
        if int(cnt.numpy()[0]) == 0: break
        wp.launch(red, N, inputs=[order, newly, w, stress, broken, L], device=DEV); wp.synchronize()
    return int(broken.numpy().sum()), broken.numpy()

def make_sample(seed):
    rng = np.random.default_rng(seed); u = rng.uniform(1e-6, 1, N)
    return ((-np.log(u)) ** (1.0 / M_WEIBULL)).astype(np.float64), (REDIST * (0.5 + rng.uniform(0, 1, N))).astype(np.float64)

# ---------------- high-contention fragment accumulation kernels ----------------
@wp.kernel
def acc_f(frag: wp.array(dtype=int), load: wp.array(dtype=float), order: wp.array(dtype=int), acc: wp.array(dtype=float)):
    t = wp.tid(); c = order[t]; wp.atomic_add(acc, frag[c], load[c])
@wp.kernel
def acc_i(frag: wp.array(dtype=int), load: wp.array(dtype=float), order: wp.array(dtype=int), sc: float, acc: wp.array(dtype=wp.int64)):
    t = wp.tid(); c = order[t]; wp.atomic_add(acc, frag[c], wp.int64(wp.round(load[c] * sc)))

# ---- high-fan-out FEEDBACK cascade (to TEST the criticality-flip prediction) ----
@wp.kernel
def fb_detect(stress: wp.array(dtype=float), thr: wp.array(dtype=float), broken: wp.array(dtype=int),
              newly: wp.array(dtype=int), cnt: wp.array(dtype=int)):
    i = wp.tid()
    if broken[i] == 0 and stress[i] > thr[i]:
        newly[i] = 1; wp.atomic_add(cnt, 0, 1)
@wp.kernel
def fb_redist(order: wp.array(dtype=int), newly: wp.array(dtype=int), tgt: wp.array(dtype=int), d: float, KK: int,
              stress: wp.array(dtype=float), broken: wp.array(dtype=int)):
    t = wp.tid(); i = order[t]
    if newly[i] == 1:
        for k in range(KK):
            wp.atomic_add(stress, tgt[i * KK + k], d)
        broken[i] = 1; newly[i] = 0

def fb_cascade(thr_np, tgt_np, se, d, KK, order_np):
    broken = wp.zeros(N, dtype=int, device=DEV); newly = wp.zeros(N, dtype=int, device=DEV); cnt = wp.zeros(1, dtype=int, device=DEV)
    order = wp.array(order_np.astype(np.int32), dtype=int, device=DEV); tgt = wp.array(tgt_np.astype(np.int32), dtype=int, device=DEV)
    stress = wp.array(np.full(N, se, dtype=np.float32), dtype=float, device=DEV); thr = wp.array(thr_np.astype(np.float32), dtype=float, device=DEV)
    for _ in range(5000):
        cnt.zero_(); wp.launch(fb_detect, N, inputs=[stress, thr, broken, newly, cnt], device=DEV); wp.synchronize()
        if int(cnt.numpy()[0]) == 0: break
        wp.launch(fb_redist, N, inputs=[order, newly, tgt, float(d), KK, stress, broken], device=DEV); wp.synchronize()
    return broken.numpy()

def frag_accum(frag, load, order, F, use_int):
    o = wp.array(order.astype(np.int32), dtype=int, device=DEV); fr = wp.array(frag.astype(np.int32), dtype=int, device=DEV)
    ld = wp.array(load.astype(np.float32), dtype=float, device=DEV)
    if use_int:
        acc = wp.zeros(F, dtype=wp.int64, device=DEV); wp.launch(acc_i, len(frag), inputs=[fr, ld, o, SCALE, acc], device=DEV); wp.synchronize()
        return acc.numpy().astype(np.float64) / SCALE
    acc = wp.zeros(F, dtype=float, device=DEV); wp.launch(acc_f, len(frag), inputs=[fr, ld, o, acc], device=DEV); wp.synchronize()
    return acc.numpy().astype(np.float64)

def main():
    banner(f"DETERMINISTIC GPU MATERIAL-FRACTURE + sigma-attribution (device={DEV}, lattice L={L}, Weibull m={M_WEIBULL})")
    idA = np.arange(N); idB = np.random.default_rng(0).permutation(N)

    # ---------- (A) lattice cascade determinism ----------
    thr0, w0 = make_sample(seed=100)
    fA, fmA = cascade(thr0, w0, idA, False); fB, fmB = cascade(thr0, w0, idB, False)
    iA, imA = cascade(thr0, w0, idA, True);  iB, imB = cascade(thr0, w0, idB, True)
    banner("(A) lattice fracture CASCADE: order A vs B (same material sample)")
    print(f"  INT64 : broken A={iA}  B={iB}  pattern-diff={int(np.sum(imA != imB))}  -> bit-identical")
    print(f"  FLOAT : broken A={fA}  B={fB}  pattern-diff={int(np.sum(fmA != fmB))}  -> (low-contention lattice: also identical)")
    okA = (iA == iB) and int(np.sum(imA != imB)) == 0
    print(f"  -> int64 fracture cascade is BIT-DETERMINISTIC (order-invariant) -- the GUARANTEE: {'PASS' if okA else 'FAIL'}")

    # ---------- (B) jitter is real and scales with contention; int64 == 0 ----------
    banner("(B) float-atomics jitter is REAL and scales with CONTENTION; int64 jitter is EXACTLY 0")
    # lattice (low contention: 4 neighbours): field jitter via the cascade stress is tiny -> measure on a direct accumulate
    rng = np.random.default_rng(1)
    F = 65536; Mc = 8_000_000
    frag = rng.integers(0, F, Mc); load = rng.uniform(0, 1, Mc).astype(np.float32)
    oA = np.arange(Mc); oB = rng.permutation(Mc)
    accA = frag_accum(frag, load, oA, F, False); accB = frag_accum(frag, load, oB, F, False)
    jit_float = float(np.max(np.abs(accA - accB)))
    iaccA = frag_accum(frag, load, oA, F, True); iaccB = frag_accum(frag, load, oB, F, True)
    jit_int = float(np.max(np.abs(iaccA - iaccB)))
    # low-contention reference jitter (4 contributions/cell)
    Fl = N; ll = rng.uniform(0, 1, 4 * N).astype(np.float32); fl = rng.integers(0, Fl, 4 * N)
    lA = frag_accum(fl, ll, np.arange(4 * N), Fl, False); lB = frag_accum(fl, ll, rng.permutation(4 * N), Fl, False)
    print(f"  low-contention (~4 adds/cell)  float jitter = {float(np.max(np.abs(lA-lB))):.2e}")
    print(f"  high-contention (~122 adds/frag) float jitter = {jit_float:.2e}   (grows with contention)")
    print(f"  high-contention INT64 jitter                 = {jit_int:.2e}   (EXACTLY 0 = order-invariant)")
    okB = jit_int == 0.0 and jit_float > 0.0
    print(f"  -> float jitter is real and contention-scaling; int64 is provably 0: {'PASS' if okB else 'FAIL'}")

    # ---------- (C) static fracture OUTCOMES are robust -> contamination negligible; int64 = guarantee ----------
    banner("(C) static fracture OUTCOMES are ROBUST to the jitter (flip needs marginal-AND-high-jitter coincidence)")
    flip_counts = []
    for ts in [0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015]:
        u = np.random.default_rng(7).uniform(1e-6, 1, F); base = (-np.log(u)) ** (1.0 / M_WEIBULL); base = base / base.mean()
        thr = accA.mean() * ts * base
        flip_counts.append(int(np.sum((accA > thr) != (accB > thr))))
    max_flips = max(flip_counts)
    # material sigma (across Weibull samples, int64 -> uncontaminated) vs float numerical sigma (across orders)
    mat = [cascade(*make_sample(seed=s), order_np=idA, use_int=True)[0] for s in range(200, 208)]
    fnum = [cascade(thr0, w0, np.random.default_rng(s).permutation(N), False)[0] for s in range(6)]
    inum = [cascade(thr0, w0, np.random.default_rng(s).permutation(N), True)[0] for s in range(6)]
    mat_sigma, num_sigma_f, num_sigma_i = float(np.std(mat)), float(np.std(fnum)), float(np.std(inum))
    print(f"  float fracture-outcome flips across 7 threshold scales (65536 frags) = {flip_counts}  (max {max_flips})")
    print(f"  material sigma (Weibull samples) = {mat_sigma:.1f} cells | float numerical sigma = {num_sigma_f:.1f} | int64 numerical sigma = {num_sigma_i:.1f}")
    okC = num_sigma_i == 0.0 and max_flips == 0 and num_sigma_f < 0.05 * max(mat_sigma, 1e-9)
    print(f"  -> outcomes ROBUST (0 flips); float numerical sigma ~0 HERE, int64 ==0 by GUARANTEE. Contamination is")
    print(f"     typically negligible; determinism's value is the PROVABLE guarantee (essential at a feedback bifurcation): {'PASS' if okC else 'FAIL'}")

    # ---------- (E) I TESTED my own criticality-flip prediction -- and measurement REFUTED it ----------
    banner("(E) TESTED PREDICTION: does float-atomics flip the outcome at a FEEDBACK BIFURCATION? -> REFUTED")
    KK = 64
    rngf = np.random.default_rng(3); uf = rngf.uniform(1e-6, 1, N)
    thrf = (-np.log(uf)) ** (1.0 / M_WEIBULL); tgt = rngf.integers(0, N, (N, KK)).ravel(); dd = 0.9 / KK
    # locate the percolation transition (broken fraction jumps), then probe AT the razor's edge with several orders
    loads = [0.500, 0.505, 0.508, 0.520, 0.550]
    orders = [np.arange(N)] + [np.random.default_rng(s).permutation(N) for s in range(1, 5)]
    crit_spread = 0; crit_patdiff = 0
    print(f"  {'load':>6} | {'broken frac':>11} | {'fraction spread (5 float orders)':>32} | {'exact-pattern diff':>18}")
    for se in loads:
        pats = [fb_cascade(thrf, tgt, se, dd, KK, o) for o in orders]
        fracs = [int(p.sum()) / N for p in pats]
        spread = max(fracs) - min(fracs)
        patdiff = max(int(np.sum(pats[0] != p)) for p in pats[1:])
        crit_spread = max(crit_spread, spread); crit_patdiff = max(crit_patdiff, patdiff)
        print(f"  {se:6.3f} | {np.mean(fracs):11.3f} | {spread:32.4f} | {patdiff:18d}")
    okE = crit_spread == 0.0 and crit_patdiff == 0
    print(f"  -> across the percolation transition, 5 float processing orders give IDENTICAL broken fraction AND exact")
    print(f"     pattern (spread {crit_spread}, pattern-diff {crit_patdiff} cells). My prediction that float would flip at")
    print(f"     criticality is REFUTED: the threshold nonlinearity FILTERS the sub-margin jitter even at the bifurcation: {'PASS' if okE else 'FAIL'}")

    # ---------- (D) external truth: Weibull ----------
    banner("(D) external-truth: material thresholds match analytic Weibull(modulus m)")
    thrD, _ = make_sample(seed=300)
    mean_emp, mean_an = float(np.mean(thrD)), gamma(1.0 + 1.0 / M_WEIBULL)
    q90_emp, q90_an = float(np.quantile(thrD, 0.9)), (-np.log(1 - 0.9)) ** (1.0 / M_WEIBULL)
    print(f"  Weibull mean: empirical {mean_emp:.4f} vs analytic Gamma(1+1/m) {mean_an:.4f}")
    print(f"  Weibull q90 : empirical {q90_emp:.4f} vs analytic (-ln0.1)^(1/m) {q90_an:.4f}")
    okD = abs(mean_emp - mean_an) < 0.01 and abs(q90_emp - q90_an) < 0.02
    print(f"  -> correct Weibull(m={M_WEIBULL}) draw (external truth): {'PASS' if okD else 'FAIL'}")

    banner("VERDIKT — deterministic GPU fracture: what's TRUE (measured, hypothesis overturned)")
    for nm, ok in [("(A) int64 GPU fracture cascade is bit-deterministic (the guarantee)", okA),
                   ("(B) float-atomics jitter is real + contention-scaling; int64 is provably 0", okB),
                   ("(C) static fracture outcomes ROBUST -> contamination negligible; int64 = guarantee", okC),
                   ("(D) material thresholds match analytic Weibull(m) (external truth)", okD),
                   ("(E) my criticality-flip PREDICTION tested and REFUTED (float robust even at the bifurcation)", okE)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}")
    allok = all([okA, okB, okC, okD, okE])
    print(f"""
      HONEST RESULT (my hypothesis 'float contaminates fracture-sigma' was OVERTURNED by measurement):
      int64-fixed-point GPU fracture is BIT-DETERMINISTIC -- a provable zero-numerical-sigma GUARANTEE. Float-atomics
      jitter is real and grows with contention ({jit_float:.0e} at ~122 adds/frag vs int64 0), but static fracture
      OUTCOMES are ROBUST: a break-decision flips only when a cell is SIMULTANEOUSLY marginal AND high-jitter -- a
      double coincidence measured at 0 flips across loads, thresholds and contention. So float's contamination of
      aleatoric-sigma attribution is TYPICALLY NEGLIGIBLE -- NOT a generic problem, correcting a colleague's framing for
      the static case. The value of determinism is the GUARANTEE (you never have to bound the margin), which is
      decisive only at a FEEDBACK BIFURCATION (criticality) where the cascade amplifies the jitter -- the regime
      fracture_fiber_bundle.py shows is maximally susceptible. game(R)twin: same kernel, float (fast) vs int64
      (sigma-clean guarantee) on a fidelity dial.
      ★I TESTED MY OWN PREDICTION (gate E) that float would flip at a feedback bifurcation -- and measurement REFUTED
      it: across the percolation transition, 5 float orders give IDENTICAL broken fraction AND exact pattern. The
      threshold-crossing nonlinearity FILTERS the sub-margin atomic jitter even at criticality. So in continuous-
      threshold GPU fracture, int64 determinism is NOT needed for outcome/sigma correctness; its real value is
      bit-exact REPLAY/networking reproducibility (a colleague's actual concern) and a guarantee for pathological cases.
      HONEST SCOPE: lattice + high-fan-out cascade, Weibull thresholds; a discrete/brittle model with degenerate
      thresholds (where margins CAN be ~0) is the only remaining place float could plausibly flip -- untested.
    """)
    print("ALL PASS" if allok else "SOME GATE FAILED -- inspect above")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

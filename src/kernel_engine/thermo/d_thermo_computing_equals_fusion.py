#!/usr/bin/env python3
r"""d_thermo_computing_equals_fusion.py -- DOES THERMODYNAMIC-COMPUTING EQUILIBRIUM SAMPLING
                                           == THE MASQUERADE (x) COMPOSITION FUSION, EXACTLY?

(COLLECTION-MODE: external-hardware grounding of the fusion.  No commits.
 CPU-only, self-contained, numpy + scipy only.  python = python3)

============================== THE EXTERNAL FACT (real hardware) ==============================
SEED_INBOX  (EXTERNALLY CONFIRMED): Normal Computing's "Thermodynamic Linear Algebra"
(arXiv 2308.05660) + "Thermodynamic Computing System for AI Applications" (arXiv 2312.04836,
Nature Comms 2025 s41467-025-59011-x).  A Stochastic Processing Unit (RLC / coupled-harmonic-
oscillator unit cells on a PCB) solves LINEAR SYSTEMS / MATRIX-INVERSE / DETERMINANTS / LYAPUNOV
by SAMPLING the thermal-equilibrium (Gaussian) distribution of  dx = -(A x - b) dt + sqrt(2 kT) dW.
It computes A^-1 / Gaussian samples via thermal fluctuations = FDT.  FORCING (SEED line 31):
"does thermodynamic-computing's thermal-equilibrium-sampling == our signed-weighted-walk-sum
resolvent EXACTLY (same object, real hardware)?"

============================== THE FUSION OBJECT (verified prior, this repo) ==============================
COMPOSITION = the resolvent  J^-1 = (I - kappa W)^-1 = sum_{k>=0} (kappa W)^k = SIGNED WEIGHTED WALK-SUM.
MASQUERADE  = a mode with ZERO effective-conductance-to-anchor = vanishing walk-sum to the anchor.
   class-A (topological): anchor in a different connected component -> no walk exists.
   class-B (algebraic weight-null / forward-null): anchor-REACHABLE, but the SIGNED walk-sum
           destructively CANCELS to zero (pure topology cannot see this; the weighted walk-sum can).

============================== THE PHYSICS (derived, not asserted) ==============================
Overdamped Langevin / coupled oscillators:  xdot = -A x + xi,  <xi(t) xi(t')^T> = 2 D delta(t-t').
Stationary covariance C = <x x^T> solves the LYAPUNOV equation   A C + C A^T = 2 D.
FDT / detailed balance: to sample Boltzmann exp(-x^T A x / 2kT) (precision A), the noise is ISOTROPIC
   D = kT * I   ==>   C = kT * A^-1   for SYMMETRIC A.   [proof: A(kT A^-1)+(kT A^-1)A = 2kT I = 2D.]
   ** SYMMETRIC-QC / instrument-vs-truth: the task-card phrasing "D = kT*A gives C = kT*A^-1" is a
      MISSTATEMENT -- D=kT*A yields C=kT*I (trivial).  The FDT relation that yields the INVERSE is the
      isotropic D=kT*I (exactly Normal Computing's sqrt(2kT) dW).  We IMPLEMENT the correct physics and
      EXPLICITLY test D=kT*A to show it does NOT give the inverse. **
So with A := (I - kappa W) = precision = J:   C / kT  ==  (I - kappa W)^-1  ==  J^-1  ==  the resolvent
==  the signed weighted walk-sum.  The thermal bath's equilibrium covariance IS the composition operator.
And  C_ij / kT = (A^-1)_ij = walk-sum(i->j); MASQUERADE (zero walk-sum to anchor) <=> zero equilibrium
cross-covariance to the sensor = a DARK MODE (bath-excited, sensor-invisible).  Both fusion halves via FDT.

============================== PRE-REGISTERED VERDICTS (frozen BEFORE the run) ==============================
PR1  EXACT identity (Lyapunov):  || C/kT - (I-kappa W)^-1 ||_max < 1e-10  for symmetric SPD A, D=kT*I.
     Triple: C/kT == direct resolvent solve == truncated signed walk-sum sum_k (kappa W)^k.  kT factor real
     (holds for kT != 1; C scales linearly in kT).
PR2  SAMPLED identity (finite-N thermo readout):  a physical Euler-Maruyama integration of the SDE reaches
     C=kT A^-1 (small relative error); the covariance-estimator error scales as ~ 1/sqrt(N) (log-log slope ~ -0.5).
PR3  MASQUERADE = DARK MODE (reads out naturally):
     (A) topological: sensor & masqueraded node in DIFFERENT components -> C_{s,m} = 0 EXACTLY, while the
         masqueraded node is bath-excited (C_{m,m} > 0).  Adding a bridge edge LIFTS it (C_{s,m} != 0).
     (B) algebraic (signed-walk cancellation, connected graph): tune one SIGNED edge so walk-sum(s->m)=0
         -> C_{s,m}=0 EXACTLY though a path exists; and a bath-excited eigenmode v with sensor c _|_ v has
         cross-cov c^T C v = 0 EXACTLY (dark mode) while Var(v^T x)=kT/a_v>0.  Thermo cross-cov = FULL signed
         walk-sum, so it detects BOTH classes -> STRICTLY FINER than the topological components object.
PR4  SYMMETRIC-vs-NONSYMMETRIC BOUNDARY (honest edge):  the identity is EXACT only for symmetric A
     (detailed balance).  Non-symmetric A (non-reciprocal edges, broken detailed balance) -> C != kT A^-1;
     rel-err grows from 0 with the skew magnitude gamma.  Clean proof: C is ALWAYS symmetric but A^-1 is NOT
     for non-symmetric A -> they cannot be equal.  D=kT*A control -> C=kT*I (not the inverse).
PR5  CONFIDENCE: HIGH that thermo-computing hardware computes EXACTLY our fusion object (resolvent =
     composition) and physically reads out masquerade (dark modes), for the symmetric/equilibrium regime.

============================== RESULTS (measured; seed=20260708) ==============================
[R1] EXACT identity  : PASS   max|C/kT - resolvent J^-1| = 2.37e-15 ; triple(resolvent,walk-sum,C/kT) all < 3.2e-15 ;
                       kT-linearity across kT in {0.3,1.0,1.7,5.0} max-dev = 2.44e-15.  C/kT == J^-1 to machine eps.
[R2] SAMPLED identity: PASS   EM-SDE physical run rel-err = 0.0118 (160k eff samples) ; iid log-log slope = -0.494
                       (pre-reg -0.5).  Finite-N thermal readout converges to the resolvent as 1/sqrt(N).
[R3] MASQUERADE dark  : PASS   (A) topological C_{s,m} = 0.0 EXACT (max cross-block 0.0) while C_{m,m}=1.880 bath-
                       excited; bridge LIFTS to |C_{s,m}|=0.635.  (B) algebraic signed-walk cancel: tuned edge
                       w*=-0.336, |C_{0,2}|=6.7e-16 in a CONNECTED SPD graph (path exists, walk-sum cancels);
                       eigenmode dark cross-cov |c^T C v|=1.11e-16 while Var(v^T x)=2.103=kT/a_v>0.  Thermo cross-cov
                       reads out BOTH classes (finer than the topological components object).
[R4] BOUNDARY         : PASS   symmetric rel-err=2.25e-15 ; non-symmetric(gamma=0.6) rel-err=0.299 (!=0), grows
                       0->0.078->0.155->0.229->0.299 monotonically ; proof: ||C - C^T||=7.6e-16 (C symmetric) but
                       ||A^-1-(A^-1)^T||=1.761 (A^-1 NOT symmetric) -> cannot be equal ; D=kT*A control -> C=kT*I
                       (rel-err vs A^-1 = 0.375, vs kT*I = 2.2e-15).  Detailed-balance / reciprocity boundary confirmed.
[R5] CONFIDENCE       : HIGH.  Thermodynamic-computing equilibrium sampling == the masquerade(x)composition fusion,
                       EXACTLY (Lyapunov, machine eps) and to 1/sqrt(N) (physical sampling).  Real hardware
                       (Normal Computing SPU) computes the resolvent=composition and physically detects
                       masquerade as dark modes.  Honest edges: exact only for symmetric A (detailed balance);
                       correct FDT noise is isotropic D=kT*I (task-card "D=kT*A" corrected).

VERDICT: the thermodynamic-computing SUBSTRATE PHYSICALLY REALIZES BOTH HALVES OF THE FUSION via FDT --
composition = the equilibrium covariance (resolvent), masquerade = a dark mode (zero sensor cross-cov).
Evidence -> artifacts/d_thermo_computing_equals_fusion_evidence.json
"""
import os
import json
import time
import numpy as np
from scipy import linalg

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)


def LOG(*a):
    print(*a, flush=True)


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    return o


# ----------------------------------------------------------------------------------------------------
# graph / operator builders
# ----------------------------------------------------------------------------------------------------
def signed_sym_coupling(n, rng, density=0.6, rho_target=1.0):
    """Random SIGNED SYMMETRIC coupling W (zero diagonal), spectral radius normalized to rho_target."""
    M = rng.standard_normal((n, n))
    M = 0.5 * (M + M.T)
    up = np.triu(rng.random((n, n)) < density, 1)
    mask = up | up.T
    W = M * mask
    np.fill_diagonal(W, 0.0)
    ev = np.linalg.eigvalsh(W)
    rho = float(np.max(np.abs(ev)))
    if rho > 0:
        W = W * (rho_target / rho)
    return W


def is_connected(W):
    n = W.shape[0]
    adj = np.abs(W) > 1e-14
    seen = np.zeros(n, bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u, v] and not seen[v]:
                seen[v] = True
                stack.append(v)
    return bool(seen.all())


def connected_signed_coupling(n, rng, density=0.6, rho_target=1.0, tries=200):
    for _ in range(tries):
        W = signed_sym_coupling(n, rng, density, rho_target)
        if is_connected(W):
            return W
    raise RuntimeError("could not build connected coupling")


def lyapunov_cov(A, D):
    """Stationary covariance C: A C + C A^T = 2 D  (scipy solves A X + X A^H = Q)."""
    return linalg.solve_continuous_lyapunov(A, 2.0 * D)


def walk_sum(kappaW, K):
    """Truncated signed weighted walk-sum sum_{k=0}^{K} (kappa W)^k -> resolvent."""
    n = kappaW.shape[0]
    S = np.eye(n)
    term = np.eye(n)
    for _ in range(K):
        term = term @ kappaW
        S = S + term
    return S


rng = np.random.default_rng(20260708)
EV = {"script": "d_thermo_computing_equals_fusion.py", "date": "", "lane": "D",
      "external_fact": "Normal Computing SPU: arXiv 2308.05660 (Thermodynamic Linear Algebra) + "
                       "arXiv 2312.04836 (Thermodynamic Computing System for AI, Nat.Comms 2025) -- "
                       "matrix-inverse/Lyapunov by sampling thermal equilibrium of coupled oscillators."}

# ====================================================================================================
# PR1 -- EXACT identity: C/kT == (I - kappa W)^-1 == signed walk-sum  (symmetric A, D = kT*I)
# ====================================================================================================
LOG("=" * 96)
LOG("PR1  EXACT IDENTITY  (thermo equilibrium covariance == resolvent == signed walk-sum)")
LOG("=" * 96)
n = 8
kappa = 0.6
W = connected_signed_coupling(n, rng, density=0.6, rho_target=1.0)   # rho(W)=1 -> rho(kappa W)=0.6 < 1
kappaW = kappa * W
A = np.eye(n) - kappaW                                               # precision J = I - kappa W
assert np.allclose(A, A.T), "A must be symmetric here"
eigA = np.linalg.eigvalsh(A)
spd = bool(eigA.min() > 0)
LOG(f"  n={n}  kappa={kappa}  rho(kappa W)={kappa:.3f}  A=I-kappaW SPD={spd}  eig(A) in [{eigA.min():.3f},{eigA.max():.3f}]")

resolvent = np.linalg.inv(A)                                         # J^-1 = composition operator
ksum = walk_sum(kappaW, K=80)                                        # signed weighted walk-sum
err_walk_vs_resolvent = float(np.max(np.abs(ksum - resolvent)))

kT = 1.7                                                             # non-trivial temperature (!=1)
D = kT * np.eye(n)                                                   # ISOTROPIC FDT noise (correct relation)
C = lyapunov_cov(A, D)                                               # thermodynamic equilibrium covariance
thermo_inv = C / kT                                                  # what the SPU "reads out" as A^-1
err_C_vs_resolvent = float(np.max(np.abs(thermo_inv - resolvent)))
err_C_vs_walk = float(np.max(np.abs(thermo_inv - ksum)))

# kT-linearity: C(kT)/kT must be identical (== A^-1) for every kT
kT_dev = 0.0
for kt_val in [0.3, 1.0, 1.7, 5.0]:
    Ck = lyapunov_cov(A, kt_val * np.eye(n))
    kT_dev = max(kT_dev, float(np.max(np.abs(Ck / kt_val - resolvent))))

pr1_pass = (err_C_vs_resolvent < 1e-10) and (err_walk_vs_resolvent < 1e-8) and (kT_dev < 1e-10) and spd
LOG(f"  max|C/kT - resolvent J^-1|         = {err_C_vs_resolvent:.2e}")
LOG(f"  max|walk-sum   - resolvent J^-1|   = {err_walk_vs_resolvent:.2e}")
LOG(f"  max|C/kT - walk-sum|               = {err_C_vs_walk:.2e}")
LOG(f"  kT-linearity max-dev (kT in .3..5) = {kT_dev:.2e}")
LOG(f"  --> PR1 {'PASS' if pr1_pass else 'FAIL'}: thermo covariance / kT == resolvent == walk-sum (machine eps)")
EV["PR1_exact_identity"] = {
    "n": n, "kappa": kappa, "kT": kT, "A_is_SPD": spd, "eigA_min": float(eigA.min()), "eigA_max": float(eigA.max()),
    "max_abs_C_over_kT_minus_resolvent": err_C_vs_resolvent,
    "max_abs_walksum_minus_resolvent": err_walk_vs_resolvent,
    "max_abs_C_over_kT_minus_walksum": err_C_vs_walk,
    "kT_linearity_max_dev": kT_dev, "pass": pr1_pass,
    "verdict": "thermo equilibrium covariance C/kT == (I-kappa W)^-1 == signed weighted walk-sum, EXACTLY (Lyapunov)"}

# ====================================================================================================
# PR2 -- SAMPLED identity: physical SDE reaches C=kT A^-1 ; estimator error ~ 1/sqrt(N)
# ====================================================================================================
LOG("\n" + "=" * 96)
LOG("PR2  SAMPLED IDENTITY  (finite-N thermal readout -> resolvent, 1/sqrt(N))")
LOG("=" * 96)

# (2a) PHYSICAL Euler-Maruyama integration of xdot=-Ax+xi (batch of independent thermal walkers)
dt = 0.02
sig = np.sqrt(2.0 * kT * dt)
n_walk = 4000
burn = 2500
snaps = 40
gap = 250
srng = np.random.default_rng(7)
X = srng.standard_normal((n_walk, n)) * np.sqrt(kT)
for _ in range(burn):                                               # relax to equilibrium
    X = X - (X @ A.T) * dt + sig * srng.standard_normal((n_walk, n))
acc = np.zeros((n, n))
cnt = 0
for _ in range(snaps):
    for _ in range(gap):                                            # decorrelate between snapshots
        X = X - (X @ A.T) * dt + sig * srng.standard_normal((n_walk, n))
    acc += X.T @ X
    cnt += n_walk
C_em = acc / cnt
em_relerr = float(np.linalg.norm(C_em / kT - resolvent) / np.linalg.norm(resolvent))
LOG(f"  EM-SDE physical run: walkers={n_walk} snaps={snaps} (~{cnt} eff samples)  rel-err(C/kT vs J^-1) = {em_relerr:.4f}")

# (2b) 1/sqrt(N) scaling of the covariance estimator (iid equilibrium samples ~ N(0, C))
L = np.linalg.cholesky(C)
Ns = [200, 1000, 5000, 20000, 100000, 500000]
reps = 8
mean_err = []
grng = np.random.default_rng(99)
for N in Ns:
    es = []
    for _ in range(reps):
        Z = grng.standard_normal((N, n)) @ L.T                      # iid equilibrium draws
        Ch = (Z.T @ Z) / N
        es.append(np.linalg.norm(Ch / kT - resolvent) / np.linalg.norm(resolvent))
    mean_err.append(float(np.mean(es)))
slope = float(np.polyfit(np.log(Ns), np.log(mean_err), 1)[0])       # expect ~ -0.5
LOG(f"  iid estimator rel-err vs N: " + "  ".join(f"N={N}:{e:.4f}" for N, e in zip(Ns, mean_err)))
LOG(f"  log-log slope = {slope:.3f}  (pre-registered -0.5)")
pr2_pass = (em_relerr < 0.03) and (abs(slope + 0.5) < 0.08)
LOG(f"  --> PR2 {'PASS' if pr2_pass else 'FAIL'}: physical thermal sampling converges to the resolvent at 1/sqrt(N)")
EV["PR2_sampled_identity"] = {
    "em_sde": {"dt": dt, "n_walkers": n_walk, "snaps": snaps, "eff_samples": cnt, "rel_err_vs_resolvent": em_relerr},
    "iid_scaling": {"N": Ns, "mean_rel_err": mean_err, "loglog_slope": slope}, "pass": pr2_pass,
    "verdict": "physical Euler-Maruyama equilibrium sampling reaches C=kT A^-1; estimator error ~ 1/sqrt(N)"}

# ====================================================================================================
# PR3 -- MASQUERADE = DARK MODE (zero equilibrium cross-covariance to the sensor)
# ====================================================================================================
LOG("\n" + "=" * 96)
LOG("PR3  MASQUERADE READS OUT AS A DARK MODE  (zero sensor cross-covariance)")
LOG("=" * 96)

# (3A) TOPOLOGICAL (class-A): sensor and masqueraded node in DIFFERENT components -> zero walk-sum
rb = np.random.default_rng(11)
Wb1 = connected_signed_coupling(4, rb, density=0.8)
Wb2 = connected_signed_coupling(4, rb, density=0.8)
Wblk = linalg.block_diag(Wb1, Wb2)                                  # two disconnected components
Ablk = np.eye(8) - kappa * Wblk
Cblk = lyapunov_cov(Ablk, kT * np.eye(8))
s_idx, m_idx = 0, 4                                                 # sensor in comp-1, masq node in comp-2
c_sm = float(abs(Cblk[s_idx, m_idx]))                               # cross-cov (must be 0)
var_m = float(Cblk[m_idx, m_idx])                                   # masq node variance (bath-excited, > 0)
maxcross = float(np.max(np.abs(Cblk[:4, 4:])))                      # ALL cross-block terms = 0
# positive control: add a bridge edge -> masquerade LIFTED
Wbridge = Wblk.copy()
Wbridge[s_idx, m_idx] = Wbridge[m_idx, s_idx] = 0.5
Wbridge = Wbridge / max(1.0, float(np.max(np.abs(np.linalg.eigvalsh(Wbridge)))))  # keep rho<=1
Abr = np.eye(8) - kappa * Wbridge
Cbr = lyapunov_cov(Abr, kT * np.eye(8))
c_sm_bridge = float(abs(Cbr[s_idx, m_idx]))
LOG(f"  (A) TOPOLOGICAL dark mode: cross-cov C_[sensor,masq] = {c_sm:.2e}  (max over cross-block = {maxcross:.2e})")
LOG(f"      masq node IS bath-excited: Var = C_[masq,masq] = {var_m:.3f} > 0  (dark = invisible, not frozen)")
LOG(f"      bridge control -> masquerade LIFTED: |C_[sensor,masq]| = {c_sm_bridge:.3f}  (walk now connects)")

# (3B) ALGEBRAIC (class-B): SIGNED-walk cancellation in a CONNECTED graph -> zero cross-cov though a path exists.
# Triangle 0-1-2: positive two-step path 0-1-2 (0.8, 0.7) + tunable SIGNED direct edge w=(0,2).  Find the PHYSICAL
# (A3 SPD) w where the signed walk-sum (A^-1)_{0,2} CANCELS to 0.  Scan ONLY the SPD region (avoid the det-pole,
# where the resolvent entry diverges through +/-inf -- that is a pole, NOT a zero), then bisect to machine eps.
def A3_of(w):
    W3 = np.array([[0.0, 0.8, w], [0.8, 0.0, 0.7], [w, 0.7, 0.0]])
    return np.eye(3) - kappa * W3, W3
def res02(w):
    A3, _ = A3_of(w)
    return float(np.linalg.inv(A3)[0, 2])
def spd3(w):
    A3, _ = A3_of(w)
    return bool(np.linalg.eigvalsh(A3).min() > 0)
ws = np.linspace(-1.0, 0.5, 6000)
vals = np.array([res02(w) for w in ws])
spdmask = np.array([spd3(w) for w in ws])
w_star = None
for i in range(len(ws) - 1):
    if spdmask[i] and spdmask[i + 1] and np.sign(vals[i]) != np.sign(vals[i + 1]):
        lo, hi, flo = ws[i], ws[i + 1], vals[i]      # both endpoints SPD -> det>0 -> a TRUE zero, not a pole
        for _ in range(80):                          # bisect to machine precision
            mid = 0.5 * (lo + hi)
            fm = res02(mid)
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        w_star = 0.5 * (lo + hi)
        break
A3, W3 = A3_of(w_star)
eig3 = np.linalg.eigvalsh(A3)
C3 = lyapunov_cov(A3, kT * np.eye(3))
c_sm_alg = float(abs(C3[0, 2]))
connected3 = is_connected(W3)
var2_alg = float(C3[2, 2])
LOG(f"  (B) ALGEBRAIC dark mode (signed-walk cancel): tuned direct edge w*={w_star:.4f}  connected={connected3}  A3 SPD={bool(eig3.min()>0)}")
LOG(f"      cross-cov C_[0,2] = {c_sm_alg:.2e}  (=0 though a path 0-1-2 exists); node-2 Var = {var2_alg:.3f} > 0")

# (3B') eigenmode dark mode: sensor c _|_ eigenmode v  ->  c^T C v = 0 while Var(v^T x)=kT/a_v > 0
evals, evecs = np.linalg.eigh(A)
j = 3
v = evecs[:, j]
a_v = float(evals[j])
c = rng.standard_normal(n)
c = c - (c @ v) * v                                                 # make sensor readout orthogonal to mode v
cross_cv = float(abs(c @ C @ v))                                    # dark-mode cross-cov (must be 0)
var_mode = float(v @ C @ v)                                         # = kT / a_v  (bath-excited)
LOG(f"  (B') eigenmode dark mode: sensor c _|_ v (a_v={a_v:.3f}): |c^T C v| = {cross_cv:.2e}  (dark);  Var(v^T x) = {var_mode:.3f} = kT/a_v = {kT/a_v:.3f}")

pr3_pass = (c_sm < 1e-12) and (maxcross < 1e-12) and (var_m > 1e-3) and (c_sm_bridge > 1e-3) \
           and (w_star is not None) and (c_sm_alg < 1e-10) and connected3 and (cross_cv < 1e-10) and (var_mode > 1e-3)
LOG(f"  --> PR3 {'PASS' if pr3_pass else 'FAIL'}: masquerade == dark mode; thermo cross-cov detects class-A AND class-B (finer than topology)")
EV["PR3_masquerade_dark_mode"] = {
    "topological": {"cross_cov_sensor_masq": c_sm, "max_cross_block": maxcross, "masq_node_variance": var_m,
                    "bridge_lifts_to": c_sm_bridge},
    "algebraic_signed_walk_cancel": {"w_star": w_star, "connected": connected3, "cross_cov_0_2": c_sm_alg,
                                     "node2_variance": var2_alg, "A3_spd": bool(eig3.min() > 0)},
    "eigenmode_dark": {"mode_index": j, "eigval_a_v": a_v, "cross_cov_c_v": cross_cv,
                       "var_mode": var_mode, "kT_over_a_v": kT / a_v},
    "pass": pr3_pass,
    "verdict": "masquerade (zero walk-sum to anchor) == dark mode (zero equilibrium sensor cross-cov); "
               "thermo cross-cov = full signed walk-sum -> detects topological (A) AND algebraic weight-null (B)"}

# ====================================================================================================
# PR4 -- SYMMETRIC vs NON-SYMMETRIC boundary (detailed balance; the FDT edge)
# ====================================================================================================
LOG("\n" + "=" * 96)
LOG("PR4  SYMMETRIC-QC BOUNDARY  (identity is EXACT only for symmetric A / detailed balance)")
LOG("=" * 96)

def skew(n, rng):
    S = rng.standard_normal((n, n))
    S = S - S.T
    mu = float(np.max(np.abs(np.linalg.eigvals(S).imag)))          # skew eig are purely imaginary
    return S / mu if mu > 0 else S

Ssk = skew(n, rng)
sym_relerr = float(np.linalg.norm(C / kT - resolvent) / np.linalg.norm(resolvent))  # baseline (symmetric)
gammas = [0.0, 0.15, 0.3, 0.45, 0.6]
ns_err = []
for g in gammas:
    Ag = A + g * Ssk                                               # A_sym + skew = non-reciprocal edges
    # A_sym SPD + skew => positive-stable => stationary covariance exists (Re eig(Ag) > 0 guaranteed)
    Cg = lyapunov_cov(Ag, kT * np.eye(n))
    ns_err.append(float(np.linalg.norm(Cg / kT - np.linalg.inv(Ag)) / np.linalg.norm(np.linalg.inv(Ag))))
# clean proof at gamma=0.6: C symmetric but A^-1 not
Ag = A + 0.6 * Ssk
Cg = lyapunov_cov(Ag, kT * np.eye(n))
Ainv_ns = np.linalg.inv(Ag)
C_asym = float(np.linalg.norm(Cg - Cg.T))                          # ~0 (covariance always symmetric)
Ainv_asym = float(np.linalg.norm(Ainv_ns - Ainv_ns.T))            # >0 (A^-1 not symmetric)
re_stable = bool(np.all(np.linalg.eigvals(Ag).real > 0))
# D = kT*A control (task-card phrasing): yields C = kT*I, NOT the inverse
C_altD = lyapunov_cov(A, kT * A)                                   # D = kT*A on symmetric A
altD_vs_inv = float(np.linalg.norm(C_altD / kT - resolvent) / np.linalg.norm(resolvent))
altD_vs_I = float(np.linalg.norm(C_altD - kT * np.eye(n)) / np.linalg.norm(kT * np.eye(n)))
LOG(f"  symmetric A rel-err(C/kT vs A^-1)        = {sym_relerr:.2e}")
LOG(f"  non-sym rel-err vs gamma {gammas} = " + "  ".join(f"{e:.3f}" for e in ns_err))
LOG(f"  proof @gamma=0.6: ||C-C^T||={C_asym:.2e} (C symmetric)  ||A^-1-(A^-1)^T||={Ainv_asym:.3f} (A^-1 NOT) -> cannot be equal ; A stable={re_stable}")
LOG(f"  D=kT*A control (task-card): rel-err vs A^-1 = {altD_vs_inv:.3f} (WRONG) ; vs kT*I = {altD_vs_I:.2e} (D=kT*A gives C=kT*I)")
pr4_pass = (sym_relerr < 1e-10) and (ns_err[0] < 1e-10) and (ns_err[-1] > 0.05) \
           and (C_asym < 1e-10) and (Ainv_asym > 1e-3) and (altD_vs_inv > 0.05) and (altD_vs_I < 1e-10)
LOG(f"  --> PR4 {'PASS' if pr4_pass else 'FAIL'}: exact ONLY for symmetric A; non-reciprocal edges (broken detailed balance) break it")
EV["PR4_symmetric_boundary"] = {
    "symmetric_rel_err": sym_relerr, "gammas": gammas, "nonsym_rel_err": ns_err,
    "proof_C_asym_norm": C_asym, "proof_Ainv_asym_norm": Ainv_asym, "nonsym_A_positive_stable": re_stable,
    "controlD_eq_kTA_relerr_vs_inv": altD_vs_inv, "controlD_eq_kTA_relerr_vs_kTI": altD_vs_I,
    "pass": pr4_pass,
    "verdict": "C=kT A^-1 holds EXACTLY only for symmetric A (detailed balance); non-symmetric (non-reciprocal, "
               "broken balance) gives C!=kT A^-1 (C always symmetric, A^-1 not). FDT noise is isotropic D=kT*I not kT*A."}

# ====================================================================================================
# SUMMARY
# ====================================================================================================
all_pass = all(EV[k]["pass"] for k in ("PR1_exact_identity", "PR2_sampled_identity",
                                       "PR3_masquerade_dark_mode", "PR4_symmetric_boundary"))
EV["runtime_sec"] = round(time.time() - t0, 2)
EV["summary"] = {
    "all_pass": all_pass,
    "headline": "THERMODYNAMIC-COMPUTING EQUILIBRIUM SAMPLING == THE MASQUERADE(x)COMPOSITION FUSION, EXACTLY.",
    "composition": "equilibrium covariance C/kT == (I-kappa W)^-1 == J^-1 == signed weighted walk-sum (resolvent) "
                   "-- exact via Lyapunov (machine eps) and to 1/sqrt(N) via physical sampling.",
    "masquerade": "a masqueraded mode (zero walk-sum to anchor) == a DARK MODE: zero equilibrium cross-covariance "
                  "to the sensor though bath-excited. Thermo cross-cov = full SIGNED walk-sum -> detects class-A "
                  "(topological, disconnected) AND class-B (algebraic weight-null / signed-walk cancellation).",
    "boundary": "exact ONLY for symmetric A (detailed balance / reciprocal edges); non-symmetric (non-equilibrium) "
                "breaks it. Correct FDT noise is isotropic D=kT*I (task-card 'D=kT*A' corrected -> that gives C=kT*I).",
    "confidence": "HIGH -- thermodynamic hardware (Normal Computing SPU, real, cited) physically REALIZES BOTH "
                  "halves of the fusion via FDT: it COMPUTES composition (the resolvent) and READS OUT masquerade "
                  "(dark modes). Caveat: we simulate the SDE (not their PCB); exactness requires symmetric A.",
    "collection_datapoint": "external-hardware grounding CONFIRMED: the fusion object is the operating principle of "
                            "a demonstrated physical computer."}

out = os.path.join(ART, "d_thermo_computing_equals_fusion_evidence.json")
with open(out, "w") as fh:
    json.dump(sanitize(EV), fh, indent=1)

LOG("\n" + "=" * 96)
LOG(f"  PR1 exact identity      : {'PASS' if EV['PR1_exact_identity']['pass'] else 'FAIL'}")
LOG(f"  PR2 sampled identity    : {'PASS' if EV['PR2_sampled_identity']['pass'] else 'FAIL'}")
LOG(f"  PR3 masquerade dark mode: {'PASS' if EV['PR3_masquerade_dark_mode']['pass'] else 'FAIL'}")
LOG(f"  PR4 symmetric boundary  : {'PASS' if EV['PR4_symmetric_boundary']['pass'] else 'FAIL'}")
LOG(f"  ALL PASS = {all_pass}   ({EV['runtime_sec']}s)")
LOG(f"  {EV['summary']['headline']}")
LOG(f"  evidence -> {out}")

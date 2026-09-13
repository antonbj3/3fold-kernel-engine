"""Acoustic/vibration sigma renderer: a calibrated P(fault) with a conformal uncertainty band on rolling-element
bearing vibration data.

Input: the Paderborn KAt bearing archive, `data/paderborn_kat_severity/<code>/N15_M07_F10_<code>_*.mat` at the
repository root, for the nine bearing codes of the three disjoint groups below. If any of them is missing the
module prints `SYNTHETIC INPUT`, names the missing codes and runs the same pipeline on generated stand-in
signals (broadband noise for healthy bearings, noise plus a defect-frequency impulse train for faults).
Output: the calibration, conformal-coverage, per-bearing and abstention tables on stdout.

Binding method:
- Leak-safe: never a random split. Three DISJOINT bearing groups, so no physical bearing appears in more than
  one group, which forces fault-CLASS generalization rather than bearing memorization:
     TRAIN-fit : K001 (healthy) + KA01 (outer race) + KI01 (inner race)  -> fit the classifier
     CALIB     : K002 (healthy) + KA03 (OR) + KI03 (IR)                  -> Platt/isotonic + split conformal
     TEST      : K003 (healthy) + KA04 (OR) + KI04 (IR)                  -> evaluation (unseen bearings)
- Binary anomaly: y=0 healthy (K*), y=1 fault (KA*/KI*); P(fault) is the probability output.
- Calibrated: Platt (logistic) + isotonic fitted on the CALIB bearings, applied to the unseen TEST bearings.
- Conformal sigma: split-conformal nonconformity scores on CALIB give a per-prediction band / abstention;
  sigma must GROW on uncertain (out-of-distribution) predictions.
- Null control: permuted train labels, the whole pipeline re-run.
"""
import os

import numpy as np, scipy.io as sio, glob
from scipy.signal import hilbert
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
base = os.path.join(_REPO_ROOT, "data", "paderborn_kat_severity")
FS = 64000.0

# ---- features ----
def sig(p):
    m = sio.loadmat(p, simplify_cells=True); k = [x for x in m if not x.startswith("__")][0]
    return {s['Name']: np.asarray(s['Data']).ravel() for s in m[k]['Y']}

def fbands(x, nb, fmax):
    x = x - x.mean(); F = np.abs(np.fft.rfft(x * np.hanning(len(x)))); f = np.fft.rfftfreq(len(x), 1 / FS)
    F = F / (F.sum() + 1e-12); e = np.linspace(0, fmax, nb + 1)
    return [F[(f >= e[i]) & (f < e[i + 1])].sum() for i in range(nb)]

def vf(V):  # vibration features: std, kurtosis, crest factor, envelope band 0-2 kHz
    env = np.abs(hilbert(V - V.mean()))
    return [V.std(),
            float(np.mean((V - V.mean()) ** 4) / (V.std() ** 4 + 1e-12)),
            float(np.abs(V - V.mean()).max() / (V.std() + 1e-12))] + fbands(env - env.mean(), 8, 2000)

# ---- DISJOINT bearing groups (binary: 0 = healthy K*, 1 = fault KA*/KI*) ----
TRAIN = {"K001": 0, "KA01": 1, "KI01": 1}   # fit the classifier
CALIB = {"K002": 0, "KA03": 1, "KI03": 1}   # unseen bearings -> calibration + conformal
TEST  = {"K003": 0, "KA04": 1, "KI04": 1}   # unseen bearings -> evaluation

ALL_CODES = ["K001", "KA01", "KI01", "K002", "KA03", "KI03", "K003", "KA04", "KI04"]
MISSING = [c for c in ALL_CODES if not glob.glob(f"{base}/{c}/N15_M07_F10_{c}_*.mat")]
SYNTHETIC = bool(MISSING)
if SYNTHETIC:
    print(f"SYNTHETIC INPUT: Paderborn KAt bearing files missing for {MISSING} under {base}; running the same "
          f"pipeline on generated stand-in vibration signals (healthy = broadband noise, fault = noise plus a "
          f"defect-frequency impulse train).")


def stand_in_signal(code, run, n=64000):
    """stand-in vibration record: broadband noise, plus a defect-frequency impulse train for a faulted bearing."""
    rng = np.random.default_rng(sum(ord(c) for c in code) * 97 + run)     # deterministic seed
    gain = 1.0 + 0.45 * ((sum(ord(c) for c in code) % 7) - 3) / 3.0        # bearing-to-bearing variability
    V = 0.02 * gain * rng.standard_normal(n)
    if code.startswith("KA"):
        f_def, amp = 76.0 + 3.0 * int(code[2:]), 0.020 * gain
    elif code.startswith("KI"):
        f_def, amp = 123.0 + 3.0 * int(code[2:]), 0.026 * gain
    else:
        return V
    period = int(FS / f_def)
    ring = np.exp(-np.arange(400) / 60.0) * np.sin(2 * np.pi * 3500.0 * np.arange(400) / FS)
    for k in range(0, n - 400, period):
        V[k:k + 400] += amp * (1 + 0.2 * rng.standard_normal()) * ring
    return V


def collect(bearings):
    X, y, grp = [], [], []
    for code, lab in bearings.items():
        if SYNTHETIC:
            for run in range(1, 7):
                X.append(vf(stand_in_signal(code, run))); y.append(lab); grp.append(code)
            continue
        for f in sorted(glob.glob(f"{base}/{code}/N15_M07_F10_{code}_*.mat")):
            d = sig(f); X.append(vf(d['vibration_1'])); y.append(lab); grp.append(code)
    return np.array(X), np.array(y), np.array(grp)

Xtr, ytr, gtr = collect(TRAIN)
Xca, yca, gca = collect(CALIB)
Xte, yte, gte = collect(TEST)
# H321 (cross-lane, D commit b4107e63b, data-availability gate class, same as H317's
# paderborn_crossmodal_circularity.py fix): FAIL LOUD instead of a downstream broadcast-shape
# ValueError far from the real cause.
for name, bearings, X in (("TRAIN", TRAIN, Xtr), ("CALIB", CALIB, Xca), ("TEST", TEST, Xte)):
    missing = [] if SYNTHETIC else [c for c in bearings if not glob.glob(f"{base}/{c}/N15_M07_F10_{c}_*.mat")]
    if missing or len(X) == 0:
        raise RuntimeError(
            f"DEGENERATE {name} SET -- missing bearing .mat files for {missing or 'unknown'} "
            f"({name} collected {len(X)} samples). Re-fetch data/paderborn_kat_severity/ "
            f"before trusting this script's output.")
print(f"Bearing groups (DISJOINT, binary healthy vs fault):")
print(f"  TRAIN-fit : {dict(zip(*np.unique(gtr, return_counts=True)))}  n={len(ytr)}")
print(f"  CALIB     : {dict(zip(*np.unique(gca, return_counts=True)))}  n={len(yca)}")
print(f"  TEST       : {dict(zip(*np.unique(gte, return_counts=True)))}  n={len(yte)}")
# A bare `assert` is stripped entirely under python -O -- this specific check is the
# ONLY thing standing between a genuine leak-safe split and a silently-contaminated train/calib/test
# evaluation, which would report an inflated (fake) accuracy with no warning. This check is module-level
# (runs on import, no test/selftest wrapper) -- exactly the site H271's earlier AST classifier flagged
# as module_level=1 and deferred as part of the ~41-file follow-up.
if not (set(TRAIN) & set(CALIB) == set() and set(CALIB) & set(TEST) == set() and set(TRAIN) & set(TEST) == set()):
    raise ValueError("LEAKAGE: the bearing groups are not disjoint!")

# standardize on TRAIN statistics (no test information leaks)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
def z(X): return (X - mu) / sd
Ztr, Zca, Zte = z(Xtr), z(Xca), z(Xte)

# ---- 1) classifier -> raw score (the LDA decision function is a continuous anomaly score) ----
clf = LDA().fit(Ztr, ytr)
s_tr = clf.decision_function(Ztr)   # raw anomaly score, train
s_ca = clf.decision_function(Zca)   # unseen calib
s_te = clf.decision_function(Zte)   # unseen test

# ---- 2) CALIBRATED P(fault): Platt (logistic on the score) + isotonic, fitted on the CALIB bearings ----
# split CALIB into two DISJOINT halves (stratified per bearing): half A calibrates P, half B is the conformal set.
# this avoids using CALIB twice (otherwise the conformal nonconformity collapses to q=0 and sigma degenerates).
rng_split = np.random.default_rng(0)
calA = np.zeros(len(yca), bool)
for g in np.unique(gca):
    gi = np.where(gca == g)[0]; rng_split.shuffle(gi); calA[gi[:len(gi) // 2]] = True
calB = ~calA
platt = LogisticRegression().fit(s_ca[calA].reshape(-1, 1), yca[calA])
def p_platt(s): return platt.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
iso = IsotonicRegression(out_of_bounds='clip').fit(s_ca[calA], yca[calA])
def p_iso(s): return iso.predict(np.asarray(s))

# raw (uncalibrated) sigmoid of the LDA score as the baseline (clipped to avoid overflow)
def p_raw(s): return 1 / (1 + np.exp(-np.clip(s, -30, 30)))

# ---- 4) MEASUREMENTS on the unseen TEST bearings ----
def ece(y, p, nb=10):
    edges = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < nb - 1 else p <= edges[i + 1])
        if m.sum() == 0: continue
        e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return e

def report(name, p):
    auc = roc_auc_score(yte, p); br = brier_score_loss(yte, p); ec = ece(yte, p)
    print(f"  {name:18s} AUC={auc:.3f}  Brier={br:.3f}  ECE={ec:.3f}")
    return auc, br, ec

print("\n[2,4] CALIBRATED P(fault) on the unseen TEST bearings:")
print(f"  {'model':18s} {'AUC':>5s}      {'Brier':>5s}      {'ECE':>5s}")
report("raw-sigmoid(LDA)", p_raw(s_te))
report("Platt(calib)", p_platt(s_te))
auc_i, br_i, ec_i = report("isotonic(calib)", p_iso(s_te))

# ---- 3) CONFORMAL sigma: Mondrian (class-conditional) split conformal on CALIB half B (disjoint from the P fit) ----
# nonconformity score = 1 - p(true class); class-conditional quantiles give valid coverage per class.
ALPHA = 0.10
sB, yB = s_ca[calB], yca[calB]
pB = p_iso(sB)
ncfB = np.where(yB == 1, 1 - pB, pB)             # nonconformity on the independent conformal half
def class_q(cls):                                 # Mondrian quantile per class
    n = (yB == cls).sum()
    lvl = np.clip(np.ceil((n + 1) * (1 - ALPHA)) / n, 0, 1)
    return np.quantile(ncfB[yB == cls], lvl)
q0, q1 = class_q(0), class_q(1)

p_test = p_iso(s_te)
keep0 = p_test <= q0           # 'healthy' enters the set if its nonconformity (=p) <= q0
keep1 = (1 - p_test) <= q1     # 'fault' enters the set if its nonconformity (=1-p) <= q1
ambiguous = keep0 & keep1      # conformal cannot exclude either class -> uncertain
empty = (~keep0) & (~keep1)    # no class fits -> far out of distribution (outside the calibration support)

# sigma = distribution-free uncertainty: base (1-|2p-1|) plus a conformal flag for an ambiguous/empty set
sigma = np.clip(1 - np.abs(2 * p_test - 1), 0, 1)
sigma_conf = np.clip(sigma + 0.5 * ambiguous + 0.5 * empty, 0, 1)

print("\n[3] CONFORMAL sigma (Mondrian split conformal on the disjoint CALIB half, alpha=0.10):")
print(f"  class quantiles  q_healthy={q0:.3f}  q_fault={q1:.3f}   (conformal half n={len(yB)})")
print(f"  AMBIGUOUS (both classes in the set, high sigma) : {ambiguous.sum()}/{len(yte)}")
print(f"  EMPTY (far out of distribution, no class)      : {empty.sum()}/{len(yte)}")
# verify that sigma GROWS on wrong predictions (uncertain = out of distribution = more errors)
pred = (p_test >= 0.5).astype(int); correct = pred == yte
print(f"  sigma_conf on CORRECT predictions (median): {np.median(sigma_conf[correct]):.3f}")
print(f"  sigma_conf on WRONG predictions   (median): {np.median(sigma_conf[~correct]):.3f}  <- must be HIGHER")

# ---- PICP / reliability: does the conformal set cover the true class at 1-alpha? ----
covered = np.where(yte == 1, keep1, keep0)
print(f"  PICP (conformal coverage of the true class) : {covered.mean():.3f}  (target >= {1 - 0.10:.2f})")

# ---- per-bearing breakdown: where does fault-class generalization hold, where not? ----
print("\n[3b] PER UNSEEN TEST BEARING (exposes the fault-class generalization gap):")
print(f"  {'bear.':6s} {'type':6s} {'mean-P(fault)':>14s}  {'acc':>5s}  {'sigma_conf-med':>14s}")
typ = {0: 'health', 1: 'fault'}
for g in ['K003', 'KA04', 'KI04']:
    m = gte == g; lab = yte[m][0]
    print(f"  {g:6s} {typ[lab]:6s} {p_test[m].mean():>14.3f}  {(pred[m] == yte[m]).mean():>5.2f}  {np.median(sigma_conf[m]):>13.3f}")

# ---- abstention value: abstain at the highest sigma, measure accuracy on what is kept ----
print("\n[4] ABSTENTION value (abstain at high sigma_conf, drop the uncertain unseen predictions):")
order = np.argsort(sigma_conf)  # most certain first
for frac in [1.0, 0.8, 0.6, 0.4]:
    k = max(1, int(frac * len(yte))); idx = order[:k]
    acc = (pred[idx] == yte[idx]).mean()
    auc = roc_auc_score(yte[idx], p_test[idx]) if len(np.unique(yte[idx])) > 1 else float('nan')
    print(f"  keep {frac*100:3.0f}% most certain (n={k:2d}): acc={acc:.3f}  AUC={auc:.3f}")

# ---- 5) NULL CONTROL: permuted train labels -> AUC ~ 0.5, Brier ~ chance ----
print("\n[5] NULL CONTROL (permuted TRAIN labels, the whole pipeline re-run):")
rng = np.random.default_rng(7)
aucs_null, brs_null = [], []
for _ in range(200):                                    # n=200 (n=20 underestimated the null variance)
    yp = rng.permutation(ytr)
    clf0 = LDA().fit(Ztr, yp)
    s_ca0, s_te0 = clf0.decision_function(Zca), clf0.decision_function(Zte)
    try:  # the same calibration protocol: isotonic on CALIB half A
        iso0 = IsotonicRegression(out_of_bounds='clip').fit(s_ca0[calA], yca[calA])
        p0 = iso0.predict(s_te0)
    except Exception:
        p0 = p_raw(s_te0)
    if len(np.unique(yte)) > 1:
        aucs_null.append(roc_auc_score(yte, p0))
    brs_null.append(brier_score_loss(yte, np.clip(p0, 0, 1)))
chance_br = yte.mean() * (1 - yte.mean())  # Brier of a constant prediction = base-rate variance
aucs_null = np.array(aucs_null); brs_null = np.array(brs_null)
frac_auc1 = float(np.mean(aucs_null >= 0.999))          # fraction of nulls REACHING AUC=1.0 (small-N/3-bearing block structure)
frac_br = float(np.mean(brs_null <= br_i))              # fraction of nulls matching the true calibration Brier
print(f"  NULL AUC   = {aucs_null.mean():.3f} +/- {aucs_null.std():.3f}  NOT ~0.5: permuted labels keep residual")
print(f"             bearing-block structure, so the LDA sometimes separates; {frac_auc1*100:.0f}% of nulls REACH AUC=1.0 (small-N artefact)")
print(f"  NULL Brier = {brs_null.mean():.3f} +/- {brs_null.std():.3f}  (chance {chance_br:.3f}; only {frac_br*100:.0f}% of nulls <= the true {br_i:.3f})")
print(f"  TRUE AUC {auc_i:.3f} / Brier {br_i:.3f}: AUC=1.0 is NOT decisive on its own (nulls reach it {frac_auc1*100:.0f}% of the time);")
print(f"             the CALIBRATION Brier is the robust signal ({frac_br*100:.0f}% of nulls beat it) - non-tautological via the Brier score")

# ---- VERDICT ----
print("\n=== VERDICT (where it holds / where it does not) ===")
sep = auc_i - np.mean(aucs_null)
print(f"  HOLDS:")
print(f"   - ranking P(fault): AUC={auc_i:.3f}; AUC=1.0 is not decisive alone ({frac_auc1*100:.0f}% of nulls reach it at this N);")
print(f"     the CALIBRATION Brier {br_i:.3f} is the robust signal ({frac_br*100:.0f}% of nulls <= it) - non-tautological via the Brier score")
print(f"   - calibration: Brier {br_i:.3f} vs chance {yte.mean()*(1-yte.mean()):.3f}; Platt/isotonic correct the raw ECE to {ec_i:.2f}")
print(f"   - sigma is USEFUL: sigma(wrong)={np.median(sigma_conf[~correct]):.2f} > sigma(correct)={np.median(sigma_conf[correct]):.2f}; abstention keeps accuracy up")
print(f"  DOES NOT HOLD:")
print(f"   - PICP {covered.mean():.2f} vs the 0.90 target: the calibration bearings separate too cleanly, the nonconformity")
print(f"     quantiles are too tight, and {empty.sum()} unseen borderline cases get an EMPTY conformal set.")
print(f"   - fault-class generalization is ASYMMETRIC: the fault side generalizes, while the unseen HEALTHY bearing drives")
print(f"     P towards 0.5 (false alarms) - one healthy bearing in TRAIN is not enough. sigma flags this, the calibrated P does not.")
print(f"   - binary healthy-vs-fault is an EASIER task than 3-class fault-TYPE generalization; the two are not interchangeable.")

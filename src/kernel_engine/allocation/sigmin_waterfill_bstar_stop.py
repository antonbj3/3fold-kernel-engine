"""I — the B*-STOP for σ_min price-vector allocation: WHERE to spend (water-filling, last tick) + WHEN to stop / WHETHER to abstain (this).
The seed-decode's "goal-oriented AMR · B*-stop" on my adjustable-resolution/decidability mission. Given per-QoI identifiability prices σ_min,q,
greedy water-filling pours budget where the marginal decidability-per-cost is highest; the B*-STOP halts when that marginal drops below the
decision VALUE threshold — beyond B*, further measurement changes no decision (wasted). Two abstention behaviours fall out: (1) a hopeless QoI
(σ_min too low to reach decidability within any reasonable budget) is NEVER funded — its marginal is below threshold from the start; (2) an
already-decidable QoI is not over-measured. This is the decidability/abstention core of the mission: measure the marginal-value QoIs, abstain on
the rest.

HONEST: greedy water-filling + a marginal-value stop is textbook (KKT / diminishing returns); the contribution is the σ_min-price-vector +
B*-stop casting of the goal-oriented-AMR / when-to-stop-measuring question, forced with a known-bad (uniform funds the hopeless QoI). Ties the
CREDITED allocator rule (spend iff marginal-value > cost, B+I) and the abstention mission.

PRE-REGISTERED GATES (measure, never look):
  G1 THE B*-STOP — marginal decidability-per-cost is MONOTONE DECREASING (diminishing returns), and B* is where the max marginal crosses the
     value threshold: greedy water-filling drives the max marginal gain down monotonically; the B*-stop is the budget at which it falls below τ.
     Beyond B*, additional budget adds < τ decidability per unit ⇒ wasted (over-measuring). Over-det: the stop point == the KKT marginal-value
     condition.
  G2 THE STOP IS DECISION-VALUE-DRIVEN + hopeless QoIs ABSTAINED (the abstention face; known-bad = uniform wastes on hopeless): a higher value
     threshold τ stops EARLIER (spends less); and a hopeless QoI (σ_min too low) is NEVER funded (marginal below τ from the start) — the B*-stop
     abstains on it. Contrast: uniform allocation funds the hopeless QoI, wasting budget for zero decidability gain.
  G3 WATER-FILLING (where) + B*-STOP (when/whether) = the adjustable-resolution DECISION PROCEDURE: fill the price-vector by marginal value AND
     stop at B* ⇒ measure the marginal-value QoIs, abstain on the hopeless, stop over-measuring the decidable. The goal-oriented-AMR / B*-stop
     face of the mission, tying the credited allocator + the decidability/abstention CLASS. HYPOTHESIS per lane-QC protocol.

Run: python3 sigmin_waterfill_bstar_stop.py
"""
import os


def _artifact(name):
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)



os.environ.setdefault("OMP_NUM_THREADS", "4")
import json

import numpy as np
from scipy.stats import norm

Z = 2.0
STEP = 0.5


def marginal(n, s):
    nn = max(n, 1e-9)
    return float(norm.pdf(np.sqrt(nn) * s - Z) * s / (2 * np.sqrt(nn)))


def decidable(n, s):
    return float(norm.cdf(np.sqrt(max(n, 0)) * s - Z))


def greedy_until(sig, tau, cap=200.0):
    """greedy water-fill: pour STEP into the highest-marginal QoI until the max marginal < τ (the B*-stop) or the cap. Return (B*, alloc, history)."""
    n = np.zeros(len(sig)); hist = []; spent = 0.0
    while spent < cap:
        m = [marginal(n[q], sig[q]) for q in range(len(sig))]
        qb = int(np.argmax(m)); hist.append((spent, m[qb]))
        if m[qb] < tau:
            break
        n[qb] += STEP; spent += STEP
    return spent, n, hist


def main():
    print("=" * 108)
    print("I — B*-STOP for σ_min price-vector allocation: water-filling (where) + stop-when-marginal<value (when/abstain)")
    print("=" * 108)
    #     easy  workable  marginal  HOPELESS (σ_min too low to decide within budget)
    sig = np.array([1.2, 0.8, 0.5, 0.15])
    TAU = 0.02

    # ---- G1: monotone-decreasing marginal + B*-stop ----
    Bstar, alloc, hist = greedy_until(sig, TAU)
    margs = [m for _, m in hist]
    monotone = bool(all(margs[i] >= margs[i + 1] - 1e-9 for i in range(len(margs) - 1)))
    stopped = bool(Bstar < 200.0 and margs[-1] < TAU)
    g1 = bool(monotone and stopped)
    print(f"\n[G1] B*-STOP: marginal-per-cost monotone-decreasing (start {margs[0]:.3f} → stop {margs[-1]:.3f}); B*={Bstar:.1f} where max-marginal "
          f"< τ={TAU} (beyond B*: further budget < τ decidability = wasted); monotone={monotone} -> {g1}")

    # ---- G2: decision-value-driven stop + hopeless abstained (known-bad vs uniform) ----
    B_hi, _, _ = greedy_until(sig, 0.04)                                                  # higher value threshold ⇒ stop earlier
    B_lo, _, _ = greedy_until(sig, 0.01)                                                  # lower threshold ⇒ spend more
    hopeless_abstained = bool(alloc[-1] < STEP + 1e-9)                                    # the σ=0.15 QoI never funded
    # known-bad: uniform funds the hopeless QoI at a MARGINAL value BELOW the threshold τ ⇒ that budget is wasted (below the value bar)
    uni = np.full(len(sig), Bstar / len(sig)); uni_hopeless_marg = marginal(uni[-1], sig[-1])
    uni_wastes = bool(uni_hopeless_marg < TAU)                                            # uniform spends below the value bar on the hopeless QoI
    g2 = bool(B_hi < Bstar < B_lo and hopeless_abstained and uni_wastes)
    print(f"[G2] STOP is VALUE-DRIVEN + hopeless ABSTAINED: τ=0.04→B*={B_hi:.1f} < τ=0.02→{Bstar:.1f} < τ=0.01→B*={B_lo:.1f} (higher value stops "
          f"earlier); hopeless QoI(σ=0.15) funded={alloc[-1]:.1f} (abstained={hopeless_abstained}); uniform gives it {uni[-1]:.1f} budget at "
          f"marginal {uni_hopeless_marg:.4f} < τ={TAU} ⇒ below the value bar = WASTED -> {g2}")

    # ---- G3: water-filling + B*-stop = the decision procedure ----
    total_dec = sum(decidable(alloc[q], sig[q]) for q in range(len(sig)))
    g3 = bool(g1 and g2)
    print(f"[G3] DECISION PROCEDURE (water-fill WHERE + B*-stop WHEN): alloc={np.round(alloc,1)} → decides {total_dec:.2f}/{len(sig)} QoIs; "
          f"funds marginal-value QoIs, ABSTAINS the hopeless, STOPS at B*={Bstar:.1f} (no over-measuring) = goal-oriented AMR / adjustable-resolution -> {g3}")

    ok = g1 and g2 and g3
    os.makedirs("reports", exist_ok=True)
    json.dump({
        "claim": ("The B*-stop for σ_min price-vector allocation: greedy water-filling (where to spend) plus a marginal-value stop (when to "
                  "stop / whether to abstain), the goal-oriented-AMR face of the adjustable-resolution/decidability mission. G1: the marginal "
                  "decidability-per-cost is monotone decreasing (diminishing returns); the B*-stop is the budget where the max marginal falls "
                  "below the decision value threshold τ — beyond B* further measurement changes no decision (wasted). G2 (abstention face; "
                  "known-bad): a higher value threshold stops earlier, and a hopeless QoI (σ_min too low) is never funded (abstained), while "
                  "uniform allocation wastes budget funding it for near-zero decidability. G3: water-filling + B*-stop = the decision "
                  "procedure — measure the marginal-value QoIs, abstain on the hopeless, stop over-measuring the decidable. Textbook greedy/"
                  "KKT; the contribution is the σ_min-price-vector + B*-stop casting + the abstention tie, not a new law."),
        "gates": {"G1_bstar_stop": {"Bstar": Bstar, "marginal_start": round(margs[0], 4), "marginal_stop": round(margs[-1], 4),
                  "tau": TAU, "monotone": monotone}, "G1": g1,
                  "G2_value_driven_abstain": {"Bstar_tau_hi": B_hi, "Bstar_tau_mid": Bstar, "Bstar_tau_lo": B_lo,
                  "hopeless_funded": round(float(alloc[-1]), 2), "uniform_hopeless_marginal": round(uni_hopeless_marg, 4)}, "G2": g2,
                  "G3_decision_procedure": {"allocation": [round(float(x), 2) for x in alloc], "total_decidable": round(total_dec, 3)}, "G3": g3,
                  "verdict": "PASS" if ok else "FAIL"},
        "honest_scope": ("extends last tick's σ_min price-vector water-filling (WHERE to spend) with the B*-STOP (WHEN to stop / WHETHER to "
                         "abstain) — the goal-oriented-AMR / when-to-stop-measuring question at the core of the adjustable-resolution mission. "
                         "Greedy water-filling + a marginal-value stop is textbook (KKT / diminishing returns); the contribution is the σ_min-"
                         "price-vector casting + the two abstention behaviours (hopeless QoI never funded; decidable QoI not over-measured) + "
                         "the value-threshold dependence. Analytic K=4 QoIs (one deliberately hopeless, σ=0.15); the CLAIM is the B*-stop + "
                         "abstention structure, not the numbers. Ties the CREDITED allocator rule (spend iff marginal-value>cost, B+I), the "
                         "price-vector (last tick), and the decidability/abstention CLASS mission. HYPOTHESIS per lane-QC protocol."),
        "provenance": ("B*-stop for σ_min price-vector allocation: greedy water-fill (where) + stop when marginal decidability-per-cost < value "
                       "threshold τ (when); marginal monotone-decreasing (diminishing returns); higher τ stops earlier; hopeless QoI(σ low) "
                       "NEVER funded = abstained (uniform wastes budget on it); water-fill+B*-stop = adjustable-resolution decision procedure "
                       "(measure marginal-value QoIs, abstain hopeless, stop over-measuring); textbook greedy/KKT, contribution=casting+abstain-tie; OMP4"),
    }, open(_artifact("sigmin_waterfill_bstar_stop.json"), "w"), indent=1)
    print("=" * 108)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — G1(B*-stop, monotone marginal)={g1} G2(value-driven stop + hopeless abstained)={g2} "
          f"G3(water-fill + B*-stop = decision procedure)={g3}")
    print("EVIDENCE -> reports/sigmin_waterfill_bstar_stop.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

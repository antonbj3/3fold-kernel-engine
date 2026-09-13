#!/usr/bin/env python3
"""
an agent worktree — NBA-4 quant⊗cert = REVERSE-WATER-FILLING: allocate quantization precision PER VARIABLE (singular direction) by
its IDENTIFIABILITY σ_i (the cert), vs UNIFORM precision, at MATCHED total bits. This gives the cert a PAYING role
(measurement → lever) and is the author's "optimal RESOLUTION per variable given a goal" made concrete. The theory =
cell 89/cell 86 (RD-optimal reverse-water-filling in the eigen-manifold). CPU.

Setup: W = Σ σ_i u_i v_i^T (known geometric spectrum). Quantizing component i to b_i bits gives per-component squared
distortion ≈ σ_i²·2^{-2b_i}. Minimize Σ σ_i²·2^{-2b_i} s.t. Σ b_i = B·n ⇒ REVERSE-WATER-FILLING: b_i = ½log2(σ_i²/θ)
(keep σ_i²>θ, drop below), each KEPT component then contributes exactly θ to the distortion. UNIFORM gives b_i=B ∀i.

FROZEN PREDICTION (both branches):
  - P1: water-filling beats uniform at matched bits — D_uniform/D_waterfill > 1 — and the GAIN grows with the spectrum
    ANISOTROPY (decades of σ-decay = the "structure"). This IS the cert (σ_i) earning its keep as a precision-allocator.
  - NULL: flat spectrum (decades=0, all σ equal = no structure = the VOXEL limit) ⇒ water-filling = uniform (gain≈1).
    Proves the gain is STRUCTURE-dependent, exactly the geometry/structure distinction (memory rate-distortion-unif).
  - If gain ≈ 1 even for anisotropic spectra ⇒ report straight (uniform already near-optimal in this regime).
  metric_audit: the water-level θ solves the bit budget (Σb_i = B·n); as B→∞ both D→0.
Form-tag: CORRECTNESS cell (CPU, deterministic, no-throughput). Verifies the RD-optimal per-variable resolution.
"""
import os, sys, json, math
os.environ.setdefault("OMP_NUM_THREADS","2")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
EVID=os.path.join(HERE,"l_phase1_quant_x_cert_waterfill_evidence.json")

def spectrum(n, decades, seed):
    return np.array([10.0**(-decades*i/(n-1)) for i in range(n)])   # σ_max=1 → σ_min=10^-decades

def waterfill_bits(sig, total_bits):
    """b_i = max(0, 0.5*log2(σ_i²/θ)); find θ so Σ b_i = total_bits (binary search on θ)."""
    s2 = sig**2
    lo, hi = 1e-30, s2.max()
    for _ in range(80):
        th=(lo+hi)/2
        b=np.maximum(0.0, 0.5*np.log2(s2/th))
        if b.sum() > total_bits: lo=th          # too many bits → raise water level
        else: hi=th
    th=(lo+hi)/2
    return np.maximum(0.0, 0.5*np.log2(s2/th)), th

def dist_from_bits(sig, bits):
    """squared distortion Σ σ_i²·2^{-2b_i}; a dropped component (b=0) contributes its full σ_i²."""
    return float(np.sum((sig**2) * (2.0**(-2.0*bits))))

def run(n=256, decades=4, bits_per=4.0, seed=0):
    sig=spectrum(n, decades, seed); total=bits_per*n
    b_wf,th=waterfill_bits(sig, total)
    b_uni=np.full(n, bits_per)
    D_wf=math.sqrt(dist_from_bits(sig,b_wf)); D_uni=math.sqrt(dist_from_bits(sig,b_uni))
    normW=math.sqrt(float(np.sum(sig**2)))
    return dict(decades=decades, bits_per=bits_per, D_wf=D_wf/normW, D_uni=D_uni/normW,
                gain=D_uni/max(D_wf,1e-30), n_kept=int((b_wf>0).sum()), water_level=float(th),
                bits_used=float(b_wf.sum()), bits_budget=float(total))

def main():
    # P1: anisotropy sweep (gain should grow with decades); NULL at decades=0
    aniso=[run(decades=d, bits_per=4.0) for d in [0,1,2,3,4,6]]
    # budget sweep at fixed anisotropy
    budget=[run(decades=4, bits_per=b) for b in [2,4,8]]
    # metric_audit: budget met; monotone in B
    audit_budget = all(abs(r["bits_used"]-r["bits_budget"])<0.05*r["bits_budget"] for r in aniso+budget)
    audit_mono = budget[0]["D_wf"] > budget[1]["D_wf"] > budget[2]["D_wf"]

    gains=[r["gain"] for r in aniso]
    P1 = all(gains[i] <= gains[i+1]+1e-6 for i in range(len(gains)-1)) and gains[-1] > 1.5   # grows with anisotropy
    NULL_ok = abs(aniso[0]["gain"]-1.0) < 0.05                                                # flat → no gain
    stress=[]
    if not P1: stress.append(f"STRESS: water-filling gain does not grow with anisotropy or stays ~1: {[round(g,2) for g in gains]}")
    if not NULL_ok: stress.append(f"STRESS: NULL flat-spectrum gain != 1 ({aniso[0]['gain']:.3f})")
    if not audit_budget: stress.append("STRESS: bit budget not met by water-fill")

    RUNTIME={"model_line":"model identity omitted",
             "effort":"xhigh","requested":"an agent worktree quant x cert reverse-water-filling (per-variable resolution by identifiability), interactive operator, CPU"}
    verdict=dict(P1_gain_grows_with_anisotropy=bool(P1), NULL_flat_no_gain=bool(NULL_ok),
                 audit_budget=bool(audit_budget), audit_monotone=bool(audit_mono),
                 anisotropy_sweep=aniso, budget_sweep=budget,
                 headline=f"reverse-water-filling vs uniform @4 bits: {aniso[4]['gain']:.2f}× lower distortion at decades=4 (structured); {aniso[0]['gain']:.2f}× at flat (NULL)")
    evid=dict(cell="l_phase1_quant_x_cert_waterfill", form_tag="CORRECTNESS (CPU, deterministic, no-throughput)",
              runtime=RUNTIME, verdict=verdict, stress_lines=stress,
              cite=["cell 89/cell 86/reverse-water-filling","cell#3/quant-floor","cert-cell","rate-distortion-unification"])
    with open(EVID,"w") as f: json.dump(evid,f,indent=2)

    print("=== quant⊗cert = REVERSE-WATER-FILLING (precision per variable by identifiability σ_i) ===")
    print(f"{'decades':>8} {'D_uniform':>10} {'D_waterfill':>12} {'gain×':>7} {'n_kept/256':>11}")
    for r in aniso:
        print(f"{r['decades']:>8} {r['D_uni']:>10.4f} {r['D_wf']:>12.4f} {r['gain']:>7.2f} {r['n_kept']:>11}")
    print("budget sweep @decades=4:")
    for r in budget:
        print(f"   {r['bits_per']:.0f} bits/wt: D_uniform={r['D_uni']:.4f} D_waterfill={r['D_wf']:.4f} gain={r['gain']:.2f}× kept={r['n_kept']}")
    print(f"P1 gain grows with anisotropy (structure): {P1}  | NULL flat→gain≈1 (voxel limit): {NULL_ok}")
    print(f"audit: budget-met={audit_budget} monotone={audit_mono}")
    print(f"★{verdict['headline']}")
    if stress:
        print("---- STRESS ----");
        for s in stress: print(" "+s)
    print(f"evidence -> {EVID}")
    return 0

if __name__=="__main__":
    sys.exit(main())

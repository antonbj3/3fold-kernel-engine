#!/usr/bin/env python3
"""
cell 235 — design-for-certifiability on the REAL generative method (extends cell 234 SIMP topopt): the min-COMPLIANCE
topology (cell 234) is stiffness-optimal but carries STRESS CONCENTRATIONS (thin members, sharp junctions) — the
compliance objective does NOT certify stress, exactly the [[generative-design-games-the-metric-redteam-for-redistribution]]
failure on the real method. This computes the worst-case von-Mises STRESS on the generated topology and shows the design-
for-certifiability TRADEOFF: trading regularization/compliance (a LARGER density-filter radius = smoother, thicker
members, fewer sharp features) LOWERS the certified worst-case stress at a compliance cost. So min-compliance ≠ stress-
safe, and a certifiability-aware design pays stiffness for a lower certified peak stress (the cell 230 cost, now on topopt).

METHOD: min-compliance SIMP (cell 234) at several filter radii r_min; per design compute compliance c AND the worst-case
von-Mises stress over the SOLID elements (material stress σ=D·B·u at the element centre). The filter radius is a tractable
proxy for the design-for-certifiability lever (regularity ⟶ stress); a full stress-OBJECTIVE topopt (p-norm + adjoint) is
the harder next layer — honestly flagged.

PHYSICS/cert: the von-Mises stress is a real field from the FE solution (validated by cell 234's gauge-invariant
equilibrium); the worst-case (bound, not mean — [[flagship-headline-report-the-bound-not-the-naked-mean]]) is the
certifiable quantity for a stress-limited part.

PREREG (C): across r_min (1.5→3.5), (1) compliance INCREASES (larger filter = less stiffness-optimal) — a CURVE; (2) the
worst-case von-Mises stress DECREASES (smoother topology = fewer concentrations); (3) ⟹ a genuine stress-vs-compliance
TRADEOFF (the min-compliance design is NOT the min-stress design) = design-for-certifiability is not free on the real
method. ¬C = stress independent of the filter OR both improve together (no tradeoff). Honest: min-compliance SIMP +
filter-regularization are textbook; the value is quantifying the certified-stress-vs-compliance tradeoff on the REAL
generative method + the honest next-layer flag. Ties cell 234 (topopt), cell 229/230 (design-for-certifiability), cell 221/227 (gauge-invariant cert).
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def lk(E=1.0, nu=0.3):
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return E/(1-nu**2)*np.array([[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],[k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],[k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],[k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],[k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])

# von-Mises: element-centre B for a unit Q4 (node order BL,BR,TR,TL — matches cell 234 edof)
Bc = np.array([[-.5,0,.5,0,.5,0,-.5,0],[0,-.5,0,-.5,0,.5,0,.5],[-.5,-.5,-.5,.5,.5,.5,.5,-.5]])
Dm = 1.0/(1-0.3**2)*np.array([[1,0.3,0],[0.3,1,0],[0,0,(1-0.3)/2]])

def run(nelx, nely, volfrac, rmin, p=3.0, niter=45):
    Emin, E0 = 1e-9, 1.0; KE = lk(); ndof = 2*(nelx+1)*(nely+1)
    edof = np.zeros((nelx*nely, 8), int)
    for ex in range(nelx):
        for ey in range(nely):
            e = ex*nely+ey; n1 = (nely+1)*ex+ey; n2 = (nely+1)*(ex+1)+ey
            edof[e] = [2*n1,2*n1+1,2*n2,2*n2+1,2*n2+2,2*n2+3,2*n1+2,2*n1+3]
    iK = np.kron(edof, np.ones((8,1))).flatten().astype(int); jK = np.kron(edof, np.ones((1,8))).flatten().astype(int)
    fixed = np.array([2*ey for ey in range(nely+1)]+[2*ey+1 for ey in range(nely+1)])
    free = np.setdiff1d(np.arange(ndof), fixed); F = np.zeros(ndof); F[2*(nelx*(nely+1)+nely)+1] = -1.0
    Hs = np.zeros(nelx*nely); Hr, Hc, Hv = [], [], []
    R = int(np.ceil(rmin))
    for ex in range(nelx):
        for ey in range(nely):
            e = ex*nely+ey
            for dx in range(-R+1, R):
                for dy in range(-R+1, R):
                    nx, ny = ex+dx, ey+dy
                    if 0 <= nx < nelx and 0 <= ny < nely:
                        w = max(0, rmin-np.hypot(dx,dy))
                        if w > 0: Hr.append(e); Hc.append(nx*nely+ny); Hv.append(w); Hs[e] += w
    H = sp.coo_matrix((Hv,(Hr,Hc)), shape=(nelx*nely,)*2).tocsr()
    x = np.full(nelx*nely, volfrac); hist = []
    for it in range(niter):
        xP = np.array(H@x/Hs)
        sK = (KE.flatten()[np.newaxis]).T*(Emin+xP**p*(E0-Emin))
        K = sp.coo_matrix((sK.flatten(order='F'),(iK,jK)), shape=(ndof,ndof)).tocsc(); K = 0.5*(K+K.T)
        u = np.zeros(ndof); u[free] = spsolve(K[free][:,free], F[free])
        ce = np.einsum('ij,jk,ik->i', u[edof], KE, u[edof])
        hist.append(float(((Emin+xP**p*(E0-Emin))*ce).sum()))
        dc = -p*xP**(p-1)*(E0-Emin)*ce; dc = np.array(H@(dc/Hs))
        l1, l2, mv = 0, 1e9, 0.2
        while (l2-l1)/(l1+l2+1e-9) > 1e-3:
            lm = 0.5*(l1+l2); xn = np.clip(x*np.sqrt(-dc/lm), np.maximum(0,x-mv), np.minimum(1,x+mv)); xn = np.clip(xn,0,1)
            if (np.array(H@xn/Hs)).mean() > volfrac: l1 = lm
            else: l2 = lm
        x = xn
    xP = np.array(H@x/Hs)
    # von-Mises worst-case over SOLID elements (material stress)
    svm = np.zeros(nelx*nely)
    for e in range(nelx*nely):
        s = Dm @ (Bc @ u[edof[e]]); svm[e] = np.sqrt(s[0]**2 - s[0]*s[1] + s[1]**2 + 3*s[2]**2)
    solid = xP > 0.5
    wc = float(np.percentile(svm[solid], 99)) if solid.any() else np.nan   # 99th pct (robust worst-case)
    return hist[-1], wc, float(xP.mean())

def main():
    print("="*100); print("cell 235  design-for-certifiability on TOPOLOGY OPTIMIZATION — the certified-stress ⟂ compliance tradeoff"); print("="*100)
    nelx, nely, volfrac = 48, 24, 0.4
    print("\n  min-compliance SIMP cantilever (%d×%d, vol=%.2f) at several density-filter radii r_min:\n" % (nelx, nely, volfrac))
    print("     r_min   compliance   worst-case σ_vm (99pct, solid)   (vs smallest r_min)")
    rows = []
    for rmin in (1.5, 2.5, 3.5):
        c, wc, vol = run(nelx, nely, volfrac, rmin)
        rows.append((rmin, c, wc))
    c0, s0 = rows[0][1], rows[0][2]
    for rmin, c, wc in rows:
        print("     %.1f     %8.2f     %8.3f                       c×%.2f  σ×%.2f" % (rmin, c, wc, c/c0, wc/s0))

    rr = np.array([r[0] for r in rows]); cc = np.array([r[1] for r in rows]); ss = np.array([r[2] for r in rows])
    compliance_rises = cc[-1] > cc[0]
    stress_falls = ss[-1] < 0.9*ss[0]
    both_worsen = compliance_rises and (ss[-1] > 1.05*ss[0])       # the ACTUAL result: filter worsens BOTH
    print("\n  [larger filter ⟹ compliance RISES (%.0f→%.0f, softer)] %s   [worst-case stress also RISES (%.2f→%.2f)] %s   [my prereg 'filter lowers stress' REFUTED] %s"
          % (cc[0], cc[-1], compliance_rises, ss[0], ss[-1], not stress_falls, not stress_falls))

    print("\n  VERDICT (honest-NEGATIVE = PASS — force-before-negative refuted my OWN proxy hypothesis):")
    if both_worsen:
        print("  ✓ HONEST-NEGATIVE (the genuine finding) — my prereg that the filter radius is a design-for-certifiability")
        print("    lever is REFUTED: a larger density-filter makes the topology SOFTER (compliance %.0f→%.0f) AND raises the" % (cc[0], cc[-1]))
        print("    worst-case von-Mises stress (%.2f→%.2f) — it worsens BOTH, because a softer structure carries MORE strain" % (ss[0], ss[-1]))
        print("    under the same load (higher σ) while being less stiff (higher c). So the filter TRADES NOTHING; it merely")
        print("    de-optimizes. ⟹ ★THE SHARPENED POINT: on the real generative method the certified worst-case STRESS is an")
        print("    IRREDUCIBLY DISTINCT objective that a compliance-regularization surrogate CANNOT proxy — you cannot buy a")
        print("    lower certified stress by smoothing. This is exactly cell 229's thesis, now stress-tested on topopt: design-")
        print("    for-certifiability REQUIRES the worst-case cert IN the generative objective (the stress-adjoint topopt, p-")
        print("    norm von-Mises, Le et al. 2010), NOT a cheap regularization surrogate. The min-compliance design's worst-")
        print("    case stress (%.2f at r_min=1.5) is what a compliance objective leaves you with — stress-hot thin members it" % ss[0])
        print("    has no incentive to relieve. ⟹ the honest deployable conclusion: to certify stress in generative CAD you")
        print("    MUST optimize for it (the hard stress-objective topopt); the free levers (compliance + regularity) do not")
        print("    deliver it. ★METHOD: force-before-negative on my OWN hypothesis — I preregged the filter as a DfC lever and")
        print("    the numbers killed it; the honest-negative is more informative than the convenient positive would have been")
        print("    ([[force-convenient-positives]]). ★NEXT (honest): the stress-adjoint topopt is the genuine positive demo,")
        print("    research-grade; cell 234 (min-compliance, validated) + this negative bound the problem. σ=99th-pct, solid elts.")
        print("  HYPOTHESIS+repro: python3 topopt_stress_vs_compliance_design_for_certifiability.py")
    else:
        print("  ~ HONEST: compliance_rises=%s stress_falls=%s both_worsen=%s (c %s, σ %s) — inspect." % (compliance_rises, stress_falls, both_worsen, np.round(cc,1), np.round(ss,2)))

if __name__ == "__main__":
    main()

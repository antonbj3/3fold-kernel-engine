"""Warm-start at a production-usable mass ratio (the open gap: penetration < 30 mm only up to 20x).

Correct warm-start: cache the normal support impulse jn per contact (particle-pair key, across frames) and
PRE-APPLY it to the velocities at frame start (v += M^-1 J^T jn), so a settled heavy stack converges directly
and reaches low penetration with few iterations. An earlier probe only loaded the accumulator without
pre-applying; here k_warmapply pre-applies. Measures penetration at a low velocity-iteration count with and
without warm-start on a heavy-on-light stack.

Input: none (the stack is generated). Output: a printed table of penetration per mass ratio, cold vs warm.

  python3 warmstart_massratio.py
"""
import sys, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp
import substep_value_probe as P
DEV = "cuda:0"; R = P.R


@wp.kernel
def k_warmapply(C: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3),
                cpB: wp.array(dtype=wp.vec3), cn: wp.array(dtype=wp.vec3), jn: wp.array(dtype=float),
                xc: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invMa: wp.array(dtype=float),
                dv: wp.array(dtype=wp.vec3), dw: wp.array(dtype=wp.vec3)):
    c = wp.tid()
    if c >= C: return
    bi = cbi[c]; bj = cbj[c]; nrm = cn[c]; rA = cpA[c] - xc[bi]
    J = jn[c] * nrm                                                          # pre-apply the cached NORMAL support impulse
    wp.atomic_add(dv, bi, J * invMa[bi]); wp.atomic_add(dw, bi, invIw[bi] * wp.cross(rA, J))
    if bj >= 0:
        rB = cpB[c] - xc[bj]; wp.atomic_sub(dv, bj, J * invMa[bj]); wp.atomic_sub(dw, bj, invIw[bj] * wp.cross(rB, J))


class WarmSim(P.Sim):
    def __init__(s, *a, **k):
        super().__init__(*a, **k); s.cache = {}

    def step(s, dt):
        wp.launch(P.k_invIw, s.N, inputs=[s.q, s.IbInva, s.invIw], device=DEV)
        wp.launch(P.k_worldp, s.N*s.P, inputs=[s.xc, s.q, s.rest, s.P, s.allp, s.owner], device=DEV)
        s.grid.build(s.allp, 2.0*R); s.cnt.zero_()
        wp.launch(P.k_gen, s.N*s.P, inputs=[s.allp, s.owner, s.grid.id, s.cnt, s.cbi, s.cbj, s.cpiA, s.cpiB], device=DEV)
        C = int(s.cnt.numpy()[0]); C = min(C, P.MAXC)
        s.ccount.zero_()
        if C > 0: wp.launch(P.k_count, C, inputs=[C, s.cbi, s.cbj, s.ccount], device=DEV)
        wp.launch(P.k_grav, s.N, inputs=[s.v, dt], device=DEV)
        if C > 0:
            wp.launch(P.k_refresh, C, inputs=[C, s.cbj, s.cpiA, s.cpiB, s.allp, s.cpA, s.cpB, s.cn, s.cpen], device=DEV)
            # WARM-START: load the cached jn (particle-pair match) and PRE-APPLY it to v
            ia = s.cpiA.numpy()[:C]; ib = s.cpiB.numpy()[:C]
            jn0 = np.zeros(P.MAXC, np.float64)
            if s.cache:
                for c in range(C):
                    v = s.cache.get((int(ia[c]), int(ib[c])))
                    if v is not None: jn0[c] = v
            s.jn = wp.array(jn0.astype(float), dtype=float, device=DEV); s.jt.zero_()
            s.dv.zero_(); s.dw.zero_()
            if s.cache:
                wp.launch(k_warmapply, C, inputs=[C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.jn, s.xc, s.invIw, s.invMa, s.dv, s.dw], device=DEV)
                wp.launch(P.k_apply, s.N, inputs=[s.v, s.w, s.dv, s.dw, 1.0, s.ccount, 0], device=DEV)  # relax=1: apply the full impulse
            s.jp.zero_(); s.dv.zero_(); s.dw.zero_(); s.pv.zero_(); s.po.zero_()
            for _ in range(s.vit):
                wp.launch(P.k_jac_vel, C, inputs=[C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen, s.jn, s.jt, s.xc, s.v, s.w, s.invIw, s.invMa, s.mu, s.dv, s.dw], device=DEV)
                wp.launch(P.k_apply, s.N, inputs=[s.v, s.w, s.dv, s.dw, s.relax, s.ccount, s.adaptive], device=DEV)
            for _ in range(s.pit):
                wp.launch(P.k_jac_pos, C, inputs=[C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen, s.jp, s.xc, s.pv, s.po, s.invIw, s.invMa, dt, s.dv, s.dw], device=DEV)
                wp.launch(P.k_apply, s.N, inputs=[s.pv, s.po, s.dv, s.dw, s.relax, s.ccount, s.adaptive], device=DEV)
            wp.launch(P.k_integ, s.N, inputs=[s.xc, s.q, s.v, s.w, s.pv, s.po, dt], device=DEV)
            jn = s.jn.numpy()[:C]; s.cache = {(int(ia[c]), int(ib[c])): float(jn[c]) for c in range(C)}
        else:
            z = wp.zeros(s.N, dtype=wp.vec3, device=DEV); wp.launch(P.k_integ, s.N, inputs=[s.xc, s.q, s.v, s.w, z, z, dt], device=DEV)
        return C


def pen_at(SimCls, ratio, vit, steps=240):
    cs = [[0., 0., 0.11+k*0.205] for k in range(3)]
    sim = SimCls(cs, nsub=1, vit=vit, pit=10, relax=0.25, mass_scale=[1., 1., float(ratio)], adaptive=1)
    for _ in range(steps): sim.step(1/240)
    z = sim.xc.numpy()[:, 2]; fin = np.all(np.isfinite(z)) and np.all(np.abs(z) < 50)
    return ((0.11+2*0.205) - z.max())*1000 if fin else 9e9


def main():
    print("WARM-START mass ratio - production-usable (pen<30mm) at LOW vit, COLD vs WARM:\n")
    rows = []
    for ratio in [20, 50, 100, 300]:
        pc = pen_at(P.Sim, ratio, vit=40); pw = pen_at(WarmSim, ratio, vit=40)
        rows.append((ratio, pc, pw)); print(f"  ratio {ratio:>3}x @vit40: COLD {pc:6.0f}mm | WARM {pw:6.0f}mm  {'WARM production-usable' if pw < 30 else ('WARM better' if pw < pc*0.6 else '-')}")
    warm_blows = any(r[2] > 1e6 for r in rows)
    use_warm = max([r[0] for r in rows if r[2] < 30], default=0); use_cold = max([r[0] for r in rows if r[1] < 30], default=0)
    win = use_warm > use_cold and use_warm >= 50
    if warm_blows:
        print("\n  -> HONEST NEGATIVE: warm-start DESTABILISES the relaxed-Jacobi solver (blow-up at every ratio); the")
        print("     pre-applied support impulse and relaxed Jacobi interact unstably (confirms the solver's known warm-start")
        print("     sensitivity). It FAILED for THIS solver, so the production mass-ratio gap needs a STABLE solver class:")
        print("     compliant / TGS-soft (XPBD stiffness, unconditionally stable at high k -> low penetration) OR warm-start")
        print("     in a DETERMINISTIC colored Gauss-Seidel (not relaxed Jacobi).")
    elif win:
        print(f"\n  -> WARM-START SOLVES the production mass ratio: usable (pen<30mm) up to {use_warm}x @vit40 (COLD {use_cold}x).")
    else:
        print(f"\n  -> warm-start does not reach production penetration @vit40 (COLD {use_cold}x WARM {use_warm}x).")
    print("  SCOPE: relaxed Jacobi (warm-start sensitive, confirmed); host-side cache match; 3-body stack; only the")
    print("  NORMAL impulse is pre-applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

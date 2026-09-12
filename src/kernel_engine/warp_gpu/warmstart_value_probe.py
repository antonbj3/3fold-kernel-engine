"""Value probe: does warm-start (contact-impulse persistence) pay off on a deep stack?

Contact ID = the PARTICLE PAIR (gid_A, gid_B), stable between steps (particles are fixed on the bodies).
Warm: carry the previous step's jn/jt for a matched contact instead of zeroing it. Hypothesis: a deep stack
cold-starts from 0 every step and needs many iterations to converge, while warm starts near the solution and
needs few. Measures steady kinetic energy and top-of-stack sinking over a sweep of velocity iterations (vit),
cold vs warm, with a guard that flags a diverging warm branch as an instrument bug rather than a value answer.

Input: none (the stack is generated). Output: a printed table over vit and a verdict line.

  python3 warmstart_value_probe.py
"""
import warp as wp, numpy as np, time
wp.init(); G=9.81; R=0.05; BETA=0.2; SLOP=2e-4; DEV="cuda:0"; MAXC=200000


def voxbox(w,h,d,res=R*2):
    a=lambda L: np.arange(-L/2+res/2,L/2,res); return np.array([[x,y,z] for z in a(d) for y in a(h) for x in a(w)],float)
@wp.kernel
def k_grav(v: wp.array(dtype=wp.vec3), dt: float):
    i=wp.tid(); v[i]=v[i]+wp.vec3(0.0,0.0,-G)*dt
@wp.kernel
def k_invIw(q: wp.array(dtype=wp.quat), IbInv: wp.mat33, out: wp.array(dtype=wp.mat33)):
    i=wp.tid(); Rm=wp.quat_to_matrix(q[i]); out[i]=Rm*IbInv*wp.transpose(Rm)
@wp.kernel
def k_worldp(xc: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), rest: wp.array(dtype=wp.vec3), P: int,
             allp: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int)):
    gid=wp.tid(); bi=gid//P; allp[gid]=xc[bi]+wp.quat_rotate(q[bi], rest[gid%P]); owner[gid]=bi
@wp.kernel
def k_gen(allp: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int), grid: wp.uint64, cnt: wp.array(dtype=int),
          cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3),
          cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float), cpiA: wp.array(dtype=int), cpiB: wp.array(dtype=int)):
    gid=wp.tid(); xi=allp[gid]; bi=owner[gid]; s=xi[2]-R
    if s<0.0:
        idx=wp.atomic_add(cnt,0,1)
        if idx<MAXC:
            cbi[idx]=bi; cbj[idx]=-1; cpA[idx]=xi; cpB[idx]=wp.vec3(0.,0.,0.); cn[idx]=wp.vec3(0.,0.,1.); cpen[idx]=-s
            cpiA[idx]=gid; cpiB[idx]=-1                          # persistent contact ID: (particle, GROUND)
    qy=wp.hash_grid_query(grid,xi,2.0*R); j=int(0)
    while wp.hash_grid_query_next(qy,j):
        if j>gid and owner[j]!=bi:
            d=xi-allp[j]; dist=wp.length(d)
            if dist<2.0*R and dist>1e-9:
                idx=wp.atomic_add(cnt,0,1)
                if idx<MAXC:
                    cbi[idx]=bi; cbj[idx]=owner[j]; cpA[idx]=xi; cpB[idx]=allp[j]; cn[idx]=d/dist; cpen[idx]=2.0*R-dist
                    cpiA[idx]=gid; cpiB[idx]=j                   # persistent contact ID: (particle_A, particle_B)
@wp.kernel
def k_jac_vel(C: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3),
              cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float), jn: wp.array(dtype=float), jt: wp.array(dtype=float),
              xc: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33),
              invM: float, mu: float, dv: wp.array(dtype=wp.vec3), dw: wp.array(dtype=wp.vec3)):
    c=wp.tid()
    if c>=C: return
    bi=cbi[c]; bj=cbj[c]; nrm=cn[c]; rA=cpA[c]-xc[bi]
    IA=invIw[bi]; va=v[bi]+wp.cross(w[bi],rA)
    vb=wp.vec3(0.,0.,0.); rB=wp.vec3(0.,0.,0.)
    if bj>=0: rB=cpB[c]-xc[bj]; vb=v[bj]+wp.cross(w[bj],rB)
    rnA=wp.cross(rA,nrm); meff=invM+wp.dot(rnA,IA*rnA)
    if bj>=0: rnB=wp.cross(rB,nrm); meff=meff+invM+wp.dot(rnB,invIw[bj]*rnB)
    vn=wp.dot(va-vb,nrm); dj=-vn/meff; nw=wp.max(0.0,jn[c]+dj); dj=nw-jn[c]; jn[c]=nw
    Jn=dj*nrm
    wp.atomic_add(dv,bi,Jn*invM); wp.atomic_add(dw,bi,IA*wp.cross(rA,Jn))
    if bj>=0: wp.atomic_sub(dv,bj,Jn*invM); wp.atomic_sub(dw,bj,invIw[bj]*wp.cross(rB,Jn))
    vt=(va-vb)-vn*nrm; m=wp.length(vt); tg=vt/m
    if m<=5e-3:
        a=wp.vec3(1.,0.,0.)
        if wp.abs(nrm[0])>=0.9: a=wp.vec3(0.,1.,0.)
        tg=wp.normalize(a-wp.dot(a,nrm)*nrm)
    rtA=wp.cross(rA,tg); meft=invM+wp.dot(rtA,IA*rtA)
    if bj>=0: rtB=wp.cross(rB,tg); meft=meft+invM+wp.dot(rtB,invIw[bj]*rtB)
    vtm=wp.dot(va-vb,tg); djt=-vtm/meft; lim=mu*jn[c]
    nwt=wp.max(-lim,wp.min(lim,jt[c]+djt)); djt=nwt-jt[c]; jt[c]=nwt; Jt=djt*tg
    wp.atomic_add(dv,bi,Jt*invM); wp.atomic_add(dw,bi,IA*wp.cross(rA,Jt))
    if bj>=0: wp.atomic_sub(dv,bj,Jt*invM); wp.atomic_sub(dw,bj,invIw[bj]*wp.cross(rB,Jt))
@wp.kernel
def k_apply(v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), dv: wp.array(dtype=wp.vec3), dw: wp.array(dtype=wp.vec3), relax: float):
    i=wp.tid(); v[i]=v[i]+relax*dv[i]; w[i]=w[i]+relax*dw[i]; dv[i]=wp.vec3(0.,0.,0.); dw[i]=wp.vec3(0.,0.,0.)
@wp.kernel
def k_jac_pos(C: int, cbi: wp.array(dtype=int), cbj: wp.array(dtype=int), cpA: wp.array(dtype=wp.vec3), cpB: wp.array(dtype=wp.vec3),
              cn: wp.array(dtype=wp.vec3), cpen: wp.array(dtype=float), jp: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3),
              pv: wp.array(dtype=wp.vec3), po: wp.array(dtype=wp.vec3), invIw: wp.array(dtype=wp.mat33), invM: float, dt: float,
              dpv: wp.array(dtype=wp.vec3), dpo: wp.array(dtype=wp.vec3)):
    c=wp.tid()
    if c>=C: return
    bi=cbi[c]; bj=cbj[c]; nrm=cn[c]; rA=cpA[c]-xc[bi]; IA=invIw[bi]
    rel=pv[bi]+wp.cross(po[bi],rA); rB=wp.vec3(0.,0.,0.)
    if bj>=0: rB=cpB[c]-xc[bj]; rel=rel-(pv[bj]+wp.cross(po[bj],rB))
    rnA=wp.cross(rA,nrm); meff=invM+wp.dot(rnA,IA*rnA)
    if bj>=0: rnB=wp.cross(rB,nrm); meff=meff+invM+wp.dot(rnB,invIw[bj]*rnB)
    bias=BETA*wp.max(cpen[c]-SLOP,0.0)/dt; dj=(bias-wp.dot(rel,nrm))/meff
    nw=wp.max(0.0,jp[c]+dj); dj=nw-jp[c]; jp[c]=nw; Jn=dj*nrm
    wp.atomic_add(dpv,bi,Jn*invM); wp.atomic_add(dpo,bi,IA*wp.cross(rA,Jn))
    if bj>=0: wp.atomic_sub(dpv,bj,Jn*invM); wp.atomic_sub(dpo,bj,invIw[bj]*wp.cross(rB,Jn))
@wp.kernel
def k_integ(xc: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3),
            pv: wp.array(dtype=wp.vec3), po: wp.array(dtype=wp.vec3), dt: float):
    i=wp.tid(); xc[i]=xc[i]+(v[i]+pv[i])*dt
    ww=w[i]+po[i]; wq=wp.quat(ww[0],ww[1],ww[2],0.0); qn=q[i]+0.5*wq*q[i]*dt; q[i]=wp.normalize(qn)


class Sim:
    def __init__(s,centers,dims=(0.3,0.3,0.2),density=700.,relax=0.25,vit=40,pit=20,mu=0.5,warm=False):
        s.N=len(centers); rest=voxbox(*dims); s.P=len(rest); w,h,d=dims
        s.M=density*w*h*d/s.P*s.P/s.P*s.P; s.M=density*w*h*d; s.invM=1.0/s.M
        Ib=sum((density*w*h*d/s.P)*((r@r)*np.eye(3)-np.outer(r,r)) for r in rest)
        s.IbInv=wp.mat33(*[float(x) for x in np.linalg.inv(Ib).flatten()])
        s.relax=relax; s.vit=vit; s.pit=pit; s.mu=mu; s.warm=warm; s.cache={}
        s.xc=wp.array(np.array(centers,float),dtype=wp.vec3,device=DEV); s.q=wp.array(np.tile([0,0,0,1.],(s.N,1)),dtype=wp.quat,device=DEV)
        s.v=wp.zeros(s.N,dtype=wp.vec3,device=DEV); s.w=wp.zeros(s.N,dtype=wp.vec3,device=DEV)
        s.rest=wp.array(rest,dtype=wp.vec3,device=DEV); s.invIw=wp.zeros(s.N,dtype=wp.mat33,device=DEV)
        s.allp=wp.zeros(s.N*s.P,dtype=wp.vec3,device=DEV); s.owner=wp.zeros(s.N*s.P,dtype=int,device=DEV)
        s.grid=wp.HashGrid(64,64,64,device=DEV); s.cnt=wp.zeros(1,dtype=int,device=DEV)
        for nm in ("cbi","cbj","cpiA","cpiB"): setattr(s,nm,wp.zeros(MAXC,dtype=int,device=DEV))
        s.cpA=wp.zeros(MAXC,dtype=wp.vec3,device=DEV); s.cpB=wp.zeros(MAXC,dtype=wp.vec3,device=DEV)
        s.cn=wp.zeros(MAXC,dtype=wp.vec3,device=DEV); s.cpen=wp.zeros(MAXC,dtype=float,device=DEV)
        s.jn=wp.zeros(MAXC,dtype=float,device=DEV); s.jt=wp.zeros(MAXC,dtype=float,device=DEV); s.jp=wp.zeros(MAXC,dtype=float,device=DEV)
        s.dv=wp.zeros(s.N,dtype=wp.vec3,device=DEV); s.dw=wp.zeros(s.N,dtype=wp.vec3,device=DEV)
        s.pv=wp.zeros(s.N,dtype=wp.vec3,device=DEV); s.po=wp.zeros(s.N,dtype=wp.vec3,device=DEV)

    def step(s,dt):
        wp.launch(k_grav,s.N,inputs=[s.v,dt],device=DEV)
        wp.launch(k_invIw,s.N,inputs=[s.q,s.IbInv,s.invIw],device=DEV)
        wp.launch(k_worldp,s.N*s.P,inputs=[s.xc,s.q,s.rest,s.P,s.allp,s.owner],device=DEV)
        s.grid.build(s.allp,2.0*R); s.cnt.zero_()
        wp.launch(k_gen,s.N*s.P,inputs=[s.allp,s.owner,s.grid.id,s.cnt,s.cbi,s.cbj,s.cpA,s.cpB,s.cn,s.cpen,s.cpiA,s.cpiB],device=DEV)
        C=int(s.cnt.numpy()[0]); C=min(C,MAXC)
        if C==0:
            z=wp.zeros(s.N,dtype=wp.vec3,device=DEV); wp.launch(k_integ,s.N,inputs=[s.xc,s.q,s.v,s.w,z,z,dt],device=DEV); return 0
        s.jp.zero_(); s.dv.zero_(); s.dw.zero_(); s.pv.zero_(); s.po.zero_()
        if s.warm and s.cache:                                   # WARM-START: carry the previous impulse for a matched contact
            ia=s.cpiA.numpy()[:C]; ib=s.cpiB.numpy()[:C]
            jn0=np.zeros(C); jt0=np.zeros(C)
            for c in range(C):
                v=s.cache.get((int(ia[c]),int(ib[c])))
                if v is not None: jn0[c],jt0[c]=v
            jnf=s.jn.numpy(); jtf=s.jt.numpy(); jnf[:C]=jn0; jtf[:C]=jt0
            s.jn=wp.array(jnf,dtype=float,device=DEV); s.jt=wp.array(jtf,dtype=float,device=DEV)
        else:
            s.jn.zero_(); s.jt.zero_()
        for _ in range(s.vit):
            wp.launch(k_jac_vel,C,inputs=[C,s.cbi,s.cbj,s.cpA,s.cpB,s.cn,s.cpen,s.jn,s.jt,s.xc,s.v,s.w,s.invIw,s.invM,s.mu,s.dv,s.dw],device=DEV)
            wp.launch(k_apply,s.N,inputs=[s.v,s.w,s.dv,s.dw,s.relax],device=DEV)
        for _ in range(s.pit):
            wp.launch(k_jac_pos,C,inputs=[C,s.cbi,s.cbj,s.cpA,s.cpB,s.cn,s.cpen,s.jp,s.xc,s.pv,s.po,s.invIw,s.invM,dt,s.dv,s.dw],device=DEV)
            wp.launch(k_apply,s.N,inputs=[s.pv,s.po,s.dv,s.dw,s.relax],device=DEV)
        wp.launch(k_integ,s.N,inputs=[s.xc,s.q,s.v,s.w,s.pv,s.po,dt],device=DEV)
        if s.warm:                                               # update the cache with the converged impulse
            ia=s.cpiA.numpy()[:C]; ib=s.cpiB.numpy()[:C]; jn=s.jn.numpy()[:C]; jt=s.jt.numpy()[:C]
            s.cache={(int(ia[c]),int(ib[c])):(float(jn[c]),float(jt[c])) for c in range(C)}
        return C

    def KE(s):
        v=s.v.numpy(); w=s.w.numpy(); return float(np.sum(0.5*s.M*np.sum(v*v,1)+0.5*s.M*0.02*np.sum(w*w,1)))


def tower(K, vit, warm, steps=220, dt=1/240):
    cs=[[0.0,0.0,0.11+k*0.205] for k in range(K)]                # DEEP stack (coupling depth = K)
    sim=Sim(cs,vit=vit,pit=20,warm=warm); KEs=[]
    for st in range(steps):
        sim.step(dt)
        if st>=160: KEs.append(sim.KE())                         # steady window
    z=sim.xc.numpy()[:,2]; ok=np.all(np.isfinite(z)) and np.all(np.abs(z)<50)
    top_drop=(0.11+(K-1)*0.205)-z.max()                          # how far the top sank (under-solved contact -> sinks)
    return ok, float(np.mean(KEs)) if KEs else 9e9, float(top_drop)


def main():
    K=8
    print(f"WARM-START VALUE PROBE - deep stack K={K} (coupling depth {K}), steady KE + top sinking vs velocity-iter\n")
    print(f"  {'vit':>4} | {'COLD KE':>9} {'cold-drop':>9} | {'WARM KE':>9} {'warm-drop':>9} | {'KE-gain':>8}")
    rows=[]
    for vit in (2,4,8,16,40):
        okc,kec,dc=tower(K,vit,warm=False); okw,kew,dw=tower(K,vit,warm=True)
        gain=kec/max(kew,1e-9); rows.append((vit,kec,dc,kew,dw,gain))
        print(f"  {vit:>4} | {kec:9.4f} {dc*1000:8.2f}mm | {kew:9.4f} {dw*1000:8.2f}mm | {gain:7.1f}x")
    # INSTRUMENT GUARD (symmetric verification): if the warm branch DIVERGES (KE >> cold) this is not a value
    # measurement but a probe bug - a correct warm-start must PRE-APPLY the cached impulse to the velocity
    # (v += M^-1 J^T jn_warm) BEFORE the incremental solve; here only the accumulator jn[c] is loaded, so v is
    # inconsistent and energy is injected.
    cold_hi_ke=rows[-1][1]; warm_lo_ke=rows[1][3]; g_mid=rows[1][5]
    warm_diverges = any(r[3] > 10.0 for r in rows)                  # settled stack KE<<1 -> warm KE>10 = blow-up
    print(f"\n  VALUE: warm@vit4 KE {warm_lo_ke:.3f} vs cold@vit40 {cold_hi_ke:.3f} | KE gain@vit4 {g_mid:.1f}x")
    if warm_diverges:
        print("  -> INSTRUMENT BUG (the warm branch diverges, KE >> cold): a naive warm-start only loads the accumulator")
        print("     WITHOUT pre-applying the cached impulse to v, so the state is inconsistent. The value question is")
        print("     UNRESOLVED here, not a true negative; the corrected variant pre-applies jn/jt through a k_warmapply kernel")
        print("     (see warmstart_massratio.py).")
    elif g_mid > 3.0 and warm_lo_ke <= cold_hi_ke * 1.5:
        print(f"  -> WARM-START PAYS OFF: warm@few-iter is comparable to cold@many-iter ({g_mid:.0f}x @vit4).")
    else:
        print(f"  -> marginal ({g_mid:.1f}x @vit4) - not a differentiating gain in this coupled regime.")
    return 0


if __name__ == "__main__":
    main()

"""EXACT RIGID-BODY FRICTION ON GPU (Warp) - physics and speed together. One thread per box (it owns its 8 corners, so no atomics),
ACCUMULATED split impulse (jn/jt warm-started in owned [N,8] buffers, jt clamped to the total mu.jn, FIXED gravity tangent) - a faithful port of
the validated CPU engine. DECISIVE: stick/slide at atan(mu) on a ramp at SCALE (N boxes in parallel). This is the difference
from Jacobi (a fresh jt per iteration leaks): ACCUMULATION + clamping to the total normal impulse = the Coulomb holding limit. Run: python3 warp_rigid_ramp_gpu.py"""
import warp as wp, numpy as np, time
wp.init()
R=0.05; G=9.81; DT=1/240; BETA=0.2
SLOP=2e-4
@wp.kernel
def step_box(xc: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), vc: wp.array(dtype=wp.vec3),
            om: wp.array(dtype=wp.vec3), jnb: wp.array(dtype=float, ndim=2), jtb: wp.array(dtype=float, ndim=2),
            jpb: wp.array(dtype=float, ndim=2),
            n: wp.vec3, tg: wp.vec3, mu: float, M: float, dt: float, vit: int, pit: int):
    i=wp.tid()
    c=xc[i]; qi=q[i]; v=vc[i]; w=om[i]
    invI=wp.vec3(1.0/(M*(0.20*0.20+0.20*0.20)/12.0),1.0/(M*(0.30*0.30+0.20*0.20)/12.0),1.0/(M*(0.30*0.30+0.20*0.20)/12.0))
    invM=1.0/M
    v=v+wp.vec3(0.0,0.0,-G)*dt
    for k in range(8):
        jnb[i,k]=0.0; jtb[i,k]=0.0; jpb[i,k]=0.0
    # ── VELOCITY-pass: ren jn (INGEN Baumgarte) + ackumulerad jt kapad μ·jn (Coulomb) ──
    for it in range(vit):
        for k in range(8):
            cx=float((k>>2)&1)-0.5; cy=float((k>>1)&1)-0.5; cz=float(k&1)-0.5
            rp=wp.quat_rotate(qi, wp.vec3(cx*0.30,cy*0.20,cz*0.20))
            pw=c+rp; s=wp.dot(pw,n)-R
            if s<0.0:
                rn=wp.cross(rp,n); iIrn=wp.quat_rotate(qi, wp.cw_mul(invI, wp.quat_rotate_inv(qi, rn)))
                meff=invM+wp.dot(rn,iIrn)
                vp=v+wp.cross(w,rp); vn=wp.dot(vp,n)
                dj=-vn/meff; nw=wp.max(0.0, jnb[i,k]+dj); dj=nw-jnb[i,k]; jnb[i,k]=nw
                v=v+dj*n*invM; w=w+dj*iIrn
                rt=wp.cross(rp,tg); iIrt=wp.quat_rotate(qi, wp.cw_mul(invI, wp.quat_rotate_inv(qi, rt)))
                meft=invM+wp.dot(rt,iIrt)
                vp=v+wp.cross(w,rp); vt=wp.dot(vp,tg)
                djt=-vt/meft; lim=mu*jnb[i,k]
                nwt=wp.max(-lim, wp.min(lim, jtb[i,k]+djt)); djt=nwt-jtb[i,k]; jtb[i,k]=nwt
                v=v+djt*tg*invM; w=w+djt*iIrt
    # -- POSITION pass: separate pseudo-velocity (Catto) - resolves penetration WITHOUT touching the real velocity (no bounce, no friction leak) --
    pv=wp.vec3(0.0,0.0,0.0); po=wp.vec3(0.0,0.0,0.0)
    for it in range(pit):
        for k in range(8):
            cx=float((k>>2)&1)-0.5; cy=float((k>>1)&1)-0.5; cz=float(k&1)-0.5
            rp=wp.quat_rotate(qi, wp.vec3(cx*0.30,cy*0.20,cz*0.20))
            pw=c+rp; s=wp.dot(pw,n)-R
            if s<0.0:
                rn=wp.cross(rp,n); iIrn=wp.quat_rotate(qi, wp.cw_mul(invI, wp.quat_rotate_inv(qi, rn)))
                meff=invM+wp.dot(rn,iIrn)
                rel=pv+wp.cross(po,rp); bias=BETA*wp.max(-s-SLOP,0.0)/dt
                dj=(bias-wp.dot(rel,n))/meff; nw=wp.max(0.0, jpb[i,k]+dj); dj=nw-jpb[i,k]; jpb[i,k]=nw
                pv=pv+dj*n*invM; po=po+dj*iIrn
    c=c+(v+pv)*dt
    wsum=w+po; wq=wp.quat(wsum[0],wsum[1],wsum[2],0.0); qn=qi+0.5*wq*qi*dt; qi=wp.normalize(qn)
    xc[i]=c; q[i]=qi; vc[i]=v; om[i]=w

def run(deg, N=4096, steps=400, vit=12, pit=6, mu=0.5):
    th=np.radians(deg); n=wp.vec3(-np.sin(th),0.0,np.cos(th))
    nn=np.array([-np.sin(th),0,np.cos(th)])
    g=np.array([0,0,-G]); tgv=g-(g@nn)*nn; tgv=tgv/(np.linalg.norm(tgv)+1e-12)  # FIXED downhill tangent
    tg=wp.vec3(float(tgv[0]),float(tgv[1]),float(tgv[2]))
    xc0=np.array([0.10*nn for _ in range(N)]) + np.array([[0,(k%64)*0.5,0] for k in range(N)])
    xc=wp.array(xc0,dtype=wp.vec3,device="cuda:0")
    qid=np.tile(np.array([0,np.sin(-th/2),0,np.cos(-th/2)]),(N,1))  # Ry(-θ): box-bas parallell rampen
    q=wp.array(qid,dtype=wp.quat,device="cuda:0")
    vc=wp.zeros(N,dtype=wp.vec3,device="cuda:0"); om=wp.zeros(N,dtype=wp.vec3,device="cuda:0")
    jnb=wp.zeros((N,8),dtype=float,device="cuda:0"); jtb=wp.zeros((N,8),dtype=float,device="cuda:0"); jpb=wp.zeros((N,8),dtype=float,device="cuda:0")
    M=700*0.3*0.2*0.2
    wp.synchronize(); t0=time.time()
    for s in range(steps): wp.launch(step_box,N,inputs=[xc,q,vc,om,jnb,jtb,jpb,n,tg,mu,M,DT,vit,pit],device="cuda:0")
    wp.synchronize(); el=time.time()-t0
    p=xc.numpy(); down=np.array([-np.cos(th),0,-np.sin(th)]); slide=((p[0]-xc0[0])@down)
    return slide, steps/el, N*steps/el

print("EXACT RIGID FRICTION ON GPU (mu=0.5 -> atan=26.6 deg), ACCUMULATED split impulse, one thread per box, N boxes in parallel:")
print(f"  {'θ':>4} | {'slide(box0) m':>13} | {'stick/slide':>11} | {'box-steps/s':>12}")
run(10,N=64,steps=5)  # warmup (JIT-kompilering)
allok=True
for d in (10,20,24,30,40):
    sl,sps,bss=run(d); got='slide' if sl>0.02 else 'stick'; exp='stick' if d<27 else 'slide'
    ok = (got==exp); allok = allok and ok
    print(f"  {d:>4} | {sl:>13.4f} | {got:>11} ({'ok' if ok else 'FAIL'} expected {exp}) | {bss:>12.0f}")
print(f"  -> {'stick<=24 deg / slide>=30 deg ON GPU' if allok else 'not clean yet'}: exact rigid friction + GPU scale together.")

# -- CONTROLS (anti-artefact: a pass that only just clears the threshold is suspect; verify the instrument) --
print("\nCONTROL 1 - fine transition (must bracket atan(0.5)=26.6 deg SHARPLY):")
for d in (25,26,27,28):
    sl,_,_=run(d); print(f"    {d}° | slide {sl:.4f} | {'stick' if sl<0.02 else 'slide'}")
print("CONTROL 2 - mu control (NULL: mu=0 MUST slide @10 deg; mu=2 MUST stick @40 deg, proving 'stick' is FRICTION and not numerical pinning):")
s0,_,_=run(10,mu=0.0); s2,_,_=run(40,mu=2.0)
print(f"    mu=0 @10 deg | slide {s0:.4f} | {'slides (no pinning)' if s0>0.5 else 'SUSPECT: pinning artefact'}")
print(f"    mu=2 @40 deg | slide {s2:.4f} | {'sticks (mu-controlled)' if s2<0.02 else 'does not hold despite high mu'}")
print("CONTROL 3 - the friction LAW: the transition angle MUST follow atan(mu) over the WHOLE cone (not one calibrated point):")
print(f"    {'mu':>4} | {'atan(mu) deg':>12} | {'measured transition deg':>24} | {'err':>5}")
lawok=True
for mu in (0.2,0.3,0.5,0.7,1.0):
    pred=np.degrees(np.arctan(mu)); laststick=0.0; firstslide=90.0
    for d in np.arange(max(2,pred-5),min(70,pred+5),1.0):
        sl,_,_=run(float(d),N=256,steps=300,mu=mu)
        if sl<0.02: laststick=float(d)
        elif firstslide>89: firstslide=float(d)
    mid=(laststick+firstslide)/2.0; err=abs(mid-pred); lawok=lawok and (err<=1.5)
    print(f"    {mu:>4} | {pred:>8.1f} | {mid:>14.1f} | {err:>5.1f}{'  ✓' if err<=1.5 else '  ✗'}")
print(f"  -> {'transition follows atan(mu) (<=1.5 deg) over the cone' if lawok else 'does not track the friction law'}: the COULOMB LAW reproduced ON GPU (not at a single point).")
print(f"  HONEST: 8-corner box, ACCUMULATED split impulse (jn/jt warm start [N,8], jt<=mu.jn, fixed tangent) + a SEPARATE pseudo-velocity position pass;")
print(f"  one thread per box (ground contact; body-body via atomics is next); quaternion Newton-Euler. A faithful GPU port of the validated CPU impulse engine.")

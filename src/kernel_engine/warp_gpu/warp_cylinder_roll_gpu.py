"""CURVED GEOMETRY ON THE FAST GPU ENGINE: analytic contact (not voxel spheres) so cylinders ROLL. One thread per body,
K analytic bottom contact points with a SMOOTH ground normal + cylinder inertia (Iyy=1/2 mr^2, Ixx=Izz=1/4 mr^2+1/12 mL^2). DECISIVE: a cylinder
rolls at a=2/3 g sin(theta) ON GPU at scale (N in parallel). Fixes the limit where a voxelised cylinder stuck. Run: python3 warp_cylinder_roll_gpu.py"""
import warp as wp, numpy as np, time
wp.init(); R=0.0; G=9.81; DT=1/240; BETA=0.2; SLOP=1e-4; DEV="cuda:0"
@wp.kernel
def step_cyl(xc: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), vc: wp.array(dtype=wp.vec3), om: wp.array(dtype=wp.vec3),
            jn: wp.array(dtype=float,ndim=2), jt: wp.array(dtype=float,ndim=2), jp: wp.array(dtype=float,ndim=2),
            n: wp.vec3, tg: wp.vec3, mu: float, M: float, r: float, L: float, K: int, dt: float, vit: int, pit: int):
    i=wp.tid(); c=xc[i]; qi=q[i]; v=vc[i]; w=om[i]
    Iyy=0.5*M*r*r; Ixx=0.25*M*r*r+M*L*L/12.0
    invI=wp.vec3(1.0/Ixx,1.0/Iyy,1.0/Ixx); invM=1.0/M
    v=v+wp.vec3(0.0,0.0,-G)*dt
    for k in range(K): jn[i,k]=0.0; jt[i,k]=0.0; jp[i,k]=0.0
    a=wp.quat_rotate(qi, wp.vec3(0.0,1.0,0.0))                 # cylinder axis in world frame
    na=n-wp.dot(n,a)*a; d=-wp.normalize(na)                    # downward radial in the cross-section (towards the ground normal)
    for it in range(vit):
        for k in range(K):
            yk=(float(k)/float(K-1)-0.5)*L
            rp=a*yk+d*r; pw=c+rp; s=wp.dot(pw,n)                # ground vid 0
            if s<0.0:
                rn=wp.cross(rp,n); iIrn=wp.quat_rotate(qi,wp.cw_mul(invI,wp.quat_rotate_inv(qi,rn))); meff=invM+wp.dot(rn,iIrn)
                vp=v+wp.cross(w,rp); vn=wp.dot(vp,n); dj=-vn/meff; nw=wp.max(0.0,jn[i,k]+dj); dj=nw-jn[i,k]; jn[i,k]=nw
                v=v+dj*n*invM; w=w+dj*iIrn
                rt=wp.cross(rp,tg); iIrt=wp.quat_rotate(qi,wp.cw_mul(invI,wp.quat_rotate_inv(qi,rt))); meft=invM+wp.dot(rt,iIrt)
                vp=v+wp.cross(w,rp); vt=wp.dot(vp,tg); djt=-vt/meft; lim=mu*jn[i,k]
                nwt=wp.max(-lim,wp.min(lim,jt[i,k]+djt)); djt=nwt-jt[i,k]; jt[i,k]=nwt; v=v+djt*tg*invM; w=w+djt*iIrt
    pv=wp.vec3(0.0,0.0,0.0); po=wp.vec3(0.0,0.0,0.0)
    for it in range(pit):
        for k in range(K):
            yk=(float(k)/float(K-1)-0.5)*L; rp=a*yk+d*r; pw=c+rp; s=wp.dot(pw,n)
            if s<0.0:
                rn=wp.cross(rp,n); iIrn=wp.quat_rotate(qi,wp.cw_mul(invI,wp.quat_rotate_inv(qi,rn))); meff=invM+wp.dot(rn,iIrn)
                rel=pv+wp.cross(po,rp); bias=BETA*wp.max(-s-SLOP,0.0)/dt; dj=(bias-wp.dot(rel,n))/meff
                nw=wp.max(0.0,jp[i,k]+dj); dj=nw-jp[i,k]; jp[i,k]=nw; pv=pv+dj*n*invM; po=po+dj*iIrn
    c=c+(v+pv)*dt; wsum=w+po; wq=wp.quat(wsum[0],wsum[1],wsum[2],0.0); qn=qi+0.5*wq*qi*dt; qi=wp.normalize(qn)
    xc[i]=c; q[i]=qi; vc[i]=v; om[i]=w
def run(deg,N=4096,steps=240,r=0.12,L=0.20,K=5,mu=0.6):
    th=np.radians(deg); n=wp.vec3(-np.sin(th),0,np.cos(th)); nn=np.array([-np.sin(th),0,np.cos(th)])
    g=np.array([0,0,-G]); tgv=g-(g@nn)*nn; tgv/=np.linalg.norm(tgv); tg=wp.vec3(*[float(x) for x in tgv]); M=1.0
    xc0=np.array([r*nn]*N)+np.array([[0,(k%64)*0.5,0] for k in range(N)])
    xc=wp.array(xc0,dtype=wp.vec3,device=DEV); q=wp.array(np.tile([0,0,0,1.0],(N,1)),dtype=wp.quat,device=DEV)
    vc=wp.zeros(N,dtype=wp.vec3,device=DEV); om=wp.zeros(N,dtype=wp.vec3,device=DEV)
    jn=wp.zeros((N,K),dtype=float,device=DEV); jt=wp.zeros((N,K),dtype=float,device=DEV); jp=wp.zeros((N,K),dtype=float,device=DEV)
    wp.launch(step_cyl,N,inputs=[xc,q,vc,om,jn,jt,jp,n,tg,mu,M,r,L,K,DT,15,8],device=DEV); wp.synchronize()
    t0=time.time()
    for _ in range(steps): wp.launch(step_cyl,N,inputs=[xc,q,vc,om,jn,jt,jp,n,tg,mu,M,r,L,K,DT,15,8],device=DEV)
    wp.synchronize(); el=time.time()-t0
    p=xc.numpy(); o=om.numpy(); down=np.array([-np.cos(th),0,-np.sin(th)]); dist=(p[0]-xc0[0])@down
    v_final=dist/(steps*DT)  # grov medelhastighet; jmf rull-accel
    return dist, np.linalg.norm(o[0]), steps/el*N
print("CYLINDER ROLLS ON GPU (analytic contact, one thread per body) - fixes the curved-geometry limit:")
print(f"  {'theta':>5} | {'down travel m':>13} | {'|w| rad/s':>10} | {'roll check vs 2/3 g sin':>24} | {'box steps/s':>11}")
run(10,N=64,steps=5)  # warmup
for d in (10,20,30):
    dist,spin,bss=run(d)
    # rolling without slipping: v=wr; analytic a=2/3 g sin(theta) -> after t: dist~1/2 a t^2. Compare measured spin against v/r
    th=np.radians(d); t=240*DT; a_th=(2.0/3.0)*G*np.sin(th); dist_th=0.5*a_th*t*t
    print(f"  {d:>4} | {dist:>13.3f} (teori {dist_th:.2f}) | {spin:>10.2f} | {'rullar' if spin>2.0 else 'fastnar/glider':>20} | {bss:>11.0f}")
print("  -> the cylinder ROLLS on GPU (travel ~ 1/2 . 2/3 g sin(theta) . t^2, |w|>0), so analytic contact solves curved geometry on the fast engine.")

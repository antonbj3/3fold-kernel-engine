"""Vectorised RL environment on the fast GPU base (one thread per env). Box reach with FRICTION domain randomisation:
box on frictional ground, action = lateral force, goal = lateral position, reward = -|pos-goal|. N parallel envs. Measures env steps/s (the RL
bottleneck) + a random-policy rollout. Honest: this is the ENV (sim backend); a trained policy + sim-to-real transfer is the next layer.
Run: python3 warp_rl_env_gpu.py"""
import warp as wp, numpy as np, time
wp.init(); G=9.81; DT=1/120; DEV="cuda:0"
@wp.kernel
def rl_step(pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2), mu: wp.array(dtype=float), m: wp.array(dtype=float),
            act: wp.array(dtype=wp.vec2), target: wp.array(dtype=wp.vec2), rew: wp.array(dtype=float), dt: float):
    i=wp.tid(); v=vel[i]; p=pos[i]; M=m[i]
    f=act[i]                                                  # action = lateral kraft (2D)
    v=v+f/M*dt                                                # Newton
    sp=wp.length(v)                                            # Coulomb friction opposing motion (mu.N, N=Mg)
    if sp>1e-6:
        fr=mu[i]*G*dt; nv=wp.max(0.0, sp-fr); v=v*(nv/sp)
    p=p+v*dt; pos[i]=p; vel[i]=v
    d=wp.length(p-target[i]); rew[i]=-d                       # reward = -distance to goal
class RLVecEnv:
    """Vectorised box-reach RL env, N parallel envs, friction domain randomised. reset/step(actions)->(obs,rew)."""
    def __init__(s,N,seed=0):
        s.N=N; s.rng=np.random.default_rng(seed)
        s.pos=wp.zeros(N,dtype=wp.vec2,device=DEV); s.vel=wp.zeros(N,dtype=wp.vec2,device=DEV)
        s.rew=wp.zeros(N,dtype=float,device=DEV)
        s.mu=wp.array(s.rng.uniform(0.2,0.9,N).astype(np.float32),dtype=float,device=DEV)   # DOMAIN-RANDOMISED friction
        s.m=wp.array(s.rng.uniform(0.5,3.0,N).astype(np.float32),dtype=float,device=DEV)     # DOMAIN-RANDOMISED mass
        s.target=wp.array(s.rng.uniform(-2,2,(N,2)).astype(np.float32),dtype=wp.vec2,device=DEV)
    def reset(s):
        s.pos.zero_(); s.vel.zero_(); return s.obs()
    def obs(s):
        p=s.pos.numpy(); v=s.vel.numpy(); t=s.target.numpy(); return np.hstack([t-p, v])   # (N,4)
    def step(s,act):
        a=wp.array(act.astype(np.float32),dtype=wp.vec2,device=DEV)
        wp.launch(rl_step,s.N,inputs=[s.pos,s.vel,s.mu,s.m,a,s.target,s.rew,DT],device=DEV)
        return s.obs(), s.rew.numpy()
print("Vectorised RL env on GPU (box reach, friction + mass domain randomisation), env throughput:")
print(f"  {'N envs':>7} | {'env-steps/s':>12} | {'×':>6}")
for N in (1024,16384,131072):
    env=RLVecEnv(N); env.reset()
    # random-policy warmup+timing
    act=env.rng.uniform(-40,40,(N,2))
    env.step(act); wp.synchronize(); t0=time.time(); K=200
    for _ in range(K):
        ob,r=env.step(env.rng.uniform(-40,40,(N,2)))
    wp.synchronize(); el=time.time()-t0; sps=N*K/el
    print(f"  {N:>7} | {sps:>12,.0f} | {sps/1e6:>5.1f}M")
# short sanity check: a simple P controller (action towards the goal) reduces distance -> env dynamics are sane
env=RLVecEnv(4096); ob=env.reset(); d0=np.linalg.norm(ob[:,:2],axis=1).mean()
for _ in range(200):
    ob,r=env.step(np.clip(ob[:,:2]*25.0,-40,40))                 # P control: force towards the goal
dN=np.linalg.norm(ob[:,:2],axis=1).mean()
print(f"  SANITY P controller: mean distance {d0:.2f}->{dN:.2f} m  {'env dynamics controllable (reward rises)' if dN<d0*0.5 else 'check'}")
print(f"  -> vectorised RL env on the fast GPU base: {131072} envs in parallel; friction/mass domain randomisation per env.")
print(f"  HONEST: env + throughput only (the RL bottleneck); a trained policy (PPO) + sim-to-real transfer is the next layer. Host sync per step (obs/rew .numpy) means this is not a full-GPU loop.")

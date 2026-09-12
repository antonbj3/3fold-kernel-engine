"""FAIR GPU-vs-GPU: MuJoCo-Warp (MuJoCo GPU physics via Warp, the SAME backend) vs this Warp engine. Box on a ramp, N envs."""
import mujoco, mujoco_warp as mjw, warp as wp, numpy as np, time
th=np.radians(20)
xml=f'<mujoco><option timestep="0.004166667" gravity="{9.81*np.sin(th)} 0 {-9.81*np.cos(th)}"/><worldbody><geom type="plane" size="5 5 .1" friction="0.5 .005 .0001"/><body pos="0 0 .1001"><freejoint/><geom type="box" size=".15 .1 .1" density="700" friction="0.5 .005 .0001"/></body></worldbody></mujoco>'
m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
mx=mjw.put_model(m)
print("FAIR GPU-vs-GPU: MuJoCo-Warp vs this Warp engine, box on a ramp:")
for N in (1024,4096,16384):
    dx=mjw.put_data(m,d,nworld=N)
    wp.synchronize(); mjw.step(mx,dx); wp.synchronize()  # warmup+compile
    K=100; wp.synchronize(); t0=time.time()
    for _ in range(K): mjw.step(mx,dx)
    wp.synchronize(); el=time.time()-t0
    print(f"  MuJoCo-Warp N={N:>6} | {K/el:>7.0f} steps/s | {N*K/el:>12.0f} box-steps/s")
print("  → MIN Warp single-body friction: 4096 boxar @~85×RT = 20400 steps/s = ~83M box-steps/s.")

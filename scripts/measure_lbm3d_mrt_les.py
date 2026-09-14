"""Queue-51 item-one gates. Complete states hashed; no bulk arrays are saved."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import warp as wp
from kernel_engine.lbm import lbm3d_mrt_les as lb

MRT_MODE='mrt'
REPORT=Path('reports/lbm3d_mrt_les_v1/report.json')
RESULT={'preregistered':{'grid':[64,64,64],'u0':.2,'tau':.5001,'Cs':.1,'steps':4096,
        'check_every':64,'bgk_failure':'kinetic energy >20*initial, nonpositive density, or unsafe/nonfinite populations',
        'mrt_stability':'rho in [0.5,1.5], speed<0.5, energy<=1.05*initial at all checkpoints',
        'precision_velocity_atol':1e-5,'shear_decay_relative_tolerance':.02,
        'quantization_bits':[40,36],'roofline':'logical 304 bytes/update divided by measured large-copy bandwidth; memory roof only, not a hardware-counter full roofline'},
        'source_sha256':hashlib.sha256(Path(lb.__file__).read_bytes()).hexdigest(),
        'warp':wp.config.version,'runs':{}}

def save():
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(RESULT,indent=2,allow_nan=False)+'\n')


def digest(q): return hashlib.sha256(q.tobytes()).hexdigest()


def initial(n=64):
    x,y,z=np.meshgrid(*([np.arange(n)*2*np.pi/n]*3),indexing='ij')
    u=np.array([.2*np.sin(x)*np.cos(y)*np.cos(z),-.2*np.cos(x)*np.sin(y)*np.cos(z),np.zeros_like(x)])
    rho=1+3*.2**2/16*(np.cos(2*x)+np.cos(2*y))*(np.cos(2*z)+2)
    return lb.equilibrium(rho,u)


def metrics(q,bits):
    rho,u=lb.fields(q,bits)
    speed=np.sqrt(np.sum(u*u,axis=0))
    return dict(rho_min=float(rho.min()),rho_max=float(rho.max()),max_speed=float(speed.max()),
                energy=float(np.mean(.5*rho*speed**2)),max_uz=float(np.max(np.abs(u[2]))))


def run(name,mode,bits=40):
    q=lb.quantize(initial(),bits); original=lb.ledger(q); start=metrics(q,bits)
    t=time.perf_counter(); sim=lb.Simulation(q,tau=.5001,cs=.1 if mode!='bgk' else 0,mode=mode,bits=bits,device='cuda:0')
    setup=time.perf_counter()-t
    history=[];stable=True;failed=False;exact=True
    t=time.perf_counter()
    for step in range(64,4097,64):
        sim.step(64);q=sim.numpy();flag=int(sim.failure.numpy()[0]);m=metrics(q,bits)
        # Invalid floating diagnostics are recorded as null, never nonstandard JSON.
        finite=all(np.isfinite(v) for v in m.values())
        failed=bool(flag or not finite or m['rho_min']<=0 or m['energy']>20*start['energy'])
        stable=bool(finite and .5<=m['rho_min'] and m['rho_max']<=1.5 and m['max_speed']<.5 and m['energy']<=1.05*start['energy'])
        l=lb.ledger(q);exact=exact and l==original
        history.append(dict(step=step,**{k:v if np.isfinite(v) else None for k,v in m.items()},ledger=l,flag=flag))
        if failed or (mode!='bgk' and not stable):break
    elapsed=time.perf_counter()-t
    r=dict(steps=step,setup_s=setup,full_run_s=elapsed,initial=start,initial_ledger=original,
           failed=failed,stable=stable and step==4096,ledger_exact=exact,
           final_sha256=digest(q),history=history)
    RESULT['runs'][name]=r;save()
    print(json.dumps({'run':name,**{k:v for k,v in r.items() if k!='history'}}),flush=True)
    return q,r


def shear():
    n=32;nu=(.8-.5)/3;k=2*np.pi/n;steps=500
    x,y,z=np.meshgrid(*([np.arange(n)]*3),indexing='ij')
    u=np.zeros((3,n,n,n));u[0]=.01*np.cos(k*y)*np.cos(k*z)
    q=lb.quantize(lb.equilibrium(np.ones((n,n,n)),u))
    sim=lb.Simulation(q,tau=.8,cs=0,mode=MRT_MODE,device='cuda:0');sim.step(steps)
    out=sim.numpy();_,uf=lb.fields(out)
    amp=float(np.sum(uf[0]*u[0])/np.sum(u[0]**2))
    truth=float(np.exp(-2*nu*k*k*steps))
    r=dict(amplitude=amp,analytic=truth,relative_error=abs(amp/truth-1),ledger_exact=lb.ledger(q)==lb.ledger(out),failure=int(sim.failure.numpy()[0]))
    RESULT['shear_decay']=r;save();print(json.dumps({'shear_decay':r}),flush=True)
    return r['relative_error']<.02 and r['ledger_exact'] and not r['failure']


def throughput():
    n=96;steps=100
    q=lb.quantize(lb.equilibrium(np.ones((n,n,n)),np.zeros((3,n,n,n))))
    sim=lb.Simulation(q,tau=.8,mode=MRT_MODE,device='cuda:0');sim.step(2);wp.synchronize()
    durations=[]
    for _ in range(2):
        t=time.perf_counter();sim.step(steps);wp.synchronize();durations.append(time.perf_counter()-t)
    # 2x256MiB; far exceeds this device's L2. Same stream, warm buffers.
    a=wp.zeros(32*1024*1024,dtype=wp.int64,device='cuda:0');b=wp.empty_like(a)
    wp.copy(b,a);wp.synchronize();copy=[]
    for _ in range(2):
        t=time.perf_counter()
        for i in range(100):wp.copy(b,a)
        wp.synchronize();copy.append(2*a.size*8*100/(time.perf_counter()-t))
    bandwidth=[n**3*steps*304/t for t in durations]
    roof=max(copy)
    RESULT['throughput']=dict(grid=[n,n,n],steps=steps,seconds=durations,mlups=[n**3*steps/t/1e6 for t in durations],
        logical_bandwidth_GBs=[v/1e9 for v in bandwidth],copy_GBs=[v/1e9 for v in copy],
        memory_roof_fraction=[v/roof for v in bandwidth],traffic_bytes_per_update=304,
        scope='warm synchronized host-launched steps, excludes setup/readback; no graph; includes collision, quantization, streaming; copy roof only',
        failure=int(sim.failure.numpy()[0]))
    save();print(json.dumps({'throughput':RESULT['throughput']}),flush=True)


def main():
    global MRT_MODE, REPORT
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['mrt','hermite_mrt'],default='mrt')
    MRT_MODE=parser.parse_args().mode
    RESULT['preregistered']['collision']=MRT_MODE
    if MRT_MODE=='hermite_mrt':REPORT=Path('reports/lbm3d_hermite_mrt_les_v1/report.json')
    wp.init();RESULT['device']=str(wp.get_device('cuda:0'));save()
    t=time.perf_counter();warm=lb.Simulation(lb.quantize(initial(4)),tau=.8,mode=MRT_MODE,device='cuda:0');warm.step();wp.synchronize()
    RESULT['first_launch_setup_build_s']=time.perf_counter()-t;save()
    shear_ok=shear()
    _,control=run('bgk','bgk')
    a,ra=run('mrt_first',MRT_MODE);b,rb=run('mrt_repeat',MRT_MODE)
    exact=bool(np.array_equal(a,b) and ra['history']==rb['history'])
    low,rl=run('mrt_precision36',MRT_MODE,36)
    _,ua=lb.fields(a);_,ul=lb.fields(low,36)
    error=float(np.max(np.abs(ua-ul)))
    RESULT["precision_max_velocity_difference"]=error
    RESULT["full_repeat_exact"]=exact
    save()
    throughput()
    gates=dict(shear=shear_ok,bgk_blows_up=control['failed'],mrt_stable=ra['stable'] and rb['stable'],
               mass_and_momentum_exact=ra['ledger_exact'] and rb['ledger_exact'] and rl['ledger_exact'],
               full_repeat_exact=exact,precision=rl['stable'] and error<=1e-5,
               three_dimensional=ra['history'][-1]['max_uz']>1e-5,
               throughput_valid=RESULT['throughput']['failure']==0)
    RESULT.update(gates=gates,precision_max_velocity_difference=error,passed=all(gates.values()))
    save();print(json.dumps({'gates':gates,'precision_error':error,'passed':RESULT['passed']}),flush=True)
    return 0 if RESULT['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

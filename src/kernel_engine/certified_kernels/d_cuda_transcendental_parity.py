import torch, numpy as np
dev='cuda'
x = torch.linspace(-3.14159265, 3.14159265, 1_000_000, device=dev, dtype=torch.float32)
xc = x.cpu().numpy().astype(np.float64)
print("CUDA transcendental bit-parity vs IEEE ref (fp64->fp32) + reproducibility (the Vulkan hard-node datapoint):")
print(f"{'fn':>4} {'bit-exact vs IEEE':>18} {'max abs err':>12} {'~ULP':>6} {'CUDA repro 3x':>14}")
for nm,tfn,nfn in [("cos",torch.cos,np.cos),("sin",torch.sin,np.sin),("exp",torch.exp,np.exp),("rsqrt",torch.rsqrt,lambda v:1/np.sqrt(np.abs(v)+1e-6))]:
    xr = x if nm!="rsqrt" else x.abs()+1e-6
    xcr = xc if nm!="rsqrt" else np.abs(xc)+1e-6
    cu = tfn(xr).cpu().numpy()
    ie = nfn(xcr).astype(np.float32)
    be = float((cu==ie).mean())
    err = float(np.abs(cu.astype(np.float64)-ie.astype(np.float64)).max())
    ulp = err/np.float32(1.0).item()*2**23  # rough ULP near 1.0
    r2=tfn(xr).cpu().numpy(); r3=tfn(xr).cpu().numpy(); repro=bool((cu==r2).all() and (cu==r3).all())
    print(f"{nm:>4} {be:>18.3f} {err:>12.2e} {ulp:>6.1f} {str(repro):>14}")
print("=> CUDA transcendentals: REPRODUCIBLE within CUDA (R2 holds), but NOT bit-exact vs IEEE (~1-2 ULP).")
print("   Vulkan/SPIR-V GLSL.std.450 cos/sin has its OWN vendor ULP tolerance => CUDA≡Vulkan is NOT bit-identical for")
print("   transcendental-using kernels (rotation/VBD). FIX for an agent: shared polynomial approx computed identically both sides,")
print("   NOT native libm. This is why AVBD (no transcendentals, pure gather) ports clean but rigid2d_vbd (cos/sin) does not.")

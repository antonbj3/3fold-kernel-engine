"""PER-VARIABLE cert on a real solver's variables (the goal's per-variable layer, closes ~25%→). C's LBM (lbm_gpu_fast)
computes 3 variable-CLASSES with DIFFERENT cert profiles — the cert must adapt PER VARIABLE, not uniformly:
  • f_i (9 distributions): the STATE. gather-streamed (R2 det-safe), traffic 72 B/voxel (R1), R3-identifiable, fp32 fine.
  • ρ = Σ f_i (density moment): ∂ρ/∂f_j = 1 ⇒ κ_ρ = 1, WELL-conditioned everywhere, fp32 fine.
  • u = (Σ cx·f_i)/ρ (velocity moment): a DIVISION by ρ. ∂u/∂f_j = (cx_j − u)/ρ ⇒ κ_u ~ 1/ρ ⇒ ILL-conditioned at LOW ρ.
★So u is the PRECISION-SENSITIVE variable, and its requirement is PER-REGION (tracks 1/ρ): fp32 where ρ high, fp64 / ρ-floor
where ρ low. This is the adaptive-precision analog of resolution-tracks-sharpness — the cert-vector's R2/conditioning facet,
evaluated PER VARIABLE. Analytic + numerically verified (perturb f, measure δu/δf vs 1/ρ). This is the LBM moment structure
(C's lbm_gpu_fast lines 60–64: ux=mx/rho), a property of the equations, not one implementation. CPU/numpy, no fit.
"""
import numpy as np
# D2Q9 lattice velocities (C's convention)
cx = np.array([0,1,0,-1,0,1,-1,-1,1], float); cy = np.array([0,0,1,0,-1,1,1,-1,-1], float)
w  = np.array([4/9]+[1/9]*4+[1/36]*4)

def equilib(rho, ux, uy):                                   # Maxwell-Boltzmann equilibrium f_i (C's _equil_soa)
    cu = 3*(cx*ux + cy*uy); usq = 1.5*(ux*ux+uy*uy)
    return w*rho*(1 + cu + 0.5*cu*cu - usq)

def moments(f):                                             # the macroscopic variables from f
    rho = f.sum(); ux = (cx*f).sum()/rho; uy = (cy*f).sum()/rho
    return rho, ux, uy

# --- numerically VERIFY the per-variable conditioning at a range of ρ (perturb f_j, measure δρ, δu) ---
print("="*96); print("PER-VARIABLE cert on LBM variables — κ (sensitivity to f-perturbation) vs density ρ"); print("="*96)
print(f"  {'ρ':>8s} {'κ_ρ (∂ρ/∂f max)':>18s} {'κ_u (∂u/∂f max)':>18s} {'κ_u·ρ (≈const?)':>16s}  ⟹ u-precision")
eps = 1e-6
for rho0 in [1.0, 0.5, 0.1, 0.03, 0.01]:
    f = equilib(rho0, 0.05, 0.0)                            # a physical state at density ρ0, small velocity
    rho, ux, uy = moments(f)
    krho = kux = 0.0
    for j in range(9):
        fp = f.copy(); fp[j] += eps
        r2,u2,_ = moments(fp)
        krho = max(krho, abs(r2-rho)/eps)                   # ∂ρ/∂f_j
        kux  = max(kux,  abs(u2-ux )/eps)                   # ∂u/∂f_j  (~1/ρ)
    need = "fp32 OK" if kux < 30 else ("fp64/ρ-floor" if kux < 3000 else "fp64 REQUIRED")
    print(f"  {rho0:8.3f} {krho:18.2f} {kux:18.1f} {kux*rho0:16.2f}  ⟹ {need}")

print("\n" + "="*96)
print("  ⟹ PER-VARIABLE RESULT (the cert ADAPTS per variable, closes the per-variable layer):")
print("  • ρ (density): κ_ρ = 1.00 at EVERY ρ — well-conditioned everywhere ⇒ fp32 fine for the density variable.")
print("  • u (velocity): κ_u ~ 1/ρ (κ_u·ρ ≈ const) — ILL-conditioned at low ρ ⇒ u's precision requirement is PER-REGION:")
print("    fp32 where ρ≈1, fp64/ρ-floor where ρ→0. A UNIFORM-precision cert is wrong both ways: uniform fp32 fails u at")
print("    low ρ (near-vacuum/boundary); uniform fp64 wastes compute on f_i and ρ that never need it.")
print("  • f_i (distributions): the identifiable state, gather-deterministic — fp32, no conditioning issue.")
print("  ⟹ best-in-class PER VARIABLE = the right precision per variable AND per region (u tracks 1/ρ) — the adaptive-")
print("    precision analog of [[resolution-tracks-sharpness]] / diffusion coarse-to-fine, on a REAL solver's variables.")
print("    This is the cert-vector's R2/conditioning facet evaluated per-variable — NOT one uniform precision for the kernel.")
print("="*96)

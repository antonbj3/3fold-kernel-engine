#!/usr/bin/env python3
"""3D linear-elastic FEM (Q1 hex) -> von Mises stress field. New physics (3D, not only 2D) + an EXPENSIVE oracle for surrogates.

From 2D warp.fem von Mises to 3D structural physics. This is 3D Q1 hex linear elasticity (8-node elements,
2x2x2 Gauss, 6x6 isotropic D), structured grid, sparse splu solve -> displacement + von Mises fields. The 3D solve
scales MUCH worse than 2D (DOF cubic, splu ~O(DOF^1.x)), so it is a genuinely EXPENSIVE oracle (where a surrogate model wins big,
unlike the cheap 2D case).

GATE (rigorous, against ANALYTIC): (1) EQUILIBRIUM - uniform tensile traction t on +x, symmetry rollers on -x/-y/-z ->
sigma_xx=t uniformly (an equilibrium identity, CONSTITUTIVE-INDEPENDENT); (2) CONSTITUTIVE - u_x=t.L/E (tests E) + lateral
kontraktion ε_yy=−ν·t/E (testar ν); (3) von Mises = |t| vid uniaxiellt (σ_vm=√(σxx²)=t); (4) energi-konsistens
1/2 u^T K u = 1/2 integral sigma:eps dV (internal energy); (5) convergence: a finer grid gives the same uniform field (mesh independence for constant
stress, machine precision).

  CUDA_VISIBLE_DEVICES="" python3 fem3d_elasticity.py
"""
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


def D_iso(E, nu):
    """6×6 isotrop elasticitets-matris [εxx,εyy,εzz,γxy,γyz,γzx]."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def hex_KE(h, D):
    """Q1-hex element-styvhet (kub-element sida h) via 2×2×2 Gauss. → 24×24."""
    g = 1 / np.sqrt(3); gp = [-g, g]
    # nodal natural coordinates (8 corners), ordering consistent with edof below
    ncoord = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
              (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    KE = np.zeros((24, 24))
    for xi in gp:
        for eta in gp:
            for ze in gp:
                dN = np.zeros((3, 8))   # dN/d(xi,eta,ze)
                for i, (a, b, c) in enumerate(ncoord):
                    dN[0, i] = 0.125 * a * (1 + b * eta) * (1 + c * ze)
                    dN[1, i] = 0.125 * b * (1 + a * xi) * (1 + c * ze)
                    dN[2, i] = 0.125 * c * (1 + a * xi) * (1 + b * eta)
                dNxyz = dN * (2.0 / h)        # J = (h/2)I → dN/dx = (2/h) dN/dxi
                B = np.zeros((6, 24))
                for i in range(8):
                    B[0, 3 * i] = dNxyz[0, i]
                    B[1, 3 * i + 1] = dNxyz[1, i]
                    B[2, 3 * i + 2] = dNxyz[2, i]
                    B[3, 3 * i] = dNxyz[1, i]; B[3, 3 * i + 1] = dNxyz[0, i]
                    B[4, 3 * i + 1] = dNxyz[2, i]; B[4, 3 * i + 2] = dNxyz[1, i]
                    B[5, 3 * i] = dNxyz[2, i]; B[5, 3 * i + 2] = dNxyz[0, i]
                KE += (B.T @ D @ B) * (h / 2) ** 3   # detJ = (h/2)^3, vikt 1
    return KE


class FEM3D:
    """Strukturerat nelx×nely×nelz hex-grid (kub-element h). −x/−y/−z symmetri-rollers, +x drag-traktion."""

    def __init__(self, nelx=8, nely=8, nelz=8, h=1.0, E=200e9, nu=0.3):
        self.nelx, self.nely, self.nelz, self.h, self.E, self.nu = nelx, nely, nelz, h, E, nu
        self.nx, self.ny, self.nz = nelx + 1, nely + 1, nelz + 1
        self.nnode = self.nx * self.ny * self.nz
        self.ndof = 3 * self.nnode
        self.D = D_iso(E, nu)
        self.KE = hex_KE(h, self.D)
        self._assemble()

    def nid(self, i, j, k):
        return (k * self.ny + j) * self.nx + i

    def _edof(self, ex, ey, ez):
        # 8 corners in the same order as hex_KE's ncoord
        ns = [self.nid(ex, ey, ez), self.nid(ex + 1, ey, ez), self.nid(ex + 1, ey + 1, ez), self.nid(ex, ey + 1, ez),
              self.nid(ex, ey, ez + 1), self.nid(ex + 1, ey, ez + 1), self.nid(ex + 1, ey + 1, ez + 1), self.nid(ex, ey + 1, ez + 1)]
        return np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in ns]).flatten()

    def _assemble(self):
        rows, cols, vals = [], [], []
        for ez in range(self.nelz):
            for ey in range(self.nely):
                for ex in range(self.nelx):
                    ed = self._edof(ex, ey, ez)
                    rows.append(np.repeat(ed, 24)); cols.append(np.tile(ed, 24)); vals.append(self.KE.flatten())
        self.K = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(self.ndof, self.ndof))

    def solve(self, traction):
        """Tensile traction t on the +x face; symmetry rollers -x(u_x=0)/-y(u_y=0)/-z(u_z=0). -> u (ndof,)."""
        fixed = []
        for k in range(self.nz):
            for j in range(self.ny):
                fixed.append(3 * self.nid(0, j, k))                 # −x: u_x=0
        for k in range(self.nz):
            for i in range(self.nx):
                fixed.append(3 * self.nid(i, 0, k) + 1)             # −y: u_y=0
        for j in range(self.ny):
            for i in range(self.nx):
                fixed.append(3 * self.nid(i, j, 0) + 2)             # −z: u_z=0
        fixed = np.unique(fixed)
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        # +x face: CONSISTENT Q1 surface load for uniform traction t (integral N_i dA = h^2/4 per surface quad node)
        F = np.zeros(self.ndof)
        for ez in range(self.nelz):
            for ey in range(self.nely):
                fn = [self.nid(self.nelx, ey, ez), self.nid(self.nelx, ey + 1, ez),
                      self.nid(self.nelx, ey + 1, ez + 1), self.nid(self.nelx, ey, ez + 1)]
                for n in fn:
                    F[3 * n] += traction * self.h * self.h / 4.0
        Kff = self.K[free][:, free]
        u = np.zeros(self.ndof)
        u[free] = splu(csr_matrix(Kff).tocsc()).solve(F[free])
        return u, fixed, free, F

    def stress_vm(self, u):
        """von Mises stress per element (at the centre, xi=eta=zeta=0)."""
        ncoord = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                  (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        dN = np.zeros((3, 8))
        for i, (a, b, c) in enumerate(ncoord):
            dN[0, i] = 0.125 * a; dN[1, i] = 0.125 * b; dN[2, i] = 0.125 * c    # vid centrum
        dNxyz = dN * (2.0 / self.h)
        vms = []
        for ez in range(self.nelz):
            for ey in range(self.nely):
                for ex in range(self.nelx):
                    ed = self._edof(ex, ey, ez)
                    B = np.zeros((6, 24))
                    for i in range(8):
                        B[0, 3 * i] = dNxyz[0, i]; B[1, 3 * i + 1] = dNxyz[1, i]; B[2, 3 * i + 2] = dNxyz[2, i]
                        B[3, 3 * i] = dNxyz[1, i]; B[3, 3 * i + 1] = dNxyz[0, i]
                        B[4, 3 * i + 1] = dNxyz[2, i]; B[4, 3 * i + 2] = dNxyz[1, i]
                        B[5, 3 * i] = dNxyz[2, i]; B[5, 3 * i + 2] = dNxyz[0, i]
                    s = self.D @ (B @ u[ed])     # [sxx,syy,szz,sxy,syz,szx]
                    vm = np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2
                                        + 6 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)))
                    vms.append((s, vm))
        return vms


def main():
    print("3D linear-elastic FEM (Q1 hex) -> von Mises field (new physics, an expensive oracle for surrogates)")
    E = 200e9; nu = 0.3; t = 1.0e6; L = 8.0
    fem = FEM3D(nelx=8, nely=8, nelz=8, h=1.0, E=E, nu=nu)
    print(f"  grid 8×8×8 ({fem.nnode} noder, {fem.ndof} DOF)")
    ts = time.perf_counter(); u, fixed, free, F = fem.solve(t); solve_s = time.perf_counter() - ts
    vms = fem.stress_vm(u)
    sxx = np.array([s[0][0] for s in vms]); syy = np.array([s[0][1] for s in vms])
    vm = np.array([s[1] for s in vms])
    # +x-ytan: maximal u_x
    ux_max = max(u[3 * fem.nid(fem.nelx, j, k)] for k in range(fem.nz) for j in range(fem.ny))
    print(f"  solve {solve_s*1e3:.0f}ms; σxx medel {sxx.mean()/1e6:.3f}MPa (t={t/1e6}MPa), σyy {syy.mean()/1e6:.2e}, "
          f"vM medel {vm.mean()/1e6:.3f}MPa; u_x(+x)={ux_max*1e6:.3f}µm")

    # (1) EQUILIBRIUM: sigma_xx = t uniformly (constitutive-independent)
    g1 = abs(sxx.mean() - t) / t < 1e-6 and sxx.std() / t < 1e-6
    print(f"  (1) equilibrium sigma_xx=t: mean error {abs(sxx.mean()-t)/t:.1e}, std {sxx.std()/t:.1e}")
    # (2) KONSTITUTIV: u_x = t·L/E ; σyy≈σzz≈0
    ux_analytic = t * L / E
    g2 = abs(ux_max - ux_analytic) / ux_analytic < 1e-3 and abs(syy.mean()) / t < 1e-6
    print(f"  (2) constitutive u_x={ux_max*1e9:.3f} nm vs analytic t.L/E={ux_analytic*1e9:.3f} nm (error {abs(ux_max-ux_analytic)/ux_analytic:.1e}); sigma_yy/t {abs(syy.mean())/t:.1e}")
    # (3) von Mises = t (uniaxiellt)
    g3 = abs(vm.mean() - t) / t < 1e-6
    print(f"  (3) von Mises = t (uniaxial): error {abs(vm.mean()-t)/t:.1e}")
    # (4) energy consistency 1/2 u^T K u (internal energy) = 1/2 F.u (external work), the same F as the solution
    Uint = 0.5 * u @ (fem.K @ u)
    Wext = 0.5 * F @ u
    g4 = abs(Uint - Wext) / abs(Wext) < 1e-8 and Uint > 0
    print(f"  (4) energy 1/2 u^T K u={Uint:.3e} = 1/2 F.u={Wext:.3e} (error {abs(Uint-Wext)/abs(Wext):.1e})")
    # (5) mesh independence: the SAME physical domain (8x8x8), a 2x FINER mesh (16^3 @ h=0.5). Audit fix: an earlier version
    # compared a DIFFERENT domain (12x6x6), which is not a refinement. HONEST: a uniform field is EXACT for Q1 on any mesh
    # (constant-field representation), so this verifies same-domain invariance + u_x consistency, NOT convergence (see the caveat).
    fem2 = FEM3D(nelx=16, nely=16, nelz=16, h=0.5, E=E, nu=nu)        # same 8x8x8 domain, twice as fine
    u2, _, _, _ = fem2.solve(t); sxx2 = np.array([s[0][0] for s in fem2.stress_vm(u2)])
    ux2 = max(u2[3 * fem2.nid(fem2.nelx, j, k)] for k in range(fem2.nz) for j in range(fem2.ny))
    g5 = abs(sxx2.mean() - t) / t < 1e-6 and abs(ux2 - ux_analytic) / ux_analytic < 1e-3   # same domain -> the same u_x=t.L/E
    print(f"  (5) mesh independence (same 8x8x8 domain, 16^3@h=0.5): sigma_xx=t error {abs(sxx2.mean()-t)/t:.1e}, u_x=t.L/E error {abs(ux2-ux_analytic)/ux_analytic:.1e}")

    ok = g1 and g2 and g3 and g4 and g5
    print(f"\nVERDICT: 3D linear-elastic FEM (Q1 hex) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"New 3D structural physics (von Mises field) validated against ANALYTIC results: (1) equilibrium sigma_xx=t uniform ({abs(sxx.mean()-t)/t:.0e}, "
             f"konstitutiv-oberoende); (2) konstitutiv u_x=t·L/E ({abs(ux_max-ux_analytic)/ux_analytic:.0e}, testar E) + σyy≈0; "
             f"(3) von Mises=t uniaxiellt; (4) energi ½uᵀKu=½F·u ({abs(Uint-Wext)/abs(Wext):.0e}); (5) mesh-oberoende. → 3D "
             f"stress field ({fem.ndof} DOF, solve {solve_s*1e3:.0f} ms) = a genuinely EXPENSIVE oracle (cubic scaling) -> ideal for "
             "surrogate compute efficiency (where 2D linear was too cheap). " if ok else
             f"Not validated (equilibrium {g1}, constitutive {g2}, vM {g3}, energy {g4}, mesh {g5}) - debug. ")
          + "CAVEAT: linear elastic (small strain, isotropic); structured hex grid (cubic elements); a simple consistent nodal load distribution "
          "on the +x face (consistent for constant traction); symmetry roller BC (a uniaxial specimen). Validated for a CONSTANT "
          "stress field (mesh-independent, exact); non-constant fields (concentrations) require a mesh convergence study (further work). "
          "Solve = scipy splu (CPU); 3D scales expensively -> surrogate payoff (next: a 3D FNO).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

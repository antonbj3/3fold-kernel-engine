"""Axisymmetric incompressible Navier-Stokes (no swirl), MAC staggered grid, Chorin projection,
pseudo-transient march to steady. Prefactored sparse pressure-Poisson (splu, geometry fixed → factor once).

Coordinates: z (axial), r (radial, r>=0, axis at r=0). Variables u_z (w), u_r (v), p.
Staggered (MAC):
  p[i,j]  cell centers,  z=zmin+(i+0.5)dz, r=(j+0.5)dr   i in[0,Nz) j in[0,Nr)
  w[i,j]  u_z on z-faces, z=zmin+i*dz,     r=(j+0.5)dr   i in[0,Nz] j in[0,Nr)   (left face of cell i)
  v[i,j]  u_r on r-faces, z=zmin+(i+0.5)dz, r=j*dr        i in[0,Nz) j in[0,Nr]   (bottom face of cell j; j=0 axis)
Geometry: a wall radius function R(z); cell (i,j) is FLUID iff r_c[j] < R(z_c[i]).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class AxisymNS:
    def __init__(self, zmin, zmax, Rmax, dz, dr, rho, mu, Rfun):
        self.dz, self.dr, self.rho, self.mu, self.nu = dz, dr, rho, mu, mu / rho
        self.Nz = int(round((zmax - zmin) / dz))
        self.Nr = int(round(Rmax / dr))
        self.zmin, self.Rmax = zmin, Rmax
        Nz, Nr = self.Nz, self.Nr
        self.zc = zmin + (np.arange(Nz) + 0.5) * dz
        self.rc = (np.arange(Nr) + 0.5) * dr
        self.rf = np.arange(Nr + 1) * dr               # r at r-faces (j=0 -> 0 axis)
        self.Rwall = Rfun(self.zc)                      # wall radius at each axial cell column
        # fluid mask on cells
        self.fluid = self.rc[None, :] < self.Rwall[:, None]    # (Nz, Nr)
        # fields
        self.w = np.zeros((Nz + 1, Nr))
        self.v = np.zeros((Nz, Nr + 1))
        self.p = np.zeros((Nz, Nr))
        self.Rfun = Rfun
        self._build_poisson()

    # ---- pressure Poisson: axisymmetric FV Laplacian on fluid cells, Neumann at walls/inlet, Dirichlet p=0 at outlet ----
    def _build_poisson(self):
        Nz, Nr, dz, dr, rc, rf = self.Nz, self.Nr, self.dz, self.dr, self.rc, self.rf
        fluid = self.fluid
        idx = -np.ones((Nz, Nr), dtype=int)
        ids = np.where(fluid)
        idx[ids] = np.arange(ids[0].size)
        self.idx = idx
        self.nfl = ids[0].size
        rows, cols, vals = [], [], []
        b_outlet = []   # cells flagged as outlet-adjacent (Dirichlet ref via ghost p=0)
        for k in range(self.nfl):
            i, j = ids[0][k], ids[1][k]
            diag = 0.0
            # axial neighbours (z): coeff 1/dz^2
            for di, nb in ((-1, 'W'), (1, 'E')):
                ii = i + di
                if 0 <= ii < Nz and fluid[ii, j]:
                    rows.append(k); cols.append(idx[ii, j]); vals.append(1.0 / dz**2); diag -= 1.0 / dz**2
                else:
                    # boundary in z: inlet (i==0 side) Neumann -> no term; outlet (i==Nz-1 side) Dirichlet p=0
                    if ii >= Nz:   # outlet face -> ghost p=0
                        diag -= 1.0 / dz**2     # Dirichlet: ghost contributes -1/dz^2 to diag, +0 to rhs
                        b_outlet.append(k)
                    # inlet/solid-in-z -> Neumann (drop)
            # radial neighbours (conservative axisym): (1/rc) * rf_face * (p_nb - p)/dr^2
            # top face at rf[j+1], bottom face at rf[j]
            cT = rf[j + 1] / (rc[j] * dr**2)
            cB = rf[j] / (rc[j] * dr**2)
            # top neighbour j+1
            if j + 1 < Nr and fluid[i, j + 1]:
                rows.append(k); cols.append(idx[i, j + 1]); vals.append(cT); diag -= cT
            # else: wall (Neumann, drop) ; outer boundary wall
            # bottom neighbour j-1 (j=0 -> axis, rf[0]=0 -> cB=0 naturally Neumann/symmetry)
            if j - 1 >= 0 and fluid[i, j - 1]:
                rows.append(k); cols.append(idx[i, j - 1]); vals.append(cB); diag -= cB
            rows.append(k); cols.append(k); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(self.nfl, self.nfl))
        self.Alu = spla.splu(A.tocsc())
        self.b_outlet = np.array(b_outlet, dtype=int)

    def set_inlet_parabolic(self, Q):
        """Fully-developed parabolic u_z at inlet plane i=0 (developed pipe flow in the inlet tube)."""
        R0 = self.Rfun(np.array([self.zmin + 0.5 * self.dz]))[0]
        Umean = Q / (np.pi * R0**2)
        prof = 2.0 * Umean * (1.0 - (self.rc / R0)**2)
        prof = np.where(self.rc < R0, prof, 0.0)
        self.w[0, :] = prof
        self.Q, self.R0, self.Umean = Q, R0, Umean

    def _wface_fluid(self):
        # w-face [i,j] fluid if both adjacent cells fluid (interior i in 1..Nz-1); inlet i=0 prescribed; outlet i=Nz free
        Nz = self.Nz
        wf = np.zeros((Nz + 1, self.Nr), dtype=bool)
        wf[1:Nz, :] = self.fluid[:-1, :] & self.fluid[1:, :]
        return wf

    def _vface_fluid(self):
        Nr = self.Nr
        vf = np.zeros((self.Nz, Nr + 1), dtype=bool)
        vf[:, 1:Nr] = self.fluid[:, :-1] & self.fluid[:, 1:]   # interior r-faces; j=0 axis & j=Nr outer stay False(=0)
        return vf

    def step(self, dt):
        Nz, Nr, dz, dr, nu, rho = self.Nz, self.Nr, self.dz, self.dr, self.nu, self.rho
        rc, rf = self.rc, self.rf
        w, v, p = self.w, self.v, self.p
        wf, vf = self._wface_fluid(), self._vface_fluid()

        # ---------- tentative w (u_z) on interior z-faces ----------
        Aw = np.zeros_like(w)
        i = np.arange(1, Nz)              # interior z-faces
        # viscous: d2/dz2 + (1/rc) d/dr(rc dw/dr)
        wW = w[0:Nz - 1, :]; wE = w[2:Nz + 1, :]; wC = w[1:Nz, :]
        lap_z = (wE - 2 * wC + wW) / dz**2
        # radial: neighbours in j; wall -> w=0 (Dirichlet at solid face center)
        wN = np.zeros_like(wC); wS = np.zeros_like(wC)
        wN[:, :Nr - 1] = w[1:Nz, 1:Nr]; wS[:, 1:Nr] = w[1:Nz, 0:Nr - 1]
        # mask solid radial neighbours to 0 (no-slip) using wf neighbour test
        nbN = np.zeros_like(wC, dtype=bool); nbS = np.zeros_like(wC, dtype=bool)
        nbN[:, :Nr - 1] = wf[1:Nz, 1:Nr]; nbS[:, 1:Nr] = wf[1:Nz, 0:Nr - 1]
        # viscous uses MIRROR ghost at walls (no-slip = Dirichlet-0 AT the face, not at ghost centre); advection uses 0-fill
        wN_v = np.where(nbN, wN, -wC); wS_v = np.where(nbS, wS, -wC)
        wN_a = np.where(nbN, wN, 0.0); wS_a = np.where(nbS, wS, 0.0)
        rfN = rf[1:Nr + 1][None, :]; rfS = rf[0:Nr][None, :]; rcb = rc[None, :]
        lap_r = (rfN * (wN_v - wC) - rfS * (wC - wS_v)) / (rcb * dr**2)
        visc = nu * (lap_z + lap_r)
        # advection (upwind, non-conservative): u_z dw/dz + u_r dw/dr
        uz = wC
        # u_r at w-location: avg of 4 v
        ur = 0.25 * (v[0:Nz - 1, 0:Nr] + v[0:Nz - 1, 1:Nr + 1] + v[1:Nz, 0:Nr] + v[1:Nz, 1:Nr + 1])
        dwdz = np.where(uz > 0, (wC - wW) / dz, (wE - wC) / dz)
        dwdr = np.where(ur > 0, (wC - wS_a) / dr, (wN_a - wC) / dr)
        adv = uz * dwdz + ur * dwdr
        Aw[1:Nz, :] = visc - adv
        w_star = w.copy()
        w_star[1:Nz, :] = w[1:Nz, :] + dt * Aw[1:Nz, :]
        w_star *= wf                          # zero on non-fluid faces (walls)
        w_star[0, :] = w[0, :]                 # inlet fixed
        w_star[Nz, :] = w_star[Nz - 1, :] * self.fluid[Nz - 1, :]   # outlet zero-gradient

        # ---------- tentative v (u_r) on interior r-faces ----------
        Av = np.zeros_like(v)
        j = np.arange(1, Nr)
        vC = v[:, 1:Nr]
        vW = np.zeros_like(vC); vE = np.zeros_like(vC)
        vW[1:Nz, :] = v[0:Nz - 1, 1:Nr]; vE[0:Nz - 1, :] = v[1:Nz, 1:Nr]
        nbW = np.zeros_like(vC, dtype=bool); nbE = np.zeros_like(vC, dtype=bool)
        nbW[1:Nz, :] = vf[0:Nz - 1, 1:Nr]; nbE[0:Nz - 1, :] = vf[1:Nz, 1:Nr]
        vW_v = np.where(nbW, vW, -vC); vE_v = np.where(nbE, vE, -vC)     # mirror for viscous
        vW_a = np.where(nbW, vW, 0.0); vE_a = np.where(nbE, vE, 0.0)     # 0-fill for advection
        vN = v[:, 2:Nr + 1]; vS = v[:, 0:Nr - 1]
        nbN2 = vf[:, 2:Nr + 1]; nbS2 = vf[:, 0:Nr - 1]
        vN_v = np.where(nbN2, vN, -vC); vS_v = np.where(nbS2, vS, -vC)
        vN_a = np.where(nbN2, vN, 0.0); vS_a = np.where(nbS2, vS, 0.0)
        rfj = rf[1:Nr][None, :]
        lap_z_v = (vE_v - 2 * vC + vW_v) / dz**2
        lap_r_v = (vN_v - 2 * vC + vS_v) / dr**2 + (1.0 / rfj) * (vN_v - vS_v) / (2 * dr) - vC / rfj**2
        visc_v = nu * (lap_z_v + lap_r_v)
        # advection of v
        uz_v = 0.25 * (w[0:Nz, 0:Nr - 1] + w[1:Nz + 1, 0:Nr - 1] + w[0:Nz, 1:Nr] + w[1:Nz + 1, 1:Nr])
        ur_v = vC
        dvdz = np.where(uz_v > 0, (vC - vW_a) / dz, (vE_a - vC) / dz)
        dvdr = np.where(ur_v > 0, (vC - vS_a) / dr, (vN_a - vC) / dr)
        adv_v = uz_v * dvdz + ur_v * dvdr
        Av[:, 1:Nr] = visc_v - adv_v
        v_star = v.copy()
        v_star[:, 1:Nr] = v[:, 1:Nr] + dt * Av[:, 1:Nr]
        v_star *= vf

        # ---------- pressure projection ----------
        div = (w_star[1:Nz + 1, :] - w_star[0:Nz, :]) / dz \
            + (rf[1:Nr + 1][None, :] * v_star[:, 1:Nr + 1] - rf[0:Nr][None, :] * v_star[:, 0:Nr]) / (rc[None, :] * dr)
        rhs = (rho / dt) * div[self.fluid]
        pvec = self.Alu.solve(rhs)
        p_new = np.zeros((Nz, Nr))
        p_new[self.fluid] = pvec
        self.p = p_new
        # correct velocities: u = u* - (dt/rho) grad p
        gradp_z = np.zeros_like(w)
        gradp_z[1:Nz, :] = (p_new[1:Nz, :] - p_new[0:Nz - 1, :]) / dz
        w_new = w_star - (dt / rho) * gradp_z
        w_new *= wf; w_new[0, :] = w[0, :]
        w_new[Nz, :] = w_new[Nz - 1, :] * self.fluid[Nz - 1, :]
        gradp_r = np.zeros_like(v)
        gradp_r[:, 1:Nr] = (p_new[:, 1:Nr] - p_new[:, 0:Nr - 1]) / dr
        v_new = v_star - (dt / rho) * gradp_r
        v_new *= vf
        dwmax = np.max(np.abs(w_new - w)); dvmax = np.max(np.abs(v_new - v))
        self.w, self.v = w_new, v_new
        return max(dwmax, dvmax)

    def solve_steady(self, Q, dt, max_steps=40000, tol=1e-7, verbose=False):
        self.set_inlet_parabolic(Q)
        hist = []
        for n in range(max_steps):
            res = self.step(dt)
            if n % 500 == 0:
                hist.append((n, res))
                if verbose:
                    print(f"  step {n:6d}  res={res:.3e}  umax={self.w.max():.4f}")
            if res < tol * max(self.Umean, 1e-9) and n > 1000:
                hist.append((n, res)); break
        self.steps_used = n
        return hist

    # ---- diagnostics ----
    def centerline_uz(self):
        """u_z on the axis (r~0): average the two near-axis w-rows weighted, take j=0 cell (rc[0]=dr/2)."""
        return self.zc, self.w[:-1, 0]   # w left-face per cell, axis column

    def axis_pressure(self):
        return self.zc, self.p[:, 0]

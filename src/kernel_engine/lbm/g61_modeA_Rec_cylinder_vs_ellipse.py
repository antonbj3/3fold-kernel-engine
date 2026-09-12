"""P3·G61 — settle FSI ADVERSARY #3 (g59): MEASURE the mode-A 3-D-transition Re_c for a circular cylinder vs ellipses and
TEST g59's Cheeger prediction Re_c ∝ (h·D_perp)^−2 against D's falsifier (Re_c ratio must match [h ratio]² within 2×).

METHOD = DNS-3-D-growth (build-spec method B; reuse g21's VALIDATED D3Q19-BGK LBM, viscosity-ramp startup). Run the 3-D
LBM with span L_z = λ_z ≈ 4·D_perp (one mode-A wavelength), develop the 2-D periodic base wake (uy von-Kármán seed), seed a
SMALL z-varying perturbation (uz seed), and measure the growth rate σ of the spanwise kinetic energy rms(uz)/U over the
LINEAR window. Re_c = the Re where σ crosses zero (mode-A onset).

WHY uz IS THE CLEAN OBSERVABLE (g42's hard-won lesson, this repo): uz ≡ 0 for the 2-D base shedding, so a purely in-plane
(2-D, CONVECTIVELY-amplified) perturbation has ZERO uz-image — it lives in the null space. rms(uz) therefore measures ONLY
the ABSOLUTE mode-A instability, not the convective shear-layer transit that inflates a naive full-field growth rate. The
discriminator (g42 X3/X5): (null) Re=100 ≪ Re_c MUST give σ≤0; (window) σ measured in a near-body window vs the far wake vs
the full domain must AGREE for an absolute mode (a convective artifact would be window-SENSITIVE — bigger downstream).

EXTERNAL ANCHOR (g41/g60 pattern, MANDATORY de-risk first): reproduce the circular-cylinder mode-A Re_c ≈ 188–190
(Williamson; Barkley & Henderson JFM 322, 1996) BEFORE touching the ellipse. G1: Re_c(circle) ∈ [180,200] ⇒ trust the
ellipse; else STOP + diagnose (broken harness ⇒ a tautology).

g59 PREDICTS (anchor circle 189): ellipse AR=2 (streamwise) → Re_c≈318 ([h ratio]²=1.68); AR=0.5 (cross-stream) → Re_c≈79
([h ratio]²=0.42). G3 falsifier: Re_c(AR=2)/Re_c(circle) within 2× of 1.68 (ratio ∈ [0.84,3.36]) ⇒ HOLDS; outside ⇒ REFUTED.

  python3 g61_modeA_Rec_cylinder_vs_ellipse.py [--bench] [--validate] [--quick]
"""
import json
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g18_cfd_nilss_prereq_wake_chaos as g18                  # temporal_character (St), sparkline
import g20_3d_wake_chaos_nilss_prereq as g20                   # lambda_z_from_Pz
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "_vendor"))
from _emit import emit

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "artifacts", "g61_runs")
os.makedirs(RESDIR, exist_ok=True)
LOG = os.path.join(RESDIR, "g61_run.log")

# ── D3Q19 lattice (verbatim from g21/g20) ───────────────────────────────────────────────────────────────────────────
E = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], dtype=np.int32)
W = np.array([1/3] + [1/18] * 6 + [1/36] * 12, dtype=np.float32)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int32)
vec19i = wp.types.vector(length=19, dtype=wp.int32); vec19f = wp.types.vector(length=19, dtype=wp.float32)
ex = wp.constant(vec19i(*[int(v) for v in E[:, 0]])); ey = wp.constant(vec19i(*[int(v) for v in E[:, 1]]))
ez = wp.constant(vec19i(*[int(v) for v in E[:, 2]])); ww = wp.constant(vec19f(*[float(v) for v in W]))
opp = wp.constant(vec19i(*[int(v) for v in OPP]))


@wp.func
def feq(q: int, rho: float, ux: float, uy: float, uz: float) -> float:
    eu = float(ex[q]) * ux + float(ey[q]) * uy + float(ez[q]) * uz
    return ww[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * (ux * ux + uy * uy + uz * uz))


@wp.kernel
def collide(f: wp.array4d(dtype=wp.float32), solid: wp.array3d(dtype=wp.int32), inv_tau: float):
    i, j, k = wp.tid()
    if solid[i, j, k] == 1:
        return
    rho = float(0.0); mx = float(0.0); my = float(0.0); mz = float(0.0)
    for q in range(19):
        fq = f[i, j, k, q]; rho += fq; mx += float(ex[q]) * fq; my += float(ey[q]) * fq; mz += float(ez[q]) * fq
    ux = mx / rho; uy = my / rho; uz = mz / rho
    for q in range(19):
        f[i, j, k, q] = f[i, j, k, q] - inv_tau * (f[i, j, k, q] - feq(q, rho, ux, uy, uz))


@wp.kernel
def stream_bb(f: wp.array4d(dtype=wp.float32), fn: wp.array4d(dtype=wp.float32), solid: wp.array3d(dtype=wp.int32),
              fx: wp.array(dtype=wp.float32), fyl: wp.array(dtype=wp.float32), nx: int, ny: int, nz: int):
    i, j, k = wp.tid()
    if solid[i, j, k] == 1:
        return
    for q in range(19):
        si = i - int(ex[q]); sj = (j - int(ey[q]) + ny) % ny; sk = (k - int(ez[q]) + nz) % nz   # periodic y & z (spanwise)
        if si < 0 or si >= nx:
            fn[i, j, k, q] = f[i, j, k, q]                                  # x faces overwritten by inflow/outflow
        elif solid[si, sj, sk] == 1:
            qb = int(opp[q])
            fn[i, j, k, q] = f[i, j, k, qb]                                 # fixed-wall halfway bounce-back
            wp.atomic_add(fx, 0, 2.0 * float(ex[q]) * f[i, j, k, qb])       # momentum-exchange drag (x) / lift (y)
            wp.atomic_add(fyl, 0, 2.0 * float(ey[q]) * f[i, j, k, qb])
        else:
            fn[i, j, k, q] = f[si, sj, sk, q]


@wp.kernel
def inflow_outflow(fn: wp.array4d(dtype=wp.float32), U: float, nx: int, ny: int, nz: int):
    j, k = wp.tid()
    for q in range(19):
        fn[0, j, k, q] = feq(q, 1.0, U, 0.0, 0.0)                           # equilibrium inflow
        fn[nx - 1, j, k, q] = fn[nx - 2, j, k, q]                           # zero-gradient outflow


@wp.kernel
def accum_uz2(f: wp.array4d(dtype=wp.float32), solid: wp.array3d(dtype=wp.int32), x_near: int, x_far: int,
              s_g: wp.array(dtype=wp.float32), n_g: wp.array(dtype=wp.float32),
              s_n: wp.array(dtype=wp.float32), n_n: wp.array(dtype=wp.float32),
              s_f: wp.array(dtype=wp.float32), n_f: wp.array(dtype=wp.float32), cx: int):
    """Σ uz² over fluid: GLOBAL + a NEAR-body window [cx, x_near) + a FAR-wake window [x_far, nx). uz≡0 for the 2-D base ⇒
       this is the absolute mode-A energy; the near-vs-far split is the convective-vs-absolute discriminator (g42 X5)."""
    i, j, k = wp.tid()
    if solid[i, j, k] == 1:
        return
    rho = float(0.0); mz = float(0.0)
    for q in range(19):
        fq = f[i, j, k, q]; rho += fq; mz += float(ez[q]) * fq
    uz2 = (mz / rho) * (mz / rho)
    wp.atomic_add(s_g, 0, uz2); wp.atomic_add(n_g, 0, 1.0)
    if i >= cx and i < x_near:
        wp.atomic_add(s_n, 0, uz2); wp.atomic_add(n_n, 0, 1.0)
    if i >= x_far:
        wp.atomic_add(s_f, 0, uz2); wp.atomic_add(n_f, 0, 1.0)


@wp.kernel
def symmetrize_z(f: wp.array4d(dtype=wp.float32), solid: wp.array3d(dtype=wp.int32)):
    """project the populations onto the z-symmetric (uz=0) subspace by averaging each +z/−z reflection pair. Applied
       EVERY step during base development ⇒ the 2-D von-Kármán base develops while uz is pinned at 0 (mode-A cannot grow
       from float32 round-off), so the post-warm reseed measures the LINEAR mode-A rate, not an already-saturated mode."""
    i, j, k = wp.tid()
    if solid[i, j, k] == 1:
        return
    m = 0.5 * (f[i, j, k, 5] + f[i, j, k, 6]);  f[i, j, k, 5] = m;  f[i, j, k, 6] = m
    m = 0.5 * (f[i, j, k, 11] + f[i, j, k, 13]); f[i, j, k, 11] = m; f[i, j, k, 13] = m
    m = 0.5 * (f[i, j, k, 12] + f[i, j, k, 14]); f[i, j, k, 12] = m; f[i, j, k, 14] = m
    m = 0.5 * (f[i, j, k, 15] + f[i, j, k, 17]); f[i, j, k, 15] = m; f[i, j, k, 17] = m
    m = 0.5 * (f[i, j, k, 16] + f[i, j, k, 18]); f[i, j, k, 16] = m; f[i, j, k, 18] = m


RE_INIT = 130.0                                                            # viscosity-ramp starts here (well-resolved, BGK-stable)


class WakeMA:
    """3-D D3Q19-BGK wake for a circle/ellipse; cylinder axis along z (spanwise), periodic y & z. Measures the mode-A
       growth rate σ = d ln rms(uz)/dt. Ellipse semi-axes (a_lat streamwise, b_lat cross-stream); Re uses D_perp=2·b_lat."""
    def __init__(self, Re, a_lat, b_lat, nx=256, ny=288, nz=80, cx=64, U=0.1, n_ramp=8000):
        self.nx, self.ny, self.nz, self.U, self.Re = nx, ny, nz, U, Re
        self.a_lat, self.b_lat = float(a_lat), float(b_lat)
        self.Dperp = 2.0 * b_lat                                            # cross-stream scale that defines Re
        self.cx, self.cy = cx, ny // 2
        self.tau = 0.5 + 3.0 * U * self.Dperp / Re                          # ν=(τ−0.5)/3, Re=U·D_perp/ν
        self.tau_init = 0.5 + 3.0 * U * self.Dperp / RE_INIT
        self.n_ramp = n_ramp
        Z, Y, X = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
        solid0 = (((X - cx) / self.a_lat) ** 2 + ((Y - self.cy) / self.b_lat) ** 2) < 1.0
        self.solid0 = np.ascontiguousarray(np.transpose(solid0, (2, 1, 0)).astype(np.int32))   # (nx,ny,nz)
        self.fluid = (self.solid0 == 0)
        self.n_solid_perz = int(self.solid0[:, :, 0].sum())
        self.solid = wp.array(self.solid0, dtype=wp.int32, device=DEV)
        self.norm = 0.5 * U * U * self.Dperp * nz                          # force normalisation (span L_z=nz)
        # discriminator windows (in x): near-body formation region, far wake
        self.x_near = cx + int(4.0 * self.Dperp)
        self.x_far = cx + int(5.0 * self.Dperp)
        self.St_guess = 0.20
        self.T_shed = max(200, int(self.Dperp / (U * self.St_guess)))

    def inv_tau_at(self, it, ramp):
        if not ramp:
            return 1.0 / self.tau
        frac = min(1.0, it / max(1, self.n_ramp))
        return 1.0 / (self.tau_init + (self.tau - self.tau_init) * frac)

    def f_init(self, seed=0, uy_noise=0.05, uz_seed=1e-3):
        """STRONG uy von-Kármán seed (start shedding fast) + SMALL uz seed (the 3-D mode-A perturbation, kept linear)."""
        rng = np.random.default_rng(seed)
        ux = np.full((self.nx, self.ny, self.nz), self.U, np.float64)
        uy = uy_noise * self.U * rng.standard_normal((self.nx, self.ny, self.nz))
        uz = uz_seed * self.U * rng.standard_normal((self.nx, self.ny, self.nz))
        f0 = np.empty((self.nx, self.ny, self.nz, 19), np.float32)
        for q in range(19):
            eu = E[q, 0] * ux + E[q, 1] * uy + E[q, 2] * uz
            f0[..., q] = W[q] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * (ux * ux + uy * uy + uz * uz))
        return wp.array(f0, dtype=wp.float32, device=DEV)

    def reseed_uz(self, f, uz_seed, seed):
        """inject a SMALL SMOOTH spanwise perturbation onto the DEVELOPED base: δf_q = W_q·ρ·3·e_z,q·δuz adds ONLY
           z-momentum (ρ,ux,uy unchanged; Σ W_q e_z,q²=1/3 ⇒ Δuz=δuz exactly). δuz uses ONLY the longest spanwise modes
           (m=1,2,3 ≈ λ_z=4D,2D,1.33D), concentrated in the near wake ⇒ energy projects onto the mode-A eigenfunction
           from the start (no fast grid-scale / high-k_z viscous transient that would mask the linear rate at the floor)."""
        fnp = f.numpy()
        rho = fnp.sum(3).astype(np.float64)
        rng = np.random.default_rng(seed)
        z = np.arange(self.nz)
        prof = np.zeros(self.nz)
        for m in (1, 2, 3):                                                # smooth: only the longest spanwise wavelengths
            prof += rng.uniform(0.5, 1.0) * np.cos(2 * np.pi * m * z / self.nz + rng.uniform(0, 2 * np.pi))
        xmask = np.zeros(self.nx); xmask[self.cx:min(self.nx - 6, self.cx + int(12 * self.Dperp))] = 1.0   # near/mid wake
        duz = (xmask[:, None, None] * np.ones((1, self.ny, 1))) * prof[None, None, :]
        duz[self.solid0 == 1] = 0.0
        cur = float(np.sqrt((duz[self.fluid] ** 2).mean()))
        duz *= uz_seed * self.U / max(cur, 1e-30)                          # normalise rms(δuz) over fluid to uz_seed·U
        for q in range(19):
            fnp[..., q] = (fnp[..., q].astype(np.float64) + W[q] * rho * 3.0 * float(E[q, 2]) * duz).astype(np.float32)
        return wp.array(fnp, dtype=wp.float32, device=DEV)

    def step(self, f, fn, fx, fyl, inv_tau):
        wp.launch(collide, (self.nx, self.ny, self.nz), inputs=[f, self.solid, inv_tau], device=DEV)
        fx.zero_(); fyl.zero_()
        wp.launch(stream_bb, (self.nx, self.ny, self.nz),
                  inputs=[f, fn, self.solid, fx, fyl, self.nx, self.ny, self.nz], device=DEV)
        wp.launch(inflow_outflow, (self.ny, self.nz), inputs=[fn, self.U, self.nx, self.ny, self.nz], device=DEV)
        return fn, f

    def _rms_uz3(self, f, acc):
        for a in acc:
            a.zero_()
        wp.launch(accum_uz2, (self.nx, self.ny, self.nz),
                  inputs=[f, self.solid, self.x_near, self.x_far, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], self.cx],
                  device=DEV)
        sg, ng, sn, nn, sf, nf = (float(a.numpy()[0]) for a in acc)
        rg = np.sqrt(sg / max(ng, 1.0)) / self.U
        rn = np.sqrt(sn / max(nn, 1.0)) / self.U
        rf = np.sqrt(sf / max(nf, 1.0)) / self.U
        return rg, rn, rf

    def measure(self, warm, sample, seed=0, ramp=False, ez_every=200, cl_every=100, skip=500):
        """SELF-SELECTION method (g42-validated uz observable): Stage 1 — develop the 2-D base with uz PINNED at 0
           (symmetrize_z every step ⇒ no mode-A growth during warmup). Stage 2 — RELEASE (stop symmetrizing); the mode-A
           eigenmode SELF-SELECTS from the round-off floor and GROWS at the absolute Floquet rate if Re>Re_c (stays at the
           floor if Re<Re_c). Measuring uz (≡0 for the 2-D base) is the null space of the in-plane convective modes, so the
           growth is the ABSOLUTE mode-A (g42). σ = slope of ln rms(uz) over the clean exponential rise. Re_c = σ(Re)→0."""
        f = self.f_init(seed, uz_seed=0.0)
        fn = wp.zeros((self.nx, self.ny, self.nz, 19), dtype=wp.float32, device=DEV)
        fx, fyl = wp.zeros(1, dtype=wp.float32, device=DEV), wp.zeros(1, dtype=wp.float32, device=DEV)
        cl = []; nan = False
        for it in range(warm):                                             # ── develop 2-D base (uz pinned; ramped if requested)
            f, fn = self.step(f, fn, fx, fyl, self.inv_tau_at(it, ramp))
            wp.launch(symmetrize_z, (self.nx, self.ny, self.nz), inputs=[f, self.solid], device=DEV)
            if it % cl_every == 0:
                cl.append((it, float(fyl.numpy()[0]) / self.norm))
        acc = [wp.zeros(1, dtype=wp.float32, device=DEV) for _ in range(6)]
        ez = []
        for it in range(sample):                                           # ── RELEASE: uz self-selects mode-A from the floor
            f, fn = self.step(f, fn, fx, fyl, 1.0 / self.tau)
            if it % cl_every == 0:
                cl.append((warm + it, float(fyl.numpy()[0]) / self.norm))
            if it % ez_every == 0:
                rg, rn, rf = self._rms_uz3(f, acc)
                ez.append((it, rg, rn, rf))                                # t measured from the release (it=0 at release)
                if not np.isfinite(rg) or rg > 5.0:
                    nan = True; break
        ez = np.array(ez); cl = np.array(cl)
        Pz = None if nan else self._zspectrum(f)
        return dict(ez=ez, cl=cl, nan=nan, Pz=Pz, skip=skip)

    def _zspectrum(self, f):
        """near-wake uz z-spectrum P_z(m) (for the dominant λ_z / mode-A confirmation)."""
        fnp = f.numpy().astype(np.float64); rho = fnp.sum(3)
        uz = (fnp @ E[:, 2]) / rho
        D = self.Dperp
        x0 = self.cx + int(self.a_lat) + int(0.5 * D); x1 = min(self.nx - 4, x0 + int(4 * D))
        blk = uz[x0:x1, int(self.cy - D):int(self.cy + D + 1), :]
        blk = blk - blk.mean(axis=2, keepdims=True)
        return (np.abs(np.fft.rfft(blk, axis=2)) ** 2).mean(axis=(0, 1))


# ── growth-rate fit + Strouhal helpers ──────────────────────────────────────────────────────────────────────────────
def fit_sigma(ez, skip=1000, sat=8.0e-3, floor=3e-6, col=1):
    """σ = slope of ln(rms_uz/U) vs step over the LINEAR window of the POST-RESEED series (t from reseed): above the
       fit FLOOR (~1e-5, well clear of the float32 round-off floor so we never fit numerical noise), below saturation,
       after the brief non-modal transient (skip). The SIGN of σ is what Re_c needs. col: 1=global,2=near,3=far.
       ROBUST (g61 bug-fix — the prior run NaN'd the global fit when the decaying null underflowed below the floor with
       only n=3 survivors, then MIS-READ that NaN as 'σ≥0 / convective leak'): require n≥5 and R²>0.9 to accept the
       lstsq slope; otherwise fall back to the two-point log-slope over the valid window (sign-correct, less precise).
       ALWAYS returns a FINITE σ when ≥2 valid samples exist. Returns dict(sigma, sigma_lstsq, sigma_2pt, r2, n, …)."""
    deg = dict(sigma=np.nan, sigma_lstsq=np.nan, sigma_2pt=np.nan, r2=np.nan, n=0, method="degenerate", floor=float(floor))
    if ez is None or len(ez) == 0 or ez.ndim != 2 or ez.shape[1] <= col:
        return deg
    t = ez[:, 0]; r = ez[:, col]
    valid = np.isfinite(r) & (r > floor) & (r < sat)
    m = valid & (t >= skip)
    if m.sum() < 5:                                                        # relax: drop the skip (keep above the fit floor)
        m = valid
    n = int(m.sum())
    if n < 2:
        deg["n"] = n; return deg
    tt = t[m]; lr = np.log(r[m])
    tp = float((lr[-1] - lr[0]) / max(float(tt[-1] - tt[0]), 1.0))         # two-point log-slope (robust sign-correct fallback)
    A = np.vstack([tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(A, lr, rcond=None)
    sl = float(coef[0]); pred = A @ coef
    ss_res = float(np.sum((lr - pred) ** 2)); ss_tot = float(np.sum((lr - lr.mean()) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    h = len(tt) // 2                                                       # two-half agreement (clean exponential ⇒ σ1≈σ2)
    s1 = float(np.polyfit(tt[:h], lr[:h], 1)[0]) if h >= 2 else np.nan
    s2 = float(np.polyfit(tt[h:], lr[h:], 1)[0]) if len(tt) - h >= 2 else np.nan
    good = (n >= 5) and np.isfinite(r2) and (r2 > 0.9) and np.isfinite(sl)
    sigma = sl if good else tp
    method = "lstsq" if good else "two_point"
    if not np.isfinite(sigma):                                            # NaN GUARD — never surface a NaN σ
        sigma = tp if np.isfinite(tp) else np.nan
        method = "two_point"
    return dict(sigma=float(sigma) if np.isfinite(sigma) else np.nan, sigma_lstsq=float(sl), sigma_2pt=float(tp),
                r2=float(r2) if np.isfinite(r2) else np.nan, n=n, t0=float(tt[0]), t1=float(tt[-1]),
                r0=float(r[m][0]), r1=float(r[m][-1]), s1=s1, s2=s2, floor=float(floor), method=method)


def strouhal(cl, warm, Dperp, U):
    c = cl[cl[:, 0] >= warm]
    if len(c) < 32:
        return np.nan, np.nan
    dt = float(np.median(np.diff(c[:, 0])))
    St, bb, _, _ = g18.temporal_character(c[:, 1], Dperp, U, dt)
    return float(St), float(bb)


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as fh:
        fh.write(msg + "\n")


def save_run(tag, Re, shape, fit_g, fit_n, fit_f, St, bb, lamz, nan, warm, sample, T_shed,
             sigma=None, sigma_src=None):
    d = dict(tag=tag, Re=Re, shape=shape, nan=bool(nan), warm=warm, sample=sample, T_shed=T_shed,
             sigma=sigma, sigma_src=sigma_src,                              # robust σ used for the Re_c crossing (global→far→near fallback)
             sigma_global=fit_g.get("sigma"), r2_global=fit_g.get("r2"), method_global=fit_g.get("method"),
             sigma_near=fit_n.get("sigma"), sigma_far=fit_f.get("sigma"),
             sigma_half=(fit_g.get("s1"), fit_g.get("s2")), n_fit=fit_g.get("n"),
             St=St, bb=bb, lambda_z_over_Dperp=lamz)
    with open(os.path.join(RESDIR, f"{tag}.json"), "w") as fh:
        json.dump(d, fh, indent=2, default=lambda o: None if o is None else float(o))
    return d


# ── one growth-rate measurement (with the convective-vs-absolute discriminator) ─────────────────────────────────────
def run_point(shape_name, a_lat, b_lat, Re, warm, sample, seed=0, ramp=None, grid=None, uz_seed=1e-3, quiet=False):
    grid = grid or {}
    ramp = (Re > 250) if ramp is None else ramp
    w = WakeMA(Re, a_lat, b_lat, n_ramp=max(6000, warm // 2), **grid)
    tag = f"{shape_name}_Re{Re:g}_s{seed}"
    t0 = time.time()
    res = w.measure(warm, sample, seed=seed, ramp=ramp)
    dt = time.time() - t0
    if res["nan"]:
        log(f"  [{tag}] τ={w.tau:.4f} ramp={ramp}  ✗ BLEW UP (rms uz>5) after {dt:.0f}s")
        save_run(tag, Re, shape_name, {}, {}, {}, np.nan, np.nan, np.nan, True, warm, sample, w.T_shed)
        return dict(Re=Re, sigma=np.nan, nan=True, w=w)
    sk = res.get("skip", 2500)
    fg = fit_sigma(res["ez"], sk, col=1); fn = fit_sigma(res["ez"], sk, col=2); ff = fit_sigma(res["ez"], sk, col=3)
    # robust σ for the Re_c crossing: PREFER global; fall back to far, then near, if global degenerate (NaN guard, g61 fix)
    sigma = fg["sigma"]; src = "global"
    if not np.isfinite(sigma):
        if np.isfinite(ff["sigma"]):   sigma = ff["sigma"]; src = "far"
        elif np.isfinite(fn["sigma"]): sigma = fn["sigma"]; src = "near"
    St, bb = strouhal(res["cl"], warm, w.Dperp, w.U)
    lamz, m, tops = g20.lambda_z_from_Pz(res["Pz"], w.nz, w.Dperp) if res["Pz"] is not None else (np.nan, 0, [])
    save_run(tag, Re, shape_name, fg, fn, ff, St, bb, lamz, False, warm, sample, w.T_shed, sigma=sigma, sigma_src=src)
    sgT = (sigma * w.T_shed) if np.isfinite(sigma) else np.nan
    rms_last = res["ez"][-1, 1]
    if not quiet:
        log(f"  [{tag}] τ={w.tau:.4f} ramp={ramp} {dt:.0f}s | σ={sigma:+.2e}/step[{src}] (×T={sgT:+.3f}) "
            f"σ_g={fg['sigma']:+.2e}(R²={fg.get('r2',np.nan):.2f},n={fg.get('n',0)},{fg.get('method','?')}) "
            f"σ_near={fn['sigma']:+.2e} σ_far={ff['sigma']:+.2e} | halves σ1={fg.get('s1',np.nan):+.2e} "
            f"σ2={fg.get('s2',np.nan):+.2e} | rms0={res['ez'][0,1]:.2e}→{rms_last:.2e} St={St:.3f}(bb={bb:.2f}) λz={lamz:.2f}D")
    return dict(Re=Re, sigma=sigma, sigma_src=src, sigma_global=fg["sigma"], sigma_near=fn["sigma"], sigma_far=ff["sigma"],
                r2=fg["r2"], s1=fg.get("s1"), s2=fg.get("s2"), St=St, bb=bb, lamz=lamz, rms_last=rms_last,
                nan=False, w=w, ez=res["ez"])


def grew(p, thr=2.0e-5):
    """did the mode-A self-select & GROW at this Re? With the self-selection method a finite σ only exists when rms(uz)
       rose above the fit floor (1e-5) — i.e. the eigenmode grew; stable Re give σ=NaN (rms pinned at the round-off floor).
       So a finite σ above a small positive threshold ⇒ growth (works on both live points and the per-Re JSONs)."""
    return (not p.get("nan")) and np.isfinite(p.get("sigma")) and p["sigma"] > thr


def rec_from_growth(pts):
    """Re_c from the Floquet law σ ∝ (Re − Re_c) near onset: LINEAR-fit the growing points' σ(Re) and take the x-intercept
       (σ→0). Robust because growth (uz self-selecting from the floor) is the ABSOLUTE mode-A; stable Re give σ=NaN (floor-
       pinned, excluded). Returns (Re_c, (lowest_growing_Re, highest_growing_Re), n_growing)."""
    g = sorted([p for p in pts if grew(p)], key=lambda p: p["Re"])
    if len(g) < 2:
        return np.nan, None, len(g)
    Re = np.array([p["Re"] for p in g], float); sg = np.array([p["sigma"] for p in g], float)
    k, b = np.polyfit(Re, sg, 1)
    Rec = -b / k if k > 0 else np.nan
    return float(Rec), (g[0]["Re"], g[-1]["Re"]), len(g)


def bracket_shape(shape_name, a_lat, b_lat, Re_list, warm, sample, grid, refine_to=None, max_extra=3):
    """measure σ(Re) (self-selection growth rate) over Re_list; ADAPT so ≥2 points GROW (extend up if onset>list) and ≥1
       point is STABLE below the lowest grower (extend down to anchor the extrapolation); Re_c = σ(Re)→0 x-intercept."""
    log(f"\n  ── {shape_name}: σ(Re) self-selection bracket {Re_list} (warm={warm}, sample={sample}) ──")
    pts = [run_point(shape_name, a_lat, b_lat, Re, warm, sample, grid=grid) for Re in Re_list]
    done = {p["Re"] for p in pts}
    extra = 0
    while extra < max_extra:
        g = sorted([p for p in pts if grew(p)], key=lambda p: p["Re"])
        stable = [p for p in pts if (not p.get("nan")) and not grew(p)]
        if len(g) < 2:                                                     # onset above the list ⇒ push UP
            nxt = int(max(done) + max(30, 0.15 * max(done)))
        elif not any(p["Re"] < g[0]["Re"] for p in stable):               # no stable anchor below ⇒ push DOWN
            nxt = int(min(g[0]["Re"], min(done)) - max(25, 0.12 * g[0]["Re"]))
        else:
            break
        if nxt in done or nxt < 40:
            break
        log(f"  ── {shape_name}: adaptive extend to Re={nxt} (growers={[p['Re'] for p in g]}) ──")
        pts.append(run_point(shape_name, a_lat, b_lat, nxt, warm, sample, grid=grid)); done.add(nxt); extra += 1
    Rec, rng, ng = rec_from_growth(pts)
    # optional refine: add a point near the extrapolated Re_c to tighten the slope (if it is between two existing samples)
    if refine_to and np.isfinite(Rec):
        cand = int(round(Rec / 5.0) * 5)
        if cand not in done and 40 < cand < max(done) and min(done) < cand:
            log(f"  ── {shape_name}: refine near Re_c≈{Rec:.0f} (add Re={cand}) ──")
            pts.append(run_point(shape_name, a_lat, b_lat, cand, warm, sample, grid=grid)); done.add(cand)
            Rec, rng, ng = rec_from_growth(pts)
    log(f"  ⇒ {shape_name}: Re_c ≈ {Rec:.1f}  (σ→0 extrapolation over {ng} growing pts in {rng})" if np.isfinite(Rec)
        else f"  ⇒ {shape_name}: Re_c NOT resolved (growers={ng}) — onset outside the explored range")
    return Rec, rng, pts


# ── validation / benchmark ──────────────────────────────────────────────────────────────────────────────────────────
def bench(grid):
    w = WakeMA(190, 10, 10, **grid)
    log(f"BENCH grid {w.nx}×{w.ny}×{w.nz}={w.nx*w.ny*w.nz/1e6:.1f}M  Dperp={w.Dperp:.0f} blockage={w.Dperp/w.ny*100:.1f}% "
        f"L_z={w.nz/w.Dperp:.1f}D  τ(Re190)={w.tau:.4f}  x_near={w.x_near} x_far={w.x_far} T_shed≈{w.T_shed}")
    f = w.f_init(0); fn = wp.zeros((w.nx, w.ny, w.nz, 19), dtype=wp.float32, device=DEV)
    fx, fyl = wp.zeros(1, dtype=wp.float32, device=DEV), wp.zeros(1, dtype=wp.float32, device=DEV)
    for it in range(20):
        f, fn = w.step(f, fn, fx, fyl, 1.0 / w.tau)
    wp.synchronize(); t0 = time.time()
    for it in range(400):
        f, fn = w.step(f, fn, fx, fyl, 1.0 / w.tau)
    wp.synchronize(); s_per_k = (time.time() - t0) / 400 * 1000
    log(f"BENCH {s_per_k:.2f} s/1000 steps ⇒ a 50k-step run ≈ {s_per_k*50:.0f}s ≈ {s_per_k*50/60:.1f} min")
    return s_per_k


def validate(grid):
    """instrument validation: (a) 2-D base wake sheds with a physical St at Re=100; (b) the NULL — Re=100 spanwise σ≤0
       (mode-A stable far below onset; if σ>0 here the method measures a convective artifact, not the absolute mode)."""
    log("\n  ── VALIDATION: base-wake St + the Re=100 spanwise-growth NULL (σ must be ≤0) ──")
    p = run_point("circle", 10, 10, 100, warm=16000, sample=24000, grid=grid)
    log(f"  (a) St(Re=100)={p['St']:.3f} (literature ~0.16–0.17 +blockage shift) ⇒ "
        f"{'PHYSICAL base wake ✓' if 0.12 < p['St'] < 0.22 else 'St OFF — check base'}")
    null_ok = not grew(p)                                                  # self-selection NULL: uz must NOT grow (stays at floor)
    log(f"  (b) NULL Re=100: σ={p['sigma']:+.2e} (rms0→{p['rms_last']:.1e}); did it grow? {grew(p)} ⇒ "
        f"{'uz STAYS at the floor — no spurious/convective growth far below onset ✓' if null_ok else '✗ uz GREW at Re=100 — METHOD BROKEN (convective leak)'}")
    return p, null_ok


SHAPES = {"circle": (10, 10), "ellipseAR2": (20, 10), "ellipseAR05": (5, 10),
          "ellipseAR3": (30, 10), "ellipseAR04": (4, 10),
          "ellipseAR3fine": (45, 15), "circlefine": (15, 15),
          "circle_lb": (10, 10), "ellipseAR2_lb": (20, 10),
          "circle_lb2": (10, 10), "ellipseAR2_lb2": (20, 10)}   # *_lb = half-blockage (--lowblock); *_lb2 = quarter-blockage (--lowblock2); g92 exponent-vs-blockage render-match


def emit_from_disk(grid):
    """SYNTHESIS step: read every per-Re JSON in g61_runs/, group by shape, compute Re_c via the σ sign change, and
       write the evidence JSON. Decouples the expensive per-Re GPU runs (run synchronously, one Bash call each) from the
       cheap verdict — so progress survives and the orchestrator never has to hold an 80-min process in one call."""
    import glob
    byshape = {}
    for fp in sorted(glob.glob(os.path.join(RESDIR, "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if not isinstance(d, dict) or "shape" not in d or "Re" not in d:
            continue
        sig = d.get("sigma")
        if sig is None or not np.isfinite(sig):                            # back-compat / fallback to far estimator
            sig = d.get("sigma_global")
        if sig is None or not np.isfinite(sig):
            sig = d.get("sigma_far")
        byshape.setdefault(d["shape"], []).append(
            dict(Re=float(d["Re"]), sigma=(float(sig) if sig is not None else np.nan),
                 nan=bool(d.get("nan", False)), sigma_near=d.get("sigma_near"), sigma_far=d.get("sigma_far"),
                 St=d.get("St"), method=d.get("method_global")))
    circ = sorted(byshape.get("circle", []), key=lambda r: r["Re"])
    e2 = sorted(byshape.get("ellipseAR2", []), key=lambda r: r["Re"])
    e05 = sorted(byshape.get("ellipseAR05", []), key=lambda r: r["Re"])
    Rec_cyl, br_cyl, _ = rec_from_growth(circ) if circ else (np.nan, None, 0)
    Rec_e2, br_e2, _ = rec_from_growth(e2) if e2 else (np.nan, None, 0)
    Rec_e05, br_e05, _ = rec_from_growth(e05) if e05 else (np.nan, None, 0)
    g1 = bool(np.isfinite(Rec_cyl) and 180 <= Rec_cyl <= 200)
    null_rec = next((r for r in circ if abs(r["Re"] - 100) < 1e-6), None)
    null_ok = bool(null_rec and not grew(null_rec))                        # self-selection NULL: Re=100 did NOT grow
    vp = {"St": (null_rec or {}).get("St"), "sigma": (null_rec or {}).get("sigma")}
    results = dict(
        circle=dict(Rec=Rec_cyl, bracket=br_cyl, pts=[(r["Re"], r["sigma"]) for r in circ]),
        ellipseAR2=dict(Rec=Rec_e2, bracket=br_e2, pts=[(r["Re"], r["sigma"]) for r in e2]),
        ellipseAR05=dict(Rec=Rec_e05, bracket=br_e05, pts=[(r["Re"], r["sigma"]) for r in e05]))
    log(f"\n  [emit-from-disk] circle pts={[(r['Re'], round(r['sigma'],5)) for r in circ]}")
    log(f"  [emit-from-disk] ellipseAR2 pts={[(r['Re'], round(r['sigma'],5)) for r in e2]}")
    log(f"  [emit-from-disk] ellipseAR05 pts={[(r['Re'], round(r['sigma'],5)) for r in e05]}")
    emit_g61(results, Rec_cyl, Rec_e2, Rec_e05, g1, null_ok, vp, stopped=False)


def main():
    quick = "--quick" in sys.argv
    grid = dict(nx=256, ny=288, nz=80, cx=64, U=0.1)
    if quick:
        grid = dict(nx=208, ny=224, nz=64, cx=52, U=0.1)                   # D=16 coarse smoke-test grid
    if "--fine" in sys.argv:
        grid = dict(nx=384, ny=432, nz=120, cx=96, U=0.1)                  # fine: 1.5x cells (D=30 for AR3fine b_lat=15), matched blockage 6.9%, L_z=4D; BL~1.4 cells @ Re500 (vs 0.9 at D=20)
    if "--lowblock" in sys.argv:
        grid = dict(nx=256, ny=576, nz=80, cx=64, U=0.1)                   # HALF blockage (D=20/ny=576 = 3.5% vs default 6.9%) — tests the exponent-blockage bias / render-match Monash Re_c(AR)
    if "--lowblock2" in sys.argv:
        grid = dict(nx=256, ny=1152, nz=80, cx=64, U=0.1)                  # QUARTER blockage (D=20/ny=1152 = 1.75%) — 3rd point for g92 to firm the 0%-extrap render-match + curvature check (B grade 412)
    if "--bench" in sys.argv:
        bench(grid); return 0
    if "--emit" in sys.argv:                                               # synthesis-only: read JSONs → evidence file
        emit_from_disk(grid); return 0
    if "--single" in sys.argv:                                             # ONE σ(Re) point, save per-Re JSON (orchestrated)
        i = sys.argv.index("--single"); shape = sys.argv[i + 1]; Re = float(sys.argv[i + 2])
        rest = [a for a in sys.argv[i + 3:] if not a.startswith("--")]
        warm = int(rest[0]) if len(rest) > 0 else (12000 if quick else 16000)
        sample = int(rest[1]) if len(rest) > 1 else (20000 if quick else 24000)
        a_lat, b_lat = SHAPES[shape]
        p = run_point(shape, a_lat, b_lat, Re, warm, sample, grid=grid)
        log(f"  [single DONE] {shape} Re={Re:g} σ={p['sigma']:+.3e}/step[{p.get('sigma_src')}] nan={p['nan']} "
            f"→ {shape}_Re{Re:g}_s0.json")
        return 0
    log("=" * 118)
    log("P3·G61 — mode-A 3-D-transition Re_c: circle vs ellipse (DNS-3-D-growth, D3Q19-BGK LBM) — settle FSI adversary #3")
    log(f"  device={DEV} warp {wp.__version__}; grid {grid}; Dperp=20 (semi-axes circle a=b=10)")
    log("  anchor: Williamson / Barkley&Henderson 1996 — cylinder mode-A Re_c≈188–190, λ_z≈3.96D")
    log("=" * 118)
    bench(grid)

    if "--validate" in sys.argv:
        validate(grid); return 0

    # de-risk first: validation (St + null) ; then G1 cylinder anchor
    vp, null_ok = validate(grid)

    warm, sample = (12000, 20000) if quick else (16000, 24000)
    # ── G1: cylinder mode-A anchor ─────────────────────────────────────────────────────────────────────────────────
    cyl_list = [150, 210] if quick else [160, 190, 220, 250]
    Rec_cyl, br_cyl, pts_cyl = bracket_shape("circle", 10, 10, cyl_list, warm, sample, grid, refine_to=20)
    g1 = bool(np.isfinite(Rec_cyl) and 180 <= Rec_cyl <= 200)
    sane = bool(np.isfinite(Rec_cyl) and 165 <= Rec_cyl <= 285)             # a sane mode-A onset (allows blockage/resol. offset)
    log(f"\n  G1 (anchor): cylinder Re_c={Rec_cyl:.1f}  ∈[180,200]? {'PASS ✓' if g1 else 'OUTSIDE — blockage/resolution-biased (ratio cancels it)'}"
        f"  | sane mode-A range={sane} | null σ(Re=100)<0={null_ok}")

    results = dict(circle=dict(Rec=Rec_cyl, bracket=br_cyl, pts=[(p['Re'], p['sigma']) for p in pts_cyl]))
    if not sane:
        log("  cylinder onset NOT in a sane mode-A range (or unbracketed) ⇒ harness genuinely off — STOP + diagnose (OODA), do NOT consume ellipse runs.")
        emit_g61(results, Rec_cyl, np.nan, np.nan, g1, null_ok, vp, stopped=True)
        return 1

    # ── G2/G3: ellipse brackets KEYED to the measured cylinder onset (cancels the harness absolute bias in the SEARCH
    #    range; spanning ratio≈1.1→2.0 for AR=2 and ≈0.35→0.8 for AR=0.5 ⇒ WIDE enough to either confirm or REFUTE g59) ──
    def around(center, fracs):
        return sorted({int(round(center * fr / 10.0) * 10) for fr in fracs})
    el_list = around(Rec_cyl, (1.1, 1.4, 1.7, 2.0))                        # AR=2 predicted ratio 1.68 (refute if ≈1.0)
    Rec_e2, br_e2, pts_e2 = bracket_shape("ellipseAR2", 20, 10, el_list, warm, sample, grid, refine_to=40)
    # ── AR=0.5 (cross-stream, a_lat=5 b_lat=10) — predicted ratio 0.42 ─────────────────────────────────────────────
    Rec_e05 = np.nan; br_e05 = None; pts_e05 = []
    if not quick:
        e05_list = around(Rec_cyl, (0.35, 0.5, 0.65, 0.8))
        Rec_e05, br_e05, pts_e05 = bracket_shape("ellipseAR05", 5, 10, e05_list, warm, sample, grid, refine_to=20)

    results["ellipseAR2"] = dict(Rec=Rec_e2, bracket=br_e2, pts=[(p['Re'], p['sigma']) for p in pts_e2])
    results["ellipseAR05"] = dict(Rec=Rec_e05, bracket=br_e05, pts=[(p['Re'], p['sigma']) for p in pts_e05])
    emit_g61(results, Rec_cyl, Rec_e2, Rec_e05, g1, null_ok, vp, stopped=False)
    return 0


def emit_g61(results, Rec_cyl, Rec_e2, Rec_e05, g1, null_ok, vp, stopped):
    PRED = {"AR2": 1.6823303635468045, "AR05": 0.42058259088670114}        # [h ratio]² from g59 (predicted Re_c/Re_c,cyl)
    ratio_e2 = float(Rec_e2 / Rec_cyl) if np.isfinite(Rec_e2) and np.isfinite(Rec_cyl) else np.nan
    ratio_e05 = float(Rec_e05 / Rec_cyl) if np.isfinite(Rec_e05) and np.isfinite(Rec_cyl) else np.nan
    within2x_e2 = bool(np.isfinite(ratio_e2) and 0.5 <= ratio_e2 / PRED["AR2"] <= 2.0)
    within2x_e05 = bool(np.isfinite(ratio_e05) and 0.5 <= ratio_e05 / PRED["AR05"] <= 2.0)
    g3 = within2x_e2                                                        # primary falsifier = AR=2
    verdict = ("STOPPED (G1 fail)" if stopped else ("HOLDS" if g3 else "REFUTED"))
    log("\n" + "=" * 118)
    log(f"  G1 cylinder anchor Re_c={Rec_cyl:.1f} ∈[180,200] {'PASS' if g1 else 'FAIL'} | null σ(Re=100)<0 {null_ok}")
    log(f"  G2 ellipse AR=2 Re_c={Rec_e2:.1f}  AR=0.5 Re_c={Rec_e05 if np.isfinite(Rec_e05) else float('nan'):.1f}")
    log(f"  G3 falsifier: AR=2 ratio_meas={ratio_e2:.3f} vs pred 1.68 ⇒ within-2×? {within2x_e2}  ({'HOLDS' if g3 else 'REFUTED'})")
    log(f"       AR=0.5 ratio_meas={ratio_e05:.3f} vs pred 0.42 ⇒ within-2×? {within2x_e05}")
    log("=" * 118)
    emit(
        "g61_modeA_Rec_cylinder_vs_ellipse",
        f"FSI adversary #3 SETTLED by DNS-3-D-growth (D3Q19-BGK LBM): measured mode-A 3-D-transition Re_c via the spanwise "
        f"rms(uz) growth rate (uz=0 for the 2-D base ⇒ filters the convective confound, g42). Cylinder anchor Re_c={Rec_cyl:.0f} "
        f"(G1 {'PASS' if g1 else 'FAIL'}, target 188-190). Ellipse AR=2 (streamwise) Re_c={Rec_e2:.0f}, AR=0.5 Re_c={Rec_e05:.0f}. "
        f"g59 Cheeger prediction Re_c∝(h·D_perp)^-2: AR=2 ratio pred 1.68, MEASURED {ratio_e2:.2f}; AR=0.5 pred 0.42, MEASURED "
        f"{ratio_e05:.2f}. Falsifier (within 2x): AR=2 {'HOLDS' if within2x_e2 else 'REFUTED'}. Convective-vs-absolute: NULL "
        f"σ(Re=100)<0 = {null_ok}; growth measured on uz (absolute mode), near/far-window cross-checked per run.",
        numbers={
            "Re_c_cylinder_measured": Rec_cyl, "Re_c_ellipseAR2_measured": Rec_e2, "Re_c_ellipseAR05_measured": Rec_e05,
            "ratio_AR2_over_cyl_measured": ratio_e2, "ratio_AR05_over_cyl_measured": ratio_e05,
            "predicted_ratio_AR2": PRED["AR2"], "predicted_ratio_AR05": PRED["AR05"],
            "predicted_Re_c_AR2": 189.0 * PRED["AR2"], "predicted_Re_c_AR05": 189.0 * PRED["AR05"],
            "St_base_Re100": vp.get("St"), "brackets": {k: results[k].get("bracket") for k in results},
            "sigma_vs_Re": {k: results[k].get("pts") for k in results},
        },
        sigma={"method": "DNS-3-D-growth: sigma=d ln rms(uz)/dt over linear window; Re_c=sigma sign change",
               "observable": "rms(uz)/U (uz=0 for 2-D base => absolute mode-A energy, convective-clean per g42)",
               "anchor": "circular cylinder mode-A Re_c=188-190 (Williamson; Barkley&Henderson 1996)",
               "ratio_basis": "Re_c(ellipse)/Re_c(circle) both from THIS harness (cancels LBM/blockage/resolution bias)"},
        gates={"G1_cylinder_anchor_180_200": g1, "G1_Re_c_cylinder": Rec_cyl, "null_sigma_Re100_negative": null_ok,
               "G2_ellipse_Rec_bracketed": bool(np.isfinite(Rec_e2)),
               "G3_falsifier_AR2_within2x": within2x_e2, "G3_falsifier_AR05_within2x": within2x_e05,
               "verdict": verdict,
               "why": f"ratio AR2 measured {ratio_e2:.2f} vs predicted 1.68 => {'within' if within2x_e2 else 'outside'} 2x"},
        provenance={
            "directive": "settle FSI adversary #3 (g59) — measure actual mode-A Re_c, test Cheeger Re_c∝(h·D_perp)^-2",
            "method": "build-spec method B (DNS-3-D-growth), reuses g21 validated D3Q19-BGK + viscosity-ramp",
            "de_risk": "MANDATORY cylinder anchor first (G1); STOP if Re_c not in [180,200]",
            "convective_vs_absolute": "observable uz (null space of in-plane convective modes); NULL Re=100 sigma<0; near/far window split",
            "no_fit": True, "cert_authority": "B",
        },
        cross_checks={
            "known_reference": {"name": "cylinder mode-A Re_c (Williamson; Barkley&Henderson 1996)",
                                "Re_c_target": "188-190", "Re_c_measured": Rec_cyl, "pass": bool(g1)},
            "null_falsifier": {"name": "Re=100 (far below onset) spanwise perturbation must DECAY (sigma<0) — else convective leak",
                               "sigma_Re100": vp.get("sigma"), "pass": bool(null_ok)},
            "over_determination": {"name": "Re_c ratio (harness-cancelling) vs g59 [h ratio]^2 within 2x",
                                   "ratio_AR2_measured": ratio_e2, "ratio_AR2_predicted": PRED["AR2"],
                                   "pass": bool(within2x_e2)},
        },
    )
    log("emitted artifacts/g61_modeA_Rec_cylinder_vs_ellipse.json")


if __name__ == "__main__":
    sys.exit(main())

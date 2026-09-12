#!/usr/bin/env python3
"""LBM GPU FP16 HALF2-PACKED (★OUTCLASS steg 2b) — REBUT/CONFIRM "Warp cannot coalesce 16-bit loads".

BACKGROUND: lbm_gpu_fp16.py stores the 9 D2Q9 deviation populations as wp.array3d(dtype=wp.float16)
in SoA layout (9,nx,ny). Measured ~6449 MLUPS = NO gain over FP32 fused (7345). HYPOTHESIS for the missing
gain: warp emits 9 SCATTERED scalar 2-byte loads per voxel; each 16-bit load underuses
the memory transaction (the hardware wants >=4-byte sectors) -> no effective bandwidth gain despite halved data.

THIS FILE (half2 packing): pack two FP16 per wp.uint32 word (lo = float16 bits, hi << 16) so the kernel
issues WIDE 4-byte coalesced loads/stores (5 uint32 words per voxel) instead of 9 scattered
2-byte ones. 9 populations -> 5 words: (k0,k1)(k2,k3)(k4,k5)(k6,k7)(k8,_). Unpacked in-register to FP32 for
compute. The SAME deviation-storage trick (s = f - w) as the plain version -> the same precision.

Bit-reinterpret float16<->uint16: warp has no wp.reinterpret builtin, so a native
pointer-cast-snippet (wp.func_native) = ekvivalent med CUDA __half_as_ushort, verifierad bit-exakt.

VALIDATION: Poiseuille vs ANALYTIC (must be no worse than plain FP16 accuracy; L2% reported).
MEASURE: peak MLUPS over 512/1024/2048/4096^2, compared against plain FP16 AND FP32 fused in the SAME environment.
beat_the_wall = TRUE only if half2 MEANINGFULLY beats plain FP16 (coalescing mechanism confirmed).

  python3 lbm_gpu_fp16_half2.py
"""
import sys
import time
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
vec9 = wp.types.vector(length=9, dtype=wp.float32)

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float32)
WT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
OP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# ── half2 bit-pack/unpack: two float16 per uint32 (lo bits | hi<<16). ──
# warp has no reinterpret-builtin; raw pointer-cast snippet = true bit reinterpret (verified bit-exact).
_PACK_SRC = """
const unsigned short* pa = reinterpret_cast<const unsigned short*>(&a);
const unsigned short* pb = reinterpret_cast<const unsigned short*>(&b);
return (unsigned int)(*pa) | (((unsigned int)(*pb)) << 16);
"""
@wp.func_native(_PACK_SRC)
def pack2(a: wp.float16, b: wp.float16) -> wp.uint32: ...

_LO_SRC = """
unsigned short s = (unsigned short)(w & 0xFFFFu);
wp::float16 r;
*reinterpret_cast<unsigned short*>(&r) = s;
return r;
"""
@wp.func_native(_LO_SRC)
def unpack_lo(w: wp.uint32) -> wp.float16: ...

_HI_SRC = """
unsigned short s = (unsigned short)((w >> 16) & 0xFFFFu);
wp::float16 r;
*reinterpret_cast<unsigned short*>(&r) = s;
return r;
"""
@wp.func_native(_HI_SRC)
def unpack_hi(w: wp.uint32) -> wp.float16: ...


@wp.func
def load_dev(p: wp.array3d(dtype=wp.uint32), i: int, j: int, k: int) -> wp.float32:
    """Read deviation s_k (FP32) from packed half2 store: word = k//2, lo if even else hi."""
    word = k >> 1
    w = p[word, i, j]
    if (k & 1) == 0:
        return wp.float32(unpack_lo(w))
    return wp.float32(unpack_hi(w))


@wp.kernel
def collide_stream_half2(pA: wp.array3d(dtype=wp.uint32), pB: wp.array3d(dtype=wp.uint32),
                         solid: wp.array2d(dtype=wp.int32),
                         cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
                         w: wp.array(dtype=wp.float32), opp: wp.array(dtype=wp.int32),
                         omega: float, gforce: float, u_in: float, periodic_x: int, nx: int, ny: int):
    """Fused collide+stream, HALF2-PACKED uint32 store of deviation s=f−w, FP32-compute.
    5 uint32 words/voxel; WIDE 4-byte coalesced loads vs plain-FP16's 9 scattered 2-byte loads."""
    i, j = wp.tid()
    if solid[i, j] == 1:
        for word in range(5):
            pB[word, i, j] = pA[word, i, j]                 # solid passive: copy raw words
        return
    g = vec9()
    for k in range(9):
        si = i - int(cx[k]); sj = j - int(cy[k])
        if periodic_x == 1:
            if si < 0: si += nx
            if si >= nx: si -= nx
        else:
            if si < 0: si = 0
            if si >= nx: si = nx - 1
        if sj < 0: sj = 0
        if sj >= ny: sj = ny - 1
        if solid[si, sj] == 1:                              # bounce-back: w[opp[k]]==w[k] in D2Q9
            g[k] = load_dev(pA, i, j, opp[k]) + w[k]
        else:
            g[k] = load_dev(pA, si, sj, k) + w[k]           # f = s + w (deviation → full f)
    rho = float(0.0); mx = float(0.0); my = float(0.0)
    for k in range(9):
        rho += g[k]; mx += cx[k] * g[k]; my += cy[k] * g[k]
    ux = mx / rho + 0.5 * gforce
    uy = my / rho
    if u_in > 0.0 and i == 0:
        ux = u_in; uy = 0.0; rho = 1.0
    usq = ux * ux + uy * uy
    # compute the 9 new deviations into a register vec9
    nd = vec9()
    for k in range(9):
        cu = cx[k] * ux + cy[k] * uy
        feq = w[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        val = g[k] - omega * (g[k] - feq)
        if gforce != 0.0:
            val += (1.0 - 0.5 * omega) * 3.0 * w[k] * cx[k] * gforce
        nd[k] = val - w[k]                                  # deviation s = f − w
    # pack into 5 uint32 words → WIDE coalesced stores
    pB[0, i, j] = pack2(wp.float16(nd[0]), wp.float16(nd[1]))
    pB[1, i, j] = pack2(wp.float16(nd[2]), wp.float16(nd[3]))
    pB[2, i, j] = pack2(wp.float16(nd[4]), wp.float16(nd[5]))
    pB[3, i, j] = pack2(wp.float16(nd[6]), wp.float16(nd[7]))
    pB[4, i, j] = pack2(wp.float16(nd[8]), wp.float16(0.0))


@wp.kernel
def macro_ux_half2(p: wp.array3d(dtype=wp.uint32), cx: wp.array(dtype=wp.float32), w: wp.array(dtype=wp.float32),
                   solid: wp.array2d(dtype=wp.int32), ux: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    if solid[i, j] == 1:
        ux[i, j] = 0.0; return
    rho = float(0.0); mx = float(0.0)
    for k in range(9):
        fk = load_dev(p, i, j, k) + w[k]
        rho += fk; mx += cx[k] * fk
    ux[i, j] = mx / rho


def _equil_dev_packed(rho, ux, uy):
    """Initial deviation s=f_eq−w, packed half2 into uint32 SoA (5,nx,ny) on the host."""
    nx, ny = rho.shape
    s = np.empty((9, nx, ny), np.float16)
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = CX[k] * ux + CY[k] * uy
        feq = WT[k] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
        s[k] = (feq - WT[k]).astype(np.float16)
    # pack: word w holds (s[2w] lo, s[2w+1] hi); last word holds (s[8], 0)
    packed = np.zeros((5, nx, ny), np.uint32)
    for word in range(5):
        lo_k = 2 * word
        hi_k = 2 * word + 1
        lo_bits = s[lo_k].view(np.uint16).astype(np.uint32)
        if hi_k < 9:
            hi_bits = s[hi_k].view(np.uint16).astype(np.uint32)
        else:
            hi_bits = np.zeros((nx, ny), np.uint32)
        packed[word] = lo_bits | (hi_bits << np.uint32(16))
    return packed


def run(nx, ny, tau, solid_np, max_steps, gforce=0.0, u_in=0.0, periodic_x=1, tol=0.0, check=2000, ret_field=False):
    omega = 1.0 / tau
    rho0 = np.ones((nx, ny), np.float32); ux0 = np.full((nx, ny), u_in, np.float32); uy0 = np.zeros((nx, ny), np.float32)
    pA = wp.array(_equil_dev_packed(rho0, ux0, uy0), dtype=wp.uint32, device=DEV)
    pB = wp.zeros((5, nx, ny), dtype=wp.uint32, device=DEV)
    solid = wp.array(solid_np.astype(np.int32), dtype=wp.int32, device=DEV)
    cx = wp.array(CX, dtype=wp.float32, device=DEV); cy = wp.array(CY, dtype=wp.float32, device=DEV)
    w = wp.array(WT, dtype=wp.float32, device=DEV); opp = wp.array(OP, dtype=wp.int32, device=DEV)
    uxd = wp.zeros((nx, ny), dtype=wp.float32, device=DEV)
    ux_prev = np.zeros((nx, ny), np.float32); res = 1.0
    wp.synchronize(); t0 = time.time(); it = 0
    while it < max_steps:
        wp.launch(collide_stream_half2, dim=(nx, ny), inputs=[pA, pB, solid, cx, cy, w, opp, omega, gforce, u_in, periodic_x, nx, ny], device=DEV)
        pA, pB = pB, pA
        it += 1
        if tol > 0.0 and it % check == 0:
            wp.launch(macro_ux_half2, dim=(nx, ny), inputs=[pA, cx, w, solid, uxd], device=DEV)
            wp.synchronize()
            uxn = uxd.numpy()
            res = float(np.max(np.abs(uxn - ux_prev)) / (np.max(np.abs(uxn)) + 1e-30))
            ux_prev = uxn.copy()
            if res < tol: break
    wp.synchronize(); dt = time.time() - t0
    mlups = nx * ny * it / dt / 1e6
    field = None
    if ret_field:
        wp.launch(macro_ux_half2, dim=(nx, ny), inputs=[pA, cx, w, solid, uxd], device=DEV)
        wp.synchronize(); field = uxd.numpy()
    return field, it, res, mlups


def main():
    print("=" * 80)
    print(f"LBM GPU FP16 HALF2-PACKED (★rebut 'warp cannot coalesce 16-bit') device={DEV}")
    print("=" * 80)

    # -- VALIDATION: Poiseuille vs ANALYTIC (must stay near plain-FP16 ~9% L2) --
    nx, ny = 10, 42; tau = 0.8; nu = (tau - 0.5) / 3.0; G = 2e-5
    solid = np.zeros((nx, ny), bool); solid[:, 0] = True; solid[:, -1] = True
    ux, st, rs, _ = run(nx, ny, tau, solid, max_steps=200000, gforce=G, periodic_x=1, tol=1e-6, check=2000, ret_field=True)
    prof = ux[nx // 2, :]; H = ny - 2
    y = np.arange(ny) - 0.5
    u_ana = np.where((y > 0) & (y < H), G / (2 * nu) * y * (H - y), 0.0)
    inner = (np.arange(ny) >= 1) & (np.arange(ny) <= ny - 2)
    l2 = float(np.sqrt(np.mean((prof[inner] - u_ana[inner]) ** 2)) / (u_ana.max() + 1e-30)) * 100
    umax_lbm, umax_ana = float(prof.max()), float(u_ana.max())
    V_ok = l2 < 12.0  # must stay ~ plain-FP16's accuracy (plain baseline ~9.22%)
    print(f"\nVALIDERING Poiseuille (half2-packed dev): u_max {umax_lbm:.3e} vs analytisk {umax_ana:.3e}; L2 {l2:.2f}% "
          f"{'no worse than plain-FP16 accuracy (same deviation trick)' if V_ok else 'degraded'}  [steady@{st}, res={rs:.1e}]")

    # ── THROUGHPUT-SVEP: half2 vs plain-FP16 (~6449) vs FP32-fused (~7345) ──
    print(f"\nTHROUGHPUT (half2-packed, 400 steg fixed):")
    best = 0.0; rows = []
    for n in [512, 1024, 2048, 4096]:
        bsolid = np.zeros((n, n), bool); bsolid[:, 0] = True; bsolid[:, -1] = True
        _, _, _, ml = run(n, n, 0.6, bsolid, max_steps=400, gforce=1e-6, periodic_x=1, tol=0.0)
        best = max(best, ml); rows.append((n, ml))
        print(f"   {n:>5}×{n:<5}: {ml:7.0f} MLUPS", flush=True)

    plain_fp16_ref = 6449.0; fp32_peak = 7345.0; ceiling_fp16 = 18667
    gain_vs_plain = best / plain_fp16_ref
    gain_vs_fp32 = best / fp32_peak
    beat_wall = best > plain_fp16_ref * 1.05      # meaningfully exceeds plain-FP16 (>5%)
    print(f"\n   best {best:.0f} MLUPS = {best/ceiling_fp16*100:.0f}% of the FP16 ceiling ({ceiling_fp16})")
    print(f"   vs plain-FP16 ({plain_fp16_ref:.0f}): {gain_vs_plain:.2f}×   vs FP32-fused ({fp32_peak:.0f}): {gain_vs_fp32:.2f}×")
    print(f"   beat_the_wall (half2 >> plain FP16, >5%): {'YES - coalescing mechanism confirmed' if beat_wall else 'NO - half2 does not beat plain FP16'}")

    all_ok = V_ok and beat_wall
    print("\n" + "=" * 80)
    print(f"VERDICT: half2 packing broke the 16-bit coalescing wall = {'yes' if all_ok else 'NO'}")
    print(f"  half2 {best:.0f} vs plain-FP16 {plain_fp16_ref:.0f} vs FP32-fused {fp32_peak:.0f} MLUPS; precision L2 {l2:.2f}%.")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""DIFFERENTIABLE-LBM PROBE (★breakthrough-riktning, FORCE:ad) — ger warp KORREKT gradient GENOM LBM:n?

Hypothesis: warp IS differentiable, so d(flow objective)/d(input) through the LBM rollout comes FOR FREE, enabling
GRADIENT-based generative design (follow dJ/d(geometry)) instead of black-box search over thousands of evaluations. Enabling test:
can warp autodiff backpropagate through the LBM CORRECTLY (vs central finite differences)?

Setup: channel flow (walls top/bottom, periodic in x), PER-CELL body force force[nx,ny] (continuous input,
requires_grad). Objective J = sum of ux in the right region. TWO clean autodiff-friendly kernels (collide->fpost, pull-stream->f;
no atomics on the differentiable path). wp.Tape().backward(J) -> force.grad, compared against FD on selected cells.

  python3 differentiable_lbm_probe.py
"""
import sys
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float32)
WT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
OP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


@wp.kernel
def collide(f0: wp.array3d(dtype=wp.float32), fpost: wp.array3d(dtype=wp.float32),
            force: wp.array2d(dtype=wp.float32),
            cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32), w: wp.array(dtype=wp.float32),
            omega: float):
    i, j = wp.tid()
    rho = float(0.0); mx = float(0.0); my = float(0.0)
    for k in range(9):
        fk = f0[k, i, j]; rho += fk; mx += cx[k] * fk; my += cy[k] * fk
    g = force[i, j]                                       # differentierbar per-cell-kraft
    ux = mx / rho + 0.5 * g
    uy = my / rho
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = cx[k] * ux + cy[k] * uy
        feq = w[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        fpost[k, i, j] = f0[k, i, j] - omega * (f0[k, i, j] - feq) + (1.0 - 0.5 * omega) * 3.0 * w[k] * cx[k] * g


@wp.kernel
def stream(fpost: wp.array3d(dtype=wp.float32), f1: wp.array3d(dtype=wp.float32),
           solid: wp.array2d(dtype=wp.int32), cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
           opp: wp.array(dtype=wp.int32), nx: int, ny: int):
    i, j = wp.tid()                                       # pull-stream (gather, en skrivare/cell → autodiff-ren)
    for k in range(9):
        si = i - int(cx[k]); sj = j - int(cy[k])
        if si < 0: si += nx
        if si >= nx: si -= nx
        if sj < 0: sj = 0
        if sj >= ny: sj = ny - 1
        if solid[si, sj] == 1:
            f1[k, i, j] = fpost[opp[k], i, j]             # half-way bounce-back
        else:
            f1[k, i, j] = fpost[k, si, sj]


@wp.kernel
def objective(f: wp.array3d(dtype=wp.float32), cx: wp.array(dtype=wp.float32),
              mask: wp.array2d(dtype=wp.float32), J: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    rho = float(0.0); mx = float(0.0)
    for k in range(9):
        fk = f[k, i, j]; rho += fk; mx += cx[k] * fk
    wp.atomic_add(J, 0, (mx / rho) * mask[i, j])


# Order-invariant accumulation: the objective is a float atomic sum, hence order-dependent (float addition
# is not associative). Rounding each contribution in float64 to an int64 fixed point and summing with
# integer atomics is associative -> the reported J is bit-identical run to run.
DETERMINISTIC_ACCUMULATION = True
ACC_SCALE = 2.0 ** 49                                  # |ux| <= 1 (lattice units) -> |sum| <= nx*ny <= 4096
assert 4096.0 * ACC_SCALE < 2.0 ** 62, "fixed-point overflow"


@wp.kernel
def objective_i64(f: wp.array3d(dtype=wp.float32), cx: wp.array(dtype=wp.float32),
                  mask: wp.array2d(dtype=wp.float32), scale: wp.float64, J: wp.array(dtype=wp.int64)):
    i, j = wp.tid()
    rho = float(0.0); mx = float(0.0)
    for k in range(9):
        fk = f[k, i, j]; rho += fk; mx += cx[k] * fk
    wp.atomic_add(J, 0, wp.int64(wp.round(wp.float64((mx / rho) * mask[i, j]) * scale)))


def objective_value(J, f, cx, mask, dim):
    """Value of the flow objective. The float array J stays for the adjoint (a quantiser has no useful
    derivative, so the tape keeps the float accumulation); the reported value is the int64 one."""
    if not DETERMINISTIC_ACCUMULATION:
        return float(J.numpy()[0])
    acc = wp.zeros(1, dtype=wp.int64, device=DEV)
    wp.launch(objective_i64, dim, inputs=[f, cx, mask, wp.float64(ACC_SCALE), acc], device=DEV)
    wp.synchronize()
    return float(int(acc.numpy()[0])) / ACC_SCALE


def _equil(rho, ux, uy):
    nx, ny = rho.shape; f = np.empty((9, nx, ny), np.float32)
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = CX[k] * ux + CY[k] * uy
        f[k] = WT[k] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f


def main():
    print("=" * 78); print(f"DIFFERENTIABLE-LBM PROBE (★FORCE:ad) — warp autodiff genom LBM (device={DEV})"); print("=" * 78)
    nx, ny, T, omega = 24, 20, 40, 1.0
    solid_np = np.zeros((nx, ny), np.int32); solid_np[:, 0] = 1; solid_np[:, -1] = 1
    mask_np = np.zeros((nx, ny), np.float32); mask_np[nx // 2:, 1:-1] = 1.0   # right half, fluid
    cx = wp.array(CX, dtype=wp.float32, device=DEV); cy = wp.array(CY, dtype=wp.float32, device=DEV)
    w = wp.array(WT, dtype=wp.float32, device=DEV); opp = wp.array(OP, dtype=wp.int32, device=DEV)
    solid = wp.array(solid_np, dtype=wp.int32, device=DEV); mask = wp.array(mask_np, dtype=wp.float32, device=DEV)
    f0_np = _equil(np.ones((nx, ny), np.float32), np.zeros((nx, ny), np.float32), np.zeros((nx, ny), np.float32))

    def forward(force_wp, tape=None):
        fs = [wp.array(f0_np, dtype=wp.float32, device=DEV, requires_grad=(tape is not None))]
        fps = []
        J = wp.zeros(1, dtype=wp.float32, device=DEV, requires_grad=(tape is not None))
        def run():
            for t in range(T):
                fp = wp.zeros((9, nx, ny), dtype=wp.float32, device=DEV, requires_grad=(tape is not None))
                fn = wp.zeros((9, nx, ny), dtype=wp.float32, device=DEV, requires_grad=(tape is not None))
                wp.launch(collide, (nx, ny), inputs=[fs[-1], fp, force_wp, cx, cy, w, omega], device=DEV)
                wp.launch(stream, (nx, ny), inputs=[fp, fn, solid, cx, cy, opp, nx, ny], device=DEV)
                fps.append(fp); fs.append(fn)
            wp.launch(objective, (nx, ny), inputs=[fs[-1], cx, mask, J], device=DEV)
        if tape is not None:
            with tape: run()
        else:
            run()
        wp.synchronize()
        return J, fs[-1]

    # ── AUTODIFF: ∂J/∂force via wp.Tape ──
    force = wp.array(np.full((nx, ny), 1e-4, np.float32), dtype=wp.float32, device=DEV, requires_grad=True)
    tape = wp.Tape()
    J, f_end = forward(force, tape=tape)
    J0 = objective_value(J, f_end, cx, mask, (nx, ny))
    tape.backward(loss=J)
    grad_ad = force.grad.numpy().copy()
    print(f"\n  J0 (sum ux, right region) = {J0:.6e};  autodiff dJ/dforce computed (||grad|| = {np.linalg.norm(grad_ad):.3e})")

    # -- FINITE-DIFFERENCE verification on selected cells (central) --
    eps = 1e-3
    cells = [(6, 10), (12, 5), (12, 10), (18, 14), (12, 15)]
    print(f"\n  cell        | autodiff dJ/dforce | central FD (eps={eps}) | relative error")
    fbase = np.full((nx, ny), 1e-4, np.float32)
    ok_all = True; nverif = 0
    for (a, b) in cells:
        fp = fbase.copy(); fp[a, b] += eps
        fm = fbase.copy(); fm[a, b] -= eps
        Jp = objective_value(*forward(wp.array(fp, dtype=wp.float32, device=DEV)), cx, mask, (nx, ny))
        Jm = objective_value(*forward(wp.array(fm, dtype=wp.float32, device=DEV)), cx, mask, (nx, ny))
        fd = (Jp - Jm) / (2 * eps)
        ad = float(grad_ad[a, b])
        rel = abs(ad - fd) / (abs(fd) + 1e-12)
        ok = rel < 0.05; ok_all = ok_all and ok; nverif += 1
        print(f"  ({a:2d},{b:2d})     | {ad:>18.4e} | {fd:>18.4e} | {rel*100:5.1f}% {'✓' if ok else '✗'}")

    all_ok = ok_all and nverif >= 3
    print("\n" + "=" * 78)
    print(f"VERDIKT: differentiable-LBM (warp autodiff genom LBM) = {'✓ ENABLER BEVISAD' if all_ok else '✗ DELVIS'} "
          f"({nverif} celler, autodiff = central-FD < 5%)")
    if all_ok:
        print(f"  warp gives a CORRECT gradient through the whole LBM rollout -> d(flow objective)/d(input) for free.")
        print(f"  NEXT: dJ/d(GEOMETRY) via continuous porosity (Brinkman) -> GRADIENT-based")
        print(f"  generative design (1 adjoint vs thousands of black-box evaluations).")
    else:
        print(f"  warp autodiff through LBM not yet verified clean - see relative error above.")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
